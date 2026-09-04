from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from BioSeqBench.datasets import SequenceDataset, StreamingSequenceDataset
from BioSeqBench.evaluation.metrics import compute_multitask_metrics
from BioSeqBench.models import build_model
from BioSeqBench.training.utils import set_seed


def _model_inputs(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key not in {"label", "metadata"} and torch.is_tensor(value)
    }


def _normalize_targets(target: torch.Tensor, mean: torch.Tensor | None, std: torch.Tensor | None) -> torch.Tensor:
    return target if mean is None or std is None else (target - mean) / std


def _restore_predictions(prediction: torch.Tensor, mean: torch.Tensor | None, std: torch.Tensor | None) -> torch.Tensor:
    return prediction if mean is None or std is None else prediction * std + mean


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device,
    target_mean: torch.Tensor | None = None, target_std: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    predictions, targets = [], []
    total_loss, total = 0.0, 0
    loss_fn = nn.MSELoss()
    for batch in loader:
        target = batch["label"].to(device)
        output = model(**_model_inputs(batch, device))
        target = target.reshape_as(output)
        normalized_target = _normalize_targets(target, target_mean, target_std)
        loss = loss_fn(output, normalized_target)
        total_loss += loss.item() * len(target)
        total += len(target)
        predictions.append(_restore_predictions(output, target_mean, target_std).cpu().numpy())
        targets.append(target.cpu().numpy())
    return np.concatenate(targets), np.concatenate(predictions), total_loss / max(1, total)


def append_results(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(row)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    dataset_dir = Path(args.data_root) / args.dataset.lower()
    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    alphabet = metadata["alphabet"]
    label_columns = args.label_columns.split(",") if args.label_columns else metadata.get("target_columns", ["label"])
    streaming = bool(metadata.get("streaming_csv", False))
    requested_max_length = args.max_length or metadata.get("max_length")
    dataset_class = StreamingSequenceDataset if streaming else SequenceDataset
    def load_split(name: str, *, vocabulary=None):
        kwargs = dict(alphabet=alphabet, max_length=requested_max_length, label_columns=label_columns, cell_type_vocabulary=vocabulary)
        if streaming:
            kwargs["num_records"] = metadata["split_sizes"][name]
        return dataset_class(dataset_dir / f"{name}.csv", **kwargs)
    train_data = load_split("train")
    max_length = train_data.max_length
    common = dict(
        alphabet=alphabet, max_length=max_length, label_columns=label_columns,
        cell_type_vocabulary=train_data.cell_type_vocabulary,
    )
    if streaming:
        common["num_records"] = None
    # A small number of primary papers (notably Sample et al. 2019) prescribed
    # a train/test split and trained for a fixed number of epochs, without a
    # validation set.  Preserve that protocol rather than silently moving part
    # of the official training set into validation.
    has_validation = int(metadata.get("split_sizes", {}).get("val", 0)) > 0
    val_data = load_split("val", vocabulary=train_data.cell_type_vocabulary) if has_validation else None
    test_data = load_split("test", vocabulary=train_data.cell_type_vocabulary)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=not isinstance(train_data, IterableDataset), num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(), generator=generator,
    )
    val_loader = (
        DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if val_data is not None else None
    )
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(
        args.model, output_dim=len(label_columns), num_cell_types=len(train_data.cell_type_vocabulary), max_length=max_length,
    ).to(device)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, momentum=args.momentum,
            weight_decay=args.weight_decay, nesterov=args.nesterov,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn: nn.Module = nn.BCEWithLogitsLoss() if args.task == "classification" else nn.MSELoss()
    target_mean = target_std = None
    if args.standardize_targets:
        if streaming or not hasattr(train_data, "frame"):
            raise ValueError("--standardize-targets currently requires an in-memory SequenceDataset")
        # Match Sample et al.'s Optimus 5-Prime notebook: fit sklearn's
        # StandardScaler on the publication training split only, then invert
        # predictions before reporting MRL-space metrics.
        values = train_data.frame[label_columns].to_numpy(dtype=np.float64)
        scaler = StandardScaler().fit(values)
        target_mean = torch.as_tensor(scaler.mean_, dtype=torch.float32, device=device)
        target_std = torch.as_tensor(scaler.scale_, dtype=torch.float32, device=device).clamp_min(1e-8)
    checkpoint_dir = Path(args.checkpoint_dir) / args.dataset.lower() / args.model.lower()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.dataset.lower()}_{args.model.lower()}.jsonl"
    best_val, best_epoch = float("inf"), 0
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            target = batch["label"].to(device)
            output = model(**_model_inputs(batch, device))
            target = target.reshape_as(output)
            loss = loss_fn(output, _normalize_targets(target, target_mean, target_std))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            running += loss.item() * len(target)
            seen += len(target)
        val_loss = (
            predict(model, val_loader, device, target_mean, target_std)[2]
            if val_loader is not None else running / max(1, seen)
        )
        record = {"epoch": epoch, "train_loss": running / max(1, seen), "val_loss": val_loss}
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        if val_loss < best_val:
            best_val, best_epoch, patience_left = val_loss, epoch, args.patience
            torch.save({
                "model_state": model.state_dict(), "model": args.model, "dataset": args.dataset,
                "label_columns": label_columns, "max_length": max_length, "cell_type_vocabulary": train_data.cell_type_vocabulary,
                "epoch": epoch, "val_loss": val_loss, "seed": args.seed,
                "target_mean": target_mean.cpu().tolist() if target_mean is not None else None,
                "target_std": target_std.cpu().tolist() if target_std is not None else None,
            }, checkpoint_dir / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    state = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    y_true, y_pred, test_loss = predict(model, test_loader, device, target_mean, target_std)
    metrics = compute_multitask_metrics(y_true, y_pred, label_columns, args.task)
    result = {
        "timestamp": int(time.time()), "dataset": args.dataset, "model": args.model,
        "seed": args.seed, "best_epoch": best_epoch, "val_loss": best_val, "test_loss": test_loss,
        **metrics,
    }
    append_results(Path(args.results_file), result)
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a BioSeqBench surrogate")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", default="BioSeqBench/dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--standardize-targets", action="store_true", help="Fit label mean/std on training data and optimize normalized MSE")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--label-columns", help="comma-separated targets; metadata targets are default")
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--device", help="e.g. cuda, cuda:1, or cpu")
    parser.add_argument("--checkpoint-dir", default="BioSeqBench/checkpoints")
    parser.add_argument("--log-dir", default="BioSeqBench/logs")
    parser.add_argument("--results-file", default="BioSeqBench/results/benchmark_results.csv")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
