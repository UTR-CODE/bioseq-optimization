"""Train the published DeepSTARR CNN on the canonical BioSeqBench split.

This is deliberately separate from the generic trainer because the original
Keras notebook has two independently summed MSE heads, unnormalised targets,
and validation-loss early stopping.  It also makes reverse-complement training
augmentation explicit while evaluating only unique assay sequences.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from BioSeqBench.evaluation.metrics import compute_multitask_metrics
from BioSeqBench.models.deepstarr import DeepSTARRPaper
from BioSeqBench.training.utils import set_seed


LABEL_COLUMNS = ["label_developmental", "label_housekeeping"]
BASE_TO_INDEX = np.full(256, -1, dtype=np.int8)
BASE_TO_INDEX[ord("A")] = 0
BASE_TO_INDEX[ord("C")] = 1
BASE_TO_INDEX[ord("G")] = 2
BASE_TO_INDEX[ord("T")] = 3


def one_hot_sequences(sequences: pd.Series) -> torch.Tensor:
    """Encode fixed-length 249-bp sequences without per-example Python work."""
    values = sequences.to_numpy(dtype="S249")
    encoded = BASE_TO_INDEX[np.frombuffer(values.tobytes(), dtype=np.uint8).reshape(len(values), 249)]
    if np.any(encoded < 0):
        raise ValueError("DeepSTARR canonical split contains a non-ACGT base")
    output = np.zeros((len(values), 4, 249), dtype=np.float32)
    rows = np.arange(len(values))[:, None]
    positions = np.arange(249)[None, :]
    output[rows, encoded, positions] = 1.0
    return torch.from_numpy(output)


def load_split(dataset_dir: Path, split: str) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    frame = pd.read_csv(dataset_dir / f"{split}.csv")
    if not frame["length"].eq(249).all():
        raise ValueError(f"DeepSTARR {split} split must contain fixed 249-bp sequences")
    return one_hot_sequences(frame["sequence"]), torch.tensor(
        frame[LABEL_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32
    ), frame


def reverse_complement_one_hot(inputs: torch.Tensor) -> torch.Tensor:
    # Channel order is A,C,G,T; the complement permutation is T,G,C,A.
    return inputs[:, [3, 2, 1, 0], :].flip(dims=[2]).contiguous()


def loss_by_head(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    developmental = nn.functional.mse_loss(prediction[:, 0], target[:, 0])
    housekeeping = nn.functional.mse_loss(prediction[:, 1], target[:, 1])
    # Keras compile(loss=["mse", "mse"], loss_weights=[1, 1]) sums the heads.
    return developmental + housekeeping, developmental, housekeeping


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    predictions, targets = [], []
    sum_total = sum_dev = sum_hk = 0.0
    seen = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        output = model(sequence_tensor=inputs)
        total, developmental, housekeeping = loss_by_head(output, labels)
        batch_size = len(labels)
        sum_total += total.item() * batch_size
        sum_dev += developmental.item() * batch_size
        sum_hk += housekeeping.item() * batch_size
        seen += batch_size
        predictions.append(output.cpu().numpy())
        targets.append(labels.cpu().numpy())
    losses = {
        "loss": sum_total / seen,
        "developmental_mse": sum_dev / seen,
        "housekeeping_mse": sum_hk / seen,
    }
    return np.concatenate(targets), np.concatenate(predictions), losses


def append_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset_dir = Path(args.data_root) / "deepstarr"
    train_x, train_y, _ = load_split(dataset_dir, "train")
    val_x, val_y, _ = load_split(dataset_dir, "val")
    test_x, test_y, test_frame = load_split(dataset_dir, "test")
    unique_train_rows = len(train_y)
    if args.rc_augment_train:
        train_x = torch.cat([train_x, reverse_complement_one_hot(train_x)], dim=0)
        train_y = torch.cat([train_y, train_y], dim=0)
    # The Zenodo publication files also contain RC copies for validation and
    # test.  Use them for source-protocol selection/reporting, and keep the
    # unique forward test set below as the benchmark score.
    paper_val_x, paper_val_y = val_x, val_y
    paper_test_x, paper_test_y = test_x, test_y
    if args.rc_augment_evaluation:
        paper_val_x = torch.cat([val_x, reverse_complement_one_hot(val_x)], dim=0)
        paper_val_y = torch.cat([val_y, val_y], dim=0)
        paper_test_x = torch.cat([test_x, reverse_complement_one_hot(test_x)], dim=0)
        paper_test_y = torch.cat([test_y, test_y], dim=0)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(TensorDataset(paper_val_x, paper_val_y), batch_size=args.batch_size, shuffle=False, **loader_options)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=args.batch_size, shuffle=False, **loader_options)
    paper_test_loader = DataLoader(TensorDataset(paper_test_x, paper_test_y), batch_size=args.batch_size, shuffle=False, **loader_options)
    model = DeepSTARRPaper().to(device)
    # Keras Adam defaults: beta_1=.9, beta_2=.999, epsilon=1e-7.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-7)
    checkpoint_dir = Path(args.checkpoint_dir) / "deepstarr" / "deepstarr_paper"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / "deepstarr_paper.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = dev_sum = hk_sum = 0.0
        seen = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(sequence_tensor=inputs)
            total, developmental, housekeeping = loss_by_head(output, labels)
            total.backward()
            optimizer.step()
            batch_size = len(labels)
            loss_sum += total.item() * batch_size
            dev_sum += developmental.item() * batch_size
            hk_sum += housekeeping.item() * batch_size
            seen += batch_size
        _, _, val_losses = evaluate(model, val_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_developmental_mse": dev_sum / seen,
            "train_housekeeping_mse": hk_sum / seen,
            "val_loss": val_losses["loss"],
            "val_developmental_mse": val_losses["developmental_mse"],
            "val_housekeeping_mse": val_losses["housekeeping_mse"],
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if record["val_loss"] < best_loss:
            best_loss, best_epoch, stale_epochs = record["val_loss"], epoch, 0
            torch.save({
                "model_state": model.state_dict(), "epoch": epoch, "val_loss": best_loss,
                "seed": args.seed, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                "rc_augment_train": args.rc_augment_train, "train_unique_rows": unique_train_rows,
                "train_rows_seen_per_epoch": len(train_y), "label_columns": LABEL_COLUMNS,
            }, checkpoint_dir / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; restoring epoch {best_epoch}.", flush=True)
                break
    state = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    y_true, y_pred, test_losses = evaluate(model, test_loader, device)
    paper_y_true, paper_y_pred, paper_test_losses = evaluate(model, paper_test_loader, device)
    benchmark_metrics = compute_multitask_metrics(y_true, y_pred, LABEL_COLUMNS)
    paper_protocol_metrics = compute_multitask_metrics(paper_y_true, paper_y_pred, LABEL_COLUMNS)
    predictions = test_frame[["sequence_id", "sequence"]].copy()
    predictions["true_developmental"] = y_true[:, 0]
    predictions["pred_developmental"] = y_pred[:, 0]
    predictions["true_housekeeping"] = y_true[:, 1]
    predictions["pred_housekeeping"] = y_pred[:, 1]
    predictions.to_csv(checkpoint_dir / "test_predictions.csv", index=False)
    result: dict[str, object] = {
        "timestamp": int(time.time()), "dataset": "deepstarr", "model": "deepstarr_paper",
        "seed": args.seed, "best_epoch": best_epoch, "best_val_loss": best_loss,
        "train_unique_rows": unique_train_rows, "train_rows_seen_per_epoch": len(train_y),
        "rc_augment_train": args.rc_augment_train, "rc_augment_evaluation": args.rc_augment_evaluation,
        "test_rows": len(test_y), "paper_protocol_test_rows": len(paper_test_y),
        "paper_developmental_pearson": 0.68, "paper_housekeeping_pearson": 0.74,
        **{f"benchmark_test_{key}": value for key, value in test_losses.items()},
        **{f"paper_protocol_test_{key}": value for key, value in paper_test_losses.items()},
        **{f"benchmark_{key}": value for key, value in benchmark_metrics.items()},
        **{f"paper_protocol_{key}": value for key, value in paper_protocol_metrics.items()},
    }
    (checkpoint_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    append_result(Path(args.results_file), result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Faithful DeepSTARR PyTorch reproduction")
    parser.add_argument("--data-root", default="BioSeqBench/dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device")
    parser.add_argument("--rc-augment-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rc-augment-evaluation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-dir", default="BioSeqBench/checkpoints/strong_deepstarr_paper_v1")
    parser.add_argument("--log-dir", default="BioSeqBench/logs/strong_deepstarr_paper_v1")
    parser.add_argument("--results-file", default="BioSeqBench/results/deepstarr_paper_reproduction.csv")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
