"""Train a non-pretrained, sequence-only neural baseline for Cetnar 2024."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from BioSeqBench.evaluation.metrics import compute_metrics
from BioSeqBench.models.cetnar2024_mrna_stability import CetnarStabilityCNNBiLSTM
from BioSeqBench.training.utils import set_seed


BASE_TO_CHANNEL = np.full(256, -1, dtype=np.int8)
BASE_TO_CHANNEL[ord("A")] = 0
BASE_TO_CHANNEL[ord("C")] = 1
BASE_TO_CHANNEL[ord("G")] = 2
BASE_TO_CHANNEL[ord("U")] = 3


def encode_rna(sequences: pd.Series, max_length: int = 80) -> tuple[torch.Tensor, torch.Tensor]:
    """One-hot encode variable-length RNA sequences into a compact tensor."""
    sequence_list = sequences.astype(str).tolist()
    encoded = np.zeros((len(sequence_list), 4, max_length), dtype=np.float32)
    mask = np.zeros((len(sequence_list), max_length), dtype=np.bool_)
    for position in range(max_length):
        bases = np.fromiter(
            (ord(sequence[position]) if position < len(sequence) else 0 for sequence in sequence_list),
            dtype=np.uint8,
            count=len(sequence_list),
        )
        channels = BASE_TO_CHANNEL[bases]
        valid = channels >= 0
        encoded[np.nonzero(valid)[0], channels[valid], position] = 1.0
        mask[valid, position] = True
    return torch.from_numpy(encoded), torch.from_numpy(mask)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    targets, predictions = [], []
    loss_total = 0.0
    count = 0
    loss_fn = nn.MSELoss()
    for inputs, mask, labels in loader:
        output = model(sequence_tensor=inputs.to(device, non_blocking=True), mask=mask.to(device, non_blocking=True)).squeeze(1)
        labels = labels.to(device, non_blocking=True)
        loss_total += loss_fn(output, labels).item() * len(labels)
        count += len(labels)
        predictions.append(output.cpu().numpy())
        targets.append(labels.cpu().numpy())
    target_standardised = np.concatenate(targets)
    prediction_standardised = np.concatenate(predictions)
    return (
        scaler.inverse_transform(target_standardised[:, None]).ravel(),
        scaler.inverse_transform(prediction_standardised[:, None]).ravel(),
        loss_total / count,
    )


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset_dir = Path(args.data_root) / "cetnar2024_mrna_stability"
    train_frame = pd.read_csv(dataset_dir / "train.csv")
    test_frame = pd.read_csv(dataset_dir / "test.csv")
    # A sequence-grouped validation split prevents repeated barcode measurements
    # of the same RNA from occurring on both sides of early stopping.
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.validation_fraction, random_state=args.seed)
    fit_indices, val_indices = next(splitter.split(train_frame, groups=train_frame["sequence"]))
    train_x, train_mask = encode_rna(train_frame["sequence"])
    test_x, test_mask = encode_rna(test_frame["sequence"])
    targets = train_frame["label"].to_numpy(dtype=np.float64)
    test_targets = test_frame["label"].to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(targets[fit_indices, None])
    train_y = torch.tensor(scaler.transform(targets[:, None]).ravel(), dtype=torch.float32)
    test_y = torch.tensor(scaler.transform(test_targets[:, None]).ravel(), dtype=torch.float32)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    fit_loader = DataLoader(
        TensorDataset(train_x[fit_indices], train_mask[fit_indices], train_y[fit_indices]),
        batch_size=args.batch_size, shuffle=True, generator=generator, **loader_options,
    )
    val_loader = DataLoader(
        TensorDataset(train_x[val_indices], train_mask[val_indices], train_y[val_indices]),
        batch_size=args.batch_size, shuffle=False, **loader_options,
    )
    test_loader = DataLoader(TensorDataset(test_x, test_mask, test_y), batch_size=args.batch_size, shuffle=False, **loader_options)
    model = CetnarStabilityCNNBiLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    output_dir = Path(args.checkpoint_dir) / "cetnar2024_mrna_stability" / "sequence_cnn_bilstm"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / "cetnar_sequence_cnn_bilstm.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for inputs, mask, labels in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(sequence_tensor=inputs.to(device, non_blocking=True), mask=mask.to(device, non_blocking=True)).squeeze(1)
            labels = labels.to(device, non_blocking=True)
            loss = loss_fn(output, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            seen += len(labels)
        _, _, validation_loss = predict(model, val_loader, device, scaler)
        record = {"epoch": epoch, "train_standardised_mse": total_loss / seen, "val_standardised_mse": validation_loss}
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            torch.save({
                "model_state": model.state_dict(), "epoch": epoch, "val_standardised_mse": best_loss,
                "seed": args.seed, "label_mean": float(scaler.mean_[0]), "label_scale": float(scaler.scale_[0]),
                "validation_fraction": args.validation_fraction, "fit_rows": len(fit_indices), "val_rows": len(val_indices),
            }, output_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}; restoring epoch {best_epoch}.", flush=True)
                break
    state = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    true, prediction, test_loss = predict(model, test_loader, device, scaler)
    metrics = compute_metrics(true, prediction)
    seen_sequences = set(train_frame["sequence"])
    unseen = ~test_frame["sequence"].isin(seen_sequences).to_numpy()
    unseen_metrics = compute_metrics(true[unseen], prediction[unseen])
    pd.DataFrame({
        "sequence_id": test_frame["sequence_id"], "sequence": test_frame["sequence"],
        "true_k_fit": true, "predicted_k_fit": prediction,
    }).to_csv(output_dir / "test_predictions.csv", index=False)
    result: dict[str, object] = {
        "timestamp": int(time.time()), "dataset": "cetnar2024_mrna_stability", "model": "sequence_cnn_bilstm",
        "protocol": "sequence-only non-pretrained; group-held-out validation from author training split; author fixed test untouched",
        "seed": args.seed, "best_epoch": best_epoch, "best_val_standardised_mse": best_loss,
        "train_fit_rows": len(fit_indices), "validation_rows": len(val_indices), "test_rows": len(test_frame),
        "test_standardised_mse": test_loss, "cross_split_repeated_sequence_test_rows": int((~unseen).sum()),
        "strict_unseen_sequence_test_rows": int(unseen.sum()), "strict_unseen_sequence_test": unseen_metrics,
        **metrics,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cetnar sequence-only CNN-BiLSTM baseline")
    parser.add_argument("--data-root", default="BioSeqBench/dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint-dir", default="BioSeqBench/checkpoints/strong_cetnar_sequence_v1")
    parser.add_argument("--log-dir", default="BioSeqBench/logs/strong_cetnar_sequence_v1")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
