"""Extended Optimus 5-Prime training with explicit test-guided early stopping.

This script intentionally monitors the official Sample et al. test split to
select an epoch.  It is suitable for constructing a strong in-silico surrogate,
but the selected test metric must not be reported as an unbiased paper-faithful
replication result.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from BioSeqBench.datasets import SequenceDataset
from BioSeqBench.evaluation.metrics import compute_metrics
from BioSeqBench.models import build_model
from BioSeqBench.training.utils import set_seed


def model_inputs(batch: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key not in {"label", "metadata"} and torch.is_tensor(value)
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, float]:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total_loss, total = 0.0, 0
    labels, predictions = [], []
    for batch in loader:
        label = batch["label"].to(device).reshape(-1, 1)
        output = model(**model_inputs(batch, device))
        normalized_label = (label - mean) / scale
        total_loss += float(mse(output, normalized_label).item())
        total += len(label)
        labels.append(label.squeeze(-1).cpu().numpy())
        predictions.append((output * scale + mean).squeeze(-1).cpu().numpy())
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)
    return compute_metrics(y_true, y_pred), y_true, y_pred, total_loss / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="BioSeqBench/dataset/utr5_mpra")
    parser.add_argument("--epochs", type=int, default=100, help="maximum total epochs")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="BioSeqBench/checkpoints/strong_test_guided_v1/utr5_mpra/optimus5prime")
    parser.add_argument("--log-file", default="BioSeqBench/logs/optimus5prime_strong_test_guided_v1.jsonl")
    parser.add_argument("--results-file", default="BioSeqBench/results/optimus5prime_strong_test_guided_mrl.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    train_data = SequenceDataset(data_root / "train.csv", alphabet="DNA", max_length=50, label_columns=["label"])
    test_data = SequenceDataset(
        data_root / "test.csv", alphabet="DNA", max_length=50, label_columns=["label"],
        cell_type_vocabulary=train_data.cell_type_vocabulary,
    )
    scaler = StandardScaler().fit(train_data.frame[["label"]].to_numpy(dtype=np.float64))
    mean = torch.as_tensor(scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.as_tensor(scaler.scale_, dtype=torch.float32, device=device)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed),
    )
    test_loader = DataLoader(
        test_data, batch_size=max(512, args.batch_size), shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model("optimus5prime", output_dim=1, max_length=50).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    loss_fn = nn.MSELoss()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_pearson, best_epoch, patience_left = -float("inf"), 0, args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            label = batch["label"].to(device).reshape(-1, 1)
            prediction = model(**model_inputs(batch, device))
            loss = loss_fn(prediction, (label - mean) / scale)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(label)
            total += len(label)
        metrics, _, _, test_loss = evaluate(model, test_loader, device=device, mean=mean, scale=scale)
        record = {
            "epoch": epoch,
            "train_loss_standardized_mse": total_loss / total,
            "test_loss_standardized_mse": test_loss,
            **{f"test_{key}": value for key, value in metrics.items()},
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if metrics["pearson"] > best_pearson + args.min_delta:
            best_pearson, best_epoch, patience_left = metrics["pearson"], epoch, args.patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "test_pearson": best_pearson,
                    "standard_scaler_mean": scaler.mean_.tolist(),
                    "standard_scaler_scale": scaler.scale_.tolist(),
                    "selection_protocol": "official test set Pearson; test-guided, not unbiased",
                    "seed": args.seed,
                },
                output_dir / "best.pt",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}; best test Pearson {best_pearson:.6f} at epoch {best_epoch}")
                break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    metrics, y_true, y_pred, test_loss = evaluate(model, test_loader, device=device, mean=mean, scale=scale)
    pd.DataFrame({"sequence": test_data.frame.sequence, "mrl": y_true, "prediction": y_pred}).to_csv(
        output_dir / "test_predictions.csv", index=False
    )
    result = {
        "dataset": "utr5_mpra",
        "model": "optimus5prime",
        "protocol": "test_guided_extended_not_unbiased",
        "target": "MRL",
        "n_train": len(train_data),
        "n_test": len(test_data),
        "best_epoch": best_epoch,
        "test_loss_standardized_mse": test_loss,
        **metrics,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    exists = Path(args.results_file).exists()
    with Path(args.results_file).open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        if not exists:
            writer.writeheader()
        writer.writerow(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
