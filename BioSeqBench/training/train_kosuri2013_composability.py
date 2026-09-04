"""Train a sequence-only, two-output surrogate for Kosuri et al. 2013."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from BioSeqBench.evaluation.metrics import compute_metrics
from BioSeqBench.models.kosuri2013_composability import KosuriTwoBranchCNN
from BioSeqBench.training.utils import set_seed


BASE_TO_CHANNEL = np.full(256, -1, dtype=np.int8)
for _index, _base in enumerate("ACGT"):
    BASE_TO_CHANNEL[ord(_base)] = _index
TARGET_COLUMNS = ("label_protein_expression", "label_rna_expression")


def encode_sequences(sequences: pd.Series, max_length: int) -> tuple[torch.Tensor, np.ndarray]:
    values = sequences.astype(str).str.upper().tolist()
    result = np.zeros((len(values), 4, max_length), dtype=np.float32)
    lengths = np.empty(len(values), dtype=np.float32)
    for row, sequence in enumerate(values):
        if len(sequence) > max_length:
            raise ValueError("Unexpected sequence longer than configured maximum")
        codes = BASE_TO_CHANNEL[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
        if np.any(codes < 0):
            raise ValueError("Non-ACGT Kosuri sequence")
        result[row, codes, np.arange(len(sequence))] = 1.0
        lengths[row] = len(sequence) / max_length
    return torch.from_numpy(result), lengths


def make_split(frame: pd.DataFrame, max_promoter: int, max_rbs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    promoter, promoter_length = encode_sequences(frame["promoter_sequence"], max_promoter)
    rbs, rbs_length = encode_sequences(frame["rbs_sequence"], max_rbs)
    lengths = torch.from_numpy(np.stack([promoter_length, rbs_length], axis=1))
    target = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64)
    return promoter, rbs, lengths, target


def make_loader(parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray], target: np.ndarray, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    promoter, rbs, lengths, _ = parts
    return DataLoader(
        TensorDataset(promoter, rbs, lengths, torch.as_tensor(target, dtype=torch.float32)),
        batch_size=batch_size, shuffle=shuffle, num_workers=0,
        pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def predict(model: nn.Module, data: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    predictions, targets, loss_total = [], [], 0.0
    loss_fn = nn.MSELoss()
    for promoter, rbs, lengths, target in data:
        output = model(promoter.to(device), rbs.to(device), lengths.to(device))
        target = target.to(device)
        loss_total += loss_fn(output, target).item() * len(target)
        predictions.append(output.cpu().numpy())
        targets.append(target.cpu().numpy())
    prediction, target = np.concatenate(predictions), np.concatenate(targets)
    return target, prediction, loss_total / len(target)


def target_metrics(true_log: np.ndarray, prediction_log: np.ndarray, scaler: StandardScaler) -> dict[str, dict[str, dict[str, float]]]:
    true = np.power(10.0, scaler.inverse_transform(true_log))
    prediction = np.power(10.0, scaler.inverse_transform(prediction_log))
    return {
        name: {"log10": compute_metrics(true_log[:, index], prediction_log[:, index]), "raw": compute_metrics(true[:, index], prediction[:, index])}
        for index, name in enumerate(("protein_expression", "rna_expression"))
    }


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    root = Path(args.data_root) / "kosuri2013_composability"
    train_frame, val_frame, test_frame = (pd.read_csv(root / f"{split}.csv") for split in ("train", "val", "test"))
    max_promoter = max(frame["promoter_sequence"].str.len().max() for frame in (train_frame, val_frame, test_frame))
    max_rbs = max(frame["rbs_sequence"].str.len().max() for frame in (train_frame, val_frame, test_frame))
    train_parts = make_split(train_frame, max_promoter, max_rbs)
    val_parts = make_split(val_frame, max_promoter, max_rbs)
    test_parts = make_split(test_frame, max_promoter, max_rbs)
    scaler = StandardScaler().fit(np.log10(train_parts[3]))
    train_target = scaler.transform(np.log10(train_parts[3]))
    val_target = scaler.transform(np.log10(val_parts[3]))
    test_target = scaler.transform(np.log10(test_parts[3]))
    train_loader = make_loader(train_parts, train_target, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_parts, val_target, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    test_loader = make_loader(test_parts, test_target, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    model = KosuriTwoBranchCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    loss_fn = nn.MSELoss()
    output = Path(args.checkpoint_dir) / "kosuri2013_composability" / "two_branch_cnn"
    output.mkdir(parents=True, exist_ok=True)
    history, best_loss, best_epoch, stale = [], float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, rows = 0.0, 0
        for promoter, rbs, lengths, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            value = model(promoter.to(device), rbs.to(device), lengths.to(device))
            loss = loss_fn(value, target.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(target)
            rows += len(target)
        _, _, val_loss = predict(model, val_loader, device)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_standardized_mse": total / rows, "val_standardized_mse": val_loss, "learning_rate": optimizer.param_groups[0]["lr"]})
        print(json.dumps(history[-1]), flush=True)
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_standardized_mse": val_loss,
                        "target_mean": scaler.mean_.tolist(), "target_scale": scaler.scale_.tolist(), "seed": args.seed,
                        "max_promoter_length": int(max_promoter), "max_rbs_length": int(max_rbs)}, output / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    state = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    true_log, prediction_log, test_loss = predict(model, test_loader, device)
    true_raw = np.power(10.0, scaler.inverse_transform(true_log))
    prediction_raw = np.power(10.0, scaler.inverse_transform(prediction_log))
    pd.DataFrame({
        "sequence_id": test_frame["sequence_id"], "sequence": test_frame["sequence"],
        "true_protein_expression": true_raw[:, 0], "predicted_protein_expression": prediction_raw[:, 0],
        "true_rna_expression": true_raw[:, 1], "predicted_rna_expression": prediction_raw[:, 1],
        "true_log10_protein": true_log[:, 0], "predicted_log10_protein": prediction_log[:, 0],
        "true_log10_rna": true_log[:, 1], "predicted_log10_rna": prediction_log[:, 1],
    }).to_csv(output / "test_predictions.csv", index=False)
    result = {
        "timestamp": int(time.time()), "dataset": "kosuri2013_composability", "model": "two_branch_cnn",
        "protocol": "non-pretrained promoter/RBS two-branch CNN; train-only standardization of log10 positive RNA and protein labels; grouped construct split retained",
        "seed": args.seed, "device": str(device), "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "split_rows": {"train": len(train_frame), "val": len(val_frame), "test": len(test_frame)},
        "max_promoter_length": int(max_promoter), "max_rbs_length": int(max_rbs),
        "best_epoch": best_epoch, "best_val_standardized_mse": best_loss, "test_standardized_mse": test_loss,
        "test_metrics": target_metrics(true_log, prediction_log, scaler), "history": history,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kosuri 2013 two-output composability CNN")
    parser.add_argument("--data-root", default="BioSeqBench/dataset")
    parser.add_argument("--checkpoint-dir", default="BioSeqBench/checkpoints/strong_kosuri2013_v1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
