"""Measure low-fitness-to-high-fitness extrapolation on exhaustive TFBind8.

The train subsets are the lowest label-ranked sequences in the full SIX6_REF_R1
landscape.  The test set is fixed to the highest 50% of the same landscape.  A
ten-member two-convolution-plus-fully-connected CNN ensemble is trained for
each requested lower-tail fraction.  This mirrors the neural model family used
by DyNA-PPO while making all otherwise unspecified architectural choices
explicit in the saved protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from BioSeqBench.evaluation.metrics import compute_metrics
from BioSeqBench.models.tfbind8 import TFBind8DyNACNN
from BioSeqBench.training.utils import set_seed


DNA_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


@dataclass(frozen=True)
class ExperimentProtocol:
    source: str
    label_column: str
    train_definition: str
    test_definition: str
    ensemble_size: int
    architecture: str
    channels: tuple[int, int]
    kernel_size: int
    hidden_units: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    base_seed: int


def one_hot(sequences: list[str]) -> torch.Tensor:
    encoded = torch.zeros((len(sequences), 4, 8), dtype=torch.float32)
    for row, sequence in enumerate(sequences):
        for position, base in enumerate(sequence):
            encoded[row, DNA_TO_INDEX[base], position] = 1.0
    return encoded


def ranked_subsets(frame: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return exact-size lower-tail train and a fixed exact-size upper-half test."""
    ordered = frame.sort_values(["label", "sequence"], kind="mergesort").reset_index(drop=True)
    total = len(ordered)
    train_count = int(round(total * fraction))
    test_count = total // 2
    if not 0 < train_count <= test_count:
        raise ValueError(f"fraction must be in (0, 0.5], got {fraction}")
    train_frame = ordered.iloc[:train_count].copy()
    test_frame = ordered.iloc[total - test_count :].copy()
    if train_frame["sequence"].isin(test_frame["sequence"]).any():
        raise RuntimeError("lower-tail training and upper-half test subsets overlap")
    return train_frame, test_frame


def fit_member(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> TFBind8DyNACNN:
    set_seed(seed)
    model = TFBind8DyNACNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x.to(device, non_blocking=True)).squeeze(-1)
            loss = loss_fn(prediction, batch_y.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.no_grad()
def predict_ensemble(models: list[nn.Module], features: torch.Tensor, *, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    means, stds = [], []
    loader = DataLoader(features, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    for batch_x in loader:
        member_predictions = torch.stack([model(batch_x.to(device, non_blocking=True)).squeeze(-1) for model in models])
        means.append(member_predictions.mean(dim=0).cpu())
        stds.append(member_predictions.std(dim=0, correction=0).cpu())
    return torch.cat(means).numpy(), torch.cat(stds).numpy()


def run_fraction(
    frame: pd.DataFrame,
    fraction: float,
    *,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict[str, float | int | str]:
    train_frame, test_frame = ranked_subsets(frame, fraction)
    train_x = one_hot(train_frame.sequence.tolist())
    test_x = one_hot(test_frame.sequence.tolist())
    train_labels = train_frame.label.to_numpy(dtype=np.float32)
    test_labels = test_frame.label.to_numpy(dtype=np.float32)
    label_mean = float(train_labels.mean())
    label_std = float(max(train_labels.std(), 1e-8))
    train_y = torch.from_numpy((train_labels - label_mean) / label_std)
    models = [
        fit_member(
            train_x,
            train_y,
            seed=args.seed + member,
            device=device,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        for member in range(args.ensemble_size)
    ]
    normalized_prediction, normalized_std = predict_ensemble(models, test_x, device=device, batch_size=args.batch_size)
    prediction = normalized_prediction * label_std + label_mean
    metrics = compute_metrics(test_labels, prediction)
    run_name = f"bottom_{fraction * 100:g}pct"
    torch.save(
        {
            "member_states": [model.state_dict() for model in models],
            "train_label_mean": label_mean,
            "train_label_std": label_std,
            "train_sequences": train_frame.sequence.tolist(),
            "test_sequences": test_frame.sequence.tolist(),
        },
        output_dir / f"{run_name}.pt",
    )
    pd.DataFrame(
        {
            "sequence": test_frame.sequence.to_numpy(),
            "label": test_labels,
            "prediction": prediction,
            "ensemble_std": normalized_std * label_std,
        }
    ).to_csv(output_dir / f"{run_name}_test_predictions.csv", index=False)
    return {
        "train_fraction": fraction,
        "train_count": len(train_frame),
        "test_fraction": 0.5,
        "test_count": len(test_frame),
        "train_label_min": float(train_labels.min()),
        "train_label_max": float(train_labels.max()),
        "test_label_min": float(test_labels.min()),
        "test_label_max": float(test_labels.max()),
        "test_prediction_mean": float(prediction.mean()),
        "test_prediction_std": float(prediction.std()),
        "mean_ensemble_std": float((normalized_std * label_std).mean()),
        **metrics,
    }


def parse_fractions(value: str) -> list[float]:
    fractions = [float(item.strip()) / 100.0 for item in value.split(",")]
    if not fractions or any(fraction <= 0 or fraction > 0.5 for fraction in fractions):
        raise argparse.ArgumentTypeError("fractions must be comma-separated percentages in (0, 50]")
    return fractions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-path", default="BioSeqBench/raw_data/tfbind8/six6_ref_r1.parquet")
    parser.add_argument("--fractions", type=parse_fractions, default=parse_fractions("1,5,10,20,50"))
    parser.add_argument("--ensemble-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="BioSeqBench/results/tfbind8_low_percentile_dynappo_cnn")
    args = parser.parse_args()
    if args.ensemble_size < 1:
        parser.error("--ensemble-size must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    frame = pd.read_parquet(args.raw_path).rename(columns={"e_score": "label"})[["sequence", "label"]]
    if len(frame) != 4**8 or frame.sequence.nunique() != 4**8:
        raise ValueError("Expected the complete, unique 4^8 TFBind8 landscape")
    protocol = ExperimentProtocol(
        source=str(Path(args.raw_path).resolve()),
        label_column="e_score",
        train_definition="exact-size lowest label-ranked fraction of all 65,536 sequences; ties broken lexicographically",
        test_definition="fixed exact-size highest 50% of all 65,536 sequences; ties broken lexicographically",
        ensemble_size=args.ensemble_size,
        architecture="one-hot DNA -> Conv1d(4,32,k=3,p=1) -> ReLU -> Conv1d(32,64,k=3,p=1) -> ReLU -> Flatten -> Linear(512,128) -> ReLU -> Linear(128,1)",
        channels=(32, 64),
        kernel_size=3,
        hidden_units=128,
        optimizer="AdamW",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        base_seed=args.seed,
    )
    (output_dir / "protocol.json").write_text(json.dumps(asdict(protocol), indent=2) + "\n")
    results = []
    for fraction in args.fractions:
        print(f"Training bottom {fraction:.0%} on {device} ...", flush=True)
        row = run_fraction(frame, fraction, args=args, device=device, output_dir=output_dir)
        results.append(row)
        print(json.dumps(row, indent=2), flush=True)
    result_path = output_dir / "results.csv"
    with result_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
