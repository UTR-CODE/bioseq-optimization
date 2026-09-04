"""Train CNN weak-surrogate ensembles on lower-activity TFBind landscapes.

The weak model sees only the lowest requested fraction of the exhaustive
landscape.  Validation is carved out of that visible lower tail.  A fixed top
5% set, which is disjoint even from the 95% training pool, is used only for OOD
diagnostics and never for early stopping.  Downstream optimization candidates
must be evaluated with the exact landscape oracle, not these diagnostic labels.
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
from torch.utils.data import DataLoader, Subset, TensorDataset

from BioSeqBench.evaluation.metrics import compute_metrics
from BioSeqBench.models.tfbind8 import TFBindCNN
from BioSeqBench.oracles.tfbind import TFBindLandscape, codes_to_encoded, load_tfbind_landscape
from BioSeqBench.training.utils import set_seed


DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


@dataclass(frozen=True)
class WeakTrainingProtocol:
    dataset: str
    landscape_size: int
    sequence_length: int
    score_definition: str
    train_pool_definition: str
    validation_definition: str
    ood_test_definition: str
    fractions: tuple[float, ...]
    ood_test_fraction: float
    validation_fraction: float
    ensemble_size: int
    architecture: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    base_seed: int


def validate_fractions(fractions: list[float] | tuple[float, ...], ood_test_fraction: float) -> tuple[float, ...]:
    values = tuple(float(value) for value in fractions)
    if not values:
        raise ValueError("At least one training fraction is required")
    if len(set(values)) != len(values):
        raise ValueError("Training fractions must be unique")
    if any(value <= 0 or value > 1 - ood_test_fraction for value in values):
        raise ValueError(
            f"Training fractions must be in (0, {1 - ood_test_fraction:g}] "
            "so they cannot overlap the fixed OOD test tail"
        )
    return tuple(sorted(values))


def one_hot_codes(codes: np.ndarray, length: int) -> torch.Tensor:
    encoded = codes_to_encoded(np.asarray(codes, dtype=np.int64), length)
    one_hot = np.eye(4, dtype=np.float32)[encoded].transpose(0, 2, 1)
    return torch.from_numpy(np.ascontiguousarray(one_hot))


def split_visible_pool(count: int, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 2:
        raise ValueError("The visible training pool must contain at least two sequences")
    validation_count = max(1, int(round(count * validation_fraction)))
    validation_count = min(validation_count, count - 1)
    permutation = np.random.default_rng(seed).permutation(count)
    return permutation[validation_count:], permutation[:validation_count]


def make_loader(
    dataset: TensorDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def predict(model: nn.Module, features: torch.Tensor, *, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    output = []
    loader = DataLoader(features, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    for batch in loader:
        output.append(model(batch.to(device, non_blocking=True)).squeeze(-1).cpu())
    return torch.cat(output).numpy()


def fit_member(
    features: torch.Tensor,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
    dataset: str,
    fraction: float,
) -> tuple[TFBindCNN, dict[str, float | int]]:
    train_labels = labels[train_indices]
    label_mean = float(train_labels.mean())
    label_std = float(max(train_labels.std(), 1e-8))
    normalized = torch.from_numpy(((labels - label_mean) / label_std).astype(np.float32))
    tensor_dataset = TensorDataset(features, normalized)
    train_loader = make_loader(
        tensor_dataset, train_indices, batch_size=args.batch_size, shuffle=True, seed=seed,
        device=device, num_workers=args.num_workers,
    )
    validation_loader = make_loader(
        tensor_dataset, validation_indices, batch_size=args.batch_size, shuffle=False, seed=seed,
        device=device, num_workers=args.num_workers,
    )

    set_seed(seed)
    model = TFBindCNN().to(device)
    # Initialize LazyLinear before saving/loading state dictionaries.
    model(features[:1].to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x.to(device, non_blocking=True)).squeeze(-1)
            loss = loss_fn(prediction, batch_y.to(device, non_blocking=True))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

        model.eval()
        validation_loss = 0.0
        validation_seen = 0
        with torch.no_grad():
            for batch_x, batch_y in validation_loader:
                prediction = model(batch_x.to(device, non_blocking=True)).squeeze(-1)
                loss = loss_fn(prediction, batch_y.to(device, non_blocking=True))
                validation_loss += float(loss) * len(batch_y)
                validation_seen += len(batch_y)
        validation_loss /= validation_seen
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            patience_left = args.patience
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "model": "tfbind_cnn",
            "dataset": dataset,
            "sequence_length": features.shape[-1],
            "train_fraction": fraction,
            "train_label_mean": label_mean,
            "train_label_std": label_std,
            "seed": seed,
            "best_epoch": best_epoch,
            "validation_loss": best_validation_loss,
        },
        checkpoint_path,
    )
    return model, {"best_epoch": best_epoch, "validation_loss": best_validation_loss}


def load_member(checkpoint_path: Path, device: torch.device) -> tuple[TFBindCNN, dict]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = TFBindCNN()
    model(torch.zeros(1, 4, int(state["sequence_length"])))
    model.load_state_dict(state["model_state"])
    return model.to(device).eval(), state


def train_fraction(
    landscape: TFBindLandscape,
    ranked_codes: np.ndarray,
    ood_codes: np.ndarray,
    ood_features: torch.Tensor,
    fraction: float,
    *,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> dict[str, float | int | str]:
    visible_count = int(round(landscape.size * fraction))
    visible_codes = ranked_codes[:visible_count]
    visible_labels = landscape.score_codes(visible_codes)
    split_seed = args.seed + int(round(fraction * 10000))
    train_indices, validation_indices = split_visible_pool(visible_count, args.validation_fraction, split_seed)
    visible_features = one_hot_codes(visible_codes, landscape.length)
    run_name = f"bottom_{fraction * 100:g}pct"
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    member_predictions = []
    member_records = []
    for member in range(args.ensemble_size):
        member_seed = split_seed + member
        checkpoint_path = run_dir / f"member_{member:02d}.pt"
        if args.resume and checkpoint_path.exists():
            model, state = load_member(checkpoint_path, device)
            expected = {
                "dataset": landscape.dataset,
                "sequence_length": landscape.length,
                "train_fraction": fraction,
                "seed": member_seed,
            }
            mismatches = {
                key: (state.get(key), value)
                for key, value in expected.items()
                if state.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"Checkpoint {checkpoint_path} does not match this run: {mismatches}. "
                    "Use a different output directory or --no-resume."
                )
            record = {
                "member": member,
                "seed": int(state["seed"]),
                "best_epoch": int(state["best_epoch"]),
                "validation_loss": float(state["validation_loss"]),
                "resumed": True,
            }
        else:
            model, fit_record = fit_member(
                visible_features, visible_labels, train_indices, validation_indices,
                args=args, seed=member_seed, device=device, checkpoint_path=checkpoint_path,
                dataset=landscape.dataset, fraction=fraction,
            )
            record = {"member": member, "seed": member_seed, **fit_record, "resumed": False}
        normalized_prediction = predict(model, ood_features, device=device, batch_size=args.batch_size)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        prediction = normalized_prediction * float(state["train_label_std"]) + float(state["train_label_mean"])
        member_predictions.append(prediction)
        member_records.append(record)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    prediction_matrix = np.stack(member_predictions)
    ensemble_prediction = prediction_matrix.mean(axis=0)
    ensemble_std = prediction_matrix.std(axis=0)
    ood_labels = landscape.score_codes(ood_codes)
    metrics = compute_metrics(ood_labels, ensemble_prediction)
    ood_name = f"ood_top_{args.ood_test_fraction * 100:g}pct_predictions.csv"
    pd.DataFrame(
        {
            "sequence": landscape.sequences_for_codes(ood_codes),
            "label": ood_labels,
            "prediction": ensemble_prediction,
            "ensemble_std": ensemble_std,
        }
    ).to_csv(run_dir / ood_name, index=False)
    (run_dir / "members.json").write_text(json.dumps(member_records, indent=2) + "\n")
    return {
        "dataset": landscape.dataset,
        "train_fraction": fraction,
        "visible_pool_count": visible_count,
        "fit_count": len(train_indices),
        "validation_count": len(validation_indices),
        "ood_test_count": len(ood_codes),
        "visible_label_max": float(visible_labels.max()),
        "ood_label_min": float(ood_labels.min()),
        "ensemble_size": args.ensemble_size,
        **metrics,
    }


def run(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    if not 0 < args.ood_test_fraction < 1:
        raise ValueError("--ood-test-fraction must be in (0, 1)")
    fractions = validate_fractions(args.fractions, args.ood_test_fraction)
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be in (0, 1)")
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    landscape = load_tfbind_landscape(args.dataset, args.raw_path)
    ranked_codes = landscape.ranked_codes()
    ood_count = int(round(landscape.size * args.ood_test_fraction))
    ood_codes = ranked_codes[-ood_count:]
    ood_features = one_hot_codes(ood_codes, landscape.length)
    output_dir = Path(args.output_dir) / landscape.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = WeakTrainingProtocol(
        dataset=landscape.dataset,
        landscape_size=landscape.size,
        sequence_length=landscape.length,
        score_definition="true exhaustive landscape activity; higher is better",
        train_pool_definition="exact lowest label-ranked fraction; ties broken by lexicographic sequence",
        validation_definition="seeded random holdout drawn only from the visible lower-tail pool",
        ood_test_definition=(
            f"fixed exact highest {args.ood_test_fraction * 100:g}% of the landscape; "
            "diagnostics only; never used for training or early stopping"
        ),
        fractions=fractions,
        ood_test_fraction=args.ood_test_fraction,
        validation_fraction=args.validation_fraction,
        ensemble_size=args.ensemble_size,
        architecture="one-hot DNA -> Conv1d(4,32,k=3,p=1) -> ReLU -> Conv1d(32,64,k=3,p=1) -> ReLU -> Flatten -> Linear(64*L,128) -> ReLU -> Linear(128,1)",
        optimizer="AdamW",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        base_seed=args.seed,
    )
    (output_dir / "protocol.json").write_text(json.dumps(asdict(protocol), indent=2) + "\n")
    results = []
    for fraction in fractions:
        print(f"Training {landscape.dataset} bottom {fraction:.0%} on {device}", flush=True)
        result = train_fraction(
            landscape, ranked_codes, ood_codes, ood_features, fraction,
            args=args, device=device, output_dir=output_dir,
        )
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["tfbind8", "tfbind10"])
    parser.add_argument("--raw-path")
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS),
        help="lower-tail fractions as decimals; default: 0.01 0.05 0.10 ... 0.90 0.95",
    )
    parser.add_argument("--ood-test-fraction", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--ensemble-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", default="BioSeqBench/results/tfbind_weak_cnns")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.ensemble_size < 1 or args.epochs < 1 or args.patience < 1:
        raise SystemExit("ensemble-size, epochs, and patience must all be positive")
    run(args)


if __name__ == "__main__":
    main()
