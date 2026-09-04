"""Train standardized WeakSeqCNN-v1 surrogates for 5'UTR lower-tail pools.

The 260k Sample et al. ``train.csv`` sequences are ranked once by measured
activity.  For each requested threshold, the visible lower-tail pool is split
into a fixed 90% fit / 10% validation partition.  The highest 5% of the same
260k sequences form one fixed, disjoint OOD test set and are never used for
training or early stopping.  The separate official ``test.csv`` is deliberately
not used by this weak-model benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from BioSeqBench.evaluation.metrics import compute_metrics
from BioSeqBench.models.weakseqcnn import WeakSeqCNN
from BioSeqBench.models.utr5_mpra.ensemble import encode_utr_sequences
from BioSeqBench.training.train_tfbind_weak_cnns import DEFAULT_FRACTIONS, validate_fractions
from BioSeqBench.training.utils import set_seed


@dataclass(frozen=True)
class UTR5WeakTrainingProtocol:
    dataset: str
    source_train_count: int
    fixed_top_test_count: int
    top_test_fraction: float
    validation_fraction: float
    sequence_length: int
    target: str
    train_pool_definition: str
    validation_definition: str
    test_definition: str
    fractions: tuple[float, ...]
    ensemble_size: int
    architecture: str
    model_config: dict[str, object]
    parameter_count: int
    optimizer: str
    scheduler: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    base_seed: int
    split_seed: int
    execution_determinism: str
    top_test_indices_sha256: str


def _encode_fixed_utr(sequences: np.ndarray) -> torch.Tensor:
    """Use the existing vectorized encoder, exposing canonical [B,L,4]."""
    return encode_utr_sequences(sequences).transpose(1, 2).contiguous()


def _index_digest(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _split_visible_pool(
    count: int, validation_fraction: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if count < 2:
        raise ValueError("The visible lower-tail pool must contain at least two sequences")
    validation_count = min(max(1, int(round(count * validation_fraction))), count - 1)
    permutation = np.random.default_rng(seed).permutation(count)
    return np.sort(permutation[validation_count:]), np.sort(permutation[:validation_count])


def _set_member_seed(seed: int, device: torch.device) -> None:
    """Seed all RNGs without forcing the exceptionally slow deterministic CNN kernels."""
    set_seed(seed)
    if device.type == "cuda":
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def _predict_normalized(
    model: nn.Module,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[torch.Tensor] = []
    loader = DataLoader(features, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    for batch in loader:
        outputs.append(model(batch.to(device, non_blocking=True)).cpu())
    return torch.cat(outputs).numpy()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = compute_metrics(y_true, y_pred)
    metrics["rmse"] = float(np.sqrt(metrics["mse"]))
    return metrics


def _prefix_metrics(prefix: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in _metrics(y_true, y_pred).items()}


def _fit_member(
    train_features: torch.Tensor,
    train_labels: np.ndarray,
    validation_features: torch.Tensor,
    validation_labels: np.ndarray,
    *,
    args: argparse.Namespace,
    seed: int,
    fraction: float,
    checkpoint_path: Path,
    split_metadata: dict[str, object],
    device: torch.device,
) -> dict[str, float | int]:
    fit_started = time.perf_counter()
    training_seconds = 0.0
    validation_seconds = 0.0
    label_mean = float(train_labels.mean())
    label_std = float(max(train_labels.std(), 1e-8))
    normalized_train = torch.from_numpy(((train_labels - label_mean) / label_std).astype(np.float32))
    normalized_validation = torch.from_numpy(
        ((validation_labels - label_mean) / label_std).astype(np.float32)
    )
    train_loader = _loader(
        train_features,
        normalized_train,
        batch_size=args.batch_size,
        shuffle=True,
        seed=seed,
        device=device,
        num_workers=args.num_workers,
    )
    validation_loader = _loader(
        validation_features,
        normalized_validation,
        batch_size=args.batch_size,
        shuffle=False,
        seed=seed,
        device=device,
        num_workers=args.num_workers,
    )
    _set_member_seed(seed, device)
    model = WeakSeqCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    patience_left = args.patience
    completed_epoch = 0
    for epoch in range(1, args.epochs + 1):
        completed_epoch = epoch
        model.train()
        phase_started = time.perf_counter()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x.to(device, non_blocking=True))
            loss = loss_fn(prediction, batch_y.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        training_seconds += time.perf_counter() - phase_started
        model.eval()
        loss_sum = 0.0
        seen = 0
        phase_started = time.perf_counter()
        with torch.no_grad():
            for batch_x, batch_y in validation_loader:
                prediction = model(batch_x.to(device, non_blocking=True))
                loss_sum += float(loss_fn(prediction, batch_y.to(device, non_blocking=True))) * len(batch_y)
                seen += len(batch_y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        validation_seconds += time.perf_counter() - phase_started
        validation_loss = loss_sum / seen
        scheduler.step()
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
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "model": "weakseqcnn_v1",
            "model_config": model.model_config,
            "dataset": "utr5_mpra",
            "sequence_length": 50,
            "train_fraction": fraction,
            "train_label_mean": label_mean,
            "train_label_std": label_std,
            "seed": seed,
            "best_epoch": best_epoch,
            "completed_epoch": completed_epoch,
            "validation_loss": best_validation_loss,
            "fit_seconds": time.perf_counter() - fit_started,
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            **split_metadata,
        },
        checkpoint_path,
    )
    return {
        "best_epoch": best_epoch,
        "completed_epoch": completed_epoch,
        "validation_loss_normalized": best_validation_loss,
        "fit_seconds": time.perf_counter() - fit_started,
        "training_seconds": training_seconds,
        "validation_seconds": validation_seconds,
    }


def _load_member(path: Path, device: torch.device) -> tuple[WeakSeqCNN, dict]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("model") != "weakseqcnn_v1":
        raise ValueError(f"Checkpoint {path} is not a WeakSeqCNN-v1 model")
    model = WeakSeqCNN(**state.get("model_config", {}))
    model.load_state_dict(state["model_state"])
    return model.to(device).eval(), state


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    fractions = validate_fractions(args.fractions, args.top_test_fraction)
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must lie in (0, 1)")
    if args.ensemble_size < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("ensemble-size, epochs, and patience must be positive")
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root = Path(args.data_root)
    source = pd.read_csv(data_root / "train.csv", usecols=["sequence", "label"])
    source_sequences = source["sequence"].astype(str).to_numpy()
    source_labels = source["label"].to_numpy(dtype=np.float64)
    ranked_indices = np.lexsort((source_sequences, source_labels))
    top_test_count = max(1, int(round(len(source) * args.top_test_fraction)))
    test_indices = ranked_indices[-top_test_count:]
    test_sequences = source_sequences[test_indices]
    test_labels = source_labels[test_indices]
    test_features = _encode_fixed_utr(test_sequences)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template = WeakSeqCNN()
    parameter_count = sum(parameter.numel() for parameter in template.parameters())
    trainable_count = sum(parameter.numel() for parameter in template.parameters() if parameter.requires_grad)
    print(f"WeakSeqCNN-v1 total parameters: {parameter_count:,}; trainable: {trainable_count:,}", flush=True)
    top_test_digest = _index_digest(test_indices)
    protocol = UTR5WeakTrainingProtocol(
        dataset="utr5_mpra",
        source_train_count=len(source),
        fixed_top_test_count=top_test_count,
        top_test_fraction=args.top_test_fraction,
        validation_fraction=args.validation_fraction,
        sequence_length=50,
        target="measured mean ribosome load (MRL); higher is better",
        train_pool_definition=(
            "each threshold is the exact lowest label-ranked fraction of train.csv; "
            "90% of that visible pool is used for fitting"
        ),
        validation_definition=(
            "a deterministic 10% holdout split inside each threshold-specific visible lower-tail pool; "
            "fixed across model seeds at that threshold"
        ),
        test_definition=(
            "the highest 5% by measured activity in train.csv; fixed, disjoint from every visible pool, "
            "and used only for final OOD evaluation; official test.csv is unused"
        ),
        fractions=fractions,
        ensemble_size=args.ensemble_size,
        architecture="WeakSeqCNN-v1 standardized surrogate CNN",
        model_config=template.model_config,
        parameter_count=parameter_count,
        optimizer="AdamW",
        scheduler="CosineAnnealingLR",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        base_seed=args.seed,
        split_seed=args.split_seed,
        execution_determinism=(
            "all RNGs seeded per member; CUDA deterministic algorithms disabled and cuDNN benchmark enabled"
        ),
        top_test_indices_sha256=top_test_digest,
    )
    (output_dir / "protocol.json").write_text(json.dumps(asdict(protocol), indent=2) + "\n")
    results: list[dict[str, object]] = []
    member_results: list[dict[str, object]] = []
    for fraction in fractions:
        visible_count = int(round(len(ranked_indices) * fraction))
        visible_indices = ranked_indices[:visible_count]
        fit_local, validation_local = _split_visible_pool(
            visible_count, args.validation_fraction, args.split_seed
        )
        train_indices = visible_indices[fit_local]
        validation_indices = visible_indices[validation_local]
        train_sequences = source_sequences[train_indices]
        train_labels = source_labels[train_indices]
        validation_sequences = source_sequences[validation_indices]
        validation_labels = source_labels[validation_indices]
        train_features = _encode_fixed_utr(train_sequences)
        validation_features = _encode_fixed_utr(validation_sequences)
        split_metadata: dict[str, object] = {
            "visible_pool_count": visible_count,
            "training_count": len(train_indices),
            "validation_count": len(validation_indices),
            "fixed_test_count": top_test_count,
            "visible_pool_indices_sha256": _index_digest(visible_indices),
            "training_indices_sha256": _index_digest(train_indices),
            "validation_indices_sha256": _index_digest(validation_indices),
            "test_indices_sha256": top_test_digest,
        }
        fraction_seed = args.seed + int(round(fraction * 10_000))
        run_dir = output_dir / f"bottom_{fraction * 100:g}pct"
        run_dir.mkdir(parents=True, exist_ok=True)
        validation_member_predictions: list[np.ndarray] = []
        test_member_predictions: list[np.ndarray] = []
        member_records: list[dict[str, object]] = []
        print(
            f"Training utr5_mpra bottom {fraction:.0%}: {visible_count:,} visible -> "
            f"{len(train_indices):,} fit / {len(validation_indices):,} in-pool val; "
            f"{top_test_count:,} fixed train.csv top-5% test on {device}",
            flush=True,
        )
        for member in range(args.ensemble_size):
            member_seed = fraction_seed + member
            checkpoint = run_dir / f"member_{member:02d}.pt"
            expected = {
                "model": "weakseqcnn_v1",
                "dataset": "utr5_mpra",
                "sequence_length": 50,
                "train_fraction": fraction,
                "seed": member_seed,
                **split_metadata,
            }
            if args.resume and checkpoint.exists():
                model, state = _load_member(checkpoint, device)
                mismatch = {
                    key: (state.get(key), value)
                    for key, value in expected.items()
                    if state.get(key) != value
                }
                if mismatch:
                    raise ValueError(f"Checkpoint {checkpoint} is incompatible: {mismatch}")
                fit_record = {
                    "best_epoch": int(state["best_epoch"]),
                    "completed_epoch": int(state["completed_epoch"]),
                    "validation_loss_normalized": float(state["validation_loss"]),
                    "fit_seconds": float(state.get("fit_seconds", float("nan"))),
                    "training_seconds": float(state.get("training_seconds", float("nan"))),
                    "validation_seconds": float(state.get("validation_seconds", float("nan"))),
                }
                resumed = True
            else:
                fit_record = _fit_member(
                    train_features,
                    train_labels,
                    validation_features,
                    validation_labels,
                    args=args,
                    seed=member_seed,
                    fraction=fraction,
                    checkpoint_path=checkpoint,
                    split_metadata=split_metadata,
                    device=device,
                )
                model, state = _load_member(checkpoint, device)
                resumed = False
            mean = float(state["train_label_mean"])
            scale = float(state["train_label_std"])
            validation_prediction = (
                _predict_normalized(model, validation_features, device=device, batch_size=args.prediction_batch_size)
                * scale
                + mean
            )
            test_prediction = (
                _predict_normalized(model, test_features, device=device, batch_size=args.prediction_batch_size)
                * scale
                + mean
            )
            validation_member_predictions.append(validation_prediction)
            test_member_predictions.append(test_prediction)
            metrics = {
                **_prefix_metrics("validation", validation_labels, validation_prediction),
                **_prefix_metrics("test", test_labels, test_prediction),
            }
            record: dict[str, object] = {
                "dataset": "utr5_mpra",
                "task": "mean_ribosome_load",
                "seed": member_seed,
                "member": member,
                "train_fraction": fraction,
                "visible_pool_fraction": fraction,
                "n_visible": visible_count,
                "n_train": len(train_indices),
                "n_val": len(validation_indices),
                "n_test": top_test_count,
                "parameter_count": parameter_count,
                **fit_record,
                **metrics,
                "resumed": resumed,
            }
            member_results.append(record)
            member_records.append(record)
            print(
                f"  member={member:02d} seed={member_seed} best_epoch={fit_record['best_epoch']} "
                f"fit_s={fit_record['fit_seconds']:.1f} val_r2={metrics['validation_r2']:.4f} "
                f"test_r2={metrics['test_r2']:.4f}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        validation_matrix = np.stack(validation_member_predictions)
        test_matrix = np.stack(test_member_predictions)
        validation_mean = validation_matrix.mean(axis=0)
        validation_std = validation_matrix.std(axis=0)
        test_mean = test_matrix.mean(axis=0)
        test_std = test_matrix.std(axis=0)
        validation_metrics = _prefix_metrics("validation", validation_labels, validation_mean)
        test_metrics = _prefix_metrics("test", test_labels, test_mean)
        pd.DataFrame(
            {
                "source_train_csv_index": validation_indices,
                "sequence": validation_sequences,
                "label": validation_labels,
                "prediction": validation_mean,
                "ensemble_std": validation_std,
            }
        ).to_csv(run_dir / "validation_predictions.csv", index=False)
        pd.DataFrame(
            {
                "source_train_csv_index": test_indices,
                "sequence": test_sequences,
                "label": test_labels,
                "prediction": test_mean,
                "ensemble_std": test_std,
            }
        ).to_csv(run_dir / "test_predictions.csv", index=False)
        (run_dir / "members.json").write_text(json.dumps(member_records, indent=2) + "\n")
        result: dict[str, object] = {
            "dataset": "utr5_mpra",
            "task": "mean_ribosome_load",
            "train_fraction": fraction,
            "visible_pool_fraction": fraction,
            "n_visible": visible_count,
            "n_train": len(train_indices),
            "n_val": len(validation_indices),
            "n_test": top_test_count,
            "ensemble_size": args.ensemble_size,
            "parameter_count": parameter_count,
            "train_label_min": float(train_labels.min()),
            "train_label_max": float(train_labels.max()),
            **validation_metrics,
            **test_metrics,
        }
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
        del train_features, validation_features
    _write_csv(output_dir / "results.csv", results)
    _write_csv(output_dir / "surrogate_results.csv", member_results)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="BioSeqBench/dataset/utr5_mpra")
    parser.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--top-test-fraction", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    # The 260k UTR task uses the task-specific large-batch allowance in
    # weak_model_description.txt; the architecture and optimizer are unchanged.
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--prediction-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--split-seed", type=int, default=20260827)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", default="BioSeqBench/results/utr5_weak_cnns")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
