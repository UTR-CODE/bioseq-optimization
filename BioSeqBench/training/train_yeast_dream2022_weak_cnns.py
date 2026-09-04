"""Train standardized WeakSeqCNN-v1 lower-tail surrogates on DREAM promoters.

`train.csv` (the FACS-scale random-promoter observations) is ranked once by
activity.  For every lower-tail fraction, a deterministic 90%/10% fit/validation
split is made *inside that visible pool*.  The highest 5% of this same training
set is the shared, disjoint high-activity OOD test set.  The official DREAM
test set has MAUDE labels on a different affine scale and is consequently not
used for the raw-scale lower-tail MSE/R2 benchmark.

The 6M-row CSV is parsed once into a compact uint8 sequence cache.  This cache
is a lossless operational representation of the DNA sequences (A/C/G/T ->
0/1/2/3, padding -> 255); it does not change the training data or its split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from BioSeqBench.evaluation.metrics import compute_metrics
from BioSeqBench.models.weakseqcnn import WeakSeqCNN
from BioSeqBench.training.train_tfbind_weak_cnns import DEFAULT_FRACTIONS, validate_fractions
from BioSeqBench.training.train_utr5_weak_cnns import _index_digest, _split_visible_pool
from BioSeqBench.training.utils import set_seed


DNA_LOOKUP = np.full(256, 255, dtype=np.uint8)
DNA_LOOKUP[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4, dtype=np.uint8)
PAD_CODE = np.uint8(255)


@dataclass(frozen=True)
class DreamWeakTrainingProtocol:
    dataset: str
    source_train_count: int
    fixed_top_test_count: int
    top_test_fraction: float
    validation_fraction: float
    max_sequence_length: int
    target: str
    train_pool_definition: str
    validation_definition: str
    test_definition: str
    official_test_note: str
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
    cache_manifest_sha256: str
    top_test_indices_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _encode_strings(sequences: np.ndarray, max_length: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(sequences, dtype=str)
    lengths = np.char.str_len(values).astype(np.uint8)
    if np.any(lengths > max_length):
        raise ValueError(f"Sequence longer than max_length={max_length}")
    joined = "".join(values.tolist()).encode("ascii")
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    codes = np.full((len(values), max_length), PAD_CODE, dtype=np.uint8)
    raw = np.frombuffer(joined, dtype=np.uint8)
    for row, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        encoded = DNA_LOOKUP[raw[start:stop]]
        if np.any(encoded == PAD_CODE):
            raise ValueError(f"Unsupported base in row {row}")
        codes[row, : stop - start] = encoded
    return codes, lengths


def _cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "manifest": cache_dir / "manifest.json",
        "codes": cache_dir / "train_codes.npy",
        "lengths": cache_dir / "train_lengths.npy",
        "labels": cache_dir / "train_labels.npy",
        "order": cache_dir / "label_rank_order.npy",
    }


def build_or_load_cache(data_root: Path, cache_dir: Path, chunk_size: int) -> tuple[dict[str, np.ndarray], dict]:
    paths = _cache_paths(cache_dir)
    if all(path.exists() for path in paths.values()):
        manifest = json.loads(paths["manifest"].read_text())
        arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items() if name != "manifest"}
        if (
            manifest.get("source_train_count") == len(arrays["labels"])
            and manifest.get("max_sequence_length") == arrays["codes"].shape[1]
        ):
            return arrays, manifest
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_root / "train.csv"
    total = sum(1 for _ in train_path.open("rb")) - 1
    max_length = 142
    codes = np.lib.format.open_memmap(paths["codes"], mode="w+", dtype=np.uint8, shape=(total, max_length))
    lengths = np.lib.format.open_memmap(paths["lengths"], mode="w+", dtype=np.uint8, shape=(total,))
    labels = np.lib.format.open_memmap(paths["labels"], mode="w+", dtype=np.float32, shape=(total,))
    cursor = 0
    for frame in pd.read_csv(train_path, usecols=["sequence", "label"], chunksize=chunk_size):
        values = frame["sequence"].astype(str).to_numpy()
        batch_codes, batch_lengths = _encode_strings(values, max_length)
        end = cursor + len(frame)
        codes[cursor:end] = batch_codes
        lengths[cursor:end] = batch_lengths
        labels[cursor:end] = frame["label"].to_numpy(dtype=np.float32)
        cursor = end
        print(f"Caching DREAM train sequences: {cursor:,}/{total:,}", flush=True)
    if cursor != total:
        raise RuntimeError(f"Cached {cursor} rows, expected {total}")
    codes.flush(); lengths.flush(); labels.flush()
    # Original row index is a deterministic tie break.  It avoids retaining 6M strings in RAM.
    order = np.lexsort((np.arange(total, dtype=np.int64), np.asarray(labels, dtype=np.float32)))
    np.save(paths["order"], order.astype(np.int64))
    manifest = {
        "source_train_csv": str(train_path),
        "source_train_size_bytes": train_path.stat().st_size,
        "source_train_count": total,
        "max_sequence_length": max_length,
        "encoding": "A/C/G/T -> 0/1/2/3; 255 padding",
        "rank_order": "ascending label, then original train.csv row index",
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items() if name != "manifest"}
    return arrays, manifest


def _set_member_seed(seed: int, device: torch.device) -> None:
    set_seed(seed)
    if device.type == "cuda":
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _features_from_codes(codes: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    batch_codes = torch.as_tensor(np.asarray(codes, dtype=np.int64), device=device)
    mask = batch_codes.ne(int(PAD_CODE))
    safe = batch_codes.clamp_max(3)
    return F.one_hot(safe, num_classes=4).float(), mask


@torch.no_grad()
def _predict(
    model: nn.Module, codes: np.ndarray, device: torch.device, batch_size: int, mean: float, scale: float,
) -> np.ndarray:
    result = np.empty(len(codes), dtype=np.float32)
    model.eval()
    for start in range(0, len(codes), batch_size):
        stop = min(start + batch_size, len(codes))
        features, mask = _features_from_codes(codes[start:stop], device)
        result[start:stop] = (model(features, mask).cpu().numpy() * scale + mean)
    return result


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    values = compute_metrics(y_true, y_pred)
    values["rmse"] = float(np.sqrt(values["mse"]))
    return values


def _prefixed(prefix: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in _metrics(y_true, y_pred).items()}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _fit_member(
    train_codes: np.ndarray, train_labels: np.ndarray, validation_codes: np.ndarray, validation_labels: np.ndarray,
    *, args: argparse.Namespace, seed: int, fraction: float, checkpoint_path: Path, split_metadata: dict, device: torch.device,
) -> dict[str, float | int]:
    started = time.perf_counter()
    mean, scale = float(train_labels.mean()), float(max(train_labels.std(), 1e-8))
    normalized = ((train_labels - mean) / scale).astype(np.float32)
    normalized_validation = ((validation_labels - mean) / scale).astype(np.float32)
    _set_member_seed(seed, device)
    model = WeakSeqCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss, best_epoch, patience_left, completed_epoch = float("inf"), 0, args.patience, 0
    for epoch in range(1, args.epochs + 1):
        completed_epoch = epoch
        model.train()
        for batch_indices in np.array_split(rng.permutation(len(train_labels)), range(args.batch_size, len(train_labels), args.batch_size)):
            if not len(batch_indices):
                continue
            features, mask = _features_from_codes(train_codes[batch_indices], device)
            target = torch.as_tensor(normalized[batch_indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features, mask), target)
            loss.backward(); optimizer.step()
        model.eval(); total, seen = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(validation_labels), args.batch_size):
                stop = min(start + args.batch_size, len(validation_labels))
                features, mask = _features_from_codes(validation_codes[start:stop], device)
                target = torch.as_tensor(normalized_validation[start:stop], device=device)
                total += float(loss_fn(model(features, mask), target)) * (stop - start)
                seen += stop - start
        validation_loss = total / seen
        scheduler.step()
        if validation_loss < best_loss:
            best_loss, best_epoch, patience_left = validation_loss, epoch, args.patience
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is None:
        raise RuntimeError("No model checkpoint created")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state, "model": "weakseqcnn_v1", "model_config": model.model_config,
        "dataset": "yeast_gpra_dream2022", "sequence_length": 142, "train_fraction": fraction,
        "train_label_mean": mean, "train_label_std": scale, "seed": seed, "best_epoch": best_epoch,
        "completed_epoch": completed_epoch, "validation_loss": best_loss, "fit_seconds": time.perf_counter() - started,
        **split_metadata,
    }, checkpoint_path)
    return {
        "best_epoch": best_epoch, "completed_epoch": completed_epoch,
        "validation_loss_normalized": best_loss, "fit_seconds": time.perf_counter() - started,
    }


def _load_member(path: Path, device: torch.device) -> tuple[WeakSeqCNN, dict]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = WeakSeqCNN(**state["model_config"])
    model.load_state_dict(state["model_state"])
    return model.to(device).eval(), state


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    fractions = validate_fractions(args.fractions, args.top_test_fraction)
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation fraction must lie in (0,1)")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root, output, cache_dir = Path(args.data_root), Path(args.output_dir), Path(args.cache_dir)
    arrays, manifest = build_or_load_cache(data_root, cache_dir, args.cache_chunk_size)
    codes, labels, order = arrays["codes"], arrays["labels"], arrays["order"]
    n, max_length = len(labels), codes.shape[1]
    test_count = int(round(n * args.top_test_fraction))
    test_indices = np.asarray(order[-test_count:], dtype=np.int64)
    test_codes, test_labels = codes[test_indices], np.asarray(labels[test_indices], dtype=np.float32)
    output.mkdir(parents=True, exist_ok=True)
    template = WeakSeqCNN(); parameter_count = sum(p.numel() for p in template.parameters())
    cache_manifest_sha = _sha256_file(_cache_paths(cache_dir)["manifest"])
    protocol = DreamWeakTrainingProtocol(
        dataset="yeast_gpra_dream2022", source_train_count=n, fixed_top_test_count=test_count,
        top_test_fraction=args.top_test_fraction, validation_fraction=args.validation_fraction, max_sequence_length=max_length,
        target="FACS promoter activity; higher is better",
        train_pool_definition="exact lowest label-ranked fraction of train.csv; 90% used for fitting",
        validation_definition="deterministic 10% split inside each threshold-specific visible lower-tail pool",
        test_definition="highest 5% label-ranked train.csv sequences; fixed and disjoint from every visible pool",
        official_test_note="Official DREAM test labels use a distinct MAUDE affine scale; only correlation diagnostics are comparable.",
        fractions=fractions, ensemble_size=args.ensemble_size, architecture="WeakSeqCNN-v1 standardized surrogate CNN",
        model_config=template.model_config, parameter_count=parameter_count, optimizer="AdamW", scheduler="CosineAnnealingLR",
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, batch_size=args.batch_size, max_epochs=args.epochs,
        patience=args.patience, base_seed=args.seed, split_seed=args.split_seed,
        execution_determinism="all RNGs seeded per member; CUDA deterministic algorithms disabled and cuDNN benchmark enabled",
        cache_manifest_sha256=cache_manifest_sha, top_test_indices_sha256=_index_digest(test_indices),
    )
    (output / "protocol.json").write_text(json.dumps(asdict(protocol), indent=2) + "\n")
    print(f"WeakSeqCNN-v1 parameters: {parameter_count:,}; DREAM train rows: {n:,}; top-5% test: {test_count:,}", flush=True)
    summaries, member_rows = [], []
    for fraction in fractions:
        visible_count = int(round(n * fraction))
        visible_indices = np.asarray(order[:visible_count], dtype=np.int64)
        fit_local, validation_local = _split_visible_pool(visible_count, args.validation_fraction, args.split_seed)
        train_indices, validation_indices = visible_indices[fit_local], visible_indices[validation_local]
        train_codes, train_labels = codes[train_indices], np.asarray(labels[train_indices], dtype=np.float32)
        validation_codes, validation_labels = codes[validation_indices], np.asarray(labels[validation_indices], dtype=np.float32)
        split_metadata = {
            "visible_pool_count": visible_count, "training_count": len(train_indices), "validation_count": len(validation_indices),
            "fixed_test_count": test_count, "visible_pool_indices_sha256": _index_digest(visible_indices),
            "training_indices_sha256": _index_digest(train_indices), "validation_indices_sha256": _index_digest(validation_indices),
            "test_indices_sha256": _index_digest(test_indices),
        }
        run_dir = output / f"bottom_{fraction * 100:g}pct"; run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Training DREAM bottom {fraction:.0%}: {visible_count:,} visible -> {len(train_indices):,} fit / {len(validation_indices):,} val; {test_count:,} top-5% test", flush=True)
        val_predictions, test_predictions, records = [], [], []
        for member in range(args.ensemble_size):
            member_seed = args.seed + int(round(fraction * 10_000)) + member
            checkpoint = run_dir / f"member_{member:02d}.pt"
            if args.resume and checkpoint.exists():
                model, state = _load_member(checkpoint, device)
                record = {
                    "best_epoch": int(state["best_epoch"]), "completed_epoch": int(state["completed_epoch"]),
                    "validation_loss_normalized": float(state["validation_loss"]), "fit_seconds": float(state.get("fit_seconds", np.nan)),
                }
                resumed = True
            else:
                record = _fit_member(train_codes, train_labels, validation_codes, validation_labels, args=args, seed=member_seed, fraction=fraction, checkpoint_path=checkpoint, split_metadata=split_metadata, device=device)
                model, state = _load_member(checkpoint, device); resumed = False
            mean, scale = float(state["train_label_mean"]), float(state["train_label_std"])
            val_prediction = _predict(model, validation_codes, device, args.prediction_batch_size, mean, scale)
            test_prediction = _predict(model, test_codes, device, args.prediction_batch_size, mean, scale)
            val_predictions.append(val_prediction); test_predictions.append(test_prediction)
            metric = {**_prefixed("validation", validation_labels, val_prediction), **_prefixed("test_top5pct", test_labels, test_prediction)}
            item = {
                "dataset": "yeast_gpra_dream2022", "task": "promoter_activity", "member": member, "seed": member_seed,
                "train_fraction": fraction, "n_visible": visible_count, "n_train": len(train_indices), "n_val": len(validation_indices),
                "n_test": test_count, "parameter_count": parameter_count, **record, **metric, "resumed": resumed,
            }
            member_rows.append(item); records.append(item)
            print(f"  member={member:02d} seed={member_seed} epoch={record['best_epoch']} fit_s={record['fit_seconds']:.1f} val_r2={metric['validation_r2']:.4f} top5_r2={metric['test_top5pct_r2']:.4f}", flush=True)
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
        val_mean, val_std = np.stack(val_predictions).mean(0), np.stack(val_predictions).std(0)
        test_mean, test_std = np.stack(test_predictions).mean(0), np.stack(test_predictions).std(0)
        pd.DataFrame({"source_train_csv_index": validation_indices, "label": validation_labels, "prediction": val_mean, "ensemble_std": val_std}).to_csv(run_dir / "validation_predictions.csv", index=False)
        pd.DataFrame({"source_train_csv_index": test_indices, "label": test_labels, "prediction": test_mean, "ensemble_std": test_std}).to_csv(run_dir / "top5_test_predictions.csv", index=False)
        (run_dir / "members.json").write_text(json.dumps(records, indent=2) + "\n")
        summary = {
            "dataset": "yeast_gpra_dream2022", "task": "promoter_activity", "train_fraction": fraction,
            "n_visible": visible_count, "n_train": len(train_indices), "n_val": len(validation_indices), "n_test": test_count,
            "ensemble_size": args.ensemble_size, "parameter_count": parameter_count,
            "train_label_min": float(train_labels.min()), "train_label_max": float(train_labels.max()),
            **_prefixed("validation", validation_labels, val_mean), **_prefixed("test_top5pct", test_labels, test_mean),
        }
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
    _write_csv(output / "results.csv", summaries)
    _write_csv(output / "surrogate_results.csv", member_rows)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="BioSeqBench/dataset/yeast_gpra_dream2022")
    parser.add_argument("--cache-dir", default="BioSeqBench/results/dream2022_weak_cnn_cache")
    parser.add_argument("--output-dir", default="BioSeqBench/results/dream2022_weak_cnns")
    parser.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--top-test-fraction", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--prediction-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cache-chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--split-seed", type=int, default=20260827)
    parser.add_argument("--device")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
