"""Faithful single-cell PARM training on the published K562 MPRA folds.

The original release trains an independent PARM model for each cell type and
each of five cross-validation folds.  This trainer preserves that protocol for
K562 while reading BioSeqBench's canonical CSV files in streaming chunks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn

from BioSeqBench.evaluation.metrics import compute_metrics
from BioSeqBench.models.parm import PARMExactCNN
from BioSeqBench.training.utils import set_seed


LABEL_COLUMN = "label_log2rpm_k562"
LENGTH = 600
ADAPTER_5 = "CAGTGAT"
ADAPTER_3 = "CACGACG"  # promoter-focused-library vector adapter from the paper
BASE_CODES = np.full(256, -1, dtype=np.int8)
for _index, _base in enumerate("ACGT"):
    BASE_CODES[ord(_base)] = _index


def _encode_batch(sequences: list[str], rng: np.random.Generator) -> torch.Tensor:
    """Port PARM's ``pad_collate``: adapter addition plus random zero padding."""
    encoded = np.zeros((len(sequences), 4, LENGTH), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        sequence = str(sequence).upper()
        if len(sequence) + len(ADAPTER_3) <= LENGTH:
            sequence += ADAPTER_3
        if len(sequence) + len(ADAPTER_5) <= LENGTH:
            sequence = ADAPTER_5 + sequence
        if len(sequence) > LENGTH:
            sequence = sequence[:LENGTH]
        offset = int(rng.integers(0, LENGTH - len(sequence) + 1))
        characters = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)
        codes = BASE_CODES[characters]
        valid = codes >= 0
        encoded[row, codes[valid], offset + np.flatnonzero(valid)] = 1.0
    return torch.from_numpy(encoded)


def _split_paths(dataset_dir: Path, folds: set[str]) -> list[Path]:
    # The held-out test fold is stored separately from the five CV folds.
    return [dataset_dir / "test.csv"] if folds == {"test"} else [dataset_dir / "train.csv", dataset_dir / "val.csv"]


def _batch_stream(
    dataset_dir: Path,
    folds: set[str],
    batch_size: int,
    seed: int,
    *,
    epoch: int,
    include_features: bool = False,
    shuffle: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, list[str] | None]]:
    """Read one shuffled epoch without materializing the 2.8-GB CSV dataset."""
    rng = np.random.default_rng(seed + epoch * 10_007)
    paths = _split_paths(dataset_dir, folds)
    usecols = ["sequence", LABEL_COLUMN, "source_fold"]
    if include_features:
        usecols.append("feature_id")
    path_order = rng.permutation(len(paths)) if shuffle else range(len(paths))
    for path_index in path_order:
        path = paths[path_index]
        for frame in pd.read_csv(path, usecols=usecols, chunksize=65_536):
            frame = frame.loc[frame["source_fold"].astype(str).isin(folds)]
            if frame.empty:
                continue
            permutation = rng.permutation(len(frame)) if shuffle else np.arange(len(frame))
            for start in range(0, len(permutation), batch_size):
                indices = permutation[start:start + batch_size]
                selected = frame.iloc[indices]
                sequences = selected["sequence"].astype(str).tolist()
                labels = torch.from_numpy(selected[LABEL_COLUMN].to_numpy(dtype=np.float32, copy=True)).unsqueeze(1)
                feature_ids = selected["feature_id"].astype(str).tolist() if include_features else None
                yield _encode_batch(sequences, rng), labels, feature_ids


def _regularized_poisson_loss(prediction: torch.Tensor, target: torch.Tensor, model: nn.Module, beta_1: float, beta_2: float) -> torch.Tensor:
    loss = nn.functional.poisson_nll_loss(prediction, target, log_input=False, reduction="mean")
    # Retain the official source implementation verbatim: both reported beta
    # terms use the L2 parameter norm (the first is named beta_1 in their CLI).
    if beta_1 or beta_2:
        norm = sum(torch.norm(parameter, p=2) for parameter in model.parameters())
        loss = loss + norm * (beta_1 + beta_2)
    return loss


def _set_learning_rate(optimizer: torch.optim.Optimizer, step: int, total_steps: int) -> float:
    """Paper schedule: 5k-batch 2e-8→1e-4 warm-up, then cosine decay."""
    if step <= 5_000:
        learning_rate = 2e-8 + (1e-4 - 2e-8) * step / 5_000
    else:
        progress = min(1.0, (step - 5_000) / max(1, total_steps - 5_000))
        learning_rate = 1e-4 * (1.0 + math.cos(math.pi * progress)) / 2.0
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset_dir: Path,
    folds: set[str],
    device: torch.device,
    batch_size: int,
    seed: int,
    *,
    epoch: int,
    save_predictions: bool = False,
    shuffle: bool = True,
) -> tuple[dict[str, float], np.ndarray | None, np.ndarray | None]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    feature_sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    total_loss = total = 0
    for inputs, labels, feature_ids in _batch_stream(
        dataset_dir, folds, batch_size, seed, epoch=epoch, include_features=True, shuffle=shuffle,
    ):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        output = model(sequence_tensor=inputs)
        total_loss += nn.functional.poisson_nll_loss(output, labels, log_input=False, reduction="sum").item()
        total += len(labels)
        truth = labels[:, 0].cpu().numpy()
        prediction = output[:, 0].cpu().numpy()
        for feature_id, true_value, predicted_value in zip(feature_ids or [], truth, prediction):
            values = feature_sums[feature_id]
            values[0] += float(true_value)
            values[1] += float(predicted_value)
            values[2] += 1.0
        if save_predictions:
            targets.append(truth)
            predictions.append(prediction)
    fragment_true = np.concatenate(targets) if save_predictions else None
    fragment_pred = np.concatenate(predictions) if save_predictions else None
    promoter_true = np.asarray([value[0] / value[2] for value in feature_sums.values()])
    promoter_pred = np.asarray([value[1] / value[2] for value in feature_sums.values()])
    metrics = {
        "poisson_nll": total_loss / max(1, total),
        **{f"promoter/{name}": value for name, value in compute_metrics(promoter_true, promoter_pred).items()},
    }
    if fragment_true is not None and fragment_pred is not None:
        metrics.update({f"fragment/{name}": value for name, value in compute_metrics(fragment_true, fragment_pred).items()})
    return metrics, fragment_true, fragment_pred


def train_one_fold(args: argparse.Namespace, fold: int) -> dict[str, object]:
    dataset_dir = Path(args.data_root) / "parm_v2_mh4"
    validation_fold = f"fold{fold}"
    all_folds = {f"fold{index}" for index in range(5)}
    training_folds = all_folds - {validation_fold}
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = PARMExactCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-8, weight_decay=args.weight_decay)
    output_dir = Path(args.checkpoint_dir) / "parm_v2_mh4" / "parm_exact_k562" / f"fold{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / "parm_v2_mh4_parm_exact_k562.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    train_rows = 3_100_172  # four of the five published training folds after canonical QC
    steps_per_epoch = math.ceil(train_rows / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    step, best_validation, best_epoch = 0, float("inf"), 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = seen = 0.0
        for inputs, labels, _ in _batch_stream(dataset_dir, training_folds, args.batch_size, args.seed, epoch=epoch):
            step += 1
            learning_rate = _set_learning_rate(optimizer, step, total_steps)
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(sequence_tensor=inputs)
            loss = _regularized_poisson_loss(output, labels, model, args.beta_1, args.beta_2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
            optimizer.step()
            train_loss += loss.item() * len(labels)
            seen += len(labels)
        validation, _, _ = evaluate(
            model, dataset_dir, {validation_fold}, device, args.batch_size, args.seed + 100_000,
            epoch=epoch,
        )
        record: dict[str, object] = {
            "fold": fold, "epoch": epoch, "step": step, "learning_rate": learning_rate,
            "train_loss": train_loss / max(1, seen), "elapsed_seconds": time.monotonic() - started,
            **{f"validation_{name.replace('/', '_')}": value for name, value in validation.items()},
        }
        if validation["poisson_nll"] < best_validation:
            best_validation, best_epoch = validation["poisson_nll"], epoch
            torch.save({"model_state": model.state_dict(), "fold": fold, "epoch": epoch, "paper_protocol": True}, output_dir / "best_val.pt")
            record["is_best_validation"] = True
        else:
            record["is_best_validation"] = False
        torch.save({"model_state": model.state_dict(), "fold": fold, "epoch": epoch, "paper_protocol": True}, output_dir / f"epoch_{epoch}.pt")
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    torch.save({"model_state": model.state_dict(), "fold": fold, "epoch": args.epochs, "paper_protocol": True}, output_dir / "last.pt")
    # The authors' trainer always serializes the model after its fixed tenth
    # epoch; validation is not used for early stopping.  ``best_val.pt`` stays
    # available as a benchmark diagnostic, but must not replace the published
    # five-fold ensemble member.
    state = torch.load(output_dir / "last.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])
    test_metrics, target, prediction = evaluate(
        model, dataset_dir, {"test"}, device, args.batch_size, args.seed + 200_000,
        epoch=0, save_predictions=True, shuffle=False,
    )
    np.savez_compressed(output_dir / "test_predictions.npz", y_true=target, y_pred=prediction)
    result: dict[str, object] = {
        "dataset": "parm_v2_mh4", "model": "parm_exact_k562", "fold": fold, "seed": args.seed,
        "best_validation_epoch": best_epoch, "primary_checkpoint": "last.pt",
        "epochs": args.epochs, "batch_size": args.batch_size,
        "train_folds": sorted(training_folds), "validation_fold": validation_fold, "test_fold": "test",
        "protocol": "Official PARM CNN / K562 single-task: 600bp, CAGTGAT/CACGACG adapters, random padding, PoissonNLL, AdamW, 10 epochs, 5k warm-up and cosine decay.",
        **{f"test_{name.replace('/', '_')}": value for name, value in test_metrics.items()},
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def _ensemble_result(args: argparse.Namespace, folds: list[int]) -> dict[str, object]:
    """Average the five independently trained PARM folds on the held-out test."""
    root = Path(args.checkpoint_dir) / "parm_v2_mh4" / "parm_exact_k562"
    archives = [np.load(root / f"fold{fold}" / "test_predictions.npz") for fold in folds]
    target = archives[0]["y_true"]
    if any(not np.array_equal(target, archive["y_true"]) for archive in archives[1:]):
        raise ValueError("PARM fold test predictions are not aligned")
    prediction = np.mean(np.stack([archive["y_pred"] for archive in archives]), axis=0)
    feature_ids = pd.read_csv(Path(args.data_root) / "parm_v2_mh4" / "test.csv", usecols=["feature_id"])["feature_id"].astype(str).to_numpy()
    if len(feature_ids) != len(prediction):
        raise ValueError("PARM test prediction length does not match canonical test CSV")
    aggregate: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for feature_id, true_value, predicted_value in zip(feature_ids, target, prediction):
        values = aggregate[feature_id]
        values[0] += float(true_value); values[1] += float(predicted_value); values[2] += 1.0
    promoter_true = np.asarray([value[0] / value[2] for value in aggregate.values()])
    promoter_pred = np.asarray([value[1] / value[2] for value in aggregate.values()])
    result: dict[str, object] = {
        "dataset": "parm_v2_mh4", "model": "parm_exact_k562_5fold_ensemble", "folds": folds,
        "primary_checkpoint": "last.pt", "test_rows": len(target), "test_features": len(promoter_true),
        **{f"test_fragment_{name}": value for name, value in compute_metrics(target, prediction).items()},
        **{f"test_promoter_{name}": value for name, value in compute_metrics(promoter_true, promoter_pred).items()},
    }
    np.savez_compressed(root / "ensemble_test_predictions.npz", y_true=target, y_pred=prediction)
    (root / "ensemble_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exact PARM K562 single-task reproduction")
    parser.add_argument("--data-root", default="BioSeqBench/dataset")
    parser.add_argument("--folds", default="0,1,2,3,4", help="comma-separated PARM validation folds")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--beta-1", type=float, default=1e-5)
    parser.add_argument("--beta-2", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint-dir", default="BioSeqBench/checkpoints/strong_parm_k562_paper_v3_5fold_ensemble")
    parser.add_argument("--log-dir", default="BioSeqBench/logs/strong_parm_k562_paper_v3_5fold_ensemble")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    folds = [int(value) for value in args.folds.split(",")]
    if not folds or any(value not in range(5) for value in folds):
        raise ValueError("--folds must be a non-empty subset of 0,1,2,3,4")
    results = [train_one_fold(args, fold) for fold in folds]
    ensemble = _ensemble_result(args, folds) if len(folds) > 1 else None
    print(json.dumps({"completed_folds": folds, "results": results, "ensemble": ensemble}, indent=2), flush=True)


if __name__ == "__main__":
    main()
