"""Reproduce MPRA-LegNet training on Agarwal et al. lentiMPRA libraries.

The author release trains separate one-output models for HepG2, K562 and
WTC11.  ``--mode demo`` reproduces the repository's inexpensive configuration
(test fold 1, validation fold 2).  ``--mode 10fold`` is BioSeqBench's practical
strong-model protocol: one deterministic validation fold per outer test fold.
``--mode cv`` executes the full 10 x 9 nested cross-validation protocol and is
intentionally opt-in.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from BioSeqBench.evaluation.metrics import compute_metrics
from BioSeqBench.models.agarwal2025_lentimpra import MPRALegNet, initialise_mpralegnet_weights
from BioSeqBench.training.utils import set_seed


# The source implementation uses this channel order, not BioSeqBench's generic
# A/C/G/T order.  Retaining it makes source checkpoints and parameters match.
BASE_CODES = np.full(256, 4, dtype=np.uint8)  # unknown bases are soft one-hot Ns
for _index, _base in enumerate("AGCT"):
    BASE_CODES[ord(_base)] = _index
COMPLEMENT = str.maketrans("AGCTN", "TCGAN")
RIGHT_FLANK = "CACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGT"
SEQUENCE_LENGTH = 230
CELL_DATASETS = {
    "hepg2": "agarwal2025_lentimpra_hepg2",
    "k562": "agarwal2025_lentimpra_k562",
    "wtc11": "agarwal2025_lentimpra_wtc11",
}


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def _encode(sequence: str, reverse_channel: float, *, use_reverse_channel: bool) -> torch.Tensor:
    characters = np.frombuffer(sequence.encode("ascii", "replace"), dtype=np.uint8)
    if len(characters) != SEQUENCE_LENGTH:
        raise ValueError(f"MPRALegNet requires {SEQUENCE_LENGTH}-nt constructs, received {len(characters)}")
    codes = BASE_CODES[characters]
    encoded = np.zeros((5 if use_reverse_channel else 4, SEQUENCE_LENGTH), dtype=np.float32)
    known = codes < 4
    positions = np.flatnonzero(known)
    encoded[codes[known], positions] = 1.0
    if np.any(~known):
        encoded[:4, ~known] = 0.25
    if use_reverse_channel:
        encoded[4, :] = reverse_channel
    return torch.from_numpy(encoded)


class _MPRALegNetDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, frame: pd.DataFrame, *, training: bool, use_shift: bool, reverse_augment: bool, use_reverse_channel: bool) -> None:
        self.sequences = frame["sequence"].astype(str).to_numpy(dtype="S230")
        self.labels = frame["label"].to_numpy(dtype=np.float32, copy=True)
        self.training = training
        self.use_shift = use_shift
        self.reverse_augment = reverse_augment
        self.use_reverse_channel = use_reverse_channel

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[index].decode("ascii")
        if self.training and self.use_shift:
            # The source default is max_shift=(0, 21): shift left and append
            # the known 3-prime vector context, including shift=0.
            shift = int(torch.randint(low=0, high=22, size=(1,)).item())
            if shift:
                sequence = sequence[shift:] + RIGHT_FLANK[:shift]
        reverse = 0.0
        if self.training and self.reverse_augment and torch.rand(()) > 0.5:
            sequence = _reverse_complement(sequence)
            reverse = 1.0
        return _encode(sequence, reverse, use_reverse_channel=self.use_reverse_channel), torch.tensor(self.labels[index], dtype=torch.float32)


def _load_cell(data_root: Path, cell: str) -> pd.DataFrame:
    dataset = data_root / CELL_DATASETS[cell]
    frames = [pd.read_csv(dataset / f"{split}.csv") for split in ("train", "val", "test")]
    frame = pd.concat(frames, ignore_index=True)
    required = {"sequence", "label", "publication_fold"}
    if not required <= set(frame):
        raise ValueError(f"{dataset} is missing required MPRALegNet columns: {required - set(frame)}")
    if not frame["length"].eq(SEQUENCE_LENGTH).all():
        raise ValueError(f"{dataset} must retain the complete 230-nt construct, including adapters")
    return frame


def _load_averaged_labels(averaged_root: Path, cell: str) -> pd.Series:
    """Read the authors' denoised evaluation labels for Fig. 3-style scoring.

    ``<cell>_averaged.tsv`` has the same constructs/folds as the training TSV
    but its mean activity values are averaged for final performance reporting.
    The source code filters its auxiliary ``rev`` full-construct copies before
    fitting, so this lookup follows the same policy.
    """
    filename = {"hepg2": "HepG2_averaged.tsv", "k562": "K562_averaged.tsv", "wtc11": "WTC11_averaged.tsv"}[cell]
    path = averaged_root / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing author averaged evaluation table: {path}")
    frame = pd.read_csv(path, sep="\t")
    required = {"seq_id", "mean_value"}
    if not required <= set(frame):
        raise ValueError(f"Unexpected averaged table columns: {list(frame.columns)}")
    if "rev" in frame:
        frame = frame.loc[pd.to_numeric(frame["rev"], errors="raise").eq(0)].copy()
    labels = pd.to_numeric(frame["mean_value"], errors="raise")
    if frame["seq_id"].duplicated().any() or labels.isna().any():
        raise ValueError(f"Averaged labels must have one finite value per seq_id: {path}")
    return pd.Series(labels.to_numpy(dtype=np.float32), index=frame["seq_id"].astype(str), name="averaged_label")


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device, *, reverse: bool = False, max_batches: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, predictions = [], []
    for batch_index, (inputs, target) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if reverse:
            # The source test dataset uses a complete reverse complement and a
            # fifth channel of ones for the reverse orientation.
            channels = [3, 2, 1, 0] + ([4] if inputs.shape[1] == 5 else [])
            inputs = inputs[:, channels, :].flip(dims=[2])
            if inputs.shape[1] == 5:
                inputs[:, 4, :] = 1.0
        inputs = inputs.to(device, non_blocking=True)
        output = model(sequence_tensor=inputs).squeeze(1)
        labels.append(target.numpy())
        predictions.append(output.cpu().float().numpy())
    return np.concatenate(labels), np.concatenate(predictions)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int | None) -> dict[str, float]:
    target, forward = _predict(model, loader, device, max_batches=max_batches)
    reverse_target, reverse = _predict(model, loader, device, reverse=True, max_batches=max_batches)
    if not np.array_equal(target, reverse_target):
        raise RuntimeError("Forward and reverse predictions lost row alignment")
    mean_prediction = (forward + reverse) / 2.0
    metrics: dict[str, float] = {}
    for prefix, prediction in (("forward", forward), ("reverse", reverse), ("tta", mean_prediction)):
        metrics.update({f"{prefix}_{name}": value for name, value in compute_metrics(target, prediction).items()})
    return metrics


@torch.no_grad()
def _predict_tta(model: nn.Module, loader: DataLoader, device: torch.device, max_batches: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target, forward = _predict(model, loader, device, max_batches=max_batches)
    reverse_target, reverse = _predict(model, loader, device, reverse=True, max_batches=max_batches)
    if not np.array_equal(target, reverse_target):
        raise RuntimeError("Forward and reverse predictions lost row alignment")
    return target, forward, reverse


def _append_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        if not exists:
            writer.writeheader()
        writer.writerow(result)


def _run_fold(args: argparse.Namespace, cell: str, frame: pd.DataFrame, averaged_labels: pd.Series, validation_fold: int, test_fold: int) -> dict[str, object]:
    if validation_fold == test_fold:
        raise ValueError("validation_fold and test_fold must differ")
    train = frame.loc[~frame["publication_fold"].isin([validation_fold, test_fold])].reset_index(drop=True)
    validation = frame.loc[frame["publication_fold"].eq(validation_fold)].reset_index(drop=True)
    test = frame.loc[frame["publication_fold"].eq(test_fold)].reset_index(drop=True)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(f"Empty CV partition for {cell}, val={validation_fold}, test={test_fold}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    generator = torch.Generator().manual_seed(args.seed + validation_fold * 100 + test_fold)
    # A full source-CV run executes 90 folds.  Persistent worker iterators for
    # train, validation and test loaders retain pipe descriptors across folds
    # under CPython's delayed cleanup, eventually hitting RLIMIT_NOFILE.
    # Keep worker parallelism for training but run the much smaller repeated
    # validation/test passes in-process; this matches the data semantics while
    # making long resume runs descriptor-stable.
    train_loader_options = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda", "persistent_workers": False}
    eval_loader_options = {"num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(
        _MPRALegNetDataset(train, training=True, use_shift=args.use_shift, reverse_augment=args.reverse_augment, use_reverse_channel=args.use_reverse_channel),
        batch_size=args.batch_size, shuffle=True, generator=generator, **train_loader_options,
    )
    val_loader = DataLoader(
        _MPRALegNetDataset(validation, training=False, use_shift=False, reverse_augment=False, use_reverse_channel=args.use_reverse_channel),
        batch_size=args.eval_batch_size, shuffle=False, **eval_loader_options,
    )
    test_loader = DataLoader(
        _MPRALegNetDataset(test, training=False, use_shift=False, reverse_augment=False, use_reverse_channel=args.use_reverse_channel),
        batch_size=args.eval_batch_size, shuffle=False, **eval_loader_options,
    )
    model = MPRALegNet(input_channels=5 if args.use_reverse_channel else 4).to(device)
    model.apply(initialise_mpralegnet_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_learning_rate / 25.0, weight_decay=args.weight_decay)
    steps_per_epoch = min(len(train_loader), args.max_train_batches) if args.max_train_batches else len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.max_learning_rate, total_steps=max(1, steps_per_epoch * args.epochs),
        pct_start=0.3, three_phase=False, cycle_momentum=False,
    )
    use_amp = device.type == "cuda" and not args.disable_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    root = Path(args.checkpoint_dir) / CELL_DATASETS[cell] / f"model_{validation_fold}_{test_fold}"
    root.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / f"{cell}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_pearson, best_epoch = -float("inf"), 0
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch_index, (inputs, target) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            inputs, target = inputs.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(sequence_tensor=inputs).squeeze(1)
                loss = nn.functional.mse_loss(prediction, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # AMP can skip the first optimizer step while lowering its scale.
            # OneCycle must advance only when the optimizer actually advanced.
            if not use_amp or scaler.get_scale() >= scale_before_step:
                scheduler.step()
            loss_sum += loss.detach().float().item() * len(target)
            seen += len(target)
        validation_metrics = _evaluate(model, val_loader, device, args.max_eval_batches)
        record: dict[str, object] = {
            "cell_type": cell, "validation_fold": validation_fold, "test_fold": test_fold, "epoch": epoch,
            "train_mse": loss_sum / max(1, seen), "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.monotonic() - started,
            **{f"validation_{name}": value for name, value in validation_metrics.items()},
        }
        if validation_metrics["tta_pearson"] > best_pearson:
            best_pearson, best_epoch = validation_metrics["tta_pearson"], epoch
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "validation_tta_pearson": best_pearson}, root / "best.pt")
            record["is_best_validation"] = True
        else:
            record["is_best_validation"] = False
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    state = torch.load(root / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model_state"])
    raw_target, forward_prediction, reverse_prediction = _predict_tta(model, test_loader, device, args.max_eval_batches)
    tta_prediction = (forward_prediction + reverse_prediction) / 2.0
    raw_test_metrics: dict[str, float] = {}
    for prefix, prediction in (("forward", forward_prediction), ("reverse", reverse_prediction), ("tta", tta_prediction)):
        raw_test_metrics.update({f"{prefix}_{name}": value for name, value in compute_metrics(raw_target, prediction).items()})
    # Smoke mode may intentionally cap evaluation batches.  DataLoader is
    # sequential, so restrict the source table to the same leading rows before
    # joining the averaged labels.
    evaluation_test = test.iloc[:len(raw_target)].copy()
    averaged_target = evaluation_test["element_id"].astype(str).map(averaged_labels)
    if averaged_target.isna().any():
        missing = test.loc[averaged_target.isna(), "element_id"].iloc[0]
        raise ValueError(f"Missing averaged label for {cell} element {missing!r}")
    averaged_test_metrics: dict[str, float] = {}
    for prefix, prediction in (("forward", forward_prediction), ("reverse", reverse_prediction), ("tta", tta_prediction)):
        averaged_test_metrics.update({f"{prefix}_{name}": value for name, value in compute_metrics(averaged_target.to_numpy(), prediction).items()})
    pd.DataFrame({
        "element_id": evaluation_test["element_id"], "sequence": evaluation_test["sequence"], "publication_fold": evaluation_test["publication_fold"],
        "label_training_raw": raw_target, "label_evaluation_averaged": averaged_target.to_numpy(),
        "prediction_forward": forward_prediction, "prediction_reverse": reverse_prediction, "prediction_tta": tta_prediction,
    }).to_csv(root / "test_predictions.csv", index=False)
    result: dict[str, object] = {
        "dataset": CELL_DATASETS[cell], "model": "mpralegnet", "cell_type": cell,
        "validation_fold": validation_fold, "test_fold": test_fold, "best_epoch": best_epoch,
        "best_validation_tta_pearson": best_pearson, "train_rows": len(train),
        "validation_rows": len(validation), "test_rows": len(test),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "protocol": "Author human_legnet: rev=0 training rows, 230-nt construct, AGCT channels, optional reverse channel, shift 0..21, random RC, AdamW, OneCycle, gradient clipping; final score uses author *_averaged.tsv labels.",
        "use_reverse_channel": args.use_reverse_channel,
        "evaluation_target": "author averaged mean_value (rev=0); raw training-table metrics retained separately",
        "smoke_limited": args.max_train_batches is not None or args.max_eval_batches is not None,
        **{f"test_raw_{name}": value for name, value in raw_test_metrics.items()},
        **{f"test_{name}": value for name, value in averaged_test_metrics.items()},
    }
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    _append_result(Path(args.results_file), result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def _fold_pairs(mode: str) -> Iterable[tuple[int, int]]:
    if mode == "demo":
        yield 2, 1  # author repository demo: fold 1 test, fold 2 validation
        return
    if mode == "10fold":
        # One reproducible inner-validation choice per outer fold.  This gives
        # ten independent held-out models per cell type, not the source repo's
        # exhaustive 90-model validation sweep.
        for test_fold in range(1, 11):
            validation_fold = test_fold % 10 + 1
            yield validation_fold, test_fold
        return
    for test_fold in range(1, 11):
        for validation_fold in range(1, 11):
            if validation_fold != test_fold:
                yield validation_fold, test_fold


def train(args: argparse.Namespace) -> list[dict[str, object]]:
    set_seed(args.seed)
    selected = [value.strip().lower() for value in args.cells.split(",") if value.strip()]
    invalid = sorted(set(selected) - set(CELL_DATASETS))
    if invalid:
        raise ValueError(f"Unknown cell types: {invalid}; choose from {sorted(CELL_DATASETS)}")
    results = []
    for cell in selected:
        frame = _load_cell(Path(args.data_root), cell)
        averaged_labels = _load_averaged_labels(Path(args.averaged_data_root), cell)
        for validation_fold, test_fold in _fold_pairs(args.mode):
            result_path = (
                Path(args.checkpoint_dir) / CELL_DATASETS[cell]
                / f"model_{validation_fold}_{test_fold}" / "result.json"
            )
            if args.skip_complete and result_path.exists():
                result = json.loads(result_path.read_text())
                result["resumed_skip"] = True
                print(json.dumps({"skipped_completed_model": str(result_path)}), flush=True)
                results.append(result)
                continue
            set_seed(args.seed + validation_fold * 100 + test_fold)
            results.append(_run_fold(args, cell, frame, averaged_labels, validation_fold, test_fold))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Faithful MPRA-LegNet reproduction for Agarwal lentiMPRA")
    parser.add_argument("--data-root", default="BioSeqBench/dataset")
    parser.add_argument("--cells", default="hepg2,k562,wtc11")
    parser.add_argument("--mode", choices=("demo", "10fold", "cv"), default="demo")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--max-learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--use-shift", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reverse-augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-reverse-channel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--averaged-data-root", default="BioSeqBench/raw_data/agarwal2025_lentimpra/human_legnet")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--skip-complete", action="store_true", help="When resuming, retain any existing result.json instead of retraining that fold pair.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Smoke-test cap per epoch; never use for final results.")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Smoke-test cap per evaluation; never use for final results.")
    parser.add_argument("--checkpoint-dir", default="BioSeqBench/checkpoints/strong_agarwal2025_mpralegnet_v1")
    parser.add_argument("--log-dir", default="BioSeqBench/logs/strong_agarwal2025_mpralegnet_v1")
    parser.add_argument("--results-file", default="BioSeqBench/results/agarwal2025_mpralegnet.csv")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
