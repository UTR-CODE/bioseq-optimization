"""Faithful trainer for the autosome.org LegNet DREAM 2022 submission."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.special import ndtr
from torch import nn

from BioSeqBench.evaluation.metrics import compute_metrics
from BioSeqBench.models.yeast_gpra_dream2022 import LegNetAutosome, initialise_autosome_weights
from BioSeqBench.training.utils import set_seed


LEFT_ADAPTER_LENGTH = 17
# Exact final 150 bases preceding the random N80 insert in the plasmid context
# used by the authors' ``preprocess_data`` implementation.
DREAM_CONTEXT_150 = "AGTCCACTATTAAAGAACGTGGACTCCAACGTCAAAGGGCGAAAAACCGTCTATCAGGGCGATGGCCCACTACGTGAACCATCACCCTAATCAAGTGCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAACTGCATTTTTTTCACATC"
BASE_TO_CHANNEL = np.full(256, 4, dtype=np.int8)  # channel 4 represents N / ambiguous -> 0.25 OHE
BASE_TO_CHANNEL[ord("A")] = 0
BASE_TO_CHANNEL[ord("G")] = 1
BASE_TO_CHANNEL[ord("C")] = 2
BASE_TO_CHANNEL[ord("T")] = 3


def preprocess_sequence(sequence: str, length: int = 150) -> str:
    """Port of autosome's GPRA plasmid-context preprocessing."""
    sequence = str(sequence).upper()
    insert_and_right_adapter = sequence[LEFT_ADAPTER_LENGTH:]
    return (DREAM_CONTEXT_150 + insert_and_right_adapter)[-length:]


def encode_batch(sequences: list[str], labels: np.ndarray, rng: np.random.Generator, *, reverse_probability: float = 0.5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    processed = [preprocess_sequence(sequence) for sequence in sequences]
    characters = np.frombuffer("".join(processed).encode("ascii", "replace"), dtype=np.uint8).reshape(len(processed), 150)
    codes = BASE_TO_CHANNEL[characters]
    one_hot = np.zeros((len(processed), 4, 150), dtype=np.float32)
    rows = np.broadcast_to(np.arange(len(processed))[:, None], codes.shape)
    positions = np.broadcast_to(np.arange(150)[None, :], codes.shape)
    known = codes < 4
    one_hot[rows[known], codes[known], positions[known]] = 1.0
    one_hot[np.repeat((~known)[:, None, :], 4, axis=1)] = 0.25
    reverse = rng.random(len(processed)) < reverse_probability
    if reverse.any():
        one_hot[reverse] = one_hot[reverse][:, [3, 2, 1, 0], ::-1]
    singleton = np.isclose(labels, np.round(labels), rtol=0.0, atol=1e-7).astype(np.float32)
    inputs = np.concatenate([one_hot, np.broadcast_to(singleton[:, None, None], (len(processed), 1, 150))], axis=1)
    means = labels.astype(np.float64) + 0.5
    edges = np.concatenate(([-np.inf], np.arange(1, 18), [np.inf]))
    probabilities = np.diff(ndtr((edges[None, :] - means[:, None]) / 0.5), axis=1).astype(np.float32)
    return torch.from_numpy(inputs.copy()), torch.from_numpy(probabilities), torch.from_numpy(labels.astype(np.float32))


class ParquetBatchStream:
    """Infinite shuffled batch stream over the official DREAM FACS training observations."""

    def __init__(self, paths: list[Path], batch_size: int, seed: int) -> None:
        self.paths, self.batch_size, self.seed = paths, batch_size, seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        while True:
            for path_index in rng.permutation(len(self.paths)):
                parquet = pq.ParquetFile(self.paths[path_index])
                for record_batch in parquet.iter_batches(batch_size=131_072, columns=["sequence", "activity"]):
                    columns = record_batch.to_pydict()
                    sequences = columns["sequence"]
                    labels = np.asarray(columns["activity"], dtype=np.float32)
                    for indices in np.array_split(rng.permutation(len(labels)), range(self.batch_size, len(labels), self.batch_size)):
                        if len(indices) != self.batch_size:
                            continue
                        batch_sequences = [sequences[index] for index in indices]
                        yield encode_batch(batch_sequences, labels[indices], rng)


@torch.no_grad()
def evaluate(model: nn.Module, test_path: Path, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    parquet = pq.ParquetFile(test_path)
    targets, predictions = [], []
    rng = np.random.default_rng(0)
    for record_batch in parquet.iter_batches(batch_size=batch_size, columns=["sequence", "activity"]):
        columns = record_batch.to_pydict()
        labels = np.asarray(columns["activity"], dtype=np.float32)
        # Use forward and reverse views separately, exactly as the official test script.
        inputs, _, _ = encode_batch(columns["sequence"], labels, rng, reverse_probability=0.0)
        inputs = inputs.to(device, non_blocking=True)
        log_probs, forward = model(sequence_tensor=inputs)
        reverse_dna = inputs[:, :4, :][:, [3, 2, 1, 0], :].flip(dims=[2])
        reverse_inputs = torch.cat([reverse_dna, inputs[:, 4:5, :]], dim=1)
        _, reverse = model(sequence_tensor=reverse_inputs)
        predictions.append(((forward + reverse) / 2).cpu().numpy())
        targets.append(labels)
    return np.concatenate(targets), np.concatenate(predictions)


def train(args: argparse.Namespace) -> dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    raw_dir = Path(args.raw_root)
    # Keep the organizer-provided validation split fully held out for checkpoint
    # selection.  The original submission could train on all labelled FACS data,
    # but that protocol cannot support a leakage-free benchmark validation curve.
    train_paths = [raw_dir / "train-00000-of-00002.parquet", raw_dir / "train-00001-of-00002.parquet"]
    validation_path = raw_dir / "validation.parquet"
    if missing := [str(path) for path in train_paths + [validation_path, raw_dir / "test.parquet"] if not path.exists()]:
        raise FileNotFoundError("Missing DREAM parquet data: " + ", ".join(missing))
    model = LegNetAutosome().to(device)
    model.apply(initialise_autosome_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_learning_rate / 25.0, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.max_learning_rate, total_steps=args.steps, pct_start=0.3, div_factor=25.0,
    )
    criterion = nn.KLDivLoss(reduction="batchmean")
    stream = iter(ParquetBatchStream(train_paths, args.batch_size, args.seed))
    output_dir = Path(args.checkpoint_dir) / "yeast_gpra_dream2022" / "autosome_legnet"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_dir) / "yeast_dream2022_autosome.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    best_validation_pearson = float("-inf")
    best_step: int | None = None
    for step in range(1, args.steps + 1):
        inputs, probabilities, labels = next(stream)
        inputs, probabilities = inputs.to(device, non_blocking=True), probabilities.to(device, non_blocking=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        log_probabilities, score = model(sequence_tensor=inputs)
        loss = criterion(log_probabilities, probabilities)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step, "kl_loss": loss.item(), "batch_pearson": float(np.corrcoef(score.detach().cpu().numpy(), labels.numpy())[0, 1]),
                "learning_rate": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.monotonic() - start,
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % args.validation_every == 0 or step == args.steps:
            validation_start = time.monotonic()
            validation_target, validation_prediction = evaluate(model, validation_path, device, args.eval_batch_size)
            validation_metrics = compute_metrics(validation_target, validation_prediction)
            validation_record = {
                "step": step,
                "split": "validation",
                **{f"validation_{name}": value for name, value in validation_metrics.items()},
                "validation_elapsed_seconds": time.monotonic() - validation_start,
                "elapsed_seconds": time.monotonic() - start,
            }
            if validation_metrics["pearson"] > best_validation_pearson:
                best_validation_pearson = validation_metrics["pearson"]
                best_step = step
                torch.save(
                    {
                        "model_state": model.state_dict(), "step": step, "seed": args.seed,
                        "official_config": True, "validation_metrics": validation_metrics,
                    },
                    output_dir / "best_val.pt",
                )
                validation_record["is_best_validation"] = True
            else:
                validation_record["is_best_validation"] = False
            with log_path.open("a") as handle:
                handle.write(json.dumps(validation_record) + "\n")
            print(json.dumps(validation_record), flush=True)
            model.train()
        if step % args.checkpoint_every == 0 or step == args.steps:
            torch.save({"model_state": model.state_dict(), "step": step, "seed": args.seed, "official_config": True}, output_dir / f"step_{step}.pt")
    target, prediction = evaluate(model, raw_dir / "test.parquet", device, args.eval_batch_size)
    last_metrics = compute_metrics(target, prediction)
    best_checkpoint = torch.load(output_dir / "best_val.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state"])
    _, best_prediction = evaluate(model, raw_dir / "test.parquet", device, args.eval_batch_size)
    best_metrics = compute_metrics(target, best_prediction)
    result: dict[str, object] = {
        "dataset": "yeast_gpra_dream2022", "model": "autosome_legnet", "seed": args.seed,
        "steps": args.steps, "batch_size": args.batch_size, "train_source_rows": 6065325,
        "validation_rows": 673925, "test_rows": len(target), "best_validation_step": best_step,
        "protocol": "Autosome official architecture/training: 150-bp plasmid context, singleton channel, random RC augmentation, soft 18-bin KL classification, AdamW/OneCycle; forward/RC test averaging.  The organizer validation split is evaluated every 5,000 steps and selects best_val without test-set selection.",
        "test_scale_warning": "Test labels are MAUDE expression on an affine scale distinct from train FACS bins; Pearson/Spearman are primary.",
        **{f"last_test_{name}": value for name, value in last_metrics.items()},
        **{f"best_val_test_{name}": value for name, value in best_metrics.items()},
    }
    np.savez_compressed(output_dir / "test_predictions.npz", y_true=target, last_prediction=prediction, best_val_prediction=best_prediction)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autosome.org LegNet reproduction for DREAM 2022")
    parser.add_argument("--raw-root", default="BioSeqBench/raw_data/yeast_gpra_dream2022")
    parser.add_argument("--steps", type=int, default=80_000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--max-learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--validation-every", type=int, default=5_000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint-dir", default="BioSeqBench/checkpoints/strong_yeast_dream2022_autosome_v3_validation_5k")
    parser.add_argument("--log-dir", default="BioSeqBench/logs/strong_yeast_dream2022_autosome_v3_validation_5k")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
