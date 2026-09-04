"""Train the Zhao 2022 Ptrc XGBoost reproduction and a CNN comparator.

The article selected XGBoost as its best predictor.  It reports prediction on
the log10(fluorescence/OD600) scale (Figure 4C), but does not release the exact
training split, parameter file, or model code.  This script therefore uses the
benchmark's deterministic 80/10/10 split, positional nucleotide one-hot
features, and validation-set hyperparameter selection.  It is a transparent
functional reproduction rather than an assertion of bitwise paper identity.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from BioSeqBench.evaluation.metrics import compute_metrics
from BioSeqBench.models.zhao2022_ptrc_promoters import PtrcPromoterCNN
from BioSeqBench.models.zhao2022_ptrc_promoters.features import (
    log10_activity,
    one_hot_features,
    torch_one_hot,
)
from BioSeqBench.training.utils import set_seed


PAPER_FIGURE4C_METRICS = {"r2": 0.8846, "mae_log10": 0.15, "pearson": 0.94, "n": 127}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def metric_bundle(y_log: np.ndarray, prediction_log: np.ndarray) -> dict[str, float]:
    """Report paper-scale and original-scale metrics without conflating them."""
    return {
        "log10_activity": {**compute_metrics(y_log, prediction_log), "mae": float(mean_absolute_error(y_log, prediction_log))},
        "fluorescence_per_od600": compute_metrics(np.power(10.0, y_log), np.power(10.0, prediction_log)),
    }


def load_splits(data_root: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_dir = Path(data_root) / "zhao2022_ptrc_promoters"
    return tuple(pd.read_csv(dataset_dir / f"{split}.csv") for split in ("train", "val", "test"))  # type: ignore[return-value]


def xgb_candidates(seed: int, n_jobs: int) -> list[dict[str, object]]:
    """A compact validation grid suitable for this 3.7k-sequence library."""
    common: dict[str, object] = {
        "objective": "reg:squarederror", "n_estimators": 1000, "learning_rate": 0.03,
        "subsample": 0.9, "colsample_bytree": 1.0, "reg_lambda": 1.0,
        "random_state": seed, "n_jobs": n_jobs, "tree_method": "hist",
    }
    return [{**common, "max_depth": depth, "min_child_weight": min_child, "gamma": gamma}
            for depth, min_child, gamma in ((3, 1.0, 0.0), (4, 1.0, 0.0), (5, 1.0, 0.0), (4, 3.0, 0.0), (4, 1.0, 0.1))]


def train_xgboost(args: argparse.Namespace, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    x_train, x_val, x_test = (one_hot_features(frame["sequence"]) for frame in (train, val, test))
    y_train, y_val, y_test = (log10_activity(frame) for frame in (train, val, test))
    trials: list[dict] = []
    best_params: dict[str, object] | None = None
    best_score = float("inf")
    for params in xgb_candidates(args.seed, args.xgb_n_jobs):
        model = XGBRegressor(**params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        prediction = model.predict(x_val)
        score = mean_absolute_error(y_val, prediction)
        trial = {"params": params, "val_metrics": metric_bundle(y_val, prediction)}
        trials.append(trial)
        if score < best_score:
            best_score, best_params = score, params
    assert best_params is not None
    # After validation determines configuration, train on every non-test row.
    x_fit = np.concatenate([x_train, x_val])
    y_fit = np.concatenate([y_train, y_val])
    model = XGBRegressor(**best_params)
    model.fit(x_fit, y_fit, verbose=False)
    prediction = model.predict(x_test)
    output_dir = Path(args.checkpoint_dir) / "zhao2022_ptrc_promoters" / "paper_xgboost"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "model.json")
    pd.DataFrame({
        "sequence_id": test["sequence_id"], "sequence": test["sequence"], "true_log10_activity": y_test,
        "predicted_log10_activity": prediction, "true_fluorescence_per_od600": test["label"],
        "predicted_fluorescence_per_od600": np.power(10.0, prediction),
    }).to_csv(output_dir / "test_predictions.csv", index=False)
    result = {
        "timestamp": int(time.time()), "dataset": "zhao2022_ptrc_promoters", "model": "paper_xgboost",
        "protocol": "functional paper reproduction: positional A/C/G/T one-hot features; validation-selected XGBoost; log10(fluorescence/OD600) target",
        "exact_paper_hyperparameters_available": False, "seed": args.seed,
        "split_rows": {"train": len(train), "val": len(val), "test": len(test), "fit_after_selection": len(x_fit)},
        "paper_figure4c": PAPER_FIGURE4C_METRICS, "validation_trials": trials,
        "selected_params": best_params, "test_metrics": metric_bundle(y_test, prediction),
    }
    write_json(output_dir / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


@torch.no_grad()
def cnn_predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    values, predictions, total_loss, count = [], [], 0.0, 0
    loss_fn = nn.MSELoss()
    for sequence, label in loader:
        sequence, label = sequence.to(device), label.to(device)
        output = model(sequence)
        total_loss += loss_fn(output, label).item() * len(label)
        count += len(label)
        values.append(label.cpu().numpy())
        predictions.append(output.cpu().numpy())
    return np.concatenate(values), np.concatenate(predictions), total_loss / max(count, 1)


def make_loader(x: torch.Tensor, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(TensorDataset(x, torch.as_tensor(y, dtype=torch.float32)), batch_size=batch_size,
                      shuffle=shuffle, generator=torch.Generator().manual_seed(seed), num_workers=0,
                      pin_memory=torch.cuda.is_available())


def train_cnn(args: argparse.Namespace, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    x_train, x_val, x_test = (torch_one_hot(frame["sequence"]) for frame in (train, val, test))
    y_train, y_val, y_test = (log10_activity(frame) for frame in (train, val, test))
    fit_loader = make_loader(x_train, y_train, args.batch_size, True, args.seed)
    val_loader = make_loader(x_val, y_val, args.batch_size, False, args.seed)
    output_dir = Path(args.checkpoint_dir) / "zhao2022_ptrc_promoters" / "sequence_cnn"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = PtrcPromoterCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    loss_fn = nn.MSELoss()
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, rows = 0.0, 0
        for sequence, label in fit_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(sequence.to(device))
            loss = loss_fn(output, label.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += loss.item() * len(label)
            rows += len(label)
        _, _, val_loss = cnn_predict(model, val_loader, device)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_mse": train_loss / rows, "val_mse": val_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_mse": val_loss, "seed": args.seed}, output_dir / "best_validation.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    # Refit the selected duration on train+validation before the one-time test evaluation.
    set_seed(args.seed)
    x_fit, y_fit = torch.cat([x_train, x_val]), np.concatenate([y_train, y_val])
    final_loader = make_loader(x_fit, y_fit, args.batch_size, True, args.seed)
    final_model = PtrcPromoterCNN().to(device)
    final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    for _ in range(best_epoch):
        final_model.train()
        for sequence, label in final_loader:
            final_optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(final_model(sequence.to(device)), label.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(final_model.parameters(), 5.0)
            final_optimizer.step()
    test_loader = make_loader(x_test, y_test, args.batch_size, False, args.seed)
    actual, prediction, test_mse = cnn_predict(final_model, test_loader, device)
    torch.save({"model_state": final_model.state_dict(), "selected_epochs": best_epoch, "seed": args.seed,
                "target": "log10(fluorescence/OD600)"}, output_dir / "model.pt")
    pd.DataFrame({
        "sequence_id": test["sequence_id"], "sequence": test["sequence"], "true_log10_activity": actual,
        "predicted_log10_activity": prediction, "true_fluorescence_per_od600": test["label"],
        "predicted_fluorescence_per_od600": np.power(10.0, prediction),
    }).to_csv(output_dir / "test_predictions.csv", index=False)
    result = {
        "timestamp": int(time.time()), "dataset": "zhao2022_ptrc_promoters", "model": "sequence_cnn",
        "protocol": "BioSeqBench non-pretrained differentiable comparator; selected epoch on validation then refit train+validation; log10(fluorescence/OD600) target",
        "seed": args.seed, "device": str(device), "parameters": int(sum(p.numel() for p in final_model.parameters())),
        "split_rows": {"train": len(train), "val": len(val), "test": len(test), "fit_after_selection": len(x_fit)},
        "best_validation_epoch": best_epoch, "best_validation_mse": best_loss, "test_log10_mse": test_mse,
        "test_metrics": metric_bundle(actual, prediction), "history": history,
    }
    write_json(output_dir / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zhao 2022 Ptrc XGBoost reproduction and CNN comparator")
    parser.add_argument("--mode", choices=("all", "xgboost", "cnn"), default="all")
    parser.add_argument("--data-root", default="BioSeqBench/dataset")
    parser.add_argument("--checkpoint-dir", default="BioSeqBench/checkpoints/strong_zhao2022_ptrc_v1")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xgb-n-jobs", type=int, default=4)
    parser.add_argument("--device", help="Defaults to CUDA when free; use --device cpu to avoid sharing the GPU.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    train, val, test = load_splits(args.data_root)
    if args.mode in {"all", "xgboost"}:
        train_xgboost(args, train, val, test)
    if args.mode in {"all", "cnn"}:
        train_cnn(args, train, val, test)


if __name__ == "__main__":
    main()
