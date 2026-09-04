from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score, roc_auc_score


def _safe_correlation(function, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(function(y_true, y_pred).statistic)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str = "regression") -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]
    if not len(y_true):
        raise ValueError("No finite prediction/target pairs")
    if task == "classification":
        probabilities = 1 / (1 + np.exp(-y_pred))
        return {
            "auroc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) > 1 else float("nan"),
            "auprc": float(average_precision_score(y_true, probabilities)),
        }
    return {
        "pearson": _safe_correlation(pearsonr, y_true, y_pred),
        "spearman": _safe_correlation(spearmanr, y_true, y_pred),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def compute_multitask_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str], task: str = "regression") -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true, y_pred = y_true[:, None], y_pred[:, None]
    result: dict[str, float] = {}
    collected: dict[str, list[float]] = {}
    for index, name in enumerate(names):
        metrics = compute_metrics(y_true[:, index], y_pred[:, index], task)
        for metric, value in metrics.items():
            result[f"{name}/{metric}"] = value
            collected.setdefault(metric, []).append(value)
    for metric, values in collected.items():
        result[f"mean/{metric}"] = float(np.nanmean(values))
    return result

