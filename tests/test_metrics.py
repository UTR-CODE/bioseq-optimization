import numpy as np

from BioSeqBench.evaluation.metrics import compute_metrics


def test_regression_metrics():
    metrics = compute_metrics(np.arange(5), np.arange(5))
    assert metrics["pearson"] == 1.0
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["mse"] == 0.0


import pytest
