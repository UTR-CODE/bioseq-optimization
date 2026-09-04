import numpy as np

from BioSeqBench.training.train_utr5_weak_cnns import (
    _split_visible_pool,
    build_parser,
)


def test_visible_pool_validation_split_is_reproducible_and_disjoint():
    fit_a, validation_a = _split_visible_pool(1_000, 0.1, 123)
    fit_b, validation_b = _split_visible_pool(1_000, 0.1, 123)
    assert np.array_equal(fit_a, fit_b)
    assert np.array_equal(validation_a, validation_b)
    assert len(fit_a) == 900
    assert len(validation_a) == 100
    assert not np.intersect1d(fit_a, validation_a).size
    assert np.array_equal(np.sort(np.concatenate((fit_a, validation_a))), np.arange(1_000))


def test_utr5_training_defaults_follow_weakseqcnn_protocol():
    args = build_parser().parse_args([])
    assert args.ensemble_size == 5
    assert args.epochs == 100
    assert args.patience == 10
    assert args.learning_rate == 1e-3
    assert args.weight_decay == 1e-4
    assert args.batch_size == 1024
    assert args.validation_fraction == 0.10
    assert args.top_test_fraction == 0.05
    assert args.fractions == [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
