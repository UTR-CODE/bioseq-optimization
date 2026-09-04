import numpy as np

from BioSeqBench.training.train_yeast_dream2022_weak_cnns import (
    PAD_CODE,
    _encode_strings,
    build_parser,
)


def test_dream_cache_encoding_preserves_variable_length_dna():
    codes, lengths = _encode_strings(np.asarray(["ACGT", "TTA"]), max_length=5)
    assert lengths.tolist() == [4, 3]
    assert codes.tolist() == [[0, 1, 2, 3, PAD_CODE], [3, 3, 0, PAD_CODE, PAD_CODE]]


def test_dream_weakseqcnn_defaults_match_standardized_protocol():
    args = build_parser().parse_args([])
    assert args.batch_size == 1024
    assert args.ensemble_size == 5
    assert args.validation_fraction == 0.1
    assert args.top_test_fraction == 0.05
    assert args.fractions == [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
