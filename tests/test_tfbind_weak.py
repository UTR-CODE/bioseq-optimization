import numpy as np
import pytest
import torch

from BioSeqBench.models.tfbind8 import TFBindCNN
from BioSeqBench.oracles.tfbind import (
    TFBindLandscape,
    codes_to_encoded,
    decode_sequences,
    encode_sequences,
    encoded_to_codes,
)
from BioSeqBench.training.train_tfbind_weak_cnns import (
    split_visible_pool,
    validate_fractions,
)


def test_base4_sequence_round_trip():
    sequences = ["AAAAAAAA", "ACGTACGT", "TTTTTTTT"]
    encoded = encode_sequences(sequences, 8)
    codes = encoded_to_codes(encoded)
    assert decode_sequences(codes_to_encoded(codes, 8)).tolist() == sequences
    assert codes.tolist() == [0, 6939, 4**8 - 1]


def test_exact_landscape_ranking_and_scoring():
    labels = np.arange(4**8, dtype=float)[::-1]
    landscape = TFBindLandscape("tfbind8", 8, labels)
    assert landscape.score(["AAAAAAAA", "TTTTTTTT"]).tolist() == [4**8 - 1, 0]
    assert landscape.ranked_codes()[0] == 4**8 - 1


def test_fraction_validation_and_visible_split():
    assert validate_fractions([0.95, 0.01, 0.05], 0.05) == (0.01, 0.05, 0.95)
    with pytest.raises(ValueError, match="cannot overlap"):
        validate_fractions([0.96], 0.05)
    train, validation = split_visible_pool(100, 0.1, seed=7)
    assert len(train) == 90
    assert len(validation) == 10
    assert not set(train) & set(validation)


@pytest.mark.parametrize("length", [8, 10])
def test_weak_cnn_supports_both_landscapes(length):
    model = TFBindCNN().eval()
    assert model(torch.randn(3, 4, length)).shape == (3, 1)
