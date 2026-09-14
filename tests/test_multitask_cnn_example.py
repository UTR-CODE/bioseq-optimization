import numpy as np

from sequence_optimization import SequencePredictor
from sequence_optimization.examples.multitask_cnn import ThreeOutputCNN, example


def test_three_output_cnn_genetic_example():
    predictor = SequencePredictor(ThreeOutputCNN())
    np.testing.assert_allclose(
        predictor.score(["G" * 20, "C" * 20, "T" * 20]).scores,
        [[0.75, -0.25, -0.25], [-0.25, 0.75, -0.25], [-0.25, -0.25, 0.75]],
    )
    result = example()
    assert result.algorithm == "genetic_algorithm"
    assert len(set(result.sequences)) == 4
    assert result.query_count <= 80
    expected = np.array(
        [
            [sequence.count(base) / 20 - 0.25 for base in "GCT"]
            for sequence in result.sequences
        ]
    )
    np.testing.assert_allclose(result.activities, expected, atol=1e-7)
    np.testing.assert_allclose(
        result.scores, (expected[:, 0] - expected[:, 1]) / 2, atol=1e-7
    )
