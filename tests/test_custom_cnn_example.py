import numpy as np
import torch

from sequence_optimization.examples.custom_cnn import (
    SimpleActivityCNN,
    load_custom_cnn,
    optimize_with_custom_cnn,
)


def test_custom_cnn_checkpoint_can_score_and_optimize(tmp_path):
    torch.manual_seed(3)
    checkpoint = tmp_path / "custom_cnn.pt"
    torch.save({"model_state": SimpleActivityCNN().state_dict()}, checkpoint)

    scorer = load_custom_cnn(checkpoint)
    scored = scorer.score(["CCACTCGATTAACATGTTAACAACATACTCGTCCGGCCGATCAGCGGCTA"])
    assert scored.scores.shape == (1, 1)
    assert np.isfinite(scored.scores).all()

    result = optimize_with_custom_cnn(checkpoint)
    assert result.algorithm == "genetic_algorithm"
    assert result.query_count == 44
    assert result.activities.shape == (4, 1)
