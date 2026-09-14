import numpy as np

from sequence_optimization.examples.custom_cnn_tasks import (
    single_task,
    multi_task,
    independent_models,
)


def test_three_custom_cnn_recipes():
    for function, signs in [
        (single_task, [1]),
        (multi_task, [1, -1, 0]),
        (independent_models, [1, -1, -1]),
    ]:
        result = function()
        assert result.algorithm == "genetic_algorithm"
        assert result.activities.shape == (4, len(signs))
        active = np.asarray(signs) != 0
        np.testing.assert_allclose(
            result.scores,
            (result.activities[:, active] * np.asarray(signs)[active]).mean(axis=1),
        )
