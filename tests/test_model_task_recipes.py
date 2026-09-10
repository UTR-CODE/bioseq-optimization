"""Exercise recipe wiring without downloading model weights."""

import numpy as np
import pytest

pytest.importorskip("bioseq_models")
from sequence_optimization.examples import model_tasks as tasks


@pytest.mark.parametrize(
    "function,args,kwargs,length,directions",
    [
        ("design_utr_code", ("utr5",), {}, 50, [1]),
        ("design_utr_code", ("utr3",), {}, 80, [1]),
        ("design_utr_code", ("cds",), {}, 60, [1]),
        ("design_cds_stability", (), {}, 60, [1]),
        ("design_cds_stability", (), {"joint": True}, 60, [1, 1]),
        ("design_deepstarr", (), {}, 249, [-1, 1]),
        ("design_dream", (), {}, 93, [1]),
        ("design_cetnar", (), {}, 80, [-1]),
        ("design_cetnar", (), {"joint": True}, 80, [-1, 1]),
        ("design_agarwal", ("hepg2",), {}, 230, [-1, 1, -1]),
        ("design_tissue", ("mouse", "heart"), {}, 1001, [1, -1, -1, -1, -1, -1]),
        ("design_tissue", ("embryo", "gut"), {}, 1001, [-1, -1, -1, 1]),
        ("design_malinois", ("HepG2",), {}, 200, [-1, 1, -1]),
    ],
)
def test_task_directions_and_context(
    monkeypatch, function, args, kwargs, length, directions
):
    def load(name, **options):
        if name == "utr_code":
            region = options["design_region"]
            assert region not in options
            assert all(key in options for key in {"utr5", "utr3", "cds"} - {region})
        if name == "dream_legnet":
            assert options["variable_region_only"] is True
        count = 2 if name == "deepstarr" else 3 if name == "malinois" else 1

        class Predictor:
            task_names = tuple(f"output_{i}" for i in range(count))

            def score(self, values):
                assert all(len(value) == length for value in values)
                return np.array(
                    [
                        [value.count("G") / length + i for i in range(count)]
                        for value in values
                    ]
                )

        return Predictor()

    monkeypatch.setattr(tasks, "load_model", load)
    result = getattr(tasks, function)(*args, budget=12, **kwargs)
    np.testing.assert_allclose(
        result.scores, (result.activities * directions).min(axis=1)
    )
    assert len(result.sequences) == 4
