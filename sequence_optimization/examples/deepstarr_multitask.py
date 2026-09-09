"""Run three objectives through the real two-output DeepSTARR model.

Run: python -m sequence_optimization.examples.deepstarr_multitask
"""

from bioseq_models import load_model
from sequence_optimization import MultiTaskObjective, SequenceSpace, basesearch_optimize


def main():
    model = load_model("deepstarr", device="cpu")
    objectives = {
        "both_high": MultiTaskObjective(high="all"),
        "developmental_high_housekeeping_low": MultiTaskObjective(
            high=["developmental"], low=["housekeeping"]
        ),
        "housekeeping_high_developmental_low": MultiTaskObjective(
            high=["housekeeping"], low=["developmental"]
        ),
    }
    for name, objective in objectives.items():
        result = basesearch_optimize(
            model,
            SequenceSpace(249),
            objective=objective,
            num_candidates=4,
            budget=100,
            mutation_rate=0.1,
            seed=7,
        )
        print(name)
        print(result.to_frame().to_string(index=False))


if __name__ == "__main__":
    main()
