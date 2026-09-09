"""Python example: three direction lists for a two-output scorer."""

from bioseq_models import load_model
from sequence_optimization import SequenceSpace, basesearch_optimize


def main():
    model = load_model("deepstarr", device="cpu")
    objectives = {
        "both_high": ["maximize", "maximize"],
        "developmental_high_housekeeping_low": ["maximize", "minimize"],
        "housekeeping_high_developmental_low": ["minimize", "maximize"],
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
