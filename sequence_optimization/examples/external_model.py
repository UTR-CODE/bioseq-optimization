"""Optimize an Optimus 5-Prime model supplied by bioseq-models."""

from bioseq_models import load_model

from sequence_optimization import (
    genetic_algorithm_optimize,
)


def main() -> None:
    model = load_model("optimus5prime", device="cpu")
    result = genetic_algorithm_optimize(
        model,
        length=50,
        budget=40,
        num_candidates=4,
        mutation_rate=0.1,
        initial=[
            "CCACTCGATTAACATGTTAACAACATACTCGTCCGGCCGATCAGCGGCTA",
            "GGTACCTCTAGTTTAATCCCTCCCTTATACGGTGCAACCCGCTCCCCTCA",
            "TGCATTTTTTTCACATCTATTTGCGTTTCACATCTTTTCTTACCTCTTGA",
            "CTCATTTTTCACCTCCCTTTTATCTCTCTGCTTACGTTTCGATCCCATCA",
        ],
        seed=0,
    )
    print(result.sequences)
    print(result.activities)


if __name__ == "__main__":
    main()
