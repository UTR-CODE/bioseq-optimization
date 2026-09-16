#!/usr/bin/env python3
"""Quick test script to verify the optimization suite framework works.

This runs a minimal test with just 2 algorithms, small population, and low budget
to quickly verify that all components are working correctly.
"""

import sys
sys.path.insert(0, "/home/hexin/bioseq-models")

from bioseq_models import load_model
from sequence_optimization import (
    DIFFERENTIABLE_ALGORITHMS,
    SequencePredictor,
    optimize_algorithms,
    random_seed_sequences,
)


def test_optimus():
    """Test Optimus 5-Prime with minimal parameters."""
    print("Testing Optimus 5-Prime...")

    model = load_model("optimus5prime", device="cpu", batch_size=256)

    # Test with only 2 algorithms: one black-box, one gradient
    algorithms = ["genetic_algorithm", "vanilla_activation_maximization"]

    differentiable_model = SequencePredictor(
        model.model,
        device="cpu",
        task_names=model.task_names,
        batch_size=256,
    )

    initial = random_seed_sequences(length=50, count=10, seed=42)

    suite = optimize_algorithms(
        model,
        algorithms=algorithms,
        differentiable_model=differentiable_model,
        length=50,
        initial=initial,
        objective=["maximize"],
        num_candidates=5,
        budget=100,
        seed=42,
    )

    print("\nSummary:")
    print(suite.summary_frame().to_string(index=False))

    print("\nTop candidates:")
    df = suite.to_frame()
    print(df.nlargest(5, 'score')[['algorithm', 'sequence', 'score']].to_string(index=False))

    print(f"\n✓ Test passed! Generated {len(df)} candidates from {len(algorithms)} algorithms.")
    return True


if __name__ == "__main__":
    try:
        test_optimus()
        print("\n✓ All tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
