"""Run a small real-checkpoint compatibility matrix, not an efficacy benchmark.

Run with bioseq-models installed or on PYTHONPATH. Model-specific tensor
transforms below are test fixtures, not public optimizer adapters.
"""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from bioseq_models import available_models, load_model, model_info
from sequence_optimization import (
    DifferentiableTorchSequenceScorer,
    OptimizationConfig,
    SequenceSpace,
    available_algorithms,
    run,
)

GRADIENT = {"vanilla_activation_maximization", "fast_seqprop", "den", "ledidi"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", help="Optional subset for targeted reruns"
    )
    parser.add_argument("--algorithms", nargs="+", choices=available_algorithms())
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = {
        "device": args.device,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "seed": 7,
        "num_candidates": 4,
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.models or available_models():
        info = model_info(name)
        if len(info.task_names) != 1:
            continue
        length = (
            info.sequence_length
            or {
                "cetnar_stability": 80,
                "parm_k562": 50,
                "utr_code": 50,
                "cds_stability": 60,
            }[name]
        )
        options = {"device": args.device, "batch_size": 8}
        if name == "utr_code":
            options.update(
                design_region="utr5", cds="ATG" + "GCT" * 18 + "TAA", utr3="ACGT" * 10
            )
        model = None
        soft = None
        try:
            model = load_model(name, **options)
            load_error = None
        except Exception as exc:
            load_error = f"{type(exc).__name__}: {exc}"
        space = SequenceSpace(length)
        initial = space.random(4, np.random.default_rng(7))
        soft_error = "requires a model-specific differentiable adapter"
        needs_gradients = bool(set(args.algorithms or available_algorithms()) & GRADIENT)
        if needs_gradients and model is not None and name not in {
            "utr_code",
            "cds_stability",
            "dream_legnet",
            "parm_k562",
        }:

            def transform(values):
                if name == "cetnar_stability":
                    return values
                values = values.transpose(1, 2)
                return (
                    values[:, [0, 2, 1, 3], :]
                    if name.startswith("agarwal_")
                    else values
                )

            try:
                soft = DifferentiableTorchSequenceScorer(
                    model.model,
                    model.encoder,
                    device=args.device,
                    task_names=model.task_names,
                    soft_input_transform=transform,
                    pad_to_length=getattr(model, "pad_to_length", None),
                    padding_side=getattr(model, "padding_side", "right"),
                    batch_size=8,
                )
                encoded = (
                    torch.nn.functional.one_hot(
                        torch.as_tensor(space.encode(initial), dtype=torch.long), 4
                    )
                    .float()
                    .to(args.device)
                    .requires_grad_()
                )
                activities, _ = soft.score_soft(encoded)
                np.testing.assert_allclose(
                    activities.detach().cpu().numpy(),
                    model.score(initial),
                    rtol=1e-4,
                    atol=1e-5,
                )
                activities.sum().backward()
                assert encoded.grad is not None and torch.isfinite(encoded.grad).all()
                del encoded, activities
            except Exception as exc:
                soft = None
                soft_error = f"{type(exc).__name__}: {exc}"
        for algorithm in args.algorithms or available_algorithms():
            start = time.monotonic()
            row = {"model": name, "algorithm": algorithm, "length": length}
            budget = 4 + length * 3 if algorithm == "ism_greedy" else 12
            row["budget"] = budget
            try:
                if load_error:
                    raise RuntimeError(load_error)
                if algorithm in GRADIENT and soft is None:
                    # Verify the actual direct-call rejection as well.
                    try:
                        run(
                            algorithm,
                            model,
                            space,
                            OptimizationConfig(12, num_candidates=4),
                            initial=initial,
                        )
                    except TypeError as exc:
                        row.update(
                            status="unsupported",
                            detail=f"{soft_error}; direct call: {exc}",
                        )
                    else:
                        raise AssertionError(
                            "unexpected gradient support; update fixture"
                        )
                else:
                    config = OptimizationConfig(
                        budget,
                        num_candidates=4,
                        mutation_rate=0.1,
                        elite_fraction=0.5,
                        generator_pretrain_epochs=1,
                        generator_update_epochs=1,
                        cbas_batch_size=4,
                        den_seed_pretrain_epochs=1,
                    )
                    result = run(
                        algorithm,
                        soft if algorithm in GRADIENT else model,
                        space,
                        config,
                        initial=initial,
                        seed=7,
                    )
                    assert len(set(result.sequences)) == 4
                    assert all(
                        len(s) == length and set(s) <= set("ACGT")
                        for s in result.sequences
                    )
                    assert result.activities.shape == (4, 1)
                    assert result.task_names == info.task_names
                    assert 4 < result.query_count <= budget
                    np.testing.assert_allclose(result.scores, result.activities[:, 0])
                    np.testing.assert_allclose(
                        result.activities,
                        model.score(result.sequences),
                        rtol=1e-4,
                        # Some GPU checkpoints (notably Agarwal K562) have
                        # small batch/order-dependent float32 drift on a
                        # second forward pass.
                        atol=1e-4,
                    )
                    row.update(
                        status="passed",
                        queries=result.query_count,
                        best_score=float(result.scores[0]),
                        interface="wrapper" if algorithm in GRADIENT else "direct",
                    )
            except Exception as exc:
                row.update(status="failed", detail=f"{type(exc).__name__}: {exc}")
            row["seconds"] = round(time.monotonic() - start, 3)
            report["results"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(name, algorithm, row["status"], row.get("detail", ""), flush=True)
        del model, soft
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
