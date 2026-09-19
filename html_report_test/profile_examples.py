"""Run unchanged example functions with per-algorithm timing and live telemetry."""

import importlib.util
import json
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import psutil
import pandas as pd
import torch
import sequence_optimization.suite as suite


def main():
    out = Path("results") / time.strftime("examples_profile_%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    print(f"OUTPUT={out.resolve()}", flush=True)
    spec = importlib.util.spec_from_file_location(
        "examples", "example/run_readme_examples.py"
    )
    examples = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(examples)
    process = psutil.Process()
    stop = threading.Event()
    state = {"example": "", "algorithm": ""}

    def monitor():
        with (out / "telemetry.jsonl").open("w") as stream:
            while not stop.is_set():
                gpu = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                row = dict(
                    time=time.time(),
                    **state,
                    cpu_percent=process.cpu_percent(),
                    rss_bytes=process.memory_info().rss,
                    gpu=gpu.stdout.strip(),
                )
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                stop.wait(5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    original = suite.optimize
    timings = []
    statistics = []

    def save_tables():
        pd.DataFrame(timings).to_csv(out / "algorithm_times.csv", index=False)
        pd.DataFrame(statistics).to_csv(out / "prediction_statistics.csv", index=False)

    def timed(*args, **kwargs):
        state["algorithm"] = kwargs["algorithm"]
        torch.cuda.synchronize()
        start = time.monotonic()
        print("START", state, flush=True)
        try:
            result = original(*args, **kwargs)
        except Exception:
            error = traceback.format_exc()
            (
                out / (state["example"] + "_" + state["algorithm"] + "_error.txt")
            ).write_text(error)
            timings.append(
                dict(**state, elapsed_seconds=time.monotonic() - start, status="error")
            )
            save_tables()
            print(error, flush=True)
            # The suite accepts a list of results. Record this failure explicitly
            # and allow the remaining algorithms to run; never invent candidates.
            return []
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        selected = result.selected if hasattr(result, "selected") else result
        frame = selected.to_frame()
        assert len(frame) == 100 and len(set(selected.sequences)) == 100
        prefix = state["example"] + "_" + state["algorithm"]
        frame.to_csv(out / (prefix + "_candidates.csv"), index=False)
        (out / (prefix + "_metadata.json")).write_text(
            json.dumps(selected.metadata, indent=2)
        )
        row = dict(
            **state,
            elapsed_seconds=elapsed,
            status="ok",
            stop_reason=getattr(
                result, "stop_reason", selected.metadata.get("stop_reason")
            ),
            query_count=getattr(result, "total_query_count", selected.query_count),
            selected_query_count=selected.query_count,
        )
        timings.append(row)
        outputs = {"objective_score": selected.scores}
        for index, task in enumerate(selected.task_names or []):
            outputs[task] = selected.activities[:, index]
        for task, values in outputs.items():
            s = pd.Series(values)
            statistics.append(
                dict(
                    **state,
                    output=task,
                    n=len(s),
                    mean=s.mean(),
                    minimum=s.min(),
                    maximum=s.max(),
                    p05=s.quantile(0.05),
                    p25=s.quantile(0.25),
                    p50=s.quantile(0.5),
                    p75=s.quantile(0.75),
                    p95=s.quantile(0.95),
                    elapsed_seconds=elapsed,
                )
            )
        save_tables()
        with (out / "algorithms.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print("DONE", row, flush=True)
        return result

    suite.optimize = timed
    try:
        requested = sys.argv[1:] or list(examples.EXAMPLES)
        unknown = sorted(set(requested) - set(examples.EXAMPLES))
        if unknown:
            raise ValueError(f"unknown examples: {', '.join(unknown)}")
        for name in requested:
            function = examples.EXAMPLES[name]
            state.update(example=name, algorithm="loading")
            start = time.monotonic()
            try:
                frame = function()
                frame.to_csv(out / (name + "_candidates.csv"), index=False)
                print("EXAMPLE_DONE", name, time.monotonic() - start, flush=True)
            except Exception:
                error = traceback.format_exc()
                (out / (name + "_error.txt")).write_text(error)
                print(error, flush=True)
    finally:
        stop.set()
        thread.join()
        suite.optimize = original
    print(
        "OUTPUT",
        out.resolve(),
        "SUCCESS",
        sum(r["status"] == "ok" for r in timings),
        "FAILURES",
        sum(r["status"] == "error" for r in timings),
        flush=True,
    )


if __name__ == "__main__":
    main()
