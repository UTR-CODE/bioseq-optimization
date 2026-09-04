"""Load every bundled checkpoint and run one real sequence through it."""

from __future__ import annotations

import json

import numpy as np

from sequence_optimization import available_models, load_model, model_info


EXAMPLE_SEQUENCES = {
    "optimus5prime": ("CCACTCGATTAACATGTTAACAACATACTCGTCCGGCCGATCAGCGGCTA"),
    "deepstarr": (
        "AACATACCCTGCTCTAGCGTATTGCTTTTTAATTGGCAGAAAAATATGTGCTCCGGTCGGTCGGTAT"
        "TAATATATGTATAAATCCAACCAATATGACACACATTTCATTCAATTTATGACGCACAGCTGACACA"
        "CTTTTCCCACACTATCTCACCCACATGCATATACACACATACACAGGTTTTTCTGTGCGAAAAATGAA"
        "TTACGATTCGTCGCGTTTTTTTTTTCTGGTCGCTTGTTTTCGTTTGG"
    ),
    "dream_legnet": (
        "TGCATTTTTTTCACATCTGTAATGTCTACAGTGGATTCTGTTATGGACCGGTGCAGGACTATGCTTAG"
        "GTGTGTGAGGTTGTTGATGGACGGGGGAGGGTTACGGCTGTT"
    ),
    "zhao_ptrc": (
        "TTGACAATTAATCATCCGGCTCGTATAATGTGCGGAATTGTAAGCGGATAACAATTTCACACAGGAAACAGACC"
    ),
    "zhou_l3": "TTGATAATTAATCATCCGGCTCGTATAAT",
}


def main() -> None:
    report = {}
    for name in available_models():
        info = model_info(name)
        sequence = EXAMPLE_SEQUENCES[name]
        model = load_model(name, device="cpu", batch_size=2)
        scores = model.score([sequence]).scores
        if len(sequence) != info.sequence_length:
            raise RuntimeError(f"{name} example has the wrong sequence length")
        if scores.shape not in {(1,), (1, len(info.task_names))}:
            raise RuntimeError(f"{name} returned an unexpected shape: {scores.shape}")
        if not np.isfinite(scores).all():
            raise RuntimeError(f"{name} returned a non-finite activity")
        report[name] = {
            "sequence_length": info.sequence_length,
            "task_names": info.task_names,
            "activities": scores.reshape(-1).tolist(),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
