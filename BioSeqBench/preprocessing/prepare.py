from __future__ import annotations

import argparse
import json

from .loaders import LOADERS
from .pipeline import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one raw sequence-function dataset to BioSeqBench schema")
    parser.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    parser.add_argument("--raw-path", required=True)
    parser.add_argument("--output-root", default="BioSeqBench/dataset")
    parser.add_argument("--report", default="BioSeqBench/reports/dataset_statistics.html")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()
    metadata = prepare_dataset(
        args.dataset, args.raw_path, args.output_root, seed=args.seed,
        train_fraction=args.train_fraction, val_fraction=args.val_fraction, report_path=args.report,
    )
    print(json.dumps(metadata["split_sizes"], indent=2))


if __name__ == "__main__":
    main()

