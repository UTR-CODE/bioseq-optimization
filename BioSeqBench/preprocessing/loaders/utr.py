from pathlib import Path

import pandas as pd

from .base import DatasetLoader, pick_column, read_table


class UTRLoader(DatasetLoader):
    name = "UTR5_MPRA"
    alphabet = "DNA"
    label_type = "mean_ribosome_load"
    organism = "Homo sapiens (HEK293T)"
    source_reference = "Sample et al. Nat Biotechnol 2019; doi:10.1038/s41587-019-0164-5; GEO:GSE114002"

    def load(self, raw_path: Path) -> pd.DataFrame:
        frame = read_table(raw_path)
        # This is the exact selection implemented in the authors' public
        # ``training_MRL_CNN.ipynb``.  It deliberately favours high-coverage
        # measurements: rank all unmodified eGFP 50-mers by total reads, keep
        # the first 280,000, and reserve the first 20,000 for the test set.
        # The deposited CSV is already in decreasing read-count order.  Use a
        # stable sort so that ties at each cut-off retain the GEO record order
        # (and therefore reproduce the authors' supplied 280k table across
        # pandas versions).  The rank is retained because the two cut-offs
        # contain tied read counts.
        if {"utr", "total_reads", "rl"}.issubset(frame.columns):
            ranked = frame.sort_values("total_reads", ascending=False, kind="mergesort").iloc[:280_000].copy()
            ranked["source_rank_by_total_reads"] = range(1, len(ranked) + 1)
            ranked["source_split"] = "train"
            ranked.iloc[:20_000, ranked.columns.get_loc("source_split")] = "test"
            return pd.DataFrame({
                "sequence": ranked["utr"],
                "label": ranked["rl"],
                "total_reads": ranked["total_reads"],
                "source_rank_by_total_reads": ranked["source_rank_by_total_reads"],
                "source_split": ranked["source_split"],
            })

        # Compatibility fallback for the older processed mirror.  It does not
        # contain total_reads, so it cannot reproduce the publication split.
        seq = pick_column(frame, ("sequence", "utr", "utr_sequence", "seq"))
        label = pick_column(frame, ("mean_ribosome_load", "mrl", "label", "rl", "score"))
        output = pd.DataFrame({"sequence": frame[seq], "label": frame[label]})
        chemistry = pick_column(frame, ("chemistry", "sample", "condition"), required=False)
        if chemistry:
            output["chemistry"] = frame[chemistry]
        return output
