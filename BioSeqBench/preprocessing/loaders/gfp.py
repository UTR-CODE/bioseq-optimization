from pathlib import Path

import pandas as pd

from .base import DatasetLoader, pick_column, read_table


class GFPLoader(DatasetLoader):
    name = "GFP"
    alphabet = "PROTEIN"
    label_type = "log_fluorescence"
    organism = "Aequorea victoria"
    source_reference = "Sarkisyan et al. Nature 2016; doi:10.1038/nature17995"

    def load(self, raw_path: Path) -> pd.DataFrame:
        frame = read_table(raw_path)
        seq = pick_column(frame, ("sequence", "protein_sequence", "seq", "genotype"))
        label = pick_column(frame, ("median_log10_brightness", "log_fluorescence", "median_brightness", "fitness", "label", "score"))
        output = pd.DataFrame({"sequence": frame[seq], "label": frame[label]})
        for candidate in ("mutation_count", "aa_mutations", "mutations", "wt_sequence"):
            column = pick_column(frame, (candidate,), required=False)
            if column:
                output[candidate] = frame[column]
        return output
