import json
from pathlib import Path

import pandas as pd

from .base import DatasetLoader, json_array_mean, pick_column, read_table


class OpenVaccineLoader(DatasetLoader):
    name = "OpenVaccine"
    alphabet = "RNA"
    label_type = "degradation_score"
    organism = "Synthetic RNA"
    source_reference = "Wayment-Steele et al. Nat Mach Intell 2022; doi:10.1038/s42256-022-00571-8"

    def load(self, raw_path: Path) -> pd.DataFrame:
        frame = read_table(raw_path)
        seq = pick_column(frame, ("sequence", "seq"))
        target_candidates = (
            "degscore", "est_k_deg", "deg_mg_ph10", "deg_ph10", "deg_mg_50c", "deg_50c", "reactivity", "label",
        )
        target = pick_column(frame, target_candidates)
        output = pd.DataFrame({
            "sequence": frame[seq],
            "label": frame[target].map(json_array_mean) if frame[target].dtype == object else frame[target],
        })
        for candidate in ("structure", "predicted_loop_type", "reactivity", "deg_mg_ph10", "deg_ph10", "deg_mg_50c", "deg_50c", "half_life", "signal_to_noise", "SN_filter"):
            column = pick_column(frame, (candidate,), required=False)
            if column:
                output[candidate] = frame[column].map(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
        return output
