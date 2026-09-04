from pathlib import Path

import pandas as pd

from .base import DatasetLoader, pick_column, read_table


class TFBind8Loader(DatasetLoader):
    name = "TFBind8"
    alphabet = "DNA"
    label_type = "binding_affinity"
    organism = "Homo sapiens"
    source_reference = "Barrera et al. Science 2016; doi:10.1126/science.aad2257; UniPROBE BAR15A"
    metadata_extras = {
        "reverse_complement_policy": "Split groups use min(sequence, reverse_complement(sequence)); PBM activity is exactly reverse-complement invariant.",
        "deepbind_reference": "Alipanahi et al., Nature Biotechnology 2015, doi:10.1038/nbt.3300; supplementary config regression/maxpool/M2.cfg.",
    }

    def load(self, raw_path: Path) -> pd.DataFrame:
        frame = read_table(raw_path)
        sequence = pick_column(frame, ("sequence", "8-mer", "8mer", "seq"))
        label = pick_column(frame, ("e_score", "normalized_e_score", "label", "score", "activity"))
        output = pd.DataFrame({"sequence": frame[sequence], "label": frame[label]})
        tf = pick_column(frame, ("tf_identity", "tf", "transcription_factor"), required=False)
        output["tf_identity"] = frame[tf] if tf else "SIX6_REF_R1"
        normalized = output["sequence"].astype(str).str.upper()
        complement = str.maketrans("ACGT", "TGCA")
        reverse_complement = normalized.str.translate(complement).str[::-1]
        output["split_group"] = normalized.where(normalized <= reverse_complement, reverse_complement)
        return output
