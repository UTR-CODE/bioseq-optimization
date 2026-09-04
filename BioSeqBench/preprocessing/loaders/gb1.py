from pathlib import Path

import pandas as pd

from .base import DatasetLoader, pick_column, read_table


GB1_WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
MUTATION_POSITIONS = (39, 40, 41, 54)


class GB1Loader(DatasetLoader):
    name = "GB1"
    alphabet = "PROTEIN"
    label_type = "binding_fitness"
    organism = "Streptococcus sp. group G"
    source_reference = "Wu et al. eLife 2016; doi:10.7554/eLife.16965"

    def load(self, raw_path: Path) -> pd.DataFrame:
        frame = read_table(raw_path)
        sequence_col = pick_column(frame, ("sequence", "seq", "full_sequence", "protein_sequence"))
        label_col = pick_column(frame, ("fitness", "label", "score", "target"))
        sequences = frame[sequence_col].astype(str).str.upper()
        # FLIP distributes the crystallographic complex chain; the mutagenized
        # GB1 domain is the N-terminal 56-aa region used by Wu et al.
        sequences = sequences.map(lambda seq: seq[: len(GB1_WT)] if len(seq) > len(GB1_WT) else seq)
        # FLIP contains the full 56-aa domain; original tables may contain only four residues.
        def expand(seq: str) -> str:
            if len(seq) != 4:
                return seq
            chars = list(GB1_WT)
            for position, residue in zip(MUTATION_POSITIONS, seq):
                chars[position - 1] = residue
            return "".join(chars)
        sequences = sequences.map(expand)
        output = pd.DataFrame({"sequence": sequences, "label": frame[label_col]})
        output["wt_sequence"] = GB1_WT
        output["mutation_positions"] = output["sequence"].map(
            lambda seq: ";".join(str(i + 1) for i, (a, b) in enumerate(zip(GB1_WT, seq)) if a != b)
        )
        output["variant_4mer"] = output["sequence"].map(lambda seq: "".join(seq[i - 1] for i in MUTATION_POSITIONS))
        return output
