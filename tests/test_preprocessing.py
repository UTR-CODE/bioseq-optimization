from pathlib import Path

import pandas as pd

from BioSeqBench.preprocessing.common import clean_table, deterministic_split


def test_cleaning_and_grouped_split():
    raw = pd.DataFrame({
        "sequence": ["acgt", "ACGT", "ACGN", "GGGG"],
        "label": [1.0, 2.0, 3.0, 4.0],
        "cell_type": ["A", "A", "A", "B"],
    })
    clean, audit = clean_table(
        raw, dataset_name="X", alphabet="DNA", label_type="activity",
        organism="test", source_reference="test",
    )
    assert len(clean) == 2
    assert audit["invalid_sequence_rows"] == 1
    assert audit["duplicate_rows_removed"] == 1
    splits = deterministic_split(clean, seed=42)
    assert sum(map(len, splits.values())) == 2

