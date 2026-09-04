import pandas as pd
import torch

from BioSeqBench.datasets import SequenceDataset


def test_dna_one_hot_and_mask(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({
        "sequence_id": ["x"], "sequence": ["ACGT"], "alphabet": ["DNA"],
        "length": [4], "label": [1.2], "cell_type": ["K562"],
    }).to_csv(path, index=False)
    item = SequenceDataset(path, max_length=6)[0]
    assert item["sequence_tensor"].shape == (4, 6)
    assert item["mask"].tolist() == [True, True, True, True, False, False]
    assert torch.allclose(item["sequence_tensor"].sum(0), item["mask"].float())


def test_protein_tokens(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({
        "sequence_id": ["p"], "sequence": ["ACD"], "alphabet": ["PROTEIN"],
        "length": [3], "label": [0.1], "cell_type": [""],
    }).to_csv(path, index=False)
    item = SequenceDataset(path, max_length=5)[0]
    assert item["sequence_tensor"].dtype == torch.long
    assert item["sequence_tensor"].tolist()[-2:] == [0, 0]

