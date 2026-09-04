import pytest
import torch

from BioSeqBench.models import build_model


@pytest.mark.parametrize("name,length,outputs", [
    ("tfbind8_deepbind", 8, 1),
    ("optimus5prime", 50, 1),
    ("deepstarr", 249, 2),
    ("igvf_legnet", 230, 1),
    ("mpralegnet", 230, 1),
    ("autosome_legnet", 150, 2),
])
def test_dna_model_shapes(name, length, outputs):
    model = build_model(name, output_dim=outputs, num_cell_types=3, max_length=length).eval()
    channels = 5 if name in {"autosome_legnet", "mpralegnet"} else 4
    batch = {
        "sequence_tensor": torch.randn(2, channels, length), "mask": torch.ones(2, length, dtype=torch.bool),
        "sequence_features": torch.randn(2, 3), "cell_type_id": torch.tensor([0, 1]),
        "structure_tensor": torch.zeros(2, length, dtype=torch.long), "loop_tensor": torch.zeros(2, length, dtype=torch.long),
    }
    output = model(**batch)
    if name == "autosome_legnet":
        assert isinstance(output, tuple) and output[0].shape == (2, 18) and output[1].shape == (2,)
    else:
        assert output.shape == (2, outputs)


@pytest.mark.parametrize("name", ["gb1_flip_cnn", "gfp_transformer"])
def test_protein_model_shapes(name):
    length = 56
    model = build_model(name, max_length=length).eval()
    tokens = torch.randint(1, 21, (2, length))
    assert model(sequence_tensor=tokens, mask=torch.ones_like(tokens, dtype=torch.bool)).shape == (2, 1)


def test_openvaccine_shape():
    length = 107
    model = build_model("openvaccine_gcn_gru").eval()
    output = model(
        sequence_tensor=torch.randn(2, 4, length), mask=torch.ones(2, length, dtype=torch.bool),
        structure_tensor=torch.zeros(2, length, dtype=torch.long), loop_tensor=torch.zeros(2, length, dtype=torch.long),
        pairing_matrix=torch.eye(length).repeat(2, 1, 1),
    )
    assert output.shape == (2, 1)
