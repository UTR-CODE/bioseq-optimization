import torch

from BioSeqBench.models.weakseqcnn import (
    WeakSeqCNN,
    encode_nucleotide_sequences,
    normalize_nucleotide_sequence,
)
from BioSeqBench.models.utr5_mpra import DifferentiableUTR5Ensemble, UTR5WeakEnsemble


def test_fixed_length_dna_shape_and_parameter_budget():
    model = WeakSeqCNN().eval()
    inputs = torch.nn.functional.one_hot(
        torch.randint(0, 4, (8, 200)), num_classes=4
    ).float()
    output = model(inputs)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert output.shape == (8,)
    assert 150_000 <= parameters <= 300_000


def test_rna_u_to_t_uses_identical_encoding():
    assert normalize_nucleotide_sequence("augcuu") == "ATGCTT"
    rna, rna_mask = encode_nucleotide_sequences(["AUGCUU"])
    dna, dna_mask = encode_nucleotide_sequences(["ATGCTT"])
    assert torch.equal(rna, dna)
    assert torch.equal(rna_mask, dna_mask)


def test_variable_length_padding_mask_preserves_prediction():
    torch.manual_seed(7)
    model = WeakSeqCNN().eval()
    unpadded, unpadded_mask = encode_nucleotide_sequences(["ACGTACGTAC"])
    padded, padded_mask = encode_nucleotide_sequences(["ACGTACGTAC"], pad_to_length=25)
    with torch.no_grad():
        first = model(unpadded, unpadded_mask)
        second = model(padded, padded_mask)
    assert torch.allclose(first, second, atol=1e-5, rtol=1e-5)


def test_soft_one_hot_propagates_finite_gradients():
    torch.manual_seed(11)
    model = WeakSeqCNN()
    logits = torch.randn(3, 40, 4, requires_grad=True)
    model(torch.softmax(logits, dim=-1)).mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_checkpoint_round_trip_is_exact(tmp_path):
    torch.manual_seed(13)
    model = WeakSeqCNN().eval()
    inputs = torch.softmax(torch.randn(2, 30, 4), dim=-1)
    with torch.no_grad():
        expected = model(inputs)
    path = tmp_path / "weakseqcnn.pt"
    torch.save(model.state_dict(), path)
    restored = WeakSeqCNN().eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        observed = restored(inputs)
    assert torch.equal(expected, observed)


def test_utr5_weak_ensemble_loads_standardized_checkpoint(tmp_path):
    torch.manual_seed(17)
    model = WeakSeqCNN().eval()
    torch.save(
        {
            "model_state": model.state_dict(),
            "model": "weakseqcnn_v1",
            "model_config": model.model_config,
            "dataset": "utr5_mpra",
            "sequence_length": 50,
            "train_label_mean": 2.0,
            "train_label_std": 0.5,
        },
        tmp_path / "member_00.pt",
    )
    ensemble = UTR5WeakEnsemble(tmp_path, device="cpu")
    mean, std = ensemble.predict(["ACGT" * 12 + "AC", "TGCA" * 12 + "TG"])
    assert mean.shape == std.shape == (2,)
    differentiable = DifferentiableUTR5Ensemble(ensemble)
    logits = torch.randn(2, 50, 4, requires_grad=True)
    prediction, _ = differentiable(torch.softmax(logits, dim=-1))
    prediction.mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_utr5_weak_ensemble_can_select_one_member(tmp_path):
    model = WeakSeqCNN().eval()
    for index in range(2):
        torch.save(
            {
                "model_state": model.state_dict(), "model": "weakseqcnn_v1",
                "model_config": model.model_config, "dataset": "utr5_mpra",
                "sequence_length": 50, "train_label_mean": float(index),
                "train_label_std": 1.0,
            },
            tmp_path / f"member_{index:02d}.pt",
        )
    ensemble = UTR5WeakEnsemble(tmp_path, device="cpu", member_indices=[1])
    assert len(ensemble.models) == 1
    assert ensemble.checkpoint_paths == [tmp_path / "member_01.pt"]
