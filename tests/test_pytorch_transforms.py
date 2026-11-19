from tests.helpers import run_kernel

import numpy as np
import pytest
import torch
from torch import nn
from sequence_optimization import SequencePredictor, onehot, random_seed_sequences, MultiTaskObjective
from sequence_optimization import available_algorithms

DESIGN_ALGORITHMS = [name for name in available_algorithms() if name != "random"]

class ActivityModel(nn.Module):
    def __init__(self, outputs=1):
        super().__init__()
        self.conv = nn.Conv1d(4, 8, kernel_size=3, padding=1)
        self.head = nn.Linear(8, outputs)
    def forward(self, x):
        return self.head(torch.relu(self.conv(x)).mean(dim=-1))


def compare_paths(scorer, sequences):
    hard = scorer.score(sequences).scores
    p = torch.tensor(onehot(sequences), requires_grad=True)
    soft, _ = scorer.score_soft(p)
    np.testing.assert_allclose(hard, soft.detach().cpu(), atol=1e-6)
    gradient, = torch.autograd.grad(soft.sum(), p)
    assert gradient.shape == p.shape and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    return hard, gradient


def test_arbitrary_pytorch_model():
    torch.manual_seed(2)
    model = ActivityModel()
    scorer = SequencePredictor(model, tasks=['activity'])
    hard, _ = compare_paths(scorer, ['ACGTACGT', 'GGGGCCCC'])
    assert hard.shape == (2, 1)
    assert not model.training and all(not p.requires_grad for p in model.parameters())


def test_channels_last_transform():
    class ChannelsLast(nn.Module):
        def forward(self, x):
            assert x.shape[1:] == (8, 4)
            return x[:, :, 0].sum(1)
    scorer = SequencePredictor(ChannelsLast(), input_transform=lambda p: p)
    values, _ = compare_paths(scorer, ['ACGTACGT', 'GGGGCCCC'])
    np.testing.assert_array_equal(values[:, 0], [2, 0])
    assert scorer.task_names == ('task_0',)


def test_context_transform_preserves_gradients():
    left, right = torch.tensor(onehot(['AC'])), torch.tensor(onehot(['TG']))
    def context(p):
        return torch.cat([left.to(p).expand(len(p), -1, -1), p,
                          right.to(p).expand(len(p), -1, -1)], dim=1).transpose(1, 2)
    class ContextModel(nn.Module):
        def forward(self, x):
            assert x.shape[1:] == (4, 12)
            return (x[:, 0] * torch.arange(1, 13)).sum(1)
    compare_paths(SequencePredictor(ContextModel(), input_transform=context), ['ACGTACGT', 'GGGGCCCC'])


def test_dictionary_output_and_task_selection():
    class DictionaryModel(nn.Module):
        def forward(self, x):
            return {'activity': x[:, 2].sum(1), 'features': x.mean(2)}
    scorer = SequencePredictor(DictionaryModel(), output_transform=lambda y: y['activity'])
    hard, _ = compare_paths(scorer, ['ACGTACGT', 'GGGGCCCC'])
    np.testing.assert_array_equal(hard[:, 0], [2, 4])


def test_multicell_contrasts_match_fitness_and_gradient():
    class MultiOutput(nn.Module):
        def forward(self, x):
            return x[:, :3].sum(2)
    def contrasts(y):
        return torch.cat([y, y[:, :1]-y[:, 1:]], dim=1)
    scorer = SequencePredictor(MultiOutput(), tasks=['target','off1','off2','margin1','margin2'],
                               output_transform=contrasts)
    p = torch.tensor(onehot(['AAAACCGG','ACCCCCGG']), requires_grad=True)
    hard = scorer.score(['AAAACCGG','ACCCCCGG']).scores
    soft, _ = scorer.score_soft(p)
    objective = MultiTaskObjective(high=[3,4], aggregation='worst')
    expected = hard[:,0]-hard[:,1:3].max(1)
    np.testing.assert_allclose(objective.scalarize(hard),expected)
    fitness = objective.scalarize_soft(soft)
    np.testing.assert_allclose(fitness.detach(),expected)
    gradient, = torch.autograd.grad(fitness.sum(),p)
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0


@pytest.mark.parametrize('algorithm', DESIGN_ALGORITHMS)
def test_all_algorithms_accept_pytorch_predictor(algorithm):
    torch.manual_seed(3)
    scorer = SequencePredictor(ActivityModel(), tasks=['activity'])
    initial = random_seed_sequences(8,4,seed=3)
    result = run_kernel(scorer,8,initial,['maximize'],4,algorithm=algorithm,
        budget=40,diversity=0,seed=3,generator_pretrain_epochs=1,generator_update_epochs=1,
        den_seed_pretrain_epochs=1)
    assert len(result.sequences)==4
    np.testing.assert_allclose(result.activities,scorer.score(result.sequences).scores,atol=1e-5)


def test_input_transform_uses_the_model_device():
    class EmbeddedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(4, 8)
            self.head = nn.Linear(8, 1)
        def forward(self, embedded):
            return self.head(embedded.mean(1))
    model = EmbeddedModel()
    scorer = SequencePredictor(model, device='cuda', tasks=['activity'],
        input_transform=lambda p: p @ model.embedding.weight)
    compare_paths(scorer, ['ACGTACGT', 'GGGGCCCC'])
