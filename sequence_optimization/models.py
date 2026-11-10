"""Assay-boundary wrappers around models loaded by bioseq_models.load_model.

No network architecture or checkpoint is copied into this repository.
Soft input convention: [batch,length,4], A/C/G/T channel order.
"""
import numpy as np
import torch
from .api import as_score_batch
from .onehot import onehot

CELLS = ('K562', 'HepG2', 'SKNSH')
PROMOTER_LEFT = 'TGCATTTTTTTCACATC'
PROMOTER_RIGHT = 'GGTTACGGCTGTT'


class MalinoisScorer:
    """Three original activities and two target-minus-offtarget margins."""
    supports_soft_scoring = True

    def __init__(self, raw, target_cell='K562'):
        from bioseq_models.adapters.malinois import LEFT_CONTEXT, RIGHT_CONTEXT
        if target_cell not in CELLS:
            raise ValueError(f'target_cell must be one of {CELLS}')
        self.raw = raw
        self.target = CELLS.index(target_cell)
        self.other = [i for i in range(3) if i != self.target]
        self.task_names = (*CELLS, *(f'{target_cell}_minus_{CELLS[i]}' for i in self.other))
        self.device, self.batch_size = raw.device, raw.batch_size
        self.left = torch.tensor(onehot([LEFT_CONTEXT], channels_first=False), device=self.device)
        self.right = torch.tensor(onehot([RIGHT_CONTEXT], channels_first=False), device=self.device)

    def score(self, sequences):
        values = list(sequences)
        if any(len(s) != 200 for s in values):
            raise ValueError('Malinois enhancer candidates must be 200 nt')
        x = as_score_batch(self.raw.score(values)).scores
        return np.column_stack((x, *(x[:, self.target]-x[:, i] for i in self.other)))

    def score_soft(self, probabilities):
        if probabilities.ndim != 3 or probabilities.shape[1:] != (200, 4):
            raise ValueError('Expected [batch,200,4] A/C/G/T probabilities')
        # Keep the official fixed MPRA contexts in the differentiable path too.
        p = probabilities.to(self.device)
        x = torch.cat((self.left.to(p).expand(len(p), -1, -1), p,
                       self.right.to(p).expand(len(p), -1, -1)), 1)
        y = self.raw.model(x.transpose(1, 2))
        return torch.cat((y, *(y[:, self.target:self.target+1]-y[:, i:i+1]
                             for i in self.other)), 1), None


class PromoterScorer:
    """Optimize only the 80-nt DREAM yeast insert; keep assay context fixed."""
    supports_soft_scoring = True
    task_names = ('promoter_activity',)

    def __init__(self, raw):
        from bioseq_models.adapters.dream_legnet import CONTEXT_PREFIX
        if not raw.variable_region_only:
            raise ValueError('Load dream_legnet with variable_region_only=True')
        self.raw = raw
        self.device, self.batch_size = raw.device, raw.batch_size
        self.left = torch.tensor(onehot([CONTEXT_PREFIX], channels_first=False), device=self.device)
        self.right = torch.tensor(onehot([PROMOTER_RIGHT], channels_first=False), device=self.device)

    def score(self, sequences):
        values = list(sequences)
        if any(len(s) != 80 for s in values):
            raise ValueError('Promoter candidates must be 80-nt inserts')
        return as_score_batch(self.raw.score([s+PROMOTER_RIGHT for s in values])).scores

    def score_soft(self, probabilities):
        if probabilities.ndim != 3 or probabilities.shape[1:] != (80, 4):
            raise ValueError('Expected [batch,80,4] A/C/G/T probabilities')
        p = probabilities.to(self.device)
        dna = torch.cat((self.left.to(p).expand(len(p), -1, -1), p,
                         self.right.to(p).expand(len(p), -1, -1)), 1)
        agct = dna[:, :, [0, 2, 1, 3]].transpose(1, 2)
        encoded = torch.cat((agct, p.new_ones((len(p), 1, 150))), 1)
        # raw.model already performs forward / reverse-complement averaging.
        return self.raw.model(encoded).reshape(-1, 1), None


def validate_scorer(scorer, sequences, *, atol=1e-4, rtol=1e-4):
    """Check exact one-hot predictions and nonzero finite input gradients."""
    hard = as_score_batch(scorer.score(sequences)).scores
    p = torch.tensor(onehot(sequences, channels_first=False),
                     device=scorer.device, requires_grad=True)
    soft, _ = scorer.score_soft(p)
    np.testing.assert_allclose(hard, soft.detach().cpu().numpy(), atol=atol, rtol=rtol)
    gradient, = torch.autograd.grad(soft.sum(), p)
    if not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
        raise ValueError('Model input gradients must be finite and nonzero')
    return {'hard_soft_max_error': float(np.abs(hard-soft.detach().cpu().numpy()).max()),
            'gradient_max_abs': float(gradient.abs().max().detach().cpu())}
