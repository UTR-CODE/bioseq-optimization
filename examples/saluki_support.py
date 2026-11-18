"""Fixed-transcript Saluki scoring and synonymous-codon attribution for eGFP."""

import numpy as np
import pandas as pd
import torch
from bioseq_models import load_model
from bioseq_models.adapters.saluki import annotation_tracks
from torch.nn import functional as F


UTR5 = 'ACATTTGCTTCTGACACAACTGTGTTCACTAGCAACCTCAAACAGACACC'
UTR3 = 'GCTGGAGCCTCGGTGGCCTAGCTTCTTGCCCCTTGGGCCTCCCCCCAGCCCCTCCTCCCCTTCCTGCACCCGTACCCCCGTGGTCTTTGAATAAAGTCTGAGTGGGCGGCA'

class ReporterScorer:
    """Score complete reporter transcripts while exposing only the CDS to search."""
    task_names = ('half_life_score',)
    supports_soft_scoring = True

    def __init__(self, length, device):
        self.device = device
        self.model = load_model('saluki', device=self.device, crosses=(0,), batch_size=32)
        self.network = self.model.model
        self.total_length = len(UTR5) + length + len(UTR3)
        self.annotations = dict(cds_start=len(UTR5), cds_end=len(UTR5)+length,
                                exon_lengths=[self.total_length])
        self.tracks = torch.from_numpy(annotation_tracks(self.total_length, **self.annotations)).to(self.device)
        self.left = self.encode([UTR5])
        self.right = self.encode([UTR3])
        # The convolution stack advances 64 nt per feature with a 320-nt
        # receptive field. Features starting after the transcript see only zeros.
        # Run the unchanged backwards GRU once over that constant right tail.
        stride, receptive_field = 1, self.network.initial.kernel_size[0]
        for block in self.network.blocks:
            receptive_field += stride * (block.conv.kernel_size[0] - 1 + block.pool.kernel_size - 1)
            stride *= block.pool.stride
        affected = (self.total_length + stride - 1) // stride
        self.prefix_length = receptive_field + (affected - 1) * stride
        total_steps = (self.network.seq_length - receptive_field) // stride + 1
        tail_steps = total_steps - affected
        assert tail_steps > 0
        with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
            feature = self.features(torch.zeros((1, 6, receptive_field), device=self.device))
            _, self.tail_hidden = self.network.gru(feature.expand(1, tail_steps, -1).contiguous())

    def encode(self, strings):
        values = list(strings)
        width = len(values[0])
        raw = np.frombuffer(''.join(values).encode('ascii'), dtype=np.uint8).reshape(len(values), width)
        return torch.from_numpy((raw[..., None] == np.frombuffer(b'ACGT', dtype=np.uint8)).astype(np.float32)).to(self.device)

    def features(self, x):
        net = self.network
        x = net.initial(x)
        for block in net.blocks:
            x = block(x)
        return torch.relu(net.final_norm(x.transpose(1, 2)))

    def forward(self, p):
        p = p.to(self.device)
        n = len(p)
        nt = torch.cat((self.left.expand(n, -1, -1), p, self.right.expand(n, -1, -1)), dim=1)
        x = torch.cat((nt, self.tracks.expand(n, -1, -1)), dim=-1).transpose(1, 2)
        x = F.pad(x, (0, self.prefix_length - self.total_length))
        net = self.network
        features = self.features(x).flip(1).contiguous()
        _, h = net.gru(features, self.tail_hidden.expand(-1, n, -1).contiguous())
        return net.output(torch.relu(net.bn2(net.dense(torch.relu(net.bn1(h[-1]))))))

    def score(self, sequences):
        values = list(sequences)
        if not values:
            return np.empty((0,1))
        with torch.no_grad():
            return np.concatenate([self.forward(self.encode(values[i:i+256])).cpu().numpy()
                                   for i in range(0,len(values),256)])

    def score_soft(self, probabilities):
        with torch.backends.cudnn.flags(enabled=False):
            return self.forward(probabilities), None

    def check(self, examples):
        transcripts = [dict(sequence=UTR5+s+UTR3, **self.annotations) for s in examples]
        expected = self.model.score_transcripts(transcripts)
        actual = self.score(examples)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        p = self.encode(examples[:2]).requires_grad_(True)
        soft, _ = self.score_soft(p)
        gradient, = torch.autograd.grad(soft.sum(), p)
        assert bool(gradient.isfinite().all() and gradient.abs().sum() > 0)
        np.testing.assert_allclose(soft.detach().cpu().numpy(), actual[:2], atol=1e-5, rtol=1e-5)
        return float(np.max(np.abs(actual-expected)))


def codon_ism(scorer, unique, space):
    tracks = []
    for row in unique.itertuples(index=False):
        parent = row.sequence
        mutants, positions, counts = [], [], np.zeros(len(parent))
        for i, choices in enumerate(space.choices):
            a = 3*i
            codon = parent[a:a+3]
            for alt in choices:
                if alt == codon:
                    continue
                mutants.append(parent[:a]+alt+parent[a+3:])
                changed = [a+k for k in range(3) if codon[k] != alt[k]]
                positions.append(changed)
            counts[a:a+3] = len(choices)
        values = scorer.score(mutants).ravel()
        attribution = np.zeros(len(parent))
        for value, positions_changed in zip(values, positions):
            attribution[positions_changed] += (row.score-float(value))/len(positions_changed)
        tracks.append(attribution/counts)
    return pd.DataFrame(tracks,index=unique.sequence_id)
