"""Verify saved workflow outputs can be resumed with the public API."""

import json

import numpy as np
import pandas as pd
import pytest

from MotifSift import call_motifs, generate_designs, load_designs, load_motifs, select_candidates
from sequence_optimization.framework import _objective


class CountBases:
    task_names = ("activity",)

    def score(self, sequences):
        return np.array([[seq.count('G')] for seq in sequences])


def test_load_designs_restores_gain_stages_and_cds_metadata(tmp_path):
    model = CountBases()
    folder = tmp_path / 'designs'
    frame = generate_designs(
        model, length=24, initial=['ATG' + 'GCT' * 6 + 'TAA'], objective='maximize',
        num_candidates=2, algorithms=['basesearch'], gain_fractions=(0.5, 0.9),
        is_CDS_optimized=True, fixed_region=[slice(-3, None)], output_dir=folder,
    )
    restored = load_designs(folder)
    pd.testing.assert_frame_equal(frame, restored)
    assert tuple(restored.attrs['cds_fixed_positions']) == (21, 22, 23)
    assert set(restored.gain_fraction) == {0.5, 0.9}
    np.testing.assert_allclose(_objective(restored, None).scalarize([[3]], ('activity',)), [3])


def test_load_designs_requires_original_custom_objective_for_attribution(tmp_path):
    pd.DataFrame({'sequence': ['ACGT'], 'score': [1.0]}).to_csv(tmp_path / 'designs.csv', index=False)
    (tmp_path / 'config.json').write_text(json.dumps({
        'objective': {'type': 'custom', 'function': 'score_fn'}, 'task_names': ['activity'],
    }))
    frame = load_designs(tmp_path)
    with pytest.raises(ValueError, match='Pass objective=score_fn'):
        _objective(frame, None)
    restored = load_designs(tmp_path, objective=lambda values: values['activity'] * 2)
    np.testing.assert_allclose(_objective(restored, None).scalarize([[3]], ('activity',)), [6])
    assert select_candidates(frame, top_n=1).sequence.tolist() == ['ACGT']


def test_load_motifs_keeps_forward_orientation_and_saved_hits(tmp_path):
    designs = pd.DataFrame({'sequence': ['ATATATAT'], 'score': [1.0]})
    pwm = np.tile([0.97, 0.01, 0.01, 0.01], (6, 1))
    fresh = call_motifs(designs, catalog={'M': pwm}, reverse_complement=False)
    hits = pd.DataFrame({
        'motif_id': ['M'], 'sequence_name': list(fresh.sequence_ids), 'strand': ['+'],
        'start': [1], 'stop': [6], 'p-value': [1e-6], 'q-value': [0.01],
    })
    folder = tmp_path / 'motifs'
    call_motifs(designs, catalog={'M': pwm}, hits=hits, reverse_complement=False, output_dir=folder)
    restored = load_motifs(folder)
    assert not restored.reverse_complement
    assert restored.sequence_ids == fresh.sequence_ids
    assert restored.q_threshold == fresh.q_threshold
    pd.testing.assert_frame_equal(restored.hits, hits)
