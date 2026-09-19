import numpy as np
import pytest
from sequence_optimization.algorithms import _select_diverse


def reference(ranked, count, diversity):
    if count == 1 or diversity == 0:
        return ranked[:count]
    best = []
    for start in range(min(len(ranked), 32)):
        selected = [ranked[start]]
        for i, item in enumerate(ranked):
            if i == start:
                continue
            if all(sum(a != b for a, b in zip(item[0], previous[0])) >= diversity for previous in selected):
                selected.append(item)
                if len(selected) == count:
                    return selected
        if len(selected) > len(best):
            best = selected
    return best


@pytest.mark.parametrize('length', [1, 7, 32, 249, 500])
@pytest.mark.parametrize('alphabet', ['ACGT', 'ACGU', 'ABCDN'])
def test_lazy_selection_matches_original(length, alphabet):
    rng = np.random.default_rng(42)
    sequences = [''.join(row) for row in rng.choice(list(alphabet), (50, length))]
    ranked = [(s, (float(i), 0.)) for i, s in enumerate(sequences)]
    for count in [1, 10, 50]:
        for diversity in [0, 1, 5, length]:
            assert _select_diverse(ranked, count, diversity) == reference(ranked, count, diversity)


def test_blocking_seed_and_empty():
    ranked = [('AA',(3.,0.)),('AC',(2.,0.)),('CA',(1.,0.))]
    assert _select_diverse(ranked,2,2) == ranked[1:]
    assert _select_diverse([],2,2) == []
