import pytest
import sequence_optimization as so


def test_twelve_public_algorithms_without_cbas():
    assert len(so.available_algorithms()) == 12
    assert 'cbas' not in so.available_algorithms()
    assert not hasattr(so, 'cbas_optimize')


def test_cbas_request_rejected_before_prediction():
    with pytest.raises(KeyError, match='unknown algorithm'):
        so.optimize(object(), 4, ['AAAA'], ['maximize'], 1,
                    algorithm='cbas', budget=2, diversity=0)
