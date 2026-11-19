import pandas as pd
import pytest

from sequence_optimization.workflows import select_sequence_panel


def test_sequence_panel_quality_and_spacing():
    d=pd.DataFrame(dict(record_id=['a','b','c','d'],sequence=['AAAA','AAAC','CCCC','GGGG'],
        algorithm=['a']*4,score=[10.,9.,8.,1.]))
    selected,distance=select_sequence_panel(d,top_n=2,min_score=8.,distance_tie_nt=0)
    assert selected.sequence.tolist()==['AAAA','CCCC'] and distance[0,1]==4
    with pytest.raises(ValueError):select_sequence_panel(d,top_n=4,min_score=8.)
