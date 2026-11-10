import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from sequence_optimization import random_seed_sequences, SaturationConfig
from sequence_optimization.workflows import (DESIGN_ALGORITHMS,generate_candidates,
    generate_candidates_parallel,deduplicate_candidates,select_sequence_panel)

class ToyScorer:
    device='cpu'
    task_names=('activity',)
    supports_soft_scoring=True
    def score(self,sequences):return np.array([[s.count('A')+.1*s.count('C')] for s in sequences])
    def score_soft(self,p):return (p[:,:,0]+.1*p[:,:,1]).sum(1,keepdim=True),None

def make_toy():return ToyScorer()

def test_all_eleven_records_and_provenance(monkeypatch,tmp_path):
    from sequence_optimization import workflows
    initial=random_seed_sequences(12,100,seed=12);calls=[]
    def fake(scorer,length,seeds,objective,count,**kw):
        calls.append(kw);scores=scorer.score(seeds)
        result=SimpleNamespace(sequences=seeds,activities=scores,scores=scores[:,0])
        return SimpleNamespace(stop_reason='plateau',total_query_count=1234,
            trajectory=[{'queries':1234,'score':4.}],at=lambda f:result)
    monkeypatch.setattr(workflows,'optimize',fake)
    d,t=generate_candidates(ToyScorer(),12,initial,['maximize'],output_dir=tmp_path/'run')
    assert len(d)==1100 and len(t)==len(DESIGN_ALGORITHMS)==11
    assert d.groupby('algorithm').size().eq(100).all()
    assert d.sequence_id.nunique()==100 and d.record_id.is_unique
    assert all(c['seed']==42 and c['saturation'].maximum_queries is None for c in calls)
    assert all(c['count_repeated'] for c in calls)
    unique,provenance=deduplicate_candidates(d)
    assert len(unique)==100 and len(provenance)==1100
    assert unique.algorithms.str.split(';').map(len).eq(11).all()
    assert json.loads((tmp_path/'run/config.json').read_text())['candidate_rows']==1100
    with pytest.raises(FileExistsError):generate_candidates(ToyScorer(),12,initial,['maximize'],output_dir=tmp_path/'run')

def test_real_serial_plateau(tmp_path):
    initial=random_seed_sequences(8,4,seed=5)
    sat=SaturationConfig(fractions=(.9,),selection='fraction',minimum_queries=4,
        plateau_patience_queries=12,plateau_patience_rounds=10)
    d,t=generate_candidates(ToyScorer(),8,initial,['maximize'],output_dir=tmp_path/'real',
        num_candidates=4,algorithms=('basesearch',),diversity=1,saturation=sat)
    assert len(d)==4 and t.stop_reason.eq('plateau').all()
    np.testing.assert_allclose(d.score,ToyScorer().score(d.sequence)[:,0])

def test_real_parallel_workflow_and_cleanup(tmp_path):
    initial=random_seed_sequences(8,4,seed=5)
    sat=SaturationConfig(fractions=(.9,),selection='fraction',minimum_queries=4,
        plateau_patience_queries=12,plateau_patience_rounds=10)
    d,t=generate_candidates_parallel(make_toy,8,initial,['maximize'],task_names=('activity',),
        output_dir=tmp_path/'parallel',num_candidates=4,algorithms=('basesearch','genetic_algorithm'),
        diversity=1,saturation=sat,max_batch_size=32,max_soft_batch_size=8)
    assert len(d)==8 and len(t)==2 and t.stop_reason.eq('plateau').all()
    assert (tmp_path/'parallel/gpu_batches.csv').exists()

def test_sequence_panel_quality_and_spacing():
    d=pd.DataFrame(dict(record_id=['a','b','c','d'],sequence=['AAAA','AAAC','CCCC','GGGG'],
        algorithm=['a']*4,score=[10.,9.,8.,1.]))
    selected,distance=select_sequence_panel(d,top_n=2,min_score=8.,distance_tie_nt=0)
    assert selected.sequence.tolist()==['AAAA','CCCC'] and distance[0,1]==4
    with pytest.raises(ValueError):select_sequence_panel(d,top_n=4,min_score=8.)
