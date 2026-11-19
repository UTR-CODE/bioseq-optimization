from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from sequence_optimization import (generate_designs, call_motifs, select_candidates,
    MotifCatalog, MultiTaskObjective, SaturationResult, OptimizationResult)
from sequence_optimization.workflows import select_sequence_panel


class CountModel(nn.Module):
    def forward(self,x):
        return x[:,2].sum(1,keepdim=True)


def test_generate_accepts_module_and_optional_seeds(tmp_path):
    designs=generate_designs(CountModel(),8,num_candidates=4,
        algorithms=['basesearch','genetic_algorithm'],score_threshold=0,diversity=1,
        tasks=['activity'],output_dir=tmp_path/'designs',seed=4)
    assert len(designs)==8 and designs.record_id.is_unique
    assert designs.sequence_id.nunique()<=8
    assert len(designs.attrs['timing'])==2
    assert designs.attrs['task_names']==['activity']
    assert (tmp_path/'designs/designs.csv').exists()
    assert (tmp_path/'designs/timing.csv').exists()
    np.testing.assert_allclose(designs.score,designs.activity_activity)


def test_default_pool_has_eleven_algorithms_and_90_percent_gain(monkeypatch):
    from sequence_optimization import suite
    calls=[]
    def fake(scorer,length,initial,objective,count,**kwargs):
        calls.append(kwargs)
        activity=scorer.score(initial[:count]).scores
        result=OptimizationResult(kwargs['algorithm'],np.array(initial[:count]),activity[:,0],
            np.zeros(count),count,activities=activity,task_names=scorer.task_names)
        return SaturationResult({.9:result},count,count,'plateau',[],selection='fraction')
    monkeypatch.setattr(suite,'optimize',fake)
    designs=generate_designs(CountModel(),30,seed=2,parallel=False)
    assert len(designs)==1100 and designs.algorithm.nunique()==11
    assert not designs.algorithm.eq('random').any()
    assert designs.groupby('algorithm').size().eq(100).all()
    assert all(c['saturation'].selection=='fraction' and c['saturation'].fractions==(.9,)
               and c['saturation'].maximum_queries is None for c in calls)


def test_default_selection_is_hamming(tmp_path):
    designs=pd.DataFrame(dict(sequence=['AAAA','AAAC','CCCC','GGGG'],
        score=[10.,9.,8.,1.],algorithm=['a']*4,record_id=list('abcd')))
    selected=select_candidates(designs,top_n=2,min_score=8.,distance_tie_nt=0,output_dir=tmp_path/'panel')
    baseline,_=select_sequence_panel(designs,top_n=2,min_score=8.,distance_tie_nt=0)
    assert selected.sequence.tolist()==baseline.sequence.tolist()==['AAAA','CCCC']
    assert selected.attrs['selection_method']=='hamming'
    assert selected.attrs['hamming_distances']==[[0,4],[4,0]]
    assert (tmp_path/'panel/selected_sequences.fa').exists()


def test_motif_catalog_selection_reproduces_reference(tmp_path):
    data=Path(__file__).resolve().parents[1]/'examples/data'
    designs=pd.read_csv(data/'k562_candidates_1100.csv')
    motifs=call_motifs(designs,catalog=data/'k562_catalog.meme',
        hits=pd.read_csv(data/'k562_fimo_sites.csv'),output_dir=tmp_path/'motifs')
    assert isinstance(motifs,MotifCatalog) and len(motifs)==11
    selected=select_candidates(designs,motifs=motifs,top_n=10,min_score=20.,output_dir=tmp_path/'selected')
    expected=pd.read_csv(Path(__file__).parent/'data/k562_top10_current.csv')
    assert selected.attrs['selection_method']=='motif'
    assert selected.sequence_id.tolist() == expected.sequence_id.tolist()
    assert len(selected.attrs['selection_trace'])==10
    assert (tmp_path/'selected/distance_matrices.npz').exists()
    assert (tmp_path/'selected/selected_sequences.fa').exists()
    assert not (tmp_path/'selected/panel').exists()


def test_explicit_empty_catalog_does_not_switch_to_hamming():
    with pytest.raises(ValueError,match='nonempty motif'):
        select_candidates(['AAAA','CCCC'],top_n=1,motifs={})


def test_precomputed_contributions_shape(monkeypatch):
    monkeypatch.setattr('sequence_optimization.framework._require_meme_tools', lambda *args: None)
    with pytest.raises(ValueError,match='one row per input record'):
        call_motifs(['ACGTACGT','GGGGCCCC'],contributions=np.zeros((1,8)))


def test_call_motifs_reuses_design_objective(monkeypatch):
    from sequence_optimization import framework
    monkeypatch.setattr(framework, '_require_meme_tools', lambda *args: None)
    designs=pd.DataFrame(dict(sequence=['AAAACCCC','GGGGTTTT'],score=[1.,0.],algorithm=['a','a']))
    designs.attrs['objective']=MultiTaskObjective(high=[0],low=[1])
    class TwoOutput(nn.Module):
        def forward(self,x):return torch.stack([x[:,0].sum(1),x[:,1].sum(1)],dim=1)
    class Done(Exception):pass
    def capture(sequences,fitness,**kwargs):
        np.testing.assert_allclose(fitness(['AAAACCCC','GGGGTTTT']),[0.,0.])
        raise Done
    monkeypatch.setattr(framework,'ism_contributions',capture)
    with pytest.raises(Done):call_motifs(designs,model=TwoOutput())


def test_cds_saturation_list_preserves_runtime_summary():
    from sequence_optimization import CallableScorer, SaturationConfig
    scorer = CallableScorer(lambda seq: np.array([s.count('G') for s in seq]))
    designs = generate_designs(scorer, 12, ['ATGGCTGCTTAA'], ['maximize'],
        algorithms=['genetic_algorithm'], num_candidates=2, diversity=0,
        is_CDS_optimized=True, seed=2,
        saturation=SaturationConfig(fractions=(.9,), selection='fraction',
            minimum_queries=2, plateau_patience_queries=1,
            plateau_patience_rounds=None))
    assert len(designs) == 2
    assert designs.attrs['timing'][0]['queries'] > 0


def test_eleven_real_optimizers_stop_on_plateau():
    from sequence_optimization import SaturationConfig
    designs = generate_designs(CountModel(), 12, num_candidates=4, diversity=1, seed=42,
        saturation=SaturationConfig(fractions=(.9,), selection='fraction',
            minimum_queries=4, plateau_patience_queries=12, plateau_patience_rounds=10),
        generator_pretrain_epochs=1, generator_update_epochs=1, den_seed_pretrain_epochs=1)
    assert len(designs) == 44 and designs.algorithm.nunique() == 11
    assert {r['stop_reason'] for r in designs.attrs['timing']} == {'plateau'}
