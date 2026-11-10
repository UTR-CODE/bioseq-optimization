import numpy as np
import pandas as pd
import pytest
from sequence_optimization.motifs import read_meme,write_meme,distinct_fimo_sites,select_motif_panel
from sequence_optimization.workflows import deduplicate_candidates

def test_meme_roundtrip_and_alphabet(tmp_path):
    p=np.full((6,4),.01);p[:,0]=.97
    write_meme(tmp_path/'motif.meme',{'A':p})
    np.testing.assert_allclose(read_meme(tmp_path/'motif.meme')['A'],p)
    (tmp_path/'bad.meme').write_text('MEME version 5\nALPHABET= AGCT\n')
    with pytest.raises(ValueError):read_meme(tmp_path/'bad.meme')

def test_site_dedup_thresholds_and_coordinates():
    d=pd.DataFrame([['M','s',2,8,'+',1e-7,.001],['M','s',2,8,'-',1e-6,.002],
        ['M','s',20,26,'+',1e-5,.01],['M','s',30,36,'+',.0001,.1]],
        columns=['motif_id','sequence_name','start','stop','strand','p-value','q-value'])
    sites=distinct_fimo_sites(d)
    assert len(sites)==2 and sites.start0.tolist()==[1,19]
    assert sites.center0.tolist()==[4.5,22.5]

def test_record_weighted_support_and_empty_evidence(tmp_path):
    records=pd.DataFrame(dict(record_id=['a','b','c','d'],sequence=['AAAAAAAA','AAAAAAAA','CCCCCCCC','GGGGGGGG'],
        algorithm=['a','b','a','a'],score=[30.,30.,29.,100.]))
    unique,records=deduplicate_candidates(records);ids=unique.set_index('sequence').sequence_id
    hits=pd.DataFrame([['M',ids['AAAAAAAA'],1,6,'+',1e-6,.001],['M',ids['CCCCCCCC'],2,7,'+',1e-6,.001]],
        columns=['motif_id','sequence_name','start','stop','strand','p-value','q-value'])
    p=np.full((6,4),.01);p[:,0]=.97
    selected,trace=select_motif_panel(records,hits,{'M':p},output_dir=tmp_path/'panel',top_n=2,min_sites=3)
    assert set(selected.sequence)=={'AAAAAAAA','CCCCCCCC'}
    support=pd.read_csv(tmp_path/'panel/motif_site_support.csv')
    assert support.sites_on_source_records.iloc[0]==3 and support.sites_on_unique_DNA.iloc[0]==2
    assert len(pd.read_csv(tmp_path/'panel/all_record_presence_matrix.csv'))==4
    with pytest.raises(ValueError,match='No motifs'):
        select_motif_panel(records,hits,{'M':p},output_dir=tmp_path/'fail',top_n=2,min_sites=4)

def test_bundled_reference_reproduces_expected_top10(tmp_path):
    from pathlib import Path
    data=Path(__file__).resolve().parents[1]/'examples/data'
    selected,_=select_motif_panel(pd.read_csv(data/'k562_candidates_1100.csv'),
        pd.read_csv(data/'k562_fimo_sites.csv'),read_meme(data/'k562_catalog.meme'),
        output_dir=tmp_path/'reference')
    expected=pd.read_csv(data/'k562_expected_top10.csv')
    assert selected.sequence.tolist()==expected.sequence.tolist()
    assert selected.sequence_id.tolist()==expected.sequence_id.tolist()
    np.testing.assert_allclose(selected.score,expected.score)
