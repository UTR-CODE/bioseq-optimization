"""Task definitions for the bioseq-optimization benchmark (Fig. 2).

Each task is a dict with keys:
  id          – short identifier used in file names
  label       – human-readable label for figures
  length      – sequence length for the optimiser
  make_scorer – zero-arg callable returning the scorer (loaded on demand)
  objective   – TargetObjective | MultiTaskObjective | list passed to optimize()
  is_cds      – bool; True activates SynonymousCodonSpace via is_CDS_optimized
  initial_cds – str; reference CDS for is_cds tasks (None otherwise)
  seq_type    – short string for axis labels
"""
import sys
sys.path.insert(0, '/home/hexin/bioseq-models')
sys.path.insert(0, '/mnt/data/home/hexin/MPRA_benchmark')

from config import DEVICE, FIXED_UTR5, FIXED_UTR3, GLUC_FASTA
from sequence_optimization import CombinedScorer
from sequence_optimization.objectives import MultiTaskObjective, TargetObjective


def _read_gluc_cds(path):
    lines = open(path).read().splitlines()
    return ''.join(l.strip() for l in lines if not l.startswith('>'))


def _make_mouse_tissue_scorer():
    from bioseq_models import load_model
    tissues = ['heart', 'limb', 'midbrain', 'forebrain', 'hindbrain', 'neural_tube']
    models  = [load_model(f'deepstarr_mouse_{t}_activity', device=DEVICE) for t in tissues]
    return CombinedScorer(models, names=tissues)


def _make_embryo_activity_scorer():
    from bioseq_models import load_model
    tissues = ['brain_vnc', 'muscle', 'epidermis', 'gut', 'brain_specific']
    models  = [load_model(f'deepstarr_embryo_{t}_activity', device=DEVICE) for t in tissues]
    return CombinedScorer(models, names=tissues)


def _make_agarwal_scorer():
    from bioseq_models import load_model
    return CombinedScorer(
        [load_model('agarwal_k562',  device=DEVICE),
         load_model('agarwal_hepg2', device=DEVICE),
         load_model('agarwal_wtc11', device=DEVICE)],
        names=['k562', 'hepg2', 'wtc11'],
    )


TASKS = [
    {
        'id':           'task1_optimus_utr5',
        'label':        "Task 1\nOptimus 5′UTR\n↑MRL",
        'length':       50,
        'make_scorer':  lambda: __import__('bioseq_models').load_model('optimus5prime', device=DEVICE),
        'objective':    TargetObjective(target=0, mode='maximize'),
        'is_cds':       False,
        'initial_cds':  None,
        'seq_type':     "5′UTR (50 nt)",
        'max_queries':  100_000,
    },
    {
        'id':           'task2_cetnar_utr3',
        'label':        "Task 2\nCetnar 3′UTR\n↓decay rate",
        'length':       80,
        'make_scorer':  lambda: __import__('bioseq_models').load_model('cetnar_stability', device=DEVICE),
        'objective':    MultiTaskObjective(directions=['minimize']),
        'is_cds':       False,
        'initial_cds':  None,
        'seq_type':     "3′UTR (80 nt)",
        'max_queries':  100_000,
    },
    {
        'id':           'task3_utrcode_cds',
        'label':        "Task 3\nUTR-CODE CDS\n↑translation",
        'length':       len(_read_gluc_cds(GLUC_FASTA)),
        'make_scorer':  lambda: __import__('bioseq_models').load_model(
                            'utr_code', design_region='cds',
                            utr5=FIXED_UTR5, utr3=FIXED_UTR3, device=DEVICE),
        'objective':    TargetObjective(target=0, mode='maximize'),
        'is_cds':       True,
        'initial_cds':  _read_gluc_cds(GLUC_FASTA),
        'seq_type':     f"CDS ({len(_read_gluc_cds(GLUC_FASTA))} nt)",
        'max_queries':  100_000,
    },
    {
        'id':           'task4_dream_promoter',
        'label':        "Task 4\nDREAM-LegNet\n↑promoter activity",
        'length':       93,
        'make_scorer':  lambda: __import__('bioseq_models').load_model(
                            'dream_legnet', variable_region_only=True, device=DEVICE),
        'objective':    TargetObjective(target=0, mode='maximize'),
        'is_cds':       False,
        'initial_cds':  None,
        'seq_type':     "Variable region (93 nt)",
        'max_queries':  100_000,
    },
    {
        'id':           'task5_deepstarr_hk',
        'label':        "Task 5\nDeepSTARR\n↑HK, ↓Dev",
        'length':       249,
        'make_scorer':  lambda: __import__('bioseq_models').load_model('deepstarr', device=DEVICE),
        # deepstarr returns [developmental, housekeeping]
        'objective':    MultiTaskObjective(directions=['minimize', 'maximize']),
        'is_cds':       False,
        'initial_cds':  None,
        'seq_type':     "Enhancer (249 bp)",
        'max_queries':  100_000,
    },
    {
        'id':           'task6_agarwal_k562',
        'label':        "Task 6\nAgarwal lentiMPRA\n↑K562, ↓HepG2, ↓WTC11",
        'length':       230,
        'make_scorer':  _make_agarwal_scorer,
        'objective':    MultiTaskObjective(directions=['maximize', 'minimize', 'minimize']),
        'is_cds':       False,
        'initial_cds':  None,
        'seq_type':     "Enhancer (230 bp)",
        'max_queries':  100_000,
    },
    {
        'id':           'task7_malinois_k562',
        'label':        "Task 7\nMalinois CRE\n↑K562, ↓HepG2, ↓SK-N-SH",
        'length':       200,
        'make_scorer':  lambda: __import__('bioseq_models').load_model('malinois', device=DEVICE),
        # malinois returns [K562, HepG2, SKNSH]
        'objective':    MultiTaskObjective(directions=['maximize', 'minimize', 'minimize']),
        'is_cds':       False,
        'initial_cds':  None,
        'seq_type':     "CRE (200 bp)",
        'max_queries':  100_000,
    },
    {
        'id':           'task8_mouse_heart',
        'label':        "Task 8\nDeepSTARR-Mouse\n↑Heart, ↓other tissues",
        'length':       1001,
        'make_scorer':  _make_mouse_tissue_scorer,
        # tissues: heart, limb, midbrain, forebrain, hindbrain, neural_tube
        'objective':    MultiTaskObjective(
                            directions=['maximize', 'minimize', 'minimize',
                                        'minimize', 'minimize', 'minimize']),
        'is_cds':       False,
        'initial_cds':  None,
        'seq_type':     "Enhancer (1001 bp)",
        'max_queries':  20_000,   # heavier model ensemble; lower cap
    },
    {
        'id':           'task9_embryo_brain',
        'label':        "Task 9\nDeepSTARR-Embryo\n↑Brain/VNC, ↓other tissues",
        'length':       1001,
        'make_scorer':  _make_embryo_activity_scorer,
        # tissues: brain_vnc, muscle, epidermis, gut, brain_specific
        'objective':    MultiTaskObjective(
                            directions=['maximize', 'minimize', 'minimize',
                                        'minimize', 'minimize']),
        'is_cds':       False,
        'initial_cds':  None,
        'seq_type':     "Enhancer (1001 bp)",
        'max_queries':  20_000,   # heavier model ensemble; lower cap
    },
]
