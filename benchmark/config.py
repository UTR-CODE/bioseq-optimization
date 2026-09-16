"""Shared configuration for the bioseq-optimization benchmark (Fig. 2)."""
import sys
import os

# Make both packages importable without pip install
sys.path.insert(0, '/home/hexin/bioseq-models')
sys.path.insert(0, '/mnt/data/home/hexin/MPRA_benchmark')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), 'plots')
GLUC_FASTA  = '/home/hexin/Gluc.fasta'

DEVICE = 'cuda'
SEED   = 42

# Fixed flanks used by context-dependent models (utr_code)
FIXED_UTR5 = 'ACGT' * 10   # 40 nt
FIXED_UTR3 = 'ACGT' * 10   # 40 nt

# Every optimizer starts from this many random sequences.
# num_candidates == NUM_INITIAL because _initial() requires len(initial) == num_candidates.
NUM_INITIAL  = 1000
# How many top sequences to keep per algorithm for the figure.
NUM_RETURN   = 10

# All 13 registered algorithms in display order.
ALGORITHMS = [
    'random',
    'ism_greedy',
    'simulated_annealing',
    'genetic_algorithm',
    'adalead',
    'basesearch',
    'beam_search',
    'cross_entropy_method',
    'cbas',
    'vanilla_activation_maximization',
    'fast_seqprop',
    'ledidi',
    'den',
]

# Algorithms that call score_soft(); bioseq_models scorers don't implement it.
# These runs are attempted and the resulting TypeError is saved, not raised.
GRADIENT_ALGORITHMS = {
    'vanilla_activation_maximization',
    'fast_seqprop',
    'ledidi',
    'den',
}

# Pretty labels for the x-axis
ALGO_LABELS = {
    'random':                         'Random',
    'ism_greedy':                     'ISM-Greedy',
    'simulated_annealing':            'Sim. Annealing',
    'genetic_algorithm':              'Genetic Alg.',
    'adalead':                        'AdaLead',
    'basesearch':                     'BaseSearch',
    'beam_search':                    'Beam Search',
    'cross_entropy_method':           'Cross-Entropy',
    'cbas':                           'CbAS',
    'vanilla_activation_maximization':'Vanilla ActMax',
    'fast_seqprop':                   'FastSeqProp',
    'ledidi':                         'Ledidi',
    'den':                            'DEN',
}

# Algorithm category for colouring
ALGO_CATEGORY = {
    'random':                         'discrete',
    'ism_greedy':                     'discrete',
    'simulated_annealing':            'discrete',
    'basesearch':                     'discrete',
    'genetic_algorithm':              'evolutionary',
    'adalead':                        'evolutionary',
    'beam_search':                    'distributional',
    'cross_entropy_method':           'distributional',
    'cbas':                           'distributional',
    'vanilla_activation_maximization':'gradient',
    'fast_seqprop':                   'gradient',
    'ledidi':                         'gradient',
    'den':                            'gradient',
}

CATEGORY_COLORS = {
    'discrete':      '#4E79A7',
    'evolutionary':  '#F28E2B',
    'distributional':'#59A14F',
    'gradient':      '#E15759',
}
