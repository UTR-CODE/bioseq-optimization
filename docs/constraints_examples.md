# Sequence Constraint Examples

This document provides example constraint functions for use with `generate_designs()`. Each constraint function accepts a batch of sequences and returns probabilities (0-1) for filtering.

## Constraint Function Interface

```python
def constraint_function(sequences: list[str]) -> np.ndarray:
    """
    Args:
        sequences: List of DNA/RNA sequences
    
    Returns:
        np.ndarray: Probabilities (0-1) for each sequence
            - 0.0: Reject (hard constraint)
            - 1.0: Accept
            - 0.0-1.0: Soft threshold (probability of acceptance)
    """
    pass
```

## GC Content Constraints

### Hard GC Bounds

```python
import numpy as np

def gc_hard_bounds(sequences, min_gc=0.4, max_gc=0.6):
    """Hard constraint: reject sequences outside GC range."""
    gc_contents = np.array([
        (seq.count('G') + seq.count('C')) / len(seq)
        for seq in sequences
    ])
    return ((gc_contents >= min_gc) & (gc_contents <= max_gc)).astype(float)
```

### Soft GC Threshold

```python
def gc_soft_threshold(sequences, target=0.5, sigma=0.1):
    """Soft constraint: prefer sequences near target GC content."""
    gc_contents = np.array([
        (seq.count('G') + seq.count('C')) / len(seq)
        for seq in sequences
    ])
    # Gaussian soft threshold centered at target
    return np.exp(-((gc_contents - target) ** 2) / (2 * sigma ** 2))
```

## Motif Exclusion

### Exclude Specific Motifs

```python
def exclude_motifs(sequences, motifs=['AAAAA', 'TTTTT', 'CCCCC', 'GGGGG']):
    """Hard constraint: reject sequences containing any specified motif."""
    return np.array([
        0.0 if any(motif in seq for motif in motifs) else 1.0
        for seq in sequences
    ])
```

### Exclude Restriction Sites

```python
def exclude_restriction_sites(sequences):
    """Hard constraint: reject sequences with common restriction sites."""
    # Common restriction enzyme recognition sites
    sites = [
        'GAATTC',  # EcoRI
        'GGATCC',  # BamHI
        'AAGCTT',  # HindIII
        'CTGCAG',  # PstI
        'GTCGAC',  # SalI
        'GCGGCCGC', # NotI
    ]
    return exclude_motifs(sequences, motifs=sites)
```

## Homopolymer Filtering

### Exclude Long Homopolymers

```python
import re

def exclude_homopolymers(sequences, max_length=4):
    """Hard constraint: reject sequences with homopolymers >= max_length."""
    pattern = r'([ACGT])\1{' + str(max_length) + r',}'
    return np.array([
        0.0 if re.search(pattern, seq) else 1.0
        for seq in sequences
    ])
```

### Soft Homopolymer Penalty

```python
def soft_homopolymer_penalty(sequences, penalty_per_base=0.1):
    """Soft constraint: penalize long homopolymers gradually."""
    results = []
    for seq in sequences:
        # Find longest homopolymer
        max_homo = 0
        current_homo = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                current_homo += 1
                max_homo = max(max_homo, current_homo)
            else:
                current_homo = 1
        
        # Penalty increases with homopolymer length
        if max_homo <= 3:
            prob = 1.0
        else:
            prob = max(0.0, 1.0 - (max_homo - 3) * penalty_per_base)
        results.append(prob)
    
    return np.array(results)
```

## Sequence Complexity

### Minimum Complexity Filter

```python
from collections import Counter

def minimum_complexity(sequences, min_unique_ratio=0.5):
    """Hard constraint: reject low-complexity sequences."""
    results = []
    for seq in sequences:
        # Count unique k-mers (k=3)
        kmers = [seq[i:i+3] for i in range(len(seq) - 2)]
        unique_ratio = len(set(kmers)) / len(kmers) if kmers else 0
        results.append(1.0 if unique_ratio >= min_unique_ratio else 0.0)
    
    return np.array(results)
```

### Dinucleotide Diversity

```python
def dinucleotide_diversity(sequences, min_diversity=0.6):
    """Soft constraint: prefer sequences with diverse dinucleotides."""
    results = []
    for seq in sequences:
        # Count dinucleotides
        dinucs = [seq[i:i+2] for i in range(len(seq) - 1)]
        if not dinucs:
            results.append(0.0)
            continue
        
        # Calculate Shannon entropy
        counter = Counter(dinucs)
        total = len(dinucs)
        entropy = -sum((count/total) * np.log2(count/total) 
                      for count in counter.values())
        max_entropy = np.log2(16)  # 16 possible dinucleotides
        
        # Normalize and apply soft threshold
        diversity = entropy / max_entropy
        prob = np.clip(diversity / min_diversity, 0, 1)
        results.append(prob)
    
    return np.array(results)
```

## RNA Secondary Structure

### Avoid Strong Hairpins

```python
def avoid_strong_hairpins(sequences, max_stem_energy=-10.0):
    """
    Soft constraint: penalize sequences with strong predicted hairpins.
    Requires RNA folding library (e.g., ViennaRNA).
    """
    try:
        import RNA  # ViennaRNA Python bindings
    except ImportError:
        # Fallback: simple complementarity check
        return simple_hairpin_check(sequences)
    
    results = []
    for seq in sequences:
        # Predict MFE structure
        structure, mfe = RNA.fold(seq)
        
        # Soft threshold: penalize strong hairpins
        if mfe >= max_stem_energy:
            prob = 1.0
        else:
            # Gradual penalty for stronger structures
            prob = max(0.1, (mfe - max_stem_energy) / max_stem_energy + 1.0)
        results.append(prob)
    
    return np.array(results)

def simple_hairpin_check(sequences, max_stem=6):
    """Simple complementarity-based hairpin check (fallback)."""
    complement = str.maketrans('ACGT', 'TGCA')
    results = []
    
    for seq in sequences:
        has_strong_hairpin = False
        seq_len = len(seq)
        
        # Check for potential stems
        for i in range(seq_len - max_stem):
            forward = seq[i:i+max_stem]
            for j in range(i + 4, seq_len - max_stem + 1):  # Min loop size = 3
                reverse = seq[j:j+max_stem][::-1].translate(complement)
                if forward == reverse:
                    has_strong_hairpin = True
                    break
            if has_strong_hairpin:
                break
        
        results.append(0.3 if has_strong_hairpin else 1.0)
    
    return np.array(results)
```

## Codon Usage (for CDS)

### Preferred Codons

```python
def preferred_codons(sequences, preferred_table=None):
    """Soft constraint: favor sequences using preferred codons."""
    if preferred_table is None:
        # Example: E. coli highly expressed genes
        preferred_table = {
            'A': ['GCT', 'GCC'],
            'C': ['TGC'],
            'D': ['GAC'],
            'E': ['GAA'],
            'F': ['TTC'],
            'G': ['GGC', 'GGT'],
            'H': ['CAC'],
            'I': ['ATC'],
            'K': ['AAA'],
            'L': ['CTG'],
            'M': ['ATG'],
            'N': ['AAC'],
            'P': ['CCG'],
            'Q': ['CAG'],
            'R': ['CGT', 'CGC'],
            'S': ['AGC'],
            'T': ['ACC'],
            'V': ['GTT', 'GTG'],
            'W': ['TGG'],
            'Y': ['TAC'],
            '*': ['TAA', 'TAG', 'TGA'],
        }
    
    # Flatten preferred codons
    preferred_codons_set = set(
        codon for codons in preferred_table.values() for codon in codons
    )
    
    results = []
    for seq in sequences:
        if len(seq) % 3 != 0:
            results.append(0.0)
            continue
        
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        preferred_count = sum(1 for codon in codons if codon in preferred_codons_set)
        ratio = preferred_count / len(codons) if codons else 0
        
        results.append(ratio)
    
    return np.array(results)
```

## Sequence Naturalness with Diffusion Models

### Example: Diffusion Model Constraint

```python
def diffusion_naturalness_score(sequences, model=None, threshold=0.7):
    """
    Soft constraint: score sequence naturalness using a diffusion model.
    
    This is a conceptual example. In practice, you would:
    1. Load a pretrained diffusion model trained on natural sequences
    2. Compute the likelihood or denoising score for each sequence
    3. Use the score as a soft constraint
    
    Args:
        sequences: List of sequences to score
        model: Pretrained diffusion model (placeholder)
        threshold: Minimum naturalness score
    
    Returns:
        Array of probabilities (0-1) based on naturalness
    """
    # Placeholder implementation
    # In reality, you would call your diffusion model here
    
    if model is None:
        # Fallback: simple k-mer frequency check against natural distribution
        return kmer_naturalness_check(sequences, threshold)
    
    results = []
    for seq in sequences:
        # Hypothetical: compute diffusion model likelihood
        # naturalness_score = model.compute_likelihood(seq)
        
        # For this example, use a placeholder score
        naturalness_score = 0.8  # Placeholder
        
        # Soft threshold
        prob = np.clip(naturalness_score / threshold, 0, 1)
        results.append(prob)
    
    return np.array(results)

def kmer_naturalness_check(sequences, threshold=0.7, k=5):
    """
    Fallback: Check k-mer frequencies against natural distribution.
    
    In practice, you would load k-mer frequencies from a large corpus
    of natural sequences and compare input sequences against them.
    """
    # Placeholder: simplified naturalness score
    results = []
    for seq in sequences:
        # Extract k-mers
        kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        unique_kmers = len(set(kmers))
        
        # Simple heuristic: natural sequences have moderate k-mer diversity
        diversity = unique_kmers / len(kmers) if kmers else 0
        
        # Expect diversity between 0.5 and 0.9 for natural sequences
        if 0.5 <= diversity <= 0.9:
            score = 1.0
        else:
            score = max(0.0, 1.0 - abs(diversity - 0.7) * 2)
        
        results.append(score)
    
    return np.array(results)
```

## Combining Multiple Constraints

### Sequential Combination

```python
def combine_constraints(sequences, constraint_functions):
    """
    Combine multiple constraints by multiplying probabilities.
    
    Args:
        sequences: List of sequences
        constraint_functions: List of constraint functions
    
    Returns:
        Combined probability array
    """
    combined_prob = np.ones(len(sequences))
    
    for constraint_fn in constraint_functions:
        probs = constraint_fn(sequences)
        combined_prob *= probs
    
    return combined_prob
```

### Usage Example

```python
from sequence_optimization import generate_designs, random_seed_sequences

# Define constraints
constraints = [
    lambda seqs: gc_soft_threshold(seqs, target=0.5, sigma=0.1),
    exclude_homopolymers,
    lambda seqs: exclude_restriction_sites(seqs),
    lambda seqs: minimum_complexity(seqs, min_unique_ratio=0.6),
]

# Run optimization with combined constraints
initial = random_seed_sequences(length=230, count=100, seed=42)

result_df = generate_designs(
    model,
    length=230,
    initial=initial,
    objective=["maximize"],
    constraint=constraints,
    seed=42,
)
```

## Performance Tips

1. **Vectorize operations**: Use NumPy operations instead of Python loops when possible
2. **Batch processing**: Constraint functions receive batches of sequences, not individual sequences
3. **Cache results**: For expensive constraints (e.g., RNA folding), consider caching
4. **Soft vs Hard**: Use soft constraints (0-1) when possible to provide gradient information
5. **Constraint ordering**: Place cheap filters before expensive ones in the list

## References

- ViennaRNA: RNA secondary structure prediction
- RNAfold: Minimum free energy (MFE) RNA folding
- Diffusion models for sequence generation: See recent literature on DNA/RNA diffusion models
