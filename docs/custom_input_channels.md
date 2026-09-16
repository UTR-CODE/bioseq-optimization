# Custom Model Input Channels

This guide shows how to use models with non-standard input beyond the default 4-channel ACGT one-hot encoding.

## Overview

The default `SequencePredictor` expects models to accept 4-channel one-hot encoded DNA/RNA sequences. For models requiring additional input channels (epigenetic marks, conservation scores, structural features, etc.), inherit `SequencePredictor` and override the encoding methods.

## Key Methods to Override

- `encode_sequences()`: Convert string sequences to tensor with custom channels
- `prepare_input()`: Reshape tensor for your model's expected input format
- `score()` and `score_soft()`: Only if you need custom batching or post-processing

## Example: Adding Conservation Scores

```python
import torch
from torch import nn
import numpy as np
from sequence_optimization import SequencePredictor, generate_designs, random_seed_sequences

class MultiChannelPredictor(SequencePredictor):
    """Predictor for models with additional input channels beyond ACGT."""
    
    def __init__(self, model, conservation_scores, **kwargs):
        super().__init__(model, **kwargs)
        self.conservation_scores = conservation_scores  # Pre-computed conservation track
    
    def encode_sequences(self, sequences):
        """Override to add conservation channel."""
        # Get standard 4-channel one-hot encoding
        onehot = super().encode_sequences(sequences)  # Shape: [batch, length, 4]
        
        # Add conservation scores as 5th channel
        batch_size = len(sequences)
        seq_length = len(sequences[0])
        conservation = torch.from_numpy(
            self.conservation_scores[:seq_length].reshape(1, seq_length, 1)
        ).repeat(batch_size, 1, 1)
        
        # Concatenate: [batch, length, 5]
        return torch.cat([onehot, conservation], dim=2)
    
    def prepare_input(self, encoded):
        """Override to handle 5-channel input."""
        # Model expects [batch, 5, length]
        return encoded.transpose(1, 2)

# Define model with 5 input channels
class MultiChannelCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(5, 64, kernel_size=8)  # 5 input channels
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# Create predictor with conservation track
model = MultiChannelCNN()
conservation_track = np.random.rand(1000)  # Example: PhyloP conservation scores

predictor = MultiChannelPredictor(
    model,
    conservation_scores=conservation_track,
    device="cuda",
    tasks=["stability"]  # Task names for output columns
)

# Run optimization
initial = random_seed_sequences(length=50, count=100, seed=42)
result_df = generate_designs(
    predictor,
    length=50,
    initial=initial,
    objective=["maximize"],
    seed=42,
)

print(result_df.head())
```

## Example: Epigenetic Marks (Multiple Tracks)

```python
class EpigeneticPredictor(SequencePredictor):
    """Predictor with DNA sequence + multiple epigenetic tracks."""
    
    def __init__(self, model, h3k4me3, h3k27ac, dnase, **kwargs):
        super().__init__(model, **kwargs)
        self.h3k4me3 = h3k4me3    # Histone modification track
        self.h3k27ac = h3k27ac    # Histone acetylation track
        self.dnase = dnase        # DNase hypersensitivity track
    
    def encode_sequences(self, sequences):
        """Encode sequences with 3 epigenetic marks (7 total channels)."""
        # Get base 4-channel one-hot
        onehot = super().encode_sequences(sequences)  # [batch, length, 4]
        
        batch_size = len(sequences)
        seq_length = len(sequences[0])
        
        # Add epigenetic tracks
        h3k4me3_tensor = torch.from_numpy(
            self.h3k4me3[:seq_length].reshape(1, seq_length, 1)
        ).repeat(batch_size, 1, 1)
        
        h3k27ac_tensor = torch.from_numpy(
            self.h3k27ac[:seq_length].reshape(1, seq_length, 1)
        ).repeat(batch_size, 1, 1)
        
        dnase_tensor = torch.from_numpy(
            self.dnase[:seq_length].reshape(1, seq_length, 1)
        ).repeat(batch_size, 1, 1)
        
        # Concatenate all channels: [batch, length, 7]
        return torch.cat([onehot, h3k4me3_tensor, h3k27ac_tensor, dnase_tensor], dim=2)
    
    def prepare_input(self, encoded):
        """Model expects [batch, 7, length]."""
        return encoded.transpose(1, 2)

# Usage
class EpigeneticCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(7, 128, kernel_size=10)  # 7 input channels
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

model = EpigeneticCNN()
predictor = EpigeneticPredictor(
    model,
    h3k4me3=np.random.rand(1000),
    h3k27ac=np.random.rand(1000),
    dnase=np.random.rand(1000),
    device="cuda",
    tasks=["expression"]
)
```

## Example: RNA Structure Features

```python
class StructuralPredictor(SequencePredictor):
    """Predictor with sequence + predicted secondary structure."""
    
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
    
    def encode_sequences(self, sequences):
        """Add predicted structure probabilities as additional channels."""
        import RNA  # ViennaRNA Python bindings
        
        # Base 4-channel encoding
        onehot = super().encode_sequences(sequences)  # [batch, length, 4]
        
        batch_size = len(sequences)
        seq_length = len(sequences[0])
        
        # Predict structure for each sequence
        structure_probs = []
        for seq in sequences:
            # Get base pairing probabilities
            _, pf = RNA.pf_fold(seq)
            bp_prob = np.array([RNA.get_pr(i+1, seq_length) for i in range(seq_length)])
            structure_probs.append(bp_prob)
        
        structure_tensor = torch.from_numpy(
            np.array(structure_probs).reshape(batch_size, seq_length, 1)
        ).float()
        
        # Concatenate: [batch, length, 5]
        return torch.cat([onehot, structure_tensor], dim=2)
    
    def prepare_input(self, encoded):
        return encoded.transpose(1, 2)
```

## Example: Position Embeddings

```python
class PositionalPredictor(SequencePredictor):
    """Predictor with position-aware encoding."""
    
    def __init__(self, model, max_length=1000, **kwargs):
        super().__init__(model, **kwargs)
        self.max_length = max_length
        # Pre-compute sinusoidal position embeddings
        self.pos_embeddings = self._create_position_embeddings()
    
    def _create_position_embeddings(self):
        """Create sinusoidal position embeddings."""
        position = torch.arange(0, self.max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 64, 2) * -(np.log(10000.0) / 64))
        
        pe = torch.zeros(self.max_length, 64)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def encode_sequences(self, sequences):
        """Add position embeddings to sequence encoding."""
        onehot = super().encode_sequences(sequences)  # [batch, length, 4]
        
        batch_size = len(sequences)
        seq_length = len(sequences[0])
        
        # Get position embeddings for this length
        pos_emb = self.pos_embeddings[:seq_length].unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Concatenate: [batch, length, 68]
        return torch.cat([onehot, pos_emb], dim=2)
    
    def prepare_input(self, encoded):
        return encoded.transpose(1, 2)
```

## Tips

1. **Data alignment**: Ensure additional tracks align correctly with sequence positions
2. **Normalization**: Normalize epigenetic/conservation scores to [0, 1] range
3. **Batch consistency**: All sequences in a batch must have the same length
4. **Device handling**: Use `.to(device)` for all tensors if needed
5. **Gradient support**: Both `encode_sequences()` and `prepare_input()` must preserve gradients for differentiable optimization

## Common Patterns

### Loading Pre-computed Tracks

```python
# Load tracks from files
import pyBigWig

class BigWigPredictor(SequencePredictor):
    def __init__(self, model, bigwig_path, chrom, start, **kwargs):
        super().__init__(model, **kwargs)
        bw = pyBigWig.open(bigwig_path)
        self.track = np.array(bw.values(chrom, start, start + 10000))
        bw.close()
```

### Dynamic Feature Computation

```python
class DynamicFeaturePredictor(SequencePredictor):
    def encode_sequences(self, sequences):
        onehot = super().encode_sequences(sequences)
        
        # Compute features on-the-fly
        gc_content = self._compute_gc_profile(sequences)
        dinuc_freq = self._compute_dinucleotide_freq(sequences)
        
        return torch.cat([onehot, gc_content, dinuc_freq], dim=2)
    
    def _compute_gc_profile(self, sequences):
        # Sliding window GC content
        ...
    
    def _compute_dinucleotide_freq(self, sequences):
        # Dinucleotide frequencies
        ...
```

### Multi-Species Conservation

```python
class ConservationPredictor(SequencePredictor):
    def __init__(self, model, conservation_tracks, **kwargs):
        """
        conservation_tracks: dict of species -> conservation array
        Example: {'human': array, 'mouse': array, 'zebrafish': array}
        """
        super().__init__(model, **kwargs)
        self.conservation_tracks = conservation_tracks
    
    def encode_sequences(self, sequences):
        onehot = super().encode_sequences(sequences)
        
        # Stack conservation from multiple species
        all_conservation = []
        for species, track in self.conservation_tracks.items():
            cons_tensor = torch.from_numpy(track[:len(sequences[0])])
            all_conservation.append(cons_tensor)
        
        conservation = torch.stack(all_conservation, dim=-1)
        # Broadcast to batch
        conservation = conservation.unsqueeze(0).repeat(len(sequences), 1, 1)
        
        return torch.cat([onehot, conservation], dim=2)
```

## See Also

- [Custom models guide](custom_models.md) - For standard 4-channel custom models
- [Constraints examples](constraints_examples.md) - For sequence filtering during optimization
