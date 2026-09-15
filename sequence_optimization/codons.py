"""Synonymous CDS operations using the NCBI standard genetic code (table 1)."""

from itertools import product
from math import prod
import numpy as np

from .api import SequenceSpace, normalize_nucleotide_sequences

_AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
GENETIC_CODE = dict(zip(map("".join, product("TCAG", repeat=3)), _AMINO))
STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})


class SynonymousCodonSpace(SequenceSpace):
    def __init__(self, initial_cds):
        if not isinstance(initial_cds, str):
            raise ValueError("initial CDS must be a nucleotide string")
        reference = str(normalize_nucleotide_sequences([initial_cds])[0])
        super().__init__(len(reference))
        if len(reference) < 6 or len(reference) % 3 or set(reference) - set("ACGT"):
            raise ValueError("initial CDS must be a complete ACGT CDS in frame")
        codons = [reference[i : i + 3] for i in range(0, len(reference), 3)]
        protein = [GENETIC_CODE[codon] for codon in codons]
        if codons[0] != "ATG":
            raise ValueError("initial CDS must begin with ATG")
        if codons[-1] not in STOP_CODONS:
            raise ValueError("initial CDS must end with TAA, TAG, or TGA")
        if "*" in protein[:-1]:
            raise ValueError("initial CDS must not contain an internal stop codon")
        table = {
            aa: tuple(c for c, value in GENETIC_CODE.items() if value == aa)
            for aa in set(_AMINO)
        }
        choices = [
            (codons[i],) if i == 0 else table[aa] for i, aa in enumerate(protein)
        ]
        object.__setattr__(self, "choices", tuple(choices))
        object.__setattr__(self, "capacity", prod(map(len, choices)))
        object.__setattr__(self, "initial_cds", reference)
        object.__setattr__(self, "protein", tuple(protein))

    def validate(self, sequences):
        values = super().validate(sequences)
        if any(sequence[:3] != "ATG" for sequence in values):
            raise ValueError("CDS candidates must begin with ATG")
        if any(sequence[-3:] not in STOP_CODONS for sequence in values):
            raise ValueError("CDS candidates must end with TAA, TAG, or TGA")
        if any(
            tuple(GENETIC_CODE[sequence[i : i + 3]] for i in range(0, self.length, 3))
            != self.protein
            for sequence in values
        ):
            raise ValueError("CDS candidates must preserve the initial protein")
        if any(
            sequence[3 * i : 3 * i + 3] not in choices
            for sequence in values
            for i, choices in enumerate(self.choices)
        ):
            raise ValueError(
                "CDS candidates must preserve the initial protein and standard codons"
            )
        return values

    def random(self, count, rng):
        columns = [rng.choice(choices, size=count) for choices in self.choices]
        return np.asarray(["".join(row) for row in zip(*columns)])

    def offspring(self, elite, count, mutation_rate, rng):
        parents = np.asarray(
            [[s[i : i + 3] for i in range(0, self.length, 3)] for s in elite]
        )
        a, b = parents[rng.integers(len(parents), size=(2, count))]
        child = np.where(rng.random(a.shape) < 0.5, a, b)
        for i, choices in enumerate(self.choices):
            mask = rng.random(count) < mutation_rate
            child[mask, i] = rng.choice(choices, size=mask.sum())
        return np.asarray(["".join(row) for row in child])

    def with_fixed_positions(self, template, positions):
        """Return this CDS space with codon choices restricted at fixed bases."""
        value = str(self.validate([template])[0])
        fixed = set(int(position) for position in positions)
        choices = []
        for codon_index, codon_choices in enumerate(self.choices):
            restricted = tuple(
                codon
                for codon in codon_choices
                if all(
                    codon[offset] == value[3 * codon_index + offset]
                    for offset in range(3)
                    if 3 * codon_index + offset in fixed
                )
            )
            choices.append(restricted)
        result = object.__new__(type(self))
        SequenceSpace.__init__(result, self.length, self.alphabet)
        object.__setattr__(result, "choices", tuple(choices))
        object.__setattr__(result, "capacity", prod(map(len, choices)))
        object.__setattr__(result, "initial_cds", self.initial_cds)
        object.__setattr__(result, "protein", self.protein)
        return result
