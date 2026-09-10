"""Synonymous CDS operations using the NCBI standard genetic code (table 1)."""

from itertools import product
from math import prod
from collections.abc import Mapping

import numpy as np

from .api import SequenceSpace, normalize_nucleotide_sequences

_AMINO = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
GENETIC_CODE = dict(zip(map("".join, product("TCAG", repeat=3)), _AMINO))


class SynonymousCodonSpace(SequenceSpace):
    def __init__(self, reference_cds, synonymous_codons=None):
        if not isinstance(reference_cds, str):
            raise ValueError("reference_cds must be a nucleotide string")
        if synonymous_codons is not None and not isinstance(synonymous_codons, Mapping):
            raise ValueError("synonymous_codons must be an amino-acid mapping")
        reference = str(normalize_nucleotide_sequences([reference_cds])[0])
        super().__init__(len(reference))
        if len(reference) < 6 or len(reference) % 3 or set(reference) - set("ACGT"):
            raise ValueError("reference_cds must be a complete ACGT CDS in frame")
        codons = [reference[i : i + 3] for i in range(0, len(reference), 3)]
        protein = [GENETIC_CODE[codon] for codon in codons]
        if codons[0] != "ATG" or protein[-1] != "*" or "*" in protein[:-1]:
            raise ValueError(
                "reference_cds requires ATG, a terminal stop, and no internal stops"
            )
        table = {
            aa: tuple(c for c, value in GENETIC_CODE.items() if value == aa)
            for aa in set(_AMINO)
        }
        for aa, choices in (synonymous_codons or {}).items():
            if aa not in table or isinstance(choices, str) or not len(choices):
                raise ValueError(
                    "synonymous_codons maps amino-acid letters to nonempty codon lists"
                )
            choices = tuple(str(c).upper().replace("U", "T") for c in choices)
            if len(set(choices)) != len(choices) or any(
                GENETIC_CODE.get(c) != aa for c in choices
            ):
                raise ValueError(f"invalid synonymous codons for {aa!r}")
            table[aa] = choices
        choices = [
            (codons[i],) if i in (0, len(codons) - 1) else table[aa]
            for i, aa in enumerate(protein)
        ]
        object.__setattr__(self, "choices", tuple(choices))
        object.__setattr__(self, "capacity", prod(map(len, choices)))
        object.__setattr__(self, "reference_cds", reference)

    def validate(self, sequences):
        values = super().validate(sequences)
        if any(
            sequence[3 * i : 3 * i + 3] not in choices
            for sequence in values
            for i, choices in enumerate(self.choices)
        ):
            raise ValueError(
                "CDS candidates must preserve the reference protein and allowed codons"
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
