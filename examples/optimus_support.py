"""Constraint used by the full Optimus 5-Prime example."""

import re

_HOMOPOLYMER = re.compile(r"([ACGT])\1{5,}")


def no_homopolymer6(sequences):
    """Allow only sequences without a run of six identical bases."""
    return [float(_HOMOPOLYMER.search(sequence) is None) for sequence in sequences]
