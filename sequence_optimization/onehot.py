"""Vectorized DNA/RNA one-hot conversion without per-base Python loops.

Reference: SeqPro's byte-array representation and alphabet-based one-hot
conversion. This independent NumPy implementation uses a cached byte LUT.
https://github.com/ML4GLand/SeqPro/blob/main/python/seqpro/_encoders.py
"""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=32)
def _tables(alphabet, dtype):
    table = np.zeros((256, 4), dtype=np.dtype(dtype))
    valid = np.zeros(256, dtype=bool)
    for column, base in enumerate(alphabet):
        for symbol in (base, base.lower()):
            table[ord(symbol), column] = 1
            valid[ord(symbol)] = True
    for symbol in "Uu":
        table[ord(symbol)] = table[ord("T")]
        valid[ord(symbol)] = True
    valid[ord("N")] = valid[ord("n")] = True
    return table, valid


def onehot(sequences, *, channels_first=False, dtype=np.float32, alphabet="ACGT"):
    """Return [N, L, 4] (or [N, 4, L]) for equal-length strings.

    Accept a string, bytes, list or one-dimensional NumPy string array.
    Lowercase is accepted, U maps to T, N maps to zeros; other symbols and
    unequal lengths raise ValueError. A single string retains the batch axis.
    ``alphabet`` specifies channel order, e.g. ACGT or AGCT.
    """
    alphabet = str(alphabet).upper().replace("U", "T")
    if len(alphabet) != 4 or set(alphabet) != set("ACGT"):
        raise ValueError("alphabet must be a permutation of ACGT (or ACGU)")
    dtype = np.dtype(dtype)
    if dtype.kind not in "buif":
        raise ValueError("onehot dtype must be boolean, integer, or floating point")
    values = np.asarray(sequences)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError(
            "onehot expects a string or a one-dimensional sequence collection"
        )
    if not len(values):
        result = np.empty((0, 0, 4), dtype=dtype)
    else:
        if values.dtype.kind == "O":
            if not all(isinstance(value, (str, bytes)) for value in values):
                raise ValueError("onehot expects strings")
            values = np.asarray(values.tolist())
        if values.dtype.kind not in "US":
            raise ValueError("onehot expects strings")
        lengths = np.char.str_len(values)
        length = int(lengths[0])
        if np.any(lengths != length):
            raise ValueError(
                "onehot requires equal-length sequences; pad explicitly first"
            )
        try:
            packed = np.ascontiguousarray(values, dtype=f"S{max(length, 1)}")
        except UnicodeEncodeError as exc:
            raise ValueError("onehot accepts ASCII nucleotide symbols only") from exc
        codes = packed.view(np.uint8).reshape(len(values), max(length, 1))[:, :length]
        table, valid = _tables(alphabet, dtype.str)
        if not valid[codes].all():
            raise ValueError("onehot sequences must contain only A/C/G/T/U/N")
        result = table[codes]
    if channels_first:
        result = np.ascontiguousarray(result.transpose(0, 2, 1))
    return result
