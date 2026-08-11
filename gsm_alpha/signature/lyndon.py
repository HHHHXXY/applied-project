"""Lyndon words and their positions inside the truncated tensor algebra.

The log-signature of a ``d``-channel path truncated at depth ``N`` lives in the
free Lie algebra, whose dimension is the number of Lyndon words of length
``<= N`` over an alphabet of ``d`` letters.  ``signatory`` exposes two ways of
coordinatising it: ``mode="brackets"`` (coefficients in the Lyndon *bracket*
basis) and ``mode="words"`` (the entries of the log tensor read off at the
positions indexed by Lyndon *words*).  ``mode="words"`` is a plain gather from
the log tensor, which is what makes the pure-PyTorch fallback in
:mod:`gsm_alpha.signature.torch_backend` possible.

This module only needs numpy, so the ordering used by the fallback can be
checked against ``signatory.lyndon_words`` in the tests.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np


def duval(d: int, max_length: int) -> List[Tuple[int, ...]]:
    """Generate every Lyndon word of length ``<= max_length`` over ``d`` letters.

    Duval's algorithm, which emits the words in lexicographic order.

    Args:
        d: Alphabet size (the number of path channels).
        max_length: Maximum word length (the signature truncation depth).

    Returns:
        Lyndon words as tuples of letters in ``range(d)``, lexicographic order.
    """
    if d < 1:
        raise ValueError(f"alphabet size must be >= 1, got {d}")
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1, got {max_length}")

    words: List[Tuple[int, ...]] = []
    w = [-1]
    while w:
        w[-1] += 1
        m = len(w)
        words.append(tuple(w))
        # Extend to length max_length by repeating the current word periodically.
        while len(w) < max_length:
            w.append(w[-m])
        # Strip the trailing letters that are already maximal.
        while w and w[-1] == d - 1:
            w.pop()
    return words


@lru_cache(maxsize=None)
def lyndon_words(d: int, depth: int) -> Tuple[Tuple[int, ...], ...]:
    """Lyndon words ordered the way ``signatory`` orders log-signature channels.

    ``signatory`` groups the channels by word length ascending and sorts
    lexicographically within each length; :func:`duval` emits a single global
    lexicographic order, so the words are regrouped here.

    Args:
        d: Alphabet size (the number of path channels).
        depth: Signature truncation depth.

    Returns:
        Lyndon words in signatory's channel order.
    """
    words = duval(d, depth)
    words.sort(key=lambda w: (len(w), w))
    return tuple(words)


@lru_cache(maxsize=None)
def logsignature_channels(d: int, depth: int) -> int:
    """Number of log-signature channels for ``d`` channels truncated at ``depth``."""
    return len(lyndon_words(d, depth))


@lru_cache(maxsize=None)
def signature_channels(d: int, depth: int) -> int:
    """Number of signature channels: ``d + d^2 + ... + d^depth``."""
    return sum(d**k for k in range(1, depth + 1))


@lru_cache(maxsize=None)
def word_gather_index(d: int, depth: int) -> Tuple[np.ndarray, ...]:
    """Positions of the Lyndon words within each level of the tensor algebra.

    Level ``k`` of the tensor algebra is stored flat with ``d**k`` entries in
    row-major order, so the word ``(i_1, ..., i_k)`` sits at
    ``sum_j i_j * d**(k - 1 - j)``.

    Args:
        d: Alphabet size.
        depth: Signature truncation depth.

    Returns:
        One index array per level ``k = 1 .. depth``; concatenating the values
        gathered with them reproduces signatory's ``mode="words"`` layout,
        because the words are already grouped by length.
    """
    per_level: List[List[int]] = [[] for _ in range(depth)]
    for word in lyndon_words(d, depth):
        flat = 0
        for letter in word:
            flat = flat * d + letter
        per_level[len(word) - 1].append(flat)
    return tuple(np.asarray(idx, dtype=np.int64) for idx in per_level)
