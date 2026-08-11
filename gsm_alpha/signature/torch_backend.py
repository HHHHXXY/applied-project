"""Pure-PyTorch signature and log-signature transforms.

``signatory`` is the fast path and the reference implementation, but it only
builds against PyTorch 1.x.  This module reimplements the same two transforms
with nothing but core PyTorch ops so the project still runs on a machine where
signatory could not be built (including PyTorch 2.x).  It is differentiable and
device-agnostic, just slower.

``tests/test_signature_backend.py`` pins the two implementations against each
other to float64 tolerance, so the fallback is a checked substitute rather than
a hopeful one.

Conventions match signatory:

* A path is a tensor of shape ``(batch, length, channels)``; its first point is
  the starting point (no basepoint is inserted for you).
* Level ``k`` of the signature is flattened row-major, so the multi-index
  ``(i_1, ..., i_k)`` sits at ``sum_j i_j * d**(k - 1 - j)`` and ``i_1`` is the
  *earliest* integration variable.
* ``logsignature(..., mode="words")`` reads the log tensor off at the positions
  indexed by Lyndon words (see :mod:`gsm_alpha.signature.lyndon`).
"""

from __future__ import annotations

from typing import List, Sequence

import torch

from .lyndon import logsignature_channels, signature_channels, word_gather_index

__all__ = [
    "signature",
    "logsignature",
    "logsignature_channels",
    "signature_channels",
]


def _segment_signature(increments: torch.Tensor, depth: int) -> List[torch.Tensor]:
    """Exact signature of straight-line segments: level ``k`` is ``dx^{ok} / k!``.

    Args:
        increments: ``(..., d)`` path increments, one per segment.
        depth: Truncation depth.

    Returns:
        ``depth`` tensors, level ``k`` having shape ``(..., d**k)``.
    """
    lead = increments.shape[:-1]
    d = increments.shape[-1]
    terms = [increments]
    current = increments
    for k in range(2, depth + 1):
        # Divide in place so the running product accumulates 1/k! and not 1/k.
        current = (current.unsqueeze(-1) * increments.unsqueeze(-2)).reshape(*lead, d**k) / k
        terms.append(current)
    return terms


def _chen(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Chen's identity: the signature of ``a`` followed by ``b``.

    Both arguments carry an implicit level-0 term equal to 1, so level ``k`` of
    the product is ``a_k + b_k + sum_{i=1}^{k-1} a_i (x) b_{k-i}``.

    Args:
        a: Levels ``1..depth`` of the first path's signature, each ``(..., d**k)``.
        b: Levels ``1..depth`` of the second path's signature.

    Returns:
        Levels ``1..depth`` of the concatenated path's signature.
    """
    depth = len(a)
    out: List[torch.Tensor] = []
    for k in range(1, depth + 1):
        acc = a[k - 1] + b[k - 1]
        for i in range(1, k):
            left, right = a[i - 1], b[k - i - 1]
            acc = acc + (left.unsqueeze(-1) * right.unsqueeze(-2)).reshape(acc.shape)
        out.append(acc)
    return out


def _reduce_chen(terms: List[torch.Tensor], seq_dim: int) -> List[torch.Tensor]:
    """Combine per-segment signatures along ``seq_dim`` by divide and conquer.

    The Chen product is associative, so folding pairwise in ``log2(n_segments)``
    rounds costs far fewer PyTorch calls than a sequential scan.

    Args:
        terms: Levels ``1..depth``, each shaped ``(..., n_segments, ..., d**k)``.
        seq_dim: Axis holding the segments.

    Returns:
        Levels ``1..depth`` with ``seq_dim`` removed.
    """
    n = terms[0].shape[seq_dim]
    while n > 1:
        pairs = n // 2
        # Pair *adjacent* segments (0,1), (2,3), ... — Chen's identity only
        # concatenates paths that meet end to end.
        even = [t.narrow(seq_dim, 0, 2 * pairs).unfold(seq_dim, 2, 2) for t in terms]
        left = [t.select(-1, 0) for t in even]
        right = [t.select(-1, 1) for t in even]
        merged = _chen(left, right)
        if n % 2:  # odd tail segment: carry it through untouched
            tail = [t.narrow(seq_dim, n - 1, 1) for t in terms]
            merged = [torch.cat([m, t], dim=seq_dim) for m, t in zip(merged, tail)]
        terms = merged
        n = terms[0].shape[seq_dim]
    return [t.squeeze(seq_dim) for t in terms]


def signature_terms(path: torch.Tensor, depth: int) -> List[torch.Tensor]:
    """Signature of ``path`` as one tensor per level.

    Args:
        path: ``(batch, length, channels)``, ``length >= 2``.
        depth: Truncation depth, ``>= 1``.

    Returns:
        ``depth`` tensors, level ``k`` shaped ``(batch, channels**k)``.
    """
    if path.dim() != 3:
        raise ValueError(f"path must be (batch, length, channels), got {tuple(path.shape)}")
    if path.shape[1] < 2:
        raise ValueError(f"path needs at least 2 points, got {path.shape[1]}")
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    increments = path[:, 1:] - path[:, :-1]  # (batch, n_seg, d)
    return _reduce_chen(_segment_signature(increments, depth), seq_dim=1)


def signature(path: torch.Tensor, depth: int) -> torch.Tensor:
    """Truncated signature, flattened level by level.

    Args:
        path: ``(batch, length, channels)``.
        depth: Truncation depth.

    Returns:
        ``(batch, signature_channels(channels, depth))``.
    """
    return torch.cat(signature_terms(path, depth), dim=-1)


def _log_terms(sig: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Tensor-algebra logarithm of a signature (a group-like element ``1 + x``).

    ``log(1 + x) = sum_{m>=1} (-1)^{m+1} x^m / m``, truncated at ``depth``.  ``x``
    has no level-0 part, so ``x^m`` vanishes below level ``m`` and the series
    terminates.

    Args:
        sig: Levels ``1..depth`` of the signature.

    Returns:
        Levels ``1..depth`` of the log-signature tensor.
    """
    depth = len(sig)
    out = [term.clone() for term in sig]  # the m = 1 contribution
    power = list(sig)  # x^m, entries below level m are structurally zero
    for m in range(2, depth + 1):
        # x^m = x^{m-1} (x) x, so level k sums over splits i in [m-1, k-1].
        next_power: List[torch.Tensor] = []
        for k in range(1, depth + 1):
            if k < m:
                next_power.append(None)  # type: ignore[arg-type]
                continue
            acc = None
            for i in range(m - 1, k):
                left, right = power[i - 1], sig[k - i - 1]
                piece = (left.unsqueeze(-1) * right.unsqueeze(-2)).reshape(
                    left.shape[0], left.shape[-1] * right.shape[-1]
                )
                acc = piece if acc is None else acc + piece
            next_power.append(acc)
        power = next_power
        coeff = (1.0 if m % 2 else -1.0) / m
        for k in range(m, depth + 1):
            out[k - 1] = out[k - 1] + coeff * power[k - 1]
    return out


def logsignature(path: torch.Tensor, depth: int, mode: str = "words") -> torch.Tensor:
    """Truncated log-signature in the Lyndon-word coordinates.

    Args:
        path: ``(batch, length, channels)``.
        depth: Truncation depth.
        mode: Only ``"words"`` is supported — the same coordinatisation as
            ``signatory.logsignature(..., mode="words")``.  ``"brackets"`` would
            need the Lyndon bracket change of basis, which nothing here uses.

    Returns:
        ``(batch, logsignature_channels(channels, depth))``.
    """
    if mode != "words":
        raise NotImplementedError(
            f"the pure-torch backend only implements mode='words', got {mode!r}"
        )
    d = path.shape[-1]
    log_levels = _log_terms(signature_terms(path, depth))
    gathers = word_gather_index(d, depth)
    picked = [
        level.index_select(-1, torch.as_tensor(idx, device=level.device))
        for level, idx in zip(log_levels, gathers)
        if len(idx)
    ]
    return torch.cat(picked, dim=-1)
