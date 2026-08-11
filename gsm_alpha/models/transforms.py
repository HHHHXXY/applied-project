"""Input transforms applied to cached features on the way into the network.

The report specifies no rescaling of the signature terms (its section 2.5
``rho`` stage, set to ``none`` because its table 7 finds that best), and the
GSM-Alpha network's first operation is therefore a ``Linear`` over raw
log-signature channels.  On this data lake that does not train: a depth-5
log-signature over a 960-step path is extraordinarily heavy tailed — measured
over the cached panel the largest absolute value of a branch is ~4 900 while
its 99th percentile is ~10 — and both fp32 and fp16 runs drove the validation
rank IC to a non-finite value inside the first epoch, which is what Lightning's
``EarlyStopping(check_finite=True)`` then terminated on.

``gaussian_rank`` is the documented deviation that makes it trainable.
"""

from __future__ import annotations

import math

import torch


def gaussian_rank(features: torch.Tensor) -> torch.Tensor:
    """Cross-sectional normal-score transform, one feature column at a time.

    Ranks each column across the day's stocks and maps rank ``i`` of ``n`` to
    ``Phi^-1((i + 0.5) / n)``, so every column arrives at the network as a
    standard normal no matter what scale or tail it had.  Three properties
    matter here:

    * It is computed **within one date's cross section**, which is exactly the
      information available when the factor is evaluated, so it introduces no
      lookahead and needs no training-set statistics.
    * Being a rank transform it is completely insensitive to outliers, which is
      the failure this exists to fix.
    * It discards magnitude and keeps only order. That is a real loss of
      information and a deliberate deviation from the report — the model can no
      longer tell a mild outlier from an extreme one.

    ``Phi^-1`` is written through ``erfinv`` rather than ``torch.special.ndtri``
    so the same code runs on the torch 1.9 stack that signatory pins.

    Gradients do not flow through the ranking, which is piecewise constant. That
    is harmless in ``features`` mode, where the transform sits on cached input
    with nothing learnable upstream, but it would cut the gradient path to a
    learnable augmentation in ``windows`` mode.

    Args:
        features: ``(n_stocks, n_features)`` for one date. Non-finite entries
            are excluded from the ranking and returned as ``NaN``.

    Returns:
        ``(n_stocks, n_features)`` standard-normal scores, same dtype and device.
    """
    if features.ndim != 2:
        raise ValueError(f"expected (n_stocks, n_features), got {tuple(features.shape)}")
    n_stocks = features.shape[0]
    if n_stocks < 2:
        return torch.zeros_like(features)

    finite = torch.isfinite(features)
    # Sort non-finite entries to the end so they never displace a real rank.
    filled = torch.where(finite, features, torch.full_like(features, float("inf")))
    order = filled.argsort(dim=0)
    positions = torch.arange(n_stocks, device=features.device).unsqueeze(1).expand_as(order)
    ranks = torch.empty_like(order)
    ranks.scatter_(0, order, positions)

    # Each column has its own valid count, so a column with missing names still
    # spans the full normal range rather than being squeezed toward zero.
    counts = finite.sum(dim=0, keepdim=True).clamp_min(1).to(features.dtype)
    quantile = (ranks.to(features.dtype) + 0.5) / counts
    scores = torch.erfinv((2.0 * quantile - 1.0).clamp(-0.999999, 0.999999)) * math.sqrt(2.0)
    return torch.where(finite, scores, torch.full_like(scores, float("nan")))
