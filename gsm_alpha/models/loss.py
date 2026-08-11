"""The weighted-correlation objective of report section 3.2.

    "损失函数选择为负向的加权相关系数，目的是降低因子的空头效应 ... 权重由生成的
     因子值决定，因子值越大，计算时的权重就越大。这里选择了指数衰减加权，半衰期为
     batch size 的一半。"

A plain cross-sectional correlation spends the same effort fitting the bottom of
the book as the top, even though only the top is investable in a long-only or
constrained portfolio.  Weighting the correlation by rank of the *predicted*
factor concentrates the fit where the strategy will actually take positions, and
is what the report means by reducing the "空头效应" (the tendency of the factor's
performance to come from the short leg).

The weights are a function of the prediction's *ordering*, which is piecewise
constant, so they carry no useful gradient and are detached: the model learns by
moving predictions, not by gaming the weights.
"""

from __future__ import annotations

import torch


def _at_least_float32(tensor: torch.Tensor) -> torch.Tensor:
    """Raise a low-precision tensor to float32, keeping the gradient path intact.

    Nothing in this module runs below float32. Two reasons, both of which bite
    under mixed precision:

    * The cross-sectional statistics accumulate thousands of terms — at full
      market a batch is ~4 300 stocks — and a float16 sum of that many small
      products loses the tail entirely, so the correlation would be quietly wrong.
    * ``torch.arange`` has no half implementation on CPU, so the rank weights
      would not even build.

    Autocast only overrides specific ops (matmul and friends), and the reductions
    used below are not among them, so a float32 input keeps the whole computation
    in float32 without needing to disable autocast explicitly.

    float64 passes through untouched. Demoting it would throw away precision the
    caller explicitly asked for, and it is what made the flat-weight Pearson
    equivalence test machine-dependent: the same assertion landed at 9.9e-9 on one
    CPU and 1.9e-8 on another, against a 1e-8 tolerance.
    """
    return tensor if tensor.dtype in (torch.float32, torch.float64) else tensor.float()


def exponential_rank_weights(
    scores: torch.Tensor,
    halflife_fraction: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Exponentially decaying weights ordered by descending score.

    Args:
        scores: ``(n,)`` predicted factor values.
        halflife_fraction: Half-life as a fraction of ``n``; the report uses
            one half, so the weight halves every ``n / 2`` rank positions.
        eps: Normalisation floor.

    Returns:
        ``(n,)`` non-negative weights summing to 1, detached from the graph.
    """
    scores = _at_least_float32(scores)
    n = scores.shape[0]
    halflife = max(halflife_fraction * n, 1.0)
    order = torch.argsort(scores.detach(), descending=True)
    rank = torch.empty(n, dtype=scores.dtype, device=scores.device)
    rank[order] = torch.arange(n, dtype=scores.dtype, device=scores.device)
    weights = torch.pow(torch.tensor(0.5, dtype=scores.dtype, device=scores.device), rank / halflife)
    return weights / weights.sum().clamp_min(eps)


def weighted_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weighted Pearson correlation, the ``rho^w`` of report section 3.2.

    Args:
        x: ``(n,)`` predictions.
        y: ``(n,)`` labels.
        weights: ``(n,)`` non-negative weights summing to 1.
        eps: Variance floor.

    Returns:
        Scalar correlation in ``[-1, 1]``.
    """
    x, y, weights = _at_least_float32(x), _at_least_float32(y), _at_least_float32(weights)
    mean_x = (weights * x).sum()
    mean_y = (weights * y).sum()
    cov = (weights * x * y).sum() - mean_x * mean_y
    var_x = (weights * x * x).sum() - mean_x * mean_x
    var_y = (weights * y * y).sum() - mean_y * mean_y
    return cov / torch.sqrt(var_x.clamp_min(eps) * var_y.clamp_min(eps))


def weighted_correlation_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    halflife_fraction: float = 0.5,
) -> torch.Tensor:
    """Negative weighted correlation between predictions and labels.

    Args:
        predictions: ``(n,)`` factor values for one trading day's cross section.
        labels: ``(n,)`` neutralised, standardised forward returns.
        halflife_fraction: See :func:`exponential_rank_weights`.

    Returns:
        Scalar loss; ``0`` when the cross section is too small to correlate.
    """
    if predictions.shape != labels.shape:
        raise ValueError(f"shape mismatch: {tuple(predictions.shape)} vs {tuple(labels.shape)}")
    predictions, labels = _at_least_float32(predictions), _at_least_float32(labels)
    finite = torch.isfinite(predictions) & torch.isfinite(labels)
    if int(finite.sum()) < 3:
        # Sum the finite subset, not the whole tensor. A degenerate batch is
        # usually degenerate *because* the predictions went non-finite, and
        # ``predictions.sum() * 0.0`` is NaN — not zero — the moment one NaN is
        # present, which turns a single bad batch into permanently NaN weights.
        # Summing an empty selection still returns a zero that carries a
        # gradient, so the graph stays connected either way.
        return predictions[finite].sum() * 0.0
    x, y = predictions[finite], labels[finite]
    weights = exponential_rank_weights(x, halflife_fraction)
    return -weighted_correlation(x, y, weights)


def rank_ic(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Spearman rank correlation of one cross section, the report's headline metric."""
    predictions, labels = _at_least_float32(predictions), _at_least_float32(labels)
    finite = torch.isfinite(predictions) & torch.isfinite(labels)
    if int(finite.sum()) < 3:
        return torch.tensor(float("nan"), device=predictions.device)
    x, y = _to_ranks(predictions[finite]), _to_ranks(labels[finite])
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum()).clamp_min(1e-12)
    return (x * y).sum() / denom


def _to_ranks(values: torch.Tensor) -> torch.Tensor:
    """Ordinal ranks, ties broken by position (adequate for continuous scores)."""
    values = _at_least_float32(values)
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(values.shape[0], dtype=values.dtype, device=values.device)
    return ranks
