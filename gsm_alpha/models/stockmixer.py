"""StockMixer, the architecture GSM-Alpha borrows from, as an ablation baseline.

    Jinyong Fan, Yanyan Shen. "StockMixer: A Simple Yet Strong MLP-Based
    Architecture for Stock Price Forecasting." AAAI-24.

The report's GSM-Alpha network is StockMixer with its *time mixing* stage
replaced by a GSM log-signature: indicator mixing and stock mixing survive, the
multi-scale temporal encoder does not.  Reproducing StockMixer as written
therefore gives the baseline that isolates what the signature actually buys,
which is the point of arm A in the three-way ablation:

    A  StockMixer          multi-scale time mixing   daily OHLCV
    B  GSM-Alpha           log-signature            daily OHLCV
    C  GSM-Alpha           log-signature            daily OHLCV + 5-minute

A -> B is the signature's contribution; B -> C is the 5-minute data's.  For that
subtraction to mean anything the three arms share everything else: the same
weighted-correlation loss, the same neutralised 20-day label, the same rolling
schedule and the same 60-day daily window.  The last of those is already a
departure from the paper, which uses a 16-day lookback; matching GSM-Alpha's
window matters more here than matching the paper's, and `window_length` is a
config knob so both can be run.

ONE STRUCTURAL ADAPTATION was unavoidable.  Equation 8 mixes stocks with
``M1 : R^{m x N}`` and ``M2 : R^{N x m}`` — learnable weights whose shape
contains ``N``, the number of stocks.  That is fine on the paper's benchmarks,
where ``N`` is a constant (1026 NASDAQ names), and impossible on a full A-share
cross section, which grows from 2 813 names in 2016 to 5 353 in 2024.  This
implementation keeps the stock -> market -> stock decomposition and its ``m``
latent market states but forms them by *attention pooling*, so the parameters no
longer carry ``N``.  It is the same inductive bias — a small number of learned
market states mediating every stock interaction, rather than a fully connected
stock graph — reached with an N-agnostic parameterisation.  The report appears
to have hit the same wall and substituted plain multi-head self-attention, which
is what :class:`gsm_alpha.models.gsm_alpha.StockMixing` implements.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class IndicatorMixing(nn.Module):
    """Equation 3: a residual MLP over the *indicator* axis.

    The stock's ``(T, F)`` window is transposed so the mixing runs across the F
    raw indicators at each time step — the open/high/low/close/volume of one day
    inform each other before any temporal reasoning happens.

    Args:
        n_indicators: ``F``.
        hidden: Width of the mixing MLP.
    """

    def __init__(self, n_indicators: int, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_indicators)
        self.fc1 = nn.Linear(n_indicators, hidden)
        self.fc2 = nn.Linear(hidden, n_indicators)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., T, F) -> (..., T, F)``."""
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class CausalTimeMixing(nn.Module):
    """Equation 4: a residual MLP over the *time* axis, masked to be causal.

    The paper's one structural departure from standard MLP-mixing.  A dense
    time-mixing layer lets step ``t`` read step ``t + 5``, which in a forecasting
    model is leakage dressed as capacity.  Masking both weight matrices to one
    triangle means information only ever flows forward in time.

    The mask multiplies the weight rather than the activation, so the masked
    entries receive no gradient and stay exactly zero.

    The paper describes the mask as *upper* triangular, which is its own index
    convention. Under the layout used here — mix along the transposed ``(F, T)``
    view — the triangle that yields "step ``t`` sees only ``t' <= t``" is the
    LOWER one: hidden unit ``j`` reads steps ``t <= j``, and output step ``t``
    reads hidden units ``j <= t``. Taking the paper's word literally produces the
    exact reverse, an anti-causal block in which step 0 reads the whole future;
    ``test_time_mixing_never_lets_the_future_reach_the_past`` pins the direction.

    The normalisation runs over the INDICATOR axis, per stock and per time step,
    which is the MLP-Mixer convention for a token-mixing block and is also what
    keeps the block causal: a LayerNorm over the time axis would pool the mean
    and variance of all ``T`` steps into every one of them, leaking the future
    around the mask before it ever applies.

    Args:
        n_steps: ``T``, the window length at this scale.
        hidden: ``H_t``; the paper sets it to ``T``.
        n_indicators: ``F``, the axis the normalisation runs over.
    """

    def __init__(self, n_steps: int, hidden: int, n_indicators: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_indicators)
        self.weight1 = nn.Parameter(torch.empty(hidden, n_steps))
        self.weight2 = nn.Parameter(torch.empty(n_steps, hidden))
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)
        self.act = nn.GELU()
        self.register_buffer("mask1", torch.tril(torch.ones(hidden, n_steps)), persistent=False)
        self.register_buffer("mask2", torch.tril(torch.ones(n_steps, hidden)), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., T, F) -> (..., T, F)``."""
        # Normalise over F first, then mix along T on the transposed view.
        h = self.norm(x).transpose(-1, -2)
        h = self.act(torch.matmul(h, (self.weight1 * self.mask1).t()))
        h = torch.matmul(h, (self.weight2 * self.mask2).t())
        return x + h.transpose(-1, -2)


class MultiScaleTemporalEncoder(nn.Module):
    """Equations 5-7: average-pool to several scales, mix each, concatenate.

    Scale ``k`` compresses the window to ``T / k`` steps before mixing, so the
    model sees the same history as a fine sequence and as one or more coarser
    tendency sequences.  The paper's ablation puts time mixing ahead of stock and
    indicator mixing in importance, and this multi-scale form is what the
    signature displaces in GSM-Alpha.

    Args:
        n_steps: ``T``.
        n_indicators: ``F``.
        scales: Pooling kernels ``k``; the paper uses ``(1, 2, 4)``.
        hidden: Indicator-mixing width.
        out_dim: Width of the fused representation.
    """

    def __init__(
        self,
        n_steps: int,
        n_indicators: int,
        scales: List[int],
        hidden: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.scales = list(scales)
        self.indicator = nn.ModuleList()
        self.time = nn.ModuleList()
        total = 0
        for k in self.scales:
            steps = n_steps // k
            if steps < 2:
                raise ValueError(f"scale {k} leaves {steps} steps of a {n_steps}-step window")
            self.indicator.append(IndicatorMixing(n_indicators, hidden))
            self.time.append(CausalTimeMixing(steps, steps, n_indicators))
            total += steps * n_indicators
        self.project = nn.Linear(total, out_dim)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """``(n_stocks, T, F) -> (n_stocks, out_dim)``."""
        parts = []
        for k, indicator, time in zip(self.scales, self.indicator, self.time):
            x = windows
            if k > 1:
                # Pool along time; avg_pool1d wants the axis last.
                x = torch.nn.functional.avg_pool1d(x.transpose(1, 2), kernel_size=k).transpose(1, 2)
            x = time(indicator(x))
            parts.append(x.flatten(start_dim=1))
        return self.project(torch.cat(parts, dim=-1))


class MarketStateStockMixing(nn.Module):
    """Equation 8, reparameterised so it does not depend on the stock count.

    The paper compresses ``N`` stocks into ``m`` market states with a learnable
    ``m x N`` matrix and expands back with an ``N x m`` one.  Both shapes contain
    ``N``, which a full A-share cross section does not hold fixed.  Here the
    ``m`` states are formed as attention-weighted averages over whatever stocks
    are present on the date, and each stock reads them back through a learned
    query — stock -> market -> stock, with ``m`` still the bottleneck that stops
    this collapsing into a fully connected stock graph.

    Args:
        dim: Width of a stock's representation.
        n_market_states: ``m``.
    """

    def __init__(self, dim: int, n_market_states: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_market = nn.Linear(dim, n_market_states)   # stock -> state affinities
        self.value = nn.Linear(dim, dim)
        self.from_market = nn.Linear(dim, n_market_states)  # stock -> state queries
        self.project = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``(n_stocks, dim) -> (n_stocks, dim)``."""
        h = self.norm(features)
        # Softmax over stocks: each market state is a weighted average of the
        # cross section, so its scale does not drift with how many names listed.
        affinity = torch.softmax(self.to_market(h), dim=0)          # (N, m)
        states = torch.matmul(affinity.t(), self.value(h))          # (m, dim)
        weights = torch.softmax(self.from_market(h), dim=-1)        # (N, m)
        return features + self.project(self.act(torch.matmul(weights, states)))


class StockMixerBackbone(nn.Module):
    """The full network: multi-scale temporal encoding, then market-aware mixing.

    Args:
        n_steps: Window length ``T``.
        n_indicators: ``F``.
        scales: Pooling kernels.
        hidden: Indicator-mixing width.
        embed_dim: Fused temporal width.
        n_market_states: ``m``.
    """

    def __init__(
        self,
        n_steps: int,
        n_indicators: int,
        scales: List[int],
        hidden: int,
        embed_dim: int,
        n_market_states: int,
    ) -> None:
        super().__init__()
        self.temporal = MultiScaleTemporalEncoder(
            n_steps, n_indicators, scales, hidden, embed_dim
        )
        self.stock = MarketStateStockMixing(embed_dim, n_market_states)
        # The paper concatenates a stock's own representation with its
        # market-influenced one before the final projection.
        self.head = nn.Linear(embed_dim * 2, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """``(n_stocks, T, F) -> (n_stocks,)``."""
        own = self.temporal(windows)
        influenced = self.stock(own)
        return self.head(torch.cat([own, influenced], dim=-1)).squeeze(-1)
