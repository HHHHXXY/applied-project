"""The StockMixer ablation baseline (AAAI-24), including its causal guarantee."""

from __future__ import annotations

import torch

from gsm_alpha.models.stockmixer import (
    CausalTimeMixing,
    IndicatorMixing,
    MarketStateStockMixing,
    MultiScaleTemporalEncoder,
    StockMixerBackbone,
)


def test_indicator_mixing_keeps_shape_and_is_per_timestep():
    block = IndicatorMixing(n_indicators=5, hidden=16)
    x = torch.randn(7, 60, 5)
    assert block(x).shape == x.shape


def test_time_mixing_never_lets_the_future_reach_the_past():
    """The upper-triangular mask is the paper's anti-leakage device; verify it."""
    torch.manual_seed(0)
    block = CausalTimeMixing(n_steps=12, hidden=12, n_indicators=3).eval()
    x = torch.randn(1, 12, 3, requires_grad=True)
    out = block(x)
    # Perturbing the LAST step must not move any earlier step's output.
    grads = torch.autograd.grad(out[0, 0].sum(), x, retain_graph=True)[0]
    assert grads[0, 11].abs().max() == 0.0, "step 0 read from step 11"
    # And the reverse direction must be live, or the block learns nothing.
    grads_last = torch.autograd.grad(out[0, 11].sum(), x)[0]
    assert grads_last[0, 0].abs().max() > 0.0, "step 11 cannot see step 0"


def test_masked_weights_receive_no_gradient():
    block = CausalTimeMixing(n_steps=8, hidden=8, n_indicators=5)
    block(torch.randn(4, 8, 5)).sum().backward()
    future = torch.triu(torch.ones(8, 8), diagonal=1).bool()
    assert block.weight1.grad[future].abs().max() == 0.0
    assert block.weight2.grad[future].abs().max() == 0.0


def test_multiscale_encoder_pools_to_every_scale():
    enc = MultiScaleTemporalEncoder(
        n_steps=60, n_indicators=5, scales=[1, 2, 4], hidden=16, out_dim=32
    )
    out = enc(torch.randn(11, 60, 5))
    assert out.shape == (11, 32)


def test_stock_mixing_handles_a_universe_that_changes_size():
    """The paper's m x N weights cannot; this reparameterisation must."""
    block = MarketStateStockMixing(dim=32, n_market_states=8).eval()
    for n_stocks in (2813, 4069, 5353):
        out = block(torch.randn(n_stocks, 32))
        assert out.shape == (n_stocks, 32)
        assert torch.isfinite(out).all()


def test_stock_mixing_actually_couples_stocks():
    """If it did not, it would be a per-stock MLP and the arm would be mislabelled."""
    torch.manual_seed(1)
    block = MarketStateStockMixing(dim=16, n_market_states=4).eval()
    base = torch.randn(50, 16)
    moved = base.clone()
    # Not a constant offset: the block LayerNorms each stock across its features,
    # so adding the same number to every feature is erased before any mixing and
    # would make this test vacuously pass on a per-stock MLP too.
    torch.manual_seed(2)
    moved[0] = moved[0] * 3.0 + torch.randn(16)   # disturb one stock only
    delta = (block(moved) - block(base))[1:].abs().max()
    assert float(delta) > 1e-6, "one stock's change left every peer's output untouched"


def test_backbone_emits_one_value_per_stock():
    net = StockMixerBackbone(
        n_steps=60, n_indicators=5, scales=[1, 2, 4],
        hidden=16, embed_dim=32, n_market_states=8,
    )
    out = net(torch.randn(37, 60, 5))
    assert out.shape == (37,)
    assert torch.isfinite(out).all()
