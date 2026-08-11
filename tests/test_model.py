"""The GSM-Alpha network, its objective, and the config loader."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gsm_alpha.config import Config
from gsm_alpha.models.gsm_alpha import GSMAlpha, IndicatorMixing, StockMixing
from gsm_alpha.models.transforms import gaussian_rank
from gsm_alpha.models.loss import (
    exponential_rank_weights,
    rank_ic,
    weighted_correlation,
    weighted_correlation_loss,
)


# -- loss -----------------------------------------------------------------


def test_rank_weights_favour_the_top_of_the_book():
    scores = torch.tensor([0.1, 0.9, 0.5, -0.3])
    weights = exponential_rank_weights(scores, 0.5)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    # Highest score gets the largest weight, lowest the smallest.
    assert weights.argmax().item() == 1
    assert weights.argmin().item() == 3


def test_rank_weights_halve_over_the_half_life():
    scores = torch.arange(100, dtype=torch.float32).flip(0)  # already descending
    weights = exponential_rank_weights(scores, 0.5)  # half-life = 50 positions
    assert torch.isclose(weights[50] / weights[0], torch.tensor(0.5), atol=1e-5)


def test_weighted_correlation_reduces_to_pearson_with_flat_weights():
    torch.manual_seed(0)
    x, y = torch.randn(50, dtype=torch.float64), torch.randn(50, dtype=torch.float64)
    flat = torch.full((50,), 1 / 50, dtype=torch.float64)
    expected = float(np.corrcoef(x.numpy(), y.numpy())[0, 1])
    assert abs(float(weighted_correlation(x, y, flat)) - expected) < 1e-8


def test_loss_is_minimised_by_a_perfect_prediction():
    torch.manual_seed(1)
    labels = torch.randn(200)
    assert weighted_correlation_loss(labels, labels) < -0.99
    assert weighted_correlation_loss(-labels, labels) > 0.99
    assert abs(float(weighted_correlation_loss(torch.randn(200), labels))) < 0.4


def test_loss_weights_carry_no_gradient():
    """The weights come from a ranking, so the model must not be able to game them."""
    predictions = torch.randn(64, requires_grad=True)
    weighted_correlation_loss(predictions, torch.randn(64)).backward()
    assert predictions.grad is not None and torch.isfinite(predictions.grad).all()


def test_loss_ignores_non_finite_entries():
    predictions = torch.randn(30)
    labels = torch.randn(30)
    labels[:5] = float("nan")
    assert torch.isfinite(weighted_correlation_loss(predictions, labels))


def test_loss_degrades_gracefully_on_a_tiny_cross_section():
    loss = weighted_correlation_loss(torch.randn(2, requires_grad=True), torch.randn(2))
    assert float(loss) == 0.0


def test_rank_ic_is_monotone_invariant():
    torch.manual_seed(2)
    x = torch.randn(100)
    y = torch.randn(100)
    assert torch.isclose(rank_ic(x, y), rank_ic(torch.exp(x), y), atol=1e-6)
    assert rank_ic(x, x) > 0.999


# -- modules --------------------------------------------------------------


def test_indicator_mixing_is_residual():
    block = IndicatorMixing(64, 32)
    out = block(torch.randn(7, 64))
    assert out.shape == (7, 32)


def test_stock_mixing_lets_a_stock_see_its_peers():
    torch.manual_seed(0)
    block = StockMixing(16, n_heads=4, dropout=0.0).eval()
    features = torch.randn(12, 16)
    baseline = block(features)

    perturbed = features.clone()
    perturbed[5] += 10.0
    after = block(perturbed)
    # Row 0 changed even though only row 5's input did: information crossed stocks.
    assert not torch.allclose(baseline[0], after[0], atol=1e-4)


def test_stock_mixing_handles_a_changing_number_of_stocks():
    block = StockMixing(16, n_heads=4, dropout=0.0).eval()
    assert block(torch.randn(9, 16)).shape == (9, 16)
    assert block(torch.randn(413, 16)).shape == (413, 16)


def test_stock_mixing_rejects_an_indivisible_width():
    with pytest.raises(ValueError, match="divisible"):
        StockMixing(18, n_heads=4)


# -- the whole network ----------------------------------------------------


def _config(**model_overrides) -> Config:
    raw = {
        "model": {"hidden_dim": 32, "n_attention_heads": 4, "attention_dropout": 0.0,
                  **model_overrides},
        "minute_gsm": {"depth": 3, "backend": "torch"},
        "daily_gsm": {"depth": 3, "backend": "torch"},
    }
    return Config.from_dict(raw)


def test_forward_from_cached_features():
    model = GSMAlpha(_config(), {"minute": 80, "daily": 40}).eval()
    batch = {"minute": torch.randn(25, 80), "daily": torch.randn(25, 40)}
    out = model(batch)
    assert out.shape == (25,)
    assert torch.isfinite(out).all()


def test_forward_from_raw_windows_runs_gsm_inline():
    config = _config()
    config.minute_gsm.augmentations = ["multi_headed_stream_preserving", "time", "basepoint"]
    config.minute_gsm.n_projections = 2
    config.minute_gsm.projection_out = 2
    from gsm_alpha.data.cache import build_gsm

    dims = {"minute": build_gsm(config.minute_gsm).out_features(15),
            "daily": build_gsm(config.daily_gsm).out_features(15)}
    model = GSMAlpha(config, dims, input_mode="windows")
    out = model({"minute": torch.randn(11, 15, 5), "daily": torch.randn(11, 15, 5)})
    assert out.shape == (11,)
    out.sum().backward()
    assert any(p.grad is not None for p in model.gsm["minute"].parameters())


def test_single_branch_model():
    model = GSMAlpha(_config(), {"daily": 40}).eval()
    assert model({"daily": torch.randn(8, 40)}).shape == (8,)


def test_disabling_stock_mixing_makes_stocks_independent():
    model = GSMAlpha(_config(use_stock_mixing=False), {"daily": 16}).eval()
    features = torch.randn(6, 16)
    baseline = model({"daily": features})
    perturbed = features.clone()
    perturbed[3] += 5.0
    after = model({"daily": perturbed})
    assert torch.allclose(baseline[0], after[0], atol=1e-6)
    assert not torch.allclose(baseline[3], after[3], atol=1e-6)


def test_training_step_lowers_the_loss_on_a_learnable_batch():
    torch.manual_seed(0)
    model = GSMAlpha(_config(), {"daily": 8})
    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
    features = torch.randn(128, 8)
    labels = features[:, 0] * 2 + features[:, 1]  # a signal the head can find
    labels = (labels - labels.mean()) / labels.std()

    first = float(weighted_correlation_loss(model({"daily": features}), labels))
    for _ in range(60):
        optimiser.zero_grad()
        loss = weighted_correlation_loss(model({"daily": features}), labels)
        loss.backward()
        optimiser.step()
    assert float(loss) < first - 0.1


# -- config ---------------------------------------------------------------


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown config key"):
        Config.from_dict({"model": {"hidden_dimm": 32}})


def test_config_overrides_use_dotted_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  max_epochs: 100\n", encoding="utf-8")
    config = Config.from_yaml(str(path), {"train.max_epochs": 3, "model.hidden_dim": 64})
    assert config.train.max_epochs == 3
    assert config.model.hidden_dim == 64


def test_shipped_default_config_loads():
    config = Config.from_yaml("configs/default.yaml")
    assert config.minute_gsm.depth == 5
    assert config.minute_gsm.augmentations == ["coordinate_projection", "time", "basepoint"]
    assert config.data.minute_lookback_days == 20
    assert config.data.daily_lookback_days == 60
    assert config.train.early_stopping_patience == 30
    assert config.train.max_epochs == 100


def test_shipped_paper_config_loads_and_matches_the_report():
    """configs/paper.yaml must encode the report's stated hyper-parameters."""
    config = Config.from_yaml("configs/paper.yaml")
    # Section 3.1: pairs projection + time + basepoint, depth-5 log-signature,
    # global window, no rescaling.
    assert config.minute_gsm.depth == 5
    assert config.minute_gsm.transform == "logsignature"
    assert config.minute_gsm.window == "global"
    assert config.minute_gsm.rescaling == "none"
    assert config.minute_gsm.projection_size == 2
    # Section 3.2: 20 days of 5-minute bars, 60 dailies, 20-day forward label.
    assert config.data.minute_lookback_days == 20
    assert config.data.daily_lookback_days == 60
    assert config.labels.horizon == 20
    # Section 3.2: 4-year lookback, last year validation, last month dropped,
    # early stopping 30, max 100 epochs, loss half-life = batch/2.
    assert config.train.lookback_years == 4
    assert config.train.validation_years == 1
    assert config.train.drop_last_month is True
    assert config.train.max_epochs == 100
    assert config.train.early_stopping_patience == 30
    assert config.model.loss_halflife_fraction == 0.5
    # Full market and every trading day, unlike the cheaper default config.
    assert config.data.universe_top_n is None
    assert config.data.train_day_stride == 1
    # The report's factor period ends 2024-05.
    assert config.data.end_date == "2024-05-31"
    # 2018 and 2019 cannot be fitted: five-minute history starts 2016.
    assert config.train.first_predict_year == 2020
    assert config.train.last_predict_year == 2024


# -- mixed precision and device readiness ----------------------------------


def test_loss_and_metrics_survive_half_precision_inputs():
    """Under autocast the model head emits half; the objective must still work.

    Two concrete failures this guards: torch.arange has no half kernel on CPU, so
    the rank weights would not build at all; and a half sum over a full-market
    cross section (~4300 names) loses the small terms, so the correlation would
    be quietly wrong rather than merely imprecise.
    """
    torch.manual_seed(0)
    n = 4300
    x = torch.randn(n)
    y = torch.randn(n) * 0.3 + x * 0.1

    assert abs(float(weighted_correlation_loss(x, y))
               - float(weighted_correlation_loss(x.half(), y.half()))) < 1e-4
    assert abs(float(rank_ic(x, y)) - float(rank_ic(x.half(), y.half()))) < 1e-4
    assert weighted_correlation_loss(x.half(), y.half()).dtype == torch.float32


def test_gradient_flows_back_into_a_half_prediction():
    predictions = torch.randn(256, dtype=torch.float16, requires_grad=True)
    weighted_correlation_loss(predictions, torch.randn(256)).backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_mixed_precision_is_downgraded_when_there_is_no_gpu():
    """configs/paper.yaml asks for precision 16; it must still run on a CPU box.

    Lightning 1.6 raises outright for precision=16 on CPU, and Lightning 2.x
    silently swaps in bfloat16 — neither is what the config meant.
    """
    from gsm_alpha.train import resolve_hardware

    config = Config.from_dict({"train": {"accelerator": "cpu", "precision": 16}})
    assert resolve_hardware(config)["precision"] == 32

    config = Config.from_dict({"train": {"accelerator": "cpu", "precision": 32}})
    assert resolve_hardware(config)["precision"] == 32

    # An explicit gpu request keeps mixed precision and is left to fail loudly
    # in Lightning if no card is present, rather than being silently downgraded.
    config = Config.from_dict({"train": {"accelerator": "gpu", "precision": 16}})
    assert resolve_hardware(config)["precision"] == 16


def test_batch_survives_lightnings_device_transfer():
    """The batch carries numpy dates and ids alongside tensors; both must pass."""
    try:
        from lightning_fabric.utilities.apply_func import move_data_to_device
    except ImportError:
        from pytorch_lightning.utilities.apply_func import move_data_to_device

    batch = {
        "date": np.datetime64("2021-03-01", "D"),
        "sid": np.arange(5, dtype=np.int64),
        "label": torch.randn(5),
        "daily": torch.randn(5, 8),
    }
    moved = move_data_to_device(batch, torch.device("cpu"))
    assert moved["date"] == batch["date"]
    assert np.array_equal(moved["sid"], batch["sid"])
    assert moved["daily"].shape == (5, 8)


def test_gaussian_rank_makes_every_column_standard_normal():
    """Whatever the scale or tail, each column leaves as a standard normal."""
    torch.manual_seed(3)
    n = 4000
    raw = torch.stack([
        torch.randn(n) * 1e-6,                    # microscopic
        torch.randn(n).abs() ** 5 * 1e3,          # the heavy tail this exists for
        torch.arange(n, dtype=torch.float32),     # perfectly uniform
    ], dim=1)
    out = gaussian_rank(raw)
    assert out.shape == raw.shape
    assert torch.isfinite(out).all()
    for k in range(raw.shape[1]):
        assert abs(float(out[:, k].mean())) < 1e-3
        assert abs(float(out[:, k].std()) - 1.0) < 0.02


def test_gaussian_rank_preserves_order_and_ignores_scale():
    torch.manual_seed(4)
    col = torch.randn(500, 1)
    a = gaussian_rank(col)
    b = gaussian_rank(col * 1e6)          # pure rescaling
    c = gaussian_rank(col.exp())          # any increasing map
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(a, c, atol=1e-6)
    assert (a.argsort(dim=0) == col.argsort(dim=0)).all()


def test_gaussian_rank_keeps_nan_out_of_the_ranking():
    values = torch.tensor([[1.0], [float("nan")], [3.0], [2.0]])
    out = gaussian_rank(values)
    assert torch.isnan(out[1, 0])
    finite = out[[0, 2, 3], 0]
    assert torch.isfinite(finite).all()
    # 1.0 < 2.0 < 3.0 must survive as an ordering over the three valid names.
    assert float(out[0, 0]) < float(out[3, 0]) < float(out[2, 0])


def test_degenerate_batch_loss_is_zero_not_nan():
    """A batch whose predictions went non-finite must not poison the weights."""
    predictions = torch.full((16,), float("nan"), requires_grad=True)
    labels = torch.randn(16)
    loss = weighted_correlation_loss(predictions, labels)
    assert float(loss) == 0.0, "a degenerate batch must contribute exactly zero"
    loss.backward()
    assert torch.isfinite(predictions.grad).all()
