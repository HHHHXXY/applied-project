"""End-to-end integration on synthetic panels written in the data contract.

This is the test that backs the portability claim: it fabricates two parquet
files that satisfy nothing but the documented contract, then runs the whole
pipeline over them — cache build, day-batched loading, a rolling fit, prediction
and evaluation.  If the contract in ``README.md`` is enough, this passes on any
machine.

Everything is kept tiny (a handful of names, a short grid) so the suite stays
fast; correctness of the pieces is covered by the other modules' tests.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gsm_alpha.config import Config
from gsm_alpha.data.cache import CacheBuilder, available_dates, read_manifest
from gsm_alpha.data.datamodule import GSMAlphaDataModule
from gsm_alpha.data.labels import LabelBuilder, LabelConfig
from gsm_alpha.data.sources import DailyPanel, IntradayPanel

N_SIDS = 30
BARS_PER_DAY = 8


def _write_panels(tmp_path, n_days: int = 90):
    """Fabricate daily and intraday parquets that satisfy the data contract.

    The intraday grid deliberately includes one label that is empty on every
    date, mirroring the real feed's right-edge 09:30/13:00 bars, so the reader's
    grid probing is exercised rather than assumed away.
    """
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-01", periods=n_days).values.astype("datetime64[D]")
    sids = np.arange(20000001, 20000001 + N_SIDS, dtype=np.int64)

    walk = np.cumsum(rng.normal(0, 0.02, size=(n_days, N_SIDS)), axis=0)
    close = (10 * np.exp(walk)).astype(np.float32)
    volume = np.abs(rng.normal(1e6, 2e5, size=(n_days, N_SIDS))).astype(np.float32)

    daily = pd.DataFrame(
        {
            "date": np.repeat(dates, N_SIDS),
            "stable_id": np.tile(sids, n_days),
            "open": close.reshape(-1) * 0.99,
            "high": close.reshape(-1) * 1.01,
            "low": close.reshape(-1) * 0.98,
            "close": close.reshape(-1),
            "volume": volume.reshape(-1),
        }
    )
    daily_path = tmp_path / "daily.parquet"
    daily.to_parquet(daily_path, index=False)

    # 09:30 is the always-empty boundary label; the rest carry bars.
    base = dt.datetime(2000, 1, 1, 9, 30)
    times = [(base + dt.timedelta(minutes=5 * i)).time() for i in range(BARS_PER_DAY + 1)]
    n_times = len(times)
    intraday_close = np.repeat(close[:, None, :], n_times, axis=1)
    intraday_close *= 1 + rng.normal(0, 0.002, size=intraday_close.shape).astype(np.float32)
    intraday_close[:, 0, :] = np.nan  # the empty label
    intraday_volume = np.abs(rng.normal(1e5, 2e4, size=intraday_close.shape)).astype(np.float32)
    intraday_volume[:, 0, :] = np.nan

    intraday = pd.DataFrame(
        {
            "date": np.repeat(dates, n_times * N_SIDS),
            "time": np.tile(np.repeat(times, N_SIDS), n_days),
            "stable_id": np.tile(sids, n_days * n_times),
            "open": intraday_close.reshape(-1) * 0.999,
            "high": intraday_close.reshape(-1) * 1.001,
            "low": intraday_close.reshape(-1) * 0.998,
            "close": intraday_close.reshape(-1),
            "volume": intraday_volume.reshape(-1),
        }
    )
    intraday_path = tmp_path / "intraday.parquet"
    # One row group per date, the layout the reader prefers.
    intraday.to_parquet(intraday_path, index=False, row_group_size=n_times * N_SIDS)
    return str(daily_path), str(intraday_path), dates, sids


def _config(tmp_path, daily_path, intraday_path, **overrides) -> Config:
    raw = {
        "data": {
            "daily_path": daily_path,
            "intraday_path": intraday_path,
            "cache_dir": str(tmp_path / "cache"),
            "start_date": "2018-01-01",
            "end_date": "2018-12-31",
            "minute_lookback_days": 4,
            "daily_lookback_days": 6,
            "train_day_stride": 1,
            "predict_day_stride": 1,
            "universe_top_n": None,
            "universe_lookback_days": 5,
            **overrides.pop("data", {}),
        },
        "minute_gsm": {"depth": 2, "backend": "torch", **overrides.pop("minute_gsm", {})},
        "daily_gsm": {"depth": 2, "backend": "torch"},
        "model": {"hidden_dim": 16, "n_attention_heads": 2, "attention_dropout": 0.0},
        "labels": {"horizon": 3},
        "train": {"num_workers": 0, "max_epochs": 2, "output_dir": str(tmp_path / "run"),
                  **overrides.pop("train", {})},
    }
    return Config.from_dict(raw)


@pytest.fixture(scope="module")
def panels(tmp_path_factory):
    return _write_panels(tmp_path_factory.mktemp("panels"))


def test_readers_honour_the_documented_contract(panels):
    daily_path, intraday_path, dates, sids = panels
    daily = DailyPanel(daily_path)
    assert daily.dates.dtype == np.dtype("datetime64[D]")
    assert np.array_equal(daily.sids, sids)
    assert daily.values.shape == (len(dates), N_SIDS, 5)
    assert np.isfinite(daily.close_matrix()).all()

    intraday = IntradayPanel(intraday_path, daily.sids)
    # The always-empty boundary label must be dropped, the rest kept.
    assert len(intraday.times) == BARS_PER_DAY
    assert dt.time(9, 30) not in intraday.times
    day = intraday.day(dates[10])
    assert day.shape == (BARS_PER_DAY, N_SIDS, 5)
    assert np.isfinite(day).all()


def test_reader_accepts_a_multiindex_keyed_file(panels, tmp_path):
    """The contract allows the key as an index or as columns."""
    daily_path, _, _, _ = panels
    frame = pd.read_parquet(daily_path).set_index(["date", "stable_id"])
    indexed = tmp_path / "indexed.parquet"
    frame.to_parquet(indexed)
    assert np.array_equal(DailyPanel(str(indexed)).dates, DailyPanel(daily_path).dates)


def test_cache_build_writes_one_entry_per_date_per_branch(panels, tmp_path):
    daily_path, intraday_path, _, _ = panels
    config = _config(tmp_path, daily_path, intraday_path)
    plan = CacheBuilder(config).run(stage="features", stride=1)

    manifest = read_manifest(config.data.cache_dir)
    # 4 lookback days x 8 bars = 32 intraday steps.
    assert manifest["minute_steps"] == 4 * BARS_PER_DAY
    # 10 pair-streams x depth-2 log-signature over 3 channels (3 + 3 words) = 60.
    assert manifest["feature_dims"] == {"minute": 60, "daily": 60}

    for branch in ("minute", "daily"):
        cached = available_dates(config.data.cache_dir, "features", branch)
        assert len(cached) == len(plan.sample_dates)
        with np.load(
            f"{config.data.cache_dir}/features/{branch}/"
            f"{str(cached[0]).replace('-', '')}.npz"
        ) as blob:
            assert blob["feature"].shape == (N_SIDS, 60)
            assert np.isfinite(blob["feature"]).all()


def test_cache_build_is_resumable(panels, tmp_path):
    daily_path, intraday_path, _, _ = panels
    config = _config(tmp_path, daily_path, intraday_path)
    builder = CacheBuilder(config)
    builder.run(stage="features", stride=1)
    before = {
        d: __import__("os").path.getmtime(
            f"{config.data.cache_dir}/features/daily/{str(d).replace('-', '')}.npz"
        )
        for d in available_dates(config.data.cache_dir, "features", "daily")
    }
    builder.run(stage="features", stride=1)  # second pass must not rewrite
    after = {
        d: __import__("os").path.getmtime(
            f"{config.data.cache_dir}/features/daily/{str(d).replace('-', '')}.npz"
        )
        for d in before
    }
    assert before == after


def test_windows_stage_round_trips(panels, tmp_path):
    daily_path, intraday_path, _, _ = panels
    config = _config(tmp_path, daily_path, intraday_path)
    CacheBuilder(config).run(stage="windows", stride=1)
    dates = available_dates(config.data.cache_dir, "windows", "minute")
    with np.load(
        f"{config.data.cache_dir}/windows/minute/{str(dates[0]).replace('-', '')}.npz"
    ) as blob:
        assert blob["window"].shape == (N_SIDS, 4 * BARS_PER_DAY, 5)


def test_learnable_augmentation_cannot_be_feature_cached(panels, tmp_path):
    daily_path, intraday_path, _, _ = panels
    config = _config(
        tmp_path,
        daily_path,
        intraday_path,
        minute_gsm={"depth": 2, "backend": "torch",
                    "augmentations": ["multi_headed_stream_preserving", "time", "basepoint"],
                    "n_projections": 2, "projection_out": 2},
    )
    with pytest.raises(ValueError, match="learnable augmentation"):
        CacheBuilder(config).run(stage="features", stride=1)


def test_datamodule_serves_aligned_day_batches(panels, tmp_path):
    daily_path, intraday_path, _, _ = panels
    config = _config(tmp_path, daily_path, intraday_path)
    CacheBuilder(config).run(stage="features", stride=1)

    panel = DailyPanel(daily_path)
    labels = LabelBuilder(LabelConfig(horizon=3), panel.dates, panel.sids).build(
        panel.close_matrix()
    )
    module = GSMAlphaDataModule(config, labels, panel.dates, panel.sids, predict_year=2018)
    # A single synthetic year cannot fill a 4-year lookback, so train is empty
    # and only the prediction split is populated; that is the correct behaviour.
    assert module.splits["predict"]
    assert module.feature_dims() == {"minute": 60, "daily": 60}

    batch = next(iter(module.predict_dataloader()))
    assert batch["minute"].shape == (N_SIDS, 60)
    assert batch["daily"].shape == (N_SIDS, 60)
    assert batch["sid"].shape == (N_SIDS,)
    assert batch["label"].shape == (N_SIDS,)


def test_rolling_fit_and_evaluation(tmp_path):
    """The full loop on enough synthetic history to fill one rolling window."""
    from gsm_alpha.evaluate import evaluate
    from gsm_alpha.train import build_labels, train_one_year

    daily_path, intraday_path, _, _ = _write_panels(tmp_path, n_days=1300)  # ~5 years
    config = _config(
        tmp_path,
        daily_path,
        intraday_path,
        data={"end_date": "2022-12-31"},
        train={"num_workers": 0, "max_epochs": 2, "output_dir": str(tmp_path / "run"),
               "early_stopping_patience": 1},
    )
    config.data.train_day_stride = 10
    CacheBuilder(config).run(stage="features", stride=10)

    factor = train_one_year(config, build_labels(config), predict_year=2022)
    assert factor is not None and len(factor) > 0
    assert set(factor.columns) == {"date", "stable_id", "factor"}
    assert np.isfinite(factor["factor"]).all()
    assert factor["date"].dt.year.unique().tolist() == [2022]

    result = evaluate(factor, daily_path, horizon=3, n_groups=3, rebalance="daily")
    assert np.isfinite(result["summary"]["rank_ic"])
    assert result["summary"]["n_periods"] > 0


def test_float16_cache_refuses_to_overflow():
    """Depth-5 log-signatures exceed float16's range; the cast must not do it silently."""
    from gsm_alpha.data.cache import _cast_features

    safe = np.array([[1.0, -2.5, 1e4]], dtype=np.float32)
    assert _cast_features(safe, "float16", "minute", "2021-01-04").dtype == np.float16
    assert _cast_features(safe, "float32", "minute", "2021-01-04").dtype == np.float32

    # A real depth-5 log-signature reaches into the hundreds of thousands.
    overflowing = np.array([[1.0, 264216.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="float16 overflows"):
        _cast_features(overflowing, "float16", "minute", "2021-01-04")

    with pytest.raises(ValueError, match="unknown data.feature_dtype"):
        _cast_features(safe, "bfloat16", "minute", "2021-01-04")


def test_validation_metric_is_always_logged(tmp_path):
    """A degenerate validation epoch must score badly, not abort the fit.

    EarlyStopping and ModelCheckpoint monitor val/rank_ic; if the key is missing
    Lightning raises and the whole run dies, which is a terrible failure mode
    hours into a rolling retrain.
    """
    import torch
    from gsm_alpha.config import Config
    from gsm_alpha.models.gsm_alpha import GSMAlpha

    config = Config.from_dict({"model": {"hidden_dim": 16, "n_attention_heads": 2},
                               "minute_gsm": {"backend": "torch"},
                               "daily_gsm": {"backend": "torch"}})
    model = GSMAlpha(config, {"daily": 8})
    logged = {}
    model.log = lambda name, value, **kw: logged.__setitem__(name, float(value))

    model.on_validation_epoch_start()
    model.on_validation_epoch_end()          # no batches contributed an IC
    assert logged["val/rank_ic"] == float("-inf")

    model.on_validation_epoch_start()
    model._val_ic = [0.05, 0.03]
    model.on_validation_epoch_end()
    assert logged["val/rank_ic"] == pytest.approx(0.04)
