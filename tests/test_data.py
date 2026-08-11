"""Preprocessing, labels and the rolling split arithmetic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gsm_alpha.data.datamodule import split_dates
from gsm_alpha.data.labels import (
    LabelBuilder,
    LabelConfig,
    cross_sectional_standardise,
    forward_return,
    neutralise,
)
from gsm_alpha.data.preprocess import build_window, fill_missing, normalise_window, valid_mask


def test_fill_missing_forward_then_backward():
    window = np.array([[[np.nan], [1.0], [np.nan], [3.0], [np.nan]]], dtype=np.float32)
    filled = fill_missing(window)
    assert np.allclose(filled[0, :, 0], [1.0, 1.0, 1.0, 3.0, 3.0])


def test_fill_missing_leaves_all_nan_series_alone():
    window = np.full((1, 4, 1), np.nan, dtype=np.float32)
    assert np.isnan(fill_missing(window)).all()


def test_valid_mask_needs_coverage_and_a_live_last_bar():
    window = np.full((3, 10, 5), 1.0, dtype=np.float32)
    window[1, :8, 3] = np.nan       # only 20% coverage
    window[2, -1, 3] = np.nan       # not trading on its own date
    assert list(valid_mask(window, 0.5)) == [True, False, False]


def test_normalise_window_standardises_each_sample_along_time():
    rng = np.random.default_rng(0)
    window = np.abs(rng.normal(10, 1, size=(4, 30, 5))).astype(np.float32) + 1.0
    out = normalise_window(window, price_zscore="joint")
    assert out.shape == window.shape
    prices = out[:, :, :4]
    assert np.allclose(prices.mean(axis=(1, 2)), 0, atol=1e-4)
    assert np.allclose(prices.std(axis=(1, 2)), 1, atol=1e-4)
    volume = out[:, :, 4]
    assert np.allclose(volume.mean(axis=1), 0, atol=1e-4)
    assert np.allclose(volume.std(axis=1), 1, atol=1e-4)


def test_joint_price_zscore_preserves_the_bar_ordering():
    """high >= close >= low must survive normalisation, or the path is nonsense."""
    rng = np.random.default_rng(1)
    low = np.abs(rng.normal(10, 1, size=(3, 20))).astype(np.float32)
    window = np.stack([low + 0.3, low + 0.5, low, low + 0.2, np.abs(rng.normal(1e6, 1e5, (3, 20)))], -1)
    out = normalise_window(window.astype(np.float32), price_zscore="joint")
    assert (out[:, :, 1] >= out[:, :, 3]).all()   # high >= close
    assert (out[:, :, 3] >= out[:, :, 2]).all()   # close >= low


def test_per_channel_price_zscore_makes_the_last_close_divisor_a_no_op():
    """Why the default is 'joint': a per-channel z-score cancels any scalar divisor."""
    rng = np.random.default_rng(2)
    window = np.abs(rng.normal(10, 1, size=(2, 15, 5))).astype(np.float32) + 1.0
    a = normalise_window(window, price_zscore="per_channel")
    b = normalise_window(window * np.float32(3.0), price_zscore="per_channel")
    assert np.allclose(a[:, :, :4], b[:, :, :4], atol=1e-4)


def test_zero_volume_bars_stay_finite():
    window = np.ones((1, 10, 5), dtype=np.float32)
    window[0, :, 4] = 0.0
    assert np.isfinite(normalise_window(window)).all()


def test_build_window_transposes_and_masks():
    buffer = np.ones((12, 4, 5), dtype=np.float32)   # (steps, sids, channels)
    buffer[:, 2, 3] = np.nan
    samples, mask = build_window(buffer)
    assert samples.shape == (4, 12, 5)
    assert list(mask) == [True, True, False, True]


def test_forward_return_looks_only_forward():
    close = np.array([[10.0], [11.0], [12.0], [13.0]], dtype=np.float32)
    out = forward_return(close, 2)
    assert np.isclose(out[0, 0], 12 / 10 - 1)
    assert np.isclose(out[1, 0], 13 / 11 - 1)
    assert np.isnan(out[2:, 0]).all()


def test_cross_sectional_standardise_ignores_nan():
    row = np.array([1.0, 2.0, 3.0, np.nan], dtype=np.float32)
    out = cross_sectional_standardise(row)
    assert np.isnan(out[3])
    assert np.isclose(np.nanmean(out), 0, atol=1e-6)
    assert np.isclose(np.nanstd(out), 1, atol=1e-6)


def test_neutralise_removes_the_exposure():
    rng = np.random.default_rng(3)
    size = rng.normal(size=200).astype(np.float32)
    row = (2.5 * size + rng.normal(scale=0.1, size=200)).astype(np.float32)
    residual = neutralise(row, size[:, None])
    assert abs(np.corrcoef(residual, size)[0, 1]) < 1e-6


def test_label_builder_neutralises_against_an_exposures_file(tmp_path):
    dates = np.array(["2020-01-02", "2020-01-03", "2020-01-06"], dtype="datetime64[D]")
    sids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    rng = np.random.default_rng(4)
    close = np.abs(rng.normal(20, 2, size=(3, 6))).astype(np.float32) + 5

    size = rng.normal(size=(3, 6)).astype(np.float32)
    frame = pd.DataFrame(
        {
            "date": np.repeat(dates, 6),
            "stable_id": np.tile(sids, 3),
            "log_mktcap": size.reshape(-1),
        }
    )
    path = tmp_path / "exposures.parquet"
    frame.to_parquet(path, index=False)

    builder = LabelBuilder(LabelConfig(horizon=1, exposures_path=str(path)), dates, sids)
    labels = builder.build(close)
    usable = np.isfinite(labels[0])
    assert usable.sum() == 6
    assert abs(np.corrcoef(labels[0][usable], size[0][usable])[0, 1]) < 1e-4


def test_label_builder_without_exposures_only_standardises(caplog):
    dates = np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[D]")
    sids = np.arange(5, dtype=np.int64)
    close = np.array([[10, 11, 12, 13, 14], [11, 11, 13, 12, 16]], dtype=np.float32)
    labels = LabelBuilder(LabelConfig(horizon=1), dates, sids).build(close)
    assert np.isclose(np.nanmean(labels[0]), 0, atol=1e-6)
    assert np.isclose(np.nanstd(labels[0]), 1, atol=1e-6)


def test_label_clipping_bounds_the_tail():
    dates = np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[D]")
    sids = np.arange(20, dtype=np.int64)
    close = np.ones((2, 20), dtype=np.float32) * 10
    close[1, 0] = 1000.0  # one enormous winner
    labels = LabelBuilder(LabelConfig(horizon=1, clip_sigma=2.0), dates, sids).build(close)
    assert np.nanmax(np.abs(labels)) <= 2.0 + 1e-6


def test_split_dates_follows_the_report_schedule():
    dates = pd.date_range("2014-01-01", "2019-12-31", freq="B").values.astype("datetime64[D]")
    splits = split_dates(dates, predict_year=2018, lookback_years=4,
                         validation_years=1, drop_last_month=True)

    train_years = {d.astype("datetime64[Y]").astype(int) + 1970 for d in splits["train"]}
    assert train_years == {2014, 2015, 2016}
    assert {d.astype("datetime64[Y]").astype(int) + 1970 for d in splits["val"]} == {2017}
    assert {d.astype("datetime64[Y]").astype(int) + 1970 for d in splits["predict"]} == {2018}

    # "剔除末月": neither fitted split may reach into its final December.
    assert max(splits["train"]) < np.datetime64("2016-12-01")
    assert max(splits["val"]) < np.datetime64("2017-12-01")
    # And no fitted date may overlap the year being predicted.
    assert max(splits["val"]) < min(splits["predict"])


def test_split_dates_moves_with_the_prediction_year():
    dates = pd.date_range("2015-01-01", "2021-12-31", freq="B").values.astype("datetime64[D]")
    splits = split_dates(dates, 2020, 4, 1, True)
    assert {d.astype("datetime64[Y]").astype(int) + 1970 for d in splits["train"]} == {2016, 2017, 2018}
    assert {d.astype("datetime64[Y]").astype(int) + 1970 for d in splits["val"]} == {2019}


def _write_forward_labels(tmp_path, dates, sids, values, column="resp_res_20d"):
    frame = pd.DataFrame(
        {
            "date": np.repeat(dates, len(sids)),
            "stable_id": np.tile(sids, len(dates)),
            column: values.reshape(-1),
        }
    )
    path = tmp_path / "labels.parquet"
    frame.to_parquet(path, index=False)
    return str(path), column


def test_precomputed_forward_label_is_used_verbatim(tmp_path):
    """A resp_* series is already forward-shifted; it must not be shifted again."""
    dates = np.array(["2020-01-02", "2020-01-03", "2020-01-06"], dtype="datetime64[D]")
    sids = np.arange(10, dtype=np.int64)
    rng = np.random.default_rng(0)
    values = rng.normal(size=(3, 10)).astype(np.float32)
    path, column = _write_forward_labels(tmp_path, dates, sids, values)

    # Closes are deliberately garbage: if the builder touched them, the labels
    # would not match the file.
    close = np.ones((3, 10), dtype=np.float32)
    labels = LabelBuilder(
        LabelConfig(horizon=20, forward_label_path=path, forward_label_column=column),
        dates, sids,
    ).build(close)

    for i in range(3):
        expected = cross_sectional_standardise(values[i])
        assert np.allclose(labels[i], expected, atol=1e-5)
    # Rank order is preserved from the file, on every row including the last —
    # a precomputed label has no unusable tail.
    assert np.array_equal(np.argsort(labels[-1]), np.argsort(values[-1]))


def test_precomputed_label_is_scale_invariant(tmp_path):
    """Basis points or decimals: standardisation removes the unit."""
    dates = np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[D]")
    sids = np.arange(8, dtype=np.int64)
    rng = np.random.default_rng(1)
    values = rng.normal(size=(2, 8)).astype(np.float32)
    close = np.ones((2, 8), dtype=np.float32)

    a_path, col = _write_forward_labels(tmp_path, dates, sids, values)
    scaled_dir = tmp_path / "b"
    scaled_dir.mkdir(exist_ok=True)
    b_path, _ = _write_forward_labels(scaled_dir, dates, sids, values * 10000.0, col)

    make = lambda p: LabelBuilder(
        LabelConfig(forward_label_path=p, forward_label_column=col), dates, sids
    ).build(close)
    assert np.allclose(make(a_path), make(b_path), atol=1e-4)


def test_forward_label_and_exposures_are_mutually_exclusive(tmp_path):
    dates = np.array(["2020-01-02"], dtype="datetime64[D]")
    sids = np.arange(4, dtype=np.int64)
    path, column = _write_forward_labels(tmp_path, dates, sids, np.zeros((1, 4), np.float32))
    with pytest.raises(ValueError, match="not both"):
        LabelBuilder(
            LabelConfig(forward_label_path=path, forward_label_column=column,
                        exposures_path=path),
            dates, sids,
        )


def test_forward_label_requires_a_column(tmp_path):
    dates = np.array(["2020-01-02"], dtype="datetime64[D]")
    sids = np.arange(4, dtype=np.int64)
    path, _ = _write_forward_labels(tmp_path, dates, sids, np.zeros((1, 4), np.float32))
    with pytest.raises(ValueError, match="forward_label_column"):
        LabelBuilder(LabelConfig(forward_label_path=path), dates, sids)


def test_forward_label_with_no_overlap_fails_loudly(tmp_path):
    """A security-id convention mismatch must not silently produce empty labels."""
    dates = np.array(["2020-01-02"], dtype="datetime64[D]")
    path, column = _write_forward_labels(
        tmp_path, dates, np.arange(900000, 900004, dtype=np.int64), np.zeros((1, 4), np.float32)
    )
    with pytest.raises(ValueError, match="no .* pair overlaps"):
        LabelBuilder(
            LabelConfig(forward_label_path=path, forward_label_column=column),
            dates, np.arange(4, dtype=np.int64),
        ).build(np.ones((1, 4), dtype=np.float32))
