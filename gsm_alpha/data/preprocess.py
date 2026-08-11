"""Sample normalisation, exactly the four steps of report section 3.2.

    "对缺失值进行填充，对成交量序列做对数处理，价格序列统一除以样本中的最后一个
     收盘价，最后进行时序 zscore 标准化。"

1. fill missing values,
2. take logs of the volume series,
3. divide *all* price series by the sample's last close,
4. z-score along the time axis.

One ambiguity is worth naming, because it changes what the network sees.  Step 3
is only meaningful if step 4 shares one mean and standard deviation across the
four price channels: a per-channel z-score is invariant to a positive scalar
divisor, so it would make step 3 a no-op *and* destroy the level relationship
between open, high, low and close (a bar's high would no longer sit above its
close).  This module therefore defaults to ``price_zscore="joint"``, the reading
under which the report's step 3 does something; ``"per_channel"`` is available
for comparison.
"""

from __future__ import annotations

import numpy as np

PRICE_CHANNELS = (0, 1, 2, 3)  # open, high, low, close
VOLUME_CHANNEL = 4


def fill_missing(window: np.ndarray) -> np.ndarray:
    """Forward-fill then backward-fill along the time axis.

    Args:
        window: ``(n_samples, n_steps, n_channels)`` with ``NaN`` gaps.

    Returns:
        A filled copy.  Series that are entirely ``NaN`` stay ``NaN`` and are
        rejected by :func:`valid_mask`.
    """
    out = np.array(window, dtype=np.float32, copy=True)
    n_steps = out.shape[1]

    # Forward fill: carry the index of the last valid observation forward.
    valid = ~np.isnan(out)
    idx = np.where(valid, np.arange(n_steps)[None, :, None], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = np.take_along_axis(out, idx, axis=1)

    # Backward fill for the leading gap, by running the same trick reversed.
    flipped = out[:, ::-1, :]
    valid = ~np.isnan(flipped)
    idx = np.where(valid, np.arange(n_steps)[None, :, None], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = np.take_along_axis(flipped, idx, axis=1)[:, ::-1, :]
    return out


def valid_mask(window: np.ndarray, min_valid_fraction: float = 0.5) -> np.ndarray:
    """Which samples carry enough real observations to be usable.

    Args:
        window: ``(n_samples, n_steps, n_channels)`` *before* filling.
        min_valid_fraction: Minimum share of time steps whose close is present.

    Returns:
        ``(n_samples,)`` boolean mask.
    """
    close_present = ~np.isnan(window[:, :, PRICE_CHANNELS[3]])
    fraction = close_present.mean(axis=1)
    last_present = close_present[:, -1]  # the sample must be alive on its own date
    return (fraction >= min_valid_fraction) & last_present


def normalise_window(
    window: np.ndarray,
    price_zscore: str = "joint",
    eps: float = 1e-8,
) -> np.ndarray:
    """Apply the section 3.2 pipeline to a batch of samples.

    Args:
        window: ``(n_samples, n_steps, 5)`` raw OHLCV, already gap-filled.
        price_zscore: ``"joint"`` (one mean/std over the whole price block, the
            default) or ``"per_channel"``.
        eps: Floor on the standard deviation, so a flat series (a suspended or
            limit-locked stock) yields zeros instead of infinities.

    Returns:
        ``(n_samples, n_steps, 5)`` float32, standardised per sample.
    """
    if window.shape[-1] != 5:
        raise ValueError(f"expected 5 OHLCV channels, got {window.shape[-1]}")
    out = np.array(window, dtype=np.float32, copy=True)

    # Step 2: log the volume series.  log1p keeps zero-volume bars finite, which
    # matter here — a suspended or limit-locked A share trades nothing all day.
    out[:, :, VOLUME_CHANNEL] = np.log1p(np.maximum(out[:, :, VOLUME_CHANNEL], 0.0))

    # Step 3: divide every price series by the sample's last close.
    last_close = out[:, -1:, PRICE_CHANNELS[3]:PRICE_CHANNELS[3] + 1]
    scale = np.where(np.abs(last_close) < eps, 1.0, last_close)
    out[:, :, PRICE_CHANNELS] /= scale

    # Step 4: z-score along time.
    #
    # The reductions accumulate in float64 on purpose. Summing ~4k float32
    # values pairwise gives an answer that depends on the array's memory layout,
    # so the same security normalised inside a full-market buffer and inside a
    # universe-screened one would differ by ~1e-5 — which a depth-5 signature
    # then amplifies by two orders of magnitude. float64 accumulation makes the
    # result layout-independent, so screening the universe before or after
    # normalising is bit-identical, and it costs nothing measurable.
    prices = out[:, :, PRICE_CHANNELS]
    if price_zscore == "joint":
        axis = (1, 2)
    elif price_zscore == "per_channel":
        axis = 1
    else:
        raise ValueError(f"unknown price_zscore mode {price_zscore!r}")
    mean = prices.mean(axis=axis, keepdims=True, dtype=np.float64)
    std = prices.std(axis=axis, keepdims=True, dtype=np.float64)
    out[:, :, PRICE_CHANNELS] = (prices - mean) / np.maximum(std, eps)

    volume = out[:, :, VOLUME_CHANNEL]
    v_mean = volume.mean(axis=1, keepdims=True, dtype=np.float64)
    v_std = volume.std(axis=1, keepdims=True, dtype=np.float64)
    out[:, :, VOLUME_CHANNEL] = (volume - v_mean) / np.maximum(v_std, eps)
    return out


def build_window(
    buffer: np.ndarray,
    price_zscore: str = "joint",
    min_valid_fraction: float = 0.5,
) -> tuple:
    """Turn a rolling raw buffer into model-ready samples plus their validity.

    Args:
        buffer: ``(n_steps, n_sids, 5)`` raw OHLCV for one lookback window,
            oldest step first.
        price_zscore: See :func:`normalise_window`.
        min_valid_fraction: See :func:`valid_mask`.

    Returns:
        ``(samples, mask)`` where ``samples`` is ``(n_sids, n_steps, 5)``
        normalised float32 and ``mask`` is ``(n_sids,)`` boolean.
    """
    raw = np.transpose(buffer, (1, 0, 2))  # (n_sids, n_steps, 5)
    mask = valid_mask(raw, min_valid_fraction)
    filled = fill_missing(raw)
    filled = np.nan_to_num(filled, nan=0.0, posinf=0.0, neginf=0.0)
    return normalise_window(filled, price_zscore=price_zscore), mask
