"""Training labels: the neutralised, cross-sectionally standardised forward return.

Report section 3.2: "标签选择行业市值中性化以及截面标准化后的未来 20 日收益率".

The OHLCV contract this project reads has no industry classification and no
market capitalisation in it, so neutralisation is a *pluggable* stage: point
``labels.exposures_path`` at a parquet of exposures keyed by ``(date,
stable_id)`` and every remaining column is used as a regressor.  With no
exposures file the labels are only cross-sectionally standardised, which is a
real deviation from the report — the paper attributes the factor's low
market-cap correlation to exactly this neutralisation — so the builder says so
out loud rather than failing quietly.  ``README.md`` shows how to generate an
exposures file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def forward_return(close: np.ndarray, horizon: int) -> np.ndarray:
    """Simple close-to-close return over the next ``horizon`` trading days.

    Args:
        close: ``(n_dates, n_sids)`` closing prices.
        horizon: Forward horizon in trading days (20 in the report).

    Returns:
        ``(n_dates, n_sids)`` returns; the final ``horizon`` rows are ``NaN``
        because their future is not in the sample.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    out = np.full_like(close, np.nan, dtype=np.float32)
    future, base = close[horizon:], close[:-horizon]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:-horizon] = np.where(base > 0, future / base - 1.0, np.nan)
    return out


def cross_sectional_standardise(row: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score one cross section, ignoring ``NaN``."""
    finite = np.isfinite(row)
    if finite.sum() < 2:
        return np.full_like(row, np.nan)
    values = row[finite]
    out = np.full_like(row, np.nan)
    out[finite] = (values - values.mean()) / max(values.std(), eps)
    return out


def neutralise(row: np.ndarray, exposures: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Residualise one cross section against a set of exposures.

    Fits ``row ~ [1, exposures]`` by least squares over the names where both
    sides are finite, and returns the residual.

    Args:
        row: ``(n_sids,)`` values to neutralise.
        exposures: ``(n_sids, n_exposures)`` regressors (industry dummies, log
            market cap, ...).
        eps: Unused ridge floor placeholder kept for signature symmetry.

    Returns:
        ``(n_sids,)`` residuals, ``NaN`` where the fit could not use the name.
    """
    usable = np.isfinite(row) & np.isfinite(exposures).all(axis=1)
    out = np.full_like(row, np.nan)
    if usable.sum() <= exposures.shape[1] + 1:
        return out
    design = np.column_stack([np.ones(usable.sum(), dtype=np.float64), exposures[usable]])
    target = row[usable].astype(np.float64)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    out[usable] = (target - design @ coef).astype(row.dtype)
    return out


@dataclass
class LabelConfig:
    """Options for :class:`LabelBuilder`.

    There are three ways to get the report's target, in decreasing order of
    fidelity to it:

    1. ``forward_label_path`` — a parquet holding an *already forward-shifted*
       return, typically a risk-model residual produced elsewhere.  Nothing is
       shifted or neutralised here; the series is standardised and used.  This
       is the best option when your platform already computes residual returns.
    2. ``exposures_path`` — the forward return is computed from closes and then
       residualised against exposures you supply (industry dummies, log market
       cap, ...).
    3. Neither — the forward return is only standardised.  A real deviation, and
       the builder warns.

    Attributes:
        horizon: Forward horizon in trading days.
        forward_label_path: Parquet keyed by ``(date, stable_id)`` whose
            ``forward_label_column`` is the forward return for the row's date.
        forward_label_column: Which column of that file to read.
        exposures_path: Optional parquet of neutralisation exposures keyed by
            ``(date, stable_id)``.  Mutually exclusive with the above.
        exposure_columns: Which exposure columns to use; ``None`` means all.
        clip_sigma: Optional winsorisation of the standardised label at this
            many standard deviations.  ``None`` (the default) is faithful to the
            report; a finite value guards the correlation loss against the fat
            right tail of 20-day A-share returns.
    """

    horizon: int = 20
    forward_label_path: Optional[str] = None
    forward_label_column: Optional[str] = None
    exposures_path: Optional[str] = None
    exposure_columns: Optional[Sequence[str]] = None
    clip_sigma: Optional[float] = None


class LabelBuilder:
    """Builds the ``(n_dates, n_sids)`` label matrix once, for reuse by every split."""

    def __init__(self, config: LabelConfig, dates: np.ndarray, sids: np.ndarray) -> None:
        if config.forward_label_path and config.exposures_path:
            raise ValueError(
                "set either labels.forward_label_path or labels.exposures_path, not both: "
                "a precomputed residual label is already neutralised, so residualising it "
                "again against exposures would be wrong."
            )
        if config.forward_label_path and not config.forward_label_column:
            raise ValueError("labels.forward_label_column is required with forward_label_path")
        self.config = config
        self.dates = dates
        self.sids = sids
        self._exposures = self._load_exposures()

    def _read_keyed_parquet(self, path: str, columns: Sequence[str]) -> np.ndarray:
        """Read a ``(date, stable_id, ...)`` parquet onto this builder's axes.

        Args:
            path: Parquet file.
            columns: Value columns to pull out, in order.

        Returns:
            ``(n_dates, n_sids, len(columns))`` float32 with ``NaN`` for
            ``(date, security)`` pairs the file does not cover.
        """
        frame = pq.read_table(path).to_pandas()
        if isinstance(frame.index, pd.MultiIndex) or frame.index.name is not None:
            frame = frame.reset_index()
        missing = ({"date", "stable_id"} | set(columns)) - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        frame["date"] = pd.to_datetime(frame["date"]).values.astype("datetime64[D]")

        d_pos = pd.Series(np.arange(len(self.dates)), index=self.dates)
        s_pos = pd.Series(np.arange(len(self.sids)), index=self.sids)
        rows = d_pos.reindex(frame["date"].values).values
        cols = s_pos.reindex(frame["stable_id"].values).values
        keep = ~pd.isna(rows) & ~pd.isna(cols)
        if not keep.any():
            raise ValueError(
                f"{path}: no (date, stable_id) pair overlaps the price panel — check that "
                "both files use the same security id convention and date range."
            )

        out = np.full((len(self.dates), len(self.sids), len(columns)), np.nan, np.float32)
        rr, cc = rows[keep].astype(np.int64), cols[keep].astype(np.int64)
        for k, name in enumerate(columns):
            out[rr, cc, k] = frame[name].values[keep].astype(np.float32)
        return out

    def _load_exposures(self) -> Optional[np.ndarray]:
        path = self.config.exposures_path
        if not path:
            if self.config.forward_label_path:
                return None  # a precomputed residual is already neutralised
            logger.warning(
                "labels.exposures_path is not set: labels will be cross-sectionally "
                "standardised but NOT industry/market-cap neutralised. The report "
                "neutralises the training target; expect a stronger size tilt than it reports."
            )
            return None

        columns = list(self.config.exposure_columns or [])
        if not columns:
            header = pq.read_schema(path).names
            columns = [c for c in header if c not in ("date", "stable_id", "__index_level_0__")]
        if not columns:
            raise ValueError(f"{path}: no exposure columns found")

        out = self._read_keyed_parquet(path, columns)
        logger.info("loaded %d neutralisation exposures from %s: %s", len(columns), path, columns)
        return out

    def build(self, close: np.ndarray) -> np.ndarray:
        """Neutralised, standardised forward returns.

        Args:
            close: ``(n_dates, n_sids)`` closing prices on this builder's axes.
                Ignored when ``forward_label_path`` supplies the return.

        Returns:
            ``(n_dates, n_sids)`` float32 labels with ``NaN`` where undefined.
        """
        raw = self._forward_returns(close)
        out = np.full_like(raw, np.nan)
        for i in range(raw.shape[0]):
            row = raw[i]
            if self._exposures is not None:
                row = neutralise(row, self._exposures[i])
            row = cross_sectional_standardise(row)
            if self.config.clip_sigma is not None:
                row = np.clip(row, -self.config.clip_sigma, self.config.clip_sigma)
            out[i] = row
        return out

    def _forward_returns(self, close: np.ndarray) -> np.ndarray:
        """The forward return, either read from file or computed from closes."""
        path = self.config.forward_label_path
        if not path:
            return forward_return(close, self.config.horizon)

        column = self.config.forward_label_column
        # The file already holds the FORWARD return for each row's date, so it
        # must not be shifted again here. Alignment is the caller's to get right;
        # scripts/export_labels.py preserves it, and the units do not matter
        # because the series is cross-sectionally standardised below.
        raw = self._read_keyed_parquet(path, [column])[:, :, 0]
        coverage = np.isfinite(raw).any(axis=1).mean()
        logger.info(
            "read forward labels from %s column %r: %.0f%% of dates covered, "
            "%.1f names per covered date (no shift applied, already forward-looking)",
            path, column, 100 * coverage,
            np.isfinite(raw).sum(axis=1)[np.isfinite(raw).any(axis=1)].mean() if coverage else 0.0,
        )
        return raw

