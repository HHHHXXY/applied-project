"""Factor evaluation in the report's own vocabulary (tables 8-14).

Rank IC, ICIR, quintile long-short performance and per-year breakdown, computed
from the stitched factor panel and the daily closes.  This is a lightweight
in-repo check that the pipeline produces a sane factor — it is not a portfolio
backtester, and it deliberately makes no attempt to model the section 3.4 index
enhancement (weight caps, tracking error, industry bands), which needs an
optimiser and a risk model this project does not carry.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .data.sources import DailyPanel

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 244


def _panel_to_matrix(factor: pd.DataFrame, dates: np.ndarray, sids: np.ndarray) -> np.ndarray:
    """Reshape a long ``(date, stable_id, factor)`` frame onto the panel axes."""
    d_pos = pd.Series(np.arange(len(dates)), index=dates)
    s_pos = pd.Series(np.arange(len(sids)), index=sids)
    rows = d_pos.reindex(pd.to_datetime(factor["date"]).values.astype("datetime64[D]")).values
    cols = s_pos.reindex(factor["stable_id"].values).values
    keep = ~pd.isna(rows) & ~pd.isna(cols)
    out = np.full((len(dates), len(sids)), np.nan, np.float32)
    out[rows[keep].astype(int), cols[keep].astype(int)] = factor["factor"].values[keep]
    return out


def _neutralize_factor(
    values: np.ndarray,
    exposures_path: str,
    dates: np.ndarray,
    sids: np.ndarray,
    winsorize_sigma: float = 3.0,
) -> np.ndarray:
    """Winsorise, standardise and residualise each cross section of the factor.

    The report's section 3.3 preprocessing.  Reuses the same
    :func:`~gsm_alpha.data.labels.neutralise` least-squares residual the label
    builder uses, so the factor side and the label side cannot drift apart.

    Args:
        values: ``(n_dates, n_sids)`` raw factor values.
        exposures_path: Parquet keyed by ``(date, stable_id)``; every other
            column is a regressor.
        dates: Date axis of ``values``.
        sids: Security axis of ``values``.
        winsorize_sigma: Clip level before neutralising.

    Returns:
        ``(n_dates, n_sids)`` neutralised factor values.
    """
    from .data.labels import LabelBuilder, LabelConfig, cross_sectional_standardise, neutralise

    # Borrow LabelBuilder purely for its (date, stable_id) -> panel alignment.
    loader = LabelBuilder(LabelConfig(exposures_path=exposures_path), dates, sids)
    exposures = loader._exposures  # noqa: SLF001 - same package, one owner of the reader
    if exposures is None:
        raise ValueError(f"{exposures_path}: no exposure columns to neutralise against")

    out = np.full_like(values, np.nan)
    for i in range(values.shape[0]):
        row = cross_sectional_standardise(values[i])
        if not np.isfinite(row).any():
            continue
        row = np.clip(row, -winsorize_sigma, winsorize_sigma)
        out[i] = neutralise(row, exposures[i])
    covered = np.isfinite(out).any(axis=1).sum()
    logger.info(
        "neutralised the factor against %s on %d of %d dates carrying factor values",
        exposures_path, covered, int(np.isfinite(values).any(axis=1).sum()),
    )
    return out


def _rank(row: np.ndarray) -> np.ndarray:
    """Cross-sectional ranks with ``NaN`` preserved."""
    out = np.full_like(row, np.nan, dtype=np.float64)
    finite = np.isfinite(row)
    if finite.sum() < 2:
        return out
    order = np.argsort(np.argsort(row[finite]))
    out[finite] = order
    return out


def evaluate(
    factor: pd.DataFrame,
    daily_path: str,
    horizon: int = 20,
    n_groups: int = 5,
    rebalance: str = "monthly",
    forward_return_path: Optional[str] = None,
    forward_return_column: Optional[str] = None,
    forward_return_scale: float = 1.0,
    neutralize_path: Optional[str] = None,
    winsorize_sigma: float = 3.0,
) -> Dict[str, object]:
    """Score a factor panel.

    Args:
        factor: Long frame with ``date``, ``stable_id`` and ``factor`` columns.
        daily_path: The daily OHLCV parquet, used for the security/date axes and,
            by default, for the forward return.
        horizon: Forward return horizon in trading days (20).
        n_groups: Quantile buckets (5, as in the report's 五分组).
        rebalance: ``"monthly"`` (the report's cadence) or ``"daily"`` —
            which dates enter the IC series and the group returns.
        forward_return_path: Optional parquet of an already forward-shifted
            return keyed by ``(date, stable_id)``, scored instead of the
            close-to-close return.  **Score a factor against the return
            definition it was trained on**: a model fitted on risk-model
            residuals deliberately ignores the style and industry beta that
            dominates the cross section of *total* returns, so scoring it on
            total returns understates it, sometimes by a lot.
        forward_return_column: Which column of that file to score against.
        neutralize_path: Optional exposures parquet keyed by ``(date, stable_id)``.
            When given, each cross section of the factor is winsorised,
            standardised and residualised against those exposures before it is
            scored — the report's "去极值、标准化以及行业市值中性化" preprocessing
            (section 3.3).  Its tables 12 and 14 are computed this way; table 13
            is the un-neutralised arm, which needs no exposures because
            winsorising and standardising are monotone and so leave both Rank IC
            and the quantile buckets unchanged.
        winsorize_sigma: Clip each cross section at this many standard deviations
            before neutralising.
        forward_return_scale: Multiplier turning that column into a decimal
            return, so the annualised figures print as real percentages.  Rank IC
            is scale-invariant and unaffected.  The zoo's residual series are in
            basis points, i.e. ``1e-4``.

    Returns:
        ``{"summary": {...}, "by_year": DataFrame, "ic_series": Series}``.
    """
    panel = DailyPanel(daily_path)
    close = panel.close_matrix()
    values = _panel_to_matrix(factor, panel.dates, panel.sids)

    if neutralize_path:
        values = _neutralize_factor(values, neutralize_path, panel.dates, panel.sids,
                                    winsorize_sigma=winsorize_sigma)

    if forward_return_path:
        if not forward_return_column:
            raise ValueError("forward_return_column is required with forward_return_path")
        frame = pd.read_parquet(forward_return_path)
        if isinstance(frame.index, pd.MultiIndex) or frame.index.name is not None:
            frame = frame.reset_index()
        frame = frame.rename(columns={forward_return_column: "factor"})
        forward = _panel_to_matrix(frame, panel.dates, panel.sids) * forward_return_scale
        logger.info("scoring against %s column %r (scale %g)",
                    forward_return_path, forward_return_column, forward_return_scale)
    else:
        forward = np.full_like(close, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            forward[:-horizon] = np.where(
                close[:-horizon] > 0, close[horizon:] / close[:-horizon] - 1.0, np.nan
            )

    dates = pd.DatetimeIndex(panel.dates)
    has_factor = np.isfinite(values).sum(axis=1) > 0
    if rebalance == "monthly":
        month = dates.year * 100 + dates.month
        is_last = np.append(month[:-1] != month[1:], True)
        active = has_factor & is_last
    elif rebalance == "daily":
        active = has_factor
    else:
        raise ValueError(f"unknown rebalance {rebalance!r}")

    ic_values, group_returns, active_dates = [], [], []
    for i in np.flatnonzero(active):
        row, fwd = values[i], forward[i]
        usable = np.isfinite(row) & np.isfinite(fwd)
        if usable.sum() < n_groups * 5:
            continue
        f_rank, r_rank = _rank(np.where(usable, row, np.nan)), _rank(np.where(usable, fwd, np.nan))
        both = np.isfinite(f_rank) & np.isfinite(r_rank)
        ic_values.append(float(np.corrcoef(f_rank[both], r_rank[both])[0, 1]))

        edges = np.quantile(row[usable], np.linspace(0, 1, n_groups + 1))
        buckets = np.clip(np.searchsorted(edges[1:-1], row[usable], side="right"), 0, n_groups - 1)
        group_returns.append([float(fwd[usable][buckets == g].mean()) for g in range(n_groups)])
        active_dates.append(panel.dates[i])

    if not ic_values:
        raise ValueError("no evaluable dates: the factor panel and the daily panel do not overlap")

    ic = pd.Series(ic_values, index=pd.DatetimeIndex(active_dates), name="rank_ic")
    groups = pd.DataFrame(
        group_returns, index=ic.index, columns=[f"group_{g + 1}" for g in range(n_groups)]
    )
    long_short = groups.iloc[:, -1] - groups.iloc[:, 0]
    periods_per_year = TRADING_DAYS_PER_YEAR / horizon

    summary = {
        "n_periods": int(len(ic)),
        "rank_ic": float(ic.mean()),
        "icir": float(ic.mean() / ic.std() * np.sqrt(periods_per_year)) if ic.std() > 0 else np.nan,
        "ic_win_rate": float((ic > 0).mean()),
        "long_short_annual": float(long_short.mean() * periods_per_year),
        "long_short_vol": float(long_short.std() * np.sqrt(periods_per_year)),
        "long_short_sharpe": (
            float(long_short.mean() / long_short.std() * np.sqrt(periods_per_year))
            if long_short.std() > 0
            else np.nan
        ),
        "top_group_annual": float(groups.iloc[:, -1].mean() * periods_per_year),
        "top_minus_equal_weight": float(
            (groups.iloc[:, -1] - groups.mean(axis=1)).mean() * periods_per_year
        ),
    }

    by_year = pd.DataFrame(
        {
            "rank_ic": ic.groupby(ic.index.year).mean(),
            "icir": ic.groupby(ic.index.year).apply(
                lambda s: s.mean() / s.std() * np.sqrt(periods_per_year) if s.std() > 0 else np.nan
            ),
            "long_short_annual": long_short.groupby(long_short.index.year).mean() * periods_per_year,
        }
    )
    return {"summary": summary, "by_year": by_year, "ic_series": ic, "groups": groups}


def format_report(result: Dict[str, object]) -> str:
    """Render :func:`evaluate`'s output the way the report's tables read."""
    summary = result["summary"]
    lines = [
        "=== factor summary ===",
        f"  periods            {summary['n_periods']}",
        f"  Rank IC            {summary['rank_ic']:.4f}",
        f"  ICIR (annualised)  {summary['icir']:.2f}",
        f"  IC win rate        {summary['ic_win_rate']:.2%}",
        f"  long-short annual  {summary['long_short_annual']:.2%}",
        f"  long-short vol     {summary['long_short_vol']:.2%}",
        f"  long-short Sharpe  {summary['long_short_sharpe']:.2f}",
        f"  top group annual   {summary['top_group_annual']:.2%}",
        "",
        "=== by year ===",
        result["by_year"].to_string(float_format=lambda v: f"{v:.4f}"),
    ]
    return "\n".join(lines)
