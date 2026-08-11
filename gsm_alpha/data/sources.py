"""Parquet readers for the daily and 5-minute price/volume panels.

The data contract — everything the project needs from a deployment's own data
lake — is deliberately small, so the code runs anywhere the same two files
exist.  See ``README.md`` for the prose version.

**Daily file** (one parquet): columns ``open, high, low, close, volume``, keyed
by ``date`` (a date) and ``stable_id`` (an integer security id).  The key may be
a pandas MultiIndex or two ordinary columns; both are accepted.

**Intraday file** (one parquet): the same five value columns keyed by ``date``,
``time`` (a time of day) and ``stable_id``.  Every date must carry the same
intraday grid.  Reading is much faster when the file is written with one row
group per date, which the reader detects and exploits, but it is not required.

Both readers hand back dense ``float32`` arrays aligned to a fixed security
axis, with missing observations left as ``NaN`` for the preprocessor to deal
with.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OHLCV: tuple = ("open", "high", "low", "close", "volume")


def _flatten(table_df: pd.DataFrame) -> pd.DataFrame:
    """Move any index levels into columns so key handling is uniform."""
    if isinstance(table_df.index, pd.MultiIndex) or table_df.index.name is not None:
        table_df = table_df.reset_index()
    return table_df


def _as_date_array(values) -> np.ndarray:
    """Normalise a date-ish column to ``datetime64[D]``."""
    return pd.to_datetime(pd.Series(values).values).values.astype("datetime64[D]")


@dataclass(frozen=True)
class PanelAxes:
    """The shared coordinate system every cached artefact is aligned to.

    Attributes:
        dates: Sorted trading dates, ``datetime64[D]``.
        sids: Sorted security ids, ``int64``.
        times: Intraday bar labels kept per day (the empty-by-construction
            bars are dropped, see :meth:`IntradayPanel.times`).
    """

    dates: np.ndarray
    sids: np.ndarray
    times: Sequence[dt.time]

    @property
    def date_index(self) -> Dict[np.datetime64, int]:
        return {d: i for i, d in enumerate(self.dates)}

    @property
    def sid_index(self) -> Dict[int, int]:
        return {int(s): i for i, s in enumerate(self.sids)}


class DailyPanel:
    """The whole daily OHLCV panel held densely in memory.

    A decade of A-share dailies is only a few hundred megabytes once densified,
    so loading it once and indexing by position beats repeated parquet reads.
    """

    def __init__(self, path: str, columns: Sequence[str] = OHLCV) -> None:
        self.path = path
        self.columns = tuple(columns)
        frame = _flatten(pq.read_table(path).to_pandas())
        missing = {"date", "stable_id", *self.columns} - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: daily panel is missing columns {sorted(missing)}")

        frame["date"] = _as_date_array(frame["date"])
        # Assigning into a DataFrame promotes dates to datetime64[ns]; force them
        # back to day precision so this axis compares equal to the intraday one.
        self.dates = np.unique(frame["date"].values).astype("datetime64[D]")
        self.sids = np.unique(frame["stable_id"].values).astype(np.int64)

        d_pos = pd.Series(np.arange(len(self.dates)), index=self.dates)
        s_pos = pd.Series(np.arange(len(self.sids)), index=self.sids)
        rows = d_pos.reindex(frame["date"].values).values
        cols = s_pos.reindex(frame["stable_id"].values).values

        self.values = np.full((len(self.dates), len(self.sids), len(self.columns)), np.nan, np.float32)
        for k, name in enumerate(self.columns):
            self.values[rows, cols, k] = frame[name].values.astype(np.float32)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"DailyPanel(dates={len(self.dates)}, sids={len(self.sids)}, "
            f"columns={self.columns})"
        )

    def day(self, date: np.datetime64) -> np.ndarray:
        """``(n_sids, n_columns)`` slice for one date."""
        return self.values[int(np.searchsorted(self.dates, date))]

    def close_matrix(self) -> np.ndarray:
        """``(n_dates, n_sids)`` closing prices, for label construction."""
        return self.values[:, :, self.columns.index("close")]

    def dollar_volume_matrix(self) -> np.ndarray:
        """``(n_dates, n_sids)`` close x volume, a liquidity proxy for universe screens.

        Memoised: the universe screen asks for this once per sample date, and
        recomputing a 2400 x 5700 product each time dominated the cache build.
        """
        if getattr(self, "_dollar_volume", None) is None:
            close = self.values[:, :, self.columns.index("close")]
            volume = self.values[:, :, self.columns.index("volume")]
            self._dollar_volume = close * volume
        return self._dollar_volume


class IntradayPanel:
    """Row-group-at-a-time reader for the 5-minute panel.

    The full file is far too large to hold densely, so dates are read on demand
    and returned aligned to a caller-supplied security axis.  The cache builder
    walks dates in order and keeps only a short rolling buffer.
    """

    def __init__(
        self,
        path: str,
        sids: np.ndarray,
        columns: Sequence[str] = OHLCV,
        drop_empty_times: bool = True,
        n_probe_dates: int = 8,
    ) -> None:
        self.path = path
        self.columns = tuple(columns)
        self.sids = np.asarray(sids, dtype=np.int64)
        self._sid_pos = pd.Series(np.arange(len(self.sids)), index=self.sids)
        self._file = pq.ParquetFile(path)
        self._row_group_of_date = self._index_row_groups()
        self.times: List[dt.time] = self._probe_times(drop_empty_times, n_probe_dates)
        self._time_pos = {t: i for i, t in enumerate(self.times)}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"IntradayPanel(dates={len(self._row_group_of_date)}, bars_per_day={len(self.times)})"

    @property
    def dates(self) -> np.ndarray:
        """Sorted dates present in the file."""
        return np.array(sorted(self._row_group_of_date), dtype="datetime64[D]")

    def _probe_times(self, drop_empty: bool, n_probe_dates: int) -> List[dt.time]:
        """Work out the intraday grid, keeping bars that trade on *any* probe date.

        A single probe date is not safe: 2016-01-04 is the first circuit-breaker
        day and halted at 13:34, so probing it alone would silently delete every
        afternoon bar from the whole history.  Several dates spread across the
        file are enough to see the real grid, while still keeping labels that are
        empty by construction on every date — the 09:30 opening call auction this
        vendor never populates — out of the path.
        """
        keys = sorted(self._row_group_of_date)
        picks = keys if len(keys) <= n_probe_dates else [
            keys[round(i * (len(keys) - 1) / (n_probe_dates - 1))] for i in range(n_probe_dates)
        ]

        all_times: set = set()
        traded: set = set()
        for date in picks:
            frame = self._read_row_groups(self._row_group_of_date[date])
            all_times |= set(frame["time"].unique())
            if drop_empty:
                filled = frame.groupby("time")["close"].apply(lambda s: s.notna().any())
                traded |= {t for t, ok in filled.items() if bool(ok)}
        return sorted(traded if drop_empty else all_times)

    def _index_row_groups(self) -> Dict[np.datetime64, List[int]]:
        """Map each date to the row groups holding it, from parquet statistics.

        Falls back to scanning the date column when the writer stored no
        statistics, which is slower but keeps the reader portable.
        """
        mapping: Dict[np.datetime64, List[int]] = {}
        metadata = self._file.metadata
        date_col = self._file.schema_arrow.names.index("date")
        for rg in range(self._file.num_row_groups):
            stats = metadata.row_group(rg).column(date_col).statistics
            if stats is None or stats.min is None:
                mapping.clear()
                break
            lo, hi = _as_date_array([stats.min])[0], _as_date_array([stats.max])[0]
            for date in np.arange(lo, hi + np.timedelta64(1, "D")):
                mapping.setdefault(np.datetime64(date, "D"), []).append(rg)
        if not mapping:
            dates = _as_date_array(self._file.read(columns=["date"]).column("date").to_pandas())
            for date in np.unique(dates):
                mapping[np.datetime64(date, "D")] = list(range(self._file.num_row_groups))
        return mapping

    def _read_row_groups(self, row_groups: Sequence[int]) -> pd.DataFrame:
        table = self._file.read_row_groups(list(row_groups))
        return _flatten(table.to_pandas())

    def day(self, date: np.datetime64) -> Optional[np.ndarray]:
        """Dense intraday slice for one date.

        Args:
            date: The trading date to read.

        Returns:
            ``(n_times, n_sids, n_columns)`` float32 with ``NaN`` where the
            security did not trade, or ``None`` if the date is absent.
        """
        date = np.datetime64(date, "D")
        row_groups = self._row_group_of_date.get(date)
        if not row_groups:
            return None
        frame = self._read_row_groups(row_groups)
        frame["date"] = _as_date_array(frame["date"])
        frame = frame[frame["date"] == date]
        if frame.empty:
            return None

        out = np.full((len(self.times), len(self.sids), len(self.columns)), np.nan, np.float32)
        rows = frame["time"].map(self._time_pos)
        cols = self._sid_pos.reindex(frame["stable_id"].values).values
        keep = rows.notna().values & ~pd.isna(cols)
        rows = rows.values[keep].astype(np.int64)
        cols = cols[keep].astype(np.int64)
        for k, name in enumerate(self.columns):
            out[rows, cols, k] = frame[name].values[keep].astype(np.float32)
        return out
