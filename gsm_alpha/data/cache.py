"""Precompute per-date model inputs in a single streaming pass over the panels.

Two things make this worth doing once instead of inside the training loop.

*Decoding cost.* A 20-day intraday window is ~980 five-minute bars per name;
assembling it from parquet for every sample would re-read the same row group
twenty times.  Walking dates in order with a rolling buffer reads each date
exactly once.

*Reuse.* The rolling retrain schedule fits one model per year over a four-year
window, so every date is consumed by about four fits, and every ablation
re-consumes all of them.

Two artefacts can be written, independently:

``windows``
    Normalised OHLCV windows, ``(n_sids, n_steps, 5)`` float16.  Needed only by
    a *learnable* augmentation such as multi-headed stream-preserving, whose
    signature features change as the network trains.

``features``
    GSM output, ``(n_sids, n_features)`` float32.  Valid only for a fixed
    augmentation (coordinate projections and friends), which is the report's
    preferred setting and the default here.  Training reads these directly and
    the GSM stage disappears from the training loop entirely.

Both are keyed by date, one ``.npz`` per date per branch, with the security ids
stored alongside so nothing depends on a global row order.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..config import Config, GSMConfig
from ..signature import GSM, backend_report
from .preprocess import build_window
from .sources import DailyPanel, IntradayPanel

logger = logging.getLogger(__name__)

BRANCHES = ("minute", "daily")


def build_gsm(config: GSMConfig, in_channels: int = 5) -> GSM:
    """Instantiate a :class:`~gsm_alpha.signature.gsm.GSM` from its config section."""
    return GSM(
        in_channels=in_channels,
        depth=config.depth,
        augmentations=config.augmentations,
        transform=config.transform,
        window=config.window,
        window_size=config.window_size,
        window_step=config.window_step,
        window_depth=config.window_depth,
        rescaling=config.rescaling,
        backend=config.backend,
        projection_size=config.projection_size,
        projection_ordered=config.projection_ordered,
        projection_out=config.projection_out,
        n_projections=config.n_projections,
        mhsp_hidden=config.mhsp_hidden,
    )


def _cast_features(feats: np.ndarray, dtype: str, branch: str, date) -> np.ndarray:
    """Narrow the feature dtype, refusing a cast that would lose values to infinity.

    float16 halves the cache and the memory a preloaded split needs, which is
    tempting for GPU training.  It is also unsafe here without checking: a
    depth-5 log-signature over a 960-step path reaches into the hundreds of
    thousands, and float16 overflows at 65 504.  Values that tip over become
    ``inf``, the network then emits ``NaN`` on those dates, and the damage shows
    up hours later as an empty validation metric rather than as an error.  So
    the cast is verified rather than assumed.
    """
    if dtype == "float32":
        return feats
    if dtype != "float16":
        raise ValueError(f"unknown data.feature_dtype {dtype!r}, expected float32 or float16")

    limit = float(np.finfo(np.float16).max)
    peak = float(np.abs(feats).max()) if feats.size else 0.0
    if peak > limit:
        raise ValueError(
            f"data.feature_dtype='float16' cannot hold the {branch} features on {date}: "
            f"largest magnitude is {peak:,.0f} but float16 overflows at {limit:,.0f}, which "
            "would silently become inf and poison training. Use float32 (the default); to "
            "cut memory instead, raise data.preload_max_gb or lower the universe."
        )
    return feats.astype(np.float16)


def branch_dir(cache_dir: str, kind: str, branch: str) -> str:
    """Directory holding one artefact kind for one branch."""
    return os.path.join(cache_dir, kind, branch)


def date_file(cache_dir: str, kind: str, branch: str, date: np.datetime64) -> str:
    """Path of the cached ``.npz`` for one date."""
    stamp = str(np.datetime64(date, "D")).replace("-", "")
    return os.path.join(branch_dir(cache_dir, kind, branch), f"{stamp}.npz")


@dataclass
class CachePlan:
    """What a cache build is about to do, resolved from the config.

    Attributes:
        sample_dates: Dates a cache entry is written for.
        warmup_dates: Earlier dates that must be streamed to fill the buffers.
        minute_steps: Intraday steps per sample.
        daily_steps: Daily steps per sample.
        feature_dims: Feature width per branch, when features are being written.
    """

    sample_dates: np.ndarray
    warmup_dates: np.ndarray
    minute_steps: int
    daily_steps: int
    feature_dims: Dict[str, int]


class CacheBuilder:
    """Streams the panels once and writes the per-date cache."""

    def __init__(self, config: Config, threads: Optional[int] = None) -> None:
        self.config = config
        self.data = config.data
        if threads:
            torch.set_num_threads(threads)

        logger.info("loading daily panel from %s", self.data.daily_path)
        self.daily = DailyPanel(self.data.daily_path)
        logger.info("%r", self.daily)

        self.intraday: Optional[IntradayPanel] = None
        if self.data.use_minute_branch:
            logger.info("indexing intraday panel at %s", self.data.intraday_path)
            self.intraday = IntradayPanel(self.data.intraday_path, self.daily.sids)
            logger.info("%r", self.intraday)

        self.gsm: Dict[str, GSM] = {}
        if self.data.use_minute_branch:
            self.gsm["minute"] = build_gsm(config.minute_gsm).eval()
        if self.data.use_daily_branch:
            self.gsm["daily"] = build_gsm(config.daily_gsm).eval()
        logger.info("%s", backend_report(config.minute_gsm.backend))

    # -- planning ---------------------------------------------------------

    def plan(self, stride: Optional[int] = None) -> CachePlan:
        """Resolve which dates are sampled and how wide each branch's window is.

        Args:
            stride: Override for ``data.predict_day_stride``.  The cache is
                built on the finest stride any split needs, and the training
                loader subsamples from it.

        Returns:
            The resolved :class:`CachePlan`.
        """
        stride = stride or 1
        dates = self.daily.dates
        if self.intraday is not None:
            dates = np.array(sorted(set(dates.tolist()) & set(self.intraday.dates.tolist())),
                             dtype="datetime64[D]")

        start = np.datetime64(self.data.start_date, "D")
        end = np.datetime64(self.data.end_date, "D")
        warmup = max(
            self.data.minute_lookback_days if self.data.use_minute_branch else 0,
            self.data.daily_lookback_days if self.data.use_daily_branch else 0,
            self.data.universe_lookback_days if self.data.universe_top_n else 0,
        )
        first_idx = int(np.searchsorted(dates, start))
        if first_idx < warmup - 1:
            logger.warning(
                "start_date %s leaves only %d days of history but %d are needed; "
                "the first samples move later",
                self.data.start_date, first_idx + 1, warmup,
            )
            first_idx = warmup - 1
        last_idx = int(np.searchsorted(dates, end, side="right"))

        candidates = dates[first_idx:last_idx]
        sample_dates = candidates[::stride]
        warmup_dates = dates[max(0, first_idx - warmup + 1):first_idx]

        minute_steps = (
            self.data.minute_lookback_days * len(self.intraday.times)
            if self.intraday is not None
            else 0
        )
        feature_dims = {
            branch: gsm.out_features(minute_steps if branch == "minute" else self.data.daily_lookback_days)
            for branch, gsm in self.gsm.items()
        }
        return CachePlan(
            sample_dates=sample_dates,
            warmup_dates=warmup_dates,
            minute_steps=minute_steps,
            daily_steps=self.data.daily_lookback_days,
            feature_dims=feature_dims,
        )

    # -- universe ---------------------------------------------------------

    def _liquidity_rank_mask(self, upto_idx: int) -> Optional[np.ndarray]:
        """Top-N liquidity screen using only information available on that date.

        Args:
            upto_idx: Positional index of the sample date in ``daily.dates``.

        Returns:
            ``(n_sids,)`` boolean mask, or ``None`` when no screen is configured.
        """
        top_n = self.data.universe_top_n
        if not top_n:
            return None
        lo = max(0, upto_idx - self.data.universe_lookback_days + 1)
        window = self.daily.dollar_volume_matrix()[lo:upto_idx + 1]
        with warnings.catch_warnings():
            # Names not yet listed are all-NaN over the window; they score -1 below.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            score = np.nanmedian(np.where(window > 0, window, np.nan), axis=0)
        score = np.nan_to_num(score, nan=-1.0)
        if (score > 0).sum() <= top_n:
            return score > 0
        cutoff = np.partition(score, -top_n)[-top_n]
        return score >= cutoff

    # -- the streaming pass -----------------------------------------------

    def run(
        self,
        stage: str = "features",
        stride: Optional[int] = None,
        overwrite: bool = False,
        progress_every: int = 20,
    ) -> CachePlan:
        """Build the cache.

        Args:
            stage: ``"features"``, ``"windows"`` or ``"both"``.
            stride: Sample every n-th trading day; defaults to every day.
            overwrite: Rebuild dates that already have a cache file.
            progress_every: Log a progress line every this many dates.

        Returns:
            The :class:`CachePlan` that was executed, also written to
            ``<cache_dir>/manifest.json``.
        """
        if stage not in ("features", "windows", "both"):
            raise ValueError(f"unknown cache stage {stage!r}")
        want_features = stage in ("features", "both")
        want_windows = stage in ("windows", "both")

        if want_features:
            for branch, gsm in self.gsm.items():
                if gsm.is_learnable:
                    raise ValueError(
                        f"the {branch} branch uses a learnable augmentation "
                        f"({self.config.minute_gsm.augmentations}), whose features change during "
                        "training and cannot be cached. Build with stage='windows' and train with "
                        "data cache mode 'windows'."
                    )

        plan = self.plan(stride=stride)
        for kind in (["features"] if want_features else []) + (["windows"] if want_windows else []):
            for branch in self.gsm:
                os.makedirs(branch_dir(self.data.cache_dir, kind, branch), exist_ok=True)

        # Keep numpy scalars on both sides: ndarray.tolist() yields datetime.date,
        # which never compares equal to the np.datetime64 the loop carries.
        sample_set = {np.datetime64(d, "D") for d in plan.sample_dates}
        stream = np.concatenate([plan.warmup_dates, plan.sample_dates])
        # Dates skipped by the stride still have to be streamed: they are inside
        # later samples' lookback windows.
        all_dates = self.daily.dates
        lo = int(np.searchsorted(all_dates, stream[0]))
        hi = int(np.searchsorted(all_dates, stream[-1], side="right"))
        stream = all_dates[lo:hi]

        minute_buf: Deque[np.ndarray] = deque(maxlen=self.data.minute_lookback_days)
        daily_buf: Deque[np.ndarray] = deque(maxlen=self.data.daily_lookback_days)

        written = skipped = 0
        for n, date in enumerate(stream):
            date = np.datetime64(date, "D")
            idx = int(np.searchsorted(all_dates, date))

            if self.intraday is not None:
                day = self.intraday.day(date)
                if day is None:  # a date the daily panel has and the intraday one lacks
                    minute_buf.clear()
                else:
                    minute_buf.append(day)
            daily_buf.append(self.daily.values[idx])

            if date not in sample_set:
                continue
            if self.data.use_minute_branch and len(minute_buf) < minute_buf.maxlen:
                continue
            if self.data.use_daily_branch and len(daily_buf) < daily_buf.maxlen:
                continue
            if not overwrite and self._already_cached(date, want_features, want_windows):
                skipped += 1
                continue

            self._emit_date(date, idx, minute_buf, daily_buf, want_features, want_windows)
            written += 1
            if progress_every and written % progress_every == 0:
                logger.info("cached %d dates (latest %s, %d skipped)", written, date, skipped)

        self._write_manifest(plan, stage)
        logger.info("cache build finished: %d dates written, %d already present", written, skipped)
        return plan

    def _already_cached(self, date: np.datetime64, features: bool, windows: bool) -> bool:
        kinds = (["features"] if features else []) + (["windows"] if windows else [])
        return all(
            os.path.exists(date_file(self.data.cache_dir, kind, branch, date))
            for kind in kinds
            for branch in self.gsm
        )

    def _emit_date(
        self,
        date: np.datetime64,
        idx: int,
        minute_buf: Sequence[np.ndarray],
        daily_buf: Sequence[np.ndarray],
        want_features: bool,
        want_windows: bool,
    ) -> None:
        """Normalise, screen and write one date's cache entries.

        The liquidity screen is applied to the security axis *before*
        normalisation, not after.  Filling and z-scoring are per-security and
        independent across securities, so the retained rows come out identical
        either way — but doing it first stops the pass from touching the ~4 000
        names that are about to be discarded, and normalisation is the single
        most expensive step in the build (it is memory-bandwidth bound and does
        not thread).
        """
        preselect = self._liquidity_rank_mask(idx)
        if preselect is None:
            preselect = np.ones(len(self.daily.sids), dtype=bool)
        elif not preselect.any():
            logger.warning("no securities pass the liquidity screen on %s, skipping", date)
            return

        prepared: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if self.data.use_minute_branch:
            raw = np.concatenate(list(minute_buf), axis=0)  # (days*bars, n_sids, 5)
            prepared["minute"] = build_window(
                raw[:, preselect], self.data.price_zscore, self.data.min_valid_fraction
            )
        if self.data.use_daily_branch:
            raw = np.stack(list(daily_buf), axis=0)  # (days, n_sids, 5)
            prepared["daily"] = build_window(
                raw[:, preselect], self.data.price_zscore, self.data.min_valid_fraction
            )

        mask = np.ones(int(preselect.sum()), dtype=bool)
        for _, branch_mask in prepared.values():
            mask &= branch_mask
        if not mask.any():
            logger.warning("no valid securities on %s, skipping", date)
            return

        sids = self.daily.sids[preselect][mask]
        for branch, (samples, _) in prepared.items():
            kept = samples[mask]
            if want_windows:
                np.savez(
                    date_file(self.data.cache_dir, "windows", branch, date),
                    sid=sids,
                    window=kept.astype(np.float16),
                )
            if want_features:
                with torch.no_grad():
                    feats = self.gsm[branch](torch.from_numpy(kept)).numpy().astype(np.float32)
                feats = _cast_features(feats, self.data.feature_dtype, branch, date)
                np.savez(
                    date_file(self.data.cache_dir, "features", branch, date),
                    sid=sids,
                    feature=feats,
                )

    def _write_manifest(self, plan: CachePlan, stage: str) -> None:
        """Record what the cache contains so training can validate its inputs."""
        os.makedirs(self.data.cache_dir, exist_ok=True)
        path = os.path.join(self.data.cache_dir, "manifest.json")
        existing: Dict = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
        existing.update(
            {
                "stage": stage,
                "minute_steps": plan.minute_steps,
                "daily_steps": plan.daily_steps,
                "feature_dims": plan.feature_dims,
                "minute_gsm": self.config.minute_gsm.__dict__,
                "daily_gsm": self.config.daily_gsm.__dict__,
                "price_zscore": self.data.price_zscore,
                "minute_lookback_days": self.data.minute_lookback_days,
                "daily_lookback_days": self.data.daily_lookback_days,
                "universe_top_n": self.data.universe_top_n,
                "feature_dtype": self.data.feature_dtype,
            }
        )
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(existing, handle, indent=2, default=str)


def read_manifest(cache_dir: str) -> Dict:
    """Load the cache manifest, with a clear error when the cache is missing."""
    path = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no cache manifest at {path}; run `python -m gsm_alpha.cli build-cache` first"
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def available_dates(cache_dir: str, kind: str, branch: str) -> List[np.datetime64]:
    """Dates present in the cache for one artefact kind and branch."""
    directory = branch_dir(cache_dir, kind, branch)
    if not os.path.isdir(directory):
        return []
    out = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".npz"):
            stem = name[:-4]
            out.append(np.datetime64(f"{stem[:4]}-{stem[4:6]}-{stem[6:]}", "D"))
    return out
