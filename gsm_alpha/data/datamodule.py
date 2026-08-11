"""Day-batched datasets and the LightningDataModule.

Report section 3.2: "每个交易日的所有股票数据作为一个 batch".  One batch is one
trading day's whole cross section, which is what stock mixing attends over and
what the weighted-correlation loss is defined on.  So the PyTorch batch size is
literally 1 — one *date* — and the stock axis lives inside the sample.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from .cache import available_dates, date_file, read_manifest

logger = logging.getLogger(__name__)


class DayBatchDataset(Dataset):
    """One item per trading date: every valid stock's inputs plus its label.

    Args:
        cache_dir: Root of the precomputed cache.
        kind: ``"features"`` or ``"windows"``.
        branches: Branch names to load, e.g. ``["daily", "minute"]``.
        dates: Dates this split covers.
        labels: ``(n_dates_all, n_sids_all)`` label matrix.
        date_index: Position of each date in ``labels``' first axis.
        sid_index: Position of each security id in ``labels``' second axis.
        require_label: Drop stocks with no label (training and validation) or
            keep them (prediction, where the future is not yet known).
    """

    def __init__(
        self,
        cache_dir: str,
        kind: str,
        branches: Sequence[str],
        dates: Sequence[np.datetime64],
        labels: Optional[np.ndarray],
        date_index: Dict[np.datetime64, int],
        sid_index: Dict[int, int],
        require_label: bool = True,
        preload: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        self.kind = kind
        self.branches = list(branches)
        self.dates = list(dates)
        self.labels = labels
        self.date_index = date_index
        self.sid_index = sid_index
        self.require_label = require_label
        self._payload = "feature" if kind == "features" else "window"
        self._memory: Optional[List[Dict]] = None
        if preload:
            self._preload()

    def nbytes(self) -> int:
        """Bytes this split would occupy in memory, read from the file sizes."""
        total = 0
        for date in self.dates:
            for branch in self.branches:
                path = date_file(self.cache_dir, self.kind, branch, date)
                if os.path.exists(path):
                    total += os.path.getsize(path)
        return total

    def _preload(self) -> None:
        """Read the whole split into memory once.

        A day-batch is tens of megabytes and a GPU step is milliseconds, so
        reading per batch would leave the device waiting on disk for most of the
        run.  Loading once up front turns every later epoch into pure compute.
        """
        logger.info(
            "preloading %d dates (%s, %s) into memory (~%.1f GB on disk)",
            len(self.dates), self.kind, "+".join(self.branches), self.nbytes() / 1e9,
        )
        self._memory = [self._read(date) for date in self.dates]

    def __len__(self) -> int:
        return len(self.dates)

    def _read(self, date) -> Dict:
        """Load one date's arrays and label from the cache."""
        arrays: Dict[str, np.ndarray] = {}
        sids: Optional[np.ndarray] = None
        for branch in self.branches:
            with np.load(date_file(self.cache_dir, self.kind, branch, date)) as blob:
                branch_sids = blob["sid"]
                values = blob[self._payload]
            if sids is None:
                sids = branch_sids
            elif not np.array_equal(sids, branch_sids):
                # Both branches are screened with the same mask, so this only
                # fires if the two caches were built from different configs.
                keep_a = np.isin(sids, branch_sids)
                keep_b = np.isin(branch_sids, sids)
                for name in arrays:
                    arrays[name] = arrays[name][keep_a]
                sids = sids[keep_a]
                values = values[keep_b]
            arrays[branch] = values
        assert sids is not None

        label = np.full(len(sids), np.nan, dtype=np.float32)
        if self.labels is not None:
            row = self.labels[self.date_index[date]]
            positions = np.array([self.sid_index.get(int(s), -1) for s in sids])
            found = positions >= 0
            label[found] = row[positions[found]]

        if self.require_label:
            keep = np.isfinite(label)
            if keep.sum() < 3:
                keep = np.ones(len(sids), dtype=bool)  # degenerate day; loss skips it
            sids = sids[keep]
            label = label[keep]
            arrays = {k: v[keep] for k, v in arrays.items()}

        item: Dict = {
            "date": np.datetime64(date, "D"),
            "sid": sids.astype(np.int64),
            "label": torch.from_numpy(label),
        }
        for branch, values in arrays.items():
            item[branch] = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
        return item

    def __getitem__(self, index: int) -> Dict:
        if self._memory is not None:
            return self._memory[index]
        return self._read(self.dates[index])


def collate_single_day(batch: List[Dict]) -> Dict:
    """Unwrap the one-item list a ``batch_size=1`` loader produces."""
    if len(batch) != 1:
        raise ValueError(f"day batches must be loaded one at a time, got {len(batch)}")
    return batch[0]


def split_dates(
    dates: Sequence[np.datetime64],
    predict_year: int,
    lookback_years: int,
    validation_years: int,
    drop_last_month: bool,
) -> Dict[str, List[np.datetime64]]:
    """Cut the rolling train/validation/predict windows of report section 3.2.

    "从 2018 年开始逐年滚动训练模型，训练模型时向过去回溯 4 年，前三年剔除末月为
     训练集，后一年剔除末月为验证集。"  Dropping each split's final month keeps a
    20-day forward label from reaching past the split boundary.

    Args:
        dates: All candidate sample dates, sorted.
        predict_year: The year this model will generate factor values for.
        lookback_years: Calendar years of history per fit (4).
        validation_years: How many of those form the validation split (1).
        drop_last_month: Whether to drop each split's final month.

    Returns:
        ``{"train": [...], "val": [...], "predict": [...]}``.
    """
    dates = np.asarray(dates, dtype="datetime64[D]")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1

    train_lo = predict_year - lookback_years
    train_hi = predict_year - validation_years - 1  # inclusive
    val_lo = predict_year - validation_years
    val_hi = predict_year - 1

    train = (years >= train_lo) & (years <= train_hi)
    val = (years >= val_lo) & (years <= val_hi)
    if drop_last_month:
        train &= ~((years == train_hi) & (months == 12))
        val &= ~((years == val_hi) & (months == 12))
    predict = years == predict_year

    return {
        "train": list(dates[train]),
        "val": list(dates[val]),
        "predict": list(dates[predict]),
    }


class GSMAlphaDataModule(pl.LightningDataModule):
    """Serves one rolling fit: its train, validation and prediction splits.

    Args:
        config: The full pipeline config.
        labels: ``(n_dates_all, n_sids_all)`` label matrix from
            :class:`~gsm_alpha.data.labels.LabelBuilder`.
        all_dates: The date axis of ``labels``.
        all_sids: The security axis of ``labels``.
        predict_year: Year this fit targets.
        kind: ``"features"`` or ``"windows"``.
    """

    def __init__(
        self,
        config: Config,
        labels: np.ndarray,
        all_dates: np.ndarray,
        all_sids: np.ndarray,
        predict_year: int,
        kind: str = "features",
    ) -> None:
        super().__init__()
        self.config = config
        self.labels = labels
        self.kind = kind
        self.predict_year = predict_year
        self.date_index = {np.datetime64(d, "D"): i for i, d in enumerate(all_dates)}
        self.sid_index = {int(s): i for i, s in enumerate(all_sids)}

        self.branches = [
            b
            for b, on in (("daily", config.data.use_daily_branch),
                          ("minute", config.data.use_minute_branch))
            if on
        ]
        if not self.branches:
            raise ValueError("at least one of use_daily_branch / use_minute_branch must be on")

        cached = set(available_dates(config.data.cache_dir, kind, self.branches[0]))
        for branch in self.branches[1:]:
            cached &= set(available_dates(config.data.cache_dir, kind, branch))
        if not cached:
            raise FileNotFoundError(
                f"cache at {config.data.cache_dir} has no '{kind}' entries for branches "
                f"{self.branches}; run `python -m gsm_alpha.cli build-cache` first"
            )

        usable = sorted(cached & set(self.date_index))
        self.splits = split_dates(
            usable,
            predict_year,
            config.train.lookback_years,
            config.train.validation_years,
            config.train.drop_last_month,
        )
        stride = max(1, config.data.train_day_stride)
        self.splits["train"] = self.splits["train"][::stride]
        self.splits["val"] = self.splits["val"][::stride]
        self.splits["predict"] = self.splits["predict"][:: max(1, config.data.predict_day_stride)]
        logger.info(
            "predict_year=%d splits: train=%d val=%d predict=%d dates",
            predict_year, len(self.splits["train"]), len(self.splits["val"]),
            len(self.splits["predict"]),
        )

    def _should_preload(self, dataset: DayBatchDataset) -> bool:
        """Whether to hold this split in RAM.

        ``auto`` preloads when the split fits the configured budget.  This is
        what keeps a GPU fed: a day-batch at full market is tens of megabytes
        and a GPU step is milliseconds, so reading per batch would make the run
        disk-bound rather than compute-bound.
        """
        mode = self.config.data.preload
        if mode == "never":
            return False
        if mode == "always":
            return True
        if mode != "auto":
            raise ValueError(f"unknown data.preload mode {mode!r}")
        budget = self.config.data.preload_max_gb * 1e9
        size = dataset.nbytes()
        if size <= budget:
            return True
        logger.info(
            "not preloading: split needs %.1f GB but data.preload_max_gb is %.1f GB; "
            "reads stay per batch, so raise the budget or use feature_dtype: float16 "
            "if the device ends up waiting on disk",
            size / 1e9, self.config.data.preload_max_gb,
        )
        return False

    def _dataset(self, split: str, require_label: bool) -> DayBatchDataset:
        dataset = DayBatchDataset(
            self.config.data.cache_dir,
            self.kind,
            self.branches,
            self.splits[split],
            self.labels,
            self.date_index,
            self.sid_index,
            require_label=require_label,
        )
        if self.splits[split] and self._should_preload(dataset):
            dataset._preload()  # noqa: SLF001 - same module, deliberate two-phase init
        return dataset

    def _loader(self, split: str, shuffle: bool, require_label: bool = True) -> DataLoader:
        dataset = self._dataset(split, require_label)
        # Workers exist to overlap IO with compute; once the split is resident
        # they only add per-batch pickling, so they are turned off.
        workers = 0 if dataset._memory is not None else self.config.train.num_workers
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=collate_single_day,
            persistent_workers=workers > 0,
            # Pinning only helps a real host-to-device copy; on CPU it is pure cost.
            pin_memory=(
                torch.cuda.is_available() and self.config.train.accelerator != "cpu"
            ),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._loader("predict", shuffle=False, require_label=False)

    def feature_dims(self) -> Dict[str, int]:
        """Width of each branch's cached feature vector, read from the cache itself."""
        if self.kind != "features":
            manifest = read_manifest(self.config.data.cache_dir)
            return {b: int(manifest["feature_dims"][b]) for b in self.branches}
        dims = {}
        for branch in self.branches:
            date = self.splits["train"][0] if self.splits["train"] else self.splits["predict"][0]
            with np.load(date_file(self.config.data.cache_dir, "features", branch, date)) as blob:
                dims[branch] = int(blob["feature"].shape[1])
        return dims
