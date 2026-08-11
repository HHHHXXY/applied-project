"""Typed configuration for the whole pipeline, loaded from YAML.

Every path, window length and hyper-parameter the report specifies lives here,
so a deployment with the same data contract only edits ``configs/*.yaml``.
Unknown keys are rejected rather than ignored — a silently misspelled option is
the kind of bug that only shows up as a disappointing IC three hours later.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, get_type_hints

import yaml


@dataclass
class DataConfig:
    """Where the panels live and how samples are cut out of them.

    Attributes:
        daily_path: Parquet of daily OHLCV keyed by ``(date, stable_id)``.
        intraday_path: Parquet of intraday OHLCV keyed by ``(date, time, stable_id)``.
        cache_dir: Directory for the precomputed feature/window cache.
        start_date: First sample date (inclusive), ``YYYY-MM-DD``.
        end_date: Last sample date (inclusive).
        minute_lookback_days: Trading days in the intraday branch's window (20).
        daily_lookback_days: Trading days in the daily branch's window (60).
        train_day_stride: Keep every n-th trading day as a training sample.  The
            label is a 20-day forward return, so consecutive daily samples
            overlap almost entirely; 1 reproduces the report, larger values cut
            cost with little information loss.
        predict_day_stride: Same, for the dates a factor value is emitted on.
        universe_top_n: Keep the ``n`` most liquid names per date by trailing
            median dollar volume; ``None`` keeps everything.
        universe_lookback_days: Window for that liquidity ranking.
        min_valid_fraction: Minimum share of non-missing bars for a usable sample.
        price_zscore: ``joint`` or ``per_channel`` (see :mod:`.data.preprocess`).
        use_minute_branch: Include the intraday branch (GSM-Alpha vs the
            daily-only ablation).
        use_daily_branch: Include the daily branch (off gives GSM-Alpha-min).
        preload: ``"auto"``, ``"always"`` or ``"never"`` — hold a split's whole
            feature set in RAM instead of reading one file per batch.  Essential
            on a GPU, where a training step is milliseconds and per-batch disk
            reads would starve the device.  ``"auto"`` preloads when the split
            fits in ``preload_max_gb``.
        preload_max_gb: Budget for that decision.
        feature_dtype: ``"float32"`` or ``"float16"`` for the cache on disk.
            float16 halves both the cache and the memory a preloaded split needs;
            features are network inputs that get projected immediately, so the
            ~1e-3 relative rounding is immaterial.
    """

    daily_path: str = ""
    intraday_path: str = ""
    cache_dir: str = ""
    start_date: str = "2017-01-01"
    end_date: str = "2024-12-31"
    minute_lookback_days: int = 20
    daily_lookback_days: int = 60
    train_day_stride: int = 5
    predict_day_stride: int = 1
    universe_top_n: Optional[int] = 1500
    universe_lookback_days: int = 60
    min_valid_fraction: float = 0.5
    price_zscore: str = "joint"
    use_minute_branch: bool = True
    use_daily_branch: bool = True
    preload: str = "auto"
    preload_max_gb: float = 24.0
    feature_dtype: str = "float32"


@dataclass
class GSMConfig:
    """One GSM branch: augmentation, window, transform, rescaling (report section 2)."""

    depth: int = 5
    augmentations: List[str] = field(
        default_factory=lambda: ["coordinate_projection", "time", "basepoint"]
    )
    transform: str = "logsignature"
    window: str = "global"
    window_size: int = 0
    window_step: int = 0
    window_depth: int = 3
    rescaling: str = "none"
    projection_size: int = 2
    projection_ordered: bool = False
    projection_out: int = 3
    n_projections: int = 10
    mhsp_hidden: int = 32
    backend: str = "auto"


@dataclass
class ModelConfig:
    """GSM-Alpha network hyper-parameters (report section 3.1, figure 4).

    Attributes:
        hidden_dim: Width each branch's log-signature block is projected to.
        n_attention_heads: Heads in the stock-mixing self-attention.
        attention_dropout: Dropout inside stock mixing.
        use_stock_mixing: Off reproduces the table 9 ablation.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        loss_halflife_fraction: Exponential decay half-life of the loss weights,
            as a fraction of the batch size (the report uses one half).
        input_transform: What to do to each branch's features before the first
            ``Linear``. ``none`` is the report's own setting and is faithful but
            does not train on this data — the raw depth-5 log-signature is heavy
            tailed enough to blow the model up inside one epoch, in fp32 as well
            as fp16. ``gauss_rank`` applies a cross-sectional normal-score
            transform per feature; see :mod:`gsm_alpha.models.transforms`.
        architecture: ``gsm_alpha`` (the report) or ``stockmixer`` (the AAAI-24
            paper it borrows from, as the ablation baseline that isolates what
            the signature contributes). ``stockmixer`` requires raw windows.
        n_indicators: ``F`` for the StockMixer arm; the OHLCV panel gives 5.
        time_scales: StockMixer's average-pooling kernels ``k`` (paper: 1, 2, 4).
        n_market_states: StockMixer's ``m``, the market-state bottleneck. Left at
            a round 32 rather than grid-searched, because the GSM-Alpha arms are
            not tuned either and tuning only the baseline would flatter it.
    """

    hidden_dim: int = 128
    n_attention_heads: int = 4
    attention_dropout: float = 0.1
    use_stock_mixing: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    loss_halflife_fraction: float = 0.5
    input_transform: str = "none"
    architecture: str = "gsm_alpha"
    n_indicators: int = 5
    time_scales: Tuple[int, ...] = (1, 2, 4)
    n_market_states: int = 32


@dataclass
class LabelSettings:
    """Forward-return target settings; see :class:`gsm_alpha.data.labels.LabelConfig`."""

    horizon: int = 20
    forward_label_path: Optional[str] = None
    forward_label_column: Optional[str] = None
    forward_label_scale: float = 1.0
    exposures_path: Optional[str] = None
    exposure_columns: Optional[List[str]] = None
    clip_sigma: Optional[float] = None


@dataclass
class TrainConfig:
    """Rolling retrain schedule and optimisation budget (report section 3.2).

    Attributes:
        first_predict_year: First year a model is fitted for (2018).
        last_predict_year: Last year a model is fitted for.
        lookback_years: Calendar years of history per fit (4).
        validation_years: How many of those years form the validation split (1).
        drop_last_month: Drop each split's final month, so a 20-day forward
            label never reaches across the split boundary.
        max_epochs: Maximum epochs per fit (100).
        early_stopping_patience: Epochs without validation improvement (30).
        accumulate_grad_batches: Optimiser steps are one per this many day-batches.
        num_workers: DataLoader workers.  Set to 0 when the split is preloaded
            into memory — there is nothing left to overlap, and worker IPC would
            copy every batch for nothing.
        seed: Global seed.
        precision: Lightning precision.  ``16`` enables mixed precision, which is
            worth a lot on a GPU because the stock-mixing attention is the
            dominant cost and is O(stocks^2).
        accelerator: ``"auto"``, ``"cpu"`` or ``"gpu"``.  Passed to Lightning.
        devices: Number of devices, or ``"auto"``.
        torch_threads: Intra-op threads for CPU training.  ``0`` leaves torch's
            own default, which is derived from the reported CPU count and can be
            catastrophically wrong in a container that misreports it — this box
            reports 1 core while having 185, so the default would run training
            single-threaded.  Set it explicitly whenever training on CPU.
        output_dir: Where checkpoints, logs and factor values are written.
    """

    first_predict_year: int = 2018
    last_predict_year: int = 2024
    lookback_years: int = 4
    validation_years: int = 1
    drop_last_month: bool = True
    max_epochs: int = 100
    early_stopping_patience: int = 30
    accumulate_grad_batches: int = 1
    num_workers: int = 4
    seed: int = 20240603
    precision: int = 32
    accelerator: str = "auto"
    devices: Any = "auto"
    torch_threads: int = 0
    output_dir: str = "runs/default"


@dataclass
class Config:
    """Top-level configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    minute_gsm: GSMConfig = field(default_factory=GSMConfig)
    daily_gsm: GSMConfig = field(default_factory=GSMConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    labels: LabelSettings = field(default_factory=LabelSettings)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str, overrides: Optional[Dict[str, Any]] = None) -> "Config":
        """Load a config file, then apply dotted-key overrides.

        Args:
            path: Path to the YAML file.
            overrides: ``{"train.max_epochs": 5}``-style overrides, typically
                from the command line.

        Returns:
            The populated config.
        """
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        for dotted, value in (overrides or {}).items():
            node = raw
            *parents, leaf = dotted.split(".")
            for key in parents:
                node = node.setdefault(key, {})
            node[leaf] = value
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        """Build a config from nested plain dictionaries, rejecting unknown keys."""
        return _build(cls, raw, path="")

    def to_dict(self) -> Dict[str, Any]:
        """Round-trip back to plain dictionaries, for logging and checkpoints."""
        return dataclasses.asdict(self)


def _build(cls, raw: Dict[str, Any], path: str):
    """Recursively instantiate nested dataclasses from dictionaries."""
    if not isinstance(raw, dict):
        raise TypeError(f"config section {path or '<root>'!r} must be a mapping, got {type(raw)}")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(
            f"unknown config key(s) {sorted(unknown)} in section {path or '<root>'!r}; "
            f"known keys are {sorted(names)}"
        )
    # ``from __future__ import annotations`` makes Field.type a string, so the
    # real classes have to be resolved before nested sections can be detected.
    hints = get_type_hints(cls)
    kwargs = {}
    for name, value in raw.items():
        field_type = hints[name]
        if dataclasses.is_dataclass(field_type):
            kwargs[name] = _build(field_type, value, path=f"{path}.{name}".lstrip("."))
        else:
            kwargs[name] = value
    return cls(**kwargs)
