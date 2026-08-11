"""The rolling yearly retrain of report section 3.2.

    "从 2018 年开始逐年滚动训练模型，训练模型时向过去回溯 4 年，前三年剔除末月为
     训练集，后一年剔除末月为验证集。设置早停轮数为 30，最大迭代轮数为 100。"

One model per prediction year, fitted on the four preceding years, each fit
producing out-of-sample factor values for its own year only.  Concatenating the
years gives a factor series where every value was produced by a model that never
saw its date — the point of the exercise.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from .config import Config
from .data.cache import read_manifest
from .data.datamodule import GSMAlphaDataModule
from .data.labels import LabelBuilder, LabelConfig
from .data.sources import DailyPanel
from .models.gsm_alpha import GSMAlpha
from .signature import backend_report

logger = logging.getLogger(__name__)


def resolve_hardware(config: Config) -> Dict:
    """Settle accelerator, devices and precision against what is actually present.

    ``configs/paper.yaml`` asks for mixed precision because that is what you want
    on a GPU, and the same file should still run on a CPU box without editing.
    Two version-specific traps make that need handling rather than hoping:

    * Lightning 1.6 raises outright for ``precision=16`` on a CPU accelerator
      (it routes to bfloat16, which needs torch >= 1.10).
    * Lightning 2.x accepts it but quietly substitutes ``bf16-mixed`` on CPU,
      so you would think you were timing fp16 and would not be.

    So mixed precision is downgraded to 32 whenever the run will not actually be
    on a GPU, and says so.

    Args:
        config: The full pipeline config.

    Returns:
        ``{"accelerator": ..., "devices": ..., "precision": ...}`` for ``Trainer``.
    """
    requested = config.train.accelerator
    on_gpu = requested == "gpu" or (requested == "auto" and torch.cuda.is_available())

    precision = config.train.precision
    if precision != 32 and not on_gpu:
        logger.warning(
            "train.precision=%s but this run is on CPU; falling back to 32. Mixed "
            "precision helps only on a GPU, where the O(stocks^2) stock-mixing "
            "attention is memory-bandwidth bound.",
            precision,
        )
        precision = 32
    return {"accelerator": requested, "devices": config.train.devices, "precision": precision}


def build_labels(config: Config) -> Dict:
    """Assemble the label matrix and the axes everything else aligns to.

    Args:
        config: The full pipeline config.

    Returns:
        ``{"labels": (n_dates, n_sids), "dates": ..., "sids": ...}``.
    """
    panel = DailyPanel(config.data.daily_path)
    builder = LabelBuilder(
        LabelConfig(
            horizon=config.labels.horizon,
            forward_label_path=config.labels.forward_label_path,
            forward_label_column=config.labels.forward_label_column,
            exposures_path=config.labels.exposures_path,
            exposure_columns=config.labels.exposure_columns,
            clip_sigma=config.labels.clip_sigma,
        ),
        panel.dates,
        panel.sids,
    )
    logger.info("building %d-day forward labels over %d dates", config.labels.horizon, len(panel.dates))
    return {"labels": builder.build(panel.close_matrix()), "dates": panel.dates, "sids": panel.sids}


def train_one_year(
    config: Config,
    label_bundle: Dict,
    predict_year: int,
    kind: str = "features",
) -> Optional[pd.DataFrame]:
    """Fit one rolling model and emit its out-of-sample factor values.

    Args:
        config: The full pipeline config.
        label_bundle: Output of :func:`build_labels`.
        predict_year: The year to fit for and predict.
        kind: Cache artefact to train on, ``"features"`` or ``"windows"``.

    Returns:
        A ``(date, stable_id, factor)`` frame for ``predict_year``, or ``None``
        when the year has no usable train or predict dates.
    """
    pl.seed_everything(config.train.seed, workers=True)
    if config.train.torch_threads:
        torch.set_num_threads(config.train.torch_threads)
    datamodule = GSMAlphaDataModule(
        config,
        label_bundle["labels"],
        label_bundle["dates"],
        label_bundle["sids"],
        predict_year,
        kind=kind,
    )
    if not datamodule.splits["train"] or not datamodule.splits["predict"]:
        logger.warning("predict_year=%d has no usable train/predict dates, skipping", predict_year)
        return None

    model = GSMAlpha(
        config,
        datamodule.feature_dims(),
        input_mode="features" if kind == "features" else "windows",
    )
    year_dir = os.path.join(config.train.output_dir, f"year_{predict_year}")
    os.makedirs(year_dir, exist_ok=True)

    monitor = "val/rank_ic" if datamodule.splits["val"] else "train/loss"
    mode = "max" if monitor == "val/rank_ic" else "min"
    checkpoint = ModelCheckpoint(
        dirpath=year_dir, filename="best", monitor=monitor, mode=mode, save_top_k=1
    )
    callbacks: List[pl.Callback] = [checkpoint]
    if datamodule.splits["val"]:
        callbacks.append(
            EarlyStopping(monitor=monitor, mode=mode, patience=config.train.early_stopping_patience)
        )

    trainer = pl.Trainer(
        default_root_dir=year_dir,
        max_epochs=config.train.max_epochs,
        callbacks=callbacks,
        logger=CSVLogger(year_dir, name="logs"),
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        **resolve_hardware(config),
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        log_every_n_steps=50,
    )
    trainer.fit(model, datamodule=datamodule)

    if checkpoint.best_model_path:
        logger.info("year %d: best %s at %s", predict_year, monitor, checkpoint.best_model_path)
        model = GSMAlpha.load_from_checkpoint(
            checkpoint.best_model_path,
            config=config,
            feature_dims=datamodule.feature_dims(),
            input_mode=model.input_mode,
        )

    outputs = trainer.predict(model, dataloaders=datamodule.predict_dataloader())
    frames = [
        pd.DataFrame(
            {
                "date": np.repeat(np.datetime64(out["date"], "D"), len(out["sid"])),
                "stable_id": out["sid"],
                "factor": np.asarray(out["factor"], dtype=np.float32),
            }
        )
        for out in (outputs or [])
        if len(out["sid"])
    ]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def run_rolling(config: Config, kind: str = "features") -> pd.DataFrame:
    """Run every rolling fit and write the stitched factor panel.

    Args:
        config: The full pipeline config.
        kind: Cache artefact to train on.

    Returns:
        The concatenated ``(date, stable_id, factor)`` panel, also written to
        ``<output_dir>/factor.parquet``.
    """
    os.makedirs(config.train.output_dir, exist_ok=True)
    logger.info("%s", backend_report(config.minute_gsm.backend))
    if config.train.torch_threads:
        torch.set_num_threads(config.train.torch_threads)
    hardware = resolve_hardware(config)
    logger.info(
        "accelerator=%s devices=%s precision=%s cuda=%s torch_threads=%d",
        hardware["accelerator"], hardware["devices"], hardware["precision"],
        torch.cuda.is_available(), torch.get_num_threads(),
    )
    logger.info("cache manifest: %s", read_manifest(config.data.cache_dir).get("feature_dims"))

    label_bundle = build_labels(config)
    pieces = []
    for year in range(config.train.first_predict_year, config.train.last_predict_year + 1):
        logger.info("=== rolling fit for %d ===", year)
        frame = train_one_year(config, label_bundle, year, kind=kind)
        if frame is None:
            continue
        frame.to_parquet(os.path.join(config.train.output_dir, f"factor_{year}.parquet"), index=False)
        pieces.append(frame)

    if not pieces:
        raise RuntimeError("no rolling fit produced predictions; check the cache date coverage")
    panel = pd.concat(pieces, ignore_index=True).sort_values(["date", "stable_id"])
    out_path = os.path.join(config.train.output_dir, "factor.parquet")
    panel.to_parquet(out_path, index=False)
    logger.info("wrote %d factor values to %s", len(panel), out_path)
    return panel
