"""The GSM-Alpha network of report section 3.1, as a LightningModule.

Figure 4 reads bottom to top:

1. Each stock's raw stream enters a **GSM & Indicator mixing** block.  GSM turns
   the multivariate path into a cross-sectional log-signature feature vector, a
   ``Linear`` drops it to the working width, and a small MLP
   (``LayerNorm -> Linear -> GELU -> Linear``) mixes the indicators, added back
   residually.  Two branches run in parallel — 20 days of 5-minute bars and 60
   days of dailies — and their outputs are concatenated per stock.
2. **Stock mixing** lets each name borrow from its peers: multi-head
   self-attention *across the stocks in the batch*, layer-normalised and added
   residually.  Attention rather than an MLP because the number of stocks
   changes from day to day.
3. A final linear layer produces the one-dimensional factor value.

A batch is one trading day's entire cross section, which is what makes step 2
well defined and what the section 3.2 loss expects.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from ..config import Config
from ..signature import GSM
from .loss import rank_ic, weighted_correlation_loss
from .stockmixer import StockMixerBackbone
from .transforms import gaussian_rank

INPUT_TRANSFORMS = ("none", "gauss_rank")
ARCHITECTURES = ("gsm_alpha", "stockmixer")

logger = logging.getLogger(__name__)


class IndicatorMixing(nn.Module):
    """Project one branch's GSM features and mix them residually.

    Args:
        in_features: Width of the GSM output for this branch.
        hidden_dim: Working width.
    """

    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.project = nn.Linear(in_features, hidden_dim)
        self.mix = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Args: ``(n_stocks, in_features)``.  Returns: ``(n_stocks, hidden_dim)``."""
        projected = self.project(features)
        return projected + self.mix(projected)


class StockMixing(nn.Module):
    """Multi-head self-attention across the stocks of one trading day.

    Args:
        dim: Feature width.
        n_heads: Attention heads.
        dropout: Attention dropout.
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"hidden width {dim} must be divisible by {n_heads} heads")
        self.attention = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Args: ``(n_stocks, dim)``.  Returns: ``(n_stocks, dim)``."""
        # One "sequence" whose tokens are the stocks, so attention is over names.
        x = features.unsqueeze(0)
        attended, _ = self.attention(x, x, x, need_weights=False)
        return features + self.norm(attended.squeeze(0))


class GSMAlpha(pl.LightningModule):
    """GSM-Alpha: two GSM branches, indicator mixing, stock mixing, one output.

    Args:
        config: The full pipeline config; only the model, GSM and data sections
            are used here.
        feature_dims: Width of each branch's GSM output, e.g.
            ``{"minute": 800, "daily": 800}``.  Taken from the cache manifest so
            the network and the cached features cannot silently disagree.
        input_mode: ``"features"`` when batches carry cached GSM output, or
            ``"windows"`` when they carry raw normalised paths and GSM runs
            inside the forward pass (needed for learnable augmentations).
    """

    def __init__(
        self,
        config: Config,
        feature_dims: Dict[str, int],
        input_mode: str = "features",
    ) -> None:
        super().__init__()
        if input_mode not in ("features", "windows"):
            raise ValueError(f"unknown input_mode {input_mode!r}")
        if config.model.input_transform not in INPUT_TRANSFORMS:
            raise ValueError(
                f"unknown model.input_transform {config.model.input_transform!r}, "
                f"expected one of {INPUT_TRANSFORMS}"
            )
        self.save_hyperparameters({"config": config.to_dict(), "feature_dims": feature_dims,
                                   "input_mode": input_mode})
        self.config = config
        self.input_mode = input_mode
        self.branches: List[str] = sorted(feature_dims)

        model_cfg = config.model
        if model_cfg.architecture not in ARCHITECTURES:
            raise ValueError(
                f"unknown model.architecture {model_cfg.architecture!r}, "
                f"expected one of {ARCHITECTURES}"
            )
        self.architecture = model_cfg.architecture

        # The StockMixer baseline is carried inside this module rather than in a
        # LightningModule of its own so the ablation arms cannot drift: loss,
        # metric, logging, early stopping, checkpointing and prediction are the
        # same code by construction, and only the feature extractor differs.
        if self.architecture == "stockmixer":
            if input_mode != "windows":
                raise ValueError(
                    "model.architecture='stockmixer' consumes raw OHLCV windows; "
                    "train with --kind windows (input_mode='windows')"
                )
            if len(self.branches) != 1:
                raise ValueError(
                    "the StockMixer baseline takes exactly one branch of raw windows, "
                    f"got {self.branches}; set data.use_minute_branch=false"
                )
            # NOT feature_dims: in windows mode that carries the GSM output
            # width (800), because it is what the *other* architecture would
            # produce from these same windows. StockMixer consumes the window
            # itself, so it needs the step count the cache actually stored.
            from ..data.cache import read_manifest

            steps = int(read_manifest(config.data.cache_dir)[f"{self.branches[0]}_steps"])
            self.backbone = StockMixerBackbone(
                n_steps=steps,
                n_indicators=model_cfg.n_indicators,
                scales=list(model_cfg.time_scales),
                hidden=model_cfg.hidden_dim,
                embed_dim=model_cfg.hidden_dim,
                n_market_states=model_cfg.n_market_states,
            )
            self._val_ic: List[float] = []
            self._val_nonfinite = 0
            return

        self.gsm = nn.ModuleDict()
        if input_mode == "windows":
            from ..data.cache import build_gsm

            for branch in self.branches:
                section = config.minute_gsm if branch == "minute" else config.daily_gsm
                self.gsm[branch] = build_gsm(section)

        self.indicator_mixing = nn.ModuleDict(
            {b: IndicatorMixing(feature_dims[b], model_cfg.hidden_dim) for b in self.branches}
        )
        width = model_cfg.hidden_dim * len(self.branches)
        self.stock_mixing = (
            StockMixing(width, model_cfg.n_attention_heads, model_cfg.attention_dropout)
            if model_cfg.use_stock_mixing
            else None
        )
        self.head = nn.Linear(width, 1)
        self._val_ic: List[float] = []
        self._val_nonfinite = 0

    # -- forward ----------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Factor values for one day's cross section.

        Args:
            batch: ``{"<branch>": tensor}`` per branch — ``(n_stocks, n_features)``
                in feature mode, ``(n_stocks, n_steps, 5)`` in window mode.

        Returns:
            ``(n_stocks,)`` factor values.
        """
        if self.architecture == "stockmixer":
            # Raw (n_stocks, n_steps, n_indicators) windows, already normalised
            # by the cache; the temporal encoder is the whole point of the arm,
            # so no signature and no gauss_rank run here.
            return self.backbone(batch[self.branches[0]].float())

        mixed = []
        for branch in self.branches:
            x = batch[branch]
            if self.input_mode == "windows":
                x = self.gsm[branch](x)
            if self.config.model.input_transform == "gauss_rank":
                x = gaussian_rank(x)
            mixed.append(self.indicator_mixing[branch](x))
        features = torch.cat(mixed, dim=-1)
        if self.stock_mixing is not None:
            features = self.stock_mixing(features)
        return self.head(features).squeeze(-1)

    # -- lightning hooks --------------------------------------------------

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        predictions = self(batch)
        loss = weighted_correlation_loss(
            predictions, batch["label"], self.config.model.loss_halflife_fraction
        )
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        predictions = self(batch)
        if not torch.isfinite(predictions).all():
            # Almost always a corrupted feature cache — an inf in the inputs, for
            # instance from a lossy feature_dtype. Say so here rather than let it
            # surface hours later as a missing early-stopping metric.
            self._val_nonfinite += 1
        loss = weighted_correlation_loss(
            predictions, batch["label"], self.config.model.loss_halflife_fraction
        )
        ic = rank_ic(predictions, batch["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        if torch.isfinite(ic):
            self._val_ic.append(float(ic))

    def on_validation_epoch_start(self) -> None:
        self._val_ic = []
        self._val_nonfinite = 0

    def on_validation_epoch_end(self) -> None:
        if self._val_nonfinite:
            logger.warning(
                "%d validation batches produced non-finite predictions; check the feature "
                "cache for inf/NaN (a lossy data.feature_dtype is the usual cause)",
                self._val_nonfinite,
            )
        # Always log the metric, even with nothing to average: EarlyStopping and
        # ModelCheckpoint monitor it, and a missing key aborts the run outright.
        # A degenerate epoch should score badly, not crash the fit.
        ic = np.asarray(self._val_ic, dtype=np.float64)
        if ic.size:
            self.log("val/rank_ic", float(ic.mean()), prog_bar=True)
            self.log("val/icir", float(ic.mean() / ic.std()) if ic.std() > 0 else 0.0)
        else:
            logger.warning("no validation batch yielded a finite rank IC this epoch")
            self.log("val/rank_ic", float("-inf"), prog_bar=True)
            self.log("val/icir", 0.0)

    def predict_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = 0) -> Dict:
        return {
            "date": batch["date"],
            "sid": batch["sid"],
            "factor": self(batch).detach().cpu().numpy(),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay,
        )
