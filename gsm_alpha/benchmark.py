"""Measure this machine before committing to a long run.

The two costs that decide the wall clock are the per-date cache build (CPU, and
signature-bound) and the per-step training throughput (dominated by the
stock-mixing attention, which is O(stocks^2) and is the part a GPU accelerates).
They live on different machines in the usual split — build the cache on CPU,
train on GPU — so each is timed independently.

    python -m gsm_alpha.cli benchmark --stocks 4300 --device cuda

The projection it prints is arithmetic on the measured step time and the rolling
schedule, not a guess.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import Config
from .models.gsm_alpha import GSMAlpha
from .models.loss import weighted_correlation_loss
from .signature import backend_report

logger = logging.getLogger(__name__)

# Trading days in one rolling fit at stride 1: three training years and one
# validation year, each minus its final month.
TRAIN_DATES_PER_FIT = 700
VAL_DATES_PER_FIT = 233
VAL_COST_FRACTION = 0.35  # a forward-only pass against a full training step


def time_training_step(
    config: Config,
    n_stocks: int,
    feature_dims: Dict[str, int],
    device: str = "cpu",
    precision: int = 32,
    iterations: int = 20,
) -> float:
    """Median seconds per optimiser step at a given cross-section size.

    Args:
        config: The pipeline config, for the model hyper-parameters.
        n_stocks: Names in the day-batch.
        feature_dims: Feature width per branch.
        device: ``"cpu"``, ``"cuda"`` or an explicit device string.
        precision: 16 to time under autocast, as mixed-precision training would.
        iterations: Timed iterations after warm-up.

    Returns:
        Median step time in seconds.
    """
    torch_device = torch.device(device)
    model = GSMAlpha(config, feature_dims).to(torch_device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = {b: torch.randn(n_stocks, d, device=torch_device) for b, d in feature_dims.items()}
    labels = torch.randn(n_stocks, device=torch_device)
    use_amp = precision == 16 and torch_device.type == "cuda"

    def one_step() -> None:
        optimiser.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                loss = weighted_correlation_loss(model(batch), labels)
        else:
            loss = weighted_correlation_loss(model(batch), labels)
        loss.backward()
        optimiser.step()

    for _ in range(5):
        one_step()
    if torch_device.type == "cuda":
        torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        one_step()
        if torch_device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def project_schedule(step_seconds: float, stride: int, n_fits: int, epochs: int) -> float:
    """Hours for the whole rolling retrain at a measured step time.

    Args:
        step_seconds: Seconds per training step.
        stride: ``data.train_day_stride``.
        n_fits: Number of rolling fits (prediction years).
        epochs: Epochs per fit.

    Returns:
        Wall-clock hours, training only — the cache build is separate.
    """
    per_epoch = (TRAIN_DATES_PER_FIT + VAL_DATES_PER_FIT * VAL_COST_FRACTION) / stride
    return per_epoch * step_seconds * epochs * n_fits / 3600


def run(
    config: Config,
    stocks: Optional[List[int]] = None,
    device: str = "cpu",
    precision: Optional[int] = None,
    threads: int = 0,
) -> None:
    """Print a measured cost table for this machine.

    Args:
        config: The pipeline config.
        stocks: Cross-section sizes to time.
        device: Torch device for the training benchmark.
        precision: Override for ``train.precision``.
        threads: CPU intra-op threads; 0 keeps torch's default.
    """
    stocks = stocks or [1500, 3000, 4300]
    threads = threads or config.train.torch_threads
    if threads:
        torch.set_num_threads(threads)
    precision = precision or config.train.precision
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda but torch.cuda.is_available() is False. On a modern card "
            "(sm_89/sm_90: 4090, L40S, H100) torch 1.9 cannot work at all — install a "
            "current torch and let the pure-python signature backend take over; "
            "training reads cached features, so signatory is not needed on this box."
        )

    print(backend_report(config.minute_gsm.backend))
    if device.startswith("cuda"):
        print(f"device: {torch.cuda.get_device_name(0)}  (precision {precision})")
    else:
        print(f"device: cpu, {torch.get_num_threads()} threads  (precision {precision})")

    n_fits = config.train.last_predict_year - config.train.first_predict_year + 1
    dims = {}
    if config.data.use_minute_branch:
        dims["minute"] = 800
    if config.data.use_daily_branch:
        dims["daily"] = 800

    print()
    print(f"{'stocks':>7} {'step':>9} {'steps/s':>9} "
          f"{'stride5 x60ep':>14} {'stride1 x60ep':>14}")
    print("-" * 58)
    for n in stocks:
        step = time_training_step(config, n, dims, device=device, precision=precision)
        s5 = project_schedule(step, 5, n_fits, 60)
        s1 = project_schedule(step, 1, n_fits, 60)
        print(f"{n:>7} {step * 1000:>7.1f}ms {1 / step:>9.1f} "
              f"{s5:>12.1f} h {s1:>12.1f} h")

    print()
    print(f"projections cover {n_fits} rolling fits at 60 epochs each, training only.")
    print("the cache build is separate and runs on CPU; see README section 8.")
    bytes_per_stock_date = sum(dims.values()) * (2 if config.data.feature_dtype == "float16" else 4)
    print(f"cache: {bytes_per_stock_date / 1024:.1f} KB per stock-date "
          f"({config.data.feature_dtype}), so 2430 dates x {stocks[-1]} names = "
          f"{2430 * stocks[-1] * bytes_per_stock_date / 1e9:.0f} GB")
