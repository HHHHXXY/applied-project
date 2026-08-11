"""Is the ablation's improvement distinguishable from noise, and do the arms combine?

The headline table shows arm C ahead of arm A16 by 0.48pp of Rank IC, which is
easy to read as a win and is not one on its own: with 53 monthly periods that
difference has a 95% interval of [-0.77pp, +1.72pp].  This script does the two
tests that decide what may honestly be claimed.

1. A paired test on the IC series.  Monthly first (the report's cadence, 53
   points), then daily, which has 20x the observations but heavily OVERLAPPING
   20-day forward windows — so its IC series is autocorrelated and a naive t
   would overstate significance.  Newey-West at lag 20, the overlap length,
   is what makes the daily test trustworthy.

2. An equal-weight combination of the z-scored arms.  The arms turn out to be
   only 0.36-0.62 rank-correlated, so "which architecture wins" may be the wrong
   question; this measures whether keeping both beats picking one.

    python scripts/significance.py --daily <daily.parquet> --exposures <exp.parquet>
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsm_alpha.evaluate import evaluate  # noqa: E402

ARMS = {
    "A16 SM T=16": "results/armA16_stockmixer_T16.parquet",
    "B  GSM daily": "results/armB_gsm_daily.parquet",
    "C  GSM+5min": "results/armC_gsm_daily_5min.parquet",
}
PAIRS = [("A16 SM T=16", "B  GSM daily"), ("B  GSM daily", "C  GSM+5min"),
         ("A16 SM T=16", "C  GSM+5min")]


def newey_west_t(diff: pd.Series, lag: int = 20):
    """t statistic on the mean of an overlapping series, HAC-corrected."""
    x = diff.values - diff.values.mean()
    n = len(x)
    gamma = (x @ x) / n
    for k in range(1, lag + 1):
        gamma += 2 * (1 - k / (lag + 1)) * ((x[k:] @ x[:-k]) / n)
    se = np.sqrt(gamma / n)
    return diff.mean() / se, se


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", required=True)
    ap.add_argument("--exposures", required=True)
    args = ap.parse_args()

    panels = {name: pd.read_parquet(path) for name, path in ARMS.items()}
    scored = {
        cadence: {
            name: evaluate(panel, daily_path=args.daily, horizon=20, n_groups=5,
                           rebalance=cadence, neutralize_path=args.exposures)
            for name, panel in panels.items()
        }
        for cadence in ("monthly", "daily")
    }

    for cadence, note in (("monthly", "the report's cadence"),
                          ("daily", "20x the observations, Newey-West lag 20")):
        ic = {n: r["ic_series"] for n, r in scored[cadence].items()}
        print(f"\n=== paired IC test, {cadence} rebalance ({note}) ===\n")
        print(f"{'comparison':30s}{'dIC':>10s}{'t':>8s}{'95% CI':>24s}")
        for a, b in PAIRS:
            d = (ic[b] - ic[a]).dropna()
            if cadence == "monthly":
                t, _ = stats.ttest_1samp(d, 0.0)
                se = d.std(ddof=1) / np.sqrt(len(d))
            else:
                t, se = newey_west_t(d)
            lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
            print(f"{a} -> {b:16s}{d.mean():+10.4f}{t:8.2f}"
                  f"{f'[{lo:+.4f}, {hi:+.4f}]':>24s}")
        print(f"  n = {len(d)}")

    print("\n=== cross-sectional rank correlation between arms (mean over dates) ===\n")
    for a, b in itertools.combinations(ARMS, 2):
        merged = panels[a].merge(panels[b], on=["date", "stable_id"], suffixes=("_a", "_b"))
        r = merged.groupby("date").apply(
            lambda g: g["factor_a"].corr(g["factor_b"], method="spearman"),
            include_groups=False,
        )
        print(f"  {a} vs {b:16s} {r.mean():.3f}")

    print("\n=== equal-weight combination of the z-scored arms ===\n")
    merged = None
    for name, panel in panels.items():
        renamed = panel.rename(columns={"factor": name})
        merged = renamed if merged is None else merged.merge(renamed, on=["date", "stable_id"])
    for name in ARMS:
        merged[name] = merged.groupby("date")[name].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0)
        )
    merged["factor"] = merged[list(ARMS)].mean(axis=1)
    combo = evaluate(merged[["date", "stable_id", "factor"]], daily_path=args.daily,
                     horizon=20, n_groups=5, rebalance="monthly",
                     neutralize_path=args.exposures)["summary"]
    for name in ARMS:
        s = scored["monthly"][name]["summary"]
        print(f"  {name:16s} Rank IC {s['rank_ic']:.4f}  ICIR {s['icir']:.2f}  "
              f"Sharpe {s['long_short_sharpe']:.2f}  L/S {s['long_short_annual']:.2%}")
    print(f"  {'equal-weight':16s} Rank IC {combo['rank_ic']:.4f}  ICIR {combo['icir']:.2f}  "
          f"Sharpe {combo['long_short_sharpe']:.2f}  L/S {combo['long_short_annual']:.2%}")


if __name__ == "__main__":
    main()
