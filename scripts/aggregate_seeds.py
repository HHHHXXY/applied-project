"""Aggregate the multi-seed ablation into the tables the report prints.

Each arm is trained under three seeds, so every headline number carries a mean
and a spread. The spread is what makes the per-year comparisons readable: a gap
between two arms only means something if it is larger than the gap the same arm
shows against itself under a different seed.

The last block is the one worth reading twice. It measures how far apart the
arms are (average daily cross-sectional rank correlation between arms, matched
seed by seed) against how far apart one arm is from itself (the same correlation
between two seeds of the same arm). If seeds of one architecture agree with each
other more than two architectures agree with each other, then the difference
between the arms is a property of the architecture and not of the optimiser.

    python scripts/aggregate_seeds.py --runs /root/runs --daily <daily.parquet> \
        --exposures <exposures.parquet>
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsm_alpha.evaluate import evaluate  # noqa: E402

ARMS = {"StockMixer": "stockmixer", "Sig-D": "sig_d", "Sig-DI": "sig_di"}
SEEDS = (42, 43, 44)
PERIODS_PER_YEAR = 243 / 20
METRICS = [
    ("RankIC", "rank_ic", "{:.2%}"),
    ("ICIR", "icir", "{:.2f}"),
    ("IC win rate", "ic_win_rate", "{:.1%}"),
    ("Long-short return", "long_short_annual", "{:.2%}"),
    ("Long-short volatility", "long_short_vol", "{:.2%}"),
    ("Long-short Sharpe", "long_short_sharpe", "{:.2f}"),
]


def load(runs: Path, slug: str, seed: int):
    path = runs / f"{slug}_s{seed}" / "factor.parquet"
    return pd.read_parquet(path) if path.exists() else None


def spread(values: List[float], fmt: str) -> str:
    """mean +/- sample standard deviation, both in the metric's own units."""
    a = np.asarray(values, dtype=float)
    if len(a) == 1:
        return fmt.format(a[0])
    lead, tail = fmt.format(a.mean()), fmt.format(a.std(ddof=1))
    return f"{lead} +/- {tail.rstrip('%')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--daily", required=True)
    ap.add_argument("--exposures", required=True)
    args = ap.parse_args()
    runs = Path(args.runs)

    panels: Dict[str, Dict[int, pd.DataFrame]] = {}
    scored: Dict[str, Dict[int, dict]] = {}
    for name, slug in ARMS.items():
        panels[name], scored[name] = {}, {}
        for seed in SEEDS:
            panel = load(runs, slug, seed)
            if panel is None:
                print(f"missing: {slug} seed {seed}", file=sys.stderr)
                continue
            panels[name][seed] = panel
            scored[name][seed] = evaluate(
                panel, daily_path=args.daily, horizon=20, n_groups=5,
                rebalance="monthly", neutralize_path=args.exposures,
            )
        print(f"scored {name}: {len(scored[name])} seeds", file=sys.stderr)

    live = [n for n in ARMS if scored[n]]
    if not live:
        raise SystemExit("no run could be scored")

    n_per = {n: {s: r["summary"]["n_periods"] for s, r in scored[n].items()} for n in live}
    print(f"\nevaluation periods: {sorted(set(v for d in n_per.values() for v in d.values()))}")

    print("\n=== headline, mean +/- sd over seeds ===\n")
    print("".ljust(24) + "".join(n.rjust(22) for n in live))
    for label, field, fmt in METRICS:
        row = label.ljust(24)
        for n in live:
            row += spread([scored[n][s]["summary"][field] for s in scored[n]], fmt).rjust(22)
        print(row)

    for label, field, fmt in (("RankIC", "rank_ic", "{:.2%}"),
                              ("ICIR", "icir", "{:.2f}"),
                              ("Long-short volatility", None, "{:.2%}"),
                              ("Long-short Sharpe", None, "{:.2f}")):
        print(f"\n=== {label} by year, mean +/- sd over seeds ===\n")
        print("year".ljust(8) + "".join(n.rjust(22) for n in live))
        years = sorted({y for n in live for s in scored[n]
                        for y in scored[n][s]["by_year"].index})
        for year in years:
            row = str(year).ljust(8)
            for n in live:
                vals = []
                for s in scored[n]:
                    if field is not None:
                        by = scored[n][s]["by_year"]
                        hit = by[by.index == year]
                        if len(hit):
                            vals.append(float(hit[field].iloc[0]))
                    else:
                        g = scored[n][s]["groups"]
                        ls = g.iloc[:, -1] - g.iloc[:, 0]
                        ls = ls[ls.index.year == year]
                        if len(ls) > 1:
                            v = ls.std() * np.sqrt(PERIODS_PER_YEAR) if "volatility" in label \
                                else ls.mean() / ls.std() * np.sqrt(PERIODS_PER_YEAR)
                            vals.append(float(v))
                row += (spread(vals, fmt) if vals else "-").rjust(22)
            print(row)

    print("\n=== architecture distance vs seed distance ===\n")
    print("Average daily cross-sectional rank correlation.\n")

    def corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
        m = a.merge(b, on=["date", "stable_id"], suffixes=("_a", "_b"))
        r = m.groupby("date").apply(
            lambda g: g["factor_a"].corr(g["factor_b"], method="spearman"),
            include_groups=False)
        return float(r.mean())

    print("  within one architecture, across seeds")
    within = {}
    for n in live:
        vals = [corr(panels[n][x], panels[n][y]) for x, y in itertools.combinations(panels[n], 2)]
        if vals:
            within[n] = float(np.mean(vals))
            print(f"    {n:14s} {within[n]:.3f}")

    print("\n  between architectures, matched seed by seed")
    between = {}
    for a, b in itertools.combinations(live, 2):
        shared = sorted(set(panels[a]) & set(panels[b]))
        vals = [corr(panels[a][s], panels[b][s]) for s in shared]
        if vals:
            between[(a, b)] = float(np.mean(vals))
            print(f"    {a} vs {b:14s} {between[(a,b)]:.3f}")

    if within and between:
        print(f"\n  mean within-architecture {np.mean(list(within.values())):.3f}"
              f"   vs   mean between-architecture {np.mean(list(between.values())):.3f}")


if __name__ == "__main__":
    main()
