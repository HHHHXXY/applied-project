"""Score every ablation arm on one identical yardstick and print the comparison.

The three arms answer two questions by subtraction:

    A  StockMixer   multi-scale time mixing   daily
    B  GSM-Alpha    log-signature             daily          A -> B = the signature
    C  GSM-Alpha    log-signature             daily + 5-min  B -> C = the 5-minute data

Every arm is scored through the same :func:`gsm_alpha.evaluate.evaluate` call with
the same neutralisation, horizon, cadence and bucket count, so the only thing that
differs between the columns is the factor panel itself.  The report's own per-year
figures (its table 12) are printed alongside, with the caveat that its 2018 and
2019 rows have no counterpart here — five-minute history starts 2016-01-04, so a
2018 fit, which needs 2014, cannot be built.

    python scripts/compare_arms.py --daily <daily.parquet> --exposures <exp.parquet> \
        --arm "A StockMixer=/root/runs_stockmixer/factor.parquet" ...
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from gsm_alpha.evaluate import evaluate  # noqa: E402

# 东北证券 2024-06-03, table 12 (neutralised), the years this data can reach.
REPORT_TABLE_12 = {
    2020: {"rank_ic": 0.1284, "icir": 2.48, "long_short_annual": 0.3853},
    2021: {"rank_ic": 0.1080, "icir": 1.60, "long_short_annual": 0.3392},
    2022: {"rank_ic": 0.1002, "icir": 2.49, "long_short_annual": 0.3177},
    2023: {"rank_ic": 0.1247, "icir": 2.55, "long_short_annual": 0.3251},
    2024: {"rank_ic": 0.1049, "icir": 1.82, "long_short_annual": 0.4298},
}


def score(path: str, daily: str, exposures: str) -> Dict[str, object]:
    """Evaluate one arm's factor panel in the report's caliber."""
    factor = pd.read_parquet(path)
    return evaluate(
        factor,
        daily_path=daily,
        horizon=20,
        n_groups=5,
        rebalance="monthly",
        neutralize_path=exposures,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily", required=True)
    parser.add_argument("--exposures", required=True)
    parser.add_argument(
        "--arm", action="append", default=[],
        help="repeatable, as 'label=/path/to/factor.parquet' (split at the last =)",
    )
    args = parser.parse_args()

    arms: List[Tuple[str, Dict[str, object]]] = []
    for spec in args.arm:
        # Split on the LAST "=", so a label may legitimately contain one
        # ("A StockMixer T=60"); paths do not carry "=" in practice.
        label, _, path = spec.rpartition("=")
        if not path:
            raise SystemExit(f"--arm expects label=path, got {spec!r}")
        try:
            arms.append((label, score(path, args.daily, args.exposures)))
            print(f"scored {label}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - one broken arm must not lose the rest
            print(f"SKIPPED {label}: {exc}", file=sys.stderr)

    if not arms:
        raise SystemExit("no arm could be scored")

    print("\n" + "=" * 78)
    print("ABLATION: what the signature buys, and what the 5-minute data buys")
    print("=" * 78)

    print("\n--- aggregate (2020-2024, report caliber: winsorise + standardise + "
          "industry/mktcap neutral) ---\n")
    keys = ["Rank IC", "ICIR", "IC win rate", "long-short annual", "long-short vol",
            "long-short Sharpe", "top group annual"]
    src = ["rank_ic", "icir", "ic_win_rate", "long_short_annual", "long_short_vol",
           "long_short_sharpe", "top_group_annual"]
    width = max(len(k) for k in keys) + 2
    header = "".ljust(width) + "".join(label.rjust(22) for label, _ in arms)
    print(header)
    for key, field in zip(keys, src):
        row = key.ljust(width)
        for _, result in arms:
            value = result["summary"].get(field)
            row += (f"{value:.4f}" if isinstance(value, float) else str(value)).rjust(22)
        print(row)

    print("\n--- per year: Rank IC ---\n")
    print("year".ljust(8) + "".join(label.rjust(22) for label, _ in arms) + "report".rjust(12))
    for year in sorted(REPORT_TABLE_12):
        row = str(year).ljust(8)
        for _, result in arms:
            by_year = result["by_year"]
            hit = by_year[by_year.index == year] if hasattr(by_year, "index") else None
            value = float(hit["rank_ic"].iloc[0]) if hit is not None and len(hit) else float("nan")
            row += f"{value:.4f}".rjust(22)
        row += f"{REPORT_TABLE_12[year]['rank_ic']:.4f}".rjust(12)
        print(row)

    print("\nreport = 东北证券 table 12. Its 2018/2019 rows are omitted: a 2018 fit needs")
    print("2014 five-minute data and this lake starts 2016-01-04.")


if __name__ == "__main__":
    main()
