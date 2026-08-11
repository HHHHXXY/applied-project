"""Export industry dummies and log market cap from the tq lake into one parquet.

Run this **on a machine that has the tq data lake**, then ship the output with
the price panels.  It unlocks the report's setting in two places at once:

* ``labels.exposures_path`` — the report's exact training target, the 20-day
  forward return residualised against industry and market cap (section 3.2).
* ``evaluate --neutralize-factor`` — the report's exact backtest, which applies
  "去极值、标准化以及行业市值中性化" to the *factor* before ranking (section 3.3,
  tables 12 and 14).

This is a narrower neutralisation than ``scripts/export_labels.py``'s risk-model
residual, which also strips the full style block.  Use this one when the point is
to line up against the report's numbers; use the residual when the point is to
build the best factor.

    python scripts/export_exposures.py --out exposures.parquet
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CAP_ROOT = "/tq/release/data/cn_equity/cap"
INDUSTRY_ROOT = "/tq/release/data/cn_equity/industry_classifications"


def _read_periods(root: str, columns) -> pd.DataFrame:
    """Concatenate every period file under a tq store, keeping ``columns``."""
    files = sorted(glob.glob(os.path.join(root, "*", "data.pq")))
    if not files:
        files = sorted(glob.glob(os.path.join(root, "*", "*", "data.pq")))
    if not files:
        raise SystemExit(f"no data.pq under {root}")

    available = pq.read_schema(files[0]).names
    wanted = [c for c in columns if c in available]
    if not wanted:
        raise SystemExit(f"{root}: none of {columns} present; it has {available[:12]}")

    pieces = []
    for path in files:
        frame = pq.read_table(path, columns=["date", "stable_id", *wanted]).to_pandas()
        if isinstance(frame.index, pd.MultiIndex) or frame.index.name is not None:
            frame = frame.reset_index()
        pieces.append(frame)
    out = pd.concat(pieces, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cap-root", default=CAP_ROOT)
    parser.add_argument("--industry-root", default=INDUSTRY_ROOT)
    parser.add_argument("--cap-column", default="total_cap",
                        help="market cap column; log is taken (default: total_cap)")
    parser.add_argument("--industry-column", default="citics_2019_level1_code",
                        help="industry classification column (default: citics_2019_level1_code)")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    cap = _read_periods(args.cap_root, [args.cap_column])
    industry = _read_periods(args.industry_root, [args.industry_column])
    frame = cap.merge(industry, on=["date", "stable_id"], how="inner")
    if args.start:
        frame = frame[frame["date"] >= pd.Timestamp(args.start)]
    if args.end:
        frame = frame[frame["date"] <= pd.Timestamp(args.end)]

    frame = frame[frame[args.cap_column] > 0].dropna(subset=[args.industry_column])
    frame["log_mktcap"] = np.log(frame[args.cap_column].astype(np.float64))

    # Industry dummies, one column dropped: the neutralisation regression already
    # fits an intercept, so a full dummy set would be collinear with it.
    dummies = pd.get_dummies(frame[args.industry_column].astype("category"),
                             prefix="ind", dtype=np.float32)
    if dummies.shape[1] > 1:
        dummies = dummies.iloc[:, 1:]

    out = pd.concat(
        [frame[["date", "stable_id", "log_mktcap"]].reset_index(drop=True),
         dummies.reset_index(drop=True)],
        axis=1,
    ).sort_values(["date", "stable_id"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False, compression="zstd")

    print(f"wrote {len(out):,} rows to {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")
    print(f"  dates      {out['date'].min().date()} .. {out['date'].max().date()} "
          f"({out['date'].nunique():,} days)")
    print(f"  names      {out['stable_id'].nunique():,}")
    print(f"  exposures  log_mktcap + {dummies.shape[1]} industry dummies "
          f"({args.industry_column}, one dropped as the base level)")
    print()
    print("use as the report's training target:")
    print(f"  --set labels.exposures_path={os.path.abspath(args.out)}")
    print("and as the report's backtest neutralisation:")
    print(f"  evaluate --neutralize-factor {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
