"""Export a risk-model residual forward return from the tq zoo into one parquet.

Run this **on a machine that has the tq data lake**, then ship the single output
file alongside the two price panels.  The training box never needs tq, DataAlchemy
or the lake — the exported file satisfies the same ``(date, stable_id, value)``
contract as everything else this project reads.

    python scripts/export_labels.py --out labels_res20d.parquet

Background: the report trains on the industry- and market-cap-neutralised 20-day
forward return.  The zoo's ``resp_res_*`` family is the same idea done more
thoroughly — the return residualised against a full RQ risk model (country,
industry, and the whole style block including size), then shifted forward.

**Which one to pick at a 20-day horizon.**  ``res_rq_cntrb`` (CNTR = the RQ
*trading*, short-horizon model) is the house default for factor evaluation, but
it is only materialised out to ``_10d``.  At 15d/20d/30d/40d the materialised
family is ``res_rq_cnltb`` (CNLT = the *long-term* model), which is in any case
the better-matched model for a 20-day holding period.  Hence the default below.

Verified alignment: ``resp_res_rq_cnltb_1500to1500_20d`` on date ``t`` correlates
0.80 with the realised close-to-close 20-day return *starting* at ``t``, and only
0.02-0.04 with the windows starting at ``t-20`` or ``t+20``.  It is the forward
return for its own row's date, matching this project's label convention, and the
sub-1.0 correlation is the risk factors having been stripped out.  Nothing shifts
it again downstream.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

DEFAULT_ROOT = (
    "/tq/release/data/zoo/cn_equity/estu_x/data/"
    "resp_statarb_composite_daily_res_rq_cnltb_returns.DAILY_BAR.Q.eod"
)
DEFAULT_COLUMN = "resp_res_rq_cnltb_1500to1500_20d"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help="zoo response store, one directory per year/quarter")
    parser.add_argument("--horizon", default="20d",
                        help="which resp_<horizon>.pq file to read (default: 20d)")
    parser.add_argument("--column", default=DEFAULT_COLUMN,
                        help="response column to export; the 1500to1500 slot is close-to-close")
    parser.add_argument("--start", default=None, help="drop dates before this YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="drop dates after this YYYY-MM-DD")
    parser.add_argument("--out", required=True, help="output parquet path")
    args = parser.parse_args(argv)

    pattern = os.path.join(args.root, "*", "*", f"resp_{args.horizon}.pq")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no files matched {pattern}", file=sys.stderr)
        return 1

    pieces = []
    for path in files:
        available = pq.read_schema(path).names
        if args.column not in available:
            print(f"{path}: no column {args.column!r}; it has "
                  f"{[c for c in available if c.startswith('resp_')][:6]}...", file=sys.stderr)
            return 1
        frame = pq.read_table(path, columns=["date", "stable_id", args.column]).to_pandas()
        if isinstance(frame.index, pd.MultiIndex) or frame.index.name is not None:
            frame = frame.reset_index()
        pieces.append(frame)

    out = pd.concat(pieces, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    if args.start:
        out = out[out["date"] >= pd.Timestamp(args.start)]
    if args.end:
        out = out[out["date"] <= pd.Timestamp(args.end)]
    out = out.dropna(subset=[args.column]).sort_values(["date", "stable_id"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False, compression="zstd")

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {len(out):,} rows to {args.out} ({size_mb:.1f} MB)")
    print(f"  column   {args.column}")
    print(f"  dates    {out['date'].min().date()} .. {out['date'].max().date()} "
          f"({out['date'].nunique():,} trading days)")
    print(f"  names    {out['stable_id'].nunique():,} unique, "
          f"{len(out) / max(out['date'].nunique(), 1):.0f} per date")
    print()
    print("point the config at it with:")
    print(f"  --set labels.forward_label_path={os.path.abspath(args.out)} \\")
    print(f"  --set labels.forward_label_column={args.column}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
