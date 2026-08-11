"""Command-line entry points: ``python -m gsm_alpha.cli <command>``.

    build-cache   stream the panels once and precompute per-date model inputs
    train         run the rolling yearly retrain and write the factor panel
    evaluate      score a factor panel (Rank IC, ICIR, quintile long-short)
    benchmark     measure step time on this machine and project the schedule
    info          print the resolved config, backend and cache status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

import pandas as pd

from .config import Config
from .signature import backend_report


def _parse_overrides(pairs: List[str]) -> Dict[str, Any]:
    """Turn ``--set train.max_epochs=5`` pairs into a dotted-key dictionary.

    Values are parsed as YAML scalars, so ``5``, ``0.1``, ``true``, ``null`` and
    ``[a, b]`` all arrive with the right type.
    """
    import yaml

    out: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        out[key.strip()] = yaml.safe_load(raw)
    return out


def _setup_logging(verbosity: str) -> None:
    logging.basicConfig(
        level=getattr(logging, verbosity.upper()),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(prog="gsm_alpha", description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="path to the YAML config")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override a config key, repeatable")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning"])
    sub = parser.add_subparsers(dest="command", required=True)

    cache = sub.add_parser("build-cache", help="precompute per-date model inputs")
    cache.add_argument("--stage", default="features", choices=["features", "windows", "both"])
    cache.add_argument("--stride", type=int, default=1,
                       help="cache every n-th trading day (1 = every day)")
    cache.add_argument("--overwrite", action="store_true", help="rebuild dates already cached")
    cache.add_argument("--threads", type=int, default=0, help="torch intra-op threads")

    train = sub.add_parser("train", help="rolling yearly retrain")
    train.add_argument("--kind", default="features", choices=["features", "windows"])

    evaluate = sub.add_parser("evaluate", help="score a factor panel")
    evaluate.add_argument("--factor", default=None, help="factor parquet (default: <output_dir>/factor.parquet)")
    evaluate.add_argument("--rebalance", default="monthly", choices=["monthly", "daily"])
    evaluate.add_argument("--groups", type=int, default=5)
    evaluate.add_argument(
        "--neutralize-factor", default=None, metavar="EXPOSURES.PARQUET",
        help="winsorise, standardise and industry/market-cap neutralise the factor before "
             "scoring — the report's section 3.3 preprocessing (its tables 12 and 14)",
    )
    evaluate.add_argument(
        "--against-label", action="store_true",
        help="score against labels.forward_label_path instead of the close-to-close return "
             "— the like-for-like comparison for a model trained on residual returns",
    )

    bench = sub.add_parser("benchmark", help="measure this machine's cache and training cost")
    bench.add_argument("--stocks", type=int, nargs="+", default=[1500, 3000, 4300])
    bench.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0")
    bench.add_argument("--precision", type=int, default=None, choices=[16, 32])
    bench.add_argument("--threads", type=int, default=0, help="CPU intra-op threads")

    sub.add_parser("info", help="print the resolved config and cache status")

    args = parser.parse_args(argv)
    _setup_logging(args.log_level)
    config = Config.from_yaml(args.config, _parse_overrides(args.overrides))

    if args.command == "build-cache":
        from .data.cache import CacheBuilder

        builder = CacheBuilder(config, threads=args.threads or None)
        builder.run(stage=args.stage, stride=args.stride, overwrite=args.overwrite)
        return 0

    if args.command == "train":
        from .train import run_rolling

        run_rolling(config, kind=args.kind)
        return 0

    if args.command == "evaluate":
        from .evaluate import evaluate as run_evaluate, format_report

        path = args.factor or os.path.join(config.train.output_dir, "factor.parquet")
        if args.against_label and not config.labels.forward_label_path:
            raise SystemExit(
                "--against-label needs labels.forward_label_path set in the config"
            )
        result = run_evaluate(
            pd.read_parquet(path),
            config.data.daily_path,
            horizon=config.labels.horizon,
            n_groups=args.groups,
            rebalance=args.rebalance,
            forward_return_path=config.labels.forward_label_path if args.against_label else None,
            forward_return_column=config.labels.forward_label_column if args.against_label else None,
            forward_return_scale=config.labels.forward_label_scale,
            neutralize_path=args.neutralize_factor,
        )
        print(format_report(result))
        return 0

    if args.command == "benchmark":
        from .benchmark import run as run_benchmark

        run_benchmark(config, stocks=args.stocks, device=args.device,
                      precision=args.precision, threads=args.threads)
        return 0

    if args.command == "info":
        from .data.cache import available_dates, read_manifest

        print(backend_report(config.minute_gsm.backend))
        print(json.dumps(config.to_dict(), indent=2, default=str))
        try:
            print("\ncache manifest:", json.dumps(read_manifest(config.data.cache_dir), indent=2))
            for branch in ("minute", "daily"):
                for kind in ("features", "windows"):
                    dates = available_dates(config.data.cache_dir, kind, branch)
                    if dates:
                        print(f"  {kind}/{branch}: {len(dates)} dates, {dates[0]} .. {dates[-1]}")
        except FileNotFoundError as exc:
            print(f"\ncache: {exc}")
        return 0

    raise SystemExit(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
