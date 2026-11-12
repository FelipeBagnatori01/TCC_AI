"""Command-line entry point: ``tcc-rca <command> [--config <yaml> ...]``."""

from __future__ import annotations

import argparse
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from tcc_rca.config import Config, load_config
from tcc_rca.dataset.build import run_tasks
from tcc_rca.e2e import run_e2e
from tcc_rca.evaluation.baselines import run_baselines
from tcc_rca.evaluation.compare import run_e5_comparison
from tcc_rca.evaluation.lstm import run_lstm
from tcc_rca.evaluation.xai import run_xai
from tcc_rca.ingestion.cases import Case, Failure, build_cases, read_failures, read_queries
from tcc_rca.ingestion.eda import plot_failure
from tcc_rca.ingestion.inventory import build_inventory, write_inventory
from tcc_rca.ingestion.manifest import (
    build_manifest,
    read_manifest,
    verify_manifest,
    write_manifest,
)
from tcc_rca.ingestion.quality import build_quality_report, day_similarity, write_quality_report
from tcc_rca.ingestion.telemetry import convert_day, list_days
from tcc_rca.series.build import run_series

TABLES_DIR = Path("reports/tables")
FIGURES_DIR = Path("reports/figures")


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024  # macOS: bytes


def _system_dir(config: Config) -> Path:
    return config.paths.data_root / config.target_system


def _cases(config: Config) -> tuple[list[Case], list[Failure]]:
    system_dir = _system_dir(config)
    queries = read_queries(system_dir / "query.csv")
    failures = read_failures(system_dir / "record.csv")
    return build_cases(queries, failures)


def cmd_manifest(config: Config, _: argparse.Namespace) -> int:
    entries = build_manifest(
        config.paths.data_root,
        list(config.data_systems),
        extra_files=list(config.manifest_extra_files),
    )
    write_manifest(entries, config.paths.manifest)
    gigabytes = sum(e.size_bytes for e in entries) / 1e9
    print(f"{len(entries)} files, {gigabytes:.2f} GB -> {config.paths.manifest}")
    return 0


def cmd_verify(config: Config, args: argparse.Namespace) -> int:
    report = verify_manifest(
        config.paths.data_root, read_manifest(config.paths.manifest), check_hashes=not args.quick
    )
    for label, items in (
        ("missing", report.missing),
        ("size mismatch", report.size_mismatch),
        ("checksum mismatch", report.checksum_mismatch),
        ("unlisted (extra)", report.unlisted),
    ):
        print(f"{label}: {len(items)}")
        for item in items[:20]:
            print(f"  {item}")
    print("OK" if report.ok else "INTEGRITY FAILURE")
    return 0 if report.ok else 1


def cmd_inventory(config: Config, _: argparse.Namespace) -> int:
    inventory = build_inventory(_system_dir(config))
    out = TABLES_DIR / f"inventory_{config.target_system}.json"
    write_inventory(inventory, out)
    print(f"{inventory['n_days']} days -> {out}")
    return 0


def cmd_convert(config: Config, args: argparse.Namespace) -> int:
    system_dir = _system_dir(config)
    days = args.days or list_days(system_dir)
    for day in days:
        start = time.perf_counter()
        convert_day(system_dir, config.target_system, day, config.paths.interim_dir, args.overwrite)
        print(
            f"{day}: {time.perf_counter() - start:.1f}s (peak RSS so far {_peak_rss_mb():.0f} MB)"
        )
    print(f"peak_rss_mb={_peak_rss_mb():.0f}")
    return 0


def cmd_cases(config: Config, _: argparse.Namespace) -> int:
    cases, orphans = _cases(config)
    n_failures = sum(len(c.failures) for c in cases)
    print(f"queries={len(cases)} failures_assigned={n_failures} orphans={len(orphans)}")
    print("queries without failures:", [c.query.row for c in cases if not c.failures])
    print("failures per query:", sorted({len(c.failures) for c in cases}))
    return 0


def cmd_quality(config: Config, _: argparse.Namespace) -> int:
    cases, orphans = _cases(config)
    report = build_quality_report(
        config.paths.interim_dir, config.target_system, cases, len(orphans)
    )
    name = config.target_system
    write_quality_report(
        report, TABLES_DIR / f"quality_{name}.json", TABLES_DIR / f"quality_{name}.md"
    )
    print(f"failures without metric coverage: {len(report['failures_without_metric_coverage'])}")
    return 0


def cmd_similarity(config: Config, _: argparse.Namespace) -> int:
    days = list_days(_system_dir(config))
    table = day_similarity(config.paths.interim_dir, config.target_system, days)
    out = TABLES_DIR / f"day_similarity_{config.target_system}.csv"
    table.write_csv(out)
    print(table.head(8))
    print(f"-> {out}")
    return 0


def cmd_eda(config: Config, args: argparse.Namespace) -> int:
    cases, _orphans = _cases(config)
    failures = [f for c in cases for f in c.failures]
    done = 0
    for failure in failures:
        if done >= args.n:
            break
        stamp = failure.start_utc.strftime("%Y%m%dT%H%M")
        out = FIGURES_DIR / f"eda_{config.target_system}_{stamp}_{failure.component}.png"
        if plot_failure(config.paths.interim_dir, config.target_system, failure, out):
            done += 1
            print(out)
    return 0


def cmd_series(config: Config, _: argparse.Namespace) -> int:
    start = time.perf_counter()
    summary = run_series(config, TABLES_DIR)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"elapsed_s={time.perf_counter() - start:.1f} peak_rss_mb={_peak_rss_mb():.0f}")
    return 0


def cmd_tasks(config: Config, _: argparse.Namespace) -> int:
    start = time.perf_counter()
    summary = run_tasks(config, TABLES_DIR)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"elapsed_s={time.perf_counter() - start:.1f} peak_rss_mb={_peak_rss_mb():.0f}")
    return 0


def cmd_baselines(config: Config, _: argparse.Namespace) -> int:
    start = time.perf_counter()
    summary = run_baselines(config, TABLES_DIR)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"elapsed_s={time.perf_counter() - start:.1f} peak_rss_mb={_peak_rss_mb():.0f}")
    return 0


def cmd_lstm(config: Config, _: argparse.Namespace) -> int:
    start = time.perf_counter()
    summary = run_lstm(config, TABLES_DIR)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"elapsed_s={time.perf_counter() - start:.1f} peak_rss_mb={_peak_rss_mb():.0f}")
    return 0


def cmd_compare_e5(config: Config, _: argparse.Namespace) -> int:
    summary = run_e5_comparison(config, TABLES_DIR)
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


def cmd_xai(config: Config, _: argparse.Namespace) -> int:
    start = time.perf_counter()
    summary = run_xai(config, TABLES_DIR)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"elapsed_s={time.perf_counter() - start:.1f} peak_rss_mb={_peak_rss_mb():.0f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tcc-rca")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", type=Path, nargs="+", default=[Path("configs/base.yaml")])
        return p

    add("e2e", "run the infrastructure smoke pipeline")
    add("manifest", "write data/MANIFEST.csv for the target system")
    verify = add("verify", "check local data against the manifest")
    verify.add_argument("--quick", action="store_true", help="sizes only, skip checksums")
    add("inventory", "inventory raw telemetry structure (schemas, rows, components, KPIs)")
    convert = add("convert", "convert raw CSVs to parquet in data/interim (lazy, per day)")
    convert.add_argument("--days", nargs="+", help="e.g. 2020_04_11 (default: all)")
    convert.add_argument("--overwrite", action="store_true")
    add("cases", "parse record.csv/query.csv and report the failure-to-query mapping")
    add("quality", "write the data-quality report")
    add("similarity", "share of equal metric values between every pair of days (duplicate check)")
    add("series", "build the multivariate KPI matrix, catalog and dependency edges (E2)")
    add("tasks", "windows, labels, rolling-origin folds and precursor check (E3)")
    add("baselines", "run and evaluate every E4 baseline over the folds (tracked run)")
    add("lstm", "search, train and evaluate the LSTM forecaster and classifier (E5)")
    add("compare-e5", "paired LSTM-vs-baseline differences for an E5 run (bootstrap CIs)")
    add("xai", "SHAP/LIME/IG explanations, XAI localization and evidence packages (E6)")
    eda = add("eda", "plot KPIs around failures")
    eda.add_argument("-n", type=int, default=10, help="number of failures to plot")

    args = parser.parse_args(argv)
    config = load_config(*args.config)
    handlers = {
        "e2e": lambda c, _a: (print(f"run dir: {run_e2e(c)}"), 0)[1],
        "manifest": cmd_manifest,
        "verify": cmd_verify,
        "inventory": cmd_inventory,
        "convert": cmd_convert,
        "cases": cmd_cases,
        "quality": cmd_quality,
        "similarity": cmd_similarity,
        "eda": cmd_eda,
        "series": cmd_series,
        "tasks": cmd_tasks,
        "baselines": cmd_baselines,
        "lstm": cmd_lstm,
        "compare-e5": cmd_compare_e5,
        "xai": cmd_xai,
    }
    return int(handlers[args.command](config, args))


if __name__ == "__main__":
    raise SystemExit(main())
