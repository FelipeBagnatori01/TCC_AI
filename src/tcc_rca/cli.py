"""Command-line entry point: ``tcc-rca <command> --config <yaml>``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tcc_rca.config import load_config
from tcc_rca.e2e import run_e2e


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tcc-rca")
    sub = parser.add_subparsers(dest="command", required=True)
    e2e = sub.add_parser("e2e", help="run the infrastructure smoke pipeline")
    e2e.add_argument("--config", type=Path, nargs="+", default=[Path("configs/base.yaml")])
    args = parser.parse_args(argv)

    if args.command == "e2e":
        config = load_config(*args.config)
        run_dir = run_e2e(config)
        print(f"run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
