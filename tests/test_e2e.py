from __future__ import annotations

import json
from pathlib import Path

from tcc_rca.config import Config, load_config
from tcc_rca.e2e import run_e2e


def _fresh_config(repo_root: Path, tmp_path: Path, name: str) -> Config:
    return load_config(
        repo_root / "configs/base.yaml",
        overrides={
            "paths": {"experiments_dir": str(tmp_path / name / "experiments")},
            "llm": {"cache_dir": str(tmp_path / name / "cache")},
        },
    )


def test_two_runs_with_same_seed_have_identical_metrics(repo_root: Path, tmp_path: Path) -> None:
    first = run_e2e(_fresh_config(repo_root, tmp_path, "a"))
    second = run_e2e(_fresh_config(repo_root, tmp_path, "b"))
    assert (first / "metrics.json").read_bytes() == (second / "metrics.json").read_bytes()


def test_different_seed_changes_metrics(repo_root: Path, tmp_path: Path) -> None:
    base = _fresh_config(repo_root, tmp_path, "a")
    other = load_config(
        repo_root / "configs/base.yaml",
        overrides={
            "experiment": {"seed": base.experiment.seed + 1},
            "paths": {"experiments_dir": str(tmp_path / "c" / "experiments")},
            "llm": {"cache_dir": str(tmp_path / "c" / "cache")},
        },
    )
    a = json.loads((run_e2e(base) / "metrics.json").read_text())
    b = json.loads((run_e2e(other) / "metrics.json").read_text())
    assert a != b


def test_e2e_second_llm_call_hits_cache(repo_root: Path, tmp_path: Path) -> None:
    run_dir = run_e2e(_fresh_config(repo_root, tmp_path, "a"))
    usage = json.loads((run_dir / "artifacts/llm_usage.json").read_text())
    assert usage["calls"] == 2
    assert usage["cache_hits"] == 1
    assert usage["paid_calls"] == 1
