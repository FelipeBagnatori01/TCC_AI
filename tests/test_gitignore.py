from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

IGNORED = [
    "data/raw/openrca/NotLinked/telemetry/x.csv",
    "data/interim/Telecom/day.parquet",
    "data/processed/windows.parquet",
    ".env",
    "models/lstm.pt",
    "weights.safetensors",
    "experiments/20260101T000000-e2e/metrics.json",
    "experiments/llm_cache/ab/abcdef.json",
]
TRACKED = ["data/MANIFEST.csv", ".env.example", "docs/RESULTADOS.md", "src/tcc_rca/cli.py"]


@pytest.mark.parametrize("path", IGNORED)
def test_sensitive_or_generated_paths_are_ignored(repo_root: Path, path: str) -> None:
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=repo_root, check=False)
    assert result.returncode == 0, f"{path} should be gitignored"


@pytest.mark.parametrize("path", TRACKED)
def test_versioned_paths_are_not_ignored(repo_root: Path, path: str) -> None:
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=repo_root, check=False)
    assert result.returncode == 1, f"{path} must not be gitignored"
