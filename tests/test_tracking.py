from __future__ import annotations

import json
from pathlib import Path

import yaml

from tcc_rca.config import Config
from tcc_rca.env import hash_config
from tcc_rca.tracking import Run


def test_run_creates_expected_layout(config: Config) -> None:
    with Run(config) as run:
        run.log_metrics({"a": 1})
        run.log_metrics({"b": 2.5})
        run_dir = run.dir
    assert run_dir.parent == config.paths.experiments_dir
    assert run_dir.name.endswith("-e2e")
    assert json.loads((run_dir / "metrics.json").read_text()) == {"a": 1, "b": 2.5}
    assert (run_dir / "run.log").read_text().strip() != ""
    assert (run_dir / "artifacts").is_dir()
    saved = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    assert saved == config.resolved()


def test_provenance_records_hashes_and_environment(config: Config, repo_root: Path) -> None:
    with Run(config, root=repo_root) as run:
        prov = json.loads((run.dir / "provenance.json").read_text())
    assert prov["seed"] == config.experiment.seed
    assert prov["config_hash"] == hash_config(config.resolved())
    assert prov["dataset_hash"] is None  # no manifest yet
    env = prov["environment"]
    assert {"python", "torch", "numpy", "cpu_count", "git"} <= env.keys()
    assert "commit" in env["git"]


def test_dataset_hash_comes_from_manifest(config: Config) -> None:
    config.paths.manifest.write_text("path,size,sha256\n", encoding="utf-8")
    with Run(config) as run:
        assert run.provenance["dataset_hash"] is not None


def test_runs_started_in_same_second_do_not_collide(config: Config) -> None:
    with Run(config) as first, Run(config) as second:
        assert first.dir != second.dir
