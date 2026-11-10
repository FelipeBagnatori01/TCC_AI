"""Experiment tracking: one directory per run under ``experiments/<timestamp>-<name>/``."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import yaml

from tcc_rca.config import Config
from tcc_rca.env import capture_environment, dataset_hash, hash_config
from tcc_rca.seed import set_global_seed


class Run:
    """A tracked execution. Use as a context manager so the log handler is released."""

    def __init__(self, config: Config, root: Path | None = None) -> None:
        self.config = config
        self.started_at = datetime.now(UTC)
        stamp = self.started_at.strftime("%Y%m%dT%H%M%S")
        base = f"{stamp}-{config.experiment.name}"
        experiments_dir = config.paths.experiments_dir
        self.dir = experiments_dir / base
        suffix = 1
        while self.dir.exists():
            suffix += 1
            self.dir = experiments_dir / f"{base}-{suffix}"
        self.dir.mkdir(parents=True)
        self.artifacts_dir = self.dir / "artifacts"
        self.artifacts_dir.mkdir()
        self._metrics: dict[str, Any] = {}

        resolved = config.resolved()
        (self.dir / "config.resolved.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
        )
        self.provenance: dict[str, Any] = {
            "started_at": self.started_at.isoformat(),
            "seed": config.experiment.seed,
            "config_hash": hash_config(resolved),
            "dataset_hash": dataset_hash(config.paths.manifest),
            "environment": capture_environment(root),
        }
        self._write_json("provenance.json", self.provenance)

        self.logger = logging.getLogger(f"tcc_rca.run.{self.dir.name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._handler = logging.FileHandler(self.dir / "run.log", encoding="utf-8")
        self._handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(self._handler)
        set_global_seed(config.experiment.seed)
        self.logger.info("run started: %s (seed=%d)", self.dir.name, config.experiment.seed)

    def _write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Merge ``metrics`` into ``metrics.json`` (values must be JSON-serializable)."""
        self._metrics.update(metrics)
        self._write_json("metrics.json", self._metrics)

    def write_artifact(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.artifacts_dir / name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @property
    def llm_call_log(self) -> Path:
        return self.dir / "llm_calls.jsonl"

    def close(self) -> None:
        self.logger.info("run finished")
        self.logger.removeHandler(self._handler)
        self._handler.close()

    def __enter__(self) -> Run:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is not None:
            self.logger.error("run failed: %r", exc)
        self.close()
