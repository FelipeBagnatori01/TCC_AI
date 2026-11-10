from __future__ import annotations

from pathlib import Path

import pytest

from tcc_rca.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Base config with every output path redirected into a temp dir."""
    return load_config(
        REPO_ROOT / "configs/base.yaml",
        overrides={
            "paths": {
                "experiments_dir": str(tmp_path / "experiments"),
                "manifest": str(tmp_path / "MANIFEST.csv"),
            },
            "llm": {"cache_dir": str(tmp_path / "llm_cache")},
        },
    )
