from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tcc_rca.config import load_config


def test_base_config_loads(repo_root: Path) -> None:
    cfg = load_config(repo_root / "configs/base.yaml")
    assert cfg.target_system == "Telecom"
    assert cfg.llm.temperature == 0.0
    assert cfg.llm.budget_usd > 0


def test_unknown_top_level_field_is_rejected(tmp_path: Path, repo_root: Path) -> None:
    with pytest.raises(ValidationError, match="bogus"):
        load_config(repo_root / "configs/base.yaml", overrides={"bogus": 1})


def test_unknown_nested_field_is_rejected(repo_root: Path) -> None:
    with pytest.raises(ValidationError, match="modle"):
        load_config(repo_root / "configs/base.yaml", overrides={"llm": {"modle": "x"}})


def test_invalid_target_system_is_rejected(repo_root: Path) -> None:
    with pytest.raises(ValidationError):
        load_config(repo_root / "configs/base.yaml", overrides={"target_system": "ESSVT"})


def test_later_files_override_earlier_and_merge_deeply(tmp_path: Path, repo_root: Path) -> None:
    extra = tmp_path / "extra.yaml"
    extra.write_text(yaml.safe_dump({"llm": {"budget_usd": 5.0}}), encoding="utf-8")
    cfg = load_config(repo_root / "configs/base.yaml", extra)
    assert cfg.llm.budget_usd == 5.0
    assert cfg.llm.model == "fake-model-v1"  # untouched sibling survives the merge


def test_config_is_immutable(repo_root: Path) -> None:
    cfg = load_config(repo_root / "configs/base.yaml")
    with pytest.raises(ValidationError):
        cfg.target_system = "Bank"  # type: ignore[misc]


def test_no_paths_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        load_config()
