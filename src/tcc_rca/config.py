"""Typed configuration loaded from YAML. Unknown fields raise an explicit error."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentConfig(_Strict):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    seed: int = Field(ge=0)


class PathsConfig(_Strict):
    data_root: Path
    interim_dir: Path
    processed_dir: Path
    experiments_dir: Path
    manifest: Path


class LLMConfig(_Strict):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int = Field(default=1024, gt=0)
    budget_usd: float = Field(ge=0.0)
    price_input_per_mtok: float = Field(ge=0.0)
    price_output_per_mtok: float = Field(ge=0.0)
    cache_dir: Path


class Config(_Strict):
    experiment: ExperimentConfig
    target_system: Literal["Telecom", "Bank", "Market"]
    paths: PathsConfig
    llm: LLMConfig

    def resolved(self) -> dict[str, Any]:
        """JSON-serializable view of the fully resolved configuration."""
        return self.model_dump(mode="json")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(*paths: Path | str, overrides: Mapping[str, Any] | None = None) -> Config:
    """Load and merge YAML files in order (later wins), then apply ``overrides``."""
    if not paths:
        raise ValueError("load_config requires at least one YAML path")
    data: dict[str, Any] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: top-level YAML must be a mapping")
        data = deep_merge(data, loaded)
    if overrides:
        data = deep_merge(data, overrides)
    return Config.model_validate(data)
