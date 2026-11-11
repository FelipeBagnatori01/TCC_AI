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
    openrca_repo: Path
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


class SelectionConfig(_Strict):
    max_missing_frac: float = Field(gt=0.0, le=1.0)
    near_constant_mode_share: float = Field(gt=0.0, le=1.0)
    # KPI categories never dropped for being (near-)constant in training; chosen a priori.
    constant_exempt_categories: tuple[str, ...] = ()


class SeriesConfig(_Strict):
    step_seconds: int = Field(gt=0)
    ffill_limit_steps: int = Field(ge=0)
    deployment_list: Path | None  # relative to paths.data_root; None when the system has none
    kpi_catalog: Path
    selection: SelectionConfig
    report_train_days: int = Field(gt=0)
    # Initial block (ADR 0012): training in every fold. Bank drain3 templates and native sampling
    # steps are measured on these days only; the ``log_top_templates`` most frequent are kept.
    initial_block_days: tuple[str, ...] = ()
    log_top_templates: int = Field(default=10, gt=0)
    # ADR 0012: per-feature forward-fill limit = native sampling step measured on the initial
    # block (initial_block_days), bounded by [ffill_limit_steps, max_ffill_steps]. Off for
    # Telecom (uniform 60 s sampling; ADR 0005).
    adaptive_ffill: bool = False
    max_ffill_steps: int = Field(default=60, gt=0)


class PrecursorConfig(_Strict):
    # Minutes between the scored minute and the failure onset: positive = before the onset
    # (precursor), zero = onset minute, negative = after the onset (detection).
    leads_minutes: tuple[int, ...] = Field(min_length=1)
    top_k: int = Field(gt=0)  # minute score = mean of the k largest |robust z|
    bootstrap: int = Field(gt=0)


class TasksConfig(_Strict):
    """E3: windows, labels and the rolling-origin split (ADR 0007/0008)."""

    excluded_days: tuple[str, ...] = ()  # removed from every split (ADR 0008)
    first_test_day: str  # days before it form the initial train/validation block
    val_days: int = Field(gt=0)  # last N days before each test day form validation
    window_minutes: int = Field(gt=0)  # W
    horizon_minutes: int = Field(ge=0)  # H: positive if an onset falls in [t, t + H]
    detection_tolerance_minutes: int = Field(ge=0)  # D: ... or in [t - D, t]
    normal_margin_before_minutes: int = Field(ge=0)
    normal_margin_after_minutes: int = Field(ge=0)
    precursor: PrecursorConfig


class LightGBMConfig(_Strict):
    n_estimators: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    num_leaves: int = Field(gt=1)
    min_child_samples: int = Field(gt=0)
    colsample_bytree: float = Field(gt=0.0, le=1.0)


class BaselinesConfig(_Strict):
    """E4 baselines and evaluation protocol (ADR 0010)."""

    zscore_threshold: float = Field(gt=0.0)  # fixed 3-sigma rule
    isolation_forest_estimators: int = Field(gt=0)
    lightgbm: LightGBMConfig
    # Threshold = best F1 on validation; without validation failures, this quantile of the
    # validation normal-window scores.
    val_fallback_quantile: float = Field(gt=0.0, lt=1.0)
    event_lookback_minutes: int = Field(ge=0)  # alarms this early still detect the failure
    event_max_delay_minutes: int = Field(gt=0)
    bootstrap: int = Field(gt=0)
    bootstrap_block_minutes: int = Field(gt=0)  # window-level CIs resample blocks of this size


class Config(_Strict):
    experiment: ExperimentConfig
    target_system: Literal["Telecom", "Bank"]
    # Every system whose raw files are listed (and verified) in the versioned data manifest.
    data_systems: tuple[Literal["Telecom", "Bank"], ...] = Field(min_length=1)
    manifest_extra_files: tuple[Path, ...] = ()  # data-root files outside system folders
    paths: PathsConfig
    llm: LLMConfig
    series: SeriesConfig
    tasks: TasksConfig
    baselines: BaselinesConfig

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
