"""Typed configuration loaded from YAML. Unknown fields raise an explicit error."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    price_cached_input_per_mtok: float = Field(ge=0.0)  # cache reads
    price_cache_write_per_mtok: float = Field(ge=0.0)  # cache writes (= input where free)
    price_output_per_mtok: float = Field(ge=0.0)  # includes reasoning tokens
    # Prompt-length pricing (e.g. Claude Haiku 5.5): a call whose prompt (full input + cache reads
    # + cache writes) exceeds the threshold pays every price times the multiplier. None = flat.
    long_prompt_threshold_tokens: int | None = Field(default=None, gt=0)
    long_prompt_price_multiplier: float = Field(default=1.0, ge=1.0)
    effort: str | None = None  # reasoning effort (provider-specific values), fixed per study
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
    # "local" method (ADR 0013): causal baseline over the previous minutes of the same day
    local_window_minutes: int = Field(gt=0)
    local_cap: float = Field(gt=0.0)
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
    # ADR 0019: unsupervised window scores (forecaster error, z-scores) read only the last N
    # minutes of the window, the span a positive label refers to (D + 1); None = the whole window
    # (the E4/E5 behaviour before ADR 0019, kept as a control).
    score_minutes: int | None = Field(gt=0)
    precursor: PrecursorConfig

    @model_validator(mode="after")
    def _score_span_fits(self) -> TasksConfig:
        # the forecaster has W - 1 error steps; the score cannot read more than that
        if self.score_minutes is not None and self.score_minutes > self.window_minutes - 1:
            raise ValueError("tasks.score_minutes must be at most window_minutes - 1")
        return self


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
    # ADR 0019: reference detector from the time of day alone (training windows within this
    # many minutes of the same minute of the day)
    time_prior_halfwidth_minutes: int = Field(gt=0, lt=720)


class LstmGridConfig(_Strict):
    hidden: tuple[int, ...] = Field(min_length=1)
    layers: tuple[int, ...] = Field(min_length=1)
    normalization: tuple[Literal["global", "local"], ...] = Field(min_length=1)


class LstmConfig(_Strict):
    """E5 (ADR 0015). Hyperparameters are searched on validation only, with this budget."""

    seeds: tuple[int, ...] = Field(min_length=1)  # offsets added to experiment.seed (finals)
    forecaster_grid: LstmGridConfig
    # The classifier reuses the normalization chosen for the forecaster; only its size is searched.
    classifier_hidden: tuple[int, ...] = Field(min_length=1)
    classifier_layers: tuple[int, ...] = Field(min_length=1)
    dropout: float = Field(ge=0.0, lt=1.0)
    learning_rate: float = Field(gt=0.0)
    batch_size: int = Field(gt=0)
    max_epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    global_clip: float = Field(gt=0.0)  # |robust z| cap for the "global" normalization
    score_top_k: int = Field(gt=0)  # minute score = mean of the k largest normalized errors
    # Control runs (e.g. without the availability exemption) reuse the main run's selection
    # instead of searching again: path to reports/tables/lstm_<system>_<name>.json (``{system}``
    # is replaced by the target system).
    reuse_selection_from: Path | None = None


class ExplainConfig(_Strict):
    """E6 (ADR 0016). Explains the E5 models of the run recorded in lstm_<system>_<lstm_name>."""

    lstm_name: str = Field(min_length=1)  # which E5 table/run to explain (lstm, lstm_sem_isencao)
    device: Literal["cpu", "mps"]
    explained_seed: int  # main explanations use this seed's models (offset, as in lstm.seeds)
    stability_seeds: tuple[int, ...]  # other seeds' models, SHAP only, for stability
    background_windows: int = Field(gt=0)
    shap_samples: int = Field(gt=0)
    lime_samples: int = Field(gt=1)
    lime_kernel_width: float = Field(gt=0.0)
    ig_steps: int = Field(gt=1)
    ig_windows_per_fold: int = Field(ge=0)
    top_k: int = Field(gt=0)  # agreement/stability/fidelity work on the top-k features
    deletion_steps: tuple[int, ...] = Field(min_length=1)
    random_orders: int = Field(gt=0)
    # window choice, score and alarm of the evidence package (best E5 detector, ADR 0021)
    evidence_detector: Literal["lstm_forecast", "lstm_classifier"]
    # model whose SHAP/LIME attributions are cited (stable across seeds, ADR 0016)
    evidence_explained: Literal["lstm_forecast", "lstm_classifier"]
    evidence_top_components: int = Field(gt=0)
    evidence_top_kpis: int = Field(gt=0)
    # Reason -> KPI categories counted as a hit for the suggested category, FIXED BEFORE MEASURING
    # (2026-11-12), per system.
    reason_categories: dict[str, dict[str, tuple[str, ...]]]


class AltModelsConfig(_Strict):
    """Alternative explainable models compared with the LSTM (ADR 0018).

    Same windows, folds, validation-only search (8 configurations per model, the forecaster's
    budget), final seeds (``lstm.seeds``) and XAI settings (``explain``) as E5/E6. ``family`` splits
    the run in two processes: LightGBM's OpenMP runtime must not meet multithreaded torch.
    """

    family: Literal["tabular", "tcn"]
    normalization: tuple[Literal["global", "local"], ...] = Field(min_length=1)
    lightgbm_num_leaves: tuple[int, ...] = Field(min_length=1)
    lightgbm_estimators: tuple[int, ...] = Field(min_length=1)  # other params: baselines.lightgbm
    iforest_estimators: tuple[int, ...] = Field(min_length=1)
    iforest_max_samples: tuple[int, ...] = Field(min_length=1)
    logistic_c: tuple[float, ...] = Field(min_length=1)
    logistic_max_iter: int = Field(gt=0)
    tcn_channels: tuple[int, ...] = Field(min_length=1)
    tcn_blocks: tuple[int, ...] = Field(min_length=1)
    tcn_kernel: int = Field(gt=1)  # training recipe (lr, batch, epochs, patience, dropout): lstm
    # reference runs for the paired comparisons (reports/tables/<kind>_<system>_<name>.json)
    lstm_name: str = Field(min_length=1)
    xai_name: str = Field(min_length=1)
    baselines_name: str = Field(min_length=1)
    # Control runs reuse the main run's selection: reports/tables/alt_models_{system}_<name>.json
    reuse_selection_from: Path | None = None
    max_folds: int | None = Field(default=None, gt=0)  # smoke runs only; None = every fold


class EvidenceSelectConfig(_Strict):
    """Evidence chosen by prequential selection: onset, component and reason (ADR 0020).

    The candidate lists are fixed in code (``evaluation.evidence_select``); only the constants
    fixed before measuring live here. Models come from the E5 run explained by ``explain``.
    """

    hazard_minutes: int = Field(gt=1)  # BOCPD constant hazard = 1 / hazard_minutes
    reference_minutes: int = Field(gt=0)  # pre-onset reference (C4/C5) and BOCPD lead-in
    min_reference_minutes: int = Field(gt=0)  # fewer -> the fold's normal training minutes
    min_onset_separation_minutes: int = Field(gt=0)  # second failure of a query
    pagerank_damping: float = Field(gt=0.0, lt=1.0)
    pagerank_backward_weight: float = Field(ge=0.0)
    pagerank_relations: tuple[str, ...] = Field(min_length=1)
    rrf_k: int = Field(gt=0)
    reason_logistic_c: float = Field(gt=0.0)
    # E6 packages the agent would receive today (the "current package" chain compared against)
    package_name: str = Field(min_length=1)
    max_folds: int | None = Field(default=None, gt=0)  # smoke runs only; None = every fold


class AgentConfig(_Strict):
    """E7 agent (ADR 0017). Model, effort and budget live in ``llm``."""

    variant: Literal["A1", "A2", "A3"]
    # test = the queries with an E6 evidence package (tested days); pilot = queries before the
    # first test day (no package): model/prompt choices are made there, never on the test set.
    case_set: Literal["test", "pilot"]
    repetitions: int = Field(ge=1)
    limit: int | None = Field(default=None, ge=1)  # first N cases (pilot runs)
    max_tool_calls: int = Field(gt=0)
    max_turns: int = Field(gt=0)
    max_retries: int = Field(ge=0)
    anomaly_top_k: int = Field(gt=0)
    evidence_name: str = Field(min_length=1)
    prompts_dir: Path


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
    lstm: LstmConfig
    explain: ExplainConfig
    alt_models: AltModelsConfig
    evidence_select: EvidenceSelectConfig
    agent: AgentConfig

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
