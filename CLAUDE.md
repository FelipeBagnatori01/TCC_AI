# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Undergraduate thesis (UNIFESP): LSTM failure detection/prediction on telecom observability KPIs, SHAP/LIME explainability, and an LLM agent that explains the root cause (timestamp, component, reason) — evaluated on the public OpenRCA benchmark (Telecom subset is the main scenario).

The full spec lives **outside** this repo: `../instrucoes_implementacao_tcc.md` (Portuguese; stages E0–E9, research questions, invariant rules R1–R11). Read it before starting a stage. Progress notes: `../acompanhamento_etapas_tcc_1.md`.

**Workflow rule from the spec:** work one stage at a time, in order. At the start of a stage present the plan and open questions before writing code; at the end report summary, changed files, how to run, acceptance criteria with evidence, ADRs, blockers, and wait for approval. Never mark a criterion met without evidence; never skip/silence tests or relax lint. Do not assume OpenRCA file or column names — inventory them (E1). Missing data/access goes to `docs/BLOQUEIOS.md`, not invented.

Status: E0 (infrastructure) approved and committed; E1 committed; E2 committed; E3 (tasks, labels, temporal split) committed with a GO decision; E4 (baselines + full evaluation protocol) committed. Plan B triggered (ADR 0009 addendum): Bank is being added as a second system (E1–E4 redone for it, logs included); Market is out of scope. `models/lstm/`, `explain/`, `agent/` are empty placeholder packages for E5–E7.

## Commands

Requires [uv](https://docs.astral.sh/uv/) (Python 3.11 is pinned and installed by uv).

```bash
make setup                      # uv sync, create .env from .env.example, install pre-commit hook
make lint                       # ruff check + ruff format --check + mypy (strict, src/ only)
make test                       # pytest
make e2e                        # smoke pipeline -> experiments/<timestamp>-e2e/
uv run pytest tests/test_llm_client.py::test_budget_cap_stops_paid_calls_but_not_cache_hits
uv run ruff format .            # apply formatting (lint only checks)
```

Data pipeline (E1; all take `--config`, target system comes from config):

```bash
uv run tcc-rca manifest | verify [--quick]   # data/MANIFEST.csv + integrity check
uv run tcc-rca inventory                      # reports/tables/inventory_<system>.json
uv run tcc-rca convert [--days 2020_04_11]    # raw CSV -> data/interim parquet (UTC)
uv run tcc-rca cases | quality | eda -n 10    # label mapping, quality report, figures
uv run tcc-rca series                         # E2: matrix -> data/processed, catalog/edges -> reports/tables
uv run tcc-rca tasks                          # E3: windows/labels -> data/processed, folds/precursor -> reports/tables
uv run tcc-rca baselines --config configs/base.yaml configs/experiment/baselines.yaml   # E4 (also run baselines_sem_isencao.yaml)
```

Analysis notebook (E1–E4 findings and baseline charts; needs the E2–E4 artifacts): `notebooks/analise_telecom_e1_e4.ipynb`. Re-execute with `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/analise_telecom_e1_e4.ipynb`; ruff also lints/formats notebooks.

Official OpenRCA scorer: run from `external/OpenRCA` as `uv run --project ../.. python -m main.evaluate -p <pred.csv> -q <query.csv> -r <report.csv>`.

`make e2e` accepts `CONFIG=...`; the CLI (`tcc-rca e2e --config a.yaml b.yaml`) merges multiple YAMLs left to right.

## Architecture (E0 infrastructure)

Everything experimental flows through three pieces that must be used together to satisfy the reproducibility rules (R5–R7):

- **Config** (`config.py`): pydantic v2 models, `extra="forbid"` and frozen. `load_config(*yamls, overrides=...)` deep-merges files then overrides; an unknown key at any depth is a hard error. Adding a config field means adding it to the model *and* `configs/base.yaml`. Tests redirect all output paths to `tmp_path` via the `config` fixture in `tests/conftest.py`.
- **Run tracking** (`tracking.py`, `env.py`, `seed.py`): `Run(config)` is a context manager that creates `experiments/<timestamp>-<name>/` (config.resolved.yaml, provenance.json with commit/config/dataset hashes and hardware, run.log, metrics.json, artifacts/) and **seeds Python/NumPy/PyTorch and enables `torch.use_deterministic_algorithms(True)`**. Dataset hash comes from the versioned `data/MANIFEST.csv`. Metrics that must be reproducible across runs go in `metrics.json`; anything that depends on cache state or wall-clock (LLM cost/latency) goes in artifacts, so two same-seed runs produce byte-identical `metrics.json`.
- **LLM client** (`llm/`): `LLMClient` wraps an `LLMProvider` (Protocol with `complete(request) -> ProviderResult`) chosen by name via a registry (`register_provider` / `build_provider`). Only the offline `fake` provider exists; real adapters are added when a provider is chosen (before E7) and swapping must stay a config change. Cache key = SHA-256 of canonical JSON of (provider, model, messages, temperature, max_tokens) — anything affecting output must be in `LLMRequest`. Cache hits cost 0, skip the provider and budget check, but are still logged to the JSONL call log. Budget cap is checked *before* each paid call (one call may overshoot).

## Data layout (E1)

- OpenRCA repo is a git submodule at `external/OpenRCA` (pinned commit in ADR 0003; excluded from ruff; never edit it — run ruff/format only via `make`, not on `external/`). Clone with `--recurse-submodules`.
- Raw Telecom data lives outside the repo (`../Telecom`) and is symlinked at `data/raw/openrca/Telecom` (git refuses paths beyond that symlink, so gitignore tests use `Bank/`). Real Telecom facts (6 h/day 00:00–06:00 UTC+8, 15 days, 60 s metric step, near-duplicate days) are in `docs/DICIONARIO_DADOS.md`.
- `ingestion/timeutil.py` is the only place that converts UTC+8 ↔ UTC (ADR 0002). `ingestion/cases.py` exposes agent-visible query data only; ground truth (`scoring_points`) is read solely via `ingestion/labels.py`, which `tcc_rca.agent` must never import (tested).
- E2 series (ADR 0005/0006): features are named `component::kpi`; every KPI must have an entry in `configs/data/kpi_catalog_telecom.yaml` (including `item_agg` for multi-item KPIs) or the build fails. `FeatureSelector`/`RobustScaler` take a boolean `train_rows` mask and must only ever be fitted on training rows (leakage test in `tests/test_series_transforms.py`). The `disponibilidade` KPI category is exempt from constancy dropping; that choice was made after seeing labels, so every headline metric from E3 on must also be reported with the control `configs/experiment/sensitivity_sem_isencao.yaml` (threat log: `docs/AMEACAS_VALIDADE.md`). Dependency edges merge trace calls with `AppTelecomDeploymentList.xlsx` (symlinked under `data/raw/openrca/`).
- E3 (ADR 0007-0009): detection framing (H=0, no precursor found), W=20, D=5; windows never cross days; labels 1/0/null (null = near a failure but outside the positive band; never trained on as normal nor scored). Days 2020_04_21/22/23 are unlabelled copies of May days and are excluded everywhere; 2020_04_11 + 2020_04_20 are the initial train/val block, so evaluation uses the 45 May failures over 10 rolling-origin folds (one test day each, last prior day = validation). `tcc_rca.dataset` reads `record.csv` and is ground truth: the agent-isolation test forbids it, `read_failures`, `build_cases`, `record.csv` and `windows.parquet` in `tcc_rca.agent`. Day strings in YAML must be quoted (YAML reads `2020_04_21` as an integer).
- E4 (ADR 0010): `tcc_rca.evaluation` holds every metric (hand-computed tests in `tests/test_metrics.py`), cluster bootstrap CIs and the official-scorer wrapper (`evaluation/openrca.py` writes a filtered `query.csv` into the run dir and calls `main.evaluate`; never reimplement it). Per fold, selection/z/robust scaling fit only on normal minutes of training days; thresholds come from the validation day (best F1, or the 0.99 quantile of normal validation scores when it has no failure). Report event recall only next to its chance level (`event_recall_by_chance`): with tens of false alarms per hour it is meaningless. LightGBM must stay `num_threads=1` (multithreaded LightGBM aborts with an OMP error once torch's libomp is loaded, which `Run` does). `sklearn` has a scoped mypy `ignore_missing_imports`.
- Telemetry readers enforce explicit schemas and fail on header drift; unit tests use the synthetic system in `tests/synthetic.py`, never real data.

## Conventions and constraints

- Code, identifiers, docstrings in English; docs, ADRs, reports and LLM-agent explanations in Portuguese. Non-obvious decisions get an ADR in `docs/adr/`.
- Telemetry (`data/raw`, `data/interim`, `data/processed`), `experiments/*` (including the LLM cache), `.env`, and model weights are gitignored; only `data/MANIFEST.csv` is versioned. OpenRCA telemetry is CC BY-NC 4.0 and must never be committed. `tests/test_gitignore.py` guards this.
- Invariants that later stages must respect (tests required): no temporal leakage (chronological day splits with guard gap, rolling-origin for the 51 Telecom cases, transformers fit on train only, forward-fill only); no label leakage to the LLM agent (`record.csv`/ground truth unreachable by agent tools, prompts contain no test labels); OpenRCA timestamps are UTC+8; the agent must be compared to the constant baseline and to the same LLM without evidence; agent results are scored with the official OpenRCA script, not a reimplementation.
- Dependencies for later stages (shap, lime, captum, drain3) are intentionally not installed yet; add them with the stage that uses them.
- Pre-commit runs ruff, ruff format check and mypy; ruff rule set is in `pyproject.toml` (line length 100).
