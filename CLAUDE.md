# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Undergraduate thesis (UNIFESP): LSTM failure detection/prediction on telecom observability KPIs, SHAP/LIME explainability, and an LLM agent that explains the root cause (timestamp, component, reason) — evaluated on the public OpenRCA benchmark (Telecom subset is the main scenario).

The full spec lives **outside** this repo: `../instrucoes_implementacao_tcc.md` (Portuguese; stages E0–E9, research questions, invariant rules R1–R11). Read it before starting a stage. Progress notes: `../acompanhamento_etapas_tcc_1.md`.

**Workflow rule from the spec:** work one stage at a time, in order. At the start of a stage present the plan and open questions before writing code; at the end report summary, changed files, how to run, acceptance criteria with evidence, ADRs, blockers, and wait for approval. Never mark a criterion met without evidence; never skip/silence tests or relax lint. Do not assume OpenRCA file or column names — inventory them (E1). Missing data/access goes to `docs/BLOQUEIOS.md`, not invented.

Status: E0 (bootstrap/infrastructure) is done and committed. `ingestion/`, `series/`, `dataset/`, `models/`, `explain/`, `agent/`, `evaluation/` are empty placeholder packages for E1–E8.

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

`make e2e` accepts `CONFIG=...`; the CLI (`tcc-rca e2e --config a.yaml b.yaml`) merges multiple YAMLs left to right.

## Architecture (E0 infrastructure)

Everything experimental flows through three pieces that must be used together to satisfy the reproducibility rules (R5–R7):

- **Config** (`config.py`): pydantic v2 models, `extra="forbid"` and frozen. `load_config(*yamls, overrides=...)` deep-merges files then overrides; an unknown key at any depth is a hard error. Adding a config field means adding it to the model *and* `configs/base.yaml`. Tests redirect all output paths to `tmp_path` via the `config` fixture in `tests/conftest.py`.
- **Run tracking** (`tracking.py`, `env.py`, `seed.py`): `Run(config)` is a context manager that creates `experiments/<timestamp>-<name>/` (config.resolved.yaml, provenance.json with commit/config/dataset hashes and hardware, run.log, metrics.json, artifacts/) and **seeds Python/NumPy/PyTorch and enables `torch.use_deterministic_algorithms(True)`**. Dataset hash comes from the versioned `data/MANIFEST.csv`. Metrics that must be reproducible across runs go in `metrics.json`; anything that depends on cache state or wall-clock (LLM cost/latency) goes in artifacts, so two same-seed runs produce byte-identical `metrics.json`.
- **LLM client** (`llm/`): `LLMClient` wraps an `LLMProvider` (Protocol with `complete(request) -> ProviderResult`) chosen by name via a registry (`register_provider` / `build_provider`). Only the offline `fake` provider exists; real adapters are added when a provider is chosen (before E7) and swapping must stay a config change. Cache key = SHA-256 of canonical JSON of (provider, model, messages, temperature, max_tokens) — anything affecting output must be in `LLMRequest`. Cache hits cost 0, skip the provider and budget check, but are still logged to the JSONL call log. Budget cap is checked *before* each paid call (one call may overshoot).

## Conventions and constraints

- Code, identifiers, docstrings in English; docs, ADRs, reports and LLM-agent explanations in Portuguese. Non-obvious decisions get an ADR in `docs/adr/`.
- Telemetry (`data/raw`, `data/interim`, `data/processed`), `experiments/*` (including the LLM cache), `.env`, and model weights are gitignored; only `data/MANIFEST.csv` is versioned. OpenRCA telemetry is CC BY-NC 4.0 and must never be committed. `tests/test_gitignore.py` guards this.
- Invariants that later stages must respect (tests required): no temporal leakage (chronological day splits with guard gap, rolling-origin for the 51 Telecom cases, transformers fit on train only, forward-fill only); no label leakage to the LLM agent (`record.csv`/ground truth unreachable by agent tools, prompts contain no test labels); OpenRCA timestamps are UTC+8; the agent must be compared to the constant baseline and to the same LLM without evidence; agent results are scored with the official OpenRCA script, not a reimplementation.
- Dependencies for later stages (polars, shap, lime, captum, lightgbm, scikit-learn, drain3) are intentionally not installed yet; add them with the stage that uses them.
- Pre-commit runs ruff, ruff format check and mypy; ruff rule set is in `pyproject.toml` (line length 100).
