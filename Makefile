.PHONY: setup lint test e2e clean-experiments audit-repo reproduce reproduce-agent reproduce-full reproduce-check notebooks

CONFIG ?= configs/base.yaml

setup:
	uv sync
	@test -f .env || cp .env.example .env
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

test:
	uv run pytest

e2e:
	uv run tcc-rca e2e --config $(CONFIG)

audit-repo:          # E9: every commit scanned for API keys, telemetry and large files
	uv run tcc-rca repo-audit

reproduce:           # E9: thesis chain E1-E8 for both systems (CONFIG may add a local paths file)
	uv run tcc-rca reproduce --level principal --config $(CONFIG)

reproduce-agent:     # E9: agent runs from the LLM cache (after `tcc-rca cache-restore`), no API key
	uv run tcc-rca reproduce --level agente-cache --config $(CONFIG)

reproduce-full:      # E9: principal + E1 reports, sensitivity controls and alternative models
	uv run tcc-rca reproduce --level completo --config $(CONFIG)

reproduce-check:     # E9: regenerated tables vs. the committed ones
	uv run tcc-rca reproduce-check --config $(CONFIG)

notebooks:           # re-execute every analysis notebook (they read reports/tables only)
	for nb in notebooks/*.ipynb; do uv run jupyter nbconvert --to notebook --execute --inplace $$nb || exit 1; done
