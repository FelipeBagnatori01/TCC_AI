.PHONY: setup lint test e2e clean-experiments

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
