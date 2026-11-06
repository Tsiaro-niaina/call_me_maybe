BASE ?= $(HOME)/goinfre
export HF_HOME ?= $(BASE)/huggingface_cache
export UV_CACHE_DIR ?= $(BASE)/uv_cache
export UV_LINK_MODE ?= copy
export UV_PROJECT_ENVIRONMENT ?= $(BASE)/call_me_maybe_venv

.PHONY: install run debug clean lint lint-strict

install:
	@uv sync

run:
	@uv run python -m src

debug:
	@uv run python -m pdb -m src

clean:
	@find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	@rm -rf .mypy_cache .pytest_cache

lint:
	@uv run flake8 .
	@uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@uv run flake8 .
	@uv run mypy . --strict