default:
    @just --list

setup:
    uv sync --group dev --no-install-project

develop: setup
    uv run maturin develop

test-rust:
    cargo test

test: develop
    uv run pytest

test-no-docker: develop
    uv run pytest -m "not needs_docker"

test-versions:
    UV_PYTHON=3.10 just test
    UV_PYTHON=3.11 just test
    UV_PYTHON=3.12 just test
    UV_PYTHON=3.13 just test
    UV_PYTHON=3.14 just test

lint-rust:
    cargo fmt --check
    cargo check
    cargo check --features pyo3

lint-pyth: setup
    uv run --no-project ruff check
    uv run --no-project ruff format --check
    uv run --no-project mypy

lint-workflows:
    actionlint

fmt-rust:
    cargo fmt

fmt-pyth:
    uv run ruff format
