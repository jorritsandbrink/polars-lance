default:
    @just --list

setup:
    uv sync --group dev

develop: setup
    uv run maturin develop

test-rust:
    cargo test

test: develop
    uv run pytest

check-rust:
    cargo fmt --check
    cargo check
    cargo check --features pyo3

check-pyth: setup
    uv run ruff check
    uv run ruff format --check
    uv run mypy

fmt-rust:
    cargo fmt

fmt-pyth:
    uv run ruff format
