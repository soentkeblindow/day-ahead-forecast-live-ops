.PHONY: help test lint backtest

.DEFAULT_GOAL := help

help:
	@echo "Available targets:"
	@echo "  test           Run the test suite (pytest)"
	@echo "  lint           Run ruff check, ruff format --check, and mypy (mirrors CI)"
	@echo "  backtest       Run the flagship LightGBM walk-forward backtest (needs data/processed/features.parquet)"

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src tests

backtest:
	uv run python scripts/backtest.py --model lgbm
