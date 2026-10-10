.DEFAULT_GOAL := help
UV ?= uv
RUN = $(UV) run

help: ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ -/' | sort

install: ## create .venv and install runtime + dev dependencies
	$(UV) sync --extra dev

test: ## run the test-suite
	$(RUN) pytest -q

cov: ## tests with coverage report
	$(RUN) pytest -q --cov=botux --cov-report=term-missing:skip-covered

lint: ## static analysis + formatting check (no changes made)
	$(RUN) ruff check .
	$(RUN) ruff format --check .

typecheck: ## strict type checking of src and tests
	$(RUN) mypy

format: ## auto-format the code
	$(RUN) ruff format .
	$(RUN) ruff check . --fix

check: lint typecheck test ## everything CI-like: lint + types + tests

paper: ## PAPER trading loop (simulated fills, public data only). Ctrl+C to stop
	$(RUN) botux paper

backtest: ## train/test back-test on the last 365 days (public data)
	$(RUN) botux backtest --days 365

.PHONY: help install test cov lint typecheck format check paper backtest
