SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

UV    ?= uv
RUFF  ?= ruff
PYTEST ?= pytest
MODELS_DIR ?= ./models

MODEL_SPECS ?= models/dummy_model.pkl
FEATURES ?= 1.0 2.0 3.0
STRATEGY ?= mean

.PHONY: help sync format check lint test run build clean preflight

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

sync: ## Install/sync dependencies including dev tools
	$(UV) sync --dev

lint: ## Format and auto-fix lint issues
	$(UV) run $(RUFF) format .
	$(UV) run $(RUFF) check . --fix

test: ## Run tests
	$(UV) run $(PYTEST) -q

run: ## Run sample multi-agent prediction
	$(UV) run ma-pred --model $(MODEL_SPECS) --features $(FEATURES) --strategy $(STRATEGY)

build: ## Build wheel and source distribution
	$(UV) build

preflight: ## Build and verify package metadata
	$(UV) build
	uvx twine check dist/*

d.models: ## Download all starter models into ./models
	$(UV) run $(PYTHON) scripts/model_downloader.py --all --output-dir $(MODELS_DIR)

d.tinyllama: ## Download TinyLlama into ./models
	$(UV) run $(PYTHON) scripts/model_downloader.py --model tinyllama --output-dir $(MODELS_DIR)

d.llama32: ## Download Llama 3.2 1B into ./models
	$(UV) run $(PYTHON) scripts/model_downloader.py --model llama32_1b --output-dir $(MODELS_DIR)

d.deepseek: ## Download DeepSeek Coder 1.3B into ./models
	$(UV) run $(PYTHON) scripts/model_downloader.py --model deepseek_coder_1_3b --output-dir $(MODELS_DIR)


clean: ## Remove build/test artifacts
	rm -rf dist build .pytest_cache .ruff_cache *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete