PY ?= python3.11
VENV := .venv
BIN := $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help install run api cycle roster backtest lint fmt clean up down logs rebuild shell

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## create the venv and install the backend (editable)
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e "backend[dev]"
	@echo "\nready. next: make run"

run: ## run the factory + control room on http://localhost:8000
	$(BIN)/python -m krish.main run

api: ## run only the API + control room (no agents)
	$(BIN)/python -m krish.main api

cycle: ## run exactly one research cycle end to end, then exit
	$(BIN)/python -m krish.main cycle --asset $(or $(ASSET),GOLD) \
	  --timeframe $(or $(TF),H1) --count $(or $(N),3)

roster: ## list the agents and what they do
	$(BIN)/python -m krish.main roster

backtest: ## backtest one IR file: make backtest FILE=path/to/strategy.json
	$(BIN)/python -m krish.main backtest $(FILE) --walk-forward

lint: ## ruff check
	$(BIN)/ruff check backend

fmt: ## ruff format
	$(BIN)/ruff format backend

clean: ## remove caches (keeps the database and delivered packages)
	rm -rf .ruff_cache .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

up: ## VPS: build and start the 24/7 stack
	docker compose up -d --build

down: ## VPS: stop the stack
	docker compose down

rebuild: ## VPS: rebuild the krish image and restart it
	docker compose up -d --build krish

logs: ## VPS: follow the factory log
	docker compose logs -f krish

shell: ## VPS: shell inside the running container
	docker compose exec krish bash
