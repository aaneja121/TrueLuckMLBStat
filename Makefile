.PHONY: setup format lint typecheck test check download-sample clean-data train demo notebook

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	@if [ ! -d "$(VENV)" ]; then python3 -m venv $(VENV); fi
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

format:
	$(PY) -m ruff format src tests

lint:
	$(PY) -m ruff check src tests

typecheck:
	$(PY) -m mypy src

test:
	$(PY) -m pytest

check: format lint typecheck test

download-sample:
	$(PY) -m mlb_luck_score.data.download_statcast \
		--start-date 2024-04-01 \
		--end-date 2024-04-07 \
		--output data/raw/statcast_2024_sample.parquet

clean-data:
	$(PY) -m mlb_luck_score.data.clean_batted_balls \
		--input data/raw/statcast_2024_sample.parquet \
		--output data/processed/cleaned_batted_balls.parquet

train:
	$(PY) -m mlb_luck_score.models.train_contact_model \
		--input data/processed/cleaned_batted_balls.parquet \
		--output-dir artifacts

demo:
	$(PY) -m mlb_luck_score.models.predict_outcomes --demo

notebook:
	$(PY) -m jupyter notebook notebooks
