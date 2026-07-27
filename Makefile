.PHONY: setup format lint typecheck test check download-sample clean-data \
	download-development-data clean-development-data train compare-models \
	build-reference-score demo notebook

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

# Downloads full regular-season Statcast data for 2021-2024 (one raw Parquet
# file per season). Requires internet access and can take a while -- see
# README.md "Full development dataset" for storage/runtime estimates before
# running this for real. Resumable: safe to re-run after an interruption,
# already-downloaded seasons are skipped unless --overwrite is passed. Never
# touches the one-week sample from `make download-sample`.
download-development-data:
	$(PY) -m mlb_luck_score.data.download_development_data \
		--seasons 2021 2022 2023 2024 \
		--output-dir data/raw

# Combines the per-season raw files from `make download-development-data`
# into one cleaned, deduplicated processed file spanning 2021-2024. Fails
# clearly if any season's raw file is missing. Never touches
# data/processed/cleaned_batted_balls.parquet (the one-week sample output).
clean-development-data:
	$(PY) -m mlb_luck_score.data.clean_development_data \
		--seasons 2021 2022 2023 2024 \
		--raw-dir data/raw \
		--output data/processed/cleaned_development_data.parquet

# Prefers the full 2021-2024 combined dataset when available (from
# `make clean-development-data`), falling back to the one-week sample (which
# has 0 rows in the 2021-2023 training window, so training will report 0
# training-eligible rows and exit non-zero).
train:
	@if [ -f data/processed/cleaned_development_data.parquet ]; then \
		echo "Training on the full development dataset (2021-2024)"; \
		INPUT=data/processed/cleaned_development_data.parquet; \
	else \
		echo "Full development dataset not found; falling back to the one-week sample" \
			"(run 'make download-development-data && make clean-development-data' for real training)"; \
		INPUT=data/processed/cleaned_batted_balls.parquet; \
	fi; \
	$(PY) -m mlb_luck_score.models.train_contact_model \
		--input $$INPUT \
		--output-dir artifacts

# Controlled comparison of contact-model variants (unweighted vs
# class-balanced vs naive-prevalence vs time-ordered post-hoc-calibrated) on
# untouched 2024 validation data. See "Model comparison" in README.md.
compare-models:
	$(PY) -m mlb_luck_score.models.compare_models \
		--input data/processed/cleaned_development_data.parquet \
		--output-dir outputs/tables \
		--figures-dir outputs/figures/model_comparison \
		--include-post-hoc-calibration

# Builds the Version 0.2 empirical public-score reference artifact: trains
# the unweighted baseline on 2021-2023, scores 2024 (out-of-sample), and
# stores positive/negative raw-contact-luck quantile tables under artifacts/
# (git-ignored). See "Version 0.2 scoring" in README.md.
build-reference-score:
	$(PY) -m mlb_luck_score.models.build_reference_score \
		--input data/processed/cleaned_development_data.parquet \
		--output-dir artifacts

demo:
	$(PY) -m mlb_luck_score.models.predict_outcomes --demo

notebook:
	$(PY) -m jupyter notebook notebooks
