.PHONY: setup format lint typecheck test check download-sample clean-data \
	download-development-data clean-development-data train compare-models \
	build-reference-score download-game-metadata join-venue-metadata \
	compare-park-aware demo notebook build-park-geometry validate-park-geometry \
	join-park-geometry compare-geometry-aware notebook-park-geometry \
	download-weather-data build-game-weather join-weather-features \
	compare-weather-aware notebook-weather

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

# Downloads per-game venue/roof/surface metadata for 2021-2024 from the
# public MLB Stats API (Statcast itself has no usable venue field). One
# Parquet file per season, resumable. See "Park-aware (Version 0.3)" in
# README.md.
download-game-metadata:
	$(PY) -m mlb_luck_score.data.download_game_metadata \
		--seasons 2021 2022 2023 2024 \
		--output-dir data/raw

# Joins the per-season game-metadata files to the cleaned development
# dataset by game_pk and reports coverage (match rate, unmatched games/rows,
# counts by venue/season). Fails clearly if a game maps to multiple venues.
join-venue-metadata:
	$(PY) -m mlb_luck_score.data.join_venue_metadata \
		--cleaned-input data/processed/cleaned_development_data.parquet \
		--metadata-dir data/raw \
		--output data/processed/cleaned_development_data_with_venue.parquet

# Controlled comparison: baseline_v02 (current unweighted model) vs
# park_aware_v03_candidate (same model + categorical venue_id) on untouched
# 2024 validation data, including calibration by venue. Does NOT
# automatically adopt the park-aware variant -- see README.md.
compare-park-aware:
	$(PY) -m mlb_luck_score.models.compare_park_aware \
		--input data/processed/cleaned_development_data_with_venue.parquet \
		--output-dir outputs/tables \
		--figures-dir outputs/figures/park_aware

demo:
	$(PY) -m mlb_luck_score.models.predict_outcomes --demo

notebook:
	$(PY) -m jupyter notebook notebooks

# Validates the reviewed park-geometry reference table (mlb_luck_score.data.
# park_geometry) and prints a coverage summary. There is no separate
# "build from raw source" step for this table -- it is small, hand-curated
# reference data edited directly in source code (like game_metadata_
# overrides.py), not downloaded, so this target and validate-park-geometry
# run the exact same command. See "Park geometry (Version 0.4)" in README.md.
build-park-geometry:
	$(PY) -m mlb_luck_score.data.park_geometry

validate-park-geometry:
	$(PY) -m mlb_luck_score.data.park_geometry

# Joins the reviewed park-geometry table onto the venue-joined cleaned
# development dataset (by venue_id + game_date + spray_angle_approx) and
# reports coverage. Requires `make join-venue-metadata` to have been run
# first. See "Park geometry (Version 0.4)" in README.md.
join-park-geometry:
	$(PY) -m mlb_luck_score.data.join_park_geometry \
		--input data/processed/cleaned_development_data_with_venue.parquet \
		--output data/processed/cleaned_development_data_with_geometry.parquet

# Four-way controlled comparison: baseline_v02 vs venue_only_v03_candidate
# vs geometry_only_v04_candidate vs venue_plus_geometry_v04_candidate on
# untouched 2024 validation data, including a paired game_pk-level
# bootstrap. Does NOT automatically adopt any v0.4 candidate -- see
# README.md.
compare-geometry-aware:
	$(PY) -m mlb_luck_score.models.compare_geometry_aware \
		--input data/processed/cleaned_development_data_with_geometry.parquet \
		--output-dir outputs/tables --figures-dir outputs/figures/geometry_aware

notebook-park-geometry:
	$(PY) -m jupyter notebook notebooks/06_park_geometry_analysis.ipynb

# Downloads historical weather for 2021-2024 ONLY (refuses 2025): per-game
# MLB schedule weather (temperature/wind/roof condition, hydrated from the
# same /schedule endpoint as download-game-metadata) plus per-station
# historical ASOS/METAR observations (humidity/pressure, from the public
# Iowa Environmental Mesonet archive -- see mlb_luck_score.data.
# venue_environment for the 30 assigned stations). Resumable: existing
# per-season/per-station-season cache files are skipped unless --overwrite.
download-weather-data:
	$(PY) -m mlb_luck_score.data.download_historical_weather \
		--seasons 2021 2022 2023 2024 --output-dir data/raw

# Combines the schedule-weather + ASOS raw caches with mlb_luck_score.data.
# venue_environment and each game's roof_type into one row per game_pk,
# matching each game to the nearest ASOS observation at/before local start
# time (see mlb_luck_score.data.build_game_weather for the exact matching
# and roof-handling rules).
build-game-weather:
	$(PY) -m mlb_luck_score.data.build_game_weather \
		--seasons 2021 2022 2023 2024 --raw-dir data/raw \
		--output data/processed/game_weather.parquet

# Joins the game-weather table onto the venue+geometry-joined cleaned
# dataset by game_pk, computing per-play following/head/crosswind relative
# to each play's own spray direction. Requires make join-park-geometry (or
# join-venue-metadata) to have been run first.
join-weather-features:
	$(PY) -m mlb_luck_score.data.join_weather_features \
		--input data/processed/cleaned_development_data_with_geometry.parquet \
		--game-weather data/processed/game_weather.parquet \
		--output data/processed/cleaned_development_data_with_weather.parquet

# Controlled comparison: selected_production_baseline vs
# weather_basic_v05_candidate vs weather_vector_v05_candidate vs (if
# geometry columns are present) geometry_plus_weather_v05_candidate on
# untouched 2024 validation data, including a paired game_pk-level
# bootstrap. Does NOT automatically adopt any v0.5 candidate -- see
# README.md.
compare-weather-aware:
	$(PY) -m mlb_luck_score.models.compare_weather_aware \
		--input data/processed/cleaned_development_data_with_weather.parquet \
		--output-dir outputs/tables --figures-dir outputs/figures/weather_aware

notebook-weather:
	$(PY) -m jupyter notebook notebooks/07_weather_air_density_analysis.ipynb
