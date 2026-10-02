"""The one pre-registered 2027 comparison: v1.3 (`outcome_free_v13`) vs v1.1.

Implements `docs/plans/v1_3_2027_preregistration.md` (approved 2026-10-01, amendment A1)
under `RESEARCH_RULES.md`, "2027: a pre-registered test season". This is the ONLY code
path that may read 2027 data, and only after the season ends, and only once:

    .venv/bin/python prospective/run_v1_3_2027_comparison.py

Order (guards first, so a refusal never downloads anything):
  1. `run_guards`: clean tree, recorded season end, no prior output, every game final.
  2. Ingest 2027 into `data/prospective/2027/v1_3/`; build the scoring dataset.
  3. Train v1.1's contact model (`baseline_v02`), v1.3 and the density-free variant on
     2021-2023 (pre-registration P1).
  4. Score every eligible 2027 batted ball; compute the §4/§5 endpoints.
  5. Write `outputs/prospective/v1_3/comparison_2027.json` exactly once.

Classification (§4). Ties in the wording are resolved by this order and are recorded here
before any 2027 data exists:
  - population below `MIN_POPULATION_BATTED_BALLS` -> "inconclusive" (overrides everything);
  - primary CI entirely above 0, or any of gates 2-4 failing -> "not_confirmed";
  - primary CI entirely below 0 -> "confirmed";
  - otherwise (CI straddles or touches 0) -> "inconclusive".
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import v1_3_2027_config as cfg
from sklearn.pipeline import Pipeline
from v1_3_2027_config import run_guards
from v1_3_2027_ingestion import build_scoring_dataset, ingest_2027

from mlb_luck_score.config import PROCESSED_DATA_DIR, TRAIN_SEASONS
from mlb_luck_score.models import evaluate_outcome_free_v13 as v13
from mlb_luck_score.models.compare_near_wall_calibration_gate import (
    MIN_SUBGROUP_GAMES,
    MIN_SUBGROUP_PLAYS,
    classify_subgroup_status,
    compute_subgroup_bootstrap,
    has_adequate_support,
)
from mlb_luck_score.models.evaluate_airball_geometry_v12b import _fit, _paired, _proba
from mlb_luck_score.models.evaluate_gated_geometry_v12 import (
    _REQUIRED,
    material_ece_regression,
    weather_residual_diagnostic,
)
from mlb_luck_score.models.train_contact_model import (
    TrainedModel,
    predict_proba_ordered,
    train_model,
)

logger = logging.getLogger(__name__)

N_BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 42
_WEATHER = ("event_id", "roof_status", "air_density_kg_m3", "following_wind_mps")
_CARRY = ("event_id", "season", "game_pk", "venue_id", "outcome_class", "bb_type")
_PREFIX = {"baseline": "base_", "candidate": "cand_", "no_density": "nod_"}


# --- data -------------------------------------------------------------------------------------


def load_development_rows() -> pd.DataFrame:
    """2021-2024 development rows with the same weather columns used in development."""
    base = pd.read_parquet(
        PROCESSED_DATA_DIR / "cleaned_development_data_with_venue.parquet",
        columns=[*_REQUIRED, "venue", "spray_sector"],
    )
    weather = pd.read_parquet(
        PROCESSED_DATA_DIR / "cleaned_development_data_with_weather.parquet",
        columns=["event_id", *v13.WEATHER_COLUMNS],
    )
    return v13.prepare_rows(base.merge(weather, on="event_id", how="left"))


def training_rows(dev: pd.DataFrame) -> pd.DataFrame:
    """Pre-registration P1: 2021-2023, the seasons v1.1 trains on."""
    return dev.loc[dev["season"].isin(TRAIN_SEASONS)].reset_index(drop=True)


def prepare_2027_rows(df: pd.DataFrame) -> pd.DataFrame:
    seasons = {int(s) for s in pd.unique(df["season"].dropna())}
    if seasons != {cfg.SEASON_2027}:
        raise ValueError(f"2027 scoring rows contain season(s) {sorted(seasons)}")
    missing = [
        c
        for c in (*v13.CANDIDATE_FEATURES, "outcome_class", "game_pk", "venue_id")
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"2027 rows are missing column(s) {missing}")
    eligible = df["eligible_for_training"].astype("boolean").fillna(False).astype(bool)
    out = df.loc[eligible].copy()
    if "following_wind_mps" not in out.columns:
        out["following_wind_mps"] = np.nan
    return out.reset_index(drop=True)


# --- models -----------------------------------------------------------------------------------


@dataclass
class Models:
    baseline: TrainedModel
    candidate: Pipeline
    no_density: Pipeline


def fit_models(train: pd.DataFrame) -> Models:
    return Models(
        baseline=train_model(train),
        candidate=_fit(train, v13.CANDIDATE_FEATURES),
        no_density=_fit(train, v13.NO_DENSITY_FEATURES),
    )


def score(models: Models, rows: pd.DataFrame) -> pd.DataFrame:
    b = models.baseline
    probas = {
        "baseline": predict_proba_ordered(b, rows[b.numeric_features + b.categorical_features]),
        "candidate": _proba(models.candidate, rows, v13.CANDIDATE_FEATURES),
        "no_density": _proba(models.no_density, rows, v13.NO_DENSITY_FEATURES),
    }
    frame = rows[list(_CARRY)].copy()
    frame["gate"] = True
    for key, prefix in _PREFIX.items():
        for cls in v13.CLASS_ORDER:
            frame[f"{prefix}{cls}"] = probas[key][cls].to_numpy()
    return frame


# --- endpoints ---------------------------------------------------------------------------------


def classify(
    *,
    n_population: int,
    primary_ci: tuple[float, float],
    calibration_ok: bool,
    no_venue_regression: bool,
    directions_ok: bool,
) -> str:
    """See the module docstring for the order of the rules."""
    lo, hi = primary_ci
    if n_population < cfg.MIN_POPULATION_BATTED_BALLS:
        return "inconclusive"
    if lo > 0 or not (calibration_ok and no_venue_regression and directions_ok):
        return "not_confirmed"
    if hi < 0:
        return "confirmed"
    return "inconclusive"


def _venues(pred: pd.DataFrame, *, n_reps: int, seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for venue_id, idx in pred.groupby("venue_id").groups.items():
        sub = pred.loc[idx]
        n_plays, n_games = len(sub), int(sub["game_pk"].nunique())
        rec: dict[str, Any] = {
            "venue_id": int(float(str(venue_id))),
            "n_plays": n_plays,
            "n_games": n_games,
            "adequate_support": n_plays >= MIN_SUBGROUP_PLAYS and n_games >= MIN_SUBGROUP_GAMES,
            "credible_regression": False,
        }
        if rec["adequate_support"]:
            ll = _paired(sub, "base_", "cand_", ("log_loss",), n_reps=n_reps, seed=seed)["log_loss"]
            rec.update(paired_log_loss=ll, credible_regression=ll["ci_low"] > 0)
            y = (sub["outcome_class"] == "home_run").to_numpy().astype(float)
            boot = compute_subgroup_bootstrap(
                y,
                sub["cand_home_run"].to_numpy(),
                sub["base_home_run"].to_numpy(),
                sub["game_pk"].to_numpy(),
                n_reps=n_reps,
                seed=seed,
            )
            support = has_adequate_support(n_plays, n_games, int(y.sum()), int((1 - y).sum()))
            for model, key in (("v1_3", "specialist_ece"), ("v1_1", "baseline_ece")):
                status, _ = classify_subgroup_status(
                    has_adequate_support=support,
                    specialist_ece_ci_low=boot[key]["ci_low"],
                    specialist_ece_ci_high=boot[key]["ci_high"],
                    paired_log_loss_delta_ci_low=None,
                )
                rec[f"{model}_hr_ece_status_reported_only"] = status
        out.append(rec)
    return out


def compare(
    models: Models,
    rows: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    n_reps: int = N_BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    pred = score(models, rows)
    primary = _paired(
        pred, "base_", "cand_", ("log_loss", "ece", "home_run_ece"), n_reps=n_reps, seed=seed
    )
    base_e, cand_e = v13._eces(pred, "base_"), v13._eces(pred, "cand_")
    calibration = {
        "v1_1": base_e,
        "v1_3": cand_e,
        "overall_material_regression": material_ece_regression(
            base_e["overall"], cand_e["overall"]
        ),
        "home_run_material_regression": material_ece_regression(
            base_e["home_run"], cand_e["home_run"]
        ),
    }
    venues = _venues(pred, n_reps=n_reps, seed=seed)
    # A row with no venue keeps its place in the pooled checks under venue -1; the
    # per-venue breakdown is reporting only.
    directions = v13.summarize_perturbations(
        v13._perturbation_sums(
            SimpleNamespace(candidate=models.candidate),  # type: ignore[arg-type]
            rows.assign(venue_id=rows["venue_id"].fillna(-1)),
        )
    )
    known = rows["air_density_kg_m3"].notna().to_numpy()
    density = _paired(pred.loc[known], "nod_", "cand_", ("log_loss",), n_reps=n_reps, seed=seed)[
        "log_loss"
    ]
    nod_view = pred.drop(columns=[f"cand_{c}" for c in v13.CLASS_ORDER]).rename(
        columns={f"nod_{c}": f"cand_{c}" for c in v13.CLASS_ORDER}
    )
    gates = {
        "calibration_ok": not (
            calibration["overall_material_regression"]
            or calibration["home_run_material_regression"]
        ),
        "no_venue_regression": not any(v["credible_regression"] for v in venues),
        "directions_ok": all(d["passed"] for d in directions.values()),
    }
    ll = primary["log_loss"]
    return {
        "n_population": len(pred),
        "classification": classify(
            n_population=len(pred), primary_ci=(ll["ci_low"], ll["ci_high"]), **gates
        ),
        "gates": gates,
        "primary": primary,
        "calibration": calibration,
        "venues": venues,
        "directions": directions,
        "density_test": {
            **density,
            "status": "supported" if density["ci_high"] < 0 else "not_supported",
        },
        "weather_residual": {
            "v1_1_and_v1_3": weather_residual_diagnostic(pred, weather, n_reps=n_reps, seed=seed),
            "density_free_variant": weather_residual_diagnostic(
                nod_view, weather, n_reps=n_reps, seed=seed
            ),
        },
    }


# --- orchestration -------------------------------------------------------------------------------


def current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cfg.PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def write_once(path: Path, record: dict[str, Any]) -> None:
    cfg.assert_in_namespace(path)
    cfg.assert_not_already_run(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as fh:
            json.dump(record, fh, indent=2, default=float)
    except FileExistsError as exc:
        raise cfg.AlreadyRunError(f"{path} appeared during the run; refusing to overwrite") from exc


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    guard_report = run_guards()
    end = date.fromisoformat(guard_report["season_end_date"])
    ingestion = ingest_2027(cfg.SEASON_2027_START_DATE, end)
    dataset, dataset_report = build_scoring_dataset()
    rows = prepare_2027_rows(dataset)
    models = fit_models(training_rows(load_development_rows()))
    weather = rows[list(_WEATHER)]
    result = compare(models, rows, weather, n_reps=N_BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED)
    record = {
        "preregistration": cfg.PREREGISTRATION,
        "code_commit": current_commit(),
        "training_seasons": list(TRAIN_SEASONS),
        "guards": guard_report,
        "ingestion": ingestion,
        "dataset": dataset_report,
        "result": result,
    }
    write_once(cfg.COMPARISON_PATH, record)
    logger.info("wrote %s: %s", cfg.COMPARISON_PATH, result["classification"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
