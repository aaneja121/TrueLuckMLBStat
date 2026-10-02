"""Contact Luck v1.3 candidate `outcome_free_v13` and its frozen evaluation.

Implements exactly `docs/plans/v1_3_outcome_free_contact_plan.md` (frozen 2026-10-01,
maintainer decisions D9-D12). Change the plan first, never this module alone. A
CANDIDATE only: `recommend_v13_adoption` is input to a maintainer decision, and even a
pass is necessary, not sufficient -- the real test is the pre-registered 2027 comparison.

- **Rows:** every training-eligible batted ball in the development seasons.
- **Inputs:** `CANDIDATE_FEATURES` -- launch conditions, batter side, batted-ball type and
  air density. No `hit_distance_sc` (it is where the ball was fielded), no wall geometry,
  no venue identity. Park effects therefore stay in luck (D11, park-neutral).
- **Model:** the v1.2b construction (`build_airball_model`): HGB with defaults and the
  repository seed; numeric inputs unimputed, so unknown density stays missing.
- **Comparator:** `baseline_v02`, retrained on the same folds.

Data plumbing (season guard, v1.2 geometry join used only for the robbery-zone report,
folds) is shared with `evaluate_gated_geometry_v12`.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from mlb_luck_score.config import CLASS_ORDER, TABLES_DIR
from mlb_luck_score.data.join_park_geometry import GEOMETRY_STATUS_OK
from mlb_luck_score.models.compare_geometry_aware import _fast_class_ece, _fast_ece
from mlb_luck_score.models.compare_near_wall_calibration_gate import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOTSTRAP_REPS,
    MIN_SUBGROUP_GAMES,
    MIN_SUBGROUP_PLAYS,
)
from mlb_luck_score.models.evaluate_airball_geometry_v12b import _fit, _paired, _proba
from mlb_luck_score.models.evaluate_gated_geometry_v12 import (
    _REQUIRED,
    DEFAULT_INPUT,
    DEFAULT_WEATHER,
    evaluation_folds,
    material_ece_regression,
    weather_residual_diagnostic,
)
from mlb_luck_score.models.evaluate_gated_geometry_v12 import prepare_rows as _prepare_v12
from mlb_luck_score.models.train_contact_model import (
    TrainedModel,
    predict_proba_ordered,
    train_model,
)

__all__ = ["CLASS_ORDER", "evaluation_folds"]

logger = logging.getLogger(__name__)

CANDIDATE_NAME = "outcome_free_v13"
CANDIDATE_FEATURES: tuple[str, ...] = (
    "launch_speed",
    "launch_angle",
    "spray_angle_approx",
    "air_density_kg_m3",
    "stand",
    "bb_type",
)
NO_DENSITY_FEATURES: tuple[str, ...] = tuple(
    f for f in CANDIDATE_FEATURES if f != "air_density_kg_m3"
)
NO_SPRAY_FEATURES: tuple[str, ...] = tuple(
    f for f in CANDIDATE_FEATURES if f != "spray_angle_approx"
)
WEATHER_COLUMNS: tuple[str, ...] = ("roof_status", "air_density_kg_m3")

PERTURB_LAUNCH_SPEED_MPH = 3.0
PERTURB_DENSITY_KG_M3 = 0.05
ROBBERY_ZONE_FT = 5.0
_HR = CLASS_ORDER.index("home_run")
_AIR = ("fly_ball", "line_drive")


def prepare_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Eligible development rows (2025/2026 refused) with weather columns present."""
    missing = [c for c in WEATHER_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing weather column(s) {missing}; merge them before preparing rows")
    return _prepare_v12(df)


def roof_group(status: pd.Series) -> pd.Series:
    mapping = {
        "outdoor_open_air": "outdoor",
        "retractable_roof_open": "roof_open",
        "retractable_roof_closed": "closed_or_indoor",
        "fixed_indoor": "closed_or_indoor",
    }
    return status.map(lambda s: mapping.get(s, "unknown") if isinstance(s, str) else "unknown")


@dataclass
class FoldModels:
    baseline: TrainedModel
    candidate: Pipeline
    no_density: Pipeline
    no_spray: Pipeline


def fit_fold(train: pd.DataFrame) -> FoldModels:
    return FoldModels(
        baseline=train_model(train),
        candidate=_fit(train, CANDIDATE_FEATURES),
        no_density=_fit(train, NO_DENSITY_FEATURES),
        no_spray=_fit(train, NO_SPRAY_FEATURES),
    )


def predict_fold(models: FoldModels, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    b = models.baseline
    return {
        "baseline": predict_proba_ordered(b, df[b.numeric_features + b.categorical_features]),
        "candidate": _proba(models.candidate, df, CANDIDATE_FEATURES),
        "no_density": _proba(models.no_density, df, NO_DENSITY_FEATURES),
        "no_spray": _proba(models.no_spray, df, NO_SPRAY_FEATURES),
    }


def _perturbation_sums(models: FoldModels, held: pd.DataFrame) -> list[dict[str, Any]]:
    known = held["air_density_kg_m3"].notna()
    scenarios = {
        # name: (rows, column, low value delta, high value delta); high = expected higher P(HR)
        "harder_hit": (held, "launch_speed", -PERTURB_LAUNCH_SPEED_MPH, +PERTURB_LAUNCH_SPEED_MPH),
        "thinner_air": (
            held.loc[known],
            "air_density_kg_m3",
            +PERTURB_DENSITY_KG_M3,
            -PERTURB_DENSITY_KG_M3,
        ),
    }
    out: list[dict[str, Any]] = []
    for name, (rows, col, low, high) in scenarios.items():
        if rows.empty:
            continue
        p = {}
        for label, delta in (("low", low), ("high", high)):
            p[label] = (
                _proba(
                    models.candidate, rows.assign(**{col: rows[col] + delta}), CANDIDATE_FEATURES
                )
                .iloc[:, _HR]
                .to_numpy()
            )
        frame = pd.DataFrame(
            {
                "venue_id": rows["venue_id"].astype(int).to_numpy(),
                "low": p["low"],
                "high": p["high"],
            }
        )
        for venue_id, g in frame.groupby("venue_id"):
            out.append(
                {
                    "check": name,
                    "venue_id": int(cast(Any, venue_id)),
                    "n": len(g),
                    "sum_low": float(g["low"].sum()),
                    "sum_high": float(g["high"].sum()),
                }
            )
    return out


@dataclass
class FoldRunResult:
    predictions: pd.DataFrame
    perturbations: list[dict[str, Any]] = field(default_factory=list)


_CARRY = (
    "event_id",
    "season",
    "game_pk",
    "venue_id",
    "outcome_class",
    "bb_type",
    "spray_sector",
    "geometry_status",
    "projected_distance_to_wall_margin",  # robbery-zone report only; never a model input
)
_PREFIXES = {"baseline": "base_", "candidate": "cand_", "no_density": "nod_", "no_spray": "nos_"}


def run_folds(rows: pd.DataFrame, folds: list[tuple[tuple[int, ...], int]]) -> FoldRunResult:
    frames: list[pd.DataFrame] = []
    result = FoldRunResult(predictions=pd.DataFrame())
    for train_seasons, held_out in folds:
        if held_out in train_seasons:
            raise ValueError(f"held-out season {held_out} is in the training seasons")
        train = rows.loc[rows["season"].isin(train_seasons)]
        held = rows.loc[rows["season"] == held_out]
        logger.info("fold: train %s -> evaluate %s", train_seasons, held_out)
        models = fit_fold(train)
        pred = predict_fold(models, held)
        frame = held[[c for c in _CARRY if c in held.columns]].copy()
        frame["roof_group"] = roof_group(held["roof_status"]).to_numpy()
        frame["gate"] = True  # every row is in scope; name shared with the v1.2 diagnostic
        for key, prefix in _PREFIXES.items():
            for cls in CLASS_ORDER:
                frame[f"{prefix}{cls}"] = pred[key][cls].to_numpy()
        frames.append(frame)
        result.perturbations.extend(_perturbation_sums(models, held))
    result.predictions = pd.concat(frames, ignore_index=True)
    return result


_EXPECTED = ("harder_hit", "thinner_air")


def summarize_perturbations(sums: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pooled mean P(HR) high minus low; both checks expect a positive delta."""
    frame = pd.DataFrame(sums)
    out: dict[str, dict[str, Any]] = {}
    for check in _EXPECTED:
        g = frame[frame["check"] == check] if not frame.empty else frame
        n = int(g["n"].sum()) if not g.empty else 0
        delta = float((g["sum_high"].sum() - g["sum_low"].sum()) / n) if n else float("nan")
        by_venue: dict[int, float] = {}
        if n:
            for venue_id, v in g.groupby("venue_id"):
                by_venue[int(cast(Any, venue_id))] = float(
                    (v["sum_high"].sum() - v["sum_low"].sum()) / v["n"].sum()
                )
        out[check] = {
            "n_rows": n,
            "delta_p_hr": delta,
            "passed": bool(n > 0 and delta > 0),
            "venues_backwards": sorted(k for k, v in by_venue.items() if v <= 0),
            "delta_by_venue": by_venue,
        }
    return out


def _group_masks(pred: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    for venue_id in sorted(pred["venue_id"].dropna().unique()):
        masks[f"venue_{int(venue_id)}"] = pred["venue_id"] == venue_id
    for b in sorted(pred["bb_type"].dropna().unique()):
        masks[f"bb_type_{b}"] = pred["bb_type"] == b
    if "spray_sector" in pred.columns:
        for s in sorted(pred["spray_sector"].dropna().unique()):
            masks[f"spray_sector_{s}"] = pred["spray_sector"] == s
    for r in ("outdoor", "roof_open", "closed_or_indoor", "unknown"):
        masks[f"roof_{r}"] = pred["roof_group"] == r
    return masks


def evaluate_groups(pred: pd.DataFrame, *, n_reps: int, seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, mask in _group_masks(pred).items():
        sub = pred.loc[mask.fillna(False).to_numpy()]
        n_plays, n_games = len(sub), int(sub["game_pk"].nunique())
        adequate = n_plays >= MIN_SUBGROUP_PLAYS and n_games >= MIN_SUBGROUP_GAMES
        record: dict[str, Any] = {
            "group": name,
            "n_plays": n_plays,
            "n_games": n_games,
            "adequate_support": adequate,
            "paired_log_loss_point": None,
            "paired_log_loss_ci_low": None,
            "paired_log_loss_ci_high": None,
        }
        if adequate:
            ll = _paired(sub, "base_", "cand_", ("log_loss",), n_reps=n_reps, seed=seed)["log_loss"]
            record.update(
                paired_log_loss_point=ll["point_estimate"],
                paired_log_loss_ci_low=ll["ci_low"],
                paired_log_loss_ci_high=ll["ci_high"],
            )
        out.append(record)
    return out


def _eces(pred: pd.DataFrame, prefix: str) -> dict[str, float]:
    y_idx = np.array([CLASS_ORDER.index(str(c)) for c in pred["outcome_class"]])
    arr = pred[[f"{prefix}{c}" for c in CLASS_ORDER]].to_numpy()
    rows = np.arange(len(pred))
    out = {"overall": _fast_ece(arr, y_idx, rows)}
    for i, cls in enumerate(CLASS_ORDER):
        out[cls] = _fast_class_ece(arr, y_idx, rows, i)
    return out


def _park_residual(pred: pd.DataFrame) -> dict[int, dict[str, float]]:
    hr = (pred["outcome_class"] == "home_run").astype(float)
    out: dict[int, dict[str, float]] = {}
    for venue_id, idx in pred.groupby("venue_id").groups.items():
        g = pred.loc[idx]
        out[int(cast(Any, venue_id))] = {
            "n": len(g),
            "baseline_hr_residual_per_100": float(100 * (hr.loc[idx] - g["base_home_run"]).mean()),
            "candidate_hr_residual_per_100": float(100 * (hr.loc[idx] - g["cand_home_run"]).mean()),
        }
    return out


def recommend_v13_adoption(summary: dict[str, Any]) -> dict[str, Any]:
    """The frozen v1.3 development gates. Necessary, not sufficient (D12)."""
    criteria = {
        "log_loss": summary["vs_baseline"]["log_loss"]["ci_high"] < 0,
        "calibration": not (
            summary["calibration"]["overall_material_regression"]
            or summary["calibration"]["home_run_material_regression"]
        ),
        "no_group_regression": not any(
            g["adequate_support"]
            and g["paired_log_loss_ci_low"] is not None
            and g["paired_log_loss_ci_low"] > 0
            for g in summary["groups"]
        ),
        "perturbations": all(c["passed"] for c in summary["perturbations"].values()),
        "confirmation_2024": not summary["confirmation_2024"]["credible_regression"],
    }
    return {
        "candidate": CANDIDATE_NAME,
        "criteria": criteria,
        "recommend_adopt": all(criteria.values()),
        "requires_maintainer_decision": True,
        "real_test": "pre-registered 2027 prospective comparison",
    }


def evaluate(
    primary: FoldRunResult,
    confirmation: FoldRunResult,
    weather: pd.DataFrame | None,
    *,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    pred = primary.predictions
    vs_base = _paired(
        pred, "base_", "cand_", ("log_loss", "ece", "home_run_ece"), n_reps=n_reps, seed=seed
    )
    base_e, cand_e = _eces(pred, "base_"), _eces(pred, "cand_")
    conf = _paired(
        confirmation.predictions, "base_", "cand_", ("log_loss",), n_reps=n_reps, seed=seed
    )["log_loss"]
    gb = pred[pred["bb_type"] == "ground_ball"]
    margin = pd.to_numeric(pred["projected_distance_to_wall_margin"], errors="coerce")
    robbery = pred.loc[
        (pred["outcome_class"] == "out")
        & pred["bb_type"].isin(_AIR)
        & (pred["geometry_status"] == GEOMETRY_STATUS_OK)
        & (margin.abs() <= ROBBERY_ZONE_FT)
    ]
    summary: dict[str, Any] = {
        "candidate": CANDIDATE_NAME,
        "n_rows": len(pred),
        "vs_baseline": vs_base,
        "calibration": {
            "baseline": {"ece": base_e["overall"], "home_run_ece": base_e["home_run"]},
            "candidate": {"ece": cand_e["overall"], "home_run_ece": cand_e["home_run"]},
            "overall_material_regression": material_ece_regression(
                base_e["overall"], cand_e["overall"]
            ),
            "home_run_material_regression": material_ece_regression(
                base_e["home_run"], cand_e["home_run"]
            ),
        },
        "groups": evaluate_groups(pred, n_reps=n_reps, seed=seed),
        "perturbations": summarize_perturbations(primary.perturbations),
        "confirmation_2024": {**conf, "credible_regression": conf["ci_low"] > 0},
        "reported_only": {
            "density_ablation": _paired(
                pred, "nod_", "cand_", ("log_loss",), n_reps=n_reps, seed=seed
            )["log_loss"],
            "spray_ablation": {
                "all": _paired(pred, "nos_", "cand_", ("log_loss",), n_reps=n_reps, seed=seed)[
                    "log_loss"
                ],
                "ground_ball": _paired(
                    gb, "nos_", "cand_", ("log_loss",), n_reps=n_reps, seed=seed
                )["log_loss"],
            },
            "park_residual": _park_residual(pred),
            "robbery_zone": {
                "n_caught_air_balls_within_5ft_of_wall": len(robbery),
                "mean_p_hr_baseline": float(robbery["base_home_run"].mean())
                if len(robbery)
                else None,
                "mean_p_hr_candidate": float(robbery["cand_home_run"].mean())
                if len(robbery)
                else None,
            },
            "per_class_ece": {"baseline": base_e, "candidate": cand_e},
            "confirmation_perturbations": summarize_perturbations(confirmation.perturbations),
        },
    }
    if weather is not None:
        summary["reported_only"]["weather_residual"] = weather_residual_diagnostic(
            pred, weather, n_reps=n_reps, seed=seed
        )
    summary["recommendation"] = recommend_v13_adoption(summary)
    return summary


OUTPUT_FILENAME = "v1_3_outcome_free_evaluation.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--weather", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    base = pd.read_parquet(args.input, columns=[*_REQUIRED, "venue", "spray_sector"])
    weather = pd.read_parquet(
        args.weather, columns=["event_id", "roof_status", "air_density_kg_m3", "following_wind_mps"]
    )
    rows = prepare_rows(
        base.merge(weather[["event_id", *WEATHER_COLUMNS]], on="event_id", how="left")
    )
    folds = evaluation_folds()
    summary = evaluate(
        run_folds(rows, folds["primary"]), run_folds(rows, folds["confirmation"]), weather
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / OUTPUT_FILENAME
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("wrote %s; recommendation: %s", out_path, summary["recommendation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
