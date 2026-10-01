"""Contact Luck v1.2b candidate `airball_geometry_v12b` and its frozen evaluation.

Implements exactly the redesign frozen in `docs/plans/v1_2_park_weather_plan.md`
("Redesign: outcome-free air-ball model", frozen 2026-10-01). Change the plan first,
never this module alone. A CANDIDATE only: `recommend_v12b_adoption` is input to a
maintainer decision, and nothing here touches production scoring.

Why it exists: `hit_distance_sc` is where a ball was fielded (glove, wall, seats), so a
model that reads it near the wall partly reads the outcome (the `gated_geometry_v12`
plausibility finding). This candidate scores every air ball at a venue with v1.2
geometry from launch conditions and the wall alone:

- **Scope:** `bb_type` in {fly_ball, line_drive} with v1.2 geometry resolved. Every other
  row keeps the `baseline_v02` prediction, bit-identical (asserted).
- **Inputs:** `CANDIDATE_FEATURES`. No `hit_distance_sc` and nothing derived from it.
  `spray_angle_approx` comes from the fielded hit coordinates and is kept, flagged.
- **Model:** `HistGradientBoostingClassifier` with defaults and the repository seed;
  numeric inputs pass through unimputed so an unknown wall height stays missing.
- **Reference:** the same model without the two wall features. It isolates what
  geometry adds without outcome-revealing inputs.

Criteria, folds and bootstraps follow the plan; see `evaluate` and
`recommend_v12b_adoption`. The data plumbing is shared with
`evaluate_gated_geometry_v12` (same v1.2 join, same folds, same season guard).
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mlb_luck_score.config import CLASS_ORDER, TABLES_DIR
from mlb_luck_score.data.join_park_geometry import GEOMETRY_STATUS_OK
from mlb_luck_score.models.compare_geometry_aware import (
    _fast_class_ece,
    _fast_ece,
    compute_paired_bootstrap,
)
from mlb_luck_score.models.compare_near_wall_calibration_gate import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOTSTRAP_REPS,
    MIN_SUBGROUP_GAMES,
    MIN_SUBGROUP_PLAYS,
    classify_subgroup_status,
    compute_subgroup_bootstrap,
    has_adequate_support,
)
from mlb_luck_score.models.evaluate_gated_geometry_v12 import (
    _REQUIRED,
    DEFAULT_INPUT,
    DEFAULT_WEATHER,
    evaluation_folds,
    join_v12_geometry,
    material_ece_regression,
    prepare_rows,
    wall_height_bin,
    weather_residual_diagnostic,
)
from mlb_luck_score.models.train_contact_model import (
    RANDOM_SEED,
    TrainedModel,
    predict_proba_ordered,
    reorder_proba_columns,
    train_model,
)

__all__ = ["evaluation_folds", "join_v12_geometry", "prepare_rows"]

logger = logging.getLogger(__name__)

CANDIDATE_NAME = "airball_geometry_v12b"
SCOPE_BB_TYPES = ("fly_ball", "line_drive")
LAUNCH_FEATURES: tuple[str, ...] = ("launch_speed", "launch_angle", "spray_angle_approx")
WALL_FEATURES: tuple[str, ...] = (
    "wall_distance_in_spray_direction",
    "wall_height_in_spray_direction",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("stand", "bb_type")
REFERENCE_FEATURES: tuple[str, ...] = (*LAUNCH_FEATURES, *CATEGORICAL_FEATURES)
CANDIDATE_FEATURES: tuple[str, ...] = (*LAUNCH_FEATURES, *WALL_FEATURES, *CATEGORICAL_FEATURES)

PERTURB_WALL_FT = 10.0
PERTURB_HEIGHT_LOW_FT = 8.0
PERTURB_HEIGHT_HIGH_FT = 16.0
PERTURB_LAUNCH_SPEED_MPH = 3.0
ROBBERY_ZONE_FT = 5.0

_HR = CLASS_ORDER.index("home_run")


def scope_mask(df: pd.DataFrame) -> pd.Series:
    """Air balls with v1.2 geometry resolved."""
    return (
        (df["geometry_status"] == GEOMETRY_STATUS_OK) & df["bb_type"].isin(SCOPE_BB_TYPES)
    ).astype(bool)


def build_airball_model(features: tuple[str, ...]) -> Pipeline:
    """HGB with repository defaults; numeric passthrough keeps missing values missing."""
    numeric = [f for f in features if f not in CATEGORICAL_FEATURES]
    categorical = [f for f in features if f in CATEGORICAL_FEATURES]
    preprocess = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("classify", HistGradientBoostingClassifier(random_state=RANDOM_SEED)),
        ]
    )


def _x(df: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    x = df[list(features)].copy()
    for col in CATEGORICAL_FEATURES:
        if col in x.columns:
            x[col] = x[col].map(lambda v: None if pd.isna(v) else str(v)).astype(object)
    for col in features:
        if col not in CATEGORICAL_FEATURES:
            x[col] = pd.to_numeric(x[col], errors="coerce").astype(float)
    return x


def _fit(df: pd.DataFrame, features: tuple[str, ...]) -> Pipeline:
    pipe = build_airball_model(features)
    pipe.fit(_x(df, features), df["outcome_class"].astype(str))
    return pipe


def _proba(pipe: Pipeline, df: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    classes = [str(c) for c in pipe.named_steps["classify"].classes_]
    return reorder_proba_columns(pipe.predict_proba(_x(df, features)), classes, df.index)


@dataclass
class FoldModels:
    baseline: TrainedModel
    reference: Pipeline
    candidate: Pipeline


def fit_fold(train: pd.DataFrame) -> FoldModels:
    """`baseline_v02` on all rows; reference and candidate on in-scope rows only."""
    in_scope = train.loc[scope_mask(train)]
    return FoldModels(
        baseline=train_model(train),
        reference=_fit(in_scope, REFERENCE_FEATURES),
        candidate=_fit(in_scope, CANDIDATE_FEATURES),
    )


def predict_fold(models: FoldModels, df: pd.DataFrame) -> dict[str, Any]:
    """Baseline (all rows), reference and candidate (in scope; baseline elsewhere)."""
    scope = scope_mask(df)
    baseline = predict_proba_ordered(
        models.baseline, df[models.baseline.numeric_features + models.baseline.categorical_features]
    )
    reference, candidate = baseline.copy(), baseline.copy()
    if scope.any():
        reference.loc[scope] = _proba(
            models.reference, df.loc[scope], REFERENCE_FEATURES
        ).to_numpy()
        candidate.loc[scope] = _proba(
            models.candidate, df.loc[scope], CANDIDATE_FEATURES
        ).to_numpy()
    if not np.array_equal(baseline.loc[~scope].to_numpy(), candidate.loc[~scope].to_numpy()):
        raise AssertionError("out-of-scope rows differ from baseline_v02")
    return {"baseline": baseline, "reference": reference, "candidate": candidate, "scope": scope}


def _perturbation_sums(models: FoldModels, in_scope: pd.DataFrame) -> list[dict[str, Any]]:
    known_height = in_scope["wall_height_in_spray_direction"].notna()
    scenarios: dict[str, tuple[pd.DataFrame, str, float, float, bool]] = {
        # name: (rows, column, low, high, values are deltas)
        "farther_wall": (
            in_scope,
            "wall_distance_in_spray_direction",
            -PERTURB_WALL_FT,
            PERTURB_WALL_FT,
            True,
        ),
        "taller_wall": (
            in_scope.loc[known_height],
            "wall_height_in_spray_direction",
            PERTURB_HEIGHT_LOW_FT,
            PERTURB_HEIGHT_HIGH_FT,
            False,
        ),
        "harder_hit": (
            in_scope,
            "launch_speed",
            -PERTURB_LAUNCH_SPEED_MPH,
            PERTURB_LAUNCH_SPEED_MPH,
            True,
        ),
    }
    out: list[dict[str, Any]] = []
    for name, (rows, col, low, high, is_delta) in scenarios.items():
        if rows.empty:
            continue
        p = {}
        for label, v in (("low", low), ("high", high)):
            scen = rows.copy()
            scen[col] = scen[col] + v if is_delta else v
            p[label] = _proba(models.candidate, scen, CANDIDATE_FEATURES).iloc[:, _HR].to_numpy()
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
    "wall_height_in_spray_direction",
    "projected_distance_to_wall_margin",  # reporting only (robbery zone); never a model input
    "spray_sector",
)


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
        frame["gate"] = pred["scope"].to_numpy()  # "gate" naming shared with v1.2 helpers
        for prefix, proba in (
            ("base_", pred["baseline"]),
            ("ref_", pred["reference"]),
            ("cand_", pred["candidate"]),
        ):
            for cls in CLASS_ORDER:
                frame[f"{prefix}{cls}"] = proba[cls].to_numpy()
        frames.append(frame)
        result.perturbations.extend(_perturbation_sums(models, held.loc[pred["scope"]]))
    result.predictions = pd.concat(frames, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_EXPECTED_SIGN = {"farther_wall": -1, "taller_wall": -1, "harder_hit": +1}


def summarize_perturbations(sums: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    frame = pd.DataFrame(sums)
    out: dict[str, dict[str, Any]] = {}
    for check, sign in _EXPECTED_SIGN.items():
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
            "expected_sign": sign,
            "n_rows": n,
            "delta_p_hr": delta,
            "passed": bool(n > 0 and np.sign(delta) == sign),
            "venues_backwards": sorted(k for k, v in by_venue.items() if np.sign(v) != sign),
            "delta_by_venue": by_venue,
        }
    return out


def _cols(pred: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return pred[[f"{prefix}{c}" for c in CLASS_ORDER]].set_axis(list(CLASS_ORDER), axis=1)


def _paired(
    pred: pd.DataFrame, a: str, b: str, metrics: tuple[str, ...], *, n_reps: int, seed: int
) -> dict[str, Any]:
    """Bootstrap of (b - a) per metric."""
    res = compute_paired_bootstrap(
        pred["outcome_class"].astype(str),
        _cols(pred, a),
        _cols(pred, b),
        pred["game_pk"],
        metrics=metrics,
        n_reps=n_reps,
        seed=seed,
    )
    return {m: dict(v) for m, v in res.items()}


def _point_eces(pred: pd.DataFrame, prefix: str) -> tuple[float, float]:
    y_idx = np.array([CLASS_ORDER.index(str(c)) for c in pred["outcome_class"]])
    arr, rows = _cols(pred, prefix).to_numpy(), np.arange(len(pred))
    return _fast_ece(arr, y_idx, rows), _fast_class_ece(arr, y_idx, rows, _HR)


def _group_masks(in_scope: pd.DataFrame) -> dict[str, pd.Series]:
    """The frozen v1.2b groups: venue, wall-height bin, spray sector, bb_type."""
    masks: dict[str, pd.Series] = {}
    for venue_id in sorted(in_scope["venue_id"].dropna().unique()):
        masks[f"venue_{int(venue_id)}"] = in_scope["venue_id"] == venue_id
    bins = wall_height_bin(in_scope["wall_height_in_spray_direction"])
    for b in ("short", "medium", "tall", "unknown"):
        masks[f"wall_height_{b}"] = bins == b
    if "spray_sector" in in_scope.columns:
        for s in sorted(in_scope["spray_sector"].dropna().unique()):
            masks[f"spray_sector_{s}"] = in_scope["spray_sector"] == s
    for b in SCOPE_BB_TYPES:
        masks[f"bb_type_{b}"] = in_scope["bb_type"] == b
    return masks


def evaluate_groups(in_scope: pd.DataFrame, *, n_reps: int, seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, mask in _group_masks(in_scope).items():
        sub = in_scope.loc[mask.fillna(False).to_numpy()]
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
            ll = _paired(sub, "ref_", "cand_", ("log_loss",), n_reps=n_reps, seed=seed)["log_loss"]
            record.update(
                paired_log_loss_point=ll["point_estimate"],
                paired_log_loss_ci_low=ll["ci_low"],
                paired_log_loss_ci_high=ll["ci_high"],
            )
            y_hr = (sub["outcome_class"] == "home_run").to_numpy().astype(float)
            boot = compute_subgroup_bootstrap(
                y_hr,
                sub["cand_home_run"].to_numpy(),
                sub["ref_home_run"].to_numpy(),
                sub["game_pk"].to_numpy(),
                n_reps=n_reps,
                seed=seed,
            )
            support = has_adequate_support(n_plays, n_games, int(y_hr.sum()), int((1 - y_hr).sum()))
            for model, key in (("candidate", "specialist_ece"), ("reference", "baseline_ece")):
                status, _ = classify_subgroup_status(
                    has_adequate_support=support,
                    specialist_ece_ci_low=boot[key]["ci_low"],
                    specialist_ece_ci_high=boot[key]["ci_high"],
                    paired_log_loss_delta_ci_low=None,
                )
                record[f"{model}_hr_ece"] = boot[key]["point_estimate"]
                record[f"{model}_hr_ece_status_reported_only"] = status
        out.append(record)
    return out


def recommend_v12b_adoption(summary: dict[str, Any]) -> dict[str, Any]:
    """The frozen v1.2b criteria. Input to a maintainer decision, never the decision."""
    criteria = {
        "geometry_adds": summary["geometry_vs_reference"]["log_loss"]["ci_high"] < 0,
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
        "out_of_scope_identical": bool(summary["out_of_scope_identical"]),
        "confirmation_2024": not summary["confirmation_2024"]["credible_regression"],
    }
    return {
        "candidate": CANDIDATE_NAME,
        "criteria": criteria,
        "recommend_adopt": all(criteria.values()),
        "requires_maintainer_decision": True,
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
    in_scope = pred.loc[pred["gate"]].reset_index(drop=True)
    geo = _paired(
        in_scope, "ref_", "cand_", ("log_loss", "ece", "home_run_ece"), n_reps=n_reps, seed=seed
    )
    ref_ece, ref_hr = _point_eces(in_scope, "ref_")
    cand_ece, cand_hr = _point_eces(in_scope, "cand_")
    base_ece, base_hr = _point_eces(in_scope, "base_")
    conf_scope = confirmation.predictions.loc[confirmation.predictions["gate"]]
    conf = _paired(conf_scope, "ref_", "cand_", ("log_loss",), n_reps=n_reps, seed=seed)["log_loss"]
    leak = _paired(
        in_scope, "base_", "cand_", ("log_loss", "ece", "home_run_ece"), n_reps=n_reps, seed=seed
    )
    margin = pd.to_numeric(in_scope["projected_distance_to_wall_margin"], errors="coerce")
    robbery = in_scope.loc[(in_scope["outcome_class"] == "out") & (margin.abs() <= ROBBERY_ZONE_FT)]
    summary: dict[str, Any] = {
        "candidate": CANDIDATE_NAME,
        "n_rows": len(pred),
        "n_in_scope": len(in_scope),
        "geometry_vs_reference": geo,
        "calibration": {
            "reference": {"ece": ref_ece, "home_run_ece": ref_hr},
            "candidate": {"ece": cand_ece, "home_run_ece": cand_hr},
            "overall_material_regression": material_ece_regression(ref_ece, cand_ece),
            "home_run_material_regression": material_ece_regression(ref_hr, cand_hr),
        },
        "groups": evaluate_groups(in_scope, n_reps=n_reps, seed=seed),
        "perturbations": summarize_perturbations(primary.perturbations),
        "out_of_scope_identical": True,  # predict_fold raises otherwise
        "confirmation_2024": {**conf, "credible_regression": conf["ci_low"] > 0},
        "reported_only": {
            "leak_gap_vs_baseline": {
                "paired_candidate_minus_baseline": leak,
                "baseline": {"ece": base_ece, "home_run_ece": base_hr},
            },
            "robbery_zone": {
                "n_caught_within_5ft_of_wall": len(robbery),
                "mean_p_hr_baseline": float(robbery["base_home_run"].mean())
                if len(robbery)
                else None,
                "mean_p_hr_candidate": float(robbery["cand_home_run"].mean())
                if len(robbery)
                else None,
            },
            "height_known_share_in_scope": float(
                in_scope["wall_height_in_spray_direction"].notna().mean()
            ),
            "confirmation_perturbations": summarize_perturbations(confirmation.perturbations),
        },
    }
    if weather is not None:
        summary["reported_only"]["weather_residual"] = weather_residual_diagnostic(
            pred, weather, n_reps=n_reps, seed=seed
        )
    summary["recommendation"] = recommend_v12b_adoption(summary)
    return summary


OUTPUT_FILENAME = "v1_2b_airball_geometry_evaluation.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--weather", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    rows = prepare_rows(pd.read_parquet(args.input, columns=[*_REQUIRED, "venue", "spray_sector"]))
    weather = pd.read_parquet(
        args.weather, columns=["event_id", "roof_status", "air_density_kg_m3", "following_wind_mps"]
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
