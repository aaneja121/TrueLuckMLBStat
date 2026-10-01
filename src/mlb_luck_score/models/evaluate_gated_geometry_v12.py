"""Contact Luck v1.2 candidate `gated_geometry_v12` and its frozen evaluation.

Implements exactly the specification frozen in `docs/plans/v1_2_park_weather_plan.md`
("Step 3", frozen 2026-09-30; wall-height amendment 2026-10-01). Change the plan first,
never this module alone. This is a CANDIDATE: `recommend_v12_adoption` is input to a
maintainer decision, never the decision, and nothing here touches production scoring.

## The candidate

- **Gate:** v1.2 geometry resolved, `bb_type` in {fly_ball, line_drive}, and
  `hit_distance_sc - wall_distance_in_spray_direction >= -20 ft`.
- **Gated rows:** a multinomial logistic specialist (`train_model`, unchanged, class_weight
  None) with the baseline features plus `SPECIALIST_FEATURES`, fitted on the gated rows of
  the training seasons only. `fit_fold` refuses to continue if the trainer's missingness
  rule dropped any of the five.
- **All other rows:** the `baseline_v02` prediction, bit-identical (asserted at runtime).
- No venue identity and no weather column in either model.

## Evaluation (all stated before any computation)

- Primary: leave-one-season-out within 2021-2023, out-of-fold predictions pooled.
- Confirmation: fit 2021-2023, evaluate 2024 (previously used; can only block).
- Game-clustered bootstrap, 500 replicates, seed 42, 95%.
- Criteria: pooled log-loss CI entirely below 0; no material overall/home-run ECE
  regression (v0.3 rule); no adequately supported venue/subgroup whose paired log-loss
  CI is entirely above 0; three perturbation directions; open field identical; no
  credible 2024 regression.
- Reported only: gate-boundary continuity, absolute home-run ECE status per group for
  both models (v0.7D part a), the weather-residual diagnostic, and coverage.

## Choices fixed here that the plan left implicit (recorded in the plan before running)

- Wall-height subgroup bins: short <= 8 ft, tall >= 15 ft (the existing
  `DEFAULT_HIGH_WALL_THRESHOLD_FT`), medium between, unknown when missing.
- Wall bands: beyond (margin >= 0) and short of the wall by 0-5, 5-10 and 10-20 ft.
- Perturbations: wall distance and hit distance +/- 10 ft; wall height 8 vs 16 ft on
  rows whose height is known. Gate membership is held at its unperturbed value.
- Gate-boundary continuity uses air balls with geometry whose margin lies within 2 ft of
  -20 ft.
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

from mlb_luck_score.config import CLASS_ORDER, DEVELOPMENT_SEASONS, PROCESSED_DATA_DIR, TABLES_DIR
from mlb_luck_score.data.join_park_geometry import (
    DEFAULT_HIGH_WALL_THRESHOLD_FT,
    GEOMETRY_STATUS_OK,
    NEAR_WALL_THRESHOLDS_FT,
    join_park_geometry,
)
from mlb_luck_score.data.park_geometry_v12 import (
    PARK_GEOMETRY_CONFIGS_V12,
    V12_EXCLUDED_VENUE_IDS,
)
from mlb_luck_score.features.build_contact_features import add_geometry_interaction_features
from mlb_luck_score.models.compare_geometry_aware import (
    _fast_class_ece,
    _fast_ece,
    compute_paired_bootstrap,
)
from mlb_luck_score.models.compare_near_wall_calibration_gate import (
    DEFAULT_BOOTSTRAP_CI,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOTSTRAP_REPS,
    MIN_SUBGROUP_GAMES,
    MIN_SUBGROUP_PLAYS,
    classify_subgroup_status,
    compute_subgroup_bootstrap,
    has_adequate_support,
)
from mlb_luck_score.models.compare_park_aware import (
    DEFAULT_MATERIAL_ECE_ABSOLUTE_MARGIN,
    DEFAULT_MATERIAL_ECE_RELATIVE_MARGIN,
)
from mlb_luck_score.models.train_contact_model import (
    TrainedModel,
    predict_proba_ordered,
    train_model,
)

logger = logging.getLogger(__name__)

CANDIDATE_NAME = "gated_geometry_v12"
GATE_BB_TYPES = ("fly_ball", "line_drive")
GATE_MARGIN_FT = -20.0
SPECIALIST_FEATURES: tuple[str, ...] = (
    "wall_distance_in_spray_direction",
    "wall_height_in_spray_direction",
    "projected_distance_to_wall_margin",
    "wall_margin_x_launch_angle",
    "launch_angle_x_wall_height",
)
PRIMARY_SEASONS: tuple[int, ...] = (2021, 2022, 2023)
CONFIRMATION_SEASON = 2024

SHORT_WALL_MAX_FT = 8.0
TALL_WALL_MIN_FT = DEFAULT_HIGH_WALL_THRESHOLD_FT
PERTURB_WALL_FT = 10.0
PERTURB_DISTANCE_FT = 10.0
PERTURB_HEIGHT_LOW_FT = 8.0
PERTURB_HEIGHT_HIGH_FT = 16.0
BOUNDARY_WINDOW_FT = 2.0

#: The 15 `wind_supported` venues from the frozen step 2 rule (plan, "Step 2 result").
WIND_SUPPORTED_VENUE_IDS: frozenset[int] = frozenset(
    {17, 31, 2681, 2395, 7, 2, 1, 2394, 3, 3309, 4, 2602, 3312, 3313, 2889}
)

_HR = CLASS_ORDER.index("home_run")
_REQUIRED = (
    "event_id",
    "season",
    "game_pk",
    "game_date",
    "venue_id",
    "has_venue_metadata",
    "launch_speed",
    "launch_angle",
    "spray_angle_approx",
    "hit_distance_sc",
    "bb_type",
    "stand",
    "eligible_for_training",
    "outcome_class",
)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def join_v12_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Join the v1.2 geometry table and add the v0.4 interaction features."""
    joined = join_park_geometry(
        df, configs=PARK_GEOMETRY_CONFIGS_V12, excluded_venue_ids=V12_EXCLUDED_VENUE_IDS
    )
    return add_geometry_interaction_features(joined)


def prepare_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Training-eligible development rows with v1.2 geometry. Refuses 2025/2026."""
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    seasons = {int(s) for s in pd.unique(df["season"].dropna())}
    outside = seasons - set(DEVELOPMENT_SEASONS)
    if outside:
        raise ValueError(f"season(s) {sorted(outside)} are not development seasons")
    eligible = df["eligible_for_training"].astype("boolean").fillna(False).astype(bool)
    out = df.loc[eligible].copy()
    if "geometry_status" not in out.columns:
        out = join_v12_geometry(out)
    return out.reset_index(drop=True)


def recompute_geometry_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute every column derived from wall distance, wall height or hit distance.

    Required for perturbations (`RESEARCH_RULES.md`, "Any override of a raw feature must
    recompute every feature DERIVED from it").
    """
    out = df.copy()
    ok = out["geometry_status"] == GEOMETRY_STATUS_OK
    dist = pd.to_numeric(out["hit_distance_sc"], errors="coerce")
    margin = (dist - out["wall_distance_in_spray_direction"]).where(ok & dist.notna())
    out["projected_distance_to_wall_margin"] = margin
    out["absolute_distance_to_wall"] = margin.abs()
    out["wall_margin_x_launch_angle"] = margin * out["launch_angle"]
    out["launch_angle_x_wall_height"] = out["launch_angle"] * out["wall_height_in_spray_direction"]
    out["spray_angle_x_wall_distance"] = (
        out["spray_angle_approx"] * out["wall_distance_in_spray_direction"]
    )

    def _flag(values: pd.Series) -> pd.Series:
        return values.where(margin.notna()).map(lambda v: None if pd.isna(v) else str(bool(v)))

    for threshold in NEAR_WALL_THRESHOLDS_FT:
        out[f"near_wall_{threshold}ft"] = _flag(margin.abs() <= threshold).astype(object)
    out["projected_beyond_wall"] = _flag(margin > 0).astype(object)
    return out


def gate_mask(df: pd.DataFrame) -> pd.Series:
    """Rows the specialist scores (see module docstring)."""
    margin = pd.to_numeric(df["projected_distance_to_wall_margin"], errors="coerce")
    return (
        (df["geometry_status"] == GEOMETRY_STATUS_OK)
        & df["bb_type"].isin(GATE_BB_TYPES)
        & (margin >= GATE_MARGIN_FT).fillna(False)
    ).astype(bool)


def wall_height_bin(height: pd.Series) -> pd.Series:
    """Fixed wall-height bins: short <= 8 ft, tall >= 15 ft, medium between, else unknown."""
    h = pd.to_numeric(height, errors="coerce")
    bins = np.where(
        h.isna(),
        "unknown",
        np.where(
            h <= SHORT_WALL_MAX_FT, "short", np.where(h >= TALL_WALL_MIN_FT, "tall", "medium")
        ),
    )
    return pd.Series(bins, index=height.index)


def evaluation_folds() -> dict[str, list[tuple[tuple[int, ...], int]]]:
    """The frozen fold layout: LOSO within 2021-2023, then 2021-2023 -> 2024."""
    primary = [(tuple(s for s in PRIMARY_SEASONS if s != held), held) for held in PRIMARY_SEASONS]
    return {"primary": primary, "confirmation": [(PRIMARY_SEASONS, CONFIRMATION_SEASON)]}


# ---------------------------------------------------------------------------
# Fitting and prediction
# ---------------------------------------------------------------------------


@dataclass
class FoldModels:
    baseline: TrainedModel
    specialist: TrainedModel


def fit_fold(train: pd.DataFrame) -> FoldModels:
    """Fit `baseline_v02` on all training rows and the specialist on the gated ones."""
    baseline = train_model(train)
    specialist = train_model(
        train.loc[gate_mask(train)], extra_numeric_features=SPECIALIST_FEATURES
    )
    dropped = [f for f in SPECIALIST_FEATURES if f not in specialist.numeric_features]
    if dropped:
        raise ValueError(
            f"the trainer dropped specialist feature(s) {dropped} (missingness rule); the frozen "
            "candidate needs all five"
        )
    return FoldModels(baseline=baseline, specialist=specialist)


def _proba(model: TrainedModel, df: pd.DataFrame) -> pd.DataFrame:
    return predict_proba_ordered(model, df[model.numeric_features + model.categorical_features])


def predict_fold(models: FoldModels, df: pd.DataFrame) -> dict[str, Any]:
    """Baseline and candidate probabilities (columns in `CLASS_ORDER`) plus the gate."""
    gate = gate_mask(df)
    baseline = _proba(models.baseline, df)
    candidate = baseline.copy()
    if gate.any():
        candidate.loc[gate] = _proba(models.specialist, df.loc[gate]).to_numpy()
    if not np.array_equal(baseline.loc[~gate].to_numpy(), candidate.loc[~gate].to_numpy()):
        raise AssertionError("open-field rows differ from baseline_v02")
    return {"baseline": baseline, "candidate": candidate, "gate": gate}


def _perturbation_sums(models: FoldModels, gated: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-venue sums of P(HR) under each perturbation pair, for pooling across folds."""
    known_height = gated["wall_height_in_spray_direction"].notna()
    scenarios = {
        "farther_wall": (
            gated,
            {"wall_distance_in_spray_direction": -PERTURB_WALL_FT},
            {"wall_distance_in_spray_direction": +PERTURB_WALL_FT},
            "set_delta",
        ),
        "taller_wall": (
            gated.loc[known_height],
            {"wall_height_in_spray_direction": PERTURB_HEIGHT_LOW_FT},
            {"wall_height_in_spray_direction": PERTURB_HEIGHT_HIGH_FT},
            "set_value",
        ),
        "longer_ball": (
            gated,
            {"hit_distance_sc": -PERTURB_DISTANCE_FT},
            {"hit_distance_sc": +PERTURB_DISTANCE_FT},
            "set_delta",
        ),
    }
    out: list[dict[str, Any]] = []
    for name, (rows, low, high, mode) in scenarios.items():
        if rows.empty:
            continue
        p: dict[str, np.ndarray] = {}
        for label, change in (("low", low), ("high", high)):
            scen = rows.copy()
            for col, v in change.items():
                scen[col] = scen[col] + v if mode == "set_delta" else v
            scen = recompute_geometry_derived(scen)
            p[label] = _proba(models.specialist, scen).iloc[:, _HR].to_numpy()
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


def _boundary_sums(models: FoldModels, df: pd.DataFrame) -> dict[str, float]:
    margin = pd.to_numeric(df["projected_distance_to_wall_margin"], errors="coerce")
    near = (
        (df["geometry_status"] == GEOMETRY_STATUS_OK)
        & df["bb_type"].isin(GATE_BB_TYPES)
        & ((margin - GATE_MARGIN_FT).abs() <= BOUNDARY_WINDOW_FT).fillna(False)
    )
    rows = df.loc[near]
    if rows.empty:
        return {"n": 0, "sum_abs_diff_p_hr": 0.0}
    spec = _proba(models.specialist, rows).iloc[:, _HR].to_numpy()
    base = _proba(models.baseline, rows).iloc[:, _HR].to_numpy()
    return {"n": len(rows), "sum_abs_diff_p_hr": float(np.abs(spec - base).sum())}


@dataclass
class FoldRunResult:
    predictions: pd.DataFrame
    perturbations: list[dict[str, Any]] = field(default_factory=list)
    boundary: list[dict[str, float]] = field(default_factory=list)


_CARRY_COLUMNS = (
    "event_id",
    "season",
    "game_pk",
    "venue_id",
    "outcome_class",
    "bb_type",
    "projected_distance_to_wall_margin",
    "wall_height_in_spray_direction",
    "geometry_status",
)


def run_folds(rows: pd.DataFrame, folds: list[tuple[tuple[int, ...], int]]) -> FoldRunResult:
    """Fit each fold on its training seasons and predict its held-out season."""
    frames: list[pd.DataFrame] = []
    result = FoldRunResult(predictions=pd.DataFrame())
    for train_seasons, held_out in folds:
        if held_out in train_seasons:
            raise ValueError(f"held-out season {held_out} is in the training seasons")
        train = rows.loc[rows["season"].isin(train_seasons)]
        held = rows.loc[rows["season"] == held_out]
        logger.info(
            "fold: train %s -> evaluate %s (%d / %d rows)",
            train_seasons,
            held_out,
            len(train),
            len(held),
        )
        models = fit_fold(train)
        pred = predict_fold(models, held)
        carry = [c for c in (*_CARRY_COLUMNS, "spray_sector") if c in held.columns]
        frame = held[carry].copy()
        frame["gate"] = pred["gate"].to_numpy()
        for prefix, proba in (("base_", pred["baseline"]), ("cand_", pred["candidate"])):
            for cls in CLASS_ORDER:
                frame[f"{prefix}{cls}"] = proba[cls].to_numpy()
        frames.append(frame)
        result.perturbations.extend(_perturbation_sums(models, held.loc[pred["gate"]]))
        result.boundary.append(_boundary_sums(models, held))
    result.predictions = pd.concat(frames, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def summarize_perturbations(sums: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pool per-venue sums: delta = mean P(HR) high minus low, with the expected sign."""
    expected = {"farther_wall": -1, "taller_wall": -1, "longer_ball": +1}
    frame = pd.DataFrame(sums)
    out: dict[str, dict[str, Any]] = {}
    for check, sign in expected.items():
        g = frame[frame["check"] == check] if not frame.empty else frame
        n = int(g["n"].sum()) if not g.empty else 0
        delta = float((g["sum_high"].sum() - g["sum_low"].sum()) / n) if n else float("nan")
        by_venue = (
            g.groupby("venue_id")
            .apply(lambda v: (v["sum_high"].sum() - v["sum_low"].sum()) / v["n"].sum())
            .to_dict()
            if n
            else {}
        )
        out[check] = {
            "expected_sign": sign,
            "n_rows": n,
            "delta_p_hr": delta,
            "passed": bool(n > 0 and np.sign(delta) == sign),
            "venues_backwards": sorted(int(k) for k, v in by_venue.items() if np.sign(v) != sign),
            "delta_by_venue": {int(k): float(v) for k, v in by_venue.items()},
        }
    return out


def material_ece_regression(baseline_ece: float, candidate_ece: float) -> bool:
    """The v0.3 material-regression rule (absolute 0.01 OR relative 50%)."""
    delta = candidate_ece - baseline_ece
    relative = delta / baseline_ece if baseline_ece > 0 else (np.inf if delta > 0 else 0.0)
    return bool(
        delta > DEFAULT_MATERIAL_ECE_ABSOLUTE_MARGIN
        or relative > DEFAULT_MATERIAL_ECE_RELATIVE_MARGIN
    )


def _probas(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pred[[f"base_{c}" for c in CLASS_ORDER]].set_axis(list(CLASS_ORDER), axis=1)
    cand = pred[[f"cand_{c}" for c in CLASS_ORDER]].set_axis(list(CLASS_ORDER), axis=1)
    return base, cand


def _boot(pred: pd.DataFrame, metrics: tuple[str, ...], **kw: Any) -> dict[str, Any]:
    base, cand = _probas(pred)
    res = compute_paired_bootstrap(
        pred["outcome_class"].astype(str), base, cand, pred["game_pk"], metrics=metrics, **kw
    )
    return {m: dict(v) for m, v in res.items()}


def _group_masks(gated: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    for venue_id in sorted(gated["venue_id"].dropna().unique()):
        masks[f"venue_{int(venue_id)}"] = gated["venue_id"] == venue_id
    for b in ("short", "medium", "tall", "unknown"):
        masks[f"wall_height_{b}"] = wall_height_bin(gated["wall_height_in_spray_direction"]) == b
    if "spray_sector" in gated.columns:
        for s in sorted(gated["spray_sector"].dropna().unique()):
            masks[f"spray_sector_{s}"] = gated["spray_sector"] == s
    for b in GATE_BB_TYPES:
        masks[f"bb_type_{b}"] = gated["bb_type"] == b
    m = pd.to_numeric(gated["projected_distance_to_wall_margin"], errors="coerce")
    masks["band_beyond_wall"] = m >= 0
    masks["band_short_0_5"] = (m >= -5) & (m < 0)
    masks["band_short_5_10"] = (m >= -10) & (m < -5)
    masks["band_short_10_20"] = (m >= -20) & (m < -10)
    return masks


def evaluate_groups(pred: pd.DataFrame, *, n_reps: int, seed: int) -> list[dict[str, Any]]:
    """Paired log-loss regression gate per venue/subgroup (gated rows), plus reported
    absolute home-run ECE status for both models (v0.7D part a)."""
    gated = pred.loc[pred["gate"]].reset_index(drop=True)
    out: list[dict[str, Any]] = []
    for name, mask in _group_masks(gated).items():
        sub = gated.loc[mask.fillna(False).to_numpy()]
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
            ll = _boot(sub, ("log_loss",), n_reps=n_reps, seed=seed)["log_loss"]
            record.update(
                paired_log_loss_point=ll["point_estimate"],
                paired_log_loss_ci_low=ll["ci_low"],
                paired_log_loss_ci_high=ll["ci_high"],
            )
            y_hr = (sub["outcome_class"] == "home_run").to_numpy().astype(float)
            boot = compute_subgroup_bootstrap(
                y_hr,
                sub["cand_home_run"].to_numpy(),
                sub["base_home_run"].to_numpy(),
                sub["game_pk"].to_numpy(),
                n_reps=n_reps,
                seed=seed,
            )
            support = has_adequate_support(n_plays, n_games, int(y_hr.sum()), int((1 - y_hr).sum()))
            for model, key in (("candidate", "specialist_ece"), ("baseline", "baseline_ece")):
                status, _reason = classify_subgroup_status(
                    has_adequate_support=support,
                    specialist_ece_ci_low=boot[key]["ci_low"],
                    specialist_ece_ci_high=boot[key]["ci_high"],
                    paired_log_loss_delta_ci_low=None,
                )
                record[f"{model}_hr_ece"] = boot[key]["point_estimate"]
                record[f"{model}_hr_ece_ci"] = [boot[key]["ci_low"], boot[key]["ci_high"]]
                record[f"{model}_hr_ece_status_reported_only"] = status
        out.append(record)
    return out


def weather_residual_diagnostic(
    pred: pd.DataFrame, weather: pd.DataFrame, *, n_reps: int, seed: int
) -> dict[str, Any]:
    """Within-venue association of the home-run residual with air density and following
    wind, gated outdoor/roof-open rows, for both models. Reported only."""
    gated = pred.loc[pred["gate"]].merge(weather, on="event_id", how="inner")
    gated = gated[gated["roof_status"].isin(("outdoor_open_air", "retractable_roof_open"))]
    y = (gated["outcome_class"] == "home_run").astype(float)
    out: dict[str, Any] = {}
    for term, rows in (
        ("air_density_kg_m3", gated[gated["air_density_kg_m3"].notna()]),
        (
            "following_wind_mps",
            gated[
                gated["following_wind_mps"].notna()
                & gated["venue_id"].isin(WIND_SUPPORTED_VENUE_IDS)
            ],
        ),
    ):
        res: dict[str, Any] = {"n_rows": len(rows), "n_games": int(rows["game_pk"].nunique())}
        if len(rows) < MIN_SUBGROUP_PLAYS:
            out[term] = res
            continue
        x = rows[term] - rows.groupby("venue_id")[term].transform("mean")
        games = rows["game_pk"].to_numpy()
        ug = np.unique(games)
        idx = {g: np.where(games == g)[0] for g in ug}
        rng = np.random.default_rng(seed)
        picks = [rng.choice(ug, size=len(ug), replace=True) for _ in range(n_reps)]
        for model in ("base", "cand"):
            r = y.loc[rows.index] - rows[f"{model}_home_run"]
            r = r - r.groupby(rows["venue_id"]).transform("mean")
            xv, rv = x.to_numpy(), r.to_numpy()
            slope = float((xv * rv).sum() / (xv * xv).sum())
            boots = []
            for pick in picks:
                ii = np.concatenate([idx[g] for g in pick])
                boots.append((xv[ii] * rv[ii]).sum() / (xv[ii] * xv[ii]).sum())
            lo, hi = np.quantile(boots, [0.025, 0.975])
            res[model] = {"slope": slope, "ci_low": float(lo), "ci_high": float(hi)}
        out[term] = res
    return out


def recommend_v12_adoption(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen adoption criteria. Input to a maintainer decision, never the decision."""
    criteria = {
        "log_loss": summary["log_loss"]["ci_high"] < 0,
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
        "open_field_identical": bool(summary["open_field_identical"]),
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
    overall = _boot(
        pred, ("log_loss", "ece", "home_run_ece"), n_reps=n_reps, seed=seed, ci=DEFAULT_BOOTSTRAP_CI
    )
    gated_ll = _boot(pred.loc[pred["gate"]], ("log_loss",), n_reps=n_reps, seed=seed)["log_loss"]
    base, cand = _probas(pred)
    y_idx = np.array([CLASS_ORDER.index(str(c)) for c in pred["outcome_class"]])
    rows = np.arange(len(pred))
    ece = {
        "baseline": _fast_ece(base.to_numpy(), y_idx, rows),
        "candidate": _fast_ece(cand.to_numpy(), y_idx, rows),
    }
    hr_ece = {
        "baseline": _fast_class_ece(base.to_numpy(), y_idx, rows, _HR),
        "candidate": _fast_class_ece(cand.to_numpy(), y_idx, rows, _HR),
    }
    conf = confirmation.predictions
    conf_ll = _boot(conf, ("log_loss",), n_reps=n_reps, seed=seed)["log_loss"]
    boundary_n = sum(b["n"] for b in primary.boundary)
    summary: dict[str, Any] = {
        "candidate": CANDIDATE_NAME,
        "n_rows": len(pred),
        "n_gated": int(pred["gate"].sum()),
        "log_loss": overall["log_loss"],
        "log_loss_gated_rows_only": gated_ll,
        "calibration": {
            "ece": ece,
            "home_run_ece": hr_ece,
            "ece_paired": overall["ece"],
            "home_run_ece_paired": overall["home_run_ece"],
            "overall_material_regression": material_ece_regression(
                ece["baseline"], ece["candidate"]
            ),
            "home_run_material_regression": material_ece_regression(
                hr_ece["baseline"], hr_ece["candidate"]
            ),
        },
        "groups": evaluate_groups(pred, n_reps=n_reps, seed=seed),
        "perturbations": summarize_perturbations(primary.perturbations),
        "open_field_identical": True,  # predict_fold raises otherwise
        "confirmation_2024": {**conf_ll, "credible_regression": conf_ll["ci_low"] > 0},
        "reported_only": {
            "gate_boundary_mean_abs_diff_p_hr": (
                sum(b["sum_abs_diff_p_hr"] for b in primary.boundary) / boundary_n
                if boundary_n
                else None
            ),
            "gate_boundary_n": boundary_n,
            "height_known_share_gated": float(
                pred.loc[pred["gate"], "wall_height_in_spray_direction"].notna().mean()
            ),
            "confirmation_perturbations": summarize_perturbations(confirmation.perturbations),
        },
    }
    if weather is not None:
        summary["reported_only"]["weather_residual"] = weather_residual_diagnostic(
            pred, weather, n_reps=n_reps, seed=seed
        )
    summary["recommendation"] = recommend_v12_adoption(summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_INPUT = PROCESSED_DATA_DIR / "cleaned_development_data_with_venue.parquet"
DEFAULT_WEATHER = PROCESSED_DATA_DIR / "cleaned_development_data_with_weather.parquet"
OUTPUT_FILENAME = "v1_2_gated_geometry_evaluation.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--weather", type=Path, default=DEFAULT_WEATHER)
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    cols = [*_REQUIRED, "venue", "spray_sector"]
    rows = prepare_rows(pd.read_parquet(args.input, columns=cols))
    weather = pd.read_parquet(
        args.weather, columns=["event_id", "roof_status", "air_density_kg_m3", "following_wind_mps"]
    )
    folds = evaluation_folds()
    primary = run_folds(rows, folds["primary"])
    confirmation = run_folds(rows, folds["confirmation"])
    summary = evaluate(primary, confirmation, weather)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / OUTPUT_FILENAME
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("wrote %s; recommendation: %s", out_path, summary["recommendation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
