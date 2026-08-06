"""Contact Luck v1.0: system-level checks and predeclared component metrics.

Versions 0.2-0.12 are FROZEN. This module computes NO new model logic --
every metric/gate function it calls is IMPORTED, UNCHANGED, from the
existing frozen `compare_*.py`/`train_*.py`/`scoring/*.py` modules. It exists
only to apply those already-frozen functions to 2025 rows without going
through the SEASON-HARDCODED orchestration wrappers those modules use for
their own 2024 development comparisons (e.g. `run_infield_final_comparison`
filters to `CALIBRATION_EVAL_SEASONS == (2024,)` internally -- not reusable
for 2025 as-is). Where a module's decision/aggregation logic IS generic
(takes pre-computed evidence objects, not a season-filtered DataFrame --
`summarize_infield_calibration`, `summarize_advancement_calibration`,
`evaluate_subgroup_calibration`, `run_infield_perturbation_checks`, `run_
advancement_perturbation_checks`, `run_near_wall_perturbation_checks`,
`compute_near_wall_subgroups`, `build_infield_subgroup_masks`, `build_
infield_venue_masks`, `compute_reached_on_error_comparison`), it is reused
VERBATIM, unchanged, on 2025 data -- only the season-scoped data-fetching
glue (trivial: the caller already passes an already-2025-scoped DataFrame)
is written fresh here.

Per the task's explicit instruction, the near-wall specialist and infield/
advancement models are EVALUATED here for their predeclared metrics, but
this module makes NO adoption decision -- Version 0.7's near-wall specialist
stays `provisional` regardless of how it performs on 2025 (see `evaluate_
near_wall_specialist`'s docstring).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from mlb_luck_score.eligibility import summarize_advancement_parse_coverage
from mlb_luck_score.models.compare_advancement_models import (
    ADVANCEMENT_MATERIAL_ECE_ABSOLUTE_THRESHOLD,
    EmpiricalAdvancementBaseline,
    predict_empirical_baseline_proba,
    run_advancement_perturbation_checks,
    summarize_advancement_calibration,
)
from mlb_luck_score.models.compare_infield_opportunity import (
    build_infield_subgroup_masks,
    build_infield_venue_masks,
    compute_reached_on_error_comparison,
    run_infield_perturbation_checks,
    summarize_infield_calibration,
)
from mlb_luck_score.models.compare_near_wall_calibration_gate import (
    compute_adaptive_calibration_table,
    compute_calibration_intercept_slope,
    evaluate_subgroup_calibration,
)
from mlb_luck_score.models.compare_near_wall_models import (
    compute_near_wall_subgroups,
    run_near_wall_perturbation_checks,
)
from mlb_luck_score.models.compare_opportunity_models import compute_binary_ece
from mlb_luck_score.models.train_advancement_model import (
    TrainedAdvancementModel,
    predict_advancement_proba,
)
from mlb_luck_score.models.train_contact_model import TrainedModel, evaluate_model
from mlb_luck_score.models.train_opportunity_model import (
    TrainedOpportunityModel,
    evaluate_opportunity_model,
    predict_opportunity_proba,
)
from mlb_luck_score.scoring.aggregate_attribution import verify_season_identity
from mlb_luck_score.scoring.attribution_ledger import (
    assert_unique_play_identifiers,
    verify_attribution_identity,
)

CLASS_ORDER: tuple[str, ...] = ("out", "single", "double", "triple", "home_run")


class SystemEvaluationError(ValueError):
    """Raised when a required system-level invariant is violated on 2025 data."""


# ---------------------------------------------------------------------------
# Accounting, routing, and reproducibility
# ---------------------------------------------------------------------------


def _abs_error_percentiles(errors: np.ndarray) -> dict[str, float | None]:
    if len(errors) == 0:
        return {"max": None, "median": None, "p95": None, "p99": None}
    return {
        "max": float(np.max(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
    }


def verify_play_level_accounting(ledger: pd.DataFrame) -> dict[str, object]:
    """Re-verify Version 0.10's exact play-level accounting identity on 2025
    rows and report max/median/p95/p99 absolute reconciliation error.
    """
    holds = verify_attribution_identity(ledger)
    resolved = holds.dropna()
    if len(resolved) == 0:
        raise SystemEvaluationError("No resolved rows to verify the play-level identity against")
    if not bool(resolved.all()):
        bad = int((~resolved).sum())
        raise SystemEvaluationError(
            f"Play-level accounting identity failed for {bad} of {len(resolved)} resolved rows"
        )

    lhs = ledger["observed_final_run_value"] - ledger["baseline_expected_contact_run_value"]
    rhs = (
        ledger["unexplained_residual"]
        + ledger["defensive_execution_contribution"].fillna(0.0)
        + ledger["advancement_execution_contribution"].fillna(0.0)
    )
    resolved_mask = ledger["observed_contact_result_run_value"].notna()
    errors = (lhs - rhs).abs().loc[resolved_mask].to_numpy()

    return {
        "identity_holds_for_every_resolved_row": True,
        "resolved_row_count": int(len(resolved)),
        "abs_reconciliation_error": _abs_error_percentiles(errors),
    }


def verify_season_level_accounting(summary: pd.DataFrame) -> dict[str, object]:
    """Re-verify Version 0.11's season-level accounting identity on 2025
    player-seasons and report max/median/p95/p99 absolute reconciliation
    error.
    """
    holds = verify_season_identity(summary)
    if not bool(holds.all()):
        bad = int((~holds).sum())
        raise SystemEvaluationError(
            f"Season-level accounting identity failed for {bad} of {len(summary)} rows"
        )

    lhs = summary["total_observed_minus_expected_runs"].to_numpy()
    rhs = (
        summary["total_contact_component_runs"]
        + summary["total_unexplained_residual_component_runs"]
        + summary["total_defensive_execution_component_runs"]
        + summary["total_advancement_component_runs"]
    ).to_numpy()
    errors = np.abs(lhs - rhs)

    return {
        "identity_holds_for_every_row": True,
        "row_count": int(len(summary)),
        "abs_reconciliation_error": _abs_error_percentiles(errors),
    }


def verify_routing_exclusivity(df: pd.DataFrame) -> dict[str, object]:
    """Confirm outfield/infield opportunity routing is mutually exclusive on
    2025 rows -- `mlb_luck_score.features.build_contact_features.
    add_opportunity_features_by_domain` already raises at construction time
    if this is violated; this re-derives the same check descriptively for
    the report.
    """
    outfield = df["outfield_opportunity_eligible"].astype(bool)
    infield = df["infield_opportunity_eligible"].astype(bool)
    both = outfield & infield
    if both.any():
        raise SystemEvaluationError(
            f"{int(both.sum())} row(s) eligible for both outfield and infield opportunity domains"
        )
    return {
        "mutually_exclusive": True,
        "outfield_eligible_rows": int(outfield.sum()),
        "infield_eligible_rows": int(infield.sum()),
        "neither_domain_rows": int((~outfield & ~infield).sum()),
    }


def verify_advancement_eligibility(df: pd.DataFrame) -> dict[str, object]:
    """Advancement eligibility must be a SUBSET of outfield-air-ball rows
    (Version 0.9 scope) -- confirm this holds on 2025.
    """
    advancement_eligible = df["advancement_eligible"].astype(bool)
    outfield_air_ball = df["bb_type"].isin({"fly_ball", "line_drive"})
    bad = advancement_eligible & ~outfield_air_ball
    if bad.any():
        raise SystemEvaluationError(
            f"{int(bad.sum())} advancement_eligible row(s) are not outfield air balls"
        )
    return {
        "advancement_eligible_rows": int(advancement_eligible.sum()),
        "subset_of_outfield_air_balls": True,
    }


def verify_event_ids(df: pd.DataFrame) -> dict[str, object]:
    assert_unique_play_identifiers(df)  # raises on duplicate/null
    return {"unique_non_null_event_ids": True, "row_count": int(len(df))}


def verify_no_experimental_components_in_official_total(ledger: pd.DataFrame) -> dict[str, object]:
    """Confirm no weather/alignment-named column exists anywhere in the
    frozen ledger's output -- Version 0.5/0.5.1/0.6 never passed adoption
    (see CLAUDE.md) and must never enter the official total.
    """
    flagged = [c for c in ledger.columns if "weather" in c.lower() or "alignment" in c.lower()]
    if flagged:
        raise SystemEvaluationError(
            f"Experimental weather/alignment column(s) found in the official ledger: {flagged}"
        )
    return {"no_weather_or_alignment_columns_in_official_ledger": True}


def verify_reproducibility(build_fn: object, *args: object, **kwargs: object) -> dict[str, object]:
    """Call `build_fn(*args, **kwargs)` TWICE and confirm bit-identical
    output -- `build_fn` must be a pure function returning a DataFrame (e.g.
    a bootstrap or aggregation call) for this check to be meaningful.
    """
    first = build_fn(*args, **kwargs)  # type: ignore[operator]
    second = build_fn(*args, **kwargs)  # type: ignore[operator]
    identical = first.equals(second)
    if not identical:
        raise SystemEvaluationError("Repeated execution did not produce identical output")
    return {"deterministic": True}


def verify_qualification_and_ranking_contract(
    public_score_table: pd.DataFrame,
) -> dict[str, object]:
    from mlb_luck_score.scoring.public_score_schema import validate_public_score_table
    from mlb_luck_score.scoring.qualification import STATUS_QUALIFIED

    validate_public_score_table(public_score_table)  # raises on any contract violation

    is_qualified = public_score_table["qualification_status"] == STATUS_QUALIFIED
    ranked = (
        public_score_table["official_rank_favorable"].notna()
        | public_score_table["official_rank_unfavorable"].notna()
    )
    non_qualified_ranked = ranked & ~is_qualified
    if non_qualified_ranked.any():
        raise SystemEvaluationError(
            f"{int(non_qualified_ranked.sum())} non-qualified row(s) carry an official rank"
        )
    has_rate = public_score_table["contact_luck_runs_per_100"].notna()
    has_interval = (
        public_score_table["lower_95_interval"].notna()
        & public_score_table["upper_95_interval"].notna()
    )
    if not (has_rate == has_interval).all():
        raise SystemEvaluationError("Some rows have a point estimate without a paired interval")

    return {
        "schema_valid": True,
        "no_rank_on_non_qualified_rows": True,
        "point_estimates_always_paired_with_intervals": True,
        "qualified_count": int(is_qualified.sum()),
    }


# ---------------------------------------------------------------------------
# Contact model (Version 0.2)
# ---------------------------------------------------------------------------


def evaluate_contact_model(trained: TrainedModel, eval_df: pd.DataFrame) -> dict[str, object]:
    metrics = evaluate_model(trained, eval_df)
    from mlb_luck_score.models.train_contact_model import _prepare_xy, predict_proba_ordered

    x_eval, y_eval = _prepare_xy(eval_df, trained.numeric_features, trained.categorical_features)
    proba_df = predict_proba_ordered(trained, x_eval)
    y_true = y_eval.to_numpy()

    classwise_ece: dict[str, float | None] = {}
    classwise_intercept_slope: dict[str, tuple[float | None, float | None]] = {}
    for cls in CLASS_ORDER:
        y_binary = (y_true == cls).astype(int)
        p_cls = proba_df[cls]
        table = compute_adaptive_calibration_table(y_binary, p_cls)
        classwise_ece[cls] = compute_binary_ece(table) if not table.empty else None
        classwise_intercept_slope[cls] = compute_calibration_intercept_slope(
            y_binary, p_cls.to_numpy()
        )

    metrics["classwise_adaptive_ece"] = classwise_ece
    metrics["classwise_calibration_intercept_slope"] = classwise_intercept_slope
    metrics["outcome_prevalence"] = {cls: float((y_true == cls).mean()) for cls in CLASS_ORDER}
    return metrics


# ---------------------------------------------------------------------------
# Outfield open-field model (Version 0.7A) and near-wall specialist (0.7C/D)
# ---------------------------------------------------------------------------


def _binary_overall_metrics(
    y_true: np.ndarray, p_out: pd.Series, *, variant: str
) -> dict[str, object]:
    table = compute_adaptive_calibration_table(y_true, p_out)
    ece = compute_binary_ece(table) if not table.empty else None
    intercept, slope = compute_calibration_intercept_slope(y_true, p_out.to_numpy())
    return {
        "variant": variant,
        "sample_count": int(len(y_true)),
        "binary_log_loss": float(log_loss(y_true, p_out.to_numpy(), labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, p_out.to_numpy())),
        "expected_calibration_error": ece,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "outcome_prevalence": float(y_true.mean()),
        "mean_predicted_p_out": float(p_out.mean()),
    }


def evaluate_open_field_outfield_model(
    trained: TrainedOpportunityModel, eval_df: pd.DataFrame
) -> dict[str, object]:
    """Predeclared metrics for the open-field outfield model (`measured_
    contact_only_v07`) on 2025 -- the model actually feeding the official
    ledger/public score for outfield rows.
    """
    from mlb_luck_score.features.build_contact_features import OUTFIELD_OPPORTUNITY_TARGET_COLUMN

    metrics = evaluate_opportunity_model(trained, eval_df)
    feature_cols = trained.numeric_features + trained.categorical_features
    p_out = predict_opportunity_proba(trained, eval_df[feature_cols])
    y_true = eval_df[OUTFIELD_OPPORTUNITY_TARGET_COLUMN].astype(int).to_numpy()
    overall = _binary_overall_metrics(y_true, p_out, variant="measured_contact_only_v07")
    metrics["expected_calibration_error"] = overall["expected_calibration_error"]
    metrics["calibration_intercept"] = overall["calibration_intercept"]
    metrics["calibration_slope"] = overall["calibration_slope"]
    return metrics


def evaluate_near_wall_specialist(
    near_wall_trained: TrainedOpportunityModel,
    open_field_trained: TrainedOpportunityModel,
    near_wall_eval_df: pd.DataFrame,
) -> dict[str, object]:
    """Predeclared metrics for the FROZEN Version 0.7C near-wall specialist
    on 2025 near-wall rows, plus a paired comparison against its frozen
    fallback (the open-field model) on IDENTICAL rows.

    This function makes NO adoption decision. Regardless of how the
    specialist performs on 2025, its status remains `provisional` -- Version
    0.7 is frozen (see CLAUDE.md "Version 0.7 is now FROZEN"), and adopting
    it would require a genuinely new development version evaluated on
    NEW (non-2024, non-2025) data, never decided from a single final
    -evaluation run. See the Version 1.0 task's explicit instruction:
    "preserve provisional status regardless of apparent improvement."
    """
    from mlb_luck_score.features.build_contact_features import OUTFIELD_OPPORTUNITY_TARGET_COLUMN

    y_true = near_wall_eval_df[OUTFIELD_OPPORTUNITY_TARGET_COLUMN].astype(int).to_numpy()
    specialist_feature_cols = (
        near_wall_trained.numeric_features + near_wall_trained.categorical_features
    )
    specialist_p_out = predict_opportunity_proba(
        near_wall_trained, near_wall_eval_df[specialist_feature_cols]
    )
    fallback_feature_cols = (
        open_field_trained.numeric_features + open_field_trained.categorical_features
    )
    fallback_p_out = predict_opportunity_proba(
        open_field_trained, near_wall_eval_df[fallback_feature_cols]
    )

    specialist_overall = _binary_overall_metrics(
        y_true, specialist_p_out, variant=near_wall_trained.variant
    )
    fallback_overall = _binary_overall_metrics(
        y_true, fallback_p_out, variant="open_field_fallback"
    )

    subgroups = compute_near_wall_subgroups(y_true, specialist_p_out, near_wall_eval_df)
    game_pks = (
        near_wall_eval_df["game_pk"]
        if "game_pk" in near_wall_eval_df.columns
        else pd.Series(range(len(near_wall_eval_df)))
    )
    paired_vs_fallback = evaluate_subgroup_calibration(
        "near_wall_vs_open_field_fallback",
        np.ones(len(near_wall_eval_df), dtype=bool),
        y_true,
        specialist_p_out,
        fallback_p_out,
        game_pks,
    )
    perturbation = run_near_wall_perturbation_checks(near_wall_trained, near_wall_eval_df)

    return {
        "specialist_overall": specialist_overall,
        "fallback_overall": fallback_overall,
        "subgroup_calibration": subgroups,
        "paired_comparison_vs_fallback": paired_vs_fallback,
        "perturbation_checks": perturbation,
        "status": "provisional",
        "status_reason": (
            "Version 0.7 is frozen; the near-wall specialist is never adopted from a single "
            "final-evaluation run regardless of 2025 performance"
        ),
    }


# ---------------------------------------------------------------------------
# Infield model (Version 0.8)
# ---------------------------------------------------------------------------


def evaluate_infield_model(
    trained: TrainedOpportunityModel, infield_df: pd.DataFrame
) -> dict[str, object]:
    from mlb_luck_score.features.build_contact_features import INFIELD_OPPORTUNITY_TARGET_COLUMN

    feature_cols = trained.numeric_features + trained.categorical_features
    p_out = predict_opportunity_proba(trained, infield_df[feature_cols])
    y_true = infield_df[INFIELD_OPPORTUNITY_TARGET_COLUMN].astype(int).to_numpy()
    overall = _binary_overall_metrics(y_true, p_out, variant=trained.variant)

    subgroup_masks = build_infield_subgroup_masks(infield_df)
    venue_masks = (
        build_infield_venue_masks(infield_df["venue_id"])
        if "venue_id" in infield_df.columns
        else {}
    )
    game_pks = (
        infield_df["game_pk"]
        if "game_pk" in infield_df.columns
        else pd.Series(range(len(infield_df)))
    )
    subgroup_evidence = [
        evaluate_subgroup_calibration(label, mask, y_true, p_out, p_out, game_pks)
        for label, mask in subgroup_masks.items()
    ]
    venue_evidence = [
        evaluate_subgroup_calibration(label, mask, y_true, p_out, p_out, game_pks)
        for label, mask in venue_masks.items()
    ]
    perturbation = run_infield_perturbation_checks(trained, infield_df)
    reached_on_error = compute_reached_on_error_comparison(infield_df, p_out)
    gate_summary = summarize_infield_calibration(
        overall, subgroup_evidence, venue_evidence, perturbation, reached_on_error
    )

    return {
        "overall": overall,
        "subgroup_count": len(subgroup_evidence),
        "venue_count": len(venue_evidence),
        "perturbation_checks": perturbation,
        "reached_on_error_comparison": reached_on_error,
        "gate_summary": gate_summary,
    }


# ---------------------------------------------------------------------------
# Advancement model (Version 0.9)
# ---------------------------------------------------------------------------


def evaluate_advancement_model(
    trained: TrainedAdvancementModel,
    empirical_baseline: EmpiricalAdvancementBaseline,
    advancement_df: pd.DataFrame,
) -> dict[str, object]:
    from mlb_luck_score.eligibility import ADVANCEMENT_LABELS
    from mlb_luck_score.models.compare_advancement_models import VARIANT_ADVANCEMENT_EMPIRICAL

    feature_cols = trained.numeric_features + trained.categorical_features
    winner_proba = predict_advancement_proba(trained, advancement_df[feature_cols])
    empirical_proba = predict_empirical_baseline_proba(empirical_baseline, advancement_df)
    y_true = advancement_df["batter_final_base"].to_numpy()
    sorted_labels = sorted(ADVANCEMENT_LABELS)

    def _summarize(proba: pd.DataFrame, variant: str) -> dict[str, object]:
        multiclass_log_loss = float(
            log_loss(y_true, proba[sorted_labels].to_numpy(), labels=sorted_labels)
        )
        per_class_ece: dict[str, float | None] = {}
        per_class_brier: dict[str, float] = {}
        for cls in ADVANCEMENT_LABELS:
            y_binary = (y_true == cls).astype(int)
            table = compute_adaptive_calibration_table(y_binary, proba[cls])
            per_class_ece[cls] = compute_binary_ece(table) if not table.empty else None
            per_class_brier[cls] = float(brier_score_loss(y_binary, proba[cls].to_numpy()))
        return {
            "variant": variant,
            "sample_count": int(len(advancement_df)),
            "multiclass_log_loss": multiclass_log_loss,
            "per_class_ece": per_class_ece,
            "per_class_brier_score": per_class_brier,
            "class_frequencies": {cls: int((y_true == cls).sum()) for cls in ADVANCEMENT_LABELS},
        }

    overall_comparison = {
        trained.feature_set_label: _summarize(winner_proba, trained.feature_set_label),
        VARIANT_ADVANCEMENT_EMPIRICAL: _summarize(empirical_proba, VARIANT_ADVANCEMENT_EMPIRICAL),
    }
    perturbation = run_advancement_perturbation_checks(trained, advancement_df)
    parse_coverage = summarize_advancement_parse_coverage(advancement_df)
    gate_summary = summarize_advancement_calibration(
        overall_comparison, trained.feature_set_label, [], perturbation
    )

    return {
        "overall_comparison": overall_comparison,
        "perturbation_checks": perturbation,
        "parse_coverage": parse_coverage,
        "gate_summary": gate_summary,
        "material_ece_absolute_threshold": ADVANCEMENT_MATERIAL_ECE_ABSOLUTE_THRESHOLD,
    }
