"""Contact Luck v1.0: public-score evaluation summary, stability analyses,
and the PREDECLARED outcome classification.

Versions 0.2-0.12 are FROZEN. This module assembles descriptive summaries
over ALREADY-COMPUTED 2025 outputs (`evaluation.v1_system_evaluation`'s
checks/metrics, `evaluation.v1_distribution_shift`'s shift report, and the
Version 0.12 public score table) -- it computes no new predictions and
alters no frozen model, threshold, or public-score value.

## Outcome classification is fixed BEFORE any 2025 result exists

`classify_final_evaluation_outcome` implements the task's own three outcome
definitions VERBATIM, as code, written and tested before any real 2025 run
of `evaluation/run_v1_final_evaluation.py` occurs. Per the task's explicit
"Do not redefine these outcomes after seeing the results," this function
must never be edited to change what counts as a failure once a real
evaluation has been sealed (`evaluation.run_v1_final_evaluation`'s sealing
mechanism) -- see that module's docstring for what a genuine defect fix vs.
a forbidden post-hoc redefinition looks like.

A critical distinction this module preserves (task decision 9): a
provisional/`not_calibrated` component status affects how the component
DECOMPOSITION should be read, NOT whether the official ledger total is
trustworthy (that rests on the accounting identity, verified separately).
`any_required_component_gate_severely_failed` therefore means predictions
are literally invalid or uncomputable (non-finite, out-of-range, or a
required step raised) -- NOT "a calibration gate reported `not_calibrated`
or `provisional`," which instead sets `has_provisional_or_subgroup_
limitations` and yields `validated_with_documented_limitations`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from mlb_luck_score.scoring.run_season_aggregation import SeasonAggregationArtifacts

OUTCOME_VALIDATED_AS_FROZEN = "validated_as_frozen"
OUTCOME_VALIDATED_WITH_DOCUMENTED_LIMITATIONS = "validated_with_documented_limitations"
OUTCOME_FINAL_EVALUATION_FAILED = "final_evaluation_failed"
OUTCOME_VALUES: tuple[str, ...] = (
    OUTCOME_VALIDATED_AS_FROZEN,
    OUTCOME_VALIDATED_WITH_DOCUMENTED_LIMITATIONS,
    OUTCOME_FINAL_EVALUATION_FAILED,
)


@dataclass(frozen=True)
class OutcomeInputs:
    """Every input `classify_final_evaluation_outcome` needs -- assembled
    from `evaluation.v1_system_evaluation`'s checks and `evaluation.
    v1_distribution_shift`'s report by `run_v1_final_evaluation.py`.
    """

    accounting_passed: bool
    routing_passed: bool
    reproducibility_passed: bool
    schema_and_public_contract_passed: bool
    frozen_artifacts_reproducible: bool
    critical_input_or_schema_issue: bool
    any_required_component_gate_severely_failed: bool
    has_provisional_or_subgroup_limitations: bool
    notes: list[str] = field(default_factory=list)


def classify_final_evaluation_outcome(inputs: OutcomeInputs) -> tuple[str, list[str]]:
    """Returns `(outcome, reasons)`. `reasons` explains exactly which
    predeclared condition(s) drove the classification -- never left implicit.
    """
    reasons: list[str] = []

    if not inputs.accounting_passed:
        reasons.append("accounting identity failed")
    if not inputs.routing_passed:
        reasons.append("routing or leakage failure")
    if not inputs.reproducibility_passed:
        reasons.append("frozen artifacts could not reproduce (non-deterministic output)")
    if not inputs.frozen_artifacts_reproducible:
        reasons.append("frozen artifact hashes did not match the pre-evaluation manifest")
    if inputs.critical_input_or_schema_issue:
        reasons.append("major schema incompatibility invalidated required inputs")
    if inputs.any_required_component_gate_severely_failed:
        reasons.append("a required production component's predictions were invalid or uncomputable")
    if not inputs.schema_and_public_contract_passed:
        reasons.append("public ranking or qualification contract was violated")

    if reasons:
        return OUTCOME_FINAL_EVALUATION_FAILED, reasons

    if inputs.has_provisional_or_subgroup_limitations:
        return OUTCOME_VALIDATED_WITH_DOCUMENTED_LIMITATIONS, [
            "one or more provisional components, subgroup limitations, or meaningful "
            "distribution shifts remain, with no critical pipeline or validity failure"
        ]

    return OUTCOME_VALIDATED_AS_FROZEN, ["all checks passed; no material failure or critical issue"]


# ---------------------------------------------------------------------------
# Public-score evaluation summary
# ---------------------------------------------------------------------------


def summarize_public_score_evaluation(
    public_score_table: pd.DataFrame,
    favorable_leaderboard: pd.DataFrame,
    unfavorable_leaderboard: pd.DataFrame,
    review_tables: dict[str, Any],
) -> dict[str, Any]:
    from mlb_luck_score.scoring.public_score_schema import (
        INTERVAL_ABOVE_ZERO,
        INTERVAL_BELOW_ZERO,
        INTERVAL_OVERLAPS_ZERO,
    )
    from mlb_luck_score.scoring.qualification import STATUS_QUALIFIED

    qualified = public_score_table[public_score_table["qualification_status"] == STATUS_QUALIFIED]
    ranked = (
        public_score_table["official_rank_favorable"].notna()
        | public_score_table["official_rank_unfavorable"].notna()
    )

    width = qualified["upper_95_interval"] - qualified["lower_95_interval"]
    interval_counts = qualified["interval_interpretation"].value_counts().to_dict()

    return {
        "n_batter_seasons": int(len(public_score_table)),
        "n_qualified": int(len(qualified)),
        "share_qualified": float(len(qualified) / len(public_score_table))
        if len(public_score_table)
        else None,
        "min_eligible_bbe_among_ranked": (
            int(public_score_table.loc[ranked, "eligible_batted_balls"].min())
            if ranked.any()
            else None
        ),
        "favorable_leaderboard_top_25": favorable_leaderboard.head(25).to_dict(orient="records"),
        "unfavorable_leaderboard_top_25": unfavorable_leaderboard.head(25).to_dict(
            orient="records"
        ),
        "interval_width_distribution": {
            "mean": float(width.mean()) if len(width) else None,
            "median": float(width.median()) if len(width) else None,
            "min": float(width.min()) if len(width) else None,
            "max": float(width.max()) if len(width) else None,
        },
        "qualified_interval_interpretation_shares": {
            "entirely_above_zero": int(interval_counts.get(INTERVAL_ABOVE_ZERO, 0)),
            "overlaps_zero": int(interval_counts.get(INTERVAL_OVERLAPS_ZERO, 0)),
            "entirely_below_zero": int(interval_counts.get(INTERVAL_BELOW_ZERO, 0)),
        },
        "provisional_component_share_distribution": {
            "mean": float(public_score_table["share_of_value_from_provisional_components"].mean()),
            "median": float(
                public_score_table["share_of_value_from_provisional_components"].median()
            ),
        },
        "defense_unavailable_play_count_total": int(
            public_score_table["defense_unavailable_play_count"].sum()
        ),
        "advancement_unavailable_play_count_total": int(
            public_score_table["advancement_unavailable_play_count"].sum()
        ),
        "provisional_exclusion_rank_sensitivity": {
            "note": "DIAGNOSTIC ONLY -- never alters the official ranking (task instruction).",
            "comparison": review_tables.get("ranking_with_vs_without_provisional_components"),
        },
    }


# ---------------------------------------------------------------------------
# Stability analyses
# ---------------------------------------------------------------------------


def compute_cross_season_correlation(
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    *,
    metric: str = "contact_luck_runs_per_100",
    require_qualified_in_both: bool = True,
) -> dict[str, Any]:
    """Correlation of `metric` between two seasons' public score tables,
    joined on `batter_id`, restricted (by default) to players meeting the
    frozen qualification rule in BOTH seasons.
    """
    from mlb_luck_score.scoring.qualification import STATUS_QUALIFIED

    left = table_a[["batter_id", "season", metric, "qualification_status"]]
    right = table_b[["batter_id", "season", metric, "qualification_status"]]
    merged = left.merge(right, on="batter_id", suffixes=("_a", "_b"))

    if require_qualified_in_both:
        merged = merged[
            (merged["qualification_status_a"] == STATUS_QUALIFIED)
            & (merged["qualification_status_b"] == STATUS_QUALIFIED)
        ]

    n = len(merged)
    if n < 2:
        return {
            "n_players_compared": n,
            "pearson_r": None,
            "spearman_r": None,
            "require_qualified_in_both": require_qualified_in_both,
        }

    col_a, col_b = f"{metric}_a", f"{metric}_b"
    return {
        "n_players_compared": n,
        "pearson_r": float(merged[col_a].corr(merged[col_b], method="pearson")),
        "spearman_r": float(merged[col_a].corr(merged[col_b], method="spearman")),
        "require_qualified_in_both": require_qualified_in_both,
    }


def summarize_2025_stability(
    artifacts_2025: SeasonAggregationArtifacts,
    public_score_table_2024: pd.DataFrame | None,
    public_score_table_2025: pd.DataFrame,
) -> dict[str, Any]:
    """Split-half/odd-even reliability WITHIN 2025 (reusing `mlb_luck_score.
    models.evaluate_aggregation_stability`'s Version 0.11 functions
    unchanged), interval width by sample size, and 2024-vs-2025 player
    correlation among dual-qualified players.

    Low year-to-year or split-half correlation must be interpreted
    consistently with Contact Luck's retrospective purpose -- see `mlb_luck_
    score.scoring.public_labels.RETROSPECTIVE_LIMITATION`. This function
    computes the numbers; it makes no predictive-validity claim itself.
    """
    from mlb_luck_score.models.evaluate_aggregation_stability import (
        compute_interval_width_by_sample_size,
        compute_split_half_reliability,
    )

    result: dict[str, Any] = {
        "split_half_2025": {
            "calendar": compute_split_half_reliability(artifacts_2025, split="calendar"),
            "odd_even_game_pk": compute_split_half_reliability(artifacts_2025, split="odd_even"),
        },
        "interval_width_by_sample_size_2025": compute_interval_width_by_sample_size(
            artifacts_2025.player_season
        ),
        "retrospective_interpretation_note": (
            "Low split-half or year-to-year correlation is CONSISTENT with Contact Luck's "
            "retrospective purpose (describing realized outcomes, not projecting future "
            "performance) and must not be read as a predictive-validity failure."
        ),
    }

    if public_score_table_2024 is not None:
        result["cross_season_2024_vs_2025"] = compute_cross_season_correlation(
            public_score_table_2024, public_score_table_2025
        )
    else:
        result["cross_season_2024_vs_2025"] = {
            "note": "2024 public score table not supplied -- cross-season correlation skipped"
        }

    return result


# ---------------------------------------------------------------------------
# Final report assembly
# ---------------------------------------------------------------------------


def assemble_final_report(
    *,
    manifest_dict: dict[str, Any],
    system_checks: dict[str, Any],
    component_metrics: dict[str, Any],
    distribution_shift_report: dict[str, Any],
    distribution_shift_flags: list[str],
    public_score_summary: dict[str, Any],
    stability_summary: dict[str, Any],
    outcome: str,
    outcome_reasons: list[str],
    limitations: list[str],
) -> dict[str, Any]:
    return {
        "manifest": manifest_dict,
        "system_checks": system_checks,
        "component_metrics": component_metrics,
        "distribution_shift": {
            "report": distribution_shift_report,
            "flags": distribution_shift_flags,
        },
        "public_score_evaluation": public_score_summary,
        "stability": stability_summary,
        "outcome_classification": {
            "outcome": outcome,
            "reasons": outcome_reasons,
        },
        "limitations": limitations,
    }
