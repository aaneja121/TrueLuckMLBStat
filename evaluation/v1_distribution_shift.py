"""Contact Luck v1.0: descriptive 2025-vs-development distribution-shift analysis.

Versions 0.2-0.12 are FROZEN -- this module computes NO model predictions and
changes NOTHING about the frozen pipeline. It only DESCRIBES how 2025's
input distributions differ from the 2021-2024 development reference, using
measures selected and thresholded BEFORE any 2025 result is examined:

  - **Standardized mean difference (SMD)** for continuous features
    (`(mean_2025 - mean_dev) / pooled_std`) -- the standard Cohen's-d-style
    effect-size measure. Flagged using the WIDELY-established, off-the-shelf
    convention (|SMD| >= 0.2 "small", >= 0.5 "medium", >= 0.8 "large") --
    these thresholds are textbook conventions, not tuned to this project's
    data.
  - **Population Stability Index (PSI)** for categorical/discrete shares
    (`sum((p_2025 - p_dev) * ln(p_2025 / p_dev))` over categories) -- the
    standard PSI formula used industry-wide for categorical drift
    monitoring. Flagged using the standard convention (PSI >= 0.1 "moderate
    shift", >= 0.25 "major shift").

This module FLAGS shifts; it does not correct them, and no result from it
feeds back into any frozen model, threshold, or eligibility rule.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SMD_SMALL_THRESHOLD = 0.2
SMD_MEDIUM_THRESHOLD = 0.5
SMD_LARGE_THRESHOLD = 0.8

PSI_MODERATE_THRESHOLD = 0.1
PSI_MAJOR_THRESHOLD = 0.25

#: Smoothing epsilon for PSI's log-ratio term -- avoids log(0)/division-by
#: -zero for a category present in one population but absent (proportion 0)
#: in the other, a standard PSI implementation detail, not a tuning choice.
PSI_EPSILON = 1e-4


def standardized_mean_difference(reference: pd.Series, current: pd.Series) -> float | None:
    """`(mean(current) - mean(reference)) / pooled_std`. `None` if either
    series has fewer than 2 non-null values or the pooled standard deviation
    is exactly 0 (a constant feature -- SMD is undefined, not zero).
    """
    ref = reference.dropna().to_numpy(dtype=float)
    cur = current.dropna().to_numpy(dtype=float)
    if len(ref) < 2 or len(cur) < 2:
        return None
    pooled_var = (np.var(ref, ddof=1) + np.var(cur, ddof=1)) / 2.0
    pooled_std = math.sqrt(pooled_var)
    if pooled_std == 0.0:
        return None
    return float((cur.mean() - ref.mean()) / pooled_std)


def classify_smd(smd: float | None) -> str | None:
    if smd is None:
        return None
    magnitude = abs(smd)
    if magnitude >= SMD_LARGE_THRESHOLD:
        return "large"
    if magnitude >= SMD_MEDIUM_THRESHOLD:
        return "medium"
    if magnitude >= SMD_SMALL_THRESHOLD:
        return "small"
    return "negligible"


def compute_population_stability_index(
    reference_shares: dict[str, float], current_shares: dict[str, float]
) -> float:
    """Standard PSI over the UNION of categories seen in either population."""
    categories = set(reference_shares) | set(current_shares)
    psi = 0.0
    for category in categories:
        ref_p = max(reference_shares.get(category, 0.0), PSI_EPSILON)
        cur_p = max(current_shares.get(category, 0.0), PSI_EPSILON)
        psi += (cur_p - ref_p) * math.log(cur_p / ref_p)
    return float(psi)


def classify_psi(psi: float) -> str:
    if psi >= PSI_MAJOR_THRESHOLD:
        return "major_shift"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "moderate_shift"
    return "stable"


def compare_continuous_feature(
    reference: pd.Series, current: pd.Series, *, label: str
) -> dict[str, object]:
    smd = standardized_mean_difference(reference, current)
    return {
        "label": label,
        "reference_mean": float(reference.mean()) if reference.notna().any() else None,
        "current_mean": float(current.mean()) if current.notna().any() else None,
        "reference_missing_fraction": float(reference.isna().mean()),
        "current_missing_fraction": float(current.isna().mean()),
        "standardized_mean_difference": smd,
        "smd_classification": classify_smd(smd),
    }


def compare_categorical_feature(
    reference: pd.Series, current: pd.Series, *, label: str
) -> dict[str, object]:
    reference_shares: dict[str, float] = {
        str(k): float(v) for k, v in reference.value_counts(normalize=True, dropna=True).items()
    }
    current_shares: dict[str, float] = {
        str(k): float(v) for k, v in current.value_counts(normalize=True, dropna=True).items()
    }
    psi = compute_population_stability_index(reference_shares, current_shares)
    return {
        "label": label,
        "reference_shares": reference_shares,
        "current_shares": current_shares,
        "reference_missing_fraction": float(reference.isna().mean()),
        "current_missing_fraction": float(current.isna().mean()),
        "population_stability_index": psi,
        "psi_classification": classify_psi(psi),
    }


def compare_boolean_share(
    reference: pd.Series, current: pd.Series, *, label: str
) -> dict[str, object]:
    ref_share = float(reference.astype(bool).mean()) if len(reference) else None
    cur_share = float(current.astype(bool).mean()) if len(current) else None
    return {
        "label": label,
        "reference_share": ref_share,
        "current_share": cur_share,
        "absolute_difference": (
            abs(cur_share - ref_share) if ref_share is not None and cur_share is not None else None
        ),
    }


def compare_feature_missingness(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, *, columns: tuple[str, ...]
) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for col in columns:
        ref_missing = (
            float(reference_df[col].isna().mean()) if col in reference_df.columns else None
        )
        cur_missing = float(current_df[col].isna().mean()) if col in current_df.columns else None
        result[col] = {
            "reference_missing_fraction": ref_missing,
            "current_missing_fraction": cur_missing,
        }
    return result


#: Continuous features compared via SMD -- the task's explicit list.
CONTINUOUS_SHIFT_FEATURES: tuple[str, ...] = ("launch_speed", "launch_angle", "spray_angle_approx")

#: Categorical features compared via PSI -- the task's explicit list
#: (venue/alignment column names as they exist in the cleaned/joined data).
CATEGORICAL_SHIFT_FEATURES: tuple[tuple[str, str], ...] = (
    ("bb_type", "batted_ball_type"),
    ("venue_id", "venue_mix"),
    ("of_fielding_alignment", "outfield_alignment_labels"),
    ("if_fielding_alignment", "infield_alignment_labels"),
)


def compare_development_vs_evaluation(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    reference_confidence: pd.DataFrame | None = None,
    current_confidence: pd.DataFrame | None = None,
) -> dict[str, object]:
    """The full Version 1.0 distribution-shift report: `reference_df` is
    2021-2024 development data, `current_df` is 2025. Both must already be
    through `compute_eligibility`/`add_outfield_opportunity_eligibility`/
    `add_infield_opportunity_eligibility`/`add_advancement_eligibility` (this
    function reads eligibility/outcome columns, never recomputes them).
    """
    report: dict[str, object] = {}

    report["continuous_features"] = {
        col: compare_continuous_feature(reference_df[col], current_df[col], label=col)
        for col in CONTINUOUS_SHIFT_FEATURES
        if col in reference_df.columns and col in current_df.columns
    }

    report["categorical_features"] = {
        key: compare_categorical_feature(reference_df[col], current_df[col], label=key)
        for col, key in CATEGORICAL_SHIFT_FEATURES
        if col in reference_df.columns and col in current_df.columns
    }

    report["outcome_class_prevalence"] = compare_categorical_feature(
        reference_df["outcome_class"], current_df["outcome_class"], label="outcome_class"
    )

    report["feature_missingness"] = compare_feature_missingness(
        reference_df,
        current_df,
        columns=(
            "launch_speed",
            "launch_angle",
            "hit_distance_sc",
            "spray_angle_approx",
            "sprint_speed",
            "wall_distance_in_spray_direction",
        ),
    )

    report["sprint_speed_coverage"] = compare_boolean_share(
        reference_df["sprint_speed"].notna(),
        current_df["sprint_speed"].notna(),
        label="sprint_speed_coverage",
    )

    if "outfield_opportunity_eligible" in reference_df.columns:
        report["outfield_opportunity_share"] = compare_boolean_share(
            reference_df["outfield_opportunity_eligible"],
            current_df["outfield_opportunity_eligible"],
            label="outfield_opportunity_eligible",
        )
    if "infield_opportunity_eligible" in reference_df.columns:
        report["infield_opportunity_share"] = compare_boolean_share(
            reference_df["infield_opportunity_eligible"],
            current_df["infield_opportunity_eligible"],
            label="infield_opportunity_eligible",
        )
    if "near_wall_20ft" in reference_df.columns:
        report["near_wall_share"] = compare_boolean_share(
            reference_df["near_wall_20ft"].astype(str) == "True",
            current_df["near_wall_20ft"].astype(str) == "True",
            label="near_wall_20ft",
        )
    if "events" in reference_df.columns:
        report["field_error_share"] = compare_boolean_share(
            reference_df["events"] == "field_error",
            current_df["events"] == "field_error",
            label="field_error",
        )
    if (
        "advancement_eligible" in reference_df.columns
        and "batter_final_base" in reference_df.columns
    ):
        adv_ref = reference_df.loc[
            reference_df["advancement_eligible"].astype(bool), "batter_final_base"
        ]
        adv_cur = current_df.loc[
            current_df["advancement_eligible"].astype(bool), "batter_final_base"
        ]
        report["advancement_class_prevalence"] = compare_categorical_feature(
            adv_ref, adv_cur, label="batter_final_base"
        )

    if reference_confidence is not None and current_confidence is not None:
        report["provisional_and_fallback_shares"] = _compare_confidence_shares(
            reference_confidence, current_confidence
        )

    return report


_PROVISIONAL_STATUSES = frozenset(
    {"provisional", "calibrated_with_limited_subgroup_evidence", "not_calibrated"}
)


def _compare_confidence_shares(
    reference_confidence: pd.DataFrame, current_confidence: pd.DataFrame
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for component_name in sorted(
        set(reference_confidence["component_name"]) | set(current_confidence["component_name"])
    ):
        ref_group = reference_confidence[reference_confidence["component_name"] == component_name]
        cur_group = current_confidence[current_confidence["component_name"] == component_name]
        ref_provisional = ref_group["model_status_confidence"].isin(_PROVISIONAL_STATUSES)
        cur_provisional = cur_group["model_status_confidence"].isin(_PROVISIONAL_STATUSES)
        ref_fallback = (
            ref_group["fallback_status"].astype(bool) if "fallback_status" in ref_group else None
        )
        cur_fallback = (
            cur_group["fallback_status"].astype(bool) if "fallback_status" in cur_group else None
        )
        result[component_name] = {
            "provisional_share": compare_boolean_share(
                ref_provisional, cur_provisional, label=f"{component_name}_provisional"
            ),
            "fallback_share": (
                compare_boolean_share(
                    ref_fallback, cur_fallback, label=f"{component_name}_fallback"
                )
                if ref_fallback is not None and cur_fallback is not None
                else None
            ),
        }
    return result


def summarize_flags(shift_report: dict[str, object]) -> list[str]:
    """Flat list of human-readable flags for every measure that crossed its
    threshold -- descriptive only, never used to alter the frozen pipeline.
    """
    flags: list[str] = []

    def _walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            if "smd_classification" in node and node["smd_classification"] in ("medium", "large"):
                flags.append(
                    f"{path}: SMD {node['smd_classification']} ({node.get('standardized_mean_difference')})"
                )
            if "psi_classification" in node and node["psi_classification"] != "stable":
                flags.append(
                    f"{path}: PSI {node['psi_classification']} ({node.get('population_stability_index')})"
                )
            for key, value in node.items():
                _walk(value, f"{path}.{key}" if path else str(key))

    _walk(shift_report, "")
    return flags
