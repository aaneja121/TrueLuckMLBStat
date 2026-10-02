"""Contact Forecast Phase 2: render the held-out 2026 report from its artifacts.

Language rules this renderer enforces structurally:

  - **Development and held-out evidence are separate sections.** No number
    pools 2024 and 2026, and there is no combined performance figure anywhere
    in the template.
  - **2026 is never described as training data.** It appears only under
    "held-out evidence", and the training seasons are named explicitly.
  - **The published metric is not claimed to be prospectively validated.** The
    contact-stage disclaimer is rendered before any result.
  - **No claim about playing time.** The conditional-forecast sentence is
    rendered with the target, every time.

Signs are printed exactly as stored under a restatement of the frozen
convention, and the prespecified classification is read from the artifact
rather than re-derived here.
"""

from __future__ import annotations

import re
from typing import Any

CLASSIFICATION_HEADLINE: dict[str, str] = {
    "established_incremental_success": "ESTABLISHED INCREMENTAL SUCCESS",
    "promising_but_inconclusive": "PROMISING BUT INCONCLUSIVE",
    "no_evidence_of_incremental_improvement": "NO EVIDENCE OF INCREMENTAL IMPROVEMENT",
    "evidence_against_incremental_value": "EVIDENCE AGAINST INCREMENTAL VALUE",
}


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _correlation(value: Any, status: str) -> str:
    if value is None:
        return f"undefined ({status})"
    return _number(value, 4)


class Phase2ReportError(ValueError):
    """Raised when a report cannot derive its own displayed sign convention."""


#: `MAE(challenger) - MAE(reference)`, as the comparison record states it.
_DEFINITION_PATTERN = re.compile(
    r"^\s*MAE\(\s*(?P<challenger>[^()\s]+)\s*\)\s*-\s*MAE\(\s*(?P<reference>[^()\s]+)\s*\)\s*$"
)


def parse_comparison_definition(comparison: dict[str, Any]) -> tuple[str, str]:
    """Read the challenger and reference out of THIS comparison's definition.

    Raises:
        Phase2ReportError: If the definition is absent or not of the form
            `MAE(challenger) - MAE(reference)`.
    """
    definition = comparison.get("definition")
    if not isinstance(definition, str):
        raise Phase2ReportError(
            "The comparison record carries no `definition`; a report may not display a "
            "sign convention it cannot derive from the comparison actually performed."
        )
    match = _DEFINITION_PATTERN.match(definition)
    if match is None:
        raise Phase2ReportError(
            f"Comparison definition {definition!r} is not of the form "
            "'MAE(challenger) - MAE(reference)', so the displayed sign convention "
            "cannot be derived from it."
        )
    return match.group("challenger"), match.group("reference")


def render_sign_convention(comparison: dict[str, Any]) -> list[str]:
    """The sign-convention block, derived from the comparison being reported.

    Deliberately NOT rendered from `forecast.metrics.DELTA_SIGN_CONVENTION`.
    That shared R1 constant describes a different pair -- shrunk deserved
    measured against shrunk realized -- so displaying it here would name the
    wrong challenger and reference. The direction of the subtraction is the
    same in both; only the operands differ, which is exactly what a reader
    needs to be told correctly.

    The names are parsed out of the comparison's own `definition`, so a report
    cannot display a convention that disagrees with the delta it prints.
    """
    challenger, reference = parse_comparison_definition(comparison)
    return [
        "## Sign convention (derived from this comparison, never reversed)",
        "",
        "```",
        f"delta_MAE = MAE({challenger}) - MAE({reference})",
        f"negative -> {challenger} better",
        "zero     -> tied",
        f"positive -> {reference} better",
        "```",
        "",
    ]


def render_phase2_report(
    *,
    authorization: dict[str, Any],
    snapshot_manifest: dict[str, Any],
    metrics: dict[str, Any],
    stability: dict[str, Any],
    shift: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines += _render_header(authorization, snapshot_manifest, metrics)
    lines += _render_result(metrics)
    lines += _render_benchmarks(metrics)
    lines += _render_stability(stability)
    lines += _render_shift(shift)
    lines += _render_bias(metrics)
    lines += _render_pending(metrics, snapshot_manifest)
    lines += _render_separation(authorization, metrics)
    return "\n".join(lines) + "\n"


def _render_header(
    authorization: dict[str, Any],
    snapshot_manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> list[str]:
    snapshot = snapshot_manifest["pinned_snapshot"]
    counts = snapshot_manifest["counts"]
    frozen = metrics["provenance"]["frozen_contract"]
    auth = authorization["authorization"]
    return [
        "# Contact Forecast Phase 2 -- held-out 2026 evaluation",
        "",
        "## What this is, and what it is not",
        "",
        "This is the FIRST held-out evaluation of the frozen Contact Forecast. The "
        "specification was sealed before 2026 was opened and nothing in this report "
        "changed it.",
        "",
        "**The forecast research measures the CONTACT STAGE (Rc - E0), not the "
        "published full-telescoping quantity (Rf - E0) that contactluck.com publishes. "
        "Nothing here prospectively validates the published metric.**",
        "",
        authorization["conditional_forecast"],
        "",
        "## Authorization",
        "",
        f"- Scope: {auth['scope']}.",
        f"- Seasons authorized for evaluation: {auth['evaluation_seasons_authorized']}. "
        f"Sealed and still forbidden: {auth['sealed_seasons_still_forbidden']}.",
        f"- Permitted training seasons: {auth['permitted_training_seasons']}.",
        "- Not authorized: " + "; ".join(auth["not_authorized"]) + ".",
        f"- Authorization record SHA-256: `{authorization['record_sha256']}`.",
        "",
        "Freeze chain verified with zero drift before 2026 was opened: "
        + ", ".join(
            f"{stage} `{digest[:16]}`"
            for stage, digest in sorted(
                authorization["freeze_chain_verification"]["manifest_sha256"].items()
            )
        )
        + ".",
        "",
        "## The frozen model, and why it was chosen",
        "",
        f"- Model: **{frozen['model']}** (full-contact-profile ridge), alpha "
        f"{frozen['alpha']}, loaded from the ridge freeze rather than restated.",
        f"- Best frozen 2024 development MAE: "
        f"{_number(frozen['model_choice_reasons']['best_frozen_2024_mae'])}, better than "
        f"shrunk deserved at "
        f"{_number(frozen['model_choice_reasons']['better_than_shrunk_deserved'])}.",
        "- HGB did not outperform ridge; its point estimate favoured ridge "
        f"({_number(frozen['model_choice_reasons']['hgb_vs_ridge_point_estimate_favoured_ridge'])}), "
        "so the prespecified nonlinear rule retained ridge.",
        "",
        "## The pinned 2026 snapshot",
        "",
        f"- Snapshot: `{snapshot['snapshot_label']}`, **data through "
        f"{snapshot['data_through_date']}**.",
        f"- Integrity: all {len(snapshot['integrity_hashes_verified'])} recorded file "
        "hashes verified before use. One snapshot for the entire evaluation; not "
        "refreshed during the analysis, not regenerated, not deployed.",
        f"- 2026 events: {counts['n_events_2026']} ({counts['n_resolved_events_2026']} resolved).",
        f"- Hitters reaching 100 resolved BBE: "
        f"{counts['n_hitters_reaching_100_resolved_bbe']}; reaching 200: "
        f"{counts['n_hitters_reaching_200_resolved_bbe']}.",
        f"- **Evaluable completed 100 -> 100 windows: "
        f"{counts['n_evaluable_completed_100_to_100_windows']}.** Forecasts still "
        f"pending an outcome: {counts['n_forecasts_with_pending_outcomes']}.",
        "",
        "Contact-stage values were derived read-only from the snapshot's own frozen "
        "contact-model probabilities. No production scoring code was modified and no "
        "model was retrained.",
        "",
        *render_sign_convention(metrics["contact_forecast_vs_shrunk_deserved"]),
    ]


def _render_result(metrics: dict[str, Any]) -> list[str]:
    comparison = metrics["contact_forecast_vs_shrunk_deserved"]
    classification = metrics["prespecified_classification"]
    headline = CLASSIFICATION_HEADLINE[classification["classification"]]
    return [
        "## HELD-OUT 2026 RESULT",
        "",
        f"### {headline}",
        "",
        f"Prespecified condition: {classification['condition']}.",
        "",
        classification["meaning"],
        "",
        f"Evaluated on {metrics['n_completed_windows']} completed 100 -> 100 windows "
        f"from {metrics['n_unique_hitters']} hitters.",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| MAE (Contact Forecast) | {_number(comparison['mae_contact_forecast'])} |",
        f"| MAE (shrunk deserved) | {_number(comparison['mae_shrunk_deserved'])} |",
        f"| **delta_MAE** | **{_number(comparison['delta_mae'])}** |",
        f"| percentage MAE improvement | {_number(comparison['pct_change_mae'], 2)}% |",
        f"| 95% paired bootstrap CI (delta_MAE) | "
        f"{_number(comparison['ci_delta_mae'][0])}, {_number(comparison['ci_delta_mae'][1])}"
        + (" (crosses zero)" if comparison["ci_crosses_zero"] else " (excludes zero)")
        + " |",
        f"| RMSE (Contact Forecast) | {_number(comparison['rmse_contact_forecast'])} |",
        f"| RMSE (shrunk deserved) | {_number(comparison['rmse_shrunk_deserved'])} |",
        f"| delta_RMSE | {_number(comparison['delta_rmse'])} |",
        f"| percentage RMSE improvement | {_number(comparison['pct_change_rmse'], 2)}% |",
        f"| 95% CI (delta_RMSE) | {_number(comparison['ci_delta_rmse'][0])}, "
        f"{_number(comparison['ci_delta_rmse'][1])} |",
        f"| replicates favouring Contact Forecast | "
        f"{_number(comparison['share_replicates_favouring_contact_forecast'], 3)} |",
        f"| replicates favouring shrunk deserved | "
        f"{_number(comparison['share_replicates_favouring_shrunk_deserved'], 3)} |",
        "",
        "MAE is the deciding metric. RMSE, R-squared and the correlations are secondary "
        "and are not used to reclassify this result.",
        "",
        metrics["clustering_note"],
        "",
    ]


def _render_benchmarks(metrics: dict[str, Any]) -> list[str]:
    lines = [
        "## All frozen benchmarks on the held-out 2026 windows",
        "",
        "| predictor | MAE | RMSE | R2 | Pearson | Spearman |",
        "|---|---|---|---|---|---|",
    ]
    for name, record in sorted(metrics["by_predictor"].items()):
        lines.append(
            f"| `{name}` | "
            + _number(record["mae"])
            + " | "
            + _number(record["rmse"])
            + " | "
            + _number(record["r_squared"])
            + " | "
            + _correlation(record["pearson"], record["pearson_status"])
            + " | "
            + _correlation(record["spearman"], record["spearman_status"])
            + " |"
        )
    lines.append("")
    return lines


def _render_stability(stability: dict[str, Any]) -> list[str]:
    sensitivity = stability["outlier_sensitivity"]
    lines = [
        "## Stability: broad improvement, or a few hitters?",
        "",
        f"`{stability['quantity']}` -- {stability['sign_meaning']}.",
        "",
        "| statistic | value |",
        "|---|---|",
        f"| hitters | {stability['n_hitters']} |",
        f"| **fraction of hitters improved** | "
        f"**{_number(stability['fraction_of_hitters_improved'], 3)}** |",
        f"| median | {_number(stability['median'])} |",
        f"| p10 / Q1 / Q3 / p90 | {_number(stability['p10'])} / {_number(stability['q1'])} "
        f"/ {_number(stability['q3'])} / {_number(stability['p90'])} |",
        f"| mean | {_number(stability['mean'])} |",
        "",
        "### Five largest improvements",
        "",
        "| hitter | error improvement |",
        "|---|---|",
    ]
    for row in stability["five_largest_improvements"]:
        lines.append(
            f"| {row['batter_name'] or row['batter']} | {_number(row['error_improvement'])} |"
        )
    lines += [
        "",
        "### Five largest deteriorations",
        "",
        "| hitter | error improvement |",
        "|---|---|",
    ]
    for row in stability["five_largest_deteriorations"]:
        lines.append(
            f"| {row['batter_name'] or row['batter']} | {_number(row['error_improvement'])} |"
        )
    lines += [
        "",
        "### Outlier-removal sensitivity",
        "",
        "| variant | delta_MAE |",
        "|---|---|",
        f"| excluding the 5 most-improved hitters | "
        f"{_number(sensitivity['delta_mae_excluding_5_most_improved'])} |",
        f"| excluding the 10 most-improved hitters | "
        f"{_number(sensitivity['delta_mae_excluding_10_most_improved'])} |",
        "",
        sensitivity["status"],
        "",
    ]
    return lines


def _render_shift(shift: dict[str, Any]) -> list[str]:
    lines = [
        "## Distribution-shift diagnostics",
        "",
        f"**{shift['status']}.**",
        "",
        f"{shift['n_features_with_material_shift']} of {shift['n_features']} features "
        f"show a standardized mean shift of at least "
        f"{shift['material_shift_threshold_standardized']}: "
        + (", ".join(f"`{f}`" for f in shift["features_with_material_shift"]) or "none")
        + ".",
        "",
        "| feature | dev mean | dev SD | 2026 mean | 2026 SD | std. shift | outside dev range | 2026 missing |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(
        shift["by_feature"],
        key=lambda r: -abs(r["standardized_mean_shift"] or 0.0),
    ):
        flag = " **(material)**" if row["material_shift"] else ""
        lines.append(
            f"| `{row['feature']}`{flag} | "
            + _number(row["development_mean"], 3)
            + " | "
            + _number(row["development_sd"], 3)
            + " | "
            + _number(row["prospective_2026_mean"], 3)
            + " | "
            + _number(row["prospective_2026_sd"], 3)
            + " | "
            + _number(row["standardized_mean_shift"], 3)
            + " | "
            + _number(row["fraction_outside_development_range"], 4)
            + " | "
            + _number(row["prospective_2026_missing_fraction"], 4)
            + " |"
        )
    lines.append("")
    return lines


def _render_bias(metrics: dict[str, Any]) -> list[str]:
    bias = metrics["forecast_bias_diagnostics"]
    calibration = bias["calibration_regression"]
    return [
        "## Forecast-bias diagnostics",
        "",
        f"**{bias['status']}.**",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| mean prediction | {_number(bias['mean_prediction'])} |",
        f"| mean realized target | {_number(bias['mean_realized_target'])} |",
        f"| mean prediction error | {_number(bias['mean_prediction_error'])} |",
        f"| median prediction error | {_number(bias['median_prediction_error'])} |",
        f"| SD of predictions | {_number(bias['sd_prediction'])} |",
        f"| SD of realized targets | {_number(bias['sd_realized_target'])} |",
        f"| calibration slope | {_number(calibration['slope'])} |",
        f"| calibration intercept | {_number(calibration['intercept'])} |",
        "",
        f"`{calibration['model']}`; perfect calibration would be slope 1, intercept 0.",
        "",
    ]


def _render_pending(metrics: dict[str, Any], snapshot_manifest: dict[str, Any]) -> list[str]:
    pending = metrics["pending_forecasts"]
    return [
        "## Pending forecasts",
        "",
        f"{pending['n_pending']} hitters received a frozen forecast but have not yet "
        "completed 100 subsequent resolved eligible batted balls.",
        "",
        pending["note"],
        "",
        "They are preserved in `phase2_2026_pending_predictions.parquet`. No hitter was "
        "evaluated on a partial future window.",
        "",
    ]


def _render_separation(authorization: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    frozen = metrics["provenance"]["frozen_contract"]
    return [
        "## Development evidence versus held-out evidence",
        "",
        "These are reported separately and are never pooled into a single performance number.",
        "",
        "| evidence class | seasons | role |",
        "|---|---|---|",
        "| development | "
        + ", ".join(str(s) for s in authorization["authorization"]["permitted_training_seasons"])
        + " | model fitting, selection and every frozen conclusion |",
        f"| held-out | {metrics['evaluation_season']} | this evaluation only; "
        "never used for fitting or selection |",
        "",
        f"Frozen development verdicts, unchanged by this evaluation: R1 "
        f'"{frozen["stage_verdicts"]["r1"]}"; ridge '
        f'"{frozen["stage_verdicts"]["ridge"]}"; HGB '
        f'"{frozen["stage_verdicts"]["hgb"]}".',
        "",
        "2026 is held-out test data. It is not training data, it was not used to select "
        "anything, and no element of the specification was altered after seeing it.",
        "",
        "## Limits of this result",
        "",
        "- The forecast is conditional on a hitter accumulating 100 more resolved "
        "eligible batted balls. It does not predict whether a hitter will receive them, "
        "nor playing time, injury, roster survival or demotion.",
        "- The quantity forecast is the contact stage, not the published full-telescoping metric.",
        "- One season, one snapshot, one cutoff view. 2025 remains sealed and was never read.",
        "- Nothing here is deployed, and no prediction intervals were built.",
        "",
    ]
