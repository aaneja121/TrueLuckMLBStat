"""Contact Forecast H=200: render the held-out 2026 report from its artifacts.

Language rules this renderer enforces structurally:

  - **The exposure disclosure is rendered before any result.** 2026 is
    outcome-held-out, not completely unseen, and the reader is told so first.
  - **H=200 is never described as the frozen Contact Forecast measured over a
    longer window.** It is a refit and a separate specification.
  - **Development and held-out evidence stay separate.** No number pools 2024
    and 2026.
  - **No claim about playing time.** The conditional-forecast sentence is
    rendered with the target, every time.
  - **The classification is read from the artifact, never re-derived here.**
"""

from __future__ import annotations

from typing import Any

from forecast.phase2.phase2_report import (
    CLASSIFICATION_HEADLINE,
    _correlation,
    _number,
    render_sign_convention,
)


def render_h200_report(
    *,
    authorization: dict[str, Any],
    snapshot_manifest: dict[str, Any],
    metrics: dict[str, Any],
    stability: dict[str, Any],
    shift: dict[str, Any],
    survivorship: dict[str, Any],
    disclosure_markdown: str,
) -> str:
    lines: list[str] = []
    lines += _render_header(authorization, snapshot_manifest, metrics)
    lines += [*disclosure_markdown.splitlines(), ""]
    lines += render_sign_convention(metrics["contact_forecast_vs_shrunk_deserved"])
    lines += _render_result(metrics)
    lines += _render_benchmarks(metrics)
    lines += _render_stability(stability)
    lines += _render_survivorship(survivorship)
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
    return [
        "# Contact Forecast H=200 -- held-out 2026 evaluation",
        "",
        "## What this is, and what it is not",
        "",
        "This is the single, prespecified evaluation of the FROZEN H=200 forecast",
        f"specification (`{frozen['spec_version']}`) against one pinned 2026 snapshot.",
        "",
        "It is **not** the frozen H=100 Contact Forecast measured over a longer window.",
        "The H=200 model is a REFIT with its own coefficients, intercept and",
        "standardization; the two results are never pooled, and the frozen H=100 Phase 2",
        "result is unchanged by anything here.",
        "",
        "H=200 was **not** prespecified in R1, whose frozen window grid used horizons 50",
        "and 100. The historical 100 -> 200 numbers were exploratory scoping, which is",
        "why the specification was frozen on development data before this run.",
        "",
        "Nothing was tuned, refit, trimmed, recalibrated or deployed after the result",
        "below was seen.",
        "",
        "## The pinned snapshot",
        "",
        f"- Snapshot label: **{snapshot['snapshot_label']}**",
        f"- Data through: **{snapshot['data_through_date']}**",
        "- Single snapshot used for the entire evaluation: "
        f"{snapshot_manifest['single_snapshot_used_for_entire_evaluation']}",
        f"- Refreshed during analysis: {snapshot_manifest['refreshed_during_analysis']}",
        f"- Resolved 2026 events read: {counts['n_resolved_events_2026']:,}",
        f"- Hitters reaching 100 resolved BBE: {counts['n_hitters_reaching_100_resolved_bbe']:,}",
        f"- Hitters reaching 300 resolved BBE: {counts['n_hitters_reaching_300_resolved_bbe']:,}",
        "- **Completed 100 -> 200 windows evaluated: "
        f"{counts['n_evaluable_completed_100_to_200_windows']:,}**",
        f"- Forecasts still pending an outcome: {counts['n_forecasts_with_pending_outcomes']:,}",
        "",
        "## The frozen specification, loaded not restated",
        "",
        f"- Model: `{frozen['model']}` (refit on the 200-BBE target)",
        f"- Alpha: {frozen['alpha']} -- frozen; never retuned on 2026",
        f"- Features: {len(frozen['features'])}, the frozen Model D list unchanged",
        f"- Trained on seasons: {frozen['trained_on_seasons']} (n = {frozen['n_train']})",
        f"- Target: {frozen['target_definition']}",
        f"- Primary benchmark: `{frozen['primary_benchmark']}`",
        f"- Primary metric: {frozen['primary_metric']}",
        "- 2025: sealed and never read.",
        "",
        "The re-derived fit reproduced the sealed coefficients, intercept and scaler",
        "exactly; a mismatch would have stopped the run before any outcome was opened.",
        "",
    ]


def _render_result(metrics: dict[str, Any]) -> list[str]:
    comparison = metrics["contact_forecast_vs_shrunk_deserved"]
    classification = metrics["prespecified_classification"]
    ci_low, ci_high = comparison["ci_delta_mae"]
    headline = CLASSIFICATION_HEADLINE[classification["classification"]]
    share = comparison["share_replicates_favouring_contact_forecast"]
    return [
        "## Result",
        "",
        f"### {headline}",
        "",
        f"- Completed 100 -> 200 windows: **{metrics['n_completed_windows']:,}** "
        f"({metrics['n_unique_hitters']:,} unique hitters)",
        f"- Contact Forecast MAE: **{_number(comparison['mae_contact_forecast'])}**",
        f"- Shrunk-deserved MAE: **{_number(comparison['mae_shrunk_deserved'])}**",
        f"- delta_MAE: **{_number(comparison['delta_mae'])}** "
        f"({_number(comparison['pct_change_mae'], 2)}% change)",
        f"- Paired 95% bootstrap CI on delta_MAE: **[{_number(ci_low)}, {_number(ci_high)}]**",
        f"- CI crosses zero: **{comparison['ci_crosses_zero']}**",
        f"- Bootstrap replicates favouring the Contact Forecast: **{_number(share, 4)}**",
        f"- Contact Forecast RMSE: {_number(comparison['rmse_contact_forecast'])} "
        f"vs shrunk-deserved {_number(comparison['rmse_shrunk_deserved'])} "
        f"(delta_RMSE {_number(comparison['delta_rmse'])})",
        "",
        f"**Classification rule (prespecified):** {classification['condition']}",
        "",
        f"**Meaning:** {classification['meaning']}",
        "",
        "The deciding metric is MAE. Secondary metrics are reported below and may not",
        "reclassify this result.",
        "",
        f"_{metrics['clustering_note']}_",
        "",
        "This forecast is conditional on the hitter subsequently accumulating 200",
        "additional resolved eligible batted balls. It does not predict playing time,",
        "injury, roster survival or demotion.",
        "",
    ]


def _render_benchmarks(metrics: dict[str, Any]) -> list[str]:
    rows = [
        "## Every predictor on the held-out 2026 windows",
        "",
        "| Predictor | MAE | RMSE | R-squared | Pearson | Spearman |",
        "|---|---|---|---|---|---|",
    ]
    frozen = metrics["provenance"]["frozen_contract"]
    ordered = ["contact_forecast", *frozen["benchmarks"]]
    by_predictor = metrics["by_predictor"]
    for name in [*ordered, *sorted(set(by_predictor) - set(ordered))]:
        record = by_predictor[name]
        rows.append(
            f"| `{name}` | {_number(record['mae'])} | {_number(record['rmse'])} | "
            f"{_number(record['r_squared'])} | "
            f"{_correlation(record['pearson'], record['pearson_status'])} | "
            f"{_correlation(record['spearman'], record['spearman_status'])} |"
        )
    rows.append("")
    return rows


def _render_stability(stability: dict[str, Any]) -> list[str]:
    sensitivity = stability["outlier_sensitivity"]
    return [
        "## Stability: is any advantage broad, or carried by a few hitters?",
        "",
        f"`{stability['quantity']}`; {stability['sign_meaning']}.",
        "",
        f"- Hitters improved by the Contact Forecast: "
        f"**{_number(stability['fraction_of_hitters_improved'], 4)}** "
        f"of {stability['n_hitters']:,}",
        f"- Median per-hitter error improvement: {_number(stability['median'])}",
        f"- Quartiles: Q1 {_number(stability['q1'])}, Q3 {_number(stability['q3'])}",
        f"- Deciles: p10 {_number(stability['p10'])}, p90 {_number(stability['p90'])}",
        f"- Mean: {_number(stability['mean'])}",
        "- delta_MAE excluding the 5 largest beneficiaries: "
        f"**{_number(sensitivity['delta_mae_excluding_5_most_improved'])}**",
        "- delta_MAE excluding the 10 largest beneficiaries: "
        f"**{_number(sensitivity['delta_mae_excluding_10_most_improved'])}**",
        "",
        f"_{sensitivity['status']}_",
        "",
    ]


def _render_survivorship(survivorship: dict[str, Any]) -> list[str]:
    development = survivorship["development_2022_2024_pooled"]
    prospective = survivorship["prospective_2026"]
    lines = [
        "## Survivorship: development versus the held-out season",
        "",
        "| | Reached 100 BBE | Reached 300 BBE | Completion rate |",
        "|---|---|---|---|",
        f"| Development 2022-2024 pooled | {development['n_reaching_100_resolved_bbe']:,} | "
        f"{development['n_subsequently_reaching_300_total_resolved_bbe']:,} | "
        f"{_number(development['completion_rate'], 4)} |",
        f"| Prospective 2026 ({prospective['snapshot_data_through']}) | "
        f"{prospective['n_reaching_100_resolved_bbe']:,} | "
        f"{prospective['n_subsequently_reaching_300_total_resolved_bbe']:,} | "
        f"{_number(prospective['completion_rate'], 4)} |",
        "",
        "Completion-rate gap (2026 minus development): "
        f"**{_number(survivorship['completion_rate_gap_2026_minus_development'], 4)}**",
        "",
        "Largest cutoff-side differences between completers and non-completers, in",
        "standardized units:",
        "",
        "| Feature | Development | 2026 |",
        "|---|---|---|",
    ]
    for row in survivorship["cutoff_side_differences_side_by_side"][:8]:
        lines.append(
            f"| `{row['feature']}` | "
            f"{_number(row['development_standardized_difference'])} | "
            f"{_number(row['prospective_2026_standardized_difference'])} |"
        )
    lines += [
        "",
        survivorship["generalization_note"],
        "",
        f"_{survivorship['status']}_",
        "",
    ]
    return lines


def _render_shift(shift: dict[str, Any]) -> list[str]:
    lines = [
        "## 2026 distribution shift against permitted development data",
        "",
        f"Material-shift threshold: |standardized mean shift| >= "
        f"{shift['material_shift_threshold_standardized']}.",
        "",
        f"- Features with a material shift: **{shift['n_features_with_material_shift']}** "
        f"of {shift['n_features']}",
    ]
    if shift["features_with_material_shift"]:
        lines.append("")
        lines += [
            "| Feature | Development mean | 2026 mean | Standardized shift |",
            "|---|---|---|---|",
        ]
        flagged = set(shift["features_with_material_shift"])
        for row in shift["by_feature"]:
            if row["feature"] in flagged:
                lines.append(
                    f"| `{row['feature']}` | {_number(row['development_mean'])} | "
                    f"{_number(row['prospective_2026_mean'])} | "
                    f"{_number(row['standardized_mean_shift'])} |"
                )
    else:
        lines.append("- No feature crossed the threshold.")
    lines += ["", f"_{shift['status']}_", ""]
    return lines


def _render_bias(metrics: dict[str, Any]) -> list[str]:
    bias = metrics["forecast_bias_diagnostics"]
    calibration = bias["calibration_regression"]
    return [
        "## Forecast bias and calibration",
        "",
        f"- Mean prediction: {_number(bias['mean_prediction'])}",
        f"- Mean realized target: {_number(bias['mean_realized_target'])}",
        f"- Mean prediction error: **{_number(bias['mean_prediction_error'])}**",
        f"- Median prediction error: {_number(bias['median_prediction_error'])}",
        f"- SD of predictions: {_number(bias['sd_prediction'])} vs SD of realized "
        f"{_number(bias['sd_realized_target'])}",
        f"- Calibration `{calibration['model']}`: slope "
        f"**{_number(calibration['slope'])}**, intercept {_number(calibration['intercept'])} "
        "(perfect would be slope 1, intercept 0)",
        "",
        f"_{bias['status']}_",
        "",
    ]


def _render_pending(metrics: dict[str, Any], snapshot_manifest: dict[str, Any]) -> list[str]:
    pending = metrics["pending_forecasts"]
    return [
        "## Forecasts still pending an outcome",
        "",
        f"- Pending: **{pending['n_pending']:,}**",
        "",
        pending["note"],
        "",
    ]


def _render_separation(authorization: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    provenance = metrics["provenance"]
    return [
        "## Provenance and separation",
        "",
        f"- Authorization record: `{provenance['authorization_sha256']}`",
        f"- H=200 specification manifest: `{provenance['h200_spec_manifest_sha256']}`",
        "- Upstream frozen stages: "
        + ", ".join(
            f"`{stage}` {provenance['frozen_stage_manifests'][stage][:16]}"
            for stage in ("r1", "ridge", "hgb")
            if stage in provenance["frozen_stage_manifests"]
        ),
        f"- Snapshot: {provenance['snapshot_label']} "
        f"(data through {provenance['snapshot_data_through']})",
        "- 2025 read: never.",
        "- 2026 used in any fit: never.",
        "- Frozen H=100 Phase 2 result modified: no.",
        "- Deployed: no.",
        "",
        "After the result above was computed, nothing was retuned, refit, trimmed,",
        "reclassified or removed. The result stands exactly as computed.",
        "",
    ]
