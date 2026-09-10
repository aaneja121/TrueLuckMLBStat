"""Render the end-of-season resolution report from its artifacts.

Language rules this renderer enforces structurally:

  - **The second-look warning comes before any number.** A reader must know
    the error rate is above nominal before seeing an interval.
  - **Only the incremental cohort carries a classification.** The full-season
    cohort renders its estimate under a NOT INDEPENDENT banner and has no
    class to quote.
  - **The sign convention is derived from the comparison**, never from the
    shared R1 constant, which names a different pair.
  - **No claim about playing time**, and the conditional-forecast sentence is
    rendered with every cohort.
"""

from __future__ import annotations

from typing import Any

from forecast.phase2.phase2_report import CLASSIFICATION_HEADLINE, _number, render_sign_convention


def render_resolution_report(
    *,
    authorization: dict[str, Any],
    metrics: dict[str, Any],
    survivorship: dict[str, Any],
    distribution_shift: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    lines += _render_header(authorization, metrics)
    lines += _render_second_look_warning(authorization)
    lines += _render_sign_convention_for(metrics)
    lines += _render_primary(metrics)
    lines += _render_full_season(metrics)
    lines += _render_never_completed(metrics)
    lines += _render_survivorship(survivorship)
    # Optional so an older caller still renders, but the runner always passes
    # it: a required analysis that is computed and never shown is an analysis
    # nobody reads.
    if distribution_shift is not None:
        lines += _render_distribution_shift(distribution_shift)
    lines += _render_provenance(authorization, metrics)
    return "\n".join(lines) + "\n"


def _render_distribution_shift(shift: dict[str, Any]) -> list[str]:
    """Required analysis 3, per cohort.

    Reported as a limit on generalization, never as a finding about the
    model: a shifted feature distribution says the cohorts differ, not that
    the forecast is wrong, and nothing may be adjusted on the strength of it.
    """
    lines = ["## Distribution-shift diagnostics", "", f"**{shift['status']}.**", ""]
    for cohort, record in shift["cohorts"].items():
        label = cohort.replace("_", "-")
        if not record.get("evaluated"):
            lines += [f"- **{label}** -- no windows to compare.", ""]
            continue
        features = record["features"]
        flagged = features["features_with_material_shift"]
        lines += [
            f"- **{label}** ({record['n_windows']} windows): "
            f"{features['n_features_with_material_shift']} of {features['n_features']} "
            f"frozen model features shift by at least "
            f"{features['material_shift_threshold_standardized']} standardized units"
            + (f" -- {', '.join(flagged)}." if flagged else "."),
        ]
    lines += ["", shift["generalization_note"], ""]
    return lines


def _render_sign_convention_for(metrics: dict[str, Any]) -> list[str]:
    """Derive the convention from whichever comparison this report prints.

    Normally the primary cohort's. If no hitter resolved late there is no
    primary comparison to derive from, so it falls back to the cohort that IS
    printed -- and renders nothing at all if neither was evaluated, rather than
    inventing a convention for a comparison the report never shows.
    """
    for cohort in ("incremental", "full_season"):
        record = metrics.get(cohort, {})
        if record.get("evaluated") and "contact_forecast_vs_shrunk_deserved" in record:
            return render_sign_convention(record["contact_forecast_vs_shrunk_deserved"])
    return []


def _render_header(authorization: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    gate = authorization["gate"]
    return [
        "# Contact Forecast -- end-of-season 2026 resolution pass",
        "",
        f"Horizon {metrics['cutoff']} -> {metrics['horizon']}. "
        f"Snapshot {gate['snapshot_reaches_season_end']['snapshot_label']}, data through "
        f"{gate['snapshot_reaches_season_end']['snapshot_data_through']}.",
        "",
        f"The 2026 regular season ended "
        f"{gate['season_has_ended']['verified_season_end_date']}. This pass attached "
        "outcomes to predictions sealed at the 2026-09-01 first look. No forecast was "
        "regenerated, nothing was refit, no hitter was removed, and nothing is deployed.",
        "",
        f"Authorized by {authorization['authorized_by']} at {authorization['authorized_at_utc']}.",
        "",
    ]


def _render_second_look_warning(authorization: dict[str, Any]) -> list[str]:
    return [
        "## Read this before any number below",
        "",
        f"**{authorization['acknowledged']}**",
        "",
        "This is the FINAL look at 2026 for these predictions. There is no third pass.",
        "",
    ]


def _render_primary(metrics: dict[str, Any]) -> list[str]:
    cohort = metrics["incremental"]
    if not cohort.get("evaluated"):
        return [
            "## PRIMARY -- incremental cohort",
            "",
            "**No hitter pending at the first look completed the horizon by season's end.**",
            "There is no primary result. The pass does not fall back to the full-season",
            "cohort, and no classification is assigned.",
            "",
        ]
    comparison = cohort["contact_forecast_vs_shrunk_deserved"]
    classification = cohort["prespecified_classification"]
    low, high = comparison["ci_delta_mae"]
    stability = cohort["stability"]
    sensitivity = stability["outlier_sensitivity"]
    return [
        "## PRIMARY -- incremental cohort (confirmatory)",
        "",
        f"### {CLASSIFICATION_HEADLINE[classification['classification']]}",
        "",
        "Hitters pending at the 2026-09-01 first look who completed the horizon by",
        "season's end. Their outcomes had never been examined before this pass.",
        "",
        f"- Windows: **{cohort['n_windows']:,}** ({cohort['n_unique_hitters']:,} unique hitters)",
        f"- Contact Forecast MAE: **{_number(comparison['mae_contact_forecast'])}**",
        f"- Shrunk-deserved MAE: **{_number(comparison['mae_shrunk_deserved'])}**",
        f"- delta_MAE: **{_number(comparison['delta_mae'])}** "
        f"({_number(comparison['pct_change_mae'], 2)}% change)",
        f"- Paired 95% bootstrap CI: **[{_number(low)}, {_number(high)}]** "
        f"(crosses zero: {comparison['ci_crosses_zero']})",
        "- Replicates favouring the Contact Forecast: "
        f"**{_number(comparison['share_replicates_favouring_contact_forecast'])}**",
        f"- delta_RMSE: {_number(comparison['delta_rmse'])}",
        "",
        f"**Prespecified condition:** {classification['condition']}",
        "",
        f"**Meaning:** {classification['meaning']}",
        "",
        "MAE decides. Secondary metrics are reported and may not reclassify this result.",
        "",
        "### Is the effect broad, or carried by a few hitters?",
        "",
        f"- Hitters improved: **{_number(stability['fraction_of_hitters_improved'])}** "
        f"of {stability['n_hitters']:,}",
        f"- Median per-hitter improvement: {_number(stability['median'])} "
        f"(Q1 {_number(stability['q1'])}, Q3 {_number(stability['q3'])})",
        "- delta_MAE excluding the 5 largest beneficiaries: "
        f"**{_number(sensitivity['delta_mae_excluding_5_most_improved'])}**",
        "- delta_MAE excluding the 10 largest beneficiaries: "
        f"**{_number(sensitivity['delta_mae_excluding_10_most_improved'])}**",
        "",
        f"_{sensitivity['status']}_",
        "",
        "This forecast is conditional on the hitter accumulating the horizon within the",
        "season. It does not predict playing time, injury, roster survival or demotion.",
        "",
    ]


def _render_full_season(metrics: dict[str, Any]) -> list[str]:
    cohort = metrics["full_season"]
    if not cohort.get("evaluated"):
        return []
    comparison = cohort["contact_forecast_vs_shrunk_deserved"]
    low, high = comparison["ci_delta_mae"]
    return [
        "## SECONDARY -- full-season cohort",
        "",
        f"> **{cohort['mandatory_label']}**",
        "",
        f"- Windows: **{cohort['n_windows']:,}**",
        f"- Contact Forecast MAE: {_number(comparison['mae_contact_forecast'])} vs "
        f"shrunk-deserved {_number(comparison['mae_shrunk_deserved'])}",
        f"- delta_MAE: **{_number(comparison['delta_mae'])}**, interval "
        f"[{_number(low)}, {_number(high)}]",
        "",
        "**No classification is assigned to this cohort.** It reuses every window",
        "already observed at the first look, so it is the most precise estimate of the",
        "effect and the least valid significance test of it. It may not upgrade the",
        "conclusion, and if it disagrees with the primary cohort, the primary stands.",
        "",
    ]


def _render_never_completed(metrics: dict[str, Any]) -> list[str]:
    cohort = metrics["never_completed"]
    return [
        "## Reported, never evaluated -- hitters who did not reach the horizon",
        "",
        f"- Hitters: **{cohort['n_hitters']:,}**",
        "",
        cohort["note"],
        "",
    ]


def _render_survivorship(survivorship: dict[str, Any]) -> list[str]:
    completion = survivorship["end_of_season_completion"]
    lines = [
        "## Survivorship",
        "",
        survivorship["why_this_matters"],
        "",
        f"- Hitters with a forecast: {completion['n_hitters_with_a_forecast']:,}",
        f"- Completed by season's end: {completion['n_completed_by_season_end']:,} "
        f"(rate {_number(completion['completion_rate'])})",
        "",
        "Largest cutoff-side differences, incremental versus first-look completers,",
        "in standardized units:",
        "",
        "| Feature | incremental vs first-look | incremental vs never-completed |",
        "|---|---|---|",
    ]
    for row in survivorship["largest_incremental_vs_first_look"]:
        lines.append(
            f"| `{row['feature']}` | "
            f"{_number(row['incremental_vs_first_look_standardized'])} | "
            f"{_number(row['incremental_vs_never_completed_standardized'])} |"
        )
    lines += [
        "",
        survivorship["generalization_note"],
        "",
        f"_{survivorship['status']}_",
        "",
    ]
    return lines


def _render_provenance(authorization: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    provenance = metrics["provenance"]
    history = metrics["cohort_classification_history"]
    return [
        "## Provenance and separation",
        "",
        f"- Authorization record: `{provenance['authorization_sha256']}`",
        "- Freeze chain: "
        + ", ".join(f"`{k}` {v[:16]}" for k, v in sorted(provenance["freeze_chain"].items())),
        "- Sealed prediction ledgers, re-hashed and unchanged: "
        + ", ".join(f"`{k}`" for k in sorted(provenance["sealed_ledger_sha256"])),
        f"- Snapshot: {provenance['snapshot_label']} "
        f"(data through {provenance['snapshot_data_through']})",
        "- Predictions regenerated: no. Refit on 2026: no. Hitters removed: none.",
        "- 2025 read: never. First-look results modified: no. Deployed: no.",
        "",
        "### Specification history",
        "",
        f"The frozen specification once flagged the full-season cohort for the four-way "
        f"classification while also stating it applied to "
        f"**{history['resolved_as']}**. That contradiction was resolved by "
        f"{history['resolved_by']}, before any outcome was opened "
        f"({history['amended_before_any_outcome_was_opened']}). This pass reads each "
        f"cohort's flag from the frozen specification rather than deciding at run time.",
        "",
        f"The full-season cohort receives {history['full_season_receives']}.",
        "",
    ]
