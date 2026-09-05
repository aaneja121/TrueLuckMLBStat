"""Contact Forecast R1: render the report FROM the machine-readable artifacts.

Every number in the report is read out of `r1_metrics.json`,
`r1_bootstrap.json` or `r1_regression_direction.json`. Nothing is recomputed
here and nothing is reformatted in a way that changes its meaning:

  - **Signs are never reversed.** A `delta_mae` is printed exactly as stored,
    under a heading that restates the frozen convention. There is no
    "improvement" column with a flipped sign anywhere in this module.
  - **Undefined correlations print as `undefined`**, with the reason, never
    as 0.
  - **Every cell is rendered by the same loop.** A cell where realized beats
    deserved, where the interval crosses zero, or where the placebo explains
    the whole regression-direction signal is written out by the identical
    code path as a favourable one, so an unfavourable result cannot be
    dropped from the write-up without being dropped from the artifact.

`forecast.targets.assert_no_banned_metric_language` is run on the finished
text by the caller, so the report cannot quietly call a contact-stage
quantity the published metric.
"""

from __future__ import annotations

from typing import Any

from forecast.forecast_config import (
    FOCAL_PRESENTATION_CUTOFF,
    PRIMARY_HORIZON,
    PRIMARY_TARGET,
)

#: How a delta's direction label is spelled in prose. Keyed by the label
#: `forecast.metrics` assigns, so the wording cannot drift from the sign.
DIRECTION_WORDS: dict[str, str] = {
    "challenger_better": "deserved better",
    "reference_better": "realized better",
    "no_difference": "no difference",
}


def _number(value: Any, digits: int = 4) -> str:
    """Format a number, or say plainly that there isn't one."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _correlation(value: Any, status: str, *, constant_within_every_season: bool = False) -> str:
    """A correlation, or `undefined (reason)` -- never 0 standing in for one.

    A rung that is constant WITHIN each season but varies across them has a
    defined pooled correlation carrying no within-season ranking information.
    That is stated next to the number rather than left for a reader to infer.
    """
    if value is None:
        return f"undefined ({status})"
    rendered = _number(value, 4)
    if constant_within_every_season:
        return f"{rendered} (between-season variation only)"
    return rendered


def _interval(summary: dict[str, Any] | None) -> str:
    """A bootstrap interval with its zero-crossing stated in words."""
    if not summary:
        return "n/a"
    lower = _number(summary.get("ci_lower"))
    upper = _number(summary.get("ci_upper"))
    crosses = summary.get("ci_crosses_zero")
    marker = ""
    if crosses is True:
        marker = " (crosses zero)"
    elif crosses is False:
        marker = " (excludes zero)"
    return f"[{lower}, {upper}]{marker}"


def _bootstrap_for(
    bootstrap: dict[str, Any], *, target: str, challenger: str, reference: str
) -> dict[str, Any] | None:
    return bootstrap.get("comparisons", {}).get(f"{target}|{challenger}_vs_{reference}")


def _cell_bootstrap(
    bootstrap: dict[str, Any],
    *,
    target: str,
    challenger: str,
    reference: str,
    cutoff: int,
    horizon: int,
) -> dict[str, Any] | None:
    entry = _bootstrap_for(bootstrap, target=target, challenger=challenger, reference=reference)
    if not entry or entry.get("status") == "no_paired_windows":
        return None
    cells = entry.get("cells", {})
    result = cells.get(f"K{int(cutoff)}_H{int(horizon)}")
    return result if isinstance(result, dict) else None


def render_report(
    *, metrics: dict[str, Any], bootstrap: dict[str, Any], regression: dict[str, Any]
) -> str:
    """Assemble the full R1 report from the three artifacts."""
    spec = metrics["specification"]
    reference = spec["primary_comparison"]["reference"]
    lines: list[str] = []

    lines += _render_header(metrics, spec)
    lines += _render_focal_view(metrics, bootstrap, spec, reference)
    lines += _render_all_cells(metrics, bootstrap, reference)
    lines += _render_batter_balanced(metrics, reference)
    lines += _render_regression_direction(regression)
    lines += _render_direction_probability(regression)
    lines += _render_limitations(metrics)
    return "\n".join(lines) + "\n"


def _render_header(metrics: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    structure = metrics["repeated_window_structure"]
    focal = spec["focal_presentation_view"]
    lines = [
        "# Contact Forecast R1 -- descriptive results",
        "",
        f"Generated {metrics['generated_at_utc']}.",
        "",
        "## What this measures, and what it does not",
        "",
        spec["contact_stage_disclaimer"],
        "",
        spec["conditional_forecast"][str(PRIMARY_HORIZON)],
        "",
        "No forecasting model is fitted in this stage. The quantities compared are the",
        "frozen baseline ladder and its matched shrinkage; the regression-direction",
        "experiment additionally fits ordinary least squares and a logistic",
        "direction-probability mapping, both on strictly earlier seasons.",
        "",
        "## Frozen specification",
        "",
        "- **Primary family**: every cutoff at the primary horizon, none promoted above",
        f"  the others: {spec['primary_comparison'] and ''}"
        + ", ".join(str(c) for c in focal["primary_family_unchanged"])
        + " BBE observed.",
        f"- **Focal presentation view**: {focal['statement']}",
        f"  (cutoff {focal['cutoff']}, horizon {focal['horizon']}, target `{focal['target']}`).",
        f"  **{focal['status']}.** " + focal["why"],
        "  No conclusion in this report rests on the selection of this cell, and every",
        "  prespecified cell remains visible in the tables below.",
        f"- **Primary comparison**: `{spec['primary_comparison']['challenger']}` versus",
        f"  `{spec['primary_comparison']['reference']}`. {spec['primary_comparison']['why']}",
        "- **Comparison ladder**: " + ", ".join(f"`{r}`" for r in spec["baseline_ladder"]) + ".",
        f"- Every other cell is a {spec['all_other_cells_are']}.",
        "",
        "## Sign convention (frozen; never reversed for presentation)",
        "",
        "```",
        "delta_MAE  = MAE(shrunk_deserved)  - MAE(shrunk_realized)",
        "delta_RMSE = RMSE(shrunk_deserved) - RMSE(shrunk_realized)",
        "",
        "negative -> deserved is better",
        "zero     -> no difference",
        "positive -> realized is better",
        "```",
        "",
        "`pct_change` shares that sign: -3.0 means deserved reduces the error by 3.0%.",
        "",
        "## Data structure",
        "",
        f"- {structure['n_windows']} windows over "
        f"{structure['n_unique_batter_seasons']} hitter-seasons and "
        f"{structure['n_unique_batters']} unique hitters.",
        f"- Mean {_number(structure['windows_per_batter_mean'], 2)} windows per hitter, "
        f"max {structure['windows_per_batter_max']}.",
        f"- The top decile of hitters supplies "
        f"{_number(100 * structure['share_of_windows_from_top_decile_of_batters'], 1)}% of windows.",
        "",
        "Windows are **not** independent observations: one hitter contributes several",
        "overlapping cutoffs and several seasons. Every interval below therefore",
        "resamples whole hitters, globally.",
        "",
        "### Structurally unavailable rung",
        "",
    ]
    for season, availability in sorted(metrics["rung_availability_by_season"].items()):
        status = availability.get("shrunk_deserved_persistence")
        if status == "available":
            continue
        lines += [
            f"- **{season}: `shrunk_deserved_persistence` is {status}.** "
            + str(availability.get("reason", "")),
        ]
    lines += [
        "",
        "That gap is carried, not filled. The primary comparison therefore covers the",
        "seasons listed with each result below, and its macro-average is over those",
        "seasons only.",
        "",
    ]
    return lines


def _render_focal_view(
    metrics: dict[str, Any],
    bootstrap: dict[str, Any],
    spec: dict[str, Any],
    reference: str,
) -> list[str]:
    challenger = spec["primary_comparison"]["challenger"]
    focal = spec["focal_presentation_view"]
    key = f"{PRIMARY_TARGET}|K{FOCAL_PRESENTATION_CUTOFF}_H{PRIMARY_HORIZON}"
    cell = metrics["cells"].get(key)
    lines = [
        "## Focal presentation view",
        "",
        "Presented first for readability. It is one of the four equally prespecified",
        "cutoffs at the primary horizon, not a headline the specification singles out --",
        "the full family follows in the next section and carries equal standing.",
        "",
    ]
    if not cell or cell.get("status") == "no_windows":
        return lines + ["No windows in the focal cell.", ""]

    comparison = cell["comparisons"][challenger]
    if comparison.get("status") != "measured":
        return lines + [
            f"The focal comparison could not be measured: {comparison.get('status')}.",
            "",
        ]

    pooled = comparison["pooled_window_weighted"]
    macro = comparison["macro_average_of_season_deltas"]
    balanced = comparison["batter_balanced"]
    magnitude = comparison["practical_magnitude"]
    boot = _cell_bootstrap(
        bootstrap,
        target=PRIMARY_TARGET,
        challenger=challenger,
        reference=reference,
        cutoff=FOCAL_PRESENTATION_CUTOFF,
        horizon=PRIMARY_HORIZON,
    )

    lines += [
        f"**{focal['statement']}** -- `{challenger}` versus `{reference}`.",
        "",
        f"- Seasons in the comparison: {comparison['seasons_in_comparison']}"
        + (
            f"; excluded for the missing rung: {comparison['seasons_excluded_for_missing_rung']}"
            if comparison["seasons_excluded_for_missing_rung"]
            else ""
        ),
        f"- Paired windows: {comparison['n_paired_windows']} "
        f"({balanced['n_unique_batters']} unique hitters)",
        "",
        "| view | delta_MAE | 95% CI | delta_RMSE | 95% CI | pct_change_MAE | direction |",
        "|---|---|---|---|---|---|---|",
        "| pooled (window-weighted) | "
        + _number(pooled["delta_mae"])
        + " | "
        + _interval((boot or {}).get("pooled", {}).get("delta_mae"))
        + " | "
        + _number(pooled["delta_rmse"])
        + " | "
        + _interval((boot or {}).get("pooled", {}).get("delta_rmse"))
        + " | "
        + _number(pooled["pct_change_mae"], 2)
        + "% | "
        + DIRECTION_WORDS[pooled["direction_mae"]]
        + " |",
        "| macro-average of season deltas | "
        + _number(macro["macro_delta_mae"])
        + " | "
        + _interval((boot or {}).get("macro_average_of_season_deltas", {}).get("delta_mae"))
        + " | "
        + _number(macro["macro_delta_rmse"])
        + " | "
        + _interval((boot or {}).get("macro_average_of_season_deltas", {}).get("delta_rmse"))
        + " | "
        + _number(macro["macro_pct_change_mae"], 2)
        + "% | "
        + (DIRECTION_WORDS.get(macro["direction_mae"], "n/a"))
        + " |",
        "| batter-balanced | "
        + _number(balanced["delta"]["delta_mae"])
        + " | n/a | "
        + _number(balanced["delta"]["delta_rmse"])
        + " | n/a | "
        + _number(balanced["delta"]["pct_change_mae"], 2)
        + "% | "
        + DIRECTION_WORDS[balanced["delta"]["direction_mae"]]
        + " |",
        "",
        "### Season by season",
        "",
        "| season | n | MAE realized | MAE deserved | delta_MAE | 95% CI | direction |",
        "|---|---|---|---|---|---|---|",
    ]
    for season, delta in sorted(comparison["by_season"].items()):
        season_boot = (boot or {}).get("by_season", {}).get(season, {})
        lines.append(
            f"| {season} | {delta['n']} | "
            + _number(delta["mae_reference"])
            + " | "
            + _number(delta["mae_challenger"])
            + " | "
            + _number(delta["delta_mae"])
            + " | "
            + _interval(season_boot.get("delta_mae"))
            + " | "
            + DIRECTION_WORDS[delta["direction_mae"]]
            + " |"
        )

    lines += [
        "",
        "### Practical magnitude",
        "",
        f"- Target SD in this cell: {_number(magnitude['target_sd'], 3)} runs/100.",
        f"- |delta_MAE| as a fraction of that SD: "
        f"{_number(magnitude['abs_delta_mae_as_fraction_of_target_sd'], 4)}.",
        f"- Percentage change in MAE: {_number(pooled['pct_change_mae'], 2)}%.",
        "",
        "A difference that is statistically distinguishable from zero but small relative",
        "to the target's own spread is a small difference, and is described here as one.",
        "",
        "### Full comparison ladder in the focal cell",
        "",
        "Each rung measured on its own complete cases. Correlations for a constant",
        "predictor are undefined, not zero. `league_mean` is constant WITHIN each",
        "season but differs between them, so its pooled correlation is a between-season",
        "artifact and is labelled as one; per season it is undefined, as",
        "`r1_metrics.json`'s `ladder_by_season` shows.",
        "",
        "| rung | n | MAE | RMSE | Pearson | Spearman |",
        "|---|---|---|---|---|---|",
    ]
    for rung in cell["ladder_pooled"]["metrics"]:
        seasonal = bool(rung.get("constant_within_every_season"))
        lines.append(
            f"| `{rung['predictor']}` | {rung['n']} | "
            + _number(rung["mae"])
            + " | "
            + _number(rung["rmse"])
            + " | "
            + _correlation(
                rung["pearson"], rung["pearson_status"], constant_within_every_season=seasonal
            )
            + " | "
            + _correlation(
                rung["spearman"], rung["spearman_status"], constant_within_every_season=seasonal
            )
            + " |"
        )
    lines.append("")
    return lines


def _render_all_cells(
    metrics: dict[str, Any], bootstrap: dict[str, Any], reference: str
) -> list[str]:
    """Every prespecified cell and comparison, favourable or not."""
    lines = [
        "## Every prespecified cell",
        "",
        "The primary family is all four cutoffs at the primary horizon, with equal",
        "standing; the focal view is marked only to locate it, not to elevate it.",
        "All targets, cutoffs, horizons and ladder rungs, reported in full. Rows where",
        "realized wins, where the interval crosses zero, or where a correlation is weak",
        "are present by construction -- this table is generated by walking the artifact.",
        "",
        "| target | cutoff | horizon | challenger vs shrunk_realized | seasons | n | delta_MAE | 95% CI | delta_RMSE | direction |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(metrics["cells"]):
        cell = metrics["cells"][key]
        if cell.get("status") == "no_windows":
            continue
        target_short = cell["target"].replace("target_", "").replace("_rv_per_100", "")
        for challenger, comparison in sorted(cell.get("comparisons", {}).items()):
            if comparison.get("status") != "measured":
                lines.append(
                    f"| {target_short} | {cell['cutoff']} | {cell['horizon']} | "
                    f"`{challenger}` | - | 0 | n/a | n/a | n/a | "
                    f"{comparison.get('status', 'unmeasured')} |"
                )
                continue
            pooled = comparison["pooled_window_weighted"]
            boot = _cell_bootstrap(
                bootstrap,
                target=cell["target"],
                challenger=challenger,
                reference=reference,
                cutoff=cell["cutoff"],
                horizon=cell["horizon"],
            )
            flag = (
                " *(focal view)*"
                if cell["is_focal_presentation_view"] and challenger.startswith("shrunk_deserved")
                else ""
            )
            lines.append(
                f"| {target_short} | {cell['cutoff']} | {cell['horizon']} | "
                f"`{challenger}`{flag} | "
                + ",".join(str(s) for s in comparison["seasons_in_comparison"])
                + f" | {comparison['n_paired_windows']} | "
                + _number(pooled["delta_mae"])
                + " | "
                + _interval((boot or {}).get("pooled", {}).get("delta_mae"))
                + " | "
                + _number(pooled["delta_rmse"])
                + " | "
                + DIRECTION_WORDS[pooled["direction_mae"]]
                + " |"
            )
    lines.append("")
    return lines


def _render_batter_balanced(metrics: dict[str, Any], reference: str) -> list[str]:
    """Whether equal-weighting hitters changes the sign or the magnitude."""
    lines = [
        "## Batter-balanced sensitivity",
        "",
        "Loss is averaged within each hitter first, then every unique hitter is weighted",
        "equally, so a hitter who qualified in three seasons counts once rather than",
        "three times. Compared against the ordinary window-weighted result.",
        "",
        "| target | cutoff | horizon | challenger | delta_MAE (window) | delta_MAE (batter-balanced) | shift | sign unchanged? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(metrics["cells"]):
        cell = metrics["cells"][key]
        if cell.get("status") == "no_windows":
            continue
        target_short = cell["target"].replace("target_", "").replace("_rv_per_100", "")
        for challenger, comparison in sorted(cell.get("comparisons", {}).items()):
            if comparison.get("status") != "measured":
                continue
            balanced = comparison["batter_balanced"]
            lines.append(
                f"| {target_short} | {cell['cutoff']} | {cell['horizon']} | `{challenger}` | "
                + _number(comparison["pooled_window_weighted"]["delta_mae"])
                + " | "
                + _number(balanced["delta"]["delta_mae"])
                + " | "
                + _number(balanced["magnitude_shift_delta_mae"])
                + " | "
                + ("yes" if balanced["sign_matches_window_weighted"] else "**NO**")
                + " |"
            )
    lines.append("")
    return lines


def _render_regression_direction(regression: dict[str, Any]) -> list[str]:
    """The regression-direction signal beside the placebo that mimics it."""
    lines = [
        "## Regression-direction experiment and its placebo",
        "",
        "Outcome: `future_realized - current_realized`. That outcome contains",
        "`-current_realized`, and `contact_result_surprise = current_realized -",
        "current_deserved` contains `+current_realized`. A negative association can",
        "therefore appear with no player-specific deserved information at all.",
        "",
        "The placebo permutes `current_deserved` across hitters within each cell and",
        "rebuilds `placebo_surprise = current_realized - permuted_current_deserved`,",
        "preserving the mechanical term and destroying only the hitter-to-deserved",
        "pairing. The meaningful comparison is real versus placebo, not real versus zero.",
        "",
        "`slope (mean-regr.)` is the realized-only baseline: `outcome ~ current_realized`,",
        "the generic tendency of extreme current performance to regress with no contact",
        "information involved.",
        "",
        "| cell | season | n | slope (mean-regr.) | slope surprise | placebo mean | p(real more negative) | incr. deserved slope | p(real more positive) | delta R2 | p(delta R2) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(regression["regression_direction_by_cell"]):
        cell = regression["regression_direction_by_cell"][key]
        label = f"K{cell['cutoff']}_H{cell['horizon']}"
        if cell.get("status") != "fitted":
            lines.append(
                f"| {label} | {cell['season']} | {cell.get('n', 0)} | "
                f"{cell.get('status', 'unmeasured')} | | | | | | | |"
            )
            continue
        baseline = cell["mean_regression_baseline"]
        surprise = cell["real_surprise_regression"]
        incremental = cell["real_incremental_regression"]
        placebo = cell["placebo"]
        marker = " *(focal view)*" if cell["is_focal_presentation_view"] else ""
        lines.append(
            f"| {label}{marker} | {cell['season']} | {cell['n']} | "
            + _number(baseline["slope_current_realized"], 3)
            + " | "
            + _number(surprise["slope_surprise"], 3)
            + " | "
            + _number(placebo["surprise_slope"]["placebo_mean"], 3)
            + " | "
            + _number(placebo["surprise_slope"]["p_lower_real_more_negative"], 3)
            + " | "
            + _number(incremental["slope_current_deserved"], 3)
            + " | "
            + _number(placebo["incremental_deserved_slope"]["p_upper_real_more_positive"], 3)
            + " | "
            + _number(incremental["delta_r_squared_vs_mean_regression_baseline"], 4)
            + " | "
            + _number(placebo["delta_r_squared"]["p_upper_real_more_positive"], 3)
            + " |"
        )
    lines += [
        "",
        "A cell whose real surprise slope sits inside the placebo distribution is a cell",
        "where the apparent regression signal is explained by the mechanical coupling,",
        "not by contact-stage information. Those cells are in the table above.",
        "",
    ]
    return lines


def _render_direction_probability(regression: dict[str, Any]) -> list[str]:
    """Proper scoring rules for the decline probability, never accuracy alone."""
    direction = regression["direction_probability"]
    lines = [
        "## Probability of decline (walk-forward)",
        "",
        f"Event: {direction['event']}.",
        "",
        direction["training_rule"] + ".",
        "",
        "Accuracy is shown only beside the majority-class rate a constant predictor",
        "would achieve; Brier score, log loss and calibration are the evaluation.",
        "",
        "**The same mechanical coupling applies here.** The event is defined as",
        "`future_realized < current_realized`, so it already contains",
        "`current_realized`; a hitter with an extreme current rate is mechanically more",
        "likely to move back toward the mean. `realized_only` beating `base_rate` is",
        "therefore expected and is not evidence about contact quality. The comparison",
        "that carries information is `realized_plus_deserved` against `realized_only`.",
        "",
        "| cell | eval season | n | model | base rate | Brier | log loss | acc @0.5 | majority acc |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(direction["cells"]):
        fold = direction["cells"][key]
        if fold.get("status") != "fitted":
            lines.append(
                f"| {key} | - | {fold.get('n_test', 0)} | {fold.get('status')} | | | | | |"
            )
            continue
        label = f"K{fold['cutoff']}_H{fold['horizon']}"
        for model_name, model in sorted(fold["models"].items()):
            lines.append(
                f"| {label} | {fold['evaluate_season']} | {model['n']} | `{model_name}` | "
                + _number(model["event_prevalence"], 3)
                + " | "
                + _number(model["brier_score"], 4)
                + " | "
                + _number(model["log_loss"], 4)
                + " | "
                + _number(model["accuracy_at_0_5"], 3)
                + " | "
                + _number(model["majority_class_accuracy"], 3)
                + " |"
            )
    lines += [
        "",
        "Calibration by probability bin for every model and fold is in",
        "`r1_regression_direction.json` under `direction_probability.cells.*.models.*.",
        "calibration_by_bin`; it is not summarised into a single number here because a",
        "single calibration number hides which part of the probability range is wrong.",
        "",
    ]
    return lines


def _render_limitations(metrics: dict[str, Any]) -> list[str]:
    return [
        "## Limitations carried forward",
        "",
        "- " + metrics["specification"]["contact_stage_disclaimer"],
        "- " + metrics["specification"]["conditional_forecast"][str(PRIMARY_HORIZON)],
        "- A hitter-season enters a cell only if it reached `cutoff + horizon` resolved",
        "  eligible batted balls. Both compared predictors share that condition, so it",
        "  does not bias the comparison, but it does bound what the results generalize",
        "  to. Per-cell exclusion rates and the realized-rate gap between included and",
        "  excluded hitters are in `r1_metrics.json` under",
        "  `assembly_audit.seasons.*.inclusion_by_cell`.",
        "- Shrinkage cannot change rank order within a cell: every hitter in a fixed",
        "  cutoff cell has the same batted-ball count and therefore the same shrinkage",
        "  factor, making the map affine. Spearman is identical for a raw rung and its",
        "  shrunk counterpart by construction, and no rank improvement from shrinkage is",
        "  claimed anywhere.",
        "- Windows repeat hitters across cutoffs and seasons, so the window count is not",
        "  a count of independent observations. Every interval resamples whole hitters.",
        "- The walk-forward contact model is a refit of the frozen current specification",
        "  on strictly earlier rows. It is not a model that historically existed in the",
        "  scored season.",
        "",
    ]
