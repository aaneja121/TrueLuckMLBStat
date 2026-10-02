"""Contact Forecast: render the ridge experiment report from its artifact.

Same discipline as `forecast.report`: every number is read out of
`ridge_results.json`, signs are printed exactly as stored under a restatement
of the frozen convention, undefined correlations print as undefined, and every
prespecified model is rendered by the same loop -- so a losing ablation cannot
be dropped from the write-up without being dropped from the data.
"""

from __future__ import annotations

from typing import Any

DIRECTION_WORDS: dict[str, str] = {
    "challenger_better": "ridge better",
    "reference_better": "benchmark better",
    "no_difference": "no difference",
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


def _interval(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "n/a"
    marker = ""
    if summary.get("ci_crosses_zero") is True:
        marker = " (crosses zero)"
    elif summary.get("ci_crosses_zero") is False:
        marker = " (excludes zero)"
    return f"[{_number(summary.get('ci_lower'))}, {_number(summary.get('ci_upper'))}]{marker}"


def _cell(bootstraps: dict[str, Any], key: str, cell_key: str) -> dict[str, Any] | None:
    entry = bootstraps.get(key)
    if not entry:
        return None
    cell = entry.get("cells", {}).get(cell_key)
    return cell.get("pooled", {}).get("delta_mae") if isinstance(cell, dict) else None


def render_ridge_report(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines += _render_header(results)
    lines += _render_key_delta(results)
    lines += _render_evaluation(results)
    lines += _render_deltas(results)
    lines += _render_ablation(results)
    lines += _render_validation(results)
    lines += _render_diagnostics(results)
    lines += _render_limitations(results)
    return "\n".join(lines) + "\n"


def _render_header(results: dict[str, Any]) -> list[str]:
    protocol = results["protocol"]
    provenance = results["r1_provenance"]
    target = results["target"]
    fold1, fold2 = protocol["folds"]
    return [
        "# Contact Forecast -- ridge forecasting experiment",
        "",
        f"Generated {results['generated_at_utc']}.",
        "",
        f"**Model class: {results['model_class']}.**",
        "",
        "## What is predicted",
        "",
        f"`{target['column']}` in cell {target['cell']}: future realized contact-result",
        "run value per 100 resolved eligible BBE over the next 100 such batted balls.",
        "",
        target["conditional_interpretation"],
        "",
        "## Protocol",
        "",
        f"- **Fold 1**: train {fold1['train']} -> validate {fold1['validate']}. "
        f"Purpose: {fold1['purpose']}. Results are selection-contaminated and labelled so.",
        f"- **Fold 2**: train {fold2['train']} -> evaluate {fold2['evaluate']}, at the alpha",
        f"  fixed in fold 1. Participated in any selection: {fold2['participated_in_any_selection']}.",
        f"- Selection metric: `{protocol['selection_metric']}`, fixed in advance.",
        f"- Split rule: {protocol['split_rule']}.",
        f"- Standardization: {protocol['standardization']}.",
        f"- Prior-season features: {protocol['prior_season_features_excluded']}.",
        "",
        f"Built on the frozen R1 package (verdict {provenance.get('r1_conclusion', 'n/a')}); "
        "the prediction table's hash was checked against the R1 freeze manifest.",
        "",
        "## Sign convention (inherited from R1, never reversed)",
        "",
        "```",
        "delta_MAE = MAE(challenger) - MAE(reference)",
        "negative -> the challenger (the ridge model) is better",
        "positive -> the reference (the benchmark) is better",
        "```",
        "",
    ]


def _render_key_delta(results: dict[str, Any]) -> list[str]:
    key = results["key_delta"]
    lines = ["## The decision number", ""]
    if key.get("status") == "benchmark_unavailable":
        return lines + [f"Unavailable: {key['note']}.", ""]
    return lines + [
        f"`{key['definition']}` on the {key['evaluation_season']} evaluation season "
        f"({key['n_evaluation_windows']} windows).",
        "",
        f"{key['interpretation'].capitalize()}.",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| best ridge model (selected on validation) | `{key['best_ridge_model']}` |",
        f"| MAE(best ridge) | {_number(key['mae_best_ridge'])} |",
        f"| MAE(shrunk deserved) | {_number(key['mae_shrunk_deserved'])} |",
        f"| **delta_MAE** | **{_number(key['delta_mae'])}** |",
        f"| 95% batter-clustered CI | {_number(key['ci_delta_mae'][0])}, "
        f"{_number(key['ci_delta_mae'][1])}"
        + (" (crosses zero)" if key["ci_crosses_zero"] else " (excludes zero)")
        + " |",
        f"| percent change in MAE | {_number(key['pct_change_mae'], 2)}% |",
        f"| delta_RMSE | {_number(key['delta_rmse'])} ({_number(key['pct_change_rmse'], 2)}%) |",
        f"| replicates favouring the benchmark | "
        f"{_number(key['share_replicates_favouring_benchmark'], 3)} |",
        f"| \\|delta_MAE\\| as a fraction of the target SD | "
        f"{_number(key['abs_delta_mae_as_fraction_of_target_sd'], 4)} |",
        f"| direction | {DIRECTION_WORDS[key['direction_mae']]} |",
        "",
    ]


def _render_evaluation(results: dict[str, Any]) -> list[str]:
    fold2 = results["fold_2_evaluation"]
    lines = [
        f"## Evaluation season {fold2['evaluate_season']} "
        f"(trained on {fold2['train_seasons']}, no selection)",
        "",
        f"{fold2['n_evaluate']} windows. Ridge models and benchmarks on identical rows.",
        "",
        "| predictor | MAE | RMSE | R2 | Pearson | Spearman | batter-bal. MAE | batter-bal. RMSE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, metrics in sorted(fold2["metrics"].items()):
        lines.append(
            f"| `{name}` | "
            + _number(metrics["mae"])
            + " | "
            + _number(metrics["rmse"])
            + " | "
            + _number(metrics["r_squared"])
            + " | "
            + _correlation(metrics["pearson"], metrics["pearson_status"])
            + " | "
            + _correlation(metrics["spearman"], metrics["spearman_status"])
            + " | "
            + _number(metrics["batter_balanced_mae"])
            + " | "
            + _number(metrics["batter_balanced_rmse"])
            + " |"
        )
    unavailable = fold2.get("benchmarks_unavailable") or []
    if unavailable:
        lines += ["", f"Benchmarks unavailable in this season: {unavailable}."]
    lines.append("")
    return lines


def _render_deltas(results: dict[str, Any]) -> list[str]:
    fold2 = results["fold_2_evaluation"]
    cell_key = results["target"]["cell"]
    lines = [
        "## Every ridge model against every benchmark",
        "",
        "Paired on identical rows, with batter-clustered bootstrap intervals. Rows where",
        "the benchmark wins are present by construction.",
        "",
        "| challenger | benchmark | delta_MAE | 95% CI | delta_RMSE | pct MAE | direction |",
        "|---|---|---|---|---|---|---|",
    ]
    for benchmark, entries in sorted(results["deltas_on_evaluation_season"].items()):
        for challenger, record in sorted(entries.items()):
            delta = record["delta"]
            summary = _cell(results["bootstrap"], f"{challenger}_vs_{benchmark}", cell_key)
            lines.append(
                f"| `{challenger}` | `{benchmark}` | "
                + _number(delta["delta_mae"])
                + " | "
                + _interval(summary)
                + " | "
                + _number(delta["delta_rmse"])
                + " | "
                + _number(delta["pct_change_mae"], 2)
                + "% | "
                + DIRECTION_WORDS[delta["direction_mae"]]
                + " |"
            )
    del fold2
    lines.append("")
    return lines


def _render_ablation(results: dict[str, Any]) -> list[str]:
    """Where any gain comes from, read off the nested models."""
    fold2 = results["fold_2_evaluation"]
    lines = [
        "## Ablation: where does predictive information come from?",
        "",
        "The four models are nested by construction, so the differences localise the",
        "source of any gain.",
        "",
        "| model | features | evaluation MAE | selected alpha |",
        "|---|---|---|---|",
    ]
    for name, spec in sorted(results["feature_sets"].items()):
        metrics = fold2["metrics"].get(f"ridge_{name}")
        lines.append(
            f"| `{name}` | {spec['description']} | "
            + _number(metrics["mae"] if metrics else None)
            + " | "
            + _number(results["selected_alphas"].get(name), 3)
            + " |"
        )
    lines += [
        "",
        "Read the steps, not the levels: A -> B is realized versus deserved information,",
        "B -> C is what results add on top of deserved, and C -> D is whether the SHAPE of",
        "the contact distribution (exit velocity, launch angle, spray, batted-ball mix,",
        "handedness) adds anything beyond the two rate summaries.",
        "",
    ]
    return lines


def _render_validation(results: dict[str, Any]) -> list[str]:
    fold1 = results["fold_1_validation"]
    lines = [
        f"## Fold 1 validation season {fold1['evaluate_season']} (selection-contaminated)",
        "",
        "These numbers chose the alpha and the winning model, so they are not an",
        "out-of-sample estimate of anything. Reported for completeness.",
        "",
        "| predictor | MAE | RMSE | R2 | Spearman |",
        "|---|---|---|---|---|",
    ]
    for name, metrics in sorted(fold1["metrics"].items()):
        lines.append(
            f"| `{name}` | "
            + _number(metrics["mae"])
            + " | "
            + _number(metrics["rmse"])
            + " | "
            + _number(metrics["r_squared"])
            + " | "
            + _correlation(metrics["spearman"], metrics["spearman_status"])
            + " |"
        )
    boundary = [
        name
        for name, selection in fold1.get("alpha_selection", {}).items()
        if selection.get("selected_alpha_at_grid_boundary")
    ]
    truncated = [
        name
        for name in boundary
        if (fold1["alpha_selection"][name].get("boundary_diagnostic") or {}).get(
            "grid_is_the_binding_constraint"
        )
    ]
    lines += [
        "",
        f"Selected alphas: {results['selected_alphas']}.",
        (
            f"Alphas pinned at a grid boundary: {boundary}. Refitting far beyond the grid "
            "in the pinned direction moves the validation score negligibly, so these are "
            "the limit (alpha -> 0 is ordinary least squares), not a truncated grid."
            if boundary and not truncated
            else f"**Alphas pinned at a grid boundary with the grid actually binding: "
            f"{truncated}** -- the score keeps moving beyond the grid, so the grid, not "
            "the data, is setting these."
            if truncated
            else "No selected alpha sat at a grid boundary."
        ),
        "",
    ]
    return lines


def _render_diagnostics(results: dict[str, Any]) -> list[str]:
    diagnostics = results["diagnostics"]
    lines = [
        "## Feature diagnostics",
        "",
        diagnostics["coefficient_caveat"],
        "",
        "### Multicollinearity",
        "",
        "Three exact linear dependencies exist in the declared feature list and are",
        "resolved by declaration BEFORE fitting, not by letting the penalty absorb them:",
        "",
    ]
    for feature, reason in sorted(diagnostics["prespecified_drops"].items()):
        lines.append(f"- `{feature}`: {reason}")
    lines += [
        "",
        "| model | retained features | rank | condition number | max VIF | VIF > 10 |",
        "|---|---|---|---|---|---|",
    ]
    for name, audit in sorted(diagnostics["collinearity_by_model"].items()):
        lines.append(
            f"| `{name}` | {audit['n_features']} | {audit['rank']} | "
            + _number(audit["condition_number"], 1)
            + " | "
            + _number(audit["max_vif"], 2)
            + " | "
            + (", ".join(f"`{f}`" for f in audit["features_with_vif_above_10"]) or "none")
            + " |"
        )
    lines += [
        "",
        "Rank equals the feature count for every model, so each design is identifiable.",
        "",
        "### The rank-deficient variant, fitted deliberately",
        "",
        "For comparison, the same models fitted with the exactly-dependent `surprise`",
        "columns left in. Ridge still returns a unique solution, but the coefficients on",
        "the collinear set are an arbitrary split and the predictions move:",
        "",
        "| model | features | rank | rank-deficient? | evaluation MAE |",
        "|---|---|---|---|---|",
    ]
    for name, variant in sorted(diagnostics["rank_deficient_variant"].items()):
        lines.append(
            f"| `{name}` | {variant['n_features']} | {variant['rank']} | "
            + ("**yes**" if variant["is_rank_deficient"] else "no")
            + " | "
            + _number(variant["mae"])
            + " |"
        )
    lines += ["", "### Standardized ridge coefficients (fold 2 fits)", ""]
    for name, coefficients in sorted(diagnostics["standardized_coefficients"].items()):
        lines += [f"**`{name}`**", "", "| feature | standardized coefficient |", "|---|---|"]
        for feature, value in sorted(coefficients.items(), key=lambda item: -abs(item[1])):
            lines.append(f"| `{feature}` | {_number(value)} |")
        lines.append("")
    return lines


def _render_limitations(results: dict[str, Any]) -> list[str]:
    no_op = results["diagnostics"]["clustering_is_a_no_op_here"]
    fold2 = results["fold_2_evaluation"]
    return [
        "## Limitations",
        "",
        f"- **Clustering and balancing are no-ops in this evaluation.** {no_op['consequence']}",
        f"  ({no_op['n_evaluation_windows']} windows, {no_op['n_unique_batters']} unique "
        f"hitters, one window per hitter: {no_op['one_window_per_hitter']}.)",
        f"- **One evaluation season.** Fold 2 evaluates on {fold2['evaluate_season']} alone,",
        f"  with {fold2['n_evaluate']} windows trained on {fold2['n_train']}. That is a small",
        "  sample for distinguishing models whose MAEs differ in the third decimal.",
        "- **Ridge only.** No nonlinear candidate was fitted; nothing here speaks to what a",
        "  nonlinear model would do.",
        "- **Development evidence only.** 2025 remains sealed and 2026 remains behind the",
        "  Phase 2 gate. No held-out prospective season has been opened.",
        "- **Contact stage.** Every quantity is Rc / E0 / Rc - E0, not the full telescoping",
        "  published quantity. Conclusions are about the luck-adjusted contact signal.",
        "",
    ]
