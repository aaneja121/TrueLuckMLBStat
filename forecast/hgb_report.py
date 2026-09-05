"""Contact Forecast: render the HGB experiment report from its artifact.

Same discipline as the earlier report renderers: every number is read out of
`hgb_results.json`, signs are printed exactly as stored, undefined
correlations print as undefined, and both formulations plus every benchmark
are rendered by the same loop, so an unfavourable result cannot leave the
write-up without leaving the data.
"""

from __future__ import annotations

from typing import Any

DIRECTION_WORDS: dict[str, str] = {
    "challenger_better": "HGB better",
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


def _pooled(results: dict[str, Any], key: str) -> dict[str, Any] | None:
    entry = results["bootstrap"].get(key)
    if not entry:
        return None
    cell = entry["cells"].get(results["target"]["cell"])
    return cell["pooled"]["delta_mae"] if cell else None


def render_hgb_report(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines += _render_header(results)
    lines += _render_verdict(results)
    lines += _render_evaluation(results)
    lines += _render_deltas(results)
    lines += _render_residual(results)
    lines += _render_stability(results)
    lines += _render_selection(results)
    lines += _render_importance(results)
    lines += _render_limitations(results)
    return "\n".join(lines) + "\n"


def _render_header(results: dict[str, Any]) -> list[str]:
    spec = results["specification"]
    grid = results["candidate_grid"]
    provenance = results["provenance"]
    return [
        "# Contact Forecast -- HistGradientBoosting experiment",
        "",
        f"Generated {results['generated_at_utc']}.",
        "",
        f"**{results['model_family']}.**",
        "Not implemented, by design: " + ", ".join(results["explicitly_not_implemented"]) + ".",
        "",
        "## What is predicted",
        "",
        f"`{results['target']['column']}` in cell {results['target']['cell']}.",
        "",
        results["target"]["conditional_interpretation"],
        "",
        "## Specification, frozen before 2024 was touched",
        "",
        f"- Primary formulation: **{spec['primary_formulation']}**; "
        f"sensitivity: **{spec['sensitivity_formulation']}**.",
        f"- {spec['formulation_designation']}",
        f"- Feature set: `{spec['feature_set']}` ({len(spec['feature_list'])} features). "
        f"{spec['feature_provenance']}.",
        f"- Preprocessing: {spec['preprocessing']}. Loss: `{spec['loss']}`. "
        f"Seed: {spec['random_seed']}.",
        f"- Grid: {grid['n_candidates']} prespecified candidates over "
        + ", ".join(f"`{k}`" for k in sorted(grid["searched_parameters"]))
        + f". {grid['expansion_rule']}.",
        f"- Selected hyperparameters: {results['fold_1_selection']['selected_searched_parameters']}.",
        "",
        "Provenance chain: R1 freeze verdict "
        f"{provenance['r1'].get('r1_conclusion', 'n/a')}, ridge freeze verdict "
        f'"{provenance["ridge"].get("ridge_verdict", "n/a")}". The frozen ridge was '
        "refit and reproduced its sealed evaluation MAE exactly "
        f"({_number(provenance['ridge'].get('reproduced_ridge_evaluation_mae'))}).",
        "",
        "Residual baseline causality, verified from the frozen R1 artifact: "
        + ", ".join(
            f"{season} "
            + (
                f"fit on {entry['fit_seasons']}"
                if entry.get("available")
                else "unavailable (no causal contact model)"
            )
            for season, entry in sorted(
                provenance["residual_baseline_causality"].get("by_season", {}).items()
            )
        )
        + ".",
        "",
        "## Sign convention (inherited, never reversed)",
        "",
        "```",
        "delta_MAE = MAE(challenger) - MAE(reference)",
        "negative -> the challenger (HGB) is better",
        "positive -> the reference (the benchmark) is better",
        "```",
        "",
    ]


def _render_verdict(results: dict[str, Any]) -> list[str]:
    verdict = results["nonlinear_value_test"]
    return [
        "## The required nonlinear-value test",
        "",
        f"**Question.** {verdict['question']}",
        "",
        f"**Answer.** {verdict['answer']}",
        "",
        f"Prespecified rule, declared before the comparison ran: {verdict['rule']} "
        f"Note that {verdict['explicitly_insufficient']}.",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| delta_MAE(HGB - frozen ridge) | {_number(verdict['delta_mae_hgb_minus_ridge'])} |",
        f"| percent change | {_number(verdict['pct_change_mae'], 2)}% |",
        f"| 95% paired batter-bootstrap CI | {_number(verdict['ci_delta_mae'][0])}, "
        f"{_number(verdict['ci_delta_mae'][1])}"
        + (" (crosses zero)" if verdict["ci_crosses_zero"] else " (excludes zero)")
        + " |",
        f"| point estimate favours HGB | {verdict['point_estimate_favours_hgb']} |",
        f"| interval excludes zero | {verdict['interval_excludes_zero']} |",
        f"| **preferred model** | **{verdict['preferred_model']}** |",
        "",
    ]


def _render_evaluation(results: dict[str, Any]) -> list[str]:
    fold2 = results["fold_2_evaluation"]
    lines = [
        f"## Evaluation season {fold2['evaluate_season']} (evaluated exactly once)",
        "",
        f"{fold2['n_evaluate']} windows. HGB-Direct trained on {fold2['train_seasons']} "
        f"({fold2['n_train_direct']} windows); HGB-Residual trained on "
        f"{fold2['residual_train_seasons']} ({fold2['n_train_residual']} windows -- the "
        "only season before 2024 with a causal shrunk-deserved baseline).",
        "",
        "| predictor | MAE | RMSE | R2 | Pearson | Spearman | batter-bal. MAE |",
        "|---|---|---|---|---|---|---|",
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
            + " |"
        )
    lines.append("")
    return lines


def _render_deltas(results: dict[str, Any]) -> list[str]:
    lines = [
        "## Both formulations against every benchmark",
        "",
        "Paired on identical rows, with batter-clustered bootstrap intervals.",
        "",
        "| challenger | benchmark | delta_MAE | 95% CI | delta_RMSE | pct MAE | direction |",
        "|---|---|---|---|---|---|---|",
    ]
    for benchmark, entries in sorted(results["deltas_on_evaluation_season"].items()):
        for challenger, delta in sorted(entries.items()):
            summary = _pooled(results, f"{challenger}_vs_{benchmark}")
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
    lines.append("")
    return lines


def _render_residual(results: dict[str, Any]) -> list[str]:
    diagnostics = results["residual_diagnostics"]
    return [
        "## Residual diagnostics",
        "",
        f"Residual target: `{diagnostics['residual_definition']}`. Does the "
        "contact-profile correction itself contain learnable signal?",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| n | {diagnostics['n']} |",
        f"| SD of the residual target | {_number(diagnostics['residual_target_sd'])} |",
        f"| mean residual | {_number(diagnostics['residual_target_mean'])} |",
        f"| MAE of predicting residual = 0 | "
        f"{_number(diagnostics['mae_predicting_residual_zero'])} |",
        f"| MAE of the HGB-predicted residual | "
        f"{_number(diagnostics['mae_hgb_predicted_residual'])} |",
        f"| correlation, predicted vs actual residual | "
        f"{_correlation(diagnostics['correlation_predicted_vs_actual_residual'], diagnostics['correlation_status'])} |",
        f"| fraction of benchmark error recovered | "
        f"{_number(diagnostics['fraction_of_benchmark_error_recovered'], 4)} |",
        "",
        diagnostics["note"],
        "",
    ]


def _render_stability(results: dict[str, Any]) -> list[str]:
    lines = [
        "## Stability: broad improvement, or a few extreme hitters?",
        "",
    ]
    for label, stability in sorted(results["stability"].items()):
        lines += [
            f"### {label.replace('_', ' ')}",
            "",
            f"`{stability['quantity']}` -- {stability['sign_meaning']}.",
            "",
            "| statistic | value |",
            "|---|---|",
            f"| hitters | {stability['n_hitters']} |",
            f"| mean | {_number(stability['mean'])} |",
            f"| median | {_number(stability['median'])} |",
            f"| Q1 / Q3 | {_number(stability['q1'])} / {_number(stability['q3'])} |",
            f"| p5 / p95 | {_number(stability['p5'])} / {_number(stability['p95'])} |",
            f"| min / max | {_number(stability['min'])} / {_number(stability['max'])} |",
            f"| fraction of hitters improved | "
            f"{_number(stability['fraction_of_hitters_improved'], 3)} |",
        ]
        for name, value in sorted(stability["trimmed_means"].items()):
            lines.append(f"| {name.replace('_', ' ')} | {_number(value)} |")
        lines += [
            "",
            "Top 5 most improved: "
            + ", ".join(_number(v, 2) for v in stability["top_5_most_improved"])
            + ". Bottom 5 most worsened: "
            + ", ".join(_number(v, 2) for v in stability["bottom_5_most_worsened"])
            + ".",
            "",
            stability["reading"].capitalize() + ".",
            "",
        ]
    return lines


def _render_selection(results: dict[str, Any]) -> list[str]:
    selection = results["fold_1_selection"]
    spread = selection["validation_mae_spread"]
    lines = [
        f"## Fold 1 selection (fit {selection['train_seasons']}, "
        f"select on {selection['validation_season']})",
        "",
        f"{selection['n_candidates']} prespecified candidates, selected on "
        f"`{selection['selection_metric']}`. {selection['split']}.",
        "",
        f"Validation MAE across the whole grid: best {_number(spread['best'])}, "
        f"median {_number(spread['median'])}, worst {_number(spread['worst'])}, "
        f"range {_number(spread['range'])}. A narrow range means the selection had "
        "little to choose between candidates, and the winner's edge is correspondingly "
        "fragile.",
        "",
        "| learning_rate | max_iter | max_leaf_nodes | l2_regularization | validation MAE |",
        "|---|---|---|---|---|",
    ]
    for row in sorted(selection["grid_scores"], key=lambda r: r["mae"]):
        p = row["parameters"]
        lines.append(
            f"| {p['learning_rate']} | {p['max_iter']} | {p['max_leaf_nodes']} | "
            f"{p['l2_regularization']} | {_number(row['mae'])} |"
        )
    lines.append("")
    return lines


def _render_importance(results: dict[str, Any]) -> list[str]:
    importance = results["permutation_importance"]
    lines = [
        "## Permutation importance (exploratory)",
        "",
        f"**{importance['status']}.** Computed on {importance['computed_on']}; "
        f"not computed on 2024: {importance['not_computed_on_2024']}; "
        f"used to revise the model: {importance['used_to_revise_the_model']}.",
        "",
        importance["caveat"],
        "",
        "| feature | mean importance | sd |",
        "|---|---|---|",
    ]
    for row in importance["ranking"][:10]:
        lines.append(
            f"| `{row['feature']}` | {_number(row['mean_importance'])} | {_number(row['sd'])} |"
        )
    lines += ["", "(Top 10 of the full ranking, which is in `hgb_results.json`.)", ""]
    return lines


def _render_limitations(results: dict[str, Any]) -> list[str]:
    fold2 = results["fold_2_evaluation"]
    return [
        "## Limitations",
        "",
        f"- **One evaluation season.** {fold2['n_evaluate']} windows in "
        f"{fold2['evaluate_season']}, one window per hitter, so batter-clustered "
        "resampling coincides with an ordinary bootstrap and batter-balanced MAE "
        "coincides with window-weighted MAE.",
        "- **The residual arm trains on one season.** 2022 has no causal "
        "shrunk-deserved value, so HGB-Residual has "
        f"{fold2['n_train_residual']} training windows against HGB-Direct's "
        f"{fold2['n_train_direct']}, and no validation estimate of its own.",
        "- **The residual arm's hyperparameters are inherited.** They were chosen by "
        "scoring on 2023, and the residual arm then trains on 2023. That is not 2024 "
        "leakage, but configurations suiting 2023's noise are mildly favoured.",
        "- **`surprise` is withheld from HGB.** It is dropped by the inherited ridge "
        "resolution because it is exactly `realized - deserved` -- a rank argument that "
        "binds for a linear model, not for a tree. A tree cannot easily synthesise that "
        "difference, so the exclusion withholds usable representation and is "
        "conservative against HGB rather than flattering to it.",
        "- **Development evidence only.** 2025 remains sealed, 2026 remains behind the "
        "Phase 2 gate, and no prediction intervals or dashboard surfaces were built.",
        "- **Importances are exploratory.** Nothing in this stage was revised on the "
        "basis of them, and they carry no causal reading.",
        "",
    ]
