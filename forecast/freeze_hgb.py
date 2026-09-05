"""Contact Forecast: freeze the completed HistGradientBoosting stage.

Seals the single nonlinear experiment: its report, results, candidate grid,
frozen specification, both formulations' fits, the residual diagnostics, the
stability distribution, the bootstrap output, and the recorded conclusion --
with the provenance chain running back through the ridge freeze to the R1
freeze.

## The conclusion is a preference, not a refutation

The prespecified rule preferred ridge, because HGB's point estimate against
the frozen ridge is WORSE and its interval crosses zero. That is a finding
about one family, one compact grid and one 283-window evaluation season. It is
not evidence that nonlinear modeling cannot help, and
`HGB_CONCLUSION["must_not_be_described_as"]` says so inside the artifact.

`assert_hgb_conclusion` re-derives the preference from the artifact, so the
frozen prose cannot outlive the numbers: if a later re-run ever made HGB the
preferred model, the freeze would refuse rather than ship a stale verdict.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from forecast.forecast_config import FORECAST_OUTPUTS_DIR
from forecast.freeze_r1 import ForecastFreezeError
from forecast.hgb import HGB_FIXED_PARAMETERS, HGB_PARAMETER_GRID
from forecast.stage_freeze import StageFreezeSpec, freeze_stage, verify_stage

logger = logging.getLogger(__name__)

HGB_FREEZE_VERSION = "contact_forecast_hgb_freeze_v1"

HGB_CONCLUSION: dict[str, Any] = {
    "verdict": (
        "PREFER RIDGE -- the single nonlinear candidate did not extract information "
        "from the contact profile that the full-profile ridge failed to capture."
    ),
    "statement": (
        "PREFER RIDGE. HistGradientBoosting was fitted as the only nonlinear family, "
        "in two prespecified formulations, over a 16-candidate grid fixed in advance. "
        "Its point estimate against the frozen full-profile ridge is WORSE, and the "
        "paired batter-bootstrap interval for that difference crosses zero, so by the "
        "prespecified preference rule ridge is retained on simplicity and "
        "interpretability grounds. HGB does beat shrunk deserved on the point "
        "estimate, but that advantage is statistically inconclusive and is carried by "
        "a small number of hitters rather than being broad."
    ),
    "focal_comparison": (
        "MAE(HGB-Direct) - MAE(full-profile ridge) = +0.0708 on the 2024 evaluation "
        "season, approximately +1.68%, with a 95% paired batter-bootstrap confidence "
        "interval of [-0.0420, +0.1891]. Positive favours ridge; the interval crosses "
        "zero, so neither model is established as better than the other."
    ),
    "evidence_class": "development-season evidence only",
    "must_not_be_described_as": [
        "evidence that nonlinear modeling cannot add value -- one family, one compact "
        "grid, 564 training windows and a single 283-window evaluation season were tested",
        "an established finding that HGB is worse than ridge: that interval also crosses zero",
        "a reason to close the nonlinear direction permanently",
        "out-of-sample evidence: 2025 is sealed and 2026 is behind the Phase 2 gate",
        "a validated production forecasting model of any kind",
    ],
    "preserved_negative_results": [
        "HGB-Direct is WORSE than the frozen full-profile ridge on the 2024 point "
        "estimate (delta_MAE +0.0708, +1.68%).",
        "HGB-Residual is worse than HGB-Direct (MAE 4.3081 versus 4.2836), so "
        "correcting the shrunk-deserved baseline did not beat modelling the target "
        "directly.",
        "The residual correction recovers only about 1.2% of the benchmark's error, "
        "with a predicted-versus-actual residual correlation near 0.14: the "
        "contact-profile correction contains very little learnable signal on this season.",
        "HGB's advantage over shrunk deserved is not broad. Only 51.9% of hitters "
        "improved -- barely a coin flip -- and the mean advantage collapses from "
        "+0.0769 to +0.0127 when the five most-improved hitters are excluded, turning "
        "negative once ten are.",
        "Every HGB-versus-shrunk-deserved interval on 2024 crosses zero.",
    ],
}

HGB_LIMITATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "one_family_one_small_grid",
        "statement": (
            "Exactly one nonlinear family was fitted, over 16 prespecified candidates "
            "spanning learning rate, iterations, leaf count and L2 regularization. "
            "min_samples_leaf was pinned at 20 on a sample-size argument made in "
            "advance. No conclusion about nonlinear modeling in general follows."
        ),
    },
    {
        "id": "residual_arm_is_structurally_handicapped",
        "statement": (
            "HGB-Residual trains on 2023 alone (284 windows against HGB-Direct's 564) "
            "because 2022 has no causal shrunk-deserved baseline, and it has no "
            "validation estimate of its own. Its hyperparameters are inherited from "
            "the direct arm, which were selected by scoring on 2023 -- the season the "
            "residual arm then trains on. That is not 2024 leakage, but configurations "
            "suiting 2023's noise are mildly favoured."
        ),
    },
    {
        "id": "surprise_withheld_from_the_tree",
        "statement": (
            "The inherited ridge resolution drops `surprise` because it is exactly "
            "realized minus deserved -- a rank argument that binds for a linear model, "
            "not for a tree. An axis-aligned splitter cannot easily synthesise that "
            "difference, so the exclusion withholds usable representation and is "
            "conservative AGAINST HGB rather than flattering to it."
        ),
    },
    {
        "id": "single_evaluation_season",
        "statement": (
            "283 windows in 2024, one per hitter, so batter-clustered resampling "
            "coincides with an ordinary bootstrap and batter-balanced MAE coincides "
            "with window-weighted MAE. Neither is additional evidence."
        ),
    },
    {
        "id": "importances_are_exploratory",
        "statement": (
            "Permutation importance was computed on the 2023 validation season with "
            "the fold-1 model, never on 2024, and no model decision in this stage "
            "consulted it. It carries no causal reading."
        ),
    },
)

_STATED_DELTA_VS_RIDGE = 0.0708
_STATED_PCT_VS_RIDGE = 1.68
_STATED_CI = (-0.0420, 0.1891)
_TOLERANCE = 5e-4
_PCT_TOLERANCE = 0.01

HGB_ARTIFACTS: tuple[str, ...] = (
    "hgb_results.json",
    "hgb_report.md",
    "hgb_candidate_grid.json",
    "hgb_specification.json",
)

HGB_SOURCE_MODULES: tuple[str, ...] = (
    "forecast/hgb.py",
    "forecast/run_hgb_forecast.py",
    "forecast/hgb_report.py",
    "forecast/stage_freeze.py",
    "forecast/freeze_hgb.py",
)


def extract_hgb_key_results(outputs_dir: Path) -> dict[str, Any]:
    """Everything the HGB conclusion rests on, pulled from its own artifacts."""
    results = json.loads((outputs_dir / "hgb_results.json").read_text())
    fold2 = results["fold_2_evaluation"]
    verdict = results["nonlinear_value_test"]
    stability = results["stability"]["vs_shrunk_deserved"]

    return {
        "nonlinear_value_test": {
            "question": verdict["question"],
            "answer": verdict["answer"],
            "preferred_model": verdict["preferred_model"],
            "delta_mae_hgb_minus_ridge": verdict["delta_mae_hgb_minus_ridge"],
            "pct_change_mae": verdict["pct_change_mae"],
            "ci_delta_mae": verdict["ci_delta_mae"],
            "ci_crosses_zero": verdict["ci_crosses_zero"],
        },
        "evaluation_mae": {
            name: metrics["mae"] for name, metrics in sorted(fold2["metrics"].items())
        },
        "evaluation_season": fold2["evaluate_season"],
        "n_evaluate": fold2["n_evaluate"],
        "n_train_direct": fold2["n_train_direct"],
        "n_train_residual": fold2["n_train_residual"],
        "deltas_vs_shrunk_deserved": {
            challenger: record["delta_mae"]
            for challenger, record in results["deltas_on_evaluation_season"][
                "shrunk_deserved_persistence"
            ].items()
        },
        "selected_hyperparameters": results["fold_1_selection"]["selected_searched_parameters"],
        "validation_mae_spread": results["fold_1_selection"]["validation_mae_spread"],
        "candidate_grid": {
            "n_candidates": results["candidate_grid"]["n_candidates"],
            "searched_parameters": results["candidate_grid"]["searched_parameters"],
            "fixed_parameters": results["candidate_grid"]["fixed_parameters"],
        },
        "feature_set": {
            "name": results["specification"]["feature_set"],
            "n_features": len(results["specification"]["feature_list"]),
            "provenance": results["specification"]["feature_provenance"],
        },
        "residual_diagnostics": results["residual_diagnostics"],
        "stability_vs_shrunk_deserved": {
            "mean": stability["mean"],
            "median": stability["median"],
            "fraction_of_hitters_improved": stability["fraction_of_hitters_improved"],
            "trimmed_means": stability["trimmed_means"],
        },
        "provenance": results["provenance"],
    }


def build_hgb_specification() -> dict[str, Any]:
    """The frozen nonlinear-stage specification, recorded with the manifest."""
    return {
        "model_family": "sklearn.ensemble.HistGradientBoostingRegressor",
        "families_explicitly_not_implemented": [
            "RandomForest",
            "XGBoost",
            "LightGBM",
            "neural network",
            "ensemble search",
            "model zoo",
        ],
        "searched_parameters": {k: list(v) for k, v in HGB_PARAMETER_GRID.items()},
        "fixed_parameters": dict(HGB_FIXED_PARAMETERS),
        "formulations": {
            "primary": "direct",
            "sensitivity": "residual",
            "designation_reason": (
                "forced by data availability -- 2022 has no causal shrunk-deserved "
                "baseline, so the residual arm cannot use the prescribed "
                "fit-2022/select-2023 structure -- and fixed before either 2024 number "
                "was computed"
            ),
        },
        "feature_provenance": (
            "inherited verbatim from the frozen ridge Model D definition and resolved "
            "by the same frozen rules, on TRAINING seasons only; no feature was chosen "
            "using 2024 ridge coefficients or any other 2024 result"
        ),
        "selection_structure": {
            "fit": [2022],
            "select": 2023,
            "final_fit": [2022, 2023],
            "evaluate": 2024,
            "evaluation_season_used_in_any_selection": False,
        },
        "stop_condition": (
            "stop after HGB evaluation; no prediction intervals, no dashboard UI, "
            "2025 sealed, 2026 behind the Phase 2 gate"
        ),
    }


def assert_hgb_conclusion(key_results: dict[str, Any]) -> None:
    """Refuse the freeze if the recorded prose no longer matches the artifact.

    Raises:
        ForecastFreezeError: If the stated focal numbers have drifted, if the
            prespecified rule no longer prefers ridge, or if a preserved
            negative result has silently turned positive.
    """
    verdict = key_results["nonlinear_value_test"]
    if abs(verdict["delta_mae_hgb_minus_ridge"] - _STATED_DELTA_VS_RIDGE) > _TOLERANCE:
        raise ForecastFreezeError(
            f"The conclusion states delta_MAE(HGB - ridge) = {_STATED_DELTA_VS_RIDGE}; "
            f"the artifact reads {verdict['delta_mae_hgb_minus_ridge']}."
        )
    if abs(verdict["pct_change_mae"] - _STATED_PCT_VS_RIDGE) > _PCT_TOLERANCE:
        raise ForecastFreezeError(
            f"The conclusion states approximately {_STATED_PCT_VS_RIDGE}%; the artifact "
            f"reads {verdict['pct_change_mae']}."
        )
    lower, upper = verdict["ci_delta_mae"]
    if abs(lower - _STATED_CI[0]) > _TOLERANCE or abs(upper - _STATED_CI[1]) > _TOLERANCE:
        raise ForecastFreezeError(
            f"The conclusion states a CI of {list(_STATED_CI)}; the artifact reads "
            f"{verdict['ci_delta_mae']}."
        )
    if verdict["preferred_model"] != "ridge":
        raise ForecastFreezeError(
            "The conclusion records PREFER RIDGE, but the artifact's prespecified rule "
            f"now prefers {verdict['preferred_model']!r}. Update the conclusion "
            "deliberately rather than freezing a stale verdict."
        )
    if not verdict["ci_crosses_zero"]:
        raise ForecastFreezeError(
            "The conclusion says neither model is established as better, but the "
            "artifact's interval no longer crosses zero."
        )

    direct = key_results["evaluation_mae"]["hgb_direct"]
    residual = key_results["evaluation_mae"]["hgb_residual"]
    if residual <= direct:
        raise ForecastFreezeError(
            "The frozen record preserves that HGB-Residual is worse than HGB-Direct, "
            f"but the artifact reads residual {residual} <= direct {direct}."
        )
    improved = key_results["stability_vs_shrunk_deserved"]["fraction_of_hitters_improved"]
    if improved >= 0.55:
        raise ForecastFreezeError(
            "The frozen record preserves that the advantage is not broad "
            f"(about half of hitters improved), but the artifact reads {improved}."
        )


HGB_FREEZE_SPEC = StageFreezeSpec(
    stage="hgb",
    version=HGB_FREEZE_VERSION,
    conclusion=HGB_CONCLUSION,
    limitations=HGB_LIMITATIONS,
    artifacts=HGB_ARTIFACTS,
    source_modules=HGB_SOURCE_MODULES,
    upstream_manifests=("r1_freeze_manifest.json", "ridge_freeze_manifest.json"),
    key_results=extract_hgb_key_results,
    checks=assert_hgb_conclusion,
    specification=build_hgb_specification,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=FORECAST_OUTPUTS_DIR)
    parser.add_argument("--verify", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.verify:
        result = verify_stage(HGB_FREEZE_SPEC, outputs_dir=args.outputs_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["matches_freeze"] else 1
    manifest = freeze_stage(HGB_FREEZE_SPEC, outputs_dir=args.outputs_dir)
    print(f"verdict: {manifest['conclusion']['verdict']}")
    print(f"manifest sha256: {manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
