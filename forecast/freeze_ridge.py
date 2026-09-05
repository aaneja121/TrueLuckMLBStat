"""Contact Forecast: freeze the completed ridge stage.

Seals the ridge forecasting experiment before any nonlinear model is written:
its report, results artifact, provenance chain back to the frozen R1 package,
the four model specifications, the selected alphas, every ablation, the
bootstrap output and the exact recorded conclusion.

## The conclusion is deliberately not a claim of superiority

The focal 2024 comparison favours the full-profile ridge over shrunk deserved
by a point estimate whose 95% batter-bootstrap interval CROSSES ZERO. That is
a promising development signal and nothing more. `assert_ridge_conclusion`
refuses the freeze if the interval ever stops crossing zero without the prose
being updated deliberately, and the phrase list in
`RIDGE_CONCLUSION["must_not_be_described_as"]` travels into the artifact so a
later reader inherits the limit with the number.

## The negative ablation is part of the frozen record

Adding realized performance on top of deserved performance did NOT improve
prediction (B -> C moved evaluation MAE the wrong way). That is preserved as a
first-class result, checked at freeze time, not left to be rediscovered.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from forecast.forecast_config import FORECAST_OUTPUTS_DIR
from forecast.freeze_r1 import ForecastFreezeError
from forecast.stage_freeze import StageFreezeSpec, freeze_stage, verify_stage

logger = logging.getLogger(__name__)

RIDGE_FREEZE_VERSION = "contact_forecast_ridge_freeze_v1"

#: The exact recorded conclusion for this stage.
RIDGE_CONCLUSION: dict[str, Any] = {
    "verdict": (
        "CONTINUE -- promising incremental development signal, but incremental "
        "forecasting value over shrunk deserved remains statistically inconclusive."
    ),
    "statement": (
        "CONTINUE -- promising incremental development signal, but incremental "
        "forecasting value over shrunk deserved remains statistically inconclusive."
    ),
    "focal_comparison": (
        "MAE(full-profile ridge) - MAE(shrunk_deserved) = -0.1477 on the 2024 "
        "evaluation season, approximately -3.39%, with a 95% batter-bootstrap "
        "confidence interval of [-0.3381, +0.0332]. The interval crosses zero."
    ),
    "evidence_class": "development-season evidence only",
    "must_not_be_described_as": [
        "established superiority over the shrunk-deserved benchmark",
        "a statistically significant improvement -- the interval crosses zero",
        "out-of-sample evidence: 2025 is sealed and 2026 is behind the Phase 2 gate",
        "evidence that combining realized with deserved information helps",
        "a validated production forecasting model",
    ],
    "preserved_negative_results": [
        "Model A (results only) is WORSE than shrunk deserved on 2024 (delta_MAE "
        "+0.1691): realized-performance features alone do not beat the benchmark.",
        "Adding realized performance on top of deserved performance did NOT improve "
        "prediction. Model B (deserved only) -> Model C (results + deserved) moved "
        "evaluation MAE the WRONG way, by +0.0138.",
        "Model B alone is statistically indistinguishable from the shrunk-deserved "
        "benchmark it approximates (delta_MAE -0.0321, interval crosses zero).",
        "Every ridge-versus-shrunk-deserved interval on 2024 crosses zero, including "
        "the best model's.",
    ],
}

RIDGE_LIMITATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "interval_crosses_zero",
        "statement": (
            "The focal delta's 95% batter-clustered bootstrap interval is "
            "[-0.3381, +0.0332] and crosses zero, with 5.9% of replicates favouring "
            "the benchmark. The point estimate favours ridge; the evidence does not "
            "establish that it is better."
        ),
    },
    {
        "id": "gain_localises_to_contact_profile_shape",
        "statement": (
            "The ablation localises the entire gain to the C -> D step (-0.1294), i.e. "
            "to the SHAPE of the contact distribution -- exit velocity, launch angle, "
            "spray, batted-ball mix, handedness -- not to combining realized with "
            "deserved rate summaries, which made prediction slightly worse."
        ),
    },
    {
        "id": "single_evaluation_season",
        "statement": (
            "Fold 2 evaluates on 2024 alone: 283 windows trained on 564. Each hitter "
            "contributes exactly one window, so batter-clustered resampling coincides "
            "with an ordinary bootstrap and batter-balanced MAE coincides with "
            "window-weighted MAE. Neither is additional evidence."
        ),
    },
    {
        "id": "alphas_at_the_ols_limit",
        "statement": (
            "Models A, B and C selected alpha at the grid's lower boundary. Refitting "
            "far beyond the grid moves validation MAE by about 1e-5, so that boundary "
            "is the ordinary-least-squares limit rather than a truncated grid."
        ),
    },
    {
        "id": "ridge_only",
        "statement": (
            "Ridge is the only model family fitted at this stage. Nothing here speaks "
            "to what a nonlinear model would do."
        ),
    },
)

#: The recorded focal numbers, and how far the artifact may drift from the
#: prose before the freeze is refused.
_STATED_DELTA_MAE = -0.1477
_STATED_PCT = -3.39
_STATED_CI = (-0.3381, 0.0332)
_DELTA_TOLERANCE = 5e-4
_PCT_TOLERANCE = 0.01
_CI_TOLERANCE = 5e-4

RIDGE_ARTIFACTS: tuple[str, ...] = ("ridge_results.json", "ridge_report.md")

RIDGE_SOURCE_MODULES: tuple[str, ...] = (
    "forecast/ridge.py",
    "forecast/run_ridge_forecast.py",
    "forecast/ridge_report.py",
    "forecast/stage_freeze.py",
    "forecast/freeze_ridge.py",
)


def extract_ridge_key_results(outputs_dir: Path) -> dict[str, Any]:
    """Everything the ridge conclusion rests on, pulled from its own artifact."""
    results = json.loads((outputs_dir / "ridge_results.json").read_text())
    fold2 = results["fold_2_evaluation"]
    key = results["key_delta"]

    model_mae = {name: fold2["metrics"][f"ridge_{name}"]["mae"] for name in results["feature_sets"]}
    ablation_steps = {
        "A_to_B_realized_versus_deserved": model_mae["B_deserved_only"]
        - model_mae["A_results_only"],
        "B_to_C_adding_realized_on_top_of_deserved": model_mae["C_results_plus_deserved"]
        - model_mae["B_deserved_only"],
        "C_to_D_adding_contact_profile_shape": model_mae["D_full_contact_profile"]
        - model_mae["C_results_plus_deserved"],
    }
    return {
        "focal_comparison": {
            "definition": key["definition"],
            "evaluation_season": key["evaluation_season"],
            "n_evaluation_windows": key["n_evaluation_windows"],
            "best_ridge_model": key["best_ridge_model"],
            "mae_best_ridge": key["mae_best_ridge"],
            "mae_shrunk_deserved": key["mae_shrunk_deserved"],
            "delta_mae": key["delta_mae"],
            "pct_change_mae": key["pct_change_mae"],
            "ci_delta_mae": key["ci_delta_mae"],
            "ci_crosses_zero": key["ci_crosses_zero"],
            "share_replicates_favouring_benchmark": key["share_replicates_favouring_benchmark"],
        },
        "evaluation_mae_by_model": model_mae,
        "benchmark_mae": {
            name: fold2["metrics"][name]["mae"] for name in fold2["benchmarks_available"]
        },
        "ablation_steps_evaluation_mae": ablation_steps,
        "selected_alphas": results["selected_alphas"],
        "selected_model": results["selected_model"],
        "feature_sets": {
            name: spec["declared_features"] for name, spec in results["feature_sets"].items()
        },
        "deltas_vs_shrunk_deserved": {
            challenger: record["delta"]["delta_mae"]
            for challenger, record in results["deltas_on_evaluation_season"][
                "shrunk_deserved_persistence"
            ].items()
        },
        "protocol": results["protocol"],
        "r1_provenance": results["r1_provenance"],
    }


def assert_ridge_conclusion(key_results: dict[str, Any]) -> None:
    """Refuse the freeze if the recorded prose no longer matches the artifact.

    Raises:
        ForecastFreezeError: On any mismatch between the stated focal numbers
            and the artifact, or if the preserved negative ablation has
            silently turned positive.
    """
    focal = key_results["focal_comparison"]
    if abs(focal["delta_mae"] - _STATED_DELTA_MAE) > _DELTA_TOLERANCE:
        raise ForecastFreezeError(
            f"The conclusion states delta_MAE = {_STATED_DELTA_MAE}; the artifact reads "
            f"{focal['delta_mae']}."
        )
    if abs(focal["pct_change_mae"] - _STATED_PCT) > _PCT_TOLERANCE:
        raise ForecastFreezeError(
            f"The conclusion states approximately {_STATED_PCT}%; the artifact reads "
            f"{focal['pct_change_mae']}."
        )
    lower, upper = focal["ci_delta_mae"]
    if abs(lower - _STATED_CI[0]) > _CI_TOLERANCE or abs(upper - _STATED_CI[1]) > _CI_TOLERANCE:
        raise ForecastFreezeError(
            f"The conclusion states a CI of {list(_STATED_CI)}; the artifact reads "
            f"{focal['ci_delta_mae']}."
        )
    if not focal["ci_crosses_zero"]:
        raise ForecastFreezeError(
            "The conclusion calls the result statistically inconclusive, but the "
            "artifact's interval no longer crosses zero. Update the conclusion "
            "deliberately rather than freezing prose that understates the finding."
        )
    step = key_results["ablation_steps_evaluation_mae"]["B_to_C_adding_realized_on_top_of_deserved"]
    if step <= 0:
        raise ForecastFreezeError(
            "The frozen record preserves a NEGATIVE ablation -- adding realized on top "
            f"of deserved did not improve prediction -- but the artifact reads {step}, "
            "which is an improvement. The recorded negative result no longer holds."
        )


RIDGE_FREEZE_SPEC = StageFreezeSpec(
    stage="ridge",
    version=RIDGE_FREEZE_VERSION,
    conclusion=RIDGE_CONCLUSION,
    limitations=RIDGE_LIMITATIONS,
    artifacts=RIDGE_ARTIFACTS,
    source_modules=RIDGE_SOURCE_MODULES,
    upstream_manifests=("r1_freeze_manifest.json",),
    key_results=extract_ridge_key_results,
    checks=assert_ridge_conclusion,
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
        result = verify_stage(RIDGE_FREEZE_SPEC, outputs_dir=args.outputs_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["matches_freeze"] else 1
    manifest = freeze_stage(RIDGE_FREEZE_SPEC, outputs_dir=args.outputs_dir)
    print(f"verdict: {manifest['conclusion']['verdict']}")
    print(f"manifest sha256: {manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
