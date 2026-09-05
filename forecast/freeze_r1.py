"""Contact Forecast R1: freeze the complete baseline package.

Seals the R1 descriptive stage before any forecasting model is written. The
freeze is a CONTENT HASH over everything that produced the result -- the
artifacts, the source modules, the walk-forward ledgers they were built from,
and the interpreter environment -- plus the conclusion, the documented
limitations and the specification amendment history.

Two properties make it a freeze rather than a summary:

  - `forecast_spec.json` is written ONCE. A later run whose specification
    hashes differently is refused, not overwritten, so the Phase 2 gate's
    first requirement ("forecast_spec.json exists, is complete, and its
    content hash is recorded") cannot be quietly satisfied by a moved
    goalpost.
  - `verify_freeze` re-hashes everything and names what drifted. A frozen
    package that no longer matches its own manifest is a finding, not a
    formatting difference.

## The conclusion is checked against the data it describes

`assert_conclusion_matches_artifacts` re-reads the artifacts and confirms the
numbers the prose states. A conclusion that said "approximately 3.9%" over a
package whose focal comparison had moved would fail the freeze rather than
ship. Conservative language is not enough on its own if the number behind it
has drifted.

## What R1 does and does not license

R1 is DEVELOPMENT evidence. 2025 stays sealed, 2026 stays behind the Phase 2
gate, and no held-out prospective season has been opened. The verdict is
recorded as PROCEED with modest positive development evidence, and the
`must_not_be_described_as` list is carried in the artifact so a later reader
inherits the limits along with the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forecast.forecast_config import (
    BASELINE_LADDER,
    CONTACT_STAGE_DISCLAIMER,
    DEFAULT_BOOTSTRAP_ALPHA,
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
    FOCAL_PRESENTATION_CUTOFF,
    FORECAST_ANALYSIS_SEASONS,
    FORECAST_CUTOFFS,
    FORECAST_DATA_DIR,
    FORECAST_DEVELOPMENT_SEASONS,
    FORECAST_HORIZONS,
    FORECAST_OUTPUTS_DIR,
    FORECAST_SEALED_SEASONS,
    FORECAST_SPEC_PATH,
    PHASE_2_GATE,
    PRIMARY_COMPARISON,
    PRIMARY_CUTOFFS,
    PRIMARY_HORIZON,
    PRIMARY_METRICS,
    PRIMARY_TARGET,
    SEASON_FORWARD_FOLDS,
    WALK_FORWARD_TRAIN_SEASONS,
    assert_path_outside_forbidden_namespaces,
    conditional_forecast_language,
)
from forecast.metrics import DELTA_SIGN_CONVENTION
from forecast.targets import assert_no_banned_metric_language

logger = logging.getLogger(__name__)

FREEZE_VERSION = "contact_forecast_r1_freeze_v1"

#: The R1 artifacts, in the order the manifest lists them.
R1_ARTIFACTS: tuple[str, ...] = (
    "r1_prediction_table.parquet",
    "r1_metrics.json",
    "r1_bootstrap.json",
    "r1_regression_direction.json",
    "r1_report.md",
)

#: Every source module that participated in producing them. Hashing the CODE
#: as well as the output is what lets a later reader tell "same numbers" from
#: "same numbers, different definition".
R1_SOURCE_MODULES: tuple[str, ...] = (
    "forecast/forecast_config.py",
    "forecast/windows.py",
    "forecast/targets.py",
    "forecast/features.py",
    "forecast/baselines.py",
    "forecast/metrics.py",
    "forecast/bootstrap.py",
    "forecast/regression_direction.py",
    "forecast/assemble.py",
    "forecast/report.py",
    "forecast/run_r1_analysis.py",
    "forecast/score_development_seasons.py",
    "forecast/freeze_r1.py",
)

# --------------------------------------------------------------------------
# The conclusion, recorded conservatively
# --------------------------------------------------------------------------

R1_CONCLUSION: dict[str, Any] = {
    "verdict": "PROCEED",
    "label": "modest positive development evidence",
    "statement": (
        "PROCEED -- modest positive development evidence. The matched primary "
        "comparison shows shrunk deserved contact performance outperforming "
        "equivalently shrunk realized performance by approximately 3.9% MAE in the "
        "focal 100 -> 100 view. The effect is small in absolute terms, pooled "
        "development evidence favors deserved, individual-season uncertainty remains "
        "substantial, and no held-out prospective season has been opened."
    ),
    "evidence_class": "development-season evidence only",
    "must_not_be_described_as": [
        "proof that the published metric predicts future performance",
        "a validated forecasting result -- no forecasting model was fitted in R1",
        "out-of-sample evidence -- 2025 is sealed and 2026 is behind the Phase 2 gate",
        "a large effect -- the focal advantage is ~3% of the target's own spread",
        "uniform across seasons -- both individual-season intervals cross zero",
    ],
    "contact_stage_disclaimer": CONTACT_STAGE_DISCLAIMER,
}

#: The percentage the conclusion states, and how far the artifact may drift
#: from it before the freeze is refused.
_STATED_PCT_CHANGE_MAE = -3.9
_STATED_PCT_TOLERANCE = 0.1

R1_DOCUMENTED_LIMITATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "primary_comparison_covers_2023_2024_only",
        "statement": (
            "The primary matched comparison (shrunk deserved versus shrunk realized) "
            "contains 2023 and 2024 only. Causal deserved shrinkage cannot be "
            "estimated for 2022: its sole strictly-earlier season is 2021, which has "
            "no walk-forward contact model and therefore no expected contact run "
            "value. 2022 contributes every other ladder rung and all 24 "
            "regression-direction cells."
        ),
    },
    {
        "id": "no_2021_cross_fitting_will_be_added",
        "statement": (
            "No 2021 cross-fitting will be added to R1 merely to fill that missing "
            "season. Cross-fitting a contact model within 2021 would be a new fitted "
            "procedure requiring its own specification and freeze; adding it after "
            "seeing R1's results would be exactly the post-hoc construction this "
            "study's freeze exists to prevent."
        ),
    },
    {
        "id": "focal_view_is_not_a_specification_narrowing",
        "statement": (
            "100 -> 100 is a FOCAL PRESENTATION VIEW, not a prespecified headline "
            "cell. A product-focused amendment proposed designating it the single "
            "headline; adopting that as a prespecification required repository "
            "history to prove the designation preceded the first computed metric, and "
            "it does not -- forecast/ was untracked, so no commit timestamps the "
            "constant, the config module, or any artifact. The original multi-cutoff "
            "specification therefore stands unamended."
        ),
    },
    {
        "id": "every_prespecified_cell_remains_visible",
        "statement": (
            "Every prespecified cutoff/horizon result remains visible in the metrics "
            "artifact, the bootstrap artifact and the report, including cells where "
            "realized beats deserved and cells whose intervals cross zero. The report "
            "is generated by walking the artifacts, so a cell cannot be dropped from "
            "the write-up without being dropped from the data."
        ),
    },
    {
        "id": "conditional_on_reaching_the_horizon",
        "statement": conditional_forecast_language(PRIMARY_HORIZON),
    },
)

SPECIFICATION_AMENDMENTS: tuple[dict[str, Any], ...] = (
    {
        "amendment_id": "A1",
        "date": "2026-09-03",
        "proposal": (
            "Designate 100 BBE observed -> next 100 BBE realized RV/100 as the "
            "study's single frozen headline cell, narrowing PRIMARY_CUTOFFS."
        ),
        "motivation": "product focus -- one cell to lead with",
        "test_applied": (
            "Adoptable as a prespecification if and only if repository history proves "
            "the designation preceded any computed metric."
        ),
        "evidence_examined": [
            "git log -- forecast/ : no commits (directory untracked)",
            "git log -- forecast/forecast_config.py : no commits",
            "git log -S PRIMARY_HEADLINE_CUTOFF --all : no commits",
            "git ls-files outputs/forecast_research/ : no tracked artifacts",
        ],
        "outcome": "NOT ADOPTED",
        "resolution": (
            "100 -> 100 is recorded as a focal presentation view. PRIMARY_CUTOFFS "
            "stands unamended: all four cutoffs at the primary horizon retain equal "
            "standing and no conclusion rests on the focal cell's selection."
        ),
    },
)

# --------------------------------------------------------------------------
# The frozen specification
# --------------------------------------------------------------------------


def build_specification() -> dict[str, Any]:
    """The complete Phase 1 specification, in the shape the Phase 2 gate needs."""
    return {
        "spec_version": FREEZE_VERSION,
        "season_policy": {
            "development_seasons": list(FORECAST_DEVELOPMENT_SEASONS),
            "analysis_seasons": list(FORECAST_ANALYSIS_SEASONS),
            "feature_source_only": [2021],
            "sealed": list(FORECAST_SEALED_SEASONS),
            "phase_2_gated": [2026],
            "walk_forward_train_seasons": {
                str(k): list(v) for k, v in WALK_FORWARD_TRAIN_SEASONS.items()
            },
            "season_forward_folds": [
                {"fit": list(fit), "evaluate": evaluate} for fit, evaluate in SEASON_FORWARD_FOLDS
            ],
        },
        "window_grid": {
            "cutoffs": list(FORECAST_CUTOFFS),
            "horizons": list(FORECAST_HORIZONS),
            "inclusion_rule": "hitter-season needs >= cutoff + horizon resolved eligible BBE",
            "denominator": "production's resolved-BBE denominator (sum of resolved)",
        },
        "targets": {
            "primary": PRIMARY_TARGET,
            "secondary": "target_deserved_rv_per_100",
            "stage": "contact stage (Rc, E0, Rc - E0); never full telescoping Rf",
            "conditional_interpretation": {
                str(h): conditional_forecast_language(h) for h in FORECAST_HORIZONS
            },
        },
        "primary_family": {
            "cutoffs": list(PRIMARY_CUTOFFS),
            "horizon": PRIMARY_HORIZON,
            "none_promoted": True,
        },
        "focal_presentation_view": {
            "cutoff": FOCAL_PRESENTATION_CUTOFF,
            "horizon": PRIMARY_HORIZON,
            "status": "presentation view only, NOT a narrowing of the specification",
        },
        "primary_comparison": {
            "reference": PRIMARY_COMPARISON[0],
            "challenger": PRIMARY_COMPARISON[1],
        },
        "baseline_ladder": list(BASELINE_LADDER),
        "metrics": {
            "primary": list(PRIMARY_METRICS),
            "sign_convention": DELTA_SIGN_CONVENTION,
            "undefined_correlations": (
                "a constant predictor's Pearson and Spearman are recorded as null "
                "with a status string, never as 0"
            ),
        },
        "resampling": {
            "cluster": "batter, globally",
            "paired": True,
            "reps": DEFAULT_BOOTSTRAP_REPS,
            "seed": DEFAULT_BOOTSTRAP_SEED,
            "alpha": DEFAULT_BOOTSTRAP_ALPHA,
        },
        "shrinkage": {
            "estimator": "empirical-Bayes / James-Stein toward a prior-season mean",
            "matched": "the identical estimator fits realized and deserved",
            "fit_rule": "strictly earlier seasons only; re-proved by assert_fit_is_causal",
            "known_property": (
                "shrinkage cannot change rank within a fixed-cutoff cell -- every "
                "hitter has the same BBE count, so the map is affine"
            ),
        },
        "regression_direction": {
            "outcome": "future_realized - current_realized",
            "placebo": (
                "current_deserved permuted across hitters within each cell; "
                "placebo_surprise = current_realized - permuted_current_deserved"
            ),
            "mean_regression_baseline": "outcome ~ current_realized",
            "prespecified_directions": {
                "surprise_slope": "real more negative than placebo",
                "incremental_deserved_slope": "real more positive than placebo",
                "delta_r_squared": "real larger than placebo",
            },
        },
        "exclusions": {
            "unresolved_batted_balls": "field_error / fielders_choice, null outcome_class",
            "structurally_unavailable_rung": (
                "shrunk_deserved_persistence for 2022 -- never imputed"
            ),
        },
        "phase_2_gate": list(PHASE_2_GATE),
    }


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------


def hash_json(payload: Any) -> str:
    """Stable SHA-256 over a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_file(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ForecastFreezeError(RuntimeError):
    """Raised when the R1 package cannot be frozen, or has drifted from its freeze."""


def _hash_tree(paths: dict[str, Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.exists():
            raise ForecastFreezeError(
                f"Cannot freeze: {name} is missing at {path}. Run `make run-forecast-r1` first."
            )
        hashes[name] = hash_file(path)
    return hashes


# --------------------------------------------------------------------------
# Conclusion / artifact agreement
# --------------------------------------------------------------------------


def extract_key_results(outputs_dir: Path) -> dict[str, Any]:
    """The handful of numbers the conclusion rests on, pulled from the artifacts."""
    metrics = json.loads((outputs_dir / "r1_metrics.json").read_text())
    bootstrap = json.loads((outputs_dir / "r1_bootstrap.json").read_text())
    regression = json.loads((outputs_dir / "r1_regression_direction.json").read_text())

    challenger = PRIMARY_COMPARISON[1]
    focal_key = f"{PRIMARY_TARGET}|K{FOCAL_PRESENTATION_CUTOFF}_H{PRIMARY_HORIZON}"
    focal = metrics["cells"][focal_key]["comparisons"][challenger]
    boot_key = f"{PRIMARY_TARGET}|{challenger}_vs_{PRIMARY_COMPARISON[0]}"
    focal_boot = bootstrap["comparisons"][boot_key]["cells"][
        f"K{FOCAL_PRESENTATION_CUTOFF}_H{PRIMARY_HORIZON}"
    ]

    family: dict[str, Any] = {}
    for cutoff in PRIMARY_CUTOFFS:
        key = f"{PRIMARY_TARGET}|K{cutoff}_H{PRIMARY_HORIZON}"
        comparison = metrics["cells"][key]["comparisons"][challenger]
        cell_boot = bootstrap["comparisons"][boot_key]["cells"][f"K{cutoff}_H{PRIMARY_HORIZON}"][
            "pooled"
        ]["delta_mae"]
        family[f"K{cutoff}_H{PRIMARY_HORIZON}"] = {
            "delta_mae": comparison["pooled_window_weighted"]["delta_mae"],
            "pct_change_mae": comparison["pooled_window_weighted"]["pct_change_mae"],
            "ci_lower": cell_boot["ci_lower"],
            "ci_upper": cell_boot["ci_upper"],
            "ci_crosses_zero": cell_boot["ci_crosses_zero"],
            "direction_mae": comparison["pooled_window_weighted"]["direction_mae"],
        }

    placebo_cells = [
        cell
        for cell in regression["regression_direction_by_cell"].values()
        if cell.get("status") == "fitted"
    ]
    return {
        "focal_primary_comparison": {
            "cell": f"K{FOCAL_PRESENTATION_CUTOFF}_H{PRIMARY_HORIZON}",
            "reference": PRIMARY_COMPARISON[0],
            "challenger": challenger,
            "seasons": focal["seasons_in_comparison"],
            "seasons_excluded": focal["seasons_excluded_for_missing_rung"],
            "n_paired_windows": focal["n_paired_windows"],
            "delta_mae": focal["pooled_window_weighted"]["delta_mae"],
            "pct_change_mae": focal["pooled_window_weighted"]["pct_change_mae"],
            "delta_rmse": focal["pooled_window_weighted"]["delta_rmse"],
            "ci_delta_mae": [
                focal_boot["pooled"]["delta_mae"]["ci_lower"],
                focal_boot["pooled"]["delta_mae"]["ci_upper"],
            ],
            "macro_delta_mae": focal["macro_average_of_season_deltas"]["macro_delta_mae"],
            "batter_balanced_delta_mae": focal["batter_balanced"]["delta"]["delta_mae"],
            "abs_delta_mae_as_fraction_of_target_sd": focal["practical_magnitude"][
                "abs_delta_mae_as_fraction_of_target_sd"
            ],
            "season_intervals_cross_zero": {
                season: entry["delta_mae"]["ci_crosses_zero"]
                for season, entry in focal_boot["by_season"].items()
            },
        },
        "primary_family_at_primary_horizon": family,
        "regression_direction": {
            "n_cells_fitted": len(placebo_cells),
            "n_cells_real_more_negative_than_placebo_mean": sum(
                1
                for cell in placebo_cells
                if cell["real_surprise_regression"]["slope_surprise"]
                < cell["placebo"]["surprise_slope"]["placebo_mean"]
            ),
            "max_p_lower_surprise_slope": max(
                cell["placebo"]["surprise_slope"]["p_lower_real_more_negative"]
                for cell in placebo_cells
            ),
            "placebo_share_of_raw_slope_note": (
                "the placebo mean is far from zero, so a substantial share of the raw "
                "surprise slope is the mechanical coupling rather than contact "
                "information; per-cell values are in r1_regression_direction.json"
            ),
        },
    }


def assert_conclusion_matches_artifacts(key_results: dict[str, Any]) -> None:
    """Refuse a freeze whose prose no longer matches its own numbers.

    Raises:
        ForecastFreezeError: If the focal percentage the conclusion states has
            drifted, if the comparison's season coverage is not the documented
            2023-2024, or if the stated direction has flipped.
    """
    focal = key_results["focal_primary_comparison"]
    pct = focal["pct_change_mae"]
    if pct is None or abs(pct - _STATED_PCT_CHANGE_MAE) > _STATED_PCT_TOLERANCE:
        raise ForecastFreezeError(
            f"The conclusion states approximately {_STATED_PCT_CHANGE_MAE}% MAE but the "
            f"focal comparison now reads {pct}. Update the conclusion deliberately "
            "rather than freezing prose that no longer describes the data."
        )
    if focal["delta_mae"] >= 0:
        raise ForecastFreezeError(
            f"The conclusion says deserved outperforms, but delta_mae is "
            f"{focal['delta_mae']} (>= 0 favours realized)."
        )
    if focal["seasons"] != [2023, 2024]:
        raise ForecastFreezeError(
            f"The documented limitation says the primary comparison covers 2023-2024; "
            f"the artifact says {focal['seasons']}."
        )
    if focal["seasons_excluded"] != [2022]:
        raise ForecastFreezeError(
            f"2022 is documented as excluded from the primary comparison; the artifact "
            f"says {focal['seasons_excluded']}."
        )


# --------------------------------------------------------------------------
# Freeze / verify
# --------------------------------------------------------------------------


def _artifact_paths(outputs_dir: Path) -> dict[str, Path]:
    return {name: outputs_dir / name for name in R1_ARTIFACTS}


def _source_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {name: root / name for name in R1_SOURCE_MODULES}


def _ledger_paths(data_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for season in FORECAST_ANALYSIS_SEASONS:
        paths[f"walk_forward_ledger_{season}.parquet"] = (
            data_dir / f"walk_forward_ledger_{season}.parquet"
        )
        paths[f"walk_forward_manifest_{season}.json"] = (
            data_dir / f"walk_forward_manifest_{season}.json"
        )
    return paths


def _environment() -> dict[str, str]:
    import numpy
    import pandas
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
    }


def freeze(
    *,
    outputs_dir: Path = FORECAST_OUTPUTS_DIR,
    data_dir: Path = FORECAST_DATA_DIR,
    force_respecify: bool = False,
) -> dict[str, Any]:
    """Hash the R1 package, check the conclusion against it, and write the freeze.

    Args:
        outputs_dir: Where the R1 artifacts live and the freeze is written.
        data_dir: Where the walk-forward ledgers live.
        force_respecify: Permit overwriting an existing `forecast_spec.json`
            whose content hash differs. Deliberately explicit -- the spec is
            written once, and a silent rewrite would hollow out the Phase 2
            gate's first requirement.

    Raises:
        ForecastFreezeError: If an artifact is missing, if the conclusion no
            longer matches the artifacts, or if the specification would change
            without `force_respecify`.
    """
    assert_path_outside_forbidden_namespaces(outputs_dir)
    assert_path_outside_forbidden_namespaces(data_dir)

    key_results = extract_key_results(outputs_dir)
    assert_conclusion_matches_artifacts(key_results)
    assert_no_banned_metric_language(
        R1_CONCLUSION["statement"] + " " + " ".join(R1_CONCLUSION["must_not_be_described_as"])
    )

    specification = build_specification()
    specification_hash = hash_json(specification)
    spec_path = outputs_dir / FORECAST_SPEC_PATH.name
    if spec_path.exists():
        existing = json.loads(spec_path.read_text())
        if hash_json(existing) != specification_hash and not force_respecify:
            raise ForecastFreezeError(
                f"{spec_path} already exists with a DIFFERENT content hash. The Phase 1 "
                "specification is written once. Re-run with force_respecify=True only "
                "as a deliberate, recorded respecification."
            )
    spec_path.write_text(json.dumps(specification, indent=2, sort_keys=True))

    manifest: dict[str, Any] = {
        "freeze_version": FREEZE_VERSION,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "conclusion": R1_CONCLUSION,
        "documented_limitations": [dict(item) for item in R1_DOCUMENTED_LIMITATIONS],
        "specification_amendments": [dict(item) for item in SPECIFICATION_AMENDMENTS],
        "specification": specification,
        "specification_sha256": specification_hash,
        "specification_path": str(spec_path),
        "key_results": key_results,
        "artifact_sha256": _hash_tree(_artifact_paths(outputs_dir)),
        "source_module_sha256": _hash_tree(_source_paths()),
        "input_ledger_sha256": _hash_tree(_ledger_paths(data_dir)),
        "environment": _environment(),
        "phase_2_gate": {
            "requirements": list(PHASE_2_GATE),
            "entry_point_exists": False,
            "sealed_seasons": list(FORECAST_SEALED_SEASONS),
        },
    }
    manifest["manifest_sha256"] = hash_json(
        {k: v for k, v in manifest.items() if k != "frozen_at_utc"}
    )

    manifest_path = outputs_dir / "r1_freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (outputs_dir / "r1_frozen_conclusion.md").write_text(render_conclusion(manifest))
    logger.info("froze R1: %s", manifest_path)
    return manifest


def verify_freeze(
    *, outputs_dir: Path = FORECAST_OUTPUTS_DIR, data_dir: Path = FORECAST_DATA_DIR
) -> dict[str, Any]:
    """Re-hash the package and name anything that drifted from its freeze."""
    manifest_path = outputs_dir / "r1_freeze_manifest.json"
    if not manifest_path.exists():
        raise ForecastFreezeError(f"No freeze manifest at {manifest_path}")
    frozen = json.loads(manifest_path.read_text())

    drift: dict[str, list[str]] = {}
    for label, recorded, current in (
        ("artifacts", frozen["artifact_sha256"], _hash_tree(_artifact_paths(outputs_dir))),
        ("source_modules", frozen["source_module_sha256"], _hash_tree(_source_paths())),
        ("input_ledgers", frozen["input_ledger_sha256"], _hash_tree(_ledger_paths(data_dir))),
    ):
        changed = [name for name, digest in recorded.items() if current.get(name) != digest]
        if changed:
            drift[label] = sorted(changed)

    specification_changed = hash_json(build_specification()) != frozen["specification_sha256"]
    if specification_changed:
        drift.setdefault("specification", []).append("build_specification() output changed")

    return {
        "frozen_at_utc": frozen["frozen_at_utc"],
        "manifest_sha256": frozen["manifest_sha256"],
        "matches_freeze": not drift,
        "drift": drift,
    }


def render_conclusion(manifest: dict[str, Any]) -> str:
    """The one-page frozen conclusion, generated from the manifest."""
    focal = manifest["key_results"]["focal_primary_comparison"]
    lines = [
        "# Contact Forecast R1 -- FROZEN",
        "",
        f"Frozen {manifest['frozen_at_utc']}.",
        f"Manifest SHA-256: `{manifest['manifest_sha256']}`.",
        f"Specification SHA-256: `{manifest['specification_sha256']}`.",
        "",
        "## Conclusion",
        "",
        f"**{manifest['conclusion']['verdict']} -- {manifest['conclusion']['label']}.**",
        "",
        manifest["conclusion"]["statement"],
        "",
        "### This result must NOT be described as",
        "",
    ]
    lines += [f"- {item}" for item in manifest["conclusion"]["must_not_be_described_as"]]
    lines += [
        "",
        manifest["conclusion"]["contact_stage_disclaimer"],
        "",
        "## The numbers behind that sentence",
        "",
        f"- Focal view: {focal['cell']}, `{focal['challenger']}` versus `{focal['reference']}`.",
        f"- Seasons: {focal['seasons']} (excluded: {focal['seasons_excluded']}); "
        f"{focal['n_paired_windows']} paired windows.",
        f"- delta_MAE {focal['delta_mae']:.4f} "
        f"(95% CI [{focal['ci_delta_mae'][0]:.4f}, {focal['ci_delta_mae'][1]:.4f}]), "
        f"{focal['pct_change_mae']:.2f}%.",
        f"- Macro-average of season deltas: {focal['macro_delta_mae']:.4f}; "
        f"batter-balanced: {focal['batter_balanced_delta_mae']:.4f}.",
        f"- |delta_MAE| is {focal['abs_delta_mae_as_fraction_of_target_sd']:.4f} of the "
        "target's own SD -- small.",
        "- Season intervals crossing zero: "
        + ", ".join(
            f"{season}: {'yes' if crosses else 'no'}"
            for season, crosses in sorted(focal["season_intervals_cross_zero"].items())
        )
        + ".",
        "",
        "Negative delta_MAE favours deserved. "
        + manifest["specification"]["metrics"]["sign_convention"]["definition"],
        "",
        "## Documented limitations",
        "",
    ]
    for item in manifest["documented_limitations"]:
        lines += [f"- **{item['id']}** -- {item['statement']}", ""]
    lines += ["## Specification amendment history", ""]
    for amendment in manifest["specification_amendments"]:
        lines += [
            f"### {amendment['amendment_id']} ({amendment['date']}) -- **{amendment['outcome']}**",
            "",
            f"- Proposal: {amendment['proposal']}",
            f"- Motivation: {amendment['motivation']}",
            f"- Test applied: {amendment['test_applied']}",
            "- Evidence examined:",
            *[f"  - `{item}`" for item in amendment["evidence_examined"]],
            f"- Resolution: {amendment['resolution']}",
            "",
        ]
    lines += [
        "## Phase 2 gate (unchanged, and not opened)",
        "",
        *[f"{i + 1}. {item}" for i, item in enumerate(manifest["phase_2_gate"]["requirements"])],
        "",
        f"Sealed seasons: {manifest['phase_2_gate']['sealed_seasons']}. "
        f"Phase 2 entry point exists: {manifest['phase_2_gate']['entry_point_exists']}.",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=FORECAST_OUTPUTS_DIR)
    parser.add_argument("--data-dir", type=Path, default=FORECAST_DATA_DIR)
    parser.add_argument("--verify", action="store_true", help="Verify an existing freeze.")
    parser.add_argument(
        "--force-respecify",
        action="store_true",
        help="Deliberately overwrite a differing forecast_spec.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    if args.verify:
        result = verify_freeze(outputs_dir=args.outputs_dir, data_dir=args.data_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["matches_freeze"] else 1
    manifest = freeze(
        outputs_dir=args.outputs_dir,
        data_dir=args.data_dir,
        force_respecify=args.force_respecify,
    )
    print(f"verdict: {manifest['conclusion']['verdict']} -- {manifest['conclusion']['label']}")
    print(f"manifest sha256: {manifest['manifest_sha256']}")
    print(f"specification sha256: {manifest['specification_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
