"""Contact Forecast Phase 2: authorization, guards, and the frozen contract.

Phase 1 built no door to 2026 on purpose (`forecast.forecast_config`'s
`PHASE_2_ENTRY_POINT_EXISTS = False`, and `assert_forecast_seasons_allowed`
with no override parameter). This module is the separately-named Phase 2
entry point that gate anticipated. It does NOT edit, relax or monkey-patch any
Phase 1 guard: `forecast.forecast_config` still refuses 2026, every frozen
stage still verifies, and Phase 1 code paths remain exactly as sealed.

## What this authorization covers, and what it does not

Authorized: evaluating the ALREADY-FROZEN forecast specification on 2026.

Not authorized, and structurally prevented rather than merely discouraged:
model tuning, feature selection, hyperparameter changes, benchmark changes,
target changes, and dashboard deployment. `FROZEN_CONTRACT` records each
frozen element with the manifest it is loaded from, and the evaluation loads
them from those manifests rather than restating them, so a 2026 result has
nothing to alter.

## 2025 stays sealed

`assert_phase2_seasons_allowed` refuses 2025 with no override, exactly as
Phase 1 did. `assert_phase2_path_allowed` additionally refuses any path
resolving inside a final-evaluation namespace or naming 2025, so the sealed
season cannot be reached by season list or by file path. Phase 2 widens
exactly one thing relative to Phase 1: the prospective 2026 roots become
READABLE. Nothing in this package writes to them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forecast.forecast_config import PHASE_2_GATE

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: The maintainer's explicit authorization -- gate requirement 10, which could
#: only ever be satisfied by a person, in the moment.
PHASE_2_AUTHORIZED = True
PHASE_2_AUTHORIZED_SCOPE = "held-out evaluation of the frozen specification on 2026 only"

#: The season this authorization opens.
PHASE2_EVALUATION_SEASONS: tuple[int, ...] = (2026,)

#: Development seasons the final model and its benchmarks may be fitted on.
#: Through 2024, per the authorization. 2025 is absent and cannot be added.
PHASE2_PERMITTED_TRAINING_SEASONS: tuple[int, ...] = (2022, 2023, 2024)

#: Permanently sealed. No parameter, flag or environment variable in this
#: package resolves it.
PHASE2_FORBIDDEN_SEASONS: tuple[int, ...] = (2025,)

#: Every season this package may touch at all.
PHASE2_ALLOWED_SEASONS: tuple[int, ...] = (
    PHASE2_PERMITTED_TRAINING_SEASONS + PHASE2_EVALUATION_SEASONS
)

#: Roots this package may READ. Read-only: nothing here writes to them.
PHASE2_READABLE_ROOTS: tuple[Path, ...] = (
    PROJECT_ROOT / "data" / "prospective" / "2026",
    PROJECT_ROOT / "outputs" / "prospective",
    PROJECT_ROOT / "artifacts" / "prospective",
    PROJECT_ROOT / "data" / "forecast",
    PROJECT_ROOT / "outputs" / "forecast_research",
)

#: Roots this package may never touch, by path.
PHASE2_FORBIDDEN_ROOTS: tuple[Path, ...] = (
    PROJECT_ROOT / "data" / "final_evaluation",
    PROJECT_ROOT / "outputs" / "final_evaluation",
    PROJECT_ROOT / "artifacts" / "final_evaluation",
)

#: The dedicated Phase 2 namespace, separate from every development artifact.
PHASE2_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "forecast_phase2"

#: Where each frozen element is LOADED from, rather than restated here. A
#: value that had to be retyped could drift from what was sealed; a value read
#: out of a verified manifest cannot.
FROZEN_CONTRACT: dict[str, dict[str, str]] = {
    "model": {
        "value": "D_full_contact_profile",
        "kind": "full-contact-profile ridge",
        "source": "ridge_freeze_manifest.json :: key_results.selected_model.model",
    },
    "feature_definition": {
        "source": "ridge_freeze_manifest.json :: key_results.feature_sets",
        "resolution": "forecast.ridge.resolve_features, unchanged",
    },
    "preprocessing": {
        "value": "StandardScaler fitted on permitted development seasons only",
        "source": "forecast.ridge.fit_ridge, unchanged",
    },
    "regularization": {
        "source": "ridge_freeze_manifest.json :: key_results.selected_alphas",
        "rule": "the frozen alpha is applied; it is never retuned on 2026",
    },
    "target": {
        "value": "target_realized_rv_per_100 over the next 100 resolved eligible BBE",
        "source": "r1_freeze_manifest.json :: specification.targets.primary",
    },
    "benchmarks": {
        "value": "league_mean, shrunk_realized_persistence, shrunk_deserved_persistence",
        "source": "r1_freeze_manifest.json :: specification.baseline_ladder",
    },
    "metrics": {
        "value": "MAE, RMSE, R2, Pearson, Spearman; paired batter-clustered bootstrap",
        "source": "r1_freeze_manifest.json :: specification.metrics",
    },
    "sign_convention": {
        "source": "r1_freeze_manifest.json :: specification.metrics.sign_convention",
        "rule": "delta = challenger - reference; negative favours the challenger",
    },
    "interpretation_rules": {
        "source": "forecast.phase2.phase2_config.SUCCESS_CLASSIFICATION",
        "rule": "prespecified before the 2026 result was computed",
    },
}

#: The prespecified classification, fixed before any 2026 metric existed. MAE
#: is the deciding metric; RMSE, R-squared and the correlations are secondary
#: and may not be used to reclassify a result after MAE has been seen.
SUCCESS_CLASSIFICATION: dict[str, dict[str, str]] = {
    "established_incremental_success": {
        "condition": "delta_MAE < 0 AND the paired 95% CI lies entirely below zero",
        "meaning": (
            "supports incremental predictive value over shrunk deserved on the "
            "held-out 2026 evaluation. Magnitude must still be reported: statistical "
            "significance alone does not make an effect large."
        ),
    },
    "promising_but_inconclusive": {
        "condition": "delta_MAE < 0 but the 95% CI crosses zero",
        "meaning": (
            "2026 points in the same favourable direction but does not establish "
            "incremental superiority."
        ),
    },
    "no_evidence_of_incremental_improvement": {
        "condition": "delta_MAE >= 0 and the 95% CI crosses zero",
        "meaning": (
            "no incremental improvement demonstrated. The model is NOT rescued using "
            "secondary metrics."
        ),
    },
    "evidence_against_incremental_value": {
        "condition": "delta_MAE > 0 AND the paired 95% CI lies entirely above zero",
        "meaning": (
            "the frozen Contact Forecast materially underperformed shrunk deserved in "
            "the held-out test."
        ),
    },
}

#: Things this authorization explicitly does not permit.
PHASE_2_NOT_AUTHORIZED: tuple[str, ...] = (
    "model tuning of any kind",
    "feature selection or engineering",
    "hyperparameter changes, including re-tuning alpha on 2026",
    "benchmark changes",
    "target changes",
    "recalibration after seeing 2026 diagnostics",
    "reopening HGB or testing any further model family",
    "prediction intervals",
    "dashboard deployment",
    "reading 2025 under any circumstance",
)


class Phase2SeasonError(ValueError):
    """Raised when a season outside the Phase 2 authorization is requested."""


class Phase2PathError(ValueError):
    """Raised when a path outside the Phase 2 authorization is requested."""


def assert_phase2_seasons_allowed(
    seasons: int | str | list[int | str] | tuple[int | str, ...],
) -> None:
    """Refuse any season this authorization does not cover.

    2025 is refused first and by name, so the error says WHY rather than
    reporting it as merely unrecognised.

    Raises:
        Phase2SeasonError: If any season is 2025, or is not a Phase 2
            training or evaluation season.
    """
    requested: tuple[int | str, ...]
    requested = (seasons,) if isinstance(seasons, int | str) else tuple(seasons)

    years: set[int] = set()
    for season in requested:
        try:
            years.add(int(season))
        except (TypeError, ValueError) as exc:
            raise Phase2SeasonError(f"Season {season!r} is not an integer year") from exc

    sealed = years & set(PHASE2_FORBIDDEN_SEASONS)
    if sealed:
        raise Phase2SeasonError(
            f"Season(s) {sorted(sealed)} are permanently sealed final-evaluation data. "
            "The Phase 2 authorization opens 2026 ONLY; it does not touch 2025, and "
            "this package has no flag that would."
        )
    unknown = years - set(PHASE2_ALLOWED_SEASONS)
    if unknown:
        raise Phase2SeasonError(
            f"Season(s) {sorted(unknown)} are outside the Phase 2 authorization "
            f"(permitted training {list(PHASE2_PERMITTED_TRAINING_SEASONS)}, "
            f"evaluation {list(PHASE2_EVALUATION_SEASONS)})"
        )


def assert_phase2_path_allowed(path: Path) -> None:
    """Refuse a path inside a sealed namespace, or naming the sealed season.

    Two independent tests: the resolved path must not sit under a
    final-evaluation root, and no path component may name 2025. The second
    catches a sealed file that has been moved somewhere unexpected.

    Raises:
        Phase2PathError: On either violation.
    """
    resolved = Path(path).resolve()
    for root in PHASE2_FORBIDDEN_ROOTS:
        if resolved == root or root in resolved.parents:
            raise Phase2PathError(
                f"{resolved} resolves inside the sealed namespace {root}. Phase 2 "
                "evaluates 2026 and never reads final-evaluation data."
            )
    for part in resolved.parts:
        if "2025" in part:
            raise Phase2PathError(
                f"{resolved} names the sealed season 2025 in path component {part!r}. "
                "2025 is never read, by season list or by path."
            )


def build_authorization_record(
    *, frozen_manifests: dict[str, str], resolved_contract: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The machine-readable Phase 2 authorization, written before 2026 is opened.

    Args:
        frozen_manifests: Stage name -> manifest SHA-256, captured from the
            verified freeze chain BEFORE any 2026 file is read.
        resolved_contract: The frozen values actually loaded from those
            manifests, so the record carries what was used rather than what
            was intended.
    """
    return {
        "authorization": {
            "granted": PHASE_2_AUTHORIZED,
            "scope": PHASE_2_AUTHORIZED_SCOPE,
            "evaluation_seasons_authorized": list(PHASE2_EVALUATION_SEASONS),
            "sealed_seasons_still_forbidden": list(PHASE2_FORBIDDEN_SEASONS),
            "permitted_training_seasons": list(PHASE2_PERMITTED_TRAINING_SEASONS),
            "not_authorized": list(PHASE_2_NOT_AUTHORIZED),
            "no_2026_information_may_alter_any_frozen_element": True,
        },
        "phase_2_gate": {
            "requirements": list(PHASE_2_GATE),
            "entry_point": "forecast.phase2 (separately named, as the gate required)",
            "phase_1_guards_modified": False,
            "phase_1_freezes_reverified_before_opening_2026": True,
        },
        "frozen_contract": FROZEN_CONTRACT,
        "resolved_frozen_contract": resolved_contract or {},
        "frozen_stage_manifests": dict(frozen_manifests),
        "interpretation_rules": SUCCESS_CLASSIFICATION,
        "deciding_metric": (
            "MAE. RMSE, R-squared, Pearson and Spearman are secondary and may not be "
            "used to reclassify a result after MAE has been observed."
        ),
        "conditional_forecast": (
            "The forecast is conditional on the hitter subsequently accumulating 100 "
            "additional resolved eligible batted balls. It does not predict playing "
            "time, injury, roster survival, demotion, or whether the hitter reaches "
            "the horizon at all."
        ),
    }
