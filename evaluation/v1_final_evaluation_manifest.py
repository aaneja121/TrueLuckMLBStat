"""Contact Luck v1.0: the pre-evaluation freeze manifest.

Versions 0.2-0.12 are FROZEN. This module builds and validates a
machine-readable manifest that MUST exist, be complete, and match the
repository's current state BEFORE `evaluation/run_v1_final_evaluation.py`
reads or downloads any 2025 data. It computes no predictions and touches no
model, ledger, confidence, aggregation, or public-score logic -- it only
RECORDS the frozen state of that logic (via commit hash + source-file
hashes) and the frozen numeric constants (feature sets, eligibility scope
constants, qualification thresholds, calibration-gate thresholds, bootstrap
settings, seeds) that this codebase already defines elsewhere.

## Why this lives outside `src/mlb_luck_score/`

`evaluation/` is a plain script directory (not part of the installed
`mlb_luck_score` package) -- consistent with the exact paths the task that
produced Version 1.0 asked for. It is NOT covered by `make check`'s default
`ruff format/check src tests` / `mypy src` commands; run `ruff format/check
evaluation` and `mypy evaluation` explicitly (see the Version 1.0 README
section). Tests import it via `tests/conftest.py`'s `pythonpath = ["evaluation"]`
pytest setting (`pyproject.toml`), not package installation.

## What "frozen artifact" means here

This codebase does not persist trained model objects to disk between runs --
`mlb_luck_score.scoring.run_season_aggregation`/`run_public_score` retrain the
four component models FRESH each invocation, deterministically (fixed
`RANDOM_SEED = 42` in every `train_*.py` module), on the frozen 2021-2023
training data. "Frozen model artifact integrity" therefore means: the SOURCE
CODE that defines training/scoring/eligibility/aggregation/presentation
hasn't changed (verified via per-file SHA-256, `FROZEN_ARTIFACT_RELATIVE_PATHS`
below), AND the training/reference DATA files those modules read haven't
changed either (same hashing mechanism, applied to `data/processed/...` and
the real `compare_*.py` detail JSON files Version 0.11/0.12 cite verbatim for
calibration status).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "1.0.0"

#: Every file whose CONTENT this manifest freezes and later re-verifies.
#: Paths are relative to the repository root. Source files first (frozen
#: logic), then the frozen training/reference data and the real, already
#: -computed calibration-status JSON files Version 0.11/0.12 cite verbatim.
FROZEN_ARTIFACT_RELATIVE_PATHS: tuple[str, ...] = (
    "src/mlb_luck_score/config.py",
    "src/mlb_luck_score/eligibility.py",
    "src/mlb_luck_score/features/build_contact_features.py",
    "src/mlb_luck_score/models/train_contact_model.py",
    "src/mlb_luck_score/models/train_opportunity_model.py",
    "src/mlb_luck_score/models/train_advancement_model.py",
    "src/mlb_luck_score/models/outfield_gating.py",
    "src/mlb_luck_score/models/compare_opportunity_models.py",
    "src/mlb_luck_score/models/compare_near_wall_models.py",
    "src/mlb_luck_score/models/compare_near_wall_calibration_gate.py",
    "src/mlb_luck_score/models/compare_infield_opportunity.py",
    "src/mlb_luck_score/models/compare_advancement_models.py",
    "src/mlb_luck_score/data/park_geometry.py",
    "src/mlb_luck_score/data/venue_environment.py",
    "src/mlb_luck_score/data/game_metadata_overrides.py",
    "src/mlb_luck_score/scoring/run_values.py",
    "src/mlb_luck_score/scoring/attribution_ledger.py",
    "src/mlb_luck_score/scoring/run_attribution_ledger.py",
    "src/mlb_luck_score/scoring/component_confidence.py",
    "src/mlb_luck_score/scoring/aggregate_attribution.py",
    "src/mlb_luck_score/scoring/aggregation_uncertainty.py",
    "src/mlb_luck_score/scoring/qualification.py",
    "src/mlb_luck_score/scoring/run_season_aggregation.py",
    "src/mlb_luck_score/scoring/public_score_schema.py",
    "src/mlb_luck_score/scoring/public_score_table.py",
    "src/mlb_luck_score/scoring/leaderboard.py",
    "src/mlb_luck_score/scoring/public_labels.py",
    "src/mlb_luck_score/scoring/run_public_score.py",
    "data/processed/cleaned_development_data_with_sprint_speed.parquet",
    "outputs/tables/opportunity_model_comparison_detail.json",
    "outputs/tables/infield_opportunity_detail.json",
    "outputs/tables/advancement_detail.json",
)


class ManifestError(ValueError):
    """Raised when the manifest cannot be built or fails validation."""


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_hashes(
    repo_root: Path, *, relative_paths: tuple[str, ...] = FROZEN_ARTIFACT_RELATIVE_PATHS
) -> dict[str, str]:
    """`{relative_path: sha256}` for every `relative_paths` entry.

    Raises:
        ManifestError: if any listed path does not exist -- a missing
            frozen artifact must fail loudly, never be silently skipped.
    """
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for rel_path in relative_paths:
        full_path = repo_root / rel_path
        if not full_path.is_file():
            missing.append(rel_path)
            continue
        hashes[rel_path] = compute_file_sha256(full_path)
    if missing:
        raise ManifestError(f"Frozen artifact(s) missing, cannot build manifest: {missing}")
    return hashes


def get_git_commit_hash(repo_root: Path) -> str:
    """The current commit hash -- works identically on a normal branch or a
    detached HEAD (`git rev-parse HEAD` resolves to the checked-out commit
    either way; detached HEAD has no special case here).

    Raises:
        ManifestError: if `repo_root` is not a git repository, has no
            commits yet (unborn HEAD), or `git` itself is unavailable --
            always a clear, typed failure, never a raw `CalledProcessError`
            or `FileNotFoundError` leaking out of this module.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise ManifestError(
            "git executable not found -- cannot determine the repository commit hash."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ManifestError(
            f"Could not determine the repository commit hash at {repo_root} (git rev-parse "
            f"HEAD failed): {exc.stderr.strip() if exc.stderr else exc}. This repository may "
            "have no commits yet, or may not be a git repository at all."
        ) from exc
    commit = result.stdout.strip()
    if not commit:
        raise ManifestError(
            f"git rev-parse HEAD returned no commit hash at {repo_root} -- refusing to build "
            "a manifest without a genuine commit identity."
        )
    return commit


def assert_clean_working_tree(repo_root: Path) -> None:
    """Raise `ManifestError` if `git status --porcelain` reports ANY change
    (staged, unstaged, or untracked) -- Version 1.0's pre-evaluation freeze
    requires a genuinely clean tree, not "just the files I happened to add."
    """
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip():
        raise ManifestError(
            "Working tree is not clean -- refusing to build a pre-evaluation manifest. "
            f"git status --porcelain:\n{result.stdout}"
        )


def _frozen_feature_sets() -> dict[str, list[str]]:
    from mlb_luck_score.features.build_contact_features import (
        ADVANCEMENT_CONTEXT_CATEGORICAL_FEATURES,
        ADVANCEMENT_CONTEXT_NUMERIC_FEATURES,
        ADVANCEMENT_NONLINEAR_CATEGORICAL_FEATURES,
        ADVANCEMENT_NONLINEAR_NUMERIC_FEATURES,
        ADVANCEMENT_SPEED_CATEGORICAL_FEATURES,
        ADVANCEMENT_SPEED_NUMERIC_FEATURES,
        INFIELD_CATEGORICAL_FEATURES,
        INFIELD_NUMERIC_FEATURES,
        NEAR_WALL_CATEGORICAL_FEATURES,
        NEAR_WALL_NUMERIC_FEATURES,
        OPPORTUNITY_CATEGORICAL_FEATURES,
        OPPORTUNITY_NUMERIC_FEATURES,
    )

    return {
        "outfield_opportunity_numeric": list(OPPORTUNITY_NUMERIC_FEATURES),
        "outfield_opportunity_categorical": list(OPPORTUNITY_CATEGORICAL_FEATURES),
        "near_wall_numeric": list(NEAR_WALL_NUMERIC_FEATURES),
        "near_wall_categorical": list(NEAR_WALL_CATEGORICAL_FEATURES),
        "infield_numeric": list(INFIELD_NUMERIC_FEATURES),
        "infield_categorical": list(INFIELD_CATEGORICAL_FEATURES),
        "advancement_context_numeric": list(ADVANCEMENT_CONTEXT_NUMERIC_FEATURES),
        "advancement_context_categorical": list(ADVANCEMENT_CONTEXT_CATEGORICAL_FEATURES),
        "advancement_speed_numeric": list(ADVANCEMENT_SPEED_NUMERIC_FEATURES),
        "advancement_speed_categorical": list(ADVANCEMENT_SPEED_CATEGORICAL_FEATURES),
        "advancement_nonlinear_numeric": list(ADVANCEMENT_NONLINEAR_NUMERIC_FEATURES),
        "advancement_nonlinear_categorical": list(ADVANCEMENT_NONLINEAR_CATEGORICAL_FEATURES),
    }


def _frozen_eligibility_definitions() -> dict[str, list[str]]:
    from mlb_luck_score.eligibility import (
        ADVANCEMENT_ELIGIBLE_BB_TYPES,
        ADVANCEMENT_SAFE_EVENTS,
        INFIELD_CLEAN_OUTCOME_EVENTS,
        OUTFIELD_AIR_BALL_TYPES,
    )

    return {
        "outfield_air_ball_bb_types": sorted(OUTFIELD_AIR_BALL_TYPES),
        "infield_clean_outcome_events": sorted(INFIELD_CLEAN_OUTCOME_EVENTS),
        "advancement_eligible_bb_types": sorted(ADVANCEMENT_ELIGIBLE_BB_TYPES),
        "advancement_safe_events": sorted(ADVANCEMENT_SAFE_EVENTS),
    }


def _frozen_qualification_thresholds() -> dict[str, dict[str, Any]]:
    from mlb_luck_score.scoring.qualification import QUALIFICATION_THRESHOLD_SETS

    return {name: asdict(thresholds) for name, thresholds in QUALIFICATION_THRESHOLD_SETS.items()}


def _frozen_calibration_gate_rules() -> dict[str, Any]:
    from mlb_luck_score.models.compare_advancement_models import (
        ADVANCEMENT_MATERIAL_ECE_ABSOLUTE_THRESHOLD,
    )
    from mlb_luck_score.models.compare_near_wall_calibration_gate import MIN_SUBGROUP_PLAYS
    from mlb_luck_score.models.compare_opportunity_models import MATERIAL_ECE_ABSOLUTE_THRESHOLD

    return {
        "material_ece_absolute_threshold": MATERIAL_ECE_ABSOLUTE_THRESHOLD,
        "advancement_material_ece_absolute_threshold": ADVANCEMENT_MATERIAL_ECE_ABSOLUTE_THRESHOLD,
        "min_subgroup_plays": MIN_SUBGROUP_PLAYS,
    }


def _frozen_bootstrap_settings() -> dict[str, Any]:
    from mlb_luck_score.scoring.aggregation_uncertainty import BootstrapDesign

    design = BootstrapDesign()
    return {
        "method": design.method,
        "resampling_unit": design.resampling_unit,
        "n_reps": design.n_reps,
        "seed": design.seed,
        "alpha": design.alpha,
        "refits_models": design.refits_models,
    }


def _model_versions() -> dict[str, str]:
    return {
        "contact": "baseline_v02",
        "outfield_defense": "measured_contact_only_v07",
        "infield_defense_expected_winner": "infield_hgb_v08",
        "advancement_expected_winner": "advancement_speed_v09",
    }


def _score_version() -> str:
    from mlb_luck_score.scoring.public_score_schema import PUBLIC_SCORE_SCHEMA_VERSION

    return PUBLIC_SCORE_SCHEMA_VERSION


@dataclass(frozen=True)
class FinalEvaluationManifest:
    manifest_version: str
    repository_commit: str
    model_versions: dict[str, str]
    score_version: str
    frozen_feature_sets: dict[str, list[str]]
    frozen_eligibility_definitions: dict[str, list[str]]
    frozen_qualification_thresholds: dict[str, dict[str, Any]]
    frozen_calibration_gate_rules: dict[str, Any]
    frozen_bootstrap_settings: dict[str, Any]
    expected_input_seasons: dict[str, list[int]]
    evaluation_date: str
    deterministic_seeds: dict[str, int]
    artifact_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FinalEvaluationManifest:
        return cls(**data)

    def content_hash(self) -> str:
        """A stable hash of the manifest's own content -- used by `evaluation.
        run_v1_final_evaluation`'s sealing mechanism to detect any change to
        the frozen state between runs.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest(
    repo_root: Path,
    *,
    train_seasons: tuple[int, ...] = (2021, 2022, 2023),
    evaluation_seasons: tuple[int, ...] = (2025,),
    require_clean_tree: bool = True,
) -> FinalEvaluationManifest:
    """Assemble the full pre-evaluation manifest. MUST be called before any
    2025 data is read or downloaded.

    Raises:
        ManifestError: if the working tree is not clean (when `require_
            clean_tree` is True) or any frozen artifact is missing.
    """
    if require_clean_tree:
        assert_clean_working_tree(repo_root)

    return FinalEvaluationManifest(
        manifest_version=MANIFEST_VERSION,
        repository_commit=get_git_commit_hash(repo_root),
        model_versions=_model_versions(),
        score_version=_score_version(),
        frozen_feature_sets=_frozen_feature_sets(),
        frozen_eligibility_definitions=_frozen_eligibility_definitions(),
        frozen_qualification_thresholds=_frozen_qualification_thresholds(),
        frozen_calibration_gate_rules=_frozen_calibration_gate_rules(),
        frozen_bootstrap_settings=_frozen_bootstrap_settings(),
        expected_input_seasons={
            "train_seasons": list(train_seasons),
            "evaluation_seasons": list(evaluation_seasons),
        },
        evaluation_date=datetime.now(UTC).isoformat(),
        deterministic_seeds={"model_random_seed": 42, "bootstrap_seed": 42},
        artifact_hashes=build_artifact_hashes(
            repo_root, relative_paths=FROZEN_ARTIFACT_RELATIVE_PATHS
        ),
    )


def validate_manifest_against_current_state(
    manifest: FinalEvaluationManifest, repo_root: Path
) -> None:
    """Recompute the commit hash and every frozen artifact's hash NOW and
    compare against `manifest`'s stored values.

    Raises:
        ManifestError: on ANY mismatch (commit changed, an artifact is
            missing, or an artifact's content hash differs) or if the
            manifest references a path missing from `FROZEN_ARTIFACT_
            RELATIVE_PATHS` (a manifest built by an older/newer version of
            this module).
    """
    current_commit = get_git_commit_hash(repo_root)
    if current_commit != manifest.repository_commit:
        raise ManifestError(
            f"Commit mismatch: manifest was built at {manifest.repository_commit}, "
            f"repository is currently at {current_commit}"
        )

    manifest_paths = set(manifest.artifact_hashes.keys())
    current_required_paths = set(FROZEN_ARTIFACT_RELATIVE_PATHS)
    added = sorted(current_required_paths - manifest_paths)
    removed = sorted(manifest_paths - current_required_paths)
    if added or removed:
        raise ManifestError(
            "The set of frozen artifacts this module requires has changed since the manifest "
            f"was built -- added (now required, not in manifest): {added or 'none'}; removed "
            f"(in manifest, no longer required): {removed or 'none'}. This manifest was built "
            "by a different version of evaluation.v1_final_evaluation_manifest -- rebuild it."
        )

    current_hashes = build_artifact_hashes(
        repo_root, relative_paths=tuple(manifest.artifact_hashes.keys())
    )
    mismatched = [
        rel_path
        for rel_path, expected_hash in manifest.artifact_hashes.items()
        if current_hashes.get(rel_path) != expected_hash
    ]
    if mismatched:
        raise ManifestError(f"Frozen artifact(s) changed since manifest was built: {mismatched}")
