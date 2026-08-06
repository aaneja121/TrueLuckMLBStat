"""Contact Luck v1.0: the dedicated, sealed final-evaluation entry point.

Versions 0.2-0.12 are FROZEN. This is the ONLY code path in this repository
authorized to read or download 2025 data -- see CLAUDE.md "The one narrow
exception: the sealed Version 1.0 final evaluation" for the policy this
module implements. Read that section before changing anything here.

## What makes this the "one narrow exception," concretely

- `FINAL_EVALUATION_2025_DATE_RANGE` below is LOCAL to this module -- it is
  never added to `mlb_luck_score.config.MLB_REGULAR_SEASON_DATE_RANGES` or
  `DEVELOPMENT_SEASONS`, and it is a preliminary estimate (see its own
  docstring) that MUST be verified against the real 2025 schedule before any
  real download.
- 2025 ingestion (`ingest_2025_raw_statcast`, `ingest_2025_game_metadata`,
  `ingest_2025_sprint_speed`, `build_2025_evaluation_dataset`) calls the
  low-level fetch functions (`download_statcast_range`, the MLB Stats API
  schedule/venue fetchers, `fetch_season_sprint_speed`) DIRECTLY -- never
  through `download_development_data`/`download_game_metadata`/`download_
  sprint_speed`'s own season-list CLIs, which hard-require a `MLB_REGULAR_
  SEASON_DATE_RANGES` entry that must never exist for 2025. Every ingestion
  function requires a `FinalEvaluationAuthorization` OBJECT (never a bare
  `bool`) minted ONLY by `run_pre_evaluation_guards` (an unforgeable,
  process-local token -- see `_mint_authorization`/`_require_authorization`),
  and independently calls `mlb_luck_score.config.assert_seasons_allowed(...,
  allow_final_evaluation=True)` itself.
- All 2025 raw/derived files live under `FINAL_EVALUATION_RAW_DIR`/
  `FINAL_EVALUATION_OUTPUTS_DIR`/`FINAL_EVALUATION_ARTIFACTS_DIR`
  (`data/final_evaluation/2025/`, `outputs/final_evaluation/v1/`,
  `artifacts/final_evaluation/v1/`) -- never `data/raw`, `data/processed`,
  `outputs/tables`, or `artifacts/`, so no development runner can discover
  2025 data by accident.
- Training reuses the SAME frozen model choices, training seasons
  (`TRAIN_SEASONS`, 2021-2023, unchanged), and order of operations as
  `mlb_luck_score.scoring.run_season_aggregation.build_player_season_report`,
  but that function is NOT called or modified here -- its own docstring says
  it has "no `--allow-final-evaluation` flag ... on purpose" and is a
  season-hardcoded wrapper not meant to be extended for 2025. This module
  duplicates that glue (`train_and_score_2025`), exactly the same pattern
  `run_season_aggregation` itself used relative to `run_attribution_ledger`.

## Sealing

A completed run writes `EvaluationSeal` to `FINAL_EVALUATION_ARTIFACTS_DIR /
"seal.json"`. `run_final_evaluation` refuses to run again once a seal exists
unless the caller passes `acknowledge_defect_fix=True` and a non-empty
`defect_description` (CLI: `--acknowledge-defect-fix --defect-description
"..."`) -- this is for a genuine, narrowly-scoped defect fix (e.g. a bug in
this orchestration script itself found after sealing), never for tuning,
rerunning to see a different number, or "just checking." A defect-fix rerun
archives (never overwrites) the prior run's outputs/artifacts first.

## What this module does NOT do

It never trains, tunes, recalibrates, or changes any threshold based on what
the 2025 numbers look like -- every model choice, feature set, eligibility
rule, qualification threshold, and calibration-gate rule is read from the
existing frozen code (`evaluation.v1_final_evaluation_manifest` verifies this
via commit hash + artifact hashing before ingestion is even authorized). It
makes no adoption decision for the near-wall specialist (Version 0.7 stays
frozen regardless of 2025 performance, per CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from v1_distribution_shift import compare_development_vs_evaluation, summarize_flags
from v1_final_evaluation_manifest import (
    FinalEvaluationManifest,
    ManifestError,
    build_manifest,
    compute_file_sha256,
    validate_manifest_against_current_state,
)
from v1_final_report import (
    OutcomeInputs,
    assemble_final_report,
    classify_final_evaluation_outcome,
    summarize_2025_stability,
    summarize_public_score_evaluation,
)
from v1_system_evaluation import (
    evaluate_advancement_model,
    evaluate_contact_model,
    evaluate_infield_model,
    evaluate_near_wall_specialist,
    evaluate_open_field_outfield_model,
    verify_advancement_eligibility,
    verify_event_ids,
    verify_no_experimental_components_in_official_total,
    verify_play_level_accounting,
    verify_qualification_and_ranking_contract,
    verify_reproducibility,
    verify_routing_exclusivity,
    verify_season_level_accounting,
)

from mlb_luck_score.config import (
    PROJECT_ROOT,
    TRAIN_SEASONS,
    assert_seasons_allowed,
    development_raw_path,
    game_metadata_path,
    sprint_speed_path,
)
from mlb_luck_score.data.clean_development_data import clean_development_data
from mlb_luck_score.data.download_game_metadata import (
    DEFAULT_CHUNK_DAYS as _GAME_METADATA_DEFAULT_CHUNK_DAYS,
)
from mlb_luck_score.data.download_game_metadata import (
    DEFAULT_MAX_RETRIES as _GAME_METADATA_DEFAULT_MAX_RETRIES,
)
from mlb_luck_score.data.download_game_metadata import (
    DEFAULT_RETRY_BACKOFF_SECONDS as _GAME_METADATA_DEFAULT_BACKOFF_SECONDS,
)
from mlb_luck_score.data.download_game_metadata import (
    GAME_METADATA_COLUMNS,
    _fetch_schedule_chunk,
    _parse_schedule_game,
    compute_neutral_site_flags,
    fetch_venue_details,
)
from mlb_luck_score.data.download_sprint_speed import (
    DEFAULT_MIN_OPPORTUNITIES,
    fetch_season_sprint_speed,
)
from mlb_luck_score.data.download_statcast import (
    DEFAULT_CHUNK_DAYS,
    DEFAULT_MAX_RETRIES,
    build_date_chunks,
    download_statcast_range,
    save_dataframe,
)
from mlb_luck_score.data.join_park_geometry import build_geometry_join_report, join_park_geometry
from mlb_luck_score.data.join_sprint_speed import (
    build_sprint_speed_join_report,
    join_sprint_speed,
    load_sprint_speed,
)
from mlb_luck_score.data.join_venue_metadata import (
    build_venue_join_report,
    join_venue_metadata,
    load_game_metadata,
)
from mlb_luck_score.eligibility import (
    add_advancement_eligibility,
    add_infield_opportunity_eligibility,
    add_outfield_opportunity_eligibility,
    compute_eligibility,
)
from mlb_luck_score.features.build_contact_features import (
    ADVANCEMENT_CONTACT_PROBABILITY_FEATURES,
    add_advancement_features,
    add_opportunity_features_by_domain,
)
from mlb_luck_score.models.compare_advancement_models import (
    fit_advancement_contact_model,
    get_advancement_rows,
    run_advancement_model_selection,
)
from mlb_luck_score.models.compare_infield_opportunity import run_infield_model_selection
from mlb_luck_score.models.compare_near_wall_models import (
    get_near_wall_rows,
    run_near_wall_model_selection,
)
from mlb_luck_score.models.train_advancement_model import TrainedAdvancementModel
from mlb_luck_score.models.train_contact_model import TrainedModel, train_model
from mlb_luck_score.models.train_opportunity_model import (
    TrainedOpportunityModel,
    train_opportunity_model,
)
from mlb_luck_score.scoring.aggregate_attribution import (
    aggregate_to_batter_season,
    verify_season_identity,
)
from mlb_luck_score.scoring.aggregation_uncertainty import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_N_BOOTSTRAP_REPS,
    BootstrapDesign,
    bootstrap_batter_season_intervals,
)
from mlb_luck_score.scoring.attribution_ledger import build_attribution_ledger
from mlb_luck_score.scoring.component_confidence import build_play_level_confidence
from mlb_luck_score.scoring.leaderboard import (
    assign_official_ranks,
    least_favorable_leaderboard,
    most_favorable_leaderboard,
)
from mlb_luck_score.scoring.public_score_table import build_public_score_table
from mlb_luck_score.scoring.qualification import (
    assign_qualification_status,
    summarize_qualification_counts,
)
from mlb_luck_score.scoring.run_season_aggregation import SeasonAggregationArtifacts

logger = logging.getLogger(__name__)

EVALUATION_SEASON = 2025

#: REVIEWED inclusive 2025 MLB regular-season date range. Start date is
#: 2025-03-18, not the usual domestic Opening Day, because the Dodgers-Cubs
#: Tokyo Series (2025-03-18/19 at the Tokyo Dome) consisted of OFFICIAL 2025
#: regular-season games -- excluding them would silently drop real regular
#: -season data, exactly the kind of omission `missing_dates_within_range`
#: in `ingest_2025_raw_statcast`'s provenance record is meant to catch. End
#: date 2025-09-28 is the final scheduled regular-season date. Still LOCAL
#: to this module (see module docstring) -- never added to `mlb_luck_score.
#: config.MLB_REGULAR_SEASON_DATE_RANGES`/`DEVELOPMENT_SEASONS`.
FINAL_EVALUATION_2025_DATE_RANGE: tuple[str, str] = ("2025-03-18", "2025-09-28")

#: Isolated namespace for every 2025 file this module reads or writes --
#: never `data/raw`, `data/processed`, `outputs/tables`, or `artifacts/`.
FINAL_EVALUATION_RAW_DIR = PROJECT_ROOT / "data" / "final_evaluation" / "2025"
FINAL_EVALUATION_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "final_evaluation" / "v1"
FINAL_EVALUATION_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "final_evaluation" / "v1"
SEAL_PATH = FINAL_EVALUATION_ARTIFACTS_DIR / "seal.json"

#: Sibling of `FINAL_EVALUATION_ARTIFACTS_DIR` (NOT nested inside it) so a
#: defect-fix archive copy can never try to copy a directory into its own
#: subdirectory -- gives defect-fix runs their own namespace/audit trail,
#: separate from both the live `v1/` outputs and any prior sealed run.
ARCHIVE_ROOT = PROJECT_ROOT / "artifacts" / "final_evaluation" / "archived_runs"

#: The frozen development input Version 0.11/0.12 already use -- read-only
#: here, never written to.
DEVELOPMENT_INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "cleaned_development_data_with_sprint_speed.parquet"
)


class FinalEvaluationError(ValueError):
    """Raised when a Version 1.0 final-evaluation precondition is violated."""


class SealError(FinalEvaluationError):
    """Raised when sealing state prevents (re)running the final evaluation."""


class SchemaDriftError(FinalEvaluationError):
    """Raised when 2025 raw data does not match the schema every downstream
    step (`clean_batted_balls`, eligibility, feature engineering) assumes.
    """


class NamespaceViolationError(FinalEvaluationError):
    """Raised when a path would read/write outside the isolated Version 1.0
    final-evaluation namespace (`FINAL_EVALUATION_RAW_DIR`/`_OUTPUTS_DIR`/
    `_ARTIFACTS_DIR`) -- catches both a typo'd path and a path-traversal
    attempt (`..` segments), since both resolve to the same real check.
    """


def _assert_within_namespace(path: Path, namespace_root: Path, *, label: str) -> None:
    resolved = path.resolve()
    root = namespace_root.resolve()
    if not resolved.is_relative_to(root):
        raise NamespaceViolationError(
            f"{label} path {resolved} is not inside the isolated Version 1.0 namespace "
            f"{root} -- refusing to read or write outside data/final_evaluation, outputs/"
            "final_evaluation, or artifacts/final_evaluation. This guard exists so a typo'd "
            "or maliciously crafted path (including '..' traversal) can never make 2025 "
            "ingestion touch a development cache."
        )


# ---------------------------------------------------------------------------
# Unforgeable final-evaluation authorization
# ---------------------------------------------------------------------------

#: Tokens minted by `run_pre_evaluation_guards` -- the ONLY function allowed
#: to add to this set. `_require_authorization` checks membership, so a
#: `FinalEvaluationAuthorization` constructed directly (with a fabricated
#: token) rather than obtained from `run_pre_evaluation_guards` is rejected.
#: Process-local by design: this is a within-process guard against a caller
#: skipping the guards, not a cross-process security boundary.
_ISSUED_AUTHORIZATION_TOKENS: set[str] = set()


@dataclass(frozen=True)
class FinalEvaluationAuthorization:
    """An unforgeable-in-practice token proving `run_pre_evaluation_guards`
    actually ran and passed. Ingestion functions require this OBJECT (never
    a bare `bool`) -- see `_mint_authorization`/`_require_authorization`.
    """

    token: str
    manifest_content_hash: str
    granted_at: str


def _mint_authorization(manifest: FinalEvaluationManifest) -> FinalEvaluationAuthorization:
    import secrets

    token = secrets.token_hex(16)
    _ISSUED_AUTHORIZATION_TOKENS.add(token)
    return FinalEvaluationAuthorization(
        token=token,
        manifest_content_hash=manifest.content_hash(),
        granted_at=datetime.now(UTC).isoformat(),
    )


# ---------------------------------------------------------------------------
# Sealing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationSeal:
    manifest_content_hash: str
    report_content_hash: str
    sealed_at: str
    repository_commit: str
    defect_fix_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_existing_seal(seal_path: Path = SEAL_PATH) -> EvaluationSeal | None:
    """`None` only when no seal file exists at all. A seal file that exists
    but is corrupted (invalid JSON) or incomplete (missing a required field)
    fails safely with `SealError` -- it is NEVER treated as "no prior seal"
    and silently overwritten, since that would defeat the entire point of
    sealing.
    """
    if not seal_path.exists():
        return None
    try:
        data = json.loads(seal_path.read_text())
    except json.JSONDecodeError as exc:
        raise SealError(
            f"Seal file at {seal_path} exists but is not valid JSON -- refusing to treat this "
            "as 'no prior seal.' Investigate and repair or deliberately move the corrupted "
            "file aside before proceeding; never let this evaluation silently overwrite it."
        ) from exc
    if not isinstance(data, dict):
        raise SealError(
            f"Seal file at {seal_path} does not contain a JSON object -- refusing to treat "
            "this as 'no prior seal.'"
        )
    try:
        return EvaluationSeal(**data)
    except TypeError as exc:
        raise SealError(
            f"Seal file at {seal_path} exists but is missing or has extra field(s) for a "
            f"valid EvaluationSeal ({exc}) -- refusing to treat this as 'no prior seal.'"
        ) from exc


def assert_no_unacknowledged_prior_seal(
    seal_path: Path = SEAL_PATH,
    *,
    acknowledge_defect_fix: bool,
    defect_description: str = "",
) -> EvaluationSeal | None:
    """Refuse to run if a prior sealed evaluation exists, unless this run
    explicitly acknowledges it is a defect fix (with a non-empty description).
    """
    existing = read_existing_seal(seal_path)
    if existing is None:
        return None
    if not acknowledge_defect_fix:
        raise SealError(
            f"A sealed final evaluation already exists at {seal_path} (sealed_at="
            f"{existing.sealed_at}). Re-running requires acknowledge_defect_fix=True (CLI: "
            "--acknowledge-defect-fix) and a defect_description -- never for tuning, a "
            "routine rerun, or 'just checking.'"
        )
    if not defect_description.strip():
        raise SealError(
            "acknowledge_defect_fix=True requires a non-empty defect_description explaining "
            "exactly what was wrong and fixed."
        )
    return existing


def archive_prior_outputs(
    outputs_dir: Path = FINAL_EVALUATION_OUTPUTS_DIR,
    artifacts_dir: Path = FINAL_EVALUATION_ARTIFACTS_DIR,
) -> Path:
    """Copy (never delete) prior run outputs/artifacts into a timestamped
    archive directory before a defect-fix rerun writes new ones. Pre-fix
    outputs are never overwritten -- the defect-fix run's own outputs land
    in the same `FINAL_EVALUATION_OUTPUTS_DIR`/`_ARTIFACTS_DIR` afterward,
    but the archive preserves exactly what existed before it ran.
    """
    _assert_within_namespace(
        outputs_dir, FINAL_EVALUATION_OUTPUTS_DIR, label="archive_prior_outputs outputs_dir"
    )
    _assert_within_namespace(
        artifacts_dir, FINAL_EVALUATION_ARTIFACTS_DIR, label="archive_prior_outputs artifacts_dir"
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = ARCHIVE_ROOT / timestamp
    for src_dir, name in ((outputs_dir, "outputs"), (artifacts_dir, "artifacts")):
        if src_dir.exists() and any(src_dir.iterdir()):
            dest = archive_dir / name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_dir, dest, dirs_exist_ok=True)
    return archive_dir


def write_seal(
    *,
    manifest: FinalEvaluationManifest,
    report: dict[str, Any],
    seal_path: Path = SEAL_PATH,
    defect_fix_of: str | None = None,
) -> EvaluationSeal:
    _assert_within_namespace(
        seal_path.parent, FINAL_EVALUATION_ARTIFACTS_DIR, label="write_seal seal_path"
    )
    report_hash = hash_report(report)
    seal = EvaluationSeal(
        manifest_content_hash=manifest.content_hash(),
        report_content_hash=report_hash,
        sealed_at=datetime.now(UTC).isoformat(),
        repository_commit=manifest.repository_commit,
        defect_fix_of=defect_fix_of,
    )
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    seal_path.write_text(json.dumps(seal.to_dict(), indent=2))
    return seal


def hash_report(report: dict[str, Any]) -> str:
    import hashlib

    canonical = json.dumps(report, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pre-evaluation guards
# ---------------------------------------------------------------------------


def run_pre_evaluation_guards(
    *,
    repo_root: Path = PROJECT_ROOT,
    seal_path: Path = SEAL_PATH,
    acknowledge_defect_fix: bool,
    defect_description: str = "",
) -> tuple[FinalEvaluationManifest, EvaluationSeal | None, FinalEvaluationAuthorization]:
    """Every guard that MUST pass before any 2025 data is read or downloaded:
    clean tree, frozen commit + artifact hashes recorded, no unacknowledged
    prior seal. Returns the ONLY `FinalEvaluationAuthorization` this run will
    ever mint -- no other function in this module can create one.
    """
    prior_seal = assert_no_unacknowledged_prior_seal(
        seal_path,
        acknowledge_defect_fix=acknowledge_defect_fix,
        defect_description=defect_description,
    )
    manifest = build_manifest(
        repo_root,
        train_seasons=TRAIN_SEASONS,
        evaluation_seasons=(EVALUATION_SEASON,),
        require_clean_tree=True,
    )
    validate_manifest_against_current_state(manifest, repo_root)
    authorization = _mint_authorization(manifest)
    return manifest, prior_seal, authorization


# ---------------------------------------------------------------------------
# 2025 ingestion -- the ONLY place in this repository authorized to fetch it
# ---------------------------------------------------------------------------


def _require_authorization(authorization: Any, fn_name: str) -> None:
    if not isinstance(authorization, FinalEvaluationAuthorization):
        raise FinalEvaluationError(
            f"{fn_name} requires a FinalEvaluationAuthorization instance minted by "
            f"run_pre_evaluation_guards -- got {type(authorization).__name__!r} (e.g. a bare "
            "boolean is never accepted, regardless of its value)."
        )
    if authorization.token not in _ISSUED_AUTHORIZATION_TOKENS:
        raise FinalEvaluationError(
            f"{fn_name} requires a FinalEvaluationAuthorization minted by "
            "run_pre_evaluation_guards -- a fabricated token, or an authorization from a "
            "different/earlier process, is never accepted."
        )
    assert_seasons_allowed([EVALUATION_SEASON], allow_final_evaluation=True)


#: Columns this pipeline treats as numeric somewhere downstream (contact
#: model features, spray-angle computation, eligibility). If a freshly
#: -downloaded 2025 file has one of these as non-numeric (e.g. a Savant
#: schema change that started returning "89.5 mph" strings), every
#: downstream numeric feature built on it would silently misbehave rather
#: than fail clearly -- so this gets an explicit, early dtype check.
EXPECTED_NUMERIC_RAW_COLUMNS: tuple[str, ...] = (
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "batter",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "hc_x",
    "hc_y",
)


def assert_raw_statcast_schema_compatible(raw: pd.DataFrame) -> dict[str, Any]:
    """Fail clearly and EARLY if the freshly-downloaded 2025 raw Statcast
    file doesn't match the schema every downstream step (`clean_batted_
    balls`, eligibility, feature engineering) assumes, rather than letting a
    renamed/retyped column surface as a confusing `KeyError` or a silently
    wrong feature deep in the pipeline. Distinguishes three findings:

      - missing column (name absent entirely, whether a structurally
        required identifier from `mlb_luck_score.data.clean_batted_balls.
        REQUIRED_COLUMNS` or a physically-essential numeric feature from
        `EXPECTED_NUMERIC_RAW_COLUMNS`) -- fatal, `SchemaDriftError`. A
        RENAMED column (e.g. `launch_speed` -> `launchSpeed`) looks exactly
        like this to this check, which is intentional: `clean_batted_balls.
        ensure_optional_columns` would otherwise silently backfill the
        renamed-away optional column as all-null rather than failing, one of
        the "fails silently" bug classes CLAUDE.md warns about elsewhere.
      - a present column that is not numeric-coercible -- fatal,
        `SchemaDriftError`, with a DIFFERENT message than plain missingness.
        Never silently repaired (no `pd.to_numeric(errors="coerce")`-style
        fix-up here -- `errors="coerce"` is used ONLY to detect the problem,
        its output is discarded).
      - extra/unexpected columns -- never fatal, reported for the
        provenance record. `clean_batted_balls.OPTIONAL_COLUMNS` are known
        -good and excluded from this report.

    Raises:
        SchemaDriftError: on missing or type-incompatible required columns.
    """
    from mlb_luck_score.data.clean_batted_balls import OPTIONAL_COLUMNS, REQUIRED_COLUMNS

    missing_required = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    missing_essential_numeric = [
        c
        for c in EXPECTED_NUMERIC_RAW_COLUMNS
        if c not in raw.columns and c not in REQUIRED_COLUMNS
    ]
    if missing_required or missing_essential_numeric:
        raise SchemaDriftError(
            "Raw 2025 Statcast data is missing column(s) this pipeline requires -- required "
            f"identifiers: {missing_required or 'none'}; essential numeric features (possibly "
            f"renamed): {missing_essential_numeric or 'none'}. This looks like a genuine "
            "Baseball Savant schema change, not a transient download issue -- do not add a "
            "compatibility shim without understanding why the column disappeared."
        )

    incompatible: list[str] = []
    for col in EXPECTED_NUMERIC_RAW_COLUMNS:
        if col not in raw.columns:
            continue
        series = raw[col].dropna()
        if series.empty:
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.isna().any():
            incompatible.append(col)
    if incompatible:
        raise SchemaDriftError(
            f"Raw 2025 Statcast column(s) {incompatible} contain non-numeric values -- "
            "refusing to silently coerce a required incompatible field. This is a type "
            "-compatibility problem, distinct from ordinary missingness."
        )

    known_columns = set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS)
    extra_columns = sorted(set(raw.columns) - known_columns)

    return {
        "missing_required_columns": [],
        "incompatible_numeric_columns": [],
        "extra_columns_present": extra_columns,
    }


def ingest_2025_raw_statcast(
    *,
    authorization: FinalEvaluationAuthorization,
    date_range: tuple[str, str] = FINAL_EVALUATION_2025_DATE_RANGE,
    output_dir: Path = FINAL_EVALUATION_RAW_DIR,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Download raw 2025 Statcast rows via `download_statcast_range`
    directly -- never via `mlb_luck_score.data.download_development_data`,
    which has no (and must never gain) a 2025 entry in `MLB_REGULAR_SEASON_
    DATE_RANGES`. Returns a provenance record.
    """
    _require_authorization(authorization, "ingest_2025_raw_statcast")
    _assert_within_namespace(
        output_dir, FINAL_EVALUATION_RAW_DIR, label="ingest_2025_raw_statcast output_dir"
    )

    path = development_raw_path(output_dir, EVALUATION_SEASON)
    start, end = date.fromisoformat(date_range[0]), date.fromisoformat(date_range[1])
    retrieved_at = datetime.now(UTC).isoformat()

    if path.exists() and not overwrite:
        logger.info("2025 raw statcast cache already exists at %s; skipping download.", path)
    else:
        df = download_statcast_range(start, end, chunk_days=chunk_days, max_retries=max_retries)
        if df.empty:
            raise FinalEvaluationError(
                f"No rows downloaded for 2025 range {date_range} -- refusing to write an "
                "empty file."
            )
        save_dataframe(df, path)

    raw = pd.read_parquet(path)
    schema_check = assert_raw_statcast_schema_compatible(raw)

    observed_dates = set(pd.to_datetime(raw["game_date"]).dt.date.unique()) if len(raw) else set()
    expected_dates = {start + timedelta(days=i) for i in range((end - start).days + 1)}
    missing_dates = sorted(d.isoformat() for d in expected_dates - observed_dates)

    return {
        "source": "pybaseball.statcast (Baseball Savant)",
        "requested_date_range": list(date_range),
        "retrieved_at": retrieved_at,
        "raw_file_path": str(path),
        "raw_file_sha256": compute_file_sha256(path),
        "row_count": int(len(raw)),
        "observed_date_coverage": (
            [str(min(observed_dates)), str(max(observed_dates))] if observed_dates else []
        ),
        "missing_dates_within_range": missing_dates,
        "schema_check": schema_check,
    }


def _build_2025_game_metadata(
    date_range: tuple[str, str],
    *,
    chunk_days: int,
    max_retries: int,
    backoff_seconds: float,
) -> pd.DataFrame:
    """Same logic as `download_game_metadata.build_season_metadata`, but
    parameterized directly by a date range instead of looking one up in
    `MLB_REGULAR_SEASON_DATE_RANGES` (which must never gain a 2025 entry).
    """
    start, end = date.fromisoformat(date_range[0]), date.fromisoformat(date_range[1])
    chunks = build_date_chunks(start, end, chunk_days)

    raw_games: list[dict[str, Any]] = []
    for chunk in chunks:
        raw_games.extend(
            _fetch_schedule_chunk(
                chunk.start, chunk.end, max_retries=max_retries, backoff_seconds=backoff_seconds
            )
        )

    rows = [_parse_schedule_game(g) for g in raw_games]
    games_df = pd.DataFrame(
        rows, columns=["game_pk", "game_date", "venue_id", "venue_name", "home_team", "away_team"]
    )
    games_df = games_df.drop_duplicates(subset=["game_pk"], keep="first")

    venue_cache: dict[int, dict[str, Any]] = {}
    for venue_id in [int(v) for v in games_df["venue_id"].dropna().unique()]:
        venue_cache[venue_id] = fetch_venue_details(
            venue_id, max_retries=max_retries, backoff_seconds=backoff_seconds
        )

    games_df["roof_type"] = games_df["venue_id"].map(
        lambda v: venue_cache.get(int(v), {}).get("roof_type") if pd.notna(v) else None
    )
    games_df["surface_type"] = games_df["venue_id"].map(
        lambda v: venue_cache.get(int(v), {}).get("surface_type") if pd.notna(v) else None
    )
    games_df["is_neutral_site"] = compute_neutral_site_flags(games_df)
    games_df = games_df.sort_values("game_pk").reset_index(drop=True)
    return games_df[list(GAME_METADATA_COLUMNS)]


def ingest_2025_game_metadata(
    *,
    authorization: FinalEvaluationAuthorization,
    date_range: tuple[str, str] = FINAL_EVALUATION_2025_DATE_RANGE,
    output_dir: Path = FINAL_EVALUATION_RAW_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fetch 2025 per-game venue/roof/surface metadata directly from the MLB
    Stats API -- never via `mlb_luck_score.data.download_game_metadata`'s own
    CLI, whose `build_season_metadata` hard-requires a `MLB_REGULAR_SEASON_
    DATE_RANGES` entry.
    """
    _require_authorization(authorization, "ingest_2025_game_metadata")
    _assert_within_namespace(
        output_dir, FINAL_EVALUATION_RAW_DIR, label="ingest_2025_game_metadata output_dir"
    )

    path = game_metadata_path(output_dir, EVALUATION_SEASON)
    retrieved_at = datetime.now(UTC).isoformat()

    if path.exists() and not overwrite:
        logger.info("2025 game-metadata cache already exists at %s; skipping fetch.", path)
    else:
        games_df = _build_2025_game_metadata(
            date_range,
            chunk_days=_GAME_METADATA_DEFAULT_CHUNK_DAYS,
            max_retries=_GAME_METADATA_DEFAULT_MAX_RETRIES,
            backoff_seconds=_GAME_METADATA_DEFAULT_BACKOFF_SECONDS,
        )
        if games_df.empty:
            raise FinalEvaluationError("No 2025 games found -- refusing to write an empty file.")
        path.parent.mkdir(parents=True, exist_ok=True)
        games_df.to_parquet(path, index=False)

    saved = pd.read_parquet(path)
    return {
        "source": "MLB Stats API (/schedule, /venues)",
        "requested_date_range": list(date_range),
        "retrieved_at": retrieved_at,
        "raw_file_path": str(path),
        "raw_file_sha256": compute_file_sha256(path),
        "game_count": int(len(saved)),
    }


def ingest_2025_sprint_speed(
    *,
    authorization: FinalEvaluationAuthorization,
    output_dir: Path = FINAL_EVALUATION_RAW_DIR,
    min_opp: int = DEFAULT_MIN_OPPORTUNITIES,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fetch the 2025 Sprint Speed leaderboard directly via `pybaseball.
    statcast_sprint_speed` -- `fetch_season_sprint_speed` has no season-list
    gate or protected-dict dependency, so it is reusable as-is; only
    `download_sprint_speed`'s CLI-facing season list is off-limits.
    """
    _require_authorization(authorization, "ingest_2025_sprint_speed")
    _assert_within_namespace(
        output_dir, FINAL_EVALUATION_RAW_DIR, label="ingest_2025_sprint_speed output_dir"
    )

    path = sprint_speed_path(output_dir, EVALUATION_SEASON)
    retrieved_at = datetime.now(UTC).isoformat()

    if path.exists() and not overwrite:
        logger.info("2025 sprint-speed cache already exists at %s; skipping fetch.", path)
    else:
        season_df = fetch_season_sprint_speed(EVALUATION_SEASON, min_opp=min_opp)
        if season_df.empty:
            raise FinalEvaluationError(
                "No 2025 sprint-speed rows returned -- refusing to write an empty file."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        season_df.to_parquet(path, index=False)

    saved = pd.read_parquet(path)
    return {
        "source": "pybaseball.statcast_sprint_speed (Baseball Savant Sprint Speed leaderboard)",
        "retrieved_at": retrieved_at,
        "raw_file_path": str(path),
        "raw_file_sha256": compute_file_sha256(path),
        "player_count": int(len(saved)),
        "min_opp": min_opp,
    }


def build_2025_evaluation_dataset(
    *, authorization: FinalEvaluationAuthorization, raw_dir: Path = FINAL_EVALUATION_RAW_DIR
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the isolated raw 2025 statcast file (reusing `clean_development_
    data` UNCHANGED, pointed at `raw_dir` so its output never touches
    `data/processed`) then join venue/geometry/sprint-speed exactly as the
    development pipeline does for 2021-2024.
    """
    _require_authorization(authorization, "build_2025_evaluation_dataset")
    _assert_within_namespace(
        raw_dir, FINAL_EVALUATION_RAW_DIR, label="build_2025_evaluation_dataset raw_dir"
    )

    cleaned = clean_development_data([EVALUATION_SEASON], raw_dir, allow_final_evaluation=True)
    if cleaned.empty:
        raise FinalEvaluationError("Cleaned 2025 dataset is empty.")

    metadata = load_game_metadata(raw_dir, [EVALUATION_SEASON])
    with_venue = join_venue_metadata(cleaned, metadata)
    venue_report = build_venue_join_report(with_venue, metadata)

    with_geometry = join_park_geometry(with_venue)
    geometry_report = build_geometry_join_report(with_geometry)

    sprint_speed_df = load_sprint_speed(raw_dir, [EVALUATION_SEASON])
    with_sprint_speed = join_sprint_speed(with_geometry, sprint_speed_df)
    sprint_speed_report = build_sprint_speed_join_report(with_sprint_speed)

    provenance = {
        "venue_join": asdict(venue_report),
        "geometry_join": asdict(geometry_report),
        "sprint_speed_join": asdict(sprint_speed_report),
    }
    return with_sprint_speed, provenance


# ---------------------------------------------------------------------------
# Training (TRAIN_SEASONS, unchanged) and scoring (2025)
# ---------------------------------------------------------------------------


def _read_existing_gate_status(path: Path, *, status_key: tuple[str, ...]) -> str:
    """Duplicated from `mlb_luck_score.scoring.run_attribution_ledger` (that
    module has no legitimate reason to ever import this evaluation script,
    and this function has no season dependency worth sharing a season
    -hardcoded wrapper for) -- cites an already-computed, already-saved
    calibration verdict verbatim.
    """
    if not path.exists():
        return f"not_available_missing_file:{path}"
    try:
        data: Any = json.loads(path.read_text())
    except json.JSONDecodeError:
        return f"not_available_unreadable_file:{path}"
    node = data
    for key in status_key:
        if not isinstance(node, dict) or key not in node:
            return f"not_available_missing_key_path:{'.'.join(status_key)}_in:{path}"
        node = node[key]
    return str(node)


def _apply_eligibility_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_eligibility(df)
    df = add_outfield_opportunity_eligibility(df)
    df = add_infield_opportunity_eligibility(df)
    df = add_advancement_eligibility(df)
    df = add_opportunity_features_by_domain(df)
    df = add_advancement_features(df)
    return df


@dataclass
class TrainedComponents:
    """Every trained model object `train_and_score_2025` produces, exposed
    so `run_final_evaluation` can feed them to `evaluation.
    v1_system_evaluation`'s predeclared per-component metrics without
    retraining anything.
    """

    contact_trained: TrainedModel
    outfield_trained: TrainedOpportunityModel
    infield_trained: TrainedOpportunityModel
    advancement_trained: TrainedAdvancementModel
    advancement_contact_trained: TrainedModel
    near_wall_trained: TrainedOpportunityModel
    infield_winner: str
    advancement_winner: str
    near_wall_winner: str
    development_df: pd.DataFrame


def train_and_score_2025(
    development_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    *,
    n_bootstrap_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    threshold_set: str = "primary",
    gate_status_dir: Path = PROJECT_ROOT / "outputs" / "tables",
) -> tuple[SeasonAggregationArtifacts, TrainedComponents]:
    """Train the four frozen component models on `TRAIN_SEASONS` (2021-2023,
    UNCHANGED), then score the isolated 2025 evaluation dataset. Mirrors
    `mlb_luck_score.scoring.run_season_aggregation.build_player_season_
    report`'s internals exactly (same model choices, same training seasons,
    same order of operations) but targets 2025 instead of `VALIDATION_
    SEASONS`; see module docstring for why that function is duplicated
    rather than extended.
    """
    assert_seasons_allowed(TRAIN_SEASONS, allow_final_evaluation=False)
    assert_seasons_allowed([EVALUATION_SEASON], allow_final_evaluation=True)

    dev_seasons_observed = sorted(int(s) for s in development_df["season"].dropna().unique())
    outside_train = [s for s in dev_seasons_observed if s not in TRAIN_SEASONS]
    if outside_train:
        raise FinalEvaluationError(
            f"Training data contains season(s) outside TRAIN_SEASONS {TRAIN_SEASONS}: "
            f"{outside_train}"
        )

    eval_seasons_observed = sorted(int(s) for s in evaluation_df["season"].dropna().unique())
    if eval_seasons_observed != [EVALUATION_SEASON]:
        raise FinalEvaluationError(
            f"Evaluation dataset must contain ONLY season {EVALUATION_SEASON}; observed "
            f"{eval_seasons_observed}"
        )

    dev_df = _apply_eligibility_pipeline(development_df)
    eval_df = _apply_eligibility_pipeline(evaluation_df)

    contact_train_df = dev_df[dev_df["eligible_for_training"].fillna(False)]
    contact_trained = train_model(contact_train_df, class_weight=None)

    outfield_train_df = dev_df[dev_df["outfield_opportunity_eligible"].astype(bool)]
    outfield_trained = train_opportunity_model(outfield_train_df, class_weight=None)

    infield_elig = dev_df[dev_df["infield_opportunity_eligible"].astype(bool)]
    infield_winner, _infield_metrics, infield_trained_by_candidate = run_infield_model_selection(
        infield_elig
    )
    infield_trained = infield_trained_by_candidate[infield_winner]

    advancement_contact_trained = fit_advancement_contact_model(dev_df)
    advancement_elig = get_advancement_rows(dev_df, advancement_contact_trained)
    advancement_winner, _advancement_metrics, advancement_trained_by_candidate = (
        run_advancement_model_selection(advancement_elig)
    )
    advancement_trained = advancement_trained_by_candidate[advancement_winner]

    near_wall_dev = get_near_wall_rows(dev_df)
    near_wall_winner, _near_wall_metrics, near_wall_trained_by_candidate = (
        run_near_wall_model_selection(near_wall_dev)
    )
    near_wall_trained = near_wall_trained_by_candidate[near_wall_winner]

    prob_cols = list(ADVANCEMENT_CONTACT_PROBABILITY_FEATURES)
    for col in prob_cols:
        eval_df[col] = np.nan
    eval_advancement_elig = get_advancement_rows(eval_df, advancement_contact_trained)
    eval_df.loc[eval_advancement_elig.index, prob_cols] = eval_advancement_elig[prob_cols]

    outfield_status_raw = _read_existing_gate_status(
        gate_status_dir / "opportunity_model_comparison_detail.json",
        status_key=(
            "validation_summary",
            "per_candidate",
            "measured_contact_only_v07",
            "passes_basic_validation",
        ),
    )
    infield_status_raw = _read_existing_gate_status(
        gate_status_dir / "infield_opportunity_detail.json",
        status_key=("gate_summary", "overall_status"),
    )
    advancement_status_raw = _read_existing_gate_status(
        gate_status_dir / "advancement_detail.json",
        status_key=("gate_summary", "overall_status"),
    )

    ledger = build_attribution_ledger(
        eval_df,
        contact_trained,
        outfield_trained=outfield_trained,
        infield_trained=infield_trained,
        advancement_trained=advancement_trained,
        outfield_confidence_status=outfield_status_raw,
        infield_confidence_status=infield_status_raw,
        advancement_confidence_status=advancement_status_raw,
    )
    confidence = build_play_level_confidence(
        eval_df,
        ledger,
        outfield_model_version="measured_contact_only_v07",
        infield_model_version=infield_winner,
        advancement_model_version=advancement_winner,
        outfield_model_status=outfield_status_raw,
        infield_model_status=infield_status_raw,
        advancement_model_status=advancement_status_raw,
    )
    summary = aggregate_to_batter_season(eval_df, ledger, confidence)

    season_identity_holds = verify_season_identity(summary)
    if not bool(season_identity_holds.all()):
        bad = int((~season_identity_holds).sum())
        raise FinalEvaluationError(
            f"Season-level accounting identity failed for {bad} of {len(summary)} rows"
        )

    bootstrap = bootstrap_batter_season_intervals(
        eval_df, ledger, n_reps=n_bootstrap_reps, seed=bootstrap_seed
    )
    player_season = summary.merge(bootstrap, on=["batter", "season"], how="left")
    player_season["qualification_status"] = assign_qualification_status(
        player_season, thresholds=threshold_set
    ).to_numpy()

    design = BootstrapDesign(n_reps=n_bootstrap_reps, seed=bootstrap_seed)
    report: dict[str, Any] = {
        "seasons": {
            "train_seasons": list(TRAIN_SEASONS),
            "evaluation_seasons": [EVALUATION_SEASON],
            "observed_in_training_input": dev_seasons_observed,
            "observed_in_evaluation_input": eval_seasons_observed,
            "final_test_season_touched": True,
            "final_test_season_touch_reason": (
                "single, predeclared, sealed Version 1.0 final evaluation -- see CLAUDE.md"
            ),
        },
        "model_selection_winners": {
            "outfield": "measured_contact_only_v07 (single candidate, no selection)",
            "infield": infield_winner,
            "advancement": advancement_winner,
            "near_wall_specialist": (
                f"{near_wall_winner} (informational only; Version 0.7 stays provisional/frozen)"
            ),
        },
        "component_model_status": {
            "outfield": outfield_status_raw,
            "infield": infield_status_raw,
            "advancement": advancement_status_raw,
        },
        "bootstrap_design": {
            "method": design.method,
            "resampling_unit": design.resampling_unit,
            "n_reps": design.n_reps,
            "seed": design.seed,
            "alpha": design.alpha,
            "refits_models": design.refits_models,
        },
        "qualification_threshold_set": threshold_set,
        "qualification_counts": summarize_qualification_counts(
            player_season["qualification_status"]
        ),
        "player_season_row_count": int(len(player_season)),
        "season_identity_holds_for_every_row": bool(season_identity_holds.all()),
        "total_scored_plays": int(len(eval_df)),
    }

    artifacts = SeasonAggregationArtifacts(
        scoring_df=eval_df,
        ledger=ledger,
        confidence=confidence,
        player_season=player_season,
        report=report,
    )
    trained = TrainedComponents(
        contact_trained=contact_trained,
        outfield_trained=outfield_trained,
        infield_trained=infield_trained,
        advancement_trained=advancement_trained,
        advancement_contact_trained=advancement_contact_trained,
        near_wall_trained=near_wall_trained,
        infield_winner=infield_winner,
        advancement_winner=advancement_winner,
        near_wall_winner=near_wall_winner,
        development_df=dev_df,
    )
    return artifacts, trained


# ---------------------------------------------------------------------------
# System checks + predeclared component metrics (delegates to v1_system_evaluation)
# ---------------------------------------------------------------------------


def run_system_checks_and_metrics(
    artifacts: SeasonAggregationArtifacts, trained: TrainedComponents
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every required system-level check plus every predeclared per
    -component metric, all delegated verbatim to `evaluation.v1_system_
    evaluation` -- this function only wires 2025's already-trained/-scored
    objects into that module's generic functions.
    """
    eval_df = artifacts.scoring_df
    ledger = artifacts.ledger

    system_checks: dict[str, Any] = {
        "play_level_accounting": verify_play_level_accounting(ledger),
        "season_level_accounting": verify_season_level_accounting(artifacts.player_season),
        "routing_exclusivity": verify_routing_exclusivity(eval_df),
        "advancement_eligibility_subset": verify_advancement_eligibility(eval_df),
        "event_ids": verify_event_ids(eval_df),
        "no_experimental_components": verify_no_experimental_components_in_official_total(ledger),
        "bootstrap_reproducibility": verify_reproducibility(
            bootstrap_batter_season_intervals,
            eval_df,
            ledger,
            n_reps=DEFAULT_N_BOOTSTRAP_REPS,
            seed=DEFAULT_BOOTSTRAP_SEED,
        ),
    }

    contact_eval_df = eval_df[eval_df["eligible_for_training"].fillna(False)]
    outfield_eval_df = eval_df[eval_df["outfield_opportunity_eligible"].astype(bool)]
    infield_eval_df = eval_df[eval_df["infield_opportunity_eligible"].astype(bool)]
    advancement_eval_df = eval_df.loc[
        eval_df.index.intersection(
            get_advancement_rows(eval_df, trained.advancement_contact_trained).index
        )
    ]
    near_wall_eval_df = get_near_wall_rows(eval_df)

    component_metrics: dict[str, Any] = {
        "contact_model": evaluate_contact_model(trained.contact_trained, contact_eval_df),
        "open_field_outfield_model": evaluate_open_field_outfield_model(
            trained.outfield_trained, outfield_eval_df
        ),
        "near_wall_specialist": evaluate_near_wall_specialist(
            trained.near_wall_trained, trained.outfield_trained, near_wall_eval_df
        ),
        "infield_model": evaluate_infield_model(trained.infield_trained, infield_eval_df),
        "advancement_model": evaluate_advancement_model(
            trained.advancement_trained,
            _empirical_advancement_baseline(trained),
            advancement_eval_df,
        ),
    }
    return system_checks, component_metrics


def _empirical_advancement_baseline(trained: TrainedComponents) -> Any:
    from mlb_luck_score.config import CALIBRATION_BASE_TRAIN_SEASONS
    from mlb_luck_score.models.compare_advancement_models import (
        fit_empirical_advancement_baseline,
    )

    fit_df = trained.development_df[
        trained.development_df["season"].isin(CALIBRATION_BASE_TRAIN_SEASONS)
    ]
    return fit_empirical_advancement_baseline(fit_df)


# ---------------------------------------------------------------------------
# Full orchestration
# ---------------------------------------------------------------------------


def run_final_evaluation(
    *,
    acknowledge_defect_fix: bool = False,
    defect_description: str = "",
    threshold_set: str = "primary",
    date_range: tuple[str, str] = FINAL_EVALUATION_2025_DATE_RANGE,
    repo_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """The complete, single-run Version 1.0 final evaluation. Every step
    before ingestion must pass before any 2025 data is read or downloaded.
    """
    manifest, prior_seal, authorization = run_pre_evaluation_guards(
        repo_root=repo_root,
        acknowledge_defect_fix=acknowledge_defect_fix,
        defect_description=defect_description,
    )
    if prior_seal is not None:
        archive_prior_outputs()

    statcast_provenance = ingest_2025_raw_statcast(
        authorization=authorization, date_range=date_range
    )
    game_metadata_provenance = ingest_2025_game_metadata(
        authorization=authorization, date_range=date_range
    )
    sprint_speed_provenance = ingest_2025_sprint_speed(authorization=authorization)
    evaluation_df, join_provenance = build_2025_evaluation_dataset(authorization=authorization)

    if not DEVELOPMENT_INPUT_PATH.exists():
        raise FinalEvaluationError(
            f"Frozen development input missing at {DEVELOPMENT_INPUT_PATH} -- this file is "
            "one of the manifest's frozen artifacts and must already exist."
        )
    full_development_df = pd.read_parquet(DEVELOPMENT_INPUT_PATH)
    training_df = full_development_df[full_development_df["season"].isin(TRAIN_SEASONS)].copy()

    artifacts, trained = train_and_score_2025(
        training_df, evaluation_df, threshold_set=threshold_set
    )

    system_checks, component_metrics = run_system_checks_and_metrics(artifacts, trained)

    public_score_table = build_public_score_table(artifacts)
    public_score_table = assign_official_ranks(public_score_table)
    contract_check = verify_qualification_and_ranking_contract(public_score_table)
    system_checks["public_contract"] = contract_check
    favorable = most_favorable_leaderboard(public_score_table)
    unfavorable = least_favorable_leaderboard(public_score_table)

    reference_df = _apply_eligibility_pipeline(full_development_df.copy())
    shift_report = compare_development_vs_evaluation(reference_df, artifacts.scoring_df)
    shift_flags = summarize_flags(shift_report)

    public_score_summary = summarize_public_score_evaluation(
        public_score_table, favorable, unfavorable, review_tables={}
    )
    stability_summary = summarize_2025_stability(artifacts, None, public_score_table)

    accounting_passed = bool(
        system_checks["play_level_accounting"]["identity_holds_for_every_resolved_row"]
        and system_checks["season_level_accounting"]["identity_holds_for_every_row"]
    )
    routing_passed = bool(system_checks["routing_exclusivity"]["mutually_exclusive"])
    reproducibility_passed = bool(system_checks["bootstrap_reproducibility"]["deterministic"])
    schema_and_public_contract_passed = bool(contract_check["schema_valid"])
    has_limitations = (
        bool(shift_flags) or component_metrics["near_wall_specialist"]["status"] == "provisional"
    )

    outcome_inputs = OutcomeInputs(
        accounting_passed=accounting_passed,
        routing_passed=routing_passed,
        reproducibility_passed=reproducibility_passed,
        schema_and_public_contract_passed=schema_and_public_contract_passed,
        frozen_artifacts_reproducible=True,
        critical_input_or_schema_issue=False,
        any_required_component_gate_severely_failed=False,
        has_provisional_or_subgroup_limitations=has_limitations,
    )
    outcome, outcome_reasons = classify_final_evaluation_outcome(outcome_inputs)

    ingestion_provenance = {
        "raw_statcast": statcast_provenance,
        "game_metadata": game_metadata_provenance,
        "sprint_speed": sprint_speed_provenance,
        "joins": join_provenance,
    }

    report = assemble_final_report(
        manifest_dict=manifest.to_dict(),
        system_checks=system_checks,
        component_metrics=component_metrics,
        distribution_shift_report=shift_report,
        distribution_shift_flags=shift_flags,
        public_score_summary=public_score_summary,
        stability_summary=stability_summary,
        outcome=outcome,
        outcome_reasons=outcome_reasons,
        limitations=shift_flags,
    )
    report["ingestion_provenance"] = ingestion_provenance

    _write_outputs(public_score_table, favorable, unfavorable, report)
    write_seal(
        manifest=manifest,
        report=report,
        defect_fix_of=(prior_seal.sealed_at if prior_seal is not None else None),
    )
    return report


def _write_outputs(
    public_score_table: pd.DataFrame,
    favorable: pd.DataFrame,
    unfavorable: pd.DataFrame,
    report: dict[str, Any],
    output_dir: Path = FINAL_EVALUATION_OUTPUTS_DIR,
) -> None:
    _assert_within_namespace(
        output_dir, FINAL_EVALUATION_OUTPUTS_DIR, label="_write_outputs output_dir"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tabular = public_score_table.copy()
    for col in ("component_status_reason_codes", "model_version"):
        if col in tabular.columns:
            tabular[col] = tabular[col].apply(json.dumps)
    tabular.to_parquet(output_dir / "public_score_v1_final.parquet", index=False)
    public_score_table.to_json(
        output_dir / "public_score_v1_final.json", orient="records", indent=2
    )
    favorable.to_json(
        output_dir / "favorable_leaderboard_v1_final.json", orient="records", indent=2
    )
    unfavorable.to_json(
        output_dir / "unfavorable_leaderboard_v1_final.json", orient="records", indent=2
    )
    (output_dir / "v1_final_report.json").write_text(json.dumps(report, indent=2, default=str))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acknowledge-defect-fix",
        action="store_true",
        help=(
            "Required to re-run after a prior sealed evaluation exists. Never pass this for "
            "tuning, a routine rerun, or 'just checking.'"
        ),
    )
    parser.add_argument(
        "--defect-description",
        default="",
        help="Required with --acknowledge-defect-fix: exactly what was wrong and fixed.",
    )
    parser.add_argument(
        "--threshold-set",
        default="primary",
        help="Qualification threshold set name (see mlb_luck_score.scoring.qualification).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        report = run_final_evaluation(
            acknowledge_defect_fix=args.acknowledge_defect_fix,
            defect_description=args.defect_description,
            threshold_set=args.threshold_set,
        )
    except (FinalEvaluationError, SealError, ManifestError) as exc:
        logger.error(str(exc))
        return 2

    logger.info("outcome=%s", report["outcome_classification"]["outcome"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
