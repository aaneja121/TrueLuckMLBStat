"""Version 0.14: the dedicated entry point for the ONE-TIME held-out 2025
Pitcher Contact Luck replication.

This is the only code path in this repository authorized to open 2025 for
the pitcher replication, and it may be run exactly once.

## Import-time behaviour: nothing

Importing this module reads no parquet, no CSV, no JSON data, no SQL, makes
no network call, and touches neither 2025 nor 2026. Every data access lives
inside `run_replication`, after `run_readiness_checks` has returned. That is
enforced by `tests/test_pitcher_replication_runner.py`, which imports this
module with every pandas reader and every `requests` method monkeypatched to
raise.

## Order of operations -- readiness first, always

    run_readiness_checks()          # freeze, authorization, manifest, tree,
                                    # protection, namespace, no prior receipt
    write_execution_start_receipt() # authorization CONSUMED here
    ingest 2025 ...                 # first byte of held-out data
    train_and_score_2025(...)       # frozen scoring path, unchanged
    questions A-G + classification
    write outputs and seal

The receipt is written between the last check and the first read, so the
authorization is spent at first look rather than at successful completion.
There is no automatic retry; see
`pitcher_replication_execution.RECOVERY_PROCEDURE`.

## The 2025 date range is LOCAL, and is the repository's verified one

`RESEARCH_RULES.md` requires the range to be defined locally inside the
sealed evaluation code, never in `mlb_luck_score.config`. It is defined
below, and it is the SAME reviewed range Version 1.0 recorded
(`evaluation.run_v1_final_evaluation.FINAL_EVALUATION_2025_DATE_RANGE`,
2025-03-18 to 2025-09-28, the early start covering the official Tokyo Series
regular-season games). A test asserts the two are identical, so this cannot
drift into a second, invented range -- and no 2025 data was fetched to
determine it.

## Scoring is the frozen path, reused not rebuilt

`train_and_score_2025` (Version 1.0) trains the four frozen component models
on `TRAIN_SEASONS` (2021-2023, unchanged) and scores the 2025 dataset,
returning `SeasonAggregationArtifacts`. Reimplementing that orchestration
would have been a new research choice; this module reuses it verbatim and
re-groups its output by pitcher through
`mlb_luck_score.scoring.pitching_contact_luck`, exactly as the frozen
Version 0.14 specification describes. Nothing is retrained, retuned or
redefined for 2025.

## Namespace

Raw 2025 lands in `data/pitcher_replication/2025/`; results in
`outputs/pitcher_replication/v0_14/`; control artifacts in
`artifacts/pitcher_replication/v0_14/`. Every write is checked by
`assert_within_namespace`, so nothing here can reach a development cache,
the Version 1.0 final-evaluation namespace, `prospective/`, `forecast/`, or
a dashboard fixture.

Usage (after the pre-execution manifest is sealed):

    make run-pitcher-replication-2025
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
for _extra in ("src", "evaluation", "replication"):
    _path = str(REPO_ROOT / _extra)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from pitcher_replication_execution import (  # noqa: E402
    EXECUTION_RECEIPT_PATH,
    ExecutionError,
    ExecutionManifest,
    ReplicationAuthorization,
    assert_within_namespace,
    build_execution_manifest,
    compute_file_hash,
    require_authorization,
    run_readiness_checks,
    validate_execution_manifest,
    write_execution_manifest,
    write_execution_start_receipt,
)
from pitcher_replication_freeze import (  # noqa: E402
    ARTIFACTS_DIR,
    REPLICATION_DATA_DIR,
    REPLICATION_OUTPUTS_DIR,
    get_git_commit_hash,
    read_freeze,
)
from pitcher_replication_questions import build_answers, role_bucket  # noqa: E402
from pitcher_replication_spec import REPLICATION_SEASON  # noqa: E402

logger = logging.getLogger(__name__)

#: REVIEWED inclusive 2025 MLB regular-season range, LOCAL to this module per
#: RESEARCH_RULES.md. Identical to the Version 1.0 final evaluation's own
#: reviewed range -- reused rather than re-derived, and never added to
#: `mlb_luck_score.config.MLB_REGULAR_SEASON_DATE_RANGES`, `DEVELOPMENT_
#: SEASONS`, `TRAIN_SEASONS` or `VALIDATION_SEASONS`. The 2025-03-18 start
#: covers the Dodgers-Cubs Tokyo Series, which were official regular-season
#: games; 2025-09-28 is the final scheduled regular-season date.
PITCHER_REPLICATION_2025_DATE_RANGE: tuple[str, str] = ("2025-03-18", "2025-09-28")

RESULTS_PATH = REPLICATION_OUTPUTS_DIR / "pitcher_replication_2025_results.json"
PROVENANCE_PATH = REPLICATION_OUTPUTS_DIR / "pitcher_replication_2025_provenance.json"
SEAL_PATH = ARTIFACTS_DIR / "pitcher_replication_2025_seal.json"

DEFAULT_CHUNK_DAYS = 5
DEFAULT_MAX_RETRIES = 3
DEFAULT_MIN_OPPORTUNITIES = 10
DEFAULT_BACKOFF_SECONDS = 1.0


class ReplicationRunError(RuntimeError):
    """Raised when the 2025 replication cannot proceed."""


# ---------------------------------------------------------------------------
# Ingestion -- the ONLY place this research line may fetch 2025
# ---------------------------------------------------------------------------


def ingest_2025_raw(
    *,
    authorization: ReplicationAuthorization,
    date_range: tuple[str, str] = PITCHER_REPLICATION_2025_DATE_RANGE,
    output_dir: Path = REPLICATION_DATA_DIR,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    """Download raw 2025 Statcast rows into the isolated namespace.

    Calls `download_statcast_range` directly, exactly as `RESEARCH_RULES.md`
    permits a dedicated sealed entry point to do -- never
    `download_development_data`, which has no 2025 date range and must never
    gain one.
    """
    require_authorization(authorization, "ingest_2025_raw")
    assert_within_namespace(output_dir, REPLICATION_DATA_DIR, label="ingest_2025_raw output_dir")

    from mlb_luck_score.config import development_raw_path
    from mlb_luck_score.data.download_statcast import download_statcast_range, save_dataframe

    path = development_raw_path(output_dir, REPLICATION_SEASON)
    start, end = date.fromisoformat(date_range[0]), date.fromisoformat(date_range[1])
    retrieved_at = datetime.now(UTC).isoformat()

    if not path.exists():
        frame = download_statcast_range(start, end, chunk_days=chunk_days, max_retries=max_retries)
        if frame.empty:
            raise ReplicationRunError(
                f"No rows downloaded for 2025 range {date_range} -- refusing to write an "
                "empty file."
            )
        save_dataframe(frame, path)

    raw = pd.read_parquet(path)
    observed = set(pd.to_datetime(raw["game_date"]).dt.date.unique()) if len(raw) else set()
    expected = {start + timedelta(days=i) for i in range((end - start).days + 1)}
    return {
        "source": "pybaseball.statcast (Baseball Savant)",
        "requested_date_range": list(date_range),
        "retrieved_at": retrieved_at,
        "raw_file_path": str(path),
        "raw_file_sha256": compute_file_hash(path),
        "row_count": int(len(raw)),
        "observed_date_coverage": ([str(min(observed)), str(max(observed))] if observed else []),
        "missing_dates_within_range": sorted(d.isoformat() for d in expected - observed),
    }


def ingest_2025_game_metadata(
    *,
    authorization: ReplicationAuthorization,
    date_range: tuple[str, str] = PITCHER_REPLICATION_2025_DATE_RANGE,
    output_dir: Path = REPLICATION_DATA_DIR,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> dict[str, Any]:
    """Per-game venue/roof metadata from the MLB Stats API, into the isolated
    namespace.
    """
    require_authorization(authorization, "ingest_2025_game_metadata")
    assert_within_namespace(
        output_dir, REPLICATION_DATA_DIR, label="ingest_2025_game_metadata output_dir"
    )

    # Reuse Version 1.0's own 2025 schedule/venue builder rather than
    # duplicating it. `download_game_metadata.build_season_metadata` cannot be
    # used: it looks its range up in `MLB_REGULAR_SEASON_DATE_RANGES`, which
    # must never gain a 2025 entry.
    from run_v1_final_evaluation import _build_2025_game_metadata

    from mlb_luck_score.config import game_metadata_path

    path = game_metadata_path(output_dir, REPLICATION_SEASON)
    retrieved_at = datetime.now(UTC).isoformat()
    if not path.exists():
        frame = _build_2025_game_metadata(
            date_range,
            chunk_days=chunk_days,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
        )
        if frame.empty:
            raise ReplicationRunError("No 2025 game metadata returned -- refusing to write.")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    saved = pd.read_parquet(path)
    return {
        "source": "MLB Stats API /schedule",
        "retrieved_at": retrieved_at,
        "raw_file_path": str(path),
        "raw_file_sha256": compute_file_hash(path),
        "game_count": int(len(saved)),
    }


def ingest_2025_sprint_speed(
    *,
    authorization: ReplicationAuthorization,
    output_dir: Path = REPLICATION_DATA_DIR,
    min_opp: int = DEFAULT_MIN_OPPORTUNITIES,
) -> dict[str, Any]:
    """The 2025 Sprint Speed leaderboard, into the isolated namespace."""
    require_authorization(authorization, "ingest_2025_sprint_speed")
    assert_within_namespace(
        output_dir, REPLICATION_DATA_DIR, label="ingest_2025_sprint_speed output_dir"
    )

    from mlb_luck_score.config import sprint_speed_path
    from mlb_luck_score.data.download_sprint_speed import fetch_season_sprint_speed

    path = sprint_speed_path(output_dir, REPLICATION_SEASON)
    retrieved_at = datetime.now(UTC).isoformat()
    if not path.exists():
        frame = fetch_season_sprint_speed(REPLICATION_SEASON, min_opp=min_opp)
        if frame.empty:
            raise ReplicationRunError("No 2025 sprint-speed rows returned -- refusing to write.")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    saved = pd.read_parquet(path)
    return {
        "source": "pybaseball.statcast_sprint_speed",
        "retrieved_at": retrieved_at,
        "raw_file_path": str(path),
        "raw_file_sha256": compute_file_hash(path),
        "player_count": int(len(saved)),
        "min_opp": min_opp,
    }


def build_2025_dataset(
    *, authorization: ReplicationAuthorization, raw_dir: Path = REPLICATION_DATA_DIR
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean and join the isolated raw 2025 files exactly as the development
    pipeline does for 2021-2024 -- same cleaning, same venue/geometry/
    sprint-speed joins, no 2025-specific step.
    """
    require_authorization(authorization, "build_2025_dataset")
    assert_within_namespace(raw_dir, REPLICATION_DATA_DIR, label="build_2025_dataset raw_dir")

    from mlb_luck_score.data.clean_development_data import clean_development_data
    from mlb_luck_score.data.join_park_geometry import (
        build_geometry_join_report,
        join_park_geometry,
    )
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

    cleaned = clean_development_data([REPLICATION_SEASON], raw_dir, allow_final_evaluation=True)
    if cleaned.empty:
        raise ReplicationRunError("Cleaned 2025 dataset is empty.")

    metadata = load_game_metadata(raw_dir, [REPLICATION_SEASON])
    with_venue = join_venue_metadata(cleaned, metadata)
    venue_report = build_venue_join_report(with_venue, metadata)

    with_geometry = join_park_geometry(with_venue)
    geometry_report = build_geometry_join_report(with_geometry)

    sprint = load_sprint_speed(raw_dir, [REPLICATION_SEASON])
    with_sprint = join_sprint_speed(with_geometry, sprint)
    sprint_report = build_sprint_speed_join_report(with_sprint)

    return with_sprint, {
        "venue_join": asdict(venue_report),
        "geometry_join": asdict(geometry_report),
        "sprint_speed_join": asdict(sprint_report),
    }


# ---------------------------------------------------------------------------
# Pitcher-side projection
# ---------------------------------------------------------------------------


def build_pitcher_frame(artifacts: Any) -> tuple[pd.DataFrame, float]:
    """Re-group the frozen ledger by pitcher and project the display fields
    questions A-G need. Recomputes no score: every value is read from
    `build_pitcher_season_table` or the ledger.

    Returns:
        `(frame, play_level_mean)` -- one row per pitcher-season, plus the
        mean per-play pitcher contribution over resolved plays.
    """
    from mlb_luck_score.scoring.pitching_contact_luck import build_pitcher_season_table

    table = build_pitcher_season_table(artifacts.scoring_df, artifacts.ledger, artifacts.confidence)
    table = table[table["season"] == REPLICATION_SEASON].copy()

    plays = artifacts.scoring_df.loc[
        :,
        ["event_id", "pitcher", "season", "game_date", "launch_speed", "launch_angle", "bb_type"],
    ].copy()
    plays["outcome_class"] = artifacts.scoring_df["outcome_class"]
    # Index-aligned, never positional: the ledger is built on scoring_df's own index.
    plays["pitching_contact_luck_runs"] = -artifacts.ledger["final_result_surprise"]
    plays = plays[plays["season"] == REPLICATION_SEASON]
    resolved = plays[plays["pitching_contact_luck_runs"].notna()].copy()
    play_level_mean = float(resolved["pitching_contact_luck_runs"].mean()) if len(resolved) else 0.0

    names = (
        artifacts.scoring_df[artifacts.scoring_df["season"] == REPLICATION_SEASON]
        .groupby("pitcher")["player_name"]
        .agg(lambda s: s.mode().iat[0] if len(s.mode()) else None)
    )
    by_pitcher = {int(pid): grp for pid, grp in resolved.groupby("pitcher")}

    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        pitcher_id = int(row["pitcher"])
        bbe = int(row["eligible_batted_balls"])
        appearances = int(row["games"])
        total = float(row["total_observed_minus_expected_runs"])
        grp = by_pitcher.get(pitcher_id)

        favorable = unfavorable = None
        largest_share = None
        if grp is not None and len(grp):
            favorable = _play_record(grp.loc[grp["pitching_contact_luck_runs"].idxmax()])
            unfavorable = _play_record(grp.loc[grp["pitching_contact_luck_runs"].idxmin()])
            largest = float(grp["pitching_contact_luck_runs"].abs().max())
            if abs(total) > 1e-9:
                largest_share = round(largest / abs(total), 4)

        rows.append(
            {
                "pitcher_id": pitcher_id,
                "name": _display_name(names.get(pitcher_id)),
                "role_bucket": role_bucket(bbe, appearances),
                "cumulative_contact_luck_runs": round(total, 4),
                "eligible_batted_balls": bbe,
                "appearances": appearances,
                "contact_luck_per_100": round(float(row["observed_minus_expected_per_100"]), 4),
                "per_100_ci_low": round(float(row["observed_minus_expected_per_100_ci_low"]), 4),
                "per_100_ci_high": round(float(row["observed_minus_expected_per_100_ci_high"]), 4),
                "cumulative_ci_low": round(float(row["observed_minus_expected_ci_low"]), 4),
                "cumulative_ci_high": round(float(row["observed_minus_expected_ci_high"]), 4),
                "largest_play_share_of_net": largest_share,
                "largest_favorable_play": favorable,
                "largest_unfavorable_play": unfavorable,
            }
        )
    return pd.DataFrame(rows), play_level_mean


def _display_name(raw: Any) -> str | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if "," not in text:
        return text
    last, first = text.split(",", 1)
    return f"{first.strip()} {last.strip()}"


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


def _play_record(row: pd.Series) -> dict[str, Any]:
    return {
        "play_id": str(row["event_id"]),
        "game_date": str(row["game_date"])[:10],
        "pitching_contact_luck_runs": round(float(row["pitching_contact_luck_runs"]), 4),
        "launch_speed_mph": _optional_float(row.get("launch_speed")),
        "launch_angle_deg": _optional_float(row.get("launch_angle")),
        "bb_type": None if pd.isna(row.get("bb_type")) else str(row["bb_type"]),
        "outcome_class": None if pd.isna(row.get("outcome_class")) else str(row["outcome_class"]),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_replication(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """The one-time held-out 2025 replication, end to end.

    Readiness runs FIRST and the receipt is written BEFORE any ingestion, so
    no 2025 byte is read until the authorization has been verified and
    consumed.
    """
    logger.info("=== readiness checks (no data is read during these) ===")
    authorization, manifest, readiness = run_readiness_checks(repo_root)
    logger.info("readiness passed: %s", json.dumps(readiness, sort_keys=True))

    logger.info("=== writing the execution-start receipt: 2025 is about to be opened ===")
    receipt = write_execution_start_receipt(authorization, manifest)
    logger.info("authorization CONSUMED, execution_id=%s", receipt["execution_id"])

    logger.info("=== ingesting 2025 into the isolated namespace ===")
    provenance = {
        "raw_statcast": ingest_2025_raw(authorization=authorization),
        "game_metadata": ingest_2025_game_metadata(authorization=authorization),
        "sprint_speed": ingest_2025_sprint_speed(authorization=authorization),
    }
    evaluation_df, join_provenance = build_2025_dataset(authorization=authorization)
    provenance["joins"] = join_provenance

    logger.info("=== scoring via the frozen Version 1.0 path (trains on 2021-2023 only) ===")
    from run_v1_final_evaluation import DEVELOPMENT_INPUT_PATH, train_and_score_2025

    development_df = pd.read_parquet(DEVELOPMENT_INPUT_PATH)
    artifacts, _trained = train_and_score_2025(development_df, evaluation_df)

    logger.info("=== answering the frozen questions A-G ===")
    frame, play_level_mean = build_pitcher_frame(artifacts)
    answers = build_answers(frame, artifacts, play_level_mean)

    freeze = read_freeze()
    results = {
        "replication_season": REPLICATION_SEASON,
        "spec_version": freeze.spec_version,
        "freeze_content_hash": freeze.freeze_content_hash(),
        "spec_content_hash": freeze.spec_content_hash,
        "execution_manifest_hash": manifest.manifest_content_hash(),
        "execution_id": receipt["execution_id"],
        "repository_commit": get_git_commit_hash(repo_root),
        "date_range": list(PITCHER_REPLICATION_2025_DATE_RANGE),
        "pitcher_season_rows": int(len(frame)),
        **answers,
        "classification_is_frozen_before_any_design_change": True,
        "limits": (
            "Held-out 2025, one time, under the frozen Version 0.14 specification. No "
            "presentation rule may change until this classification is reported and "
            "frozen."
        ),
    }
    _write_outputs(results, provenance)
    _write_seal(results, manifest, receipt, repo_root)
    return results


def _write_outputs(results: dict[str, Any], provenance: dict[str, Any]) -> None:
    for path, payload in ((RESULTS_PATH, results), (PROVENANCE_PATH, provenance)):
        assert_within_namespace(path, REPLICATION_OUTPUTS_DIR, label="output")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _write_seal(
    results: dict[str, Any],
    manifest: ExecutionManifest,
    receipt: dict[str, Any],
    repo_root: Path,
) -> None:
    assert_within_namespace(SEAL_PATH, ARTIFACTS_DIR, label="seal")
    seal = {
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "execution_id": receipt["execution_id"],
        "execution_manifest_hash": manifest.manifest_content_hash(),
        "freeze_content_hash": manifest.freeze_content_hash,
        "spec_content_hash": manifest.spec_content_hash,
        "repository_commit": get_git_commit_hash(repo_root),
        "classification": results["classification"]["classification"],
        "primary_disagreements": results["classification"]["primary_disagreements"],
        "results_sha256": None,
    }
    SEAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    seal["results_sha256"] = compute_file_hash(RESULTS_PATH)
    SEAL_PATH.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "The one-time held-out 2025 Pitcher Contact Luck replication. Requires a "
            "sealed pre-execution manifest and refuses to run twice."
        )
    )
    parser.add_argument(
        "--seal-execution-manifest",
        action="store_true",
        help=(
            "Build and write the pre-execution manifest, then exit WITHOUT opening 2025. "
            "Must be run from a clean tree before the replication itself."
        ),
    )
    parser.add_argument(
        "--check-readiness",
        action="store_true",
        help="Run every readiness check and report, WITHOUT opening 2025. Read-only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_arg_parser().parse_args(argv)

    if args.seal_execution_manifest:
        manifest = build_execution_manifest()
        path, digest = write_execution_manifest(manifest)
        report = validate_execution_manifest(manifest)
        print(f"=== Sealed {path} ===")
        print(f"execution_manifest_hash = {digest}")
        print(json.dumps(report, indent=2, sort_keys=True))
        print("2025 was NOT opened.")
        return 0

    if args.check_readiness:
        try:
            _authorization, manifest, report = run_readiness_checks()
        except (ExecutionError, Exception) as exc:  # noqa: BLE001 -- reported, not swallowed
            print(f"NOT READY: {exc}")
            return 1
        print("READY -- every precondition passed. 2025 was NOT opened.")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if not EXECUTION_RECEIPT_PATH.exists():
        logger.info("This run WILL open held-out 2025 data, once, and cannot be repeated.")
    results = run_replication()
    print(json.dumps(results["classification"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
