"""Contact Luck v1.1: the dedicated, repeatable prospective 2026 scoring entry point.

Versions 0.2-1.0 are FROZEN. This script applies the exact frozen Version 1.0
system (contact model, outfield/infield opportunity models, advancement
model, attribution ledger, confidence, aggregation, qualification, public
-score contract) to 2026 season-to-date data. It NEVER refits, recalibrates,
retunes, or selects a feature/threshold based on 2026 data -- see CLAUDE.md
"Version 1.1: prospective 2026 scoring."

Unlike `evaluation/run_v1_final_evaluation.py` (a one-time sealed event),
this script is meant to run repeatedly across a season, once per
`--data-through` date. Immutability comes from refusing to overwrite a
completed, DIFFERENT snapshot while accepting an identical rerun as a no-op
-- see `prospective.prospective_manifest` for the deterministic-hash design
this relies on.

## Exposed CLI surface (deliberately narrow)

    --data-through YYYY-MM-DD   (required) -- inclusive upper bound on the
                                 2026 data this snapshot scores.
    --snapshot-label            (optional) -- a short label appended to the
                                 snapshot directory name.
    --force-redownload          (optional) -- refresh the SHARED, MUTABLE raw
                                 cache under data/prospective/2026/ before
                                 building a NEW snapshot. Can never touch or
                                 invalidate an already-completed snapshot
                                 directory.

There is deliberately NO model-selection, calibration, feature-selection, or
threshold-tuning flag here -- see CLAUDE.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from prospective_config import (
    PROJECT_ROOT,
    PROSPECTIVE_ARTIFACTS_DIR,
    PROSPECTIVE_OUTPUTS_DIR,
    _assert_within_namespace,
    assert_not_sealed_v1_namespace,
)
from prospective_ingestion import (
    ProspectiveError,
    assert_data_through_date_is_complete,
    build_2026_scoring_dataset,
    ingest_2026_game_metadata,
    ingest_2026_raw_statcast,
    ingest_2026_sprint_speed,
    run_prospective_guards,
)
from prospective_manifest import (
    SnapshotConflictError,
    SnapshotManifest,
    build_snapshot_manifest,
    compute_file_sha256,
    with_output_hashes,
)
from prospective_player_names import apply_player_name_overlay
from prospective_scoring import ProspectiveScoringError, train_and_score_2026
from run_v1_final_evaluation import DEVELOPMENT_INPUT_PATH

from mlb_luck_score.config import TRAIN_SEASONS
from mlb_luck_score.data.download_player_names import fetch_player_names
from mlb_luck_score.scoring.leaderboard import (
    assign_official_ranks,
    least_favorable_leaderboard,
    most_favorable_leaderboard,
)
from mlb_luck_score.scoring.public_score_table import build_public_score_table

logger = logging.getLogger(__name__)

#: Frozen default -- never exposed as a CLI flag (no threshold-tuning surface).
DEFAULT_THRESHOLD_SET = "primary"


class RunProspectiveScoringError(ValueError):
    """Raised when this script's own preconditions are not met."""


def _snapshot_dir_name(data_through_date: str, snapshot_label: str | None) -> str:
    return f"{data_through_date}__{snapshot_label}" if snapshot_label else data_through_date


def _flatten_for_tabular_export(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("component_status_reason_codes", "model_version"):
        if col in out.columns:
            out[col] = out[col].apply(json.dumps)
    return out


def _build_score_card(public_score_table: pd.DataFrame) -> dict[str, Any]:
    from mlb_luck_score.scoring.public_score_schema import PUBLIC_SCORE_SCHEMA_VERSION
    from mlb_luck_score.scoring.qualification import STATUS_QUALIFIED

    qualified = public_score_table["qualification_status"] == STATUS_QUALIFIED
    qualification_counts = public_score_table["qualification_status"].value_counts().to_dict()
    return {
        "score_version": PUBLIC_SCORE_SCHEMA_VERSION,
        "model_version": public_score_table["model_version"].iloc[0]
        if len(public_score_table)
        else {},
        "generated_at": public_score_table["generated_at"].iloc[0]
        if len(public_score_table)
        else None,
        "data_through_date": sorted(
            public_score_table["data_through_date"].dropna().unique().tolist()
        ),
        "seasons": sorted(int(s) for s in public_score_table["season"].unique().tolist()),
        "row_count": int(len(public_score_table)),
        "qualified_count": int(qualified.sum()),
        "qualification_counts": {str(k): int(v) for k, v in qualification_counts.items()},
        "ranking_method": "competition_ranking_min_with_batter_id_tiebreak",
        "official_metric": "contact_luck_runs_per_100",
        "interval_method": "game_pk_clustered_percentile_bootstrap_95pct",
    }


def resolve_snapshot_conflict(
    prior_manifest: SnapshotManifest | None,
    candidate_manifest: SnapshotManifest,
    *,
    dir_name: str,
    artifacts_snapshot_dir: Path,
) -> str:
    """Pure decision logic, factored out of `run_prospective_snapshot` so it
    is directly unit-testable without running the full ingestion/scoring
    pipeline. Returns `"new"` if there is no prior snapshot to compare
    against, `"idempotent_no_op"` if a prior snapshot exists and is
    deterministically identical, or raises `SnapshotConflictError` if a
    prior snapshot exists and differs.
    """
    if prior_manifest is None:
        return "new"
    if (
        prior_manifest.deterministic_content_hash()
        == candidate_manifest.deterministic_content_hash()
    ):
        logger.info(
            "Snapshot %s already exists and is deterministically identical to this run -- "
            "idempotent no-op, nothing rewritten.",
            dir_name,
        )
        return "idempotent_no_op"
    raise SnapshotConflictError(
        f"A completed snapshot already exists at {artifacts_snapshot_dir} for this "
        "data-through date, and it is NOT deterministically identical to this run's "
        "inputs/outputs -- refusing to overwrite it. Use a different --snapshot-label to "
        "record a distinct snapshot, or investigate why the same data-through date now "
        "produces different results."
    )


def _read_prior_manifest(artifacts_snapshot_dir: Path) -> SnapshotManifest | None:
    manifest_path = artifacts_snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise SnapshotConflictError(
            f"Existing snapshot directory {artifacts_snapshot_dir} has a manifest.json that is "
            "not valid JSON -- refusing to treat this as either 'no prior snapshot' or a clean "
            "rerun target. Investigate manually; never auto-delete."
        ) from exc
    try:
        return SnapshotManifest.from_dict(data)
    except TypeError as exc:
        raise SnapshotConflictError(
            f"Existing snapshot directory {artifacts_snapshot_dir} has a manifest.json that "
            f"does not match the current SnapshotManifest schema ({exc}) -- refusing to treat "
            "this as either 'no prior snapshot' or a clean rerun target."
        ) from exc


def run_prospective_snapshot(
    *,
    data_through_date: str,
    snapshot_label: str | None = None,
    force_redownload: bool = False,
    repo_root: Path = PROJECT_ROOT,
    outputs_root: Path = PROSPECTIVE_OUTPUTS_DIR,
    artifacts_root: Path = PROSPECTIVE_ARTIFACTS_DIR,
) -> dict[str, Any]:
    """The complete Version 1.1 prospective snapshot run. Returns a summary
    dict; writes nothing to disk if an identical snapshot already exists
    (idempotent no-op) and raises `SnapshotConflictError` if a DIFFERENT
    completed snapshot already exists for the same directory name.

    Guard order (cheap/local first, network last): namespace safety ->
    working-tree cleanliness (`run_prospective_guards`, now BLOCKING -- see
    its docstring) -> data-through-date completeness (`assert_data_through_
    date_is_complete`, one network call) -> ingestion.
    """
    date.fromisoformat(data_through_date)  # validate format early, fail loudly
    assert_not_sealed_v1_namespace(outputs_root, label="run_prospective_snapshot outputs_root")
    assert_not_sealed_v1_namespace(artifacts_root, label="run_prospective_snapshot artifacts_root")

    authorization = run_prospective_guards(repo_root=repo_root)
    completeness_result = assert_data_through_date_is_complete(data_through_date)

    statcast_provenance = ingest_2026_raw_statcast(
        authorization=authorization,
        data_through_date=data_through_date,
        overwrite=force_redownload,
    )
    game_metadata_provenance = ingest_2026_game_metadata(
        authorization=authorization,
        data_through_date=data_through_date,
        overwrite=force_redownload,
    )
    sprint_speed_provenance = ingest_2026_sprint_speed(
        authorization=authorization, overwrite=force_redownload
    )
    scoring_df, join_provenance = build_2026_scoring_dataset(authorization=authorization)

    observed_max_date = pd.to_datetime(scoring_df["game_date"]).max()
    if pd.notna(observed_max_date) and observed_max_date.date() > date.fromisoformat(
        data_through_date
    ):
        raise RunProspectiveScoringError(
            f"Scoring dataset contains game_date {observed_max_date.date()} beyond the "
            f"requested --data-through {data_through_date} -- refusing to score data past the "
            "requested cutoff."
        )

    if not DEVELOPMENT_INPUT_PATH.exists():
        raise RunProspectiveScoringError(
            f"Frozen development input missing at {DEVELOPMENT_INPUT_PATH} -- this is the same "
            "frozen 2021-2024 input Version 1.0/0.12 already use."
        )
    full_development_df = pd.read_parquet(DEVELOPMENT_INPUT_PATH)
    training_df = full_development_df[full_development_df["season"].isin(TRAIN_SEASONS)].copy()

    artifacts, _trained = train_and_score_2026(
        training_df, scoring_df, threshold_set=DEFAULT_THRESHOLD_SET
    )

    public_score_table = build_public_score_table(artifacts)
    public_score_table = assign_official_ranks(public_score_table)
    favorable = most_favorable_leaderboard(public_score_table)
    unfavorable = least_favorable_leaderboard(public_score_table)

    names_df = fetch_player_names(public_score_table["batter_id"].astype(int).tolist())
    public_score_table, name_report = apply_player_name_overlay(public_score_table, names_df)

    score_card = _build_score_card(public_score_table)
    coverage_and_schema_report = {
        "statcast_schema_check": statcast_provenance["schema_check"],
        "missing_dates_within_range": statcast_provenance["missing_dates_within_range"],
        "data_through_completeness": completeness_result.to_dict(),
        "venue_join": join_provenance["venue_join"],
        "geometry_join": join_provenance["geometry_join"],
        "sprint_speed_join": join_provenance["sprint_speed_join"],
    }
    component_status_summary = {
        "model_selection_winners": artifacts.report["model_selection_winners"],
        "component_model_status": artifacts.report["component_model_status"],
    }

    observed_date_coverage = statcast_provenance["observed_date_coverage"]
    retrieval_timestamps = {
        "raw_statcast": statcast_provenance["retrieved_at"],
        "game_metadata": game_metadata_provenance["retrieved_at"],
        "sprint_speed": sprint_speed_provenance["retrieved_at"],
        "player_names": name_report.retrieved_at,
    }
    source_hashes = {
        "raw_statcast": statcast_provenance["raw_file_sha256"],
        "game_metadata": game_metadata_provenance["raw_file_sha256"],
        "sprint_speed": sprint_speed_provenance["raw_file_sha256"],
    }
    raw_row_counts = {
        "raw_statcast": statcast_provenance["row_count"],
        "game_metadata": game_metadata_provenance["game_count"],
        "sprint_speed": sprint_speed_provenance["player_count"],
    }

    manifest = build_snapshot_manifest(
        repo_root,
        data_through_date=data_through_date,
        requested_date_range=tuple(statcast_provenance["requested_date_range"]),
        observed_date_coverage=observed_date_coverage,
        snapshot_label=snapshot_label,
        retrieval_timestamps=retrieval_timestamps,
        source_hashes=source_hashes,
        raw_row_counts=raw_row_counts,
        processed_row_count=int(len(scoring_df)),
        schema_checks=coverage_and_schema_report,
        join_coverage=join_provenance,
        qualification_threshold_set=DEFAULT_THRESHOLD_SET,
    )

    dir_name = _snapshot_dir_name(data_through_date, snapshot_label)
    outputs_snapshot_dir = outputs_root / dir_name
    artifacts_snapshot_dir = artifacts_root / dir_name
    _assert_within_namespace(
        outputs_snapshot_dir, outputs_root, label="run_prospective_snapshot outputs_snapshot_dir"
    )
    _assert_within_namespace(
        artifacts_snapshot_dir,
        artifacts_root,
        label="run_prospective_snapshot artifacts_snapshot_dir",
    )

    prior_manifest = _read_prior_manifest(artifacts_snapshot_dir)
    conflict_status = resolve_snapshot_conflict(
        prior_manifest, manifest, dir_name=dir_name, artifacts_snapshot_dir=artifacts_snapshot_dir
    )
    if conflict_status == "idempotent_no_op":
        assert prior_manifest is not None  # noqa: S101 -- narrows type for mypy
        return {
            "status": "idempotent_no_op",
            "snapshot_dir": str(outputs_snapshot_dir),
            "manifest": prior_manifest.to_dict(),
        }

    outputs_snapshot_dir.mkdir(parents=True, exist_ok=True)
    artifacts_snapshot_dir.mkdir(parents=True, exist_ok=True)

    tabular = _flatten_for_tabular_export(public_score_table)
    tabular.to_csv(outputs_snapshot_dir / "public_score.csv", index=False)
    tabular.to_parquet(outputs_snapshot_dir / "public_score.parquet", index=False)
    public_score_table.to_json(
        outputs_snapshot_dir / "public_score.json", orient="records", indent=2
    )
    favorable.to_json(
        outputs_snapshot_dir / "favorable_leaderboard.json", orient="records", indent=2
    )
    unfavorable.to_json(
        outputs_snapshot_dir / "unfavorable_leaderboard.json", orient="records", indent=2
    )
    (outputs_snapshot_dir / "scorecard.json").write_text(
        json.dumps(score_card, indent=2, default=str)
    )
    (outputs_snapshot_dir / "coverage_and_schema_report.json").write_text(
        json.dumps(coverage_and_schema_report, indent=2, default=str)
    )
    (outputs_snapshot_dir / "component_status_summary.json").write_text(
        json.dumps(component_status_summary, indent=2, default=str)
    )
    (outputs_snapshot_dir / "name_resolution_report.json").write_text(
        json.dumps(name_report.to_dict(), indent=2, default=str)
    )

    output_hashes = {
        f"outputs/{p.name}": compute_file_sha256(p)
        for p in sorted(outputs_snapshot_dir.iterdir())
        if p.is_file()
    }
    manifest = with_output_hashes(manifest, output_hashes)
    (artifacts_snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, default=str)
    )

    integrity_hashes = {f"outputs/{k.split('/', 1)[1]}": v for k, v in output_hashes.items()}
    integrity_hashes["artifacts/manifest.json"] = compute_file_sha256(
        artifacts_snapshot_dir / "manifest.json"
    )
    (artifacts_snapshot_dir / "integrity_hashes.json").write_text(
        json.dumps(integrity_hashes, indent=2, default=str)
    )

    logger.info(
        "Wrote snapshot %s to %s / %s", dir_name, outputs_snapshot_dir, artifacts_snapshot_dir
    )
    return {
        "status": "written",
        "snapshot_dir": str(outputs_snapshot_dir),
        "artifacts_dir": str(artifacts_snapshot_dir),
        "manifest": manifest.to_dict(),
        "score_card": score_card,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-through",
        required=True,
        help="Inclusive upper bound (YYYY-MM-DD) on the 2026 data this snapshot scores.",
    )
    parser.add_argument(
        "--snapshot-label",
        default=None,
        help="Optional short label appended to the snapshot directory name.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help=(
            "Refresh the shared, mutable 2026 raw cache before building a NEW snapshot. Never "
            "touches or invalidates an already-completed snapshot directory."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = run_prospective_snapshot(
            data_through_date=args.data_through,
            snapshot_label=args.snapshot_label,
            force_redownload=args.force_redownload,
        )
    except (
        ProspectiveError,
        ProspectiveScoringError,
        SnapshotConflictError,
        RunProspectiveScoringError,
    ) as exc:
        logger.error(str(exc))
        return 2

    logger.info("status=%s snapshot_dir=%s", result["status"], result["snapshot_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
