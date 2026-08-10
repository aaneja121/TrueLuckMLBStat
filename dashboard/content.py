"""Contact Luck v1.2: turns a validated snapshot's raw JSON payloads into
page-ready view-models.

Every value in every dataclass below is copied verbatim from a frozen
Version 1.1 snapshot output file -- this module recomputes nothing (no
score, no rank, no interval, no qualification decision). Its only jobs are:
load the handful of JSON files a validated `DiscoveredSnapshot` points at,
and reshape them into the specific fields each page needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dashboard_config import DASHBOARD_VERSION
from snapshot_data import DiscoveredSnapshot, SnapshotHistoryEntry

__all__ = [
    "ComponentBreakdown",
    "DashboardBuildManifest",
    "LeaderboardRow",
    "PlayerDetail",
    "PlayerIndexEntry",
    "SnapshotPayloadError",
    "SnapshotPayloads",
    "StatusPageData",
    "TrendPoint",
    "build_dashboard_manifest",
    "build_favorable_leaderboard",
    "build_player_detail",
    "build_player_index",
    "build_player_trend",
    "build_status_page_data",
    "build_unfavorable_leaderboard",
    "find_player_record",
    "load_snapshot_payloads",
]


class SnapshotPayloadError(Exception):
    """A required output file for an otherwise-valid snapshot could not be
    read/parsed. Distinct from `snapshot_data`'s integrity validation (which
    already confirmed the file exists and its hash matches the manifest) --
    this only guards against a JSON-parsing surprise.
    """


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotPayloadError(f"failed to load {path}: {exc}") from exc


@dataclass(frozen=True)
class SnapshotPayloads:
    snapshot: DiscoveredSnapshot
    public_score: list[dict[str, Any]]
    favorable_leaderboard: list[dict[str, Any]]
    unfavorable_leaderboard: list[dict[str, Any]]
    scorecard: dict[str, Any]
    coverage_and_schema_report: dict[str, Any]
    component_status_summary: dict[str, Any]
    name_resolution_report: dict[str, Any]


def load_snapshot_payloads(snapshot: DiscoveredSnapshot) -> SnapshotPayloads:
    """Refuses to load an invalid snapshot at all -- there is no partial
    rendering of a snapshot that failed integrity validation.
    """
    if not snapshot.integrity.valid:
        raise SnapshotPayloadError(
            f"refusing to load payloads for invalid snapshot {snapshot.directory_name}: "
            f"{snapshot.integrity.errors}"
        )
    outputs = snapshot.outputs_dir
    return SnapshotPayloads(
        snapshot=snapshot,
        public_score=_load_json(outputs / "public_score.json"),
        favorable_leaderboard=_load_json(outputs / "favorable_leaderboard.json"),
        unfavorable_leaderboard=_load_json(outputs / "unfavorable_leaderboard.json"),
        scorecard=_load_json(outputs / "scorecard.json"),
        coverage_and_schema_report=_load_json(outputs / "coverage_and_schema_report.json"),
        component_status_summary=_load_json(outputs / "component_status_summary.json"),
        name_resolution_report=_load_json(outputs / "name_resolution_report.json"),
    )


@dataclass(frozen=True)
class LeaderboardRow:
    official_rank: int
    batter_id: int
    batter_name: str | None
    contact_luck_runs_per_100: float
    lower_95_interval: float
    upper_95_interval: float
    interval_interpretation: str
    total_contact_luck_runs: float
    eligible_batted_balls: int
    games: int


def _leaderboard_row_from_record(record: dict[str, Any], rank_key: str) -> LeaderboardRow:
    return LeaderboardRow(
        official_rank=record[rank_key],
        batter_id=record["batter_id"],
        batter_name=record.get("batter_name"),
        contact_luck_runs_per_100=record["contact_luck_runs_per_100"],
        lower_95_interval=record["lower_95_interval"],
        upper_95_interval=record["upper_95_interval"],
        interval_interpretation=record["interval_interpretation"],
        total_contact_luck_runs=record["total_contact_luck_runs"],
        eligible_batted_balls=record["eligible_batted_balls"],
        games=record["games"],
    )


def build_favorable_leaderboard(payloads: SnapshotPayloads) -> list[LeaderboardRow]:
    """Sorted by the STORED official rank -- this is the frozen public-score
    contract's own ordering, never re-derived from the displayed metric.
    """
    rows = [
        _leaderboard_row_from_record(r, "official_rank_favorable")
        for r in payloads.favorable_leaderboard
    ]
    return sorted(rows, key=lambda r: r.official_rank)


def build_unfavorable_leaderboard(payloads: SnapshotPayloads) -> list[LeaderboardRow]:
    rows = [
        _leaderboard_row_from_record(r, "official_rank_unfavorable")
        for r in payloads.unfavorable_leaderboard
    ]
    return sorted(rows, key=lambda r: r.official_rank)


@dataclass(frozen=True)
class PlayerIndexEntry:
    batter_id: int
    batter_name: str | None


def build_player_index(payloads: SnapshotPayloads) -> list[PlayerIndexEntry]:
    """Every player in the snapshot (not just qualified ones) -- backs
    player-search, which must be able to find any player page by ID/name
    regardless of qualification status.
    """
    entries = [
        PlayerIndexEntry(batter_id=r["batter_id"], batter_name=r.get("batter_name"))
        for r in payloads.public_score
    ]
    return sorted(entries, key=lambda e: (e.batter_name or "", e.batter_id))


def find_player_record(payloads: SnapshotPayloads, batter_id: int) -> dict[str, Any] | None:
    for record in payloads.public_score:
        if record["batter_id"] == batter_id:
            return record
    return None


@dataclass(frozen=True)
class ComponentBreakdown:
    contact_per_100: float | None
    unexplained_residual_per_100: float | None
    defensive_execution_per_100: float | None
    advancement_per_100: float | None
    share_of_value_from_provisional_components: float | None
    component_status_reason_codes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlayerDetail:
    batter_id: int
    batter_name: str | None
    contact_luck_runs_per_100: float
    lower_95_interval: float
    upper_95_interval: float
    interval_interpretation: str
    total_contact_luck_runs: float
    eligible_batted_balls: int
    games: int
    qualification_status: str
    official_rank_favorable: int | None
    official_rank_unfavorable: int | None
    components: ComponentBreakdown


def build_player_detail(record: dict[str, Any]) -> PlayerDetail:
    return PlayerDetail(
        batter_id=record["batter_id"],
        batter_name=record.get("batter_name"),
        contact_luck_runs_per_100=record["contact_luck_runs_per_100"],
        lower_95_interval=record["lower_95_interval"],
        upper_95_interval=record["upper_95_interval"],
        interval_interpretation=record["interval_interpretation"],
        total_contact_luck_runs=record["total_contact_luck_runs"],
        eligible_batted_balls=record["eligible_batted_balls"],
        games=record["games"],
        qualification_status=record["qualification_status"],
        official_rank_favorable=record.get("official_rank_favorable"),
        official_rank_unfavorable=record.get("official_rank_unfavorable"),
        components=ComponentBreakdown(
            contact_per_100=record.get("contact_component_per_100"),
            unexplained_residual_per_100=record.get("unexplained_residual_component_per_100"),
            defensive_execution_per_100=record.get("defensive_execution_component_per_100"),
            advancement_per_100=record.get("advancement_execution_component_per_100"),
            share_of_value_from_provisional_components=record.get(
                "share_of_value_from_provisional_components"
            ),
            component_status_reason_codes=record.get("component_status_reason_codes", {}),
        ),
    )


@dataclass(frozen=True)
class TrendPoint:
    data_through_date: str
    snapshot_type: str
    contact_luck_runs_per_100: float
    lower_95_interval: float
    upper_95_interval: float
    eligible_batted_balls: int
    games: int
    qualification_status: str


def build_player_trend(
    history: list[SnapshotHistoryEntry],
    batter_id: int,
    *,
    payload_cache: dict[str, SnapshotPayloads] | None = None,
) -> list[TrendPoint]:
    """One point per historical date whose PREFERRED snapshot actually
    contains this batter_id -- never interpolated, never fabricated for a
    missing date (e.g. 2026-08-07, or a date before the player's first
    eligible batted ball). `snapshot_type` travels with each point so a
    future retrospective-backfill point can be rendered distinctly from a
    genuine/corrected one.
    """
    cache = payload_cache if payload_cache is not None else {}
    points: list[TrendPoint] = []
    for entry in sorted(history, key=lambda e: e.data_through_date):
        if entry.preferred is None:
            continue
        key = entry.preferred.directory_name
        if key not in cache:
            cache[key] = load_snapshot_payloads(entry.preferred)
        record = find_player_record(cache[key], batter_id)
        if record is None:
            continue
        points.append(
            TrendPoint(
                data_through_date=entry.data_through_date,
                snapshot_type=entry.preferred.snapshot_type,
                contact_luck_runs_per_100=record["contact_luck_runs_per_100"],
                lower_95_interval=record["lower_95_interval"],
                upper_95_interval=record["upper_95_interval"],
                eligible_batted_balls=record["eligible_batted_balls"],
                games=record["games"],
                qualification_status=record["qualification_status"],
            )
        )
    return points


@dataclass(frozen=True)
class StatusPageData:
    data_through_date: str
    snapshot_directory_name: str
    snapshot_label: str | None
    snapshot_type: str
    generated_at: str | None
    source_season: int | None
    score_version: str
    model_versions: dict[str, str]
    cache_decision: str | None
    cache_decision_reason: str | None
    coverage_missing_dates: list[str]
    final_pre_scoring_missing_dates: list[str]
    component_model_status: dict[str, Any]
    model_selection_winners: dict[str, Any]
    name_resolution_summary: dict[str, Any]
    integrity_valid: bool
    row_count: int
    qualified_count: int
    qualification_counts: dict[str, int]
    snapshot_history: list[SnapshotHistoryEntry] = field(default_factory=list)


def build_status_page_data(
    payloads: SnapshotPayloads, history: list[SnapshotHistoryEntry]
) -> StatusPageData:
    snapshot = payloads.snapshot
    coverage = payloads.coverage_and_schema_report
    cache_validation = coverage.get("raw_statcast_cache_coverage_validation", {})
    final_check = coverage.get("final_pre_scoring_coverage_check", {})
    scorecard = payloads.scorecard
    seasons = scorecard.get("seasons") or []
    return StatusPageData(
        data_through_date=snapshot.data_through_date,
        snapshot_directory_name=snapshot.directory_name,
        snapshot_label=snapshot.snapshot_label,
        snapshot_type=snapshot.snapshot_type,
        generated_at=snapshot.generated_at,
        source_season=seasons[0] if seasons else None,
        score_version=scorecard.get("score_version") or snapshot.score_version or "unknown",
        model_versions=snapshot.model_versions or {},
        cache_decision=cache_validation.get("decision"),
        cache_decision_reason=cache_validation.get("reason"),
        coverage_missing_dates=coverage.get("missing_dates_within_range", []),
        final_pre_scoring_missing_dates=final_check.get("missing_completed_game_dates", []),
        component_model_status=payloads.component_status_summary.get("component_model_status", {}),
        model_selection_winners=payloads.component_status_summary.get(
            "model_selection_winners", {}
        ),
        name_resolution_summary={
            "source": payloads.name_resolution_report.get("source"),
            "requested_batter_count": payloads.name_resolution_report.get("requested_batter_count"),
            "resolved_batter_count": payloads.name_resolution_report.get("resolved_batter_count"),
            "unresolved_batter_count": payloads.name_resolution_report.get(
                "unresolved_batter_count"
            ),
        },
        integrity_valid=snapshot.integrity.valid,
        row_count=scorecard.get("row_count", len(payloads.public_score)),
        qualified_count=scorecard.get("qualified_count", 0),
        qualification_counts=scorecard.get("qualification_counts", {}),
        snapshot_history=history,
    )


@dataclass(frozen=True)
class DashboardBuildManifest:
    """Presentation provenance for one dashboard build -- NOT a new model
    artifact. Records which snapshot the site was built from and when, so a
    viewer (or a future rebuild) can tell exactly what generated a given
    `dist/` output.
    """

    dashboard_version: str
    repository_commit: str
    preferred_snapshot_directory_name: str
    data_through_date: str
    snapshot_manifest_hash_reference: str
    build_timestamp: str
    player_count: int
    qualified_count: int


def _snapshot_manifest_hash_reference(snapshot: DiscoveredSnapshot) -> str:
    integrity_hashes = _load_json(snapshot.artifacts_dir / "integrity_hashes.json")
    value = integrity_hashes.get("artifacts/manifest.json", "unknown")
    return str(value)


def build_dashboard_manifest(
    *,
    repository_commit: str,
    snapshot: DiscoveredSnapshot,
    player_count: int,
    qualified_count: int,
    build_timestamp: str,
) -> DashboardBuildManifest:
    return DashboardBuildManifest(
        dashboard_version=DASHBOARD_VERSION,
        repository_commit=repository_commit,
        preferred_snapshot_directory_name=snapshot.directory_name,
        data_through_date=snapshot.data_through_date,
        snapshot_manifest_hash_reference=_snapshot_manifest_hash_reference(snapshot),
        build_timestamp=build_timestamp,
        player_count=player_count,
        qualified_count=qualified_count,
    )
