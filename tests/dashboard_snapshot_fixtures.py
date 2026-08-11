"""Shared synthetic snapshot-directory builders for the dashboard test
suite. Builds fake `outputs/`+`artifacts/` snapshot directory pairs shaped
exactly like a real Version 1.1 prospective snapshot, with real integrity
hashes computed over the files actually written -- never touches this
repo's real `outputs/prospective/`/`artifacts/prospective/` directories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from snapshot_data import compute_file_sha256

__all__ = ["default_player_record", "write_snapshot"]


def default_player_record(
    *,
    batter_id: int,
    batter_name: str | None,
    score: float,
    lower: float,
    upper: float,
    bbe: int = 300,
    games: int = 100,
    qualification_status: str = "qualified",
) -> dict[str, Any]:
    total_runs = score * bbe / 100.0
    if lower > 0:
        interpretation = "entirely_above_zero"
    elif upper < 0:
        interpretation = "entirely_below_zero"
    else:
        interpretation = "overlaps_zero"
    return {
        "batter_id": batter_id,
        "batter_name": batter_name,
        "season": 2026,
        "games": games,
        "eligible_batted_balls": bbe,
        "total_contact_luck_runs": total_runs,
        "contact_luck_runs_per_100": score,
        "lower_95_interval": lower,
        "upper_95_interval": upper,
        "interval_interpretation": interpretation,
        "qualification_status": qualification_status,
        "official_rank_eligible": qualification_status == "qualified",
        "official_rank_favorable": None,
        "official_rank_unfavorable": None,
        "total_contact_component_runs": total_runs * 0.5,
        "total_unexplained_residual_component_runs": total_runs * 0.1,
        "total_defensive_execution_component_runs": total_runs * 0.3,
        "total_advancement_component_runs": total_runs * 0.1,
        "contact_component_per_100": score * 0.5,
        "unexplained_residual_component_per_100": score * 0.1,
        "defensive_execution_component_per_100": score * 0.3,
        "advancement_execution_component_per_100": score * 0.1,
        "share_of_value_from_provisional_components": 0.4,
        "defense_unavailable_play_count": 0,
        "advancement_unavailable_play_count": 0,
        "component_status_reason_codes": {
            "contact": {"model_status_values": ["calibrated"], "reason_codes": []},
            "outfield_defense": {
                "model_status_values": ["provisional"],
                "reason_codes": ["near_wall_provisional"],
            },
            "infield_defense": {
                "model_status_values": ["calibrated_with_limited_subgroup_evidence"],
                "reason_codes": [],
            },
            "advancement": {
                "model_status_values": ["calibrated_with_limited_subgroup_evidence"],
                "reason_codes": [],
            },
        },
    }


def _leaderboard_record(player: dict[str, Any], rank_key: str, rank: int) -> dict[str, Any]:
    return {
        rank_key: rank,
        "batter_id": player["batter_id"],
        "batter_name": player["batter_name"],
        "season": player["season"],
        "contact_luck_runs_per_100": player["contact_luck_runs_per_100"],
        "lower_95_interval": player["lower_95_interval"],
        "upper_95_interval": player["upper_95_interval"],
        "interval_interpretation": player["interval_interpretation"],
        "eligible_batted_balls": player["eligible_batted_balls"],
        "games": player["games"],
        "total_contact_luck_runs": player["total_contact_luck_runs"],
    }


def write_snapshot(
    outputs_root: Path,
    artifacts_root: Path,
    *,
    directory_name: str,
    data_through_date: str,
    snapshot_label: str | None,
    generated_at: str,
    players: list[dict[str, Any]],
    corrupt: str | None = None,
    favorable_top_n: int | None = None,
    unfavorable_top_n: int | None = None,
) -> None:
    """Writes one full synthetic snapshot directory pair.

    `corrupt` selects a specific integrity-failure mode for negative tests:
    - "missing_manifest": artifacts/manifest.json is never written.
    - "bad_hash": one output file is rewritten after hashing, so its
      recorded hash no longer matches.
    - "mismatched_label": the manifest's own snapshot_label disagrees with
      the directory name's parsed label.
    - "malformed_manifest_json": manifest.json is truncated invalid JSON.
    - None: a fully valid snapshot.

    `favorable_top_n`/`unfavorable_top_n` mirror the optional `top_n` cut
    `mlb_luck_score.scoring.leaderboard._leaderboard` supports in
    production (unused there today -- both tables list the full qualified
    population -- but the parameter exists, so dashboard code must not
    silently assume the two tables always contain the same players). Use
    these to build a snapshot where the two leaderboard tables genuinely
    diverge, for tests that need to prove the dashboard's shared domain is
    derived from BOTH tables' displayed data, not just one.
    """
    out_dir = outputs_root / directory_name
    art_dir = artifacts_root / directory_name
    out_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    for player in players:
        player.setdefault("official_rank_favorable", None)
        player.setdefault("official_rank_unfavorable", None)

    qualified = [p for p in players if p["qualification_status"] == "qualified"]
    favorable_sorted = sorted(qualified, key=lambda p: -p["contact_luck_runs_per_100"])
    unfavorable_sorted = sorted(qualified, key=lambda p: p["contact_luck_runs_per_100"])
    favorable_records = [
        _leaderboard_record(p, "official_rank_favorable", i)
        for i, p in enumerate(favorable_sorted, start=1)
    ]
    unfavorable_records = [
        _leaderboard_record(p, "official_rank_unfavorable", i)
        for i, p in enumerate(unfavorable_sorted, start=1)
    ]
    if favorable_top_n is not None:
        favorable_records = favorable_records[:favorable_top_n]
    if unfavorable_top_n is not None:
        unfavorable_records = unfavorable_records[:unfavorable_top_n]
    for i, p in enumerate(favorable_sorted, start=1):
        p["official_rank_favorable"] = i
    for i, p in enumerate(unfavorable_sorted, start=1):
        p["official_rank_unfavorable"] = i

    manifest_label = snapshot_label
    if corrupt == "mismatched_label":
        manifest_label = (snapshot_label or "") + "_mismatch"

    manifest = {
        "manifest_version": "1.1.0",
        "repository_commit": "deadbeefcafefeed0000000000000000000000",
        "working_tree_clean": True,
        "prospective_season": 2026,
        "model_versions": {
            "contact": "baseline_v02",
            "outfield_defense": "measured_contact_only_v07",
            "infield_defense": "infield_hgb_v08",
            "advancement": "advancement_speed_v09",
        },
        "score_version": "0.12.0",
        "qualification_threshold_set": "primary",
        "requested_date_range": ["2026-03-25", data_through_date],
        "observed_date_coverage": ["2026-03-25", data_through_date],
        "data_through_date": data_through_date,
        "snapshot_label": manifest_label,
        "retrieval_timestamps": {"raw_statcast": generated_at},
        "source_hashes": {},
        "raw_row_counts": {"raw_statcast": len(players)},
        "processed_row_count": len(players),
        "schema_checks": {},
        "join_coverage": {},
        "deterministic_seeds": {"model_random_seed": 42, "bootstrap_seed": 42},
        "v1_seal_verification": {"verified": True},
        "generated_at": generated_at,
        "frozen_artifact_hashes": {},
        "output_hashes": {},
    }

    scorecard = {
        "score_version": "0.12.0",
        "model_version": manifest["model_versions"],
        "generated_at": generated_at,
        "data_through_date": [data_through_date],
        "seasons": [2026],
        "row_count": len(players),
        "qualified_count": len(qualified),
        "qualification_counts": {
            "qualified": len(qualified),
            "small_sample": len(players) - len(qualified),
        },
        "ranking_method": "competition_ranking_min_with_batter_id_tiebreak",
        "official_metric": "contact_luck_runs_per_100",
        "interval_method": "game_pk_clustered_percentile_bootstrap_95pct",
    }

    coverage_and_schema_report = {
        "statcast_schema_check": {"passed": True},
        "missing_dates_within_range": [],
        "data_through_completeness": {"complete": True},
        "raw_statcast_cache_coverage_validation": {
            "decision": "reused",
            "reason": "cached_coverage_verified_sufficient",
            "cached_provenance": {},
        },
        "raw_statcast_cached_provenance_before_decision": {},
        "raw_statcast_final_provenance": {},
        "final_pre_scoring_coverage_check": {
            "verified_at": generated_at,
            "observed_game_date_count": 1,
            "missing_completed_game_dates": [],
        },
        "venue_join": {},
        "geometry_join": {},
        "sprint_speed_join": {},
    }

    component_status_summary = {
        "model_selection_winners": {
            "outfield": "measured_contact_only_v07",
            "infield": "infield_hgb_v08",
            "advancement": "advancement_speed_v09",
        },
        "component_model_status": {
            "outfield": "False",
            "infield": "calibrated_with_limited_subgroup_evidence",
            "advancement": "calibrated_with_limited_subgroup_evidence",
        },
    }

    name_resolution_report = {
        "source": "MLB Stats API (/people)",
        "retrieved_at": generated_at,
        "requested_batter_count": len(players),
        "resolved_batter_count": len(players),
        "unresolved_batter_count": 0,
        "unresolved_batter_ids": [],
        "reason_codes_by_batter_id": {str(p["batter_id"]): "resolved" for p in players},
    }

    output_files = {
        "public_score.json": players,
        "favorable_leaderboard.json": favorable_records,
        "unfavorable_leaderboard.json": unfavorable_records,
        "scorecard.json": scorecard,
        "coverage_and_schema_report.json": coverage_and_schema_report,
        "component_status_summary.json": component_status_summary,
        "name_resolution_report.json": name_resolution_report,
    }
    for name, payload in output_files.items():
        (out_dir / name).write_text(json.dumps(payload, indent=2, default=str))

    if corrupt == "bad_hash":
        integrity_hashes = {
            f"outputs/{name}": compute_file_sha256(out_dir / name) for name in output_files
        }
        # Rewrite AFTER hashing so the recorded hash goes stale.
        (out_dir / "public_score.json").write_text(json.dumps(players + [{"tampered": True}]))
    else:
        integrity_hashes = {
            f"outputs/{name}": compute_file_sha256(out_dir / name) for name in output_files
        }

    if corrupt == "missing_manifest":
        (art_dir / "integrity_hashes.json").write_text(json.dumps(integrity_hashes, indent=2))
        return

    if corrupt == "malformed_manifest_json":
        (art_dir / "manifest.json").write_text("{not valid json")
        (art_dir / "integrity_hashes.json").write_text(json.dumps(integrity_hashes, indent=2))
        return

    (art_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    integrity_hashes["artifacts/manifest.json"] = compute_file_sha256(art_dir / "manifest.json")
    (art_dir / "integrity_hashes.json").write_text(json.dumps(integrity_hashes, indent=2))
