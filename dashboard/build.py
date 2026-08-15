"""Contact Luck v1.2: the static-site build script.

Reads validated Version 1.1 prospective snapshots (via `snapshot_data`/
`content`), renders them into plain HTML + a couple of small JSON payloads
under `dashboard/dist/`, and stops. This script never downloads Statcast,
never trains or fits anything, never scores a play, and never writes into a
snapshot directory -- see `tests/test_dashboard_isolation.py` for the
structural no-scoring-import guarantee and
`tests/test_dashboard_build.py::TestDeterminismAndMutation` for the
read-only guarantee.

Run: `.venv/bin/python dashboard/build.py`. Output lands in
`dashboard/dist/` (gitignored, like every other build/output directory in
this repo) -- serve it locally with `python -m http.server` from that
directory, or point any static host at it (see the Phase 10 delivery
report for the recommended target).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import content as c
import demo_content as dc
import snapshot_data as sd
import visuals as v
from dashboard_config import (
    DASHBOARD_DIST_DIR,
    DASHBOARD_STATIC_DIR,
    DASHBOARD_TEMPLATES_DIR,
    DASHBOARD_VERSION,
    DEMO_FIXTURE_PATH,
    PROJECT_ROOT,
)
from jinja2 import Environment, FileSystemLoader, select_autoescape

__all__ = ["DashboardBuildError", "build_dashboard"]


class DashboardBuildError(Exception):
    """Raised when the build cannot proceed -- e.g. no valid snapshot found.
    Never a reason to fall back to an invalid/incomplete snapshot.
    """


def _get_repository_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(DASHBOARD_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _leaderboard_view_rows(
    rows: list[c.LeaderboardRow], domain: tuple[float, float], root_prefix: str
) -> list[dict[str, Any]]:
    view_rows = []
    for row in rows:
        view_rows.append(
            {
                "official_rank": row.official_rank,
                "batter_id": row.batter_id,
                "batter_name": row.batter_name or f"Player {row.batter_id}",
                "url": f"{root_prefix}players/{row.batter_id}/",
                "score": row.contact_luck_runs_per_100,
                "lower": row.lower_95_interval,
                "upper": row.upper_95_interval,
                "interval_width": row.upper_95_interval - row.lower_95_interval,
                "total_runs": row.total_contact_luck_runs,
                "bbe": row.eligible_batted_balls,
                "games": row.games,
                "interval_svg": v.render_interval_bar_svg(
                    point=row.contact_luck_runs_per_100,
                    lower=row.lower_95_interval,
                    upper=row.upper_95_interval,
                    domain=domain,
                    compact=True,
                ),
            }
        )
    return view_rows


_COMPONENT_LABELS = {
    "contact": "Contact",
    "outfield_defense": "Outfield defense",
    "infield_defense": "Infield defense",
    "advancement": "Advancement",
}


def _player_view(
    detail: c.PlayerDetail,
    trend_points: list[c.TrendPoint],
    qualified_intervals: Sequence[tuple[float, float]],
) -> dict[str, Any]:
    # Single-pass padding over the SAME raw interval set the leaderboard
    # domain is built from, plus this player's own raw (unpadded) interval --
    # never re-pad an already-padded domain (that silently shifts the zero
    # fraction and puts this page's bar on a different effective scale than
    # the player's own leaderboard row for the identical numbers). For a
    # qualified player, this player's interval is already inside
    # `qualified_intervals`, so `player_domain` comes out identical to the
    # leaderboard's shared domain; for a non-qualified player, the domain
    # widens (still with exactly one padding pass) to fit their own interval.
    player_domain = v.compute_interval_domain(
        [*qualified_intervals, (detail.lower_95_interval, detail.upper_95_interval)]
    )
    component_value_rows = [
        {"label": "Contact", "value": detail.components.contact_per_100},
        {"label": "Defensive execution", "value": detail.components.defensive_execution_per_100},
        {"label": "Advancement", "value": detail.components.advancement_per_100},
        {"label": "Unexplained residual", "value": detail.components.unexplained_residual_per_100},
    ]
    component_status_rows = [
        {
            "label": _COMPONENT_LABELS.get(key, key),
            "status_summary": c.summarize_component_status(value.get("model_status_values", [])),
            "model_status_values": value.get("model_status_values", []),
            "reason_codes": value.get("reason_codes", []),
        }
        for key, value in detail.components.component_status_reason_codes.items()
    ]
    return {
        "player": {
            "batter_id": detail.batter_id,
            "batter_name": detail.batter_name or f"Player {detail.batter_id}",
            "score": detail.contact_luck_runs_per_100,
            "lower": detail.lower_95_interval,
            "upper": detail.upper_95_interval,
            "interval_interpretation": detail.interval_interpretation,
            "total_runs": detail.total_contact_luck_runs,
            "bbe": detail.eligible_batted_balls,
            "games": detail.games,
            "qualification_status": detail.qualification_status,
            "official_rank_favorable": detail.official_rank_favorable,
            "official_rank_unfavorable": detail.official_rank_unfavorable,
            "provisional_share": detail.components.share_of_value_from_provisional_components,
            "interval_svg": v.render_interval_bar_svg(
                point=detail.contact_luck_runs_per_100,
                lower=detail.lower_95_interval,
                upper=detail.upper_95_interval,
                domain=player_domain,
                compact=False,
            ),
        },
        "component_value_rows": component_value_rows,
        "component_status_rows": component_status_rows,
        "trend_svg": v.render_trend_chart_svg(trend_points) if len(trend_points) >= 2 else None,
    }


def _demo_view(page_data: dc.DemoPageData) -> dict[str, Any]:
    examples = [
        {
            "example_id": ex.example_id,
            "contact_description": ex.contact_description,
            "batter_name": ex.batter_name,
            "batter_team": ex.batter_team,
            "opponent_team": ex.opponent_team,
            "game_date": ex.game_date,
            "description": ex.description,
            "exit_velocity_mph": ex.exit_velocity_mph,
            "launch_angle_deg": ex.launch_angle_deg,
            "probability_rows": ex.probability_rows,
            "expected_run_value": ex.expected_run_value,
            "outcome_label": ex.outcome_label,
            "observed_run_value": ex.observed_run_value,
            "contact_luck_runs": ex.contact_luck_runs,
            "explanation": ex.explanation,
            "field_svg": v.render_demo_field_svg(
                example_id=ex.example_id,
                spray_angle_deg=ex.spray_angle_deg,
                hit_distance_ft=ex.hit_distance_ft,
                favorable=ex.contact_luck_runs >= 0,
            ),
        }
        for ex in page_data.examples
    ]
    return {"examples": examples}


@dataclass(frozen=True)
class BuildResult:
    manifest: c.DashboardBuildManifest
    invalid_snapshot_count: int
    out_dir: Path


def build_dashboard(
    *,
    out_dir: Path = DASHBOARD_DIST_DIR,
    outputs_root: Path = sd.PROSPECTIVE_OUTPUTS_ROOT,
    artifacts_root: Path = sd.PROSPECTIVE_ARTIFACTS_ROOT,
    demo_fixture_path: Path = DEMO_FIXTURE_PATH,
    repo_root: Path = PROJECT_ROOT,
    build_timestamp: str | None = None,
) -> BuildResult:
    root_prefix = "/"
    asset_prefix = "/"
    build_timestamp = build_timestamp or datetime.now(UTC).isoformat()

    all_snapshots = sd.discover_snapshots(outputs_root=outputs_root, artifacts_root=artifacts_root)
    invalid = sd.invalid_snapshots(all_snapshots)
    for snap in invalid:
        print(
            f"[dashboard build] WARNING: excluding invalid snapshot {snap.directory_name}: {snap.integrity.errors}"
        )

    latest = sd.resolve_latest_snapshot(all_snapshots)
    if latest is None:
        raise DashboardBuildError(
            "No valid, non-backfill prospective snapshot was found under "
            f"{outputs_root} / {artifacts_root} -- refusing to build a dashboard with nothing to show."
        )

    payloads = c.load_snapshot_payloads(latest)
    history = sd.build_snapshot_history(all_snapshots)

    favorable_rows = c.build_favorable_leaderboard(payloads)
    unfavorable_rows = c.build_unfavorable_leaderboard(payloads)
    # The shared x-domain for every interval chart in this leaderboard view
    # (both tabs) must be derived from every CI endpoint actually displayed
    # in that view -- not just one table. `favorable_rows`/`unfavorable_rows`
    # happen to list the same qualified population today (both are built
    # with no `top_n` cut -- see `mlb_luck_score.scoring.leaderboard`), but
    # that is not a contract this module should silently depend on: a future
    # top-N cut on either table must not leave the other table's bars
    # clipped against a domain that never saw their data. Duplicate
    # intervals (the common case today) do not change the computed min/max,
    # so this is a no-op change in practice, only a correctness one.
    qualified_intervals = [
        (r.lower_95_interval, r.upper_95_interval) for r in (*favorable_rows, *unfavorable_rows)
    ]
    qualified_domain = v.compute_interval_domain(qualified_intervals)

    player_index = c.build_player_index(payloads)
    player_index_json = json.dumps(
        [
            {
                "batter_id": e.batter_id,
                "batter_name": e.batter_name,
                "url": f"{root_prefix}players/{e.batter_id}/",
            }
            for e in player_index
        ],
        sort_keys=True,
    )

    status_data = c.build_status_page_data(payloads, history)

    env = _make_jinja_env()
    base_context = {
        "root_prefix": root_prefix,
        "asset_prefix": asset_prefix,
        "data_through_date": latest.data_through_date,
        "data_through_date_display": date.fromisoformat(latest.data_through_date).strftime(
            "%b. %-d, %Y"
        ),
        "dashboard_version": DASHBOARD_VERSION,
        "snapshot_directory_name": latest.directory_name,
        "build_timestamp": build_timestamp,
        "player_index_json": player_index_json,
    }

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_template = env.get_template("index.html")
    (out_dir / "index.html").write_text(
        index_template.render(
            **base_context,
            active_page="leaderboard",
            favorable_rows=_leaderboard_view_rows(favorable_rows, qualified_domain, root_prefix),
            unfavorable_rows=_leaderboard_view_rows(
                unfavorable_rows, qualified_domain, root_prefix
            ),
            qualified_count=status_data.qualified_count,
            player_count=len(player_index),
        )
    )

    methodology_dir = out_dir / "methodology"
    methodology_dir.mkdir(parents=True, exist_ok=True)
    (methodology_dir / "index.html").write_text(
        env.get_template("methodology.html").render(**base_context, active_page="methodology")
    )

    # "/demo/" -- reads ONLY the committed, hand-reviewed demo fixture (see
    # `demo/build_demo_fixture.py` and `dashboard/demo_content.py`'s module
    # docstrings); this build never scores anything or regenerates that
    # file.
    demo_page_data = dc.load_demo_page_data(demo_fixture_path)
    demo_dir = out_dir / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    (demo_dir / "index.html").write_text(
        env.get_template("demo.html").render(
            **base_context, active_page="demo", **_demo_view(demo_page_data)
        )
    )

    status_dir = out_dir / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    (status_dir / "index.html").write_text(
        env.get_template("status.html").render(
            **base_context, active_page="status", status=status_data
        )
    )

    payload_cache: dict[str, c.SnapshotPayloads] = {latest.directory_name: payloads}
    player_template = env.get_template("player.html")
    for entry in player_index:
        record = c.find_player_record(payloads, entry.batter_id)
        assert record is not None
        detail = c.build_player_detail(record)
        trend_points = c.build_player_trend(history, entry.batter_id, payload_cache=payload_cache)
        player_dir = out_dir / "players" / str(entry.batter_id)
        player_dir.mkdir(parents=True, exist_ok=True)
        view = _player_view(detail, trend_points, qualified_intervals)
        (player_dir / "index.html").write_text(
            player_template.render(**base_context, active_page=None, **view)
        )

    shutil.copytree(DASHBOARD_STATIC_DIR, out_dir / "static", dirs_exist_ok=True)

    manifest = c.build_dashboard_manifest(
        repository_commit=_get_repository_commit(repo_root),
        snapshot=latest,
        player_count=len(player_index),
        qualified_count=status_data.qualified_count,
        build_timestamp=build_timestamp,
    )
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "dashboard_build_manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True)
    )

    return BuildResult(manifest=manifest, invalid_snapshot_count=len(invalid), out_dir=out_dir)


if __name__ == "__main__":
    result = build_dashboard()
    print(f"[dashboard build] wrote {result.out_dir}")
    print(
        f"[dashboard build] preferred snapshot: {result.manifest.preferred_snapshot_directory_name}"
    )
    print(f"[dashboard build] data through: {result.manifest.data_through_date}")
    print(
        f"[dashboard build] players: {result.manifest.player_count} (qualified: {result.manifest.qualified_count})"
    )
    if result.invalid_snapshot_count:
        print(
            f"[dashboard build] WARNING: {result.invalid_snapshot_count} invalid snapshot(s) excluded -- see warnings above"
        )
