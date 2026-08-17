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
import demo_counterfactual_content as dcc
import explore_content as ec
import snapshot_data as sd
import visuals as v
from dashboard_config import (
    DASHBOARD_DIST_DIR,
    DASHBOARD_STATIC_DIR,
    DASHBOARD_TEMPLATES_DIR,
    DASHBOARD_VERSION,
    DEMO_COUNTERFACTUAL_GRID_PATH,
    DEMO_FIXTURE_PATH,
    EXPLORE_GAMES_DIR,
    EXPLORE_METADATA_PATH,
    EXPLORE_PLAYERS_DIR,
    EXPLORE_PLAYERS_PATH,
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
    return {"examples": examples, "simulator_field_svg": v.render_simulator_field_svg()}


def _homepage_proof_examples(page_data: dc.DemoPageData) -> list[dict[str, Any]]:
    """Version 1.3.3 homepage "BELIEVE" teaser -- the SAME two committed,
    hand-reviewed examples `_demo_view` renders on `/demo/`, reduced to just
    the fields the compact homepage proof strip needs. This reads directly
    from the already-loaded `page_data` (never a second fixture load, never
    a re-derivation) so the homepage numbers can never drift from `/demo/`'s
    own -- if the fixture ever changes, both pages update together from the
    one source of truth.
    """
    return [
        {
            "batter_name": ex.batter_name,
            "exit_velocity_mph": ex.exit_velocity_mph,
            "launch_angle_deg": ex.launch_angle_deg,
            "expected_run_value": ex.expected_run_value,
            "outcome_label": ex.outcome_label,
            "contact_luck_runs": ex.contact_luck_runs,
            "favorable": ex.contact_luck_runs >= 0,
        }
        for ex in page_data.examples
    ]


#: Display labels for the "Try It Yourself" simulator's probability rows AND
#: its realized-outcome buttons -- ONE list reused for both (the outcome
#: buttons are styled uppercase via CSS, not a second label set), matching
#: `demo_content.CLASS_DISPLAY_LABELS`' existing short labels for visual
#: consistency with the walkthrough above.
_SIMULATOR_OUTCOME_OPTIONS: list[dict[str, str]] = [
    {"cls": "out", "label": "Out"},
    {"cls": "single", "label": "1B"},
    {"cls": "double", "label": "2B"},
    {"cls": "triple", "label": "3B"},
    {"cls": "home_run", "label": "HR"},
]


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
    demo_counterfactual_grid_path: Path = DEMO_COUNTERFACTUAL_GRID_PATH,
    explore_players_path: Path = EXPLORE_PLAYERS_PATH,
    explore_players_dir: Path = EXPLORE_PLAYERS_DIR,
    explore_games_dir: Path = EXPLORE_GAMES_DIR,
    explore_metadata_path: Path = EXPLORE_METADATA_PATH,
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

    # Loaded here (before the index page renders) rather than down in the
    # "/demo/" section below, so the homepage's compact proof strip can read
    # from the SAME loaded `demo_page_data` `_demo_view` uses for `/demo/`
    # itself -- one fixture load, one source of truth, no second copy of
    # Greene's/Lindor's numbers that could drift from the committed fixture.
    demo_page_data = dc.load_demo_page_data(demo_fixture_path)

    # Version 1.4.0 Phase 4: the Play Explorer fixture is OPTIONAL at build
    # time (unlike the demo fixture/counterfactual grid above) -- a build
    # against a repo checkout that hasn't generated
    # `dashboard/explore_fixture/` yet (e.g. a fresh clone before `demo/
    # build_play_explorer_dev_ledger.py` has been run locally) must still
    # produce a complete, working site with every other page unaffected.
    # `explore_available` gates both the nav link (never a link that
    # predictably 404s) and the `/explore/`+`/plays/` generation below.
    #
    # Phase 4.1: if `explore-metadata.json` EXISTS (i.e. the fixture is
    # present at all), its `play_ledger_version`/`explorer_artifact_version`
    # are validated here -- a SECOND, independent fail-closed check (see
    # `explore_content.load_explore_metadata`'s docstring), after the
    # generator's own. A present-but-incompatible fixture is a hard build
    # failure, never silently skipped like a genuinely ABSENT one.
    #
    # Phase 4.2: the fixture is now sharded (`players.json` +
    # `players/<batter_id>.json`, replacing the monolithic
    # `search-index.json`) -- `ec.load_explore_catalog` loads and
    # cross-validates every shard (see that function's docstring for why
    # this offline build-time validation is unrelated to what the browser
    # itself fetches at runtime).
    explore_available = explore_players_path.exists()
    explore_data: ec.ExploreLoadedData | None = None
    if explore_available:
        explore_metadata = ec.load_explore_metadata(explore_metadata_path)
        explore_data = ec.load_explore_catalog(
            explore_players_path, explore_players_dir, explore_games_dir
        )
        if explore_metadata.play_count != explore_data.total_play_count:
            raise DashboardBuildError(
                f"explore-metadata.json play_count={explore_metadata.play_count} does not match "
                f"the actual total row count across players/*.json ({explore_data.total_play_count}) "
                "-- refusing to build a Play Explorer whose own provenance disagrees with its data."
            )
        if explore_metadata.player_count != len(explore_data.players):
            raise DashboardBuildError(
                f"explore-metadata.json player_count={explore_metadata.player_count} does not "
                f"match the actual players.json row count ({len(explore_data.players)}) -- "
                "refusing to build a Play Explorer whose own provenance disagrees with its data."
            )
        if explore_metadata.game_count != len(explore_data.game_pks):
            raise DashboardBuildError(
                f"explore-metadata.json game_count={explore_metadata.game_count} does not match "
                f"the actual distinct game_pk count referenced by players/*.json "
                f"({len(explore_data.game_pks)}) -- refusing to build a Play Explorer whose own "
                "provenance disagrees with its data."
            )

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
        "explore_available": explore_available,
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
            proof_examples=_homepage_proof_examples(demo_page_data),
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
    # file. `demo_page_data` was already loaded above (before the index page
    # render) so the homepage proof strip and this page share one load.
    demo_dir = out_dir / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    # "Try It Yourself" counterfactual grid (v1.3.1) -- same read-only
    # contract as the fixture above (see `demo/build_counterfactual_grid.py`
    # and `dashboard/demo_counterfactual_content.py`'s module docstrings).
    # Validated here, then the RAW committed file is copied byte-for-byte
    # into dist/demo/ for the browser to fetch once on page load -- never
    # re-serialized, so there is exactly one copy of this (large) grid on
    # the wire, identical to what was validated.
    dcc.load_counterfactual_grid_data(demo_counterfactual_grid_path)
    shutil.copy(demo_counterfactual_grid_path, demo_dir / "counterfactual-grid.json")

    (demo_dir / "index.html").write_text(
        env.get_template("demo.html").render(
            **base_context,
            active_page="demo",
            **_demo_view(demo_page_data),
            simulator_outcome_options=_SIMULATOR_OUTCOME_OPTIONS,
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

    # "/explore/" + "/plays/" -- Version 1.4.0 Phase 4 (routing revised in
    # Phase 4.1: a SINGLE static play-page shell, never one directory per
    # play -- see `dashboard/templates/play.html`/`dashboard/static/
    # play.js`'s module docstrings for why: at full-season scale (~95K-125K
    # scored plays), one HTML directory per play is an unnecessary,
    # operationally undesirable multiplication of files that are all just
    # an identical shell around the same client-side renderer). Reads ONLY
    # the committed, bounded Play Explorer fixture (see `demo/
    # build_play_explorer_fixture.py`/`dashboard/explore_content.py`'s
    # module docstrings); this build never scores anything, never derives
    # Rf, and never touches `mlb_luck_score`. `players.json`, every
    # `players/<batter_id>.json` and `games/<game_pk>.json` file, and
    # `explore-metadata.json` are copied byte-for-byte (already validated
    # above) -- never re-serialized, so the wire copy is identical to what
    # was validated.
    #
    # Phase 4.2: `players.json` replaced the monolithic search-index.json
    # (see `dashboard_config.EXPLORE_PLAYERS_PATH`'s docstring for the
    # 25 MiB Cloudflare Pages asset-size motivation). The Explorer landing
    # page fetches only `players.json` up front; a hitter's own
    # `players/<batter_id>.json` is fetched only after that hitter is
    # selected (`dashboard/static/explore.js`). The single play-page shell
    # is unchanged from Phase 4.1: it embeds NOTHING play-specific -- it
    # reads `?id=<play_id>` from the URL client-side, derives `game_pk`
    # from that same stable identity, and fetches ONLY that one
    # already-copied per-game JSON file, lazily, on page load.
    if explore_available and explore_data is not None:
        explore_dir = out_dir / "explore"
        explore_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(explore_players_path, explore_dir / "players.json")
        shutil.copy(explore_metadata_path, explore_dir / "explore-metadata.json")
        shutil.copytree(explore_players_dir, explore_dir / "players", dirs_exist_ok=True)
        shutil.copytree(explore_games_dir, explore_dir / "games", dirs_exist_ok=True)

        (explore_dir / "index.html").write_text(
            env.get_template("explore.html").render(**base_context, active_page="explore")
        )

        plays_dir = out_dir / "plays"
        plays_dir.mkdir(parents=True, exist_ok=True)
        (plays_dir / "index.html").write_text(
            env.get_template("play.html").render(**base_context, active_page=None)
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
