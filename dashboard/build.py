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

## Play Explorer: fail-closed by default (Version 1.4.0 Phase 5)

A bare invocation with no flags builds the Play Explorer DISABLED -- there
is no default Explorer artifact source, and in particular no implicit
fallback to the committed, bounded `dashboard/explore_fixture/`
development fixture. Pass `--explore-artifacts-dir DIR` to enable it, where
`DIR` contains `players.json`, `players/<batter_id>.json`,
`games/<game_pk>.json`, and `explore-metadata.json`:

    .venv/bin/python dashboard/build.py --explore-artifacts-dir dashboard/explore_fixture

for local/dev use of the committed fixture, or

    .venv/bin/python dashboard/build.py --explore-artifacts-dir outputs/explorer_build/<snapshot>

for a real snapshot's own generated artifacts (see
`scripts/generate_production_explorer_artifacts.py`, which
`scripts/publish_snapshot.sh` always invokes and passes explicitly before
building -- see `tests/test_dashboard_build_cli.py` for the regression
proving a normal production invocation can never publish
`dashboard/explore_fixture/` accidentally).
"""

from __future__ import annotations

import argparse
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
import spectrum_labels as sl
import visuals as v
from dashboard_config import (
    DASHBOARD_DIST_DIR,
    DASHBOARD_STATIC_DIR,
    DASHBOARD_TEMPLATES_DIR,
    DASHBOARD_VERSION,
    DEMO_COUNTERFACTUAL_GRID_PATH,
    DEMO_FIXTURE_PATH,
    OG_IMAGE_PATH,
    PROJECT_ROOT,
    SITE_URL,
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


def _spectrum_marker(
    *,
    batter_id: int,
    batter_name: str | None,
    score: float,
    url: str | None,
    domain: tuple[float, float],
    tier: str = "secondary",
    sublabel: str | None = None,
) -> dict[str, Any]:
    """One Luck Spectrum marker -- see `_macros.html::luck_spectrum`'s own
    docstring for the shape this feeds. `value_display` is always the
    plain signed number (never color-only), matching every other signed
    figure on this site (`formatSigned` in the JS, `"%+.2f"` in Jinja).

    `tier` is one of `"primary"` (the most prominent label on its side --
    always visible), `"secondary"` (a labeled but less prominent marker),
    or `"unlabeled"` (plotted as a small dot; its label only appears on
    hover/focus) -- see `spectrum_labels.select_spectrum_labels`/
    `label_tier` for how callers with more than a couple of candidates per
    side (`_homepage_spectrum`) decide which tier each marker gets.
    """
    return {
        "position_pct": v.spectrum_position_pct(score, domain),
        "value_display": f"{score:+.2f}",
        "label": batter_name or f"Player {batter_id}",
        "sublabel": sublabel,
        "favorable": score >= 0,
        "tier": tier,
        "url": url,
    }


#: Position-percent thresholds (0-100) beyond which a marker's label would
#: otherwise run past the track's own left/right edge and needs to grow
#: inward instead of staying centered on the marker -- see `_spectrum`.
_SPECTRUM_EDGE_MARGIN_PCT = 10.0


def _spectrum(markers: list[dict[str, Any]], domain: tuple[float, float]) -> dict[str, Any]:
    """Wraps a list of already-built markers with the shared zero position,
    and assigns each marker a horizontal "edge" alignment so its label
    doesn't run past the track's own left/right edge. Markers within
    `_SPECTRUM_EDGE_MARGIN_PCT` of either end get "edge"="start"/"end" so
    their label grows inward from the marker instead of staying centered
    on it (which would otherwise push text off the visible page).

    Two markers close in VALUE land close in PIXEL position too -- often
    close enough that their labels would overlap even side by side. That
    can't be resolved here at build time: it depends on each label's
    actual rendered width (font metrics, player name length) and the
    track's actual pixel width, neither of which exists yet in a static
    HTML string. It's handled at runtime instead, in the browser, by
    `declutterLuckSpectrum` (duplicated in `static/app.js` and
    `static/explore.js` per this codebase's existing convention for
    independent, unbundled <script> tags) -- it measures real
    `getBoundingClientRect()` positions after render and nudges only the
    labels that actually collide sideways, keeping every label on one row
    rather than stacking close markers into separate vertical bands
    (an earlier version did that; real user feedback was that it read as
    "these two names are unrelated" rather than "these two are close in
    value", the opposite of the intent).

    Mutates and returns the same marker dicts passed in -- these are
    freshly built by `_spectrum_marker` immediately before this call,
    never shared/cached elsewhere, so mutating them in place is safe.
    """
    for m in markers:
        if m["position_pct"] <= _SPECTRUM_EDGE_MARGIN_PCT:
            m["edge"] = "start"
        elif m["position_pct"] >= 100.0 - _SPECTRUM_EDGE_MARGIN_PCT:
            m["edge"] = "end"
        else:
            m["edge"] = "mid"
    return {"zero_pct": v.spectrum_position_pct(0.0, domain), "markers": markers}


def _homepage_spectrum(
    favorable_rows: list[c.LeaderboardRow],
    unfavorable_rows: list[c.LeaderboardRow],
    domain: tuple[float, float],
    root_prefix: str,
    *,
    top_n: int = 3,
    max_labels_per_side: int = 2,
) -> dict[str, Any]:
    """The homepage's Luck Spectrum: the top `top_n` most favorable plus the
    top `top_n` most unfavorable qualified players are plotted, one each (a
    player already shown on the favorable side is never duplicated on the
    unfavorable side -- only relevant when the qualified population itself
    is smaller than `2 * top_n`).

    Not every plotted player gets a visible label, though -- at most
    `max_labels_per_side` per side (see `spectrum_labels.
    select_spectrum_labels`, called once per side on that side's real
    values): the most extreme player is always labeled, a second only when
    it's adequately separated from the first. This is what keeps a
    tightly-clustered top-3 (e.g. three players within a few hundredths of
    each other) from producing overlapping label text -- the rest still
    plot as small unlabeled dots, so the spectrum still shows the shape of
    the top of the population, just without forcing every value into text.
    """
    seen: set[int] = set()
    markers: list[dict[str, Any]] = []
    for rows, side in ((favorable_rows, "favorable"), (unfavorable_rows, "unfavorable")):
        pool = [row for row in rows[:top_n] if row.batter_id not in seen]
        seen.update(row.batter_id for row in pool)
        if not pool:
            continue
        values = [row.contact_luck_runs_per_100 for row in pool]
        ranks = sl.select_spectrum_labels(values, max_labels=max_labels_per_side)
        for row, rank in zip(pool, ranks, strict=True):
            markers.append(
                _spectrum_marker(
                    batter_id=row.batter_id,
                    batter_name=row.batter_name,
                    score=row.contact_luck_runs_per_100,
                    url=f"{root_prefix}players/{row.batter_id}/",
                    domain=domain,
                    tier=sl.label_tier(rank),
                    sublabel=f"#{row.official_rank} {side}",
                )
            )
    return _spectrum(markers, domain)


def _player_spectrum(
    detail: c.PlayerDetail,
    domain: tuple[float, float],
    favorable_rows: list[c.LeaderboardRow],
    unfavorable_rows: list[c.LeaderboardRow],
    root_prefix: str,
) -> dict[str, Any]:
    """This player's own Luck Spectrum: their own marker (highlighted),
    plus the #1 most-favorable and #1 most-unfavorable qualified player as
    context anchors -- both real, already-loaded leaderboard rows, never
    invented -- so a visitor can see where this player sits relative to the
    extremes of the actual population, not just an isolated dot. Either or
    both anchors are skipped when they'd duplicate the player's own marker.
    """
    markers: list[dict[str, Any]] = []
    if favorable_rows and favorable_rows[0].batter_id != detail.batter_id:
        top = favorable_rows[0]
        markers.append(
            _spectrum_marker(
                batter_id=top.batter_id,
                batter_name=top.batter_name,
                score=top.contact_luck_runs_per_100,
                url=f"{root_prefix}players/{top.batter_id}/",
                domain=domain,
                sublabel="Most favorable",
            )
        )
    if unfavorable_rows and unfavorable_rows[0].batter_id != detail.batter_id:
        bottom = unfavorable_rows[0]
        markers.append(
            _spectrum_marker(
                batter_id=bottom.batter_id,
                batter_name=bottom.batter_name,
                score=bottom.contact_luck_runs_per_100,
                url=f"{root_prefix}players/{bottom.batter_id}/",
                domain=domain,
                sublabel="Most unfavorable",
            )
        )
    markers.append(
        _spectrum_marker(
            batter_id=detail.batter_id,
            batter_name=detail.batter_name,
            score=detail.contact_luck_runs_per_100,
            url=None,
            domain=domain,
            tier="primary",
            sublabel="This player",
        )
    )
    return _spectrum(markers, domain)


#: Minimum absolute per-100 score for the "Where the score came from"
#: generated summary sentence to be shown at all -- below this the sentence
#: would be dividing by a number close enough to zero that the resulting
#: percentage is not a meaningful description of anything (see
#: `_component_summary_sentence`'s own docstring).
_SUMMARY_SENTENCE_MIN_ABS_SCORE = 0.01


def _component_summary_sentence(
    component_value_rows: list[dict[str, Any]],
    total_score: float,
    batter_name: str,
) -> str | None:
    """A single deterministic, arithmetic-only sentence naming the
    largest-magnitude component and its share of the total score (e.g. "48%
    of Aaron Judge's favorable Contact Luck comes from contact outcomes.") --
    never a new number, only a percentage of two figures already shown
    verbatim in the component table below it. Returns `None` (template
    omits the sentence) whenever that percentage would not honestly
    describe the total: the total itself is too close to zero to normalize
    against, or the dominant component's sign disagrees with the total's
    own sign (which would produce a share outside a sane 0-100% range) --
    this can genuinely happen when components partly offset each other, and
    silently showing a nonsense percentage would be worse than omitting the
    sentence.
    """
    scored = [r for r in component_value_rows if r["value"] is not None]
    if not scored or abs(total_score) < _SUMMARY_SENTENCE_MIN_ABS_SCORE:
        return None
    dominant = max(scored, key=lambda r: abs(r["value"]))
    share = dominant["value"] / total_score
    if share <= 0 or share > 1.0001:
        return None
    direction = "favorable" if total_score >= 0 else "unfavorable"
    pct = round(min(share, 1.0) * 100)
    return f"{pct}% of {batter_name}'s {direction} Contact Luck comes from {dominant['label'].lower()}."


def _player_view(
    detail: c.PlayerDetail,
    trend_points: list[c.TrendPoint],
    qualified_intervals: Sequence[tuple[float, float]],
    *,
    qualified_count: int,
    favorable_rows: list[c.LeaderboardRow],
    unfavorable_rows: list[c.LeaderboardRow],
    root_prefix: str,
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
    component_value_rows: list[dict[str, Any]] = [
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
    batter_display_name = detail.batter_name or f"Player {detail.batter_id}"
    percentile = (
        c.compute_percentile(detail.official_rank_favorable, qualified_count)
        if detail.official_rank_favorable is not None
        else None
    )
    return {
        "qualified_count": qualified_count,
        "player": {
            "batter_id": detail.batter_id,
            "batter_name": batter_display_name,
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
            "percentile": percentile,
            "provisional_share": detail.components.share_of_value_from_provisional_components,
            "interval_svg": v.render_interval_bar_svg(
                point=detail.contact_luck_runs_per_100,
                lower=detail.lower_95_interval,
                upper=detail.upper_95_interval,
                domain=player_domain,
                compact=False,
            ),
        },
        "spectrum": _player_spectrum(
            detail, player_domain, favorable_rows, unfavorable_rows, root_prefix
        ),
        "component_value_rows": component_value_rows,
        # The shared scale every component bar in the (new, purely visual)
        # horizontal-bar breakdown is normalized against -- the largest
        # magnitude among this player's OWN components only (never compared
        # across players), so a component with no value (None, e.g. an
        # unavailable component) never affects the scale. 0.0 when every
        # component is None -- the template checks this before dividing.
        "component_max_abs": max(
            (abs(r["value"]) for r in component_value_rows if r["value"] is not None),
            default=0.0,
        ),
        "component_status_rows": component_status_rows,
        "summary_sentence": _component_summary_sentence(
            component_value_rows, detail.contact_luck_runs_per_100, batter_display_name
        ),
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
            "batter_id": ex.batter_id,
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
    explore_players_path: Path | None = None,
    explore_players_dir: Path | None = None,
    explore_games_dir: Path | None = None,
    explore_metadata_path: Path | None = None,
    explore_showcase_path: Path | None = None,
    explore_showcase_sensitivity_dir: Path | None = None,
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
    homepage_spectrum = _homepage_spectrum(
        favorable_rows, unfavorable_rows, qualified_domain, root_prefix
    )

    # Biggest movers (homepage): the immediately preceding stored snapshot's
    # payloads, looked up by walking `history` (newest first) to the entry
    # right after `latest`'s own date -- not simply `history[1]`, since
    # `history` (unlike `resolve_latest_snapshot`) does not exclude a date
    # whose only valid snapshot is a backfill, so the two lists can diverge
    # in principle. A missing/unloadable previous snapshot degrades to "no
    # movers module" (see `c.build_biggest_movers_by_direction`'s own
    # docstring), never a build failure -- this is presentation-only,
    # additive information.
    payload_cache: dict[str, c.SnapshotPayloads] = {latest.directory_name: payloads}
    previous_payloads: c.SnapshotPayloads | None = None
    history_dates = [h.data_through_date for h in history]
    if latest.data_through_date in history_dates:
        idx = history_dates.index(latest.data_through_date)
        if idx + 1 < len(history):
            prev_entry = history[idx + 1]
            if prev_entry.preferred is not None:
                try:
                    previous_payloads = payload_cache.setdefault(
                        prev_entry.preferred.directory_name,
                        c.load_snapshot_payloads(prev_entry.preferred),
                    )
                except c.SnapshotPayloadError:
                    previous_payloads = None
    biggest_movers = c.build_biggest_movers_by_direction(payloads, previous_payloads)

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

    # Version 1.4.0 Phase 4: the Play Explorer artifact directory is
    # OPTIONAL at build time (unlike the demo fixture/counterfactual grid
    # above) -- a build with no Explorer source given must still produce a
    # complete, working site with every other page unaffected.
    # `explore_available` gates both the nav link (never a link that
    # predictably 404s) and the `/explore/`+`/plays/` generation below.
    #
    # Phase 5 fail-closed rule: this function has NO default Explorer
    # source of its own -- `explore_players_path`/`explore_players_dir`/
    # `explore_games_dir`/`explore_metadata_path` default to `None`
    # (Explorer disabled), never to the committed, bounded
    # `dashboard/explore_fixture/` development fixture. A caller must pass
    # all four explicitly to enable the Explorer, whether that's the
    # committed fixture (local/dev/tests, see
    # `demo/build_play_explorer_fixture.py`) or a real production
    # snapshot's generated artifacts (see
    # `scripts/generate_production_explorer_artifacts.py`). The CLI wrapper
    # (`main()` below) mirrors this: a bare `dashboard/build.py` invocation
    # with no `--explore-artifacts-dir` flag builds with the Explorer
    # disabled, never falling back to the fixture -- see
    # `tests/test_dashboard_build_cli.py` for the regression proving a
    # normal production invocation can never publish
    # `dashboard/explore_fixture/` accidentally.
    #
    # Phase 4.1: if `explore-metadata.json` EXISTS (i.e. an Explorer source
    # was given), its `play_ledger_version`/`explorer_artifact_version` are
    # validated here -- a SECOND, independent fail-closed check (see
    # `explore_content.load_explore_metadata`'s docstring), after the
    # generator's own. A present-but-incompatible artifact set is a hard
    # build failure, never silently skipped like a genuinely ABSENT one.
    #
    # Phase 4.2: the artifact set is sharded (`players.json` +
    # `players/<batter_id>.json`, replacing the monolithic
    # `search-index.json`) -- `ec.load_explore_catalog` loads and
    # cross-validates every shard (see that function's docstring for why
    # this offline build-time validation is unrelated to what the browser
    # itself fetches at runtime).
    # Version 1.4.1: `showcase.json` (Showcase Plays) is now a REQUIRED
    # part of the artifact set -- `explore_showcase_path` joins the other
    # four in the `explore_available` check below, and `ec.
    # SUPPORTED_EXPLORER_ARTIFACT_VERSION`/`ec.load_explore_metadata`
    # already reject an older "2.0" artifact set (no showcase.json) as
    # incompatible before this function ever gets here.
    explore_available = (
        explore_players_path is not None
        and explore_players_dir is not None
        and explore_games_dir is not None
        and explore_metadata_path is not None
        and explore_showcase_path is not None
        and explore_players_path.exists()
    )
    explore_data: ec.ExploreLoadedData | None = None
    showcase_rows: list[ec.ShowcaseRow] | None = None
    if explore_available:
        assert explore_players_path is not None
        assert explore_players_dir is not None
        assert explore_games_dir is not None
        assert explore_metadata_path is not None
        assert explore_showcase_path is not None
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
        # `ec.load_showcase` itself reconciles favorable/unfavorable row
        # counts and rank contiguity against explore_metadata -- see that
        # function's docstring; no duplicate check needed here.
        showcase_rows = ec.load_showcase(explore_showcase_path, explore_metadata)
        # `showcase-sensitivity/<play_id>.json` files are OPTIONAL (a
        # showcase can have zero interactive plays, e.g. every true extreme
        # is missing a required raw measurement -- see `demo/
        # build_play_explorer_fixture.py`'s "Interactive eligibility"). Only
        # the rows this build ITSELF claims are interactive are validated;
        # an extra, unreferenced file in the sensitivity directory is not
        # this build's concern.
        if explore_showcase_sensitivity_dir is not None:
            for row in showcase_rows:
                if not row.interactive_available:
                    continue
                sensitivity_path = explore_showcase_sensitivity_dir / f"{row.play_id}.json"
                grid = ec.load_showcase_sensitivity(sensitivity_path)
                if grid.play_id != row.play_id:
                    raise DashboardBuildError(
                        f"showcase-sensitivity/{row.play_id}.json declares play_id="
                        f"{grid.play_id!r}, expected {row.play_id!r}"
                    )
        elif explore_metadata.showcase_interactive_count:
            raise DashboardBuildError(
                f"explore-metadata.json showcase_interactive_count="
                f"{explore_metadata.showcase_interactive_count} but no "
                "explore_showcase_sensitivity_dir was provided -- refusing to build a Play "
                "Explorer whose own provenance disagrees with its data."
            )

    env = _make_jinja_env()
    base_context = {
        "root_prefix": root_prefix,
        "asset_prefix": asset_prefix,
        "site_url": SITE_URL,
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
            homepage_spectrum=homepage_spectrum,
            biggest_movers=biggest_movers,
            most_fortunate=favorable_rows[0] if favorable_rows else None,
            most_unfortunate=unfavorable_rows[0] if unfavorable_rows else None,
            total_analyzed_bbe=sum(
                r.get("eligible_batted_balls") or 0 for r in payloads.public_score
            ),
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

    player_template = env.get_template("player.html")
    for entry in player_index:
        record = c.find_player_record(payloads, entry.batter_id)
        assert record is not None
        detail = c.build_player_detail(record)
        trend_points = c.build_player_trend(history, entry.batter_id, payload_cache=payload_cache)
        player_dir = out_dir / "players" / str(entry.batter_id)
        player_dir.mkdir(parents=True, exist_ok=True)
        view = _player_view(
            detail,
            trend_points,
            qualified_intervals,
            qualified_count=status_data.qualified_count,
            favorable_rows=favorable_rows,
            unfavorable_rows=unfavorable_rows,
            root_prefix=root_prefix,
        )
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
        assert explore_players_path is not None
        assert explore_players_dir is not None
        assert explore_games_dir is not None
        assert explore_metadata_path is not None
        assert explore_showcase_path is not None
        explore_dir = out_dir / "explore"
        explore_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(explore_players_path, explore_dir / "players.json")
        shutil.copy(explore_metadata_path, explore_dir / "explore-metadata.json")
        shutil.copy(explore_showcase_path, explore_dir / "showcase.json")
        shutil.copytree(explore_players_dir, explore_dir / "players", dirs_exist_ok=True)
        shutil.copytree(explore_games_dir, explore_dir / "games", dirs_exist_ok=True)
        if (
            explore_showcase_sensitivity_dir is not None
            and explore_showcase_sensitivity_dir.is_dir()
        ):
            shutil.copytree(
                explore_showcase_sensitivity_dir,
                explore_dir / "showcase-sensitivity",
                dirs_exist_ok=True,
            )

        (explore_dir / "index.html").write_text(
            env.get_template("explore.html").render(**base_context, active_page="explore")
        )

        plays_dir = out_dir / "plays"
        plays_dir.mkdir(parents=True, exist_ok=True)
        (plays_dir / "index.html").write_text(
            env.get_template("play.html").render(**base_context, active_page=None)
        )

    shutil.copytree(DASHBOARD_STATIC_DIR, out_dir / "static", dirs_exist_ok=True)
    # Copied to the dist ROOT (not under static/) so it serves at
    # `{SITE_URL}/og-image.png`, matching the `og:image`/`twitter:image` URLs
    # in `base.html` -- see `OG_IMAGE_PATH`'s docstring.
    shutil.copy(OG_IMAGE_PATH, out_dir / "og-image.png")

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


def _resolve_explore_paths(
    explore_artifacts_dir: Path | None,
) -> tuple[Path | None, Path | None, Path | None, Path | None, Path | None, Path | None]:
    """`--explore-artifacts-dir DIR` -> the six Explorer artifact paths
    `build_dashboard()` expects, all `None` if no directory was given.

    Deliberately generic over WHICH directory is passed -- the committed,
    bounded `dashboard/explore_fixture/` (local/dev/tests, opted into
    explicitly) and a real production snapshot's generated artifact
    directory (`scripts/generate_production_explorer_artifacts.py`) are
    both just directories with this same shape to this function. There is
    no implicit fallback to the fixture: `None` in, all-`None` out -- see
    `build_dashboard()`'s "Phase 5 fail-closed rule" docstring above.

    `showcase_sensitivity_dir` is returned even when that subdirectory
    doesn't exist on disk -- `build_dashboard()` itself treats a missing
    directory as "no interactive showcase plays," never an error (see its
    own docstring), so resolving the path unconditionally here keeps this
    function a pure, side-effect-free string-joining helper.
    """
    if explore_artifacts_dir is None:
        return None, None, None, None, None, None
    return (
        explore_artifacts_dir / "players.json",
        explore_artifacts_dir / "players",
        explore_artifacts_dir / "games",
        explore_artifacts_dir / "explore-metadata.json",
        explore_artifacts_dir / "showcase.json",
        explore_artifacts_dir / "showcase-sensitivity",
    )


def _build_cli_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--explore-artifacts-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the Play Explorer browser artifacts "
            "(players.json, players/<batter_id>.json, games/<game_pk>.json, "
            "explore-metadata.json, showcase.json, and optionally "
            "showcase-sensitivity/<play_id>.json). NOT defaulted -- omitting this flag "
            "builds the site with the Play Explorer disabled, it never falls back to the "
            "committed, bounded dashboard/explore_fixture/ development fixture. "
            "scripts/publish_snapshot.sh always passes this explicitly, pointing at "
            "that run's own generated production artifacts (see "
            "scripts/generate_production_explorer_artifacts.py). For local/dev use "
            "of the committed fixture, pass "
            "--explore-artifacts-dir dashboard/explore_fixture explicitly."
        ),
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    outputs_root: Path | None = None,
    artifacts_root: Path | None = None,
    out_dir: Path | None = None,
) -> BuildResult:
    """CLI entry point. `outputs_root`/`artifacts_root`/`out_dir` are
    injectable only for tests (e.g. to point a build at a synthetic
    snapshot without touching the real `outputs/prospective/v1_1/` on
    disk) -- ordinary invocations (including `scripts/publish_snapshot.sh`)
    never pass them and get `build_dashboard()`'s own real-path defaults.
    """
    args = _build_cli_arg_parser().parse_args(argv)
    players_path, players_dir, games_dir, metadata_path, showcase_path, showcase_sensitivity_dir = (
        _resolve_explore_paths(args.explore_artifacts_dir)
    )

    kwargs: dict[str, Any] = {
        "explore_players_path": players_path,
        "explore_players_dir": players_dir,
        "explore_games_dir": games_dir,
        "explore_metadata_path": metadata_path,
        "explore_showcase_path": showcase_path,
        "explore_showcase_sensitivity_dir": showcase_sensitivity_dir,
    }
    if outputs_root is not None:
        kwargs["outputs_root"] = outputs_root
    if artifacts_root is not None:
        kwargs["artifacts_root"] = artifacts_root
    if out_dir is not None:
        kwargs["out_dir"] = out_dir

    result = build_dashboard(**kwargs)
    print(f"[dashboard build] wrote {result.out_dir}")
    print(
        f"[dashboard build] preferred snapshot: {result.manifest.preferred_snapshot_directory_name}"
    )
    print(f"[dashboard build] data through: {result.manifest.data_through_date}")
    print(
        f"[dashboard build] players: {result.manifest.player_count} (qualified: {result.manifest.qualified_count})"
    )
    print(f"[dashboard build] explore available: {args.explore_artifacts_dir is not None}")
    if result.invalid_snapshot_count:
        print(
            f"[dashboard build] WARNING: {result.invalid_snapshot_count} invalid snapshot(s) excluded -- see warnings above"
        )
    return result


if __name__ == "__main__":
    main()
