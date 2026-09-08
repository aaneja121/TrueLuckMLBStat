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
import math
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import content as c
import demo_content as dc
import demo_counterfactual_content as dcc
import explore_content as ec
import pitcher_prototype_content as ppc
import snapshot_data as sd
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


#: Redesign Phase 1 numeric primitive. Contact Luck is a SIGNED quantity, so
#: every rendering of one carries its sign explicitly -- the sign glyph is
#: the non-colour channel for the favorable/unfavorable pair, and it is the
#: only channel a screen reader, a monochrome display, or a colour-blind
#: reader receives.
#:
#: The minus is U+2212 MINUS SIGN, not ASCII hyphen-minus. In a tabular-figure
#: face the true minus is drawn to the same width and at the same height as
#: the plus, so "-6.41" and "+7.62" align in a column; the hyphen is narrower
#: and sits lower, which is exactly the misalignment the Zero Spine's value
#: column cannot afford. See docs/design/typography.md § Numeric typography.
#:
#: SCOPE: Contact Luck values and run values ONLY. Exit velocity, launch
#: angle, probabilities, shares, counts and interval WIDTHS are not signed
#: quantities and must never be given a "+" -- a plus on an interval width
#: would be meaningless.
MINUS_SIGN = "\u2212"


def format_signed(value: float | None, digits: int = 2) -> str:
    """`+7.62` / `\u22126.41` / `+0.00`, with a true minus."""
    if value is None:
        return "\u2014"
    return f"{value:+.{digits}f}".replace("-", MINUS_SIGN)


def _make_jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(DASHBOARD_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["signed"] = format_signed
    return env


def _pct(fraction: float) -> str:
    """A CSS percentage for the Zero Spine plot field.

    Four decimals for the same reason `--cl-zero` carries four: at a ~900px
    field, one decimal is a ~0.9px error, which is visible as a mark sitting
    off the spine. Values are NOT clamped -- `.cl-scale-field` clips, and the
    clip state is reported separately so the row can draw a caret.
    """
    return f"{fraction * 100:.4f}%"


def _axis_ticks(scale: v.ZeroScale) -> list[dict[str, Any]]:
    """Labelled tick positions for the leaderboard's shared axis header.

    Rendered as HTML text at a real CSS size, deliberately not as SVG:
    `docs/design/guardrails.md` anti-pattern 11 bans SVG text below 12 CSS px
    after viewBox scaling, and a scaled axis label is exactly how the baseline
    produced 5.5px tick text on mobile. HTML text does not scale with a
    viewBox, so this surface cannot reproduce that defect.

    Step is chosen so the axis carries roughly 5-9 ticks whatever the span,
    always including zero. Zero is flagged so the header can mark it without
    the template deciding what zero is.
    """
    span = scale.span
    for step in (1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0):
        if span / step <= 9:
            break
    start = math.ceil(scale.domain_min / step) * step
    ticks: list[dict[str, Any]] = []
    value = start
    while value <= scale.domain_max + 1e-9:
        # `-0.0` formats as "-0.00"; normalise it away before it reaches a label.
        clean = 0.0 if abs(value) < 1e-9 else value
        ticks.append(
            {
                "value": clean,
                "label": format_signed(clean, 0 if step >= 1 else 1),
                "left": _pct(scale.fraction_of(clean)),
                "is_zero": abs(clean) < 1e-9,
            }
        )
        value += step
    return ticks


def _distribution_marks(
    rows: list[c.LeaderboardRow], scale: v.ZeroScale
) -> list[dict[str, Any]]:
    """Every qualified hitter as one mark on the shared axis.

    This is the league distribution the reader should be able to READ before
    reading any individual row (`docs/design/information-architecture.md`, the
    10-second budget). It is the same scale and the same percentage basis the
    rows below use, so the shape above and the rows beneath are literally the
    same measurement -- not an illustration of it.
    """
    marks = []
    for row in rows:
        point = row.contact_luck_runs_per_100
        marks.append(
            {
                "left": _pct(scale.clamped_fraction_of(point)),
                "favorable": point >= 0,
                "name": row.batter_name or f"Player {row.batter_id}",
                "score": point,
            }
        )
    return sorted(marks, key=lambda m: m["score"])


#: The standardized MLB "silo" headshot, keyed by MLBAM id: a square,
#: transparent-background head-and-shoulders portrait, framed identically for
#: every player (measured across 31 ids: content is flush to the frame's
#: bottom, with 6-10px of clearance above the cap at w_213).
#:
#: `d_people:generic:headshot:silo:current.png` is the CDN's OWN neutral
#: silhouette, served whenever a player has no photo -- so "no headshot" is a
#: portrait-shaped placeholder in the same framing, never a 404 or a broken
#: image. `app.js` adds a second, offline fallback (the player's initials) for
#: the case where the request itself fails.
#:
#: `f_auto` negotiates WebP where the browser accepts it (~2.9 KB per
#: portrait against ~14 KB of PNG) and serves PNG where it does not; both keep
#: the alpha channel, which is what lets the portrait sit on the page with no
#: plate, box, or circle behind it.
#:
#: This is the site's FIRST third-party request and first external dependency
#: -- the product/privacy decision `docs/design/information-architecture.md`
#: § "Imagery and team context" says must be raised explicitly rather than
#: made silently. It is a display asset only: no score, rank, interval or
#: probability depends on it, and every row is complete and correct with the
#: name alone (`DESIGN.md` rule 9 -- identity is name-first).
HEADSHOT_ORIGIN = "https://img.mlbstatic.com"
HEADSHOT_URL_TEMPLATE = (
    HEADSHOT_ORIGIN + "/mlb-photos/image/upload"
    "/d_people:generic:headshot:silo:current.png"
    "/w_120,q_auto:best,f_auto"
    "/v1/people/{batter_id}/headshot/silo/current"
)

#: Generational suffixes are not a surname: "Vladimir Guerrero Jr." initials
#: to VG, not VJ.
_NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})


def player_initials(name: str) -> str:
    """One or two letters for the offline portrait fallback.

    Never more than two, and never empty for a non-empty name -- this is the
    last thing standing where both the photo and the CDN's own silhouette are
    unavailable.
    """
    tokens = [t for t in re.split(r"[\s.]+", name or "") if t]
    tokens = [t for t in tokens if t.rstrip(".").lower() not in _NAME_SUFFIXES] or tokens
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0][0].upper()
    return (tokens[0][0] + tokens[-1][0]).upper()


def _leaderboard_view_rows(
    rows: list[c.LeaderboardRow], scale: v.ZeroScale, root_prefix: str
) -> list[dict[str, Any]]:
    """One dict per ranked row, carrying LAYOUT PERCENTAGES rather than an SVG.

    Redesign Phase 3. The leaderboard's interval bar was 124 independent
    inline SVGs, each scaling on its own inside a `margin: 0 auto` cell, so a
    row's absolute zero x depended on that cell's width -- which is why the
    baseline spine read as 124 unrelated ticks. These percentages feed the
    `.cl-scale` CSS primitive instead: every field shares one containing block
    and one percentage basis, so one rule at `--cl-zero` registers against all
    124 rows by construction.

    A layout percentage is not a score, rank, interval or probability, and no
    scoring code is imported to produce it (`CLAUDE.md` rule 6).
    """
    view_rows = []
    for row in rows:
        point = row.contact_luck_runs_per_100
        lower = row.lower_95_interval
        upper = row.upper_95_interval
        view_rows.append(
            {
                "official_rank": row.official_rank,
                "batter_id": row.batter_id,
                "batter_name": row.batter_name or f"Player {row.batter_id}",
                "url": f"{root_prefix}players/{row.batter_id}/",
                "portrait_url": HEADSHOT_URL_TEMPLATE.format(batter_id=row.batter_id),
                "initials": player_initials(row.batter_name or ""),
                "score": point,
                "lower": lower,
                "upper": upper,
                "interval_width": upper - lower,
                "total_runs": row.total_contact_luck_runs,
                "bbe": row.eligible_batted_balls,
                "games": row.games,
                # Sign follows the POINT ESTIMATE only, identically whether or
                # not the interval crosses zero (preserved product invariant).
                "favorable": point >= 0,
                "pt_pct": _pct(scale.clamped_fraction_of(point)),
                "lo_pct": _pct(scale.clamped_fraction_of(lower)),
                "hi_pct": _pct(scale.clamped_fraction_of(upper)),
                "clip_state": scale.clip_state(point=point, lower=lower, upper=upper),
                "point_state": scale.point_state(point),
            }
        )
    return view_rows


_COMPONENT_LABELS = {
    "contact": "Contact",
    "outfield_defense": "Outfield defense",
    "infield_defense": "Infield defense",
    "advancement": "Advancement",
}

#: Which model status entries belong to which DISPLAYED component value.
#: The snapshot records status per MODEL (`contact`, `outfield_defense`,
#: `infield_defense`, `advancement`); the decomposition a reader sees is per
#: VALUE, and defensive execution is one value produced by two models. Phase
#: 4 requires every displayed component value to carry its status
#: programmatically, so the two are joined here rather than shown as the
#: baseline's two disconnected tables -- one listing values, one listing
#: statuses, with nothing tying a row in either to a row in the other.
_COMPONENT_VALUE_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Contact", ("contact",)),
    ("Defensive execution", ("outfield_defense", "infield_defense")),
    ("Advancement", ("advancement",)),
    ("Unexplained residual", ()),
)

#: Why a hitter holds no official rank, per status. NOT one sentence for all
#: of them: the baseline said "hasn't yet reached the minimum eligible
#: batted-ball threshold" for every unqualified hitter, which is simply false
#: for three of the four statuses -- Bobby Witt Jr. is
#: `provisionally_qualified` with 325 eligible BBE, MORE than the #1 ranked
#: hitter's 321. Saying "too few batted balls" there misdescribes the data.
#:
#: These restate `mlb_luck_score.scoring.qualification`'s own documented
#: precedence in plain language and introduce no new rule, threshold or
#: claim. They are duplicated here rather than imported because no module
#: under `dashboard/` may import scoring code (`CLAUDE.md` rule 6, enforced
#: by `tests/test_dashboard_isolation.py`) -- the same convention
#: `content.py`'s component-status labels already follow. If the precedence
#: in that module changes, this map changes with it.
_QUALIFICATION_EXPLANATIONS: dict[str, str] = {
    "small_sample": (
        "This player is below the minimum number of eligible batted balls and scored games "
        "the official leaderboard requires. Not enough plays to rank, however clean the ones "
        "there are."
    ),
    "insufficient_component_coverage": (
        "There are enough plays here. Too much of this player's batted-ball profile falls "
        "outside both the defense and advancement models for the official leaderboard, which "
        "is a different problem from a small sample."
    ),
    # Position-neutral on purpose: this paragraph moved below the sample line
    # so it would stop delaying the score, and "the share printed below" was
    # then pointing upward. It names the figure instead of its location, the
    # same correction the leaderboard's axis caption already carries.
    "provisionally_qualified": (
        "This player clears the volume and coverage requirements. Too much of the value "
        "comes through provisional or limited-evidence model pathways (the “From provisional "
        "components” figure), or too many plays needed a fallback for a missing input. Where "
        "the value comes from is what holds this player off the official leaderboard, not "
        "how large the sample is."
    ),
    "not_reportable": (
        "No eligible batted balls in this snapshot, so there is no rate to compute or rank."
    ),
}

#: What the residual row says where the other rows carry a model status. It
#: has none, and inventing one would misrepresent it: the residual is what
#: the three modelled components do not account for, so "no model status"
#: is the honest statement, not a gap to be filled.
_RESIDUAL_STATUS_NOTE = "Not modelled. What the three components above leave over."


def _component_view(
    detail: c.PlayerDetail, zero_fraction: float
) -> dict[str, Any]:
    """The component decomposition, on the `component_per_100` scale.

    A per-player domain (Invariant D does not reach a component breakdown --
    see `ZeroScale.from_values`), padded so its zero lands on the SAME
    `--cl-zero` the hero figure and the leaderboard use. Because the scale is
    this player's own, the figure owes visible tick labels carrying its unit,
    and the template renders them.
    """
    values_by_label: dict[str, float | None] = {
        "Contact": detail.components.contact_per_100,
        "Defensive execution": detail.components.defensive_execution_per_100,
        "Advancement": detail.components.advancement_per_100,
        "Unexplained residual": detail.components.unexplained_residual_per_100,
    }
    statuses = detail.components.component_status_reason_codes

    known = [v for v in values_by_label.values() if v is not None]
    scale = v.ZeroScale.from_values(
        "component_per_100",
        "Runs / 100 eligible BBE",
        known + [detail.contact_luck_runs_per_100],
        zero_fraction=zero_fraction,
    )

    def bar(value: float) -> dict[str, Any]:
        """A bar runs FROM zero TO the value -- it is not an interval, and
        it must not be drawn with the interval's vocabulary."""
        low, high = sorted((scale.zero_fraction, scale.clamped_fraction_of(value)))
        return {"lo_pct": _pct(low), "hi_pct": _pct(high)}

    rows: list[dict[str, Any]] = []
    for label, source_keys in _COMPONENT_VALUE_SOURCES:
        value = values_by_label[label]
        model_status_values: list[str] = []
        reason_codes: list[str] = []
        for key in source_keys:
            entry = statuses.get(key, {})
            model_status_values += entry.get("model_status_values", [])
            reason_codes += entry.get("reason_codes", [])
        rows.append(
            {
                "label": label,
                "value": value,
                "favorable": value is not None and value >= 0,
                "status_summary": (
                    _RESIDUAL_STATUS_NOTE
                    if not source_keys
                    else c.summarize_component_status(sorted(set(model_status_values)))
                ),
                "has_model_status": bool(source_keys),
                "model_status_values": sorted(set(model_status_values)),
                "reason_codes": sorted(set(reason_codes)),
                "source_labels": [_COMPONENT_LABELS.get(k, k) for k in source_keys],
                **(bar(value) if value is not None else {"lo_pct": None, "hi_pct": None}),
            }
        )

    # The components sum to the headline by construction; drawing the total
    # on the SAME scale is what makes that visible rather than asserted.
    # `sum_of_parts` is carried so a test can hold the claim to account, and
    # so the template never has to add anything up itself.
    return {
        "rows": rows,
        "total": {
            "label": "Total Contact Luck",
            "value": detail.contact_luck_runs_per_100,
            "favorable": detail.contact_luck_runs_per_100 >= 0,
            **bar(detail.contact_luck_runs_per_100),
        },
        "sum_of_parts": sum(known),
        "unit_label": scale.unit_label,
        "ticks": _axis_ticks(scale),
        "domain_min": scale.domain_min,
        "domain_max": scale.domain_max,
    }


def _player_view(
    detail: c.PlayerDetail,
    trend_points: list[c.TrendPoint],
    scale: v.ZeroScale,
    *,
    league: dict[str, Any],
    play_count: int | None,
    root_prefix: str,
) -> dict[str, Any]:
    # Redesign Phase 1, decision D1a: this page uses the ONE canonical
    # `league_per_100` scale, built from the qualified comparison
    # population -- the identical scale the leaderboard row for this same
    # player is drawn on. Invariant D.
    #
    # This replaces a per-player domain that was widened, with one padding
    # pass, to fit a non-qualified player's own (often very wide,
    # small-sample) interval. That widening kept the bar from clipping, but
    # it moved the zero line to a different fraction of the field on those
    # pages -- so the same page could not be read against the leaderboard,
    # and under the Zero Spine direction it is the one place the site would
    # draw two different zeros. A value outside the canonical domain now
    # CLIPS, with an explicit continuation caret and the true numbers intact
    # in text; the scale never moves to accommodate it. Qualified players are
    # unaffected: their interval was already inside this domain.
    #
    # Redesign Phase 4: the hero draws through the `.cl-scale` CSS primitive
    # (renderer A), not an inline SVG, for the same reason the leaderboard
    # does -- the axis rail, the league distribution and this hitter's own
    # mark then share one containing block and one percentage basis, so the
    # amber rule descending the figure registers against all three by
    # construction rather than by arithmetic.
    point = detail.contact_luck_runs_per_100
    point_fraction = scale.clamped_fraction_of(point)

    # Where the headline numeral sits. The plan's guard against this hero
    # becoming a KPI panel is that the number sits AT the mark rather than in
    # a tile of its own; near either edge, anchoring it centred would push it
    # off the field, so the anchor -- not the position -- changes.
    if point_fraction < 0.14:
        value_anchor = "start"
    elif point_fraction > 0.86:
        value_anchor = "end"
    else:
        value_anchor = "center"

    qualified = detail.qualification_status == "qualified"
    return {
        "qualification_explanation": _QUALIFICATION_EXPLANATIONS.get(
            detail.qualification_status,
            "This player does not meet the official leaderboard's requirements, so no rank "
            "is assigned.",
        ),
        "player": {
            "batter_id": detail.batter_id,
            "batter_name": detail.batter_name or f"Player {detail.batter_id}",
            "score": point,
            "lower": detail.lower_95_interval,
            "upper": detail.upper_95_interval,
            "interval_interpretation": detail.interval_interpretation,
            "total_runs": detail.total_contact_luck_runs,
            "bbe": detail.eligible_batted_balls,
            "games": detail.games,
            "qualification_status": detail.qualification_status,
            "qualified": qualified,
            "official_rank_favorable": detail.official_rank_favorable,
            "official_rank_unfavorable": detail.official_rank_unfavorable,
            "provisional_share": detail.components.share_of_value_from_provisional_components,
            # Sign follows the POINT ESTIMATE only, identically whether or not
            # the interval crosses zero (preserved product invariant).
            "favorable": point >= 0,
            "pt_pct": _pct(point_fraction),
            "lo_pct": _pct(scale.clamped_fraction_of(detail.lower_95_interval)),
            "hi_pct": _pct(scale.clamped_fraction_of(detail.upper_95_interval)),
            "clip_state": scale.clip_state(
                point=point,
                lower=detail.lower_95_interval,
                upper=detail.upper_95_interval,
            ),
            "point_state": scale.point_state(point),
            "value_anchor": value_anchor,
        },
        "league": league,
        "components": _component_view(detail, scale.zero_fraction),
        "trend": v.build_trend_figure(trend_points, league_scale=scale),
        "trend_point_count": len(trend_points),
        # The play-level path (`docs/design/information-architecture.md`
        # question 6). A link is emitted ONLY when this hitter actually has
        # an entry in the Play Explorer catalog, so the page never offers a
        # route that lands on an empty selection.
        "plays": (
            {"url": f"{root_prefix}explore/?batter={detail.batter_id}", "count": play_count}
            if play_count
            else None
        ),
    }


#: Snapshot-type display labels. The raw values are `snake_case` schema
#: tokens (`snapshot_data.classify_snapshot_type`); a reader gets the words.
#: Display only -- nothing here changes precedence or which snapshot is used.
_SNAPSHOT_TYPE_LABELS: dict[str, str] = {
    "genuine_prospective_snapshot": "Genuine prospective",
    "corrected_prospective_snapshot": "Corrected prospective",
    "retrospective_backfill": "Retrospective backfill",
    "invalid_incomplete_snapshot": "Invalid or incomplete",
}

#: Qualification-status display labels, matching `mlb_luck_score.scoring.
#: qualification`'s own five statuses. Duplicated rather than imported for
#: the same reason `_QUALIFICATION_EXPLANATIONS` is (CLAUDE.md rule 6).
_QUALIFICATION_STATUS_LABELS: dict[str, str] = {
    "qualified": "Qualified",
    "provisionally_qualified": "Provisionally qualified",
    "small_sample": "Small sample",
    "insufficient_component_coverage": "Insufficient component coverage",
    "not_reportable": "Not reportable",
}


_KNOWN_STATUS_LABELS: frozenset[str] = frozenset(
    {
        "Calibrated",
        "Not calibrated",
        "Provisional",
        "Limited subgroup evidence",
        "Unavailable",
    }
)


def _humanize_token(value: str) -> str:
    """A `snake_case` schema token as words, first letter capitalised."""
    words = str(value).replace("_", " ").strip()
    return words[:1].upper() + words[1:] if words else words


def _display_date(iso_date: str | None) -> str:
    """`2026-08-14` -> `Aug. 14, 2026`, matching the header's own badge."""
    if not iso_date:
        return "\u2014"
    try:
        return date.fromisoformat(iso_date).strftime("%b. %-d, %Y")
    except ValueError:
        return iso_date


def _display_timestamp(iso_timestamp: str | None) -> str:
    """`2026-08-15T13:35:37.370905+00:00` -> `Aug. 15, 2026, 13:35 UTC`.

    Design principle 8, *format the data rather than engineering around it*:
    the raw value is a 32-character unbreakable token that collapsed the
    status history table's last columns to ~31px at narrow widths. The ISO
    value is not lost -- it stays in the `<time datetime>` attribute. This
    reformats an already-recorded string; it derives no new fact.
    """
    if not iso_timestamp:
        return "\u2014"
    try:
        parsed = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return iso_timestamp
    if parsed.tzinfo is None:
        return parsed.strftime("%b. %-d, %Y, %H:%M")
    return parsed.astimezone(UTC).strftime("%b. %-d, %Y, %H:%M UTC")


def _snapshot_type_label(snapshot_type: str) -> str:
    return _SNAPSHOT_TYPE_LABELS.get(snapshot_type, _humanize_token(snapshot_type))


#: The snapshot records model information under THREE key vocabularies that
#: do not line up: `model_versions` is keyed per model
#: (`infield_defense_expected_winner`), `component_model_status` per
#: component (`infield`), and `model_selection_winners` per component again
#: plus `near_wall_specialist`, which has no version entry at all. The
#: baseline shipped them as three disconnected lists, so a reader had to do
#: the join themselves and could not tell which status went with which
#: version. This maps every known key onto one canonical component, in the
#: order the score is actually built. Unknown keys are never dropped -- they
#: fall through to their own row (see `_model_rows`), so a future schema
#: addition appears rather than disappearing.
_MODEL_COMPONENT_CANONICAL: dict[str, str] = {
    "contact": "Contact",
    "outfield": "Outfield defense",
    "outfield_defense": "Outfield defense",
    "infield": "Infield defense",
    "infield_defense": "Infield defense",
    "infield_defense_expected_winner": "Infield defense",
    "advancement": "Advancement",
    "advancement_expected_winner": "Advancement",
    "near_wall_specialist": "Near-wall specialist",
}
_MODEL_COMPONENT_ORDER: tuple[str, ...] = (
    "Contact",
    "Outfield defense",
    "Infield defense",
    "Advancement",
    "Near-wall specialist",
)


def _model_rows(status: c.StatusPageData) -> list[dict[str, Any]]:
    """One row per component, joining version, recorded status and selection.

    Display only. `summarize_component_status` is the dashboard's existing
    fail-soft labeller: a documented status becomes its public label, and
    anything else comes back close to verbatim, which is the honest handling
    for a value this dashboard is not entitled to reinterpret.
    """
    rows: dict[str, dict[str, Any]] = {}

    def cell(key: str) -> dict[str, Any]:
        label = _MODEL_COMPONENT_CANONICAL.get(key) or _humanize_token(key)
        return rows.setdefault(
            label, {"component": label, "version": None, "status": None, "selection": None}
        )

    for name, version in status.model_versions.items():
        cell(name)["version"] = str(version)
    for name, value in status.component_model_status.items():
        raw = str(value)
        entry = cell(name)
        entry["status"] = c.summarize_component_status([raw])
        #: True when the snapshot recorded something outside the documented
        #: status vocabulary. It is shown in the identifier register rather
        #: than reworded, so it never reads as a label this site chose.
        entry["status_is_raw"] = entry["status"] not in _KNOWN_STATUS_LABELS
    for name, winner in status.model_selection_winners.items():
        cell(name)["selection"] = str(winner)

    ordered = [rows.pop(label) for label in _MODEL_COMPONENT_ORDER if label in rows]
    return ordered + [rows[label] for label in sorted(rows)]


def _history_gaps(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Interleave the snapshot history with an explicit marker for any date
    that holds no valid snapshot.

    A date missing from the history is a real provenance fact, and the
    reader cannot tell an intentional gap from a rendering bug if the list
    simply skips from the 8th to the 6th. This states it. It never asserts
    a cause, because the dashboard does not know one, and it never
    interpolates: a gap is drawn as a gap.

    Calendar arithmetic over dates the snapshot already recorded. No score,
    rank, interval or probability is derived here.
    """
    rows: list[dict[str, Any]] = []
    previous: date | None = None
    for entry in history:
        try:
            current = date.fromisoformat(entry["data_through_date"])
        except (TypeError, ValueError):
            rows.append({"kind": "snapshot", "entry": entry})
            previous = None
            continue
        if previous is not None and (previous - current).days > 1:
            newest_missing = previous - timedelta(days=1)
            oldest_missing = current + timedelta(days=1)
            label = (
                _display_date(oldest_missing.isoformat())
                if newest_missing == oldest_missing
                else f"{_display_date(oldest_missing.isoformat())} to "
                f"{_display_date(newest_missing.isoformat())}"
            )
            rows.append({"kind": "gap", "label": label})
        rows.append({"kind": "snapshot", "entry": entry})
        previous = current
    return rows


def _status_view(status: c.StatusPageData) -> dict[str, Any]:
    """Presentation-only view model for `/status/`.

    Every value here is a display FORM of something already in the snapshot:
    a date reformatted, a `snake_case` token turned into words, a dict turned
    into an ordered list of rows. No status, count, threshold or precedence
    decision is made here -- `content.build_status_page_data` already made
    them, and this never disagrees with it.
    """
    history = [
        {
            "data_through_date": entry.data_through_date,
            "data_through_display": _display_date(entry.data_through_date),
            "snapshots": [
                {
                    "directory_name": snap.directory_name,
                    "snapshot_type": snap.snapshot_type,
                    "type_label": _snapshot_type_label(snap.snapshot_type),
                    "generated_at": snap.generated_at,
                    "generated_at_display": _display_timestamp(snap.generated_at),
                    "is_displayed": bool(
                        entry.preferred and snap.directory_name == entry.preferred.directory_name
                    ),
                    #: The one snapshot this build was rendered from. Amber
                    #: means "the frozen official record", so exactly one row
                    #: in the history carries it -- not every published row.
                    "is_current": snap.directory_name == status.snapshot_directory_name,
                }
                for snap in entry.all_valid_snapshots
            ],
        }
        for entry in status.snapshot_history
    ]
    return {
        "data_through_display": _display_date(status.data_through_date),
        "generated_at_display": _display_timestamp(status.generated_at),
        "snapshot_type_label": _snapshot_type_label(status.snapshot_type),
        "model_rows": _model_rows(status),
        "qualification_rows": [
            {
                "label": _QUALIFICATION_STATUS_LABELS.get(name, _humanize_token(name)),
                "count": count,
            }
            for name, count in sorted(
                status.qualification_counts.items(), key=lambda kv: -kv[1]
            )
        ],
        "history": _history_gaps(history),
        "snapshot_count": sum(len(entry["snapshots"]) for entry in history),
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


# ══ Version 0.13.1 LOCAL PROTOTYPE · pitcher views ═══════════════════════
#
# Every function below is a PROJECTION of already-scored fixture values into
# layout percentages and display strings. Nothing here computes a score, a
# rank, an interval or a probability: `board_rank` is a position in a list
# the fixture's own cumulative totals already ordered, and a percentage is a
# layout coordinate (the same argument `cl_zero_fraction_css` carries).

#: The board's ordering quantity, named once. Every label on the surface
#: that says what is being ranked reads from here, so no template can
#: describe the board as ranking something it does not.
PITCHER_RANKED_QUANTITY_LABEL = "Cumulative Contact Luck allowed, runs"

_PITCHER_BOARD_DEFINITIONS: dict[str, str] = {
    "starter_like": (
        "Observed usage of at least 10 resolved batted balls per appearance. A description "
        "of how these pitchers were used this season, not a roster role."
    ),
    "reliever_like": (
        "Observed usage of 8 or fewer resolved batted balls per appearance. A description "
        "of how these pitchers were used this season, not a roster role — no pitcher here "
        "is identified as a closer."
    ),
}

_BB_TYPE_LABELS: dict[str, str] = {
    "fly_ball": "fly ball",
    "line_drive": "line drive",
    "ground_ball": "ground ball",
    "popup": "popup",
}

#: Outcome names in plain words. These restate `mlb_luck_score.eligibility`'s
#: own outcome classes for a reader and introduce no new class -- duplicated
#: rather than imported, the same convention `_QUALIFICATION_EXPLANATIONS`
#: above follows, because no module under `dashboard/` may import scoring
#: code.
_OUTCOME_LABELS: dict[str, str] = {
    "out": "out",
    "single": "single",
    "double": "double",
    "triple": "triple",
    "home_run": "home run",
}

#: Why a pitcher-season is on no board. Two different reasons, said
#: differently, because collapsing them into one sentence would be false for
#: whichever half it did not describe. Neither is a qualification rule: the
#: first is a display choice about a ranked board, and the second is the
#: honest consequence of a two-population split that does not claim to cover
#: a continuum.
_REASON_MIXED_USAGE = (
    "Shown here but on neither board. This season's batted balls per appearance fall between "
    "the starter-like and reliever-like descriptions, and the two boards are separate because "
    "their opportunity is genuinely different \u2014 so there is no board this season belongs "
    "to. That is a limit of a two-way usage split, not a requirement this season failed."
)

#: Short band tags for the per-row workload marker. Display organization on a
#: continuous axis -- never a tier, never a qualification.
_BAND_SHORT_LABELS: dict[str, str] = {
    "high": "150+",
    "moderate": "60\u2013149",
    "below_board_minimum": "<60",
}


def _pitcher_row_view(row: ppc.PitcherRow, scale: v.ZeroScale, root_prefix: str) -> dict[str, Any]:
    """One pitcher, projected onto the cumulative-runs scale.

    The scale field is built from the CUMULATIVE interval, never the per-100
    interval, because the cumulative total is what this surface ranks. Mixing
    the two would draw a mark whose position and whose error bar answer
    different questions.
    """
    point = row.cumulative_contact_luck_runs
    lower = row.cumulative_ci_low
    upper = row.cumulative_ci_high
    return {
        "pitcher_id": row.pitcher_id,
        "name": row.name,
        "url": f"{root_prefix}pitchers/{row.pitcher_id}/",
        "total": point,
        "bbe": row.eligible_batted_balls,
        "appearances": row.appearances,
        "bbe_per_appearance": row.bbe_per_appearance,
        "per_100": row.contact_luck_per_100,
        "per_100_low": row.per_100_ci_low,
        "per_100_high": row.per_100_ci_high,
        "per_100_width": f"{row.per_100_ci_high - row.per_100_ci_low:.1f}",
        "band_short": _BAND_SHORT_LABELS[row.workload_band],
        "role_label": ppc.ROLE_LABELS[row.role_bucket],
        "on_board": row.on_board,
        # Sign follows the POINT ESTIMATE only, identically whether or not
        # the interval crosses zero (preserved product invariant).
        "favorable": point >= 0,
        "pt_pct": _pct(scale.clamped_fraction_of(point)),
        "lo_pct": _pct(scale.clamped_fraction_of(lower)),
        "hi_pct": _pct(scale.clamped_fraction_of(upper)),
        "clip_state": scale.clip_state(point=point, lower=lower, upper=upper),
        "point_state": scale.point_state(point),
    }


def _pitcher_band_summary(rows: list[ppc.PitcherRow]) -> str:
    high = sum(1 for r in rows if r.workload_band == "high")
    moderate = sum(1 for r in rows if r.workload_band == "moderate")
    return (
        f"{len(rows)} ranked here: {high} at 150 or more resolved batted balls, "
        f"{moderate} between 60 and 149."
    )


def _pitcher_board_view(
    data: ppc.PitcherPrototypeData,
    role_bucket: str,
    scale: v.ZeroScale,
    root_prefix: str,
) -> dict[str, Any]:
    """One ranked board. Built from ONE usage population and never merged
    with another: two populations whose opportunity differs by roughly a
    factor of four have no shared ranking to be first of.
    """
    rows = data.board(role_bucket)
    views = [_pitcher_row_view(r, scale, root_prefix) for r in rows]
    for position, view in enumerate(views, start=1):
        view["board_rank"] = position
    marks = sorted(
        (
            {
                "left": _pct(scale.clamped_fraction_of(r.cumulative_contact_luck_runs)),
                "favorable": r.cumulative_contact_luck_runs >= 0,
                "score": r.cumulative_contact_luck_runs,
            }
            for r in rows
        ),
        key=lambda m: m["score"],
    )
    label = ppc.ROLE_LABELS[role_bucket]
    return {
        "slug": role_bucket.replace("_", "-"),
        "label": label,
        "short_label": label,
        "definition": _PITCHER_BOARD_DEFINITIONS[role_bucket],
        "band_summary": _pitcher_band_summary(rows),
        "season": data.season,
        "rows": views,
        "marks": marks,
        # Per-row band tags on the RELIEVER-LIKE board only. On the
        # starter-like board 176 of 231 rows carry the same tag, so the
        # column reads as decoration applied uniformly rather than as
        # information (guardrails anti-pattern 1's argument, applied to a
        # table cell). Reliever-like workload is the axis that actually
        # varies fast down its board, and it is the one a reader most needs
        # marked -- so that is where the marker is spent. The band counts
        # for BOTH boards are stated above every board regardless.
        "show_bands": role_bucket == "reliever_like",
    }


def _pitcher_off_board_groups(
    data: ppc.PitcherPrototypeData, root_prefix: str
) -> list[dict[str, Any]]:
    """Name-and-link index of every pitcher-season the boards withhold.

    Two disjoint groups, for the two different reasons a season is on no
    board: below the display minimum, or usage that falls between the two
    descriptions. A season below the minimum is listed under the minimum
    even when its usage is also ambiguous -- one season, one reason, the
    one that actually withheld it.

    This exists because "player-page-only" has to mean the page is
    reachable. The site's player search covers hitters, and no board links
    a withheld season, so these pages have no other route in.

    Alphabetical by name, never by score: this is an index, and ordering it
    by the quantity the boards rank would make it a third ranked board of
    exactly the seasons the display rules withheld from ranking.
    """

    def entries(rows: list[ppc.PitcherRow]) -> list[dict[str, Any]]:
        return [
            {
                "name": r.name,
                "url": f"{root_prefix}pitchers/{r.pitcher_id}/",
                "bbe": r.eligible_batted_balls,
            }
            for r in sorted(rows, key=lambda r: r.name)
        ]

    below = [r for r in data.rows if not r.on_board]
    mixed = [r for r in data.rows if r.on_board and r.role_bucket == "ambiguous"]
    groups = []
    if below:
        groups.append(
            {
                "summary": (
                    f"All {len(below)} seasons under "
                    f"{data.board_display_minimum_bbe} resolved batted balls"
                ),
                "entries": entries(below),
            }
        )
    if mixed:
        groups.append(
            {
                "summary": f"All {len(mixed)} mixed-usage seasons",
                "entries": entries(mixed),
            }
        )
    return groups


def _pitcher_play_view(play: ppc.PlayHighlight, label: str) -> dict[str, Any]:
    bb_type = _BB_TYPE_LABELS.get(play.bb_type or "", play.bb_type or "batted ball")
    outcome = _OUTCOME_LABELS.get(play.outcome_class or "", play.outcome_class or "\u2014")
    return {
        "label": label,
        "value": play.pitching_contact_luck_runs,
        "favorable": play.pitching_contact_luck_runs >= 0,
        "launch_speed": (
            f"{play.launch_speed_mph:.1f}" if play.launch_speed_mph is not None else None
        ),
        "launch_angle": (
            format_signed(play.launch_angle_deg, 0) if play.launch_angle_deg is not None else None
        ),
        "bb_type_label": bb_type,
        "outcome_label": outcome,
        "game_date_display": _display_date(play.game_date),
    }


def _plural(count: float, singular: str, plural: str | None = None) -> str:
    """`1 batted ball` / `2 batted balls`. A stat line that reads "1 resolved
    batted balls" is a small thing that tells a reader nobody looked at the
    smallest cases, which on this surface are exactly the cases the display
    rules are about.
    """
    return singular if count == 1 else (plural or f"{singular}s")


def _largest_play_sentence(row: ppc.PitcherRow) -> str:
    """States how much of the season total the single biggest batted ball
    accounts for, in the reader's own terms.

    The share is `largest |play| / |net total|`, and it is only a meaningful
    percentage while the denominator is a real quantity. Josh Winckowski's
    2024 net is -0.10 runs over 241 batted balls, which makes that ratio
    1365% -- a true number that tells a reader nothing except that something
    small is in a denominator. Above 2x the sentence stops quoting a
    percentage and says the thing the percentage was standing in for: the
    season's batted balls very nearly cancelled, and the net is smaller than
    a single play on this page.

    Never a warning and never a de-emphasis: a season carried by one play is
    a true description of that season, and the product's standing rule is
    that an uncertain number is rendered exactly like a certain one.
    """
    share = row.largest_play_share_of_net
    if share is None:
        return (
            "These are the two batted balls whose outcomes differed most from what the "
            "contact predicted."
        )
    if share >= 2.0:
        return (
            "This season's favorable and unfavorable batted balls very nearly cancelled: the "
            "net total above is smaller than either of the two single batted balls below. "
            "Read it as roughly zero across the whole sample, not as a small finding in "
            "either direction."
        )
    pct = f"{share * 100:.0f}%"
    if share >= 1.0:
        return (
            f"The larger of these two batted balls is worth {pct} of this season's net total "
            "on its own \u2014 more than the whole of it, with the rest of the season pulling "
            "the other way. A total this size is one or two batted balls, not a season-long "
            "pattern."
        )
    if share >= 0.5:
        return (
            f"The larger of these two batted balls accounts for {pct} of this season's net "
            "total on its own. Most of what the number above says comes from very few plays."
        )
    return (
        f"The larger of these two batted balls accounts for {pct} of this season's net total. "
        "The rest is spread across the other batted balls."
    )


def _pitcher_card_view(
    row: ppc.PitcherRow,
    data: ppc.PitcherPrototypeData,
    scale: v.ZeroScale,
    root_prefix: str,
) -> dict[str, Any]:
    view = _pitcher_row_view(row, scale, root_prefix)

    # A season with ONE resolved batted ball has the same play at both ends
    # of its own distribution. Printing it twice, once labelled "largest
    # favorable" and once "largest unfavorable", states two findings where
    # there is one -- and gives a positive value a label reading
    # "unfavorable", which is simply false. Seven 2024 pitcher-seasons are in
    # this position; each gets one entry, named for what it is.
    favorable_play = row.largest_favorable_play
    unfavorable_play = row.largest_unfavorable_play
    single_play = (
        favorable_play is not None
        and unfavorable_play is not None
        and favorable_play.play_id == unfavorable_play.play_id
    )
    plays = []
    if single_play and favorable_play is not None:
        plays.append(_pitcher_play_view(favorable_play, "The season's only batted ball"))
    else:
        if favorable_play is not None:
            plays.append(_pitcher_play_view(favorable_play, "Largest favorable batted ball"))
        if unfavorable_play is not None:
            plays.append(_pitcher_play_view(unfavorable_play, "Largest unfavorable batted ball"))
    if not row.on_board:
        off_board_reason: str | None = ppc.REASON_BELOW_BOARD_MINIMUM
    elif row.role_bucket == "ambiguous":
        off_board_reason = _REASON_MIXED_USAGE
    else:
        off_board_reason = None
    # A 95% interval of width zero is not precision. The bootstrap resamples
    # GAMES within a pitcher-season, so a pitcher who appeared in exactly one
    # game has nothing to resample and every replicate returns the same
    # number. Seventy-three 2024 pitcher-seasons are in this position and all
    # 73 appeared once. Printing "+16.30 to +16.30" beside the standard
    # sentence about interval width would present the total ABSENCE of an
    # uncertainty estimate as the tightest one on the site, which inverts the
    # thing this whole surface is careful about. None of the 73 is on a
    # ranked board, but every one of them has a page.
    interval_width = row.per_100_ci_high - row.per_100_ci_low
    degenerate_interval = interval_width < 0.005 and row.appearances <= 1
    view.update(
        {
            "season": data.season,
            "role_usage_phrase": ppc.ROLE_USAGE_PHRASES[row.role_bucket],
            "off_board_reason": off_board_reason,
            "single_play_season": single_play,
            "degenerate_interval": degenerate_interval,
            "bbe_noun": _plural(row.eligible_batted_balls, "resolved batted ball"),
            "appearances_noun": _plural(row.appearances, "appearance"),
            "largest_favorable": row.largest_favorable_play is not None,
            "largest_unfavorable": row.largest_unfavorable_play is not None,
            "plays": plays,
            "largest_play_sentence": (
                "This season has one resolved batted ball, so it is both the most and the "
                "least favorable one, and the whole of the total above."
                if single_play
                else _largest_play_sentence(row)
            ),
        }
    )
    return view


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
    pitcher_prototype_fixture_path: Path | None = None,
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
    # Redesign Phase 1: the ONE canonical scale for Contact Luck Runs per
    # 100, built once here and threaded to every renderer (Invariant D).
    # `qualified_domain` is kept as its raw tuple for the call sites that
    # still take one.
    league_scale = v.ZeroScale.from_intervals(
        "league_per_100",
        "Runs / 100 eligible BBE",
        qualified_intervals,
    )
    qualified_domain = league_scale.domain

    player_index = c.build_player_index(payloads)
    # Redesign Phase 2: the global search marks a player it finds who holds no
    # official rank, rather than suppressing them -- unqualified players keep a
    # page, a score and an interval and are never de-emphasised
    # (`docs/design/guardrails.md`, preserved product invariants). `ranked` is a
    # verbatim read of the snapshot's own `qualification_status`, resolved here
    # at build time so no rank semantics are ever decided in JavaScript
    # (`CLAUDE.md` rule 6).
    _qualification_by_id = {
        record["batter_id"]: record.get("qualification_status") for record in payloads.public_score
    }
    player_index_json = json.dumps(
        [
            {
                "batter_id": e.batter_id,
                "batter_name": e.batter_name,
                "url": f"{root_prefix}players/{e.batter_id}/",
                "ranked": _qualification_by_id.get(e.batter_id) == "qualified",
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

    # Version 0.13.1 LOCAL PROTOTYPE. Fail-closed exactly like the Play
    # Explorer above: this parameter has NO default of its own, so a bare
    # `build.py` -- and therefore every production invocation, including
    # `scripts/publish_snapshot.sh`, which does not pass it -- builds the
    # site with no pitcher route at all and no nav entry pointing at one.
    # A development-season surface can never reach the published site by
    # omission; it takes an explicit flag naming the fixture.
    pitcher_data: ppc.PitcherPrototypeData | None = None
    if pitcher_prototype_fixture_path is not None:
        pitcher_data = ppc.load_pitcher_prototype_data(pitcher_prototype_fixture_path)
    pitcher_prototype_available = pitcher_data is not None

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
        # Redesign Phase 1, Invariant Z: the single zero locus, written onto
        # <html> so CSS and Python can never disagree about where zero is.
        # This is a LAYOUT percentage derived from already-computed snapshot
        # values -- not a score, rank, interval or probability -- so it does
        # not cross the "the dashboard displays, it never computes" line in
        # PRODUCT.md #5; no scoring code is imported to produce it.
        "cl_zero_fraction_css": f"{league_scale.zero_fraction * 100:.4f}%",
        "dashboard_version": DASHBOARD_VERSION,
        "snapshot_directory_name": latest.directory_name,
        "build_timestamp": build_timestamp,
        # Phase 8: the footer printed this raw ISO string on every route
        # while /status/ formatted its own timestamps for a reader -- one
        # product, two treatments of the same kind of value. Both now go
        # through `_display_timestamp`, and the machine value stays in
        # `<time datetime>` exactly as it does on /status/.
        "build_timestamp_display": _display_timestamp(build_timestamp),
        "player_index_json": player_index_json,
        "explore_available": explore_available,
        "pitcher_prototype_available": pitcher_prototype_available,
    }

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_template = env.get_template("index.html")
    # Built ONCE and reused verbatim by the homepage and by every player
    # page. The player hero draws the whole league as its ground (Phase 4),
    # so this geometry would otherwise be recomputed ~600 times, and -- worse
    # -- two independently computed copies of "the league" could drift.
    league_axis_ticks = _axis_ticks(league_scale)
    league_distribution_marks = _distribution_marks(favorable_rows, league_scale)

    (out_dir / "index.html").write_text(
        index_template.render(
            **base_context,
            active_page="leaderboard",
            headshot_origin=HEADSHOT_ORIGIN,
            axis_ticks=league_axis_ticks,
            axis_unit_label=league_scale.unit_label,
            distribution_marks=league_distribution_marks,
            favorable_rows=_leaderboard_view_rows(favorable_rows, league_scale, root_prefix),
            unfavorable_rows=_leaderboard_view_rows(unfavorable_rows, league_scale, root_prefix),
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
            **base_context,
            active_page="status",
            status=status_data,
            **_status_view(status_data),
        )
    )

    payload_cache: dict[str, c.SnapshotPayloads] = {latest.directory_name: payloads}
    player_template = env.get_template("player.html")
    league_context = {
        "axis_ticks": league_axis_ticks,
        "distribution_marks": league_distribution_marks,
        "unit_label": league_scale.unit_label,
        "domain_min": league_scale.domain_min,
        "domain_max": league_scale.domain_max,
        "qualified_count": status_data.qualified_count,
    }
    # `play_count` per hitter, from the already-validated Explore catalog.
    # Empty when the Play Explorer is disabled for this build, which is the
    # fail-closed default -- the player page then emits no plays link at all
    # rather than one that 404s.
    play_counts: dict[int, int] = (
        {p.batter_id: p.play_count for p in explore_data.players}
        if explore_available and explore_data is not None
        else {}
    )
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
            league_scale,
            league=league_context,
            play_count=play_counts.get(entry.batter_id),
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

        # Redesign Phase 5, Invariant D applied to Explore: ONE snapshot-level
        # `run_value` domain over every published play, padded by
        # `ZeroScale.from_values` until its zero lands on the same
        # `--cl-zero` the leaderboard and the player hero use. Explore draws
        # every hitter's plays on it, so two hitters' plays are comparable and
        # a one-play hitter is not stretched across the whole field.
        #
        # It is NOT `league_per_100` and must never look like it: the figure
        # prints its own ticks with its own unit (the anti-fake-alignment
        # guard, docs/design/dataviz.md). The domain is computed here, once,
        # and handed to the browser as data -- `explore.js` interpolates a
        # LAYOUT PERCENTAGE against it and never derives a domain of its own.
        run_value_scale = v.ZeroScale.from_values(
            "run_value",
            "Contact Luck on the play, runs",
            [explore_data.contact_luck_min, explore_data.contact_luck_max],
            zero_fraction=league_scale.zero_fraction,
        )
        # Built ONCE and rendered into BOTH routes. Redesign Phase 6 draws
        # the play page's expected-vs-observed figure on this scale, and
        # "the same domain" has to be true by construction rather than by
        # two call sites agreeing: one object, one dict, two templates.
        #
        # `unit_label` names what EXPLORE's marks are (one Contact Luck value
        # per play). The play page's figure carries expected and observed run
        # value as well, so it prints its own unit on the figure. The shared
        # thing is the DOMAIN and the zero fraction; each figure still has to
        # say honestly what its own marks measure.
        run_value_context = {
            "ticks": _axis_ticks(run_value_scale),
            "unit_label": run_value_scale.unit_label,
            "domain_min": run_value_scale.domain_min,
            "domain_max": run_value_scale.domain_max,
            "play_count": explore_data.total_play_count,
        }
        run_value_scale_json = json.dumps(
            {
                "domain_min": run_value_scale.domain_min,
                "domain_max": run_value_scale.domain_max,
                "unit_label": run_value_scale.unit_label,
            },
            sort_keys=True,
        )
        (explore_dir / "index.html").write_text(
            env.get_template("explore.html").render(
                **base_context,
                active_page="explore",
                run_value=run_value_context,
                run_value_scale_json=run_value_scale_json,
            )
        )

        plays_dir = out_dir / "plays"
        plays_dir.mkdir(parents=True, exist_ok=True)
        (plays_dir / "index.html").write_text(
            env.get_template("play.html").render(
                **base_context,
                active_page=None,
                run_value=run_value_context,
                run_value_scale_json=run_value_scale_json,
            )
        )

    # ── "/pitchers/" + "/pitchers/<pitcher_id>/" · Version 0.13.1 LOCAL
    #    PROTOTYPE ────────────────────────────────────────────────────────
    #
    # A FOURTH `ZeroScale`. `visuals.ZeroScale`'s docstring says the product
    # has exactly three and that adding one is a design review rather than a
    # code change -- so this is that decision made deliberately and written
    # down, not slipped in. The reason a fourth is needed: this surface's
    # ranked quantity is a CUMULATIVE RUN TOTAL, which no existing scale
    # measures. Drawing it on `league_per_100` would place a +20-run season
    # off the end of a rate axis; drawing it on `run_value` would put a
    # season total on a single-play axis. Both would be a false alignment.
    #
    # Invariant Z still holds and is what makes the fourth scale safe: it is
    # built with `from_values(..., zero_fraction=league_scale.zero_fraction)`,
    # so zero sits at the SAME `--cl-zero` every other figure on the site
    # registers against. Invariant D deliberately does NOT apply (same
    # exception the `run_value` scale takes): this is a development-season
    # population, not the snapshot's own comparison population. The price of
    # the exception, per that docstring, is paid in the template -- both
    # boards and every card print this scale's own ticks and its own unit,
    # so no figure borrows the shared zero while hiding its own domain.
    #
    # ONE scale across BOTH boards. Starter-like and reliever-like are never
    # ranked together, but they are drawn together, which is what makes the
    # opportunity difference between them legible instead of hidden.
    if pitcher_data is not None:
        board_rows = [r for r in pitcher_data.rows if r.on_board]
        pitcher_scale = v.ZeroScale.from_values(
            "pitcher_cumulative_runs",
            f"Contact Luck allowed, cumulative runs, {pitcher_data.season}",
            [r.cumulative_ci_low for r in board_rows] + [r.cumulative_ci_high for r in board_rows],
            zero_fraction=league_scale.zero_fraction,
        )
        pitcher_scale_ticks = _axis_ticks(pitcher_scale)
        boards = [
            _pitcher_board_view(pitcher_data, "starter_like", pitcher_scale, root_prefix),
            _pitcher_board_view(pitcher_data, "reliever_like", pitcher_scale, root_prefix),
        ]
        pitchers_dir = out_dir / "pitchers"
        pitchers_dir.mkdir(parents=True, exist_ok=True)
        (pitchers_dir / "index.html").write_text(
            env.get_template("pitchers.html").render(
                **base_context,
                active_page="pitchers",
                pitchers=pitcher_data,
                boards=boards,
                pitcher_scale=pitcher_scale,
                pitcher_scale_ticks=pitcher_scale_ticks,
                below_minimum_count=sum(1 for r in pitcher_data.rows if not r.on_board),
                # Seasons that CLEAR the display minimum and are still on no
                # board, because their usage falls between the two
                # descriptions. Counted separately and stated separately:
                # folding them into the below-minimum sentence would give
                # them a reason that is not theirs, and leaving them out
                # entirely -- which the first version of this page did --
                # meant 16 seasons with real workload were absent from the
                # surface without explanation.
                mixed_usage_count=sum(
                    1 for r in pitcher_data.rows if r.role_bucket == "ambiguous" and r.on_board
                ),
                off_board_groups=_pitcher_off_board_groups(pitcher_data, root_prefix),
                off_board_total=sum(
                    1 for r in pitcher_data.rows if not r.on_board or r.role_bucket == "ambiguous"
                ),
            )
        )

        # Every pitcher-season gets a page, INCLUDING the ones the board
        # withholds -- that withholding is the whole reason those seasons
        # need somewhere to be shown in full, and a display rule that
        # removed them from the site entirely would be the qualification
        # rule this surface is careful not to have.
        marks_by_role = {b["label"]: b["marks"] for b in boards}
        count_by_role = {b["label"]: len(b["rows"]) for b in boards}
        pitcher_card_template = env.get_template("pitcher.html")
        for row in pitcher_data.rows:
            card = _pitcher_card_view(row, pitcher_data, pitcher_scale, root_prefix)
            role_label = card["role_label"]
            card_dir = pitchers_dir / str(row.pitcher_id)
            card_dir.mkdir(parents=True, exist_ok=True)
            (card_dir / "index.html").write_text(
                pitcher_card_template.render(
                    **base_context,
                    active_page=None,
                    p=card,
                    pitcher_scale=pitcher_scale,
                    pitcher_scale_ticks=pitcher_scale_ticks,
                    board_marks=marks_by_role.get(role_label, []),
                    board_count=count_by_role.get(role_label, 0),
                )
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
    parser.add_argument(
        "--pitcher-prototype-fixture",
        type=Path,
        default=None,
        help=(
            "Path to the committed Pitcher Contact Luck development fixture "
            "(dashboard/pitcher_prototype_fixture.json), enabling the LOCAL 2024 pitcher "
            "prototype at /pitchers/. NOT defaulted: omitting this flag builds the site "
            "with no pitcher route and no nav entry, which is what every production "
            "invocation does -- scripts/publish_snapshot.sh never passes it. The fixture "
            "is 2024 DEVELOPMENT-season data and must not be published."
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
        "pitcher_prototype_fixture_path": args.pitcher_prototype_fixture,
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
    print(
        "[dashboard build] pitcher prototype (2024 development data): "
        f"{args.pitcher_prototype_fixture is not None}"
    )
    if result.invalid_snapshot_count:
        print(
            f"[dashboard build] WARNING: {result.invalid_snapshot_count} invalid snapshot(s) excluded -- see warnings above"
        )
    return result


if __name__ == "__main__":
    main()
