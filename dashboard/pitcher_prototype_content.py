"""Version 0.13.1 LOCAL PROTOTYPE: turns the committed Pitcher Contact Luck
development fixture into page-ready view-models.

Same contract as every other module under `dashboard/`: it recomputes
nothing. Every run total, rate, interval, batted-ball count and per-play
value below is read verbatim from `dashboard/pitcher_prototype_fixture.json`,
which `demo/build_pitcher_prototype_fixture.py` produced offline from real
2024 development data. This module derives no score, rank, interval or
probability, and imports no scoring code (`CLAUDE.md` rule 6, enforced by
`tests/test_dashboard_isolation.py`).

## What this surface is, and is not

It is a LOCAL prototype of a presentation, gated behind an explicit
`dashboard/build.py --pitcher-prototype-fixture PATH` flag exactly like the
Play Explorer's `--explore-artifacts-dir`: omit the flag and the route is
not built and does not appear in navigation. It is 2024 development data
only -- never a 2026 prospective snapshot, never part of the published
site's snapshot contract, and never a claim about a future season.

## Two vocabulary rules this module enforces rather than assumes

1. **`starter_like` / `reliever_like` are DESCRIPTIVE usage buckets** read
   off batted balls per appearance. This repository holds no authoritative
   role metadata, so no value here may be rendered as "starter",
   "reliever", or "closer". `ROLE_LABELS` is the only permitted vocabulary
   and the templates take their strings from it.
2. **`board_display_minimum_bbe` is a PRESENTATION rule, not qualification.**
   A pitcher-season below it keeps its full page, its total, its rate and
   its interval; it is only left off the ranked board, because a normalized
   rate over a handful of batted balls carries an interval wide enough to
   place a one-batted-ball season at either end of the population. It is
   never called qualification, eligibility, a minimum, or an MLB rule --
   see `REASON_BELOW_BOARD_MINIMUM`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PitcherPrototypeData",
    "PitcherRow",
    "PitcherPrototypeError",
    "PlayHighlight",
    "PITCHER_RETROSPECTIVE_LIMITATION",
    "ROLE_LABELS",
    "ROLE_USAGE_PHRASES",
    "WORKLOAD_BAND_LABELS",
    "load_pitcher_prototype_data",
]

#: The ONLY role vocabulary this surface may use. Deliberately hyphenated
#: adjectives, never role nouns: the repository has no role metadata, and
#: "closer" in particular is nowhere in this data and must never be inferred
#: from a low batted-balls-per-appearance figure.
ROLE_LABELS: dict[str, str] = {
    "starter_like": "Starter-like",
    "reliever_like": "Reliever-like",
    "ambiguous": "Mixed usage",
}

#: The same three buckets, phrased as a standalone usage description.
#: `ROLE_LABELS["ambiguous"]` already carries the noun, so a template that
#: appended " usage" to it rendered "Mixed usage usage" on every ambiguous
#: pitcher's page. The phrase is written out once here rather than assembled
#: in a template, for the same reason `ROLE_LABELS` is: this surface's role
#: vocabulary is fixed, and a template must not be able to compose a new
#: variant of it.
ROLE_USAGE_PHRASES: dict[str, str] = {
    "starter_like": "Starter-like usage",
    "reliever_like": "Reliever-like usage",
    "ambiguous": "Mixed usage",
}

#: The retrospective-limitation sentence for the PITCHER surface, duplicated
#: verbatim from `mlb_luck_score.scoring.public_labels.
#: PITCHER_RETROSPECTIVE_LIMITATION`.
#:
#: Duplicated rather than imported for the same reason every other public
#: string on this surface is: no module under `dashboard/` may import
#: scoring code (`CLAUDE.md` rule 6, enforced by
#: `tests/test_dashboard_isolation.py`). The copy is bound to its source by
#: an equality test, which is what stops the two drifting apart.
#:
#: The site footer's shared sentence names *batting* talent, which is right
#: on a hitter page and wrong on this one. Pitcher routes render this
#: instead; hitter routes are untouched.
PITCHER_RETROSPECTIVE_LIMITATION = (
    "Pitcher Contact Luck is retrospective. It describes how favorable or unfavorable the "
    "outcomes on a pitcher's contact were relative to what that contact predicted. It is "
    "not a measure of stable pitching talent or of future performance."
)

#: Display organization for the reliever-like board. These band a continuous
#: workload axis for legibility; none of them is a qualification tier.
WORKLOAD_BAND_LABELS: dict[str, str] = {
    "high": "150+ resolved BBE",
    "moderate": "60-149 resolved BBE",
    "below_board_minimum": "Under 60 resolved BBE",
}

#: Why a pitcher-season is off the ranked board. Says what the display rule
#: is and why, and never implies the season failed a requirement.
REASON_BELOW_BOARD_MINIMUM = (
    "Shown here but not on the ranked board. Below 60 resolved batted balls the "
    "per-100 interval is wide enough to reach both ends of the pitcher population, "
    "so a ranked position would read as a finding it cannot support. This is a "
    "display choice about the board, not a requirement this season failed."
)

#: The fixture schema this module accepts. Fail-closed, exactly like
#: `explore_content`'s `play_ledger_version`/`explorer_artifact_version`
#: checks: a fixture written against a different schema is refused rather
#: than partially read.
SUPPORTED_FIXTURE_VERSIONS: frozenset[str] = frozenset({"0.1"})


class PitcherPrototypeError(Exception):
    """The prototype fixture is missing, unparseable, or written against a
    schema version this module does not accept. Never a reason to render the
    route with partial data.
    """


@dataclass(frozen=True)
class PlayHighlight:
    """One batted ball, read verbatim off the fixture."""

    play_id: str
    game_date: str
    pitching_contact_luck_runs: float
    launch_speed_mph: float | None
    launch_angle_deg: float | None
    bb_type: str | None
    outcome_class: str | None


@dataclass(frozen=True)
class PitcherRow:
    pitcher_id: int
    name: str
    role_bucket: str
    workload_band: str
    cumulative_contact_luck_runs: float
    eligible_batted_balls: int
    appearances: int
    bbe_per_appearance: float | None
    contact_luck_per_100: float
    per_100_ci_low: float
    per_100_ci_high: float
    cumulative_ci_low: float
    cumulative_ci_high: float
    largest_play_share_of_net: float | None
    largest_favorable_play: PlayHighlight | None
    largest_unfavorable_play: PlayHighlight | None

    @property
    def on_board(self) -> bool:
        return self.workload_band != "below_board_minimum"


@dataclass(frozen=True)
class PitcherPrototypeData:
    season: int
    sign_convention: str
    board_display_minimum_bbe: int
    workload_band_high_min_bbe: int
    starter_like_min_bbe_per_appearance: float
    reliever_like_max_bbe_per_appearance: float
    batter_side_reproduction: dict[str, Any]
    rows: list[PitcherRow]

    def board(self, role_bucket: str) -> list[PitcherRow]:
        """One ranked board: a single usage population, ordered by the
        cumulative total, with the below-minimum rows withheld.

        Each board is built independently and never concatenated with
        another -- two populations whose opportunity differs by a factor of
        four have no shared ranking to be #1 of.
        """
        return [r for r in self.rows if r.role_bucket == role_bucket and r.on_board]

    def find(self, pitcher_id: int) -> PitcherRow | None:
        for row in self.rows:
            if row.pitcher_id == pitcher_id:
                return row
        return None


def _play(record: dict[str, Any] | None) -> PlayHighlight | None:
    if not record:
        return None
    return PlayHighlight(
        play_id=str(record["play_id"]),
        game_date=str(record["game_date"]),
        pitching_contact_luck_runs=float(record["pitching_contact_luck_runs"]),
        launch_speed_mph=record.get("launch_speed_mph"),
        launch_angle_deg=record.get("launch_angle_deg"),
        bb_type=record.get("bb_type"),
        outcome_class=record.get("outcome_class"),
    )


def load_pitcher_prototype_data(path: Path) -> PitcherPrototypeData:
    """Read and validate the committed fixture. Fails closed."""
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PitcherPrototypeError(f"failed to load {path}: {exc}") from exc

    version = raw.get("pitcher_prototype_fixture_version")
    if version not in SUPPORTED_FIXTURE_VERSIONS:
        raise PitcherPrototypeError(
            f"{path}: pitcher_prototype_fixture_version {version!r} is not one of "
            f"{sorted(SUPPORTED_FIXTURE_VERSIONS)} -- refusing to render it."
        )

    reproduction = raw.get("batter_side_reproduction") or {}
    if not reproduction.get("reproduces"):
        # The generator already refuses to write a fixture whose pitcher
        # numbers failed the batter-side self-check. This is the second,
        # independent refusal at the point of DISPLAY, on the same
        # two-layer principle the prospective cache-coverage guard follows:
        # never let a single upstream check be the only thing standing
        # between untrustworthy values and a rendered page.
        raise PitcherPrototypeError(
            f"{path}: batter_side_reproduction did not pass ({reproduction}) -- the pitcher "
            "values in this fixture are not trustworthy and must not be displayed."
        )

    rules = raw.get("presentation_rules") or {}
    rows = [
        PitcherRow(
            pitcher_id=int(r["pitcher_id"]),
            name=r["name"] or f"Pitcher {r['pitcher_id']}",
            role_bucket=r["role_bucket"],
            workload_band=r["workload_band"],
            cumulative_contact_luck_runs=float(r["cumulative_contact_luck_runs"]),
            eligible_batted_balls=int(r["eligible_batted_balls"]),
            appearances=int(r["appearances"]),
            bbe_per_appearance=r.get("bbe_per_appearance"),
            contact_luck_per_100=float(r["contact_luck_per_100"]),
            per_100_ci_low=float(r["per_100_ci_low"]),
            per_100_ci_high=float(r["per_100_ci_high"]),
            cumulative_ci_low=float(r["cumulative_ci_low"]),
            cumulative_ci_high=float(r["cumulative_ci_high"]),
            largest_play_share_of_net=r.get("largest_play_share_of_net"),
            largest_favorable_play=_play(r.get("largest_favorable_play")),
            largest_unfavorable_play=_play(r.get("largest_unfavorable_play")),
        )
        for r in raw["pitchers"]
    ]
    unknown_roles = {r.role_bucket for r in rows} - set(ROLE_LABELS)
    if unknown_roles:
        raise PitcherPrototypeError(
            f"{path}: unknown role bucket(s) {sorted(unknown_roles)} -- this surface has a "
            "fixed, descriptive role vocabulary and never invents a label for a new one."
        )

    # Cumulative runs, descending. THE ordering of this surface: the board's
    # rank is a position in this list and nothing else, so no template can
    # present a different quantity as the ranking key.
    rows.sort(key=lambda r: -r.cumulative_contact_luck_runs)

    return PitcherPrototypeData(
        season=int(raw["season"]),
        sign_convention=str(raw["sign_convention"]),
        board_display_minimum_bbe=int(rules["board_display_minimum_bbe"]),
        workload_band_high_min_bbe=int(rules["workload_band_high_min_bbe"]),
        starter_like_min_bbe_per_appearance=float(rules["starter_like_min_bbe_per_appearance"]),
        reliever_like_max_bbe_per_appearance=float(rules["reliever_like_max_bbe_per_appearance"]),
        batter_side_reproduction=reproduction,
        rows=rows,
    )
