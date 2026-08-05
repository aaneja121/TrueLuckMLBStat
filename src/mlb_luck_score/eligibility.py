"""Centralized fair-batted-ball eligibility rules for Version 0.1.

This is the ONE place that decides which Statcast events count as an
eligible "fair batted ball" event for the Contact Luck Prototype, and how a
raw `events` value maps onto the five-class outcome used for training. Do
not duplicate or reimplement this logic elsewhere -- import from here.

The event list below is a documented Version 0.1 research choice, not a
permanent or authoritative taxonomy of baseball events. It is intentionally
conservative: ambiguous or rare situations are flagged rather than guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from mlb_luck_score.config import CLASS_ORDER

#: Statcast `events` values treated as an eligible fair-batted-ball event in
#: Version 0.1. Everything else (strikeouts, walks, HBP, catcher
#: interference, foul-related non-events, stolen bases, pickoffs, balks,
#: etc.) is excluded by construction because it simply is not in this set.
ELIGIBLE_EVENTS_V0_1: frozenset[str] = frozenset(
    {
        "field_out",
        "force_out",
        "grounded_into_double_play",
        "double_play",
        "fielders_choice",
        "fielders_choice_out",
        "single",
        "double",
        "triple",
        "home_run",
        "field_error",
        "sac_bunt",
        "sac_fly",
        "sac_fly_double_play",
    }
)

#: `events` values that map unambiguously to the batter-runner being put out,
#: with no ambiguity about the resulting base (there isn't one -- the batter
#: didn't reach safely on this play type).
_UNAMBIGUOUS_OUT_EVENTS: frozenset[str] = frozenset(
    {
        "field_out",
        "force_out",
        "grounded_into_double_play",
        "double_play",
        "fielders_choice_out",
        "sac_bunt",
        "sac_fly",
        "sac_fly_double_play",
    }
)

#: `events` values that map unambiguously to a hit outcome class.
_DIRECT_HIT_EVENTS: dict[str, str] = {
    "single": "single",
    "double": "double",
    "triple": "triple",
    "home_run": "home_run",
}

#: `events` values where the batter-runner's resulting base cannot be
#: reliably determined from public Statcast fields alone. Version 0.1 keeps
#: these rows in the cleaned dataset but excludes them from model training
#: rather than inventing a label.
_AMBIGUOUS_OUTCOME_EVENTS: frozenset[str] = frozenset({"fielders_choice", "field_error"})

assert _AMBIGUOUS_OUTCOME_EVENTS | _UNAMBIGUOUS_OUT_EVENTS | set(_DIRECT_HIT_EVENTS) == set(
    ELIGIBLE_EVENTS_V0_1
), "Every eligible event must be classified as out / direct-hit / ambiguous"

assert set(CLASS_ORDER) == {"out"} | set(_DIRECT_HIT_EVENTS.values())

#: Required contact variables. A row missing these cannot be used for
#: contact-feature modeling even if the event type is otherwise eligible.
REQUIRED_CONTACT_FIELDS: tuple[str, ...] = ("launch_speed", "launch_angle")

#: Keyword heuristics used for the best-effort, conservative text flags
#: below. These are intentionally narrow: Version 0.1 does not attempt to
#: aggressively infer rare-event flags from ambiguous free-text descriptions.
_INTERFERENCE_KEYWORDS = ("fan interference", "spectator interference")
_PARK_RULE_KEYWORDS = ("ground rule", "ground-rule")
_RULE_VIOLATION_KEYWORDS = ("batting out of turn", "rule violation")
_WALKOFF_KEYWORDS = ("walk-off", "walkoff", "walk off")

#: Exclusion reasons recorded in the cleaned event table's reason column.
REASON_NOT_ELIGIBLE_EVENT = "not_fair_batted_ball_event_v0_1"
REASON_MISSING_CONTACT_DATA = "missing_required_contact_data"
REASON_AMBIGUOUS_FIELDERS_CHOICE = "ambiguous_outcome_fielders_choice"
REASON_AMBIGUOUS_FIELD_ERROR = "ambiguous_outcome_field_error"
REASON_EXTERNAL_INTERFERENCE = "external_interference_suspected"
REASON_UNMODELED_PARK_RULE = "unmodeled_park_rule_suspected"
REASON_PLAYER_RULE_VIOLATION = "player_rule_violation_suspected"
REASON_WALKOFF_TRUNCATION = "walkoff_truncation_suspected"


@dataclass(frozen=True)
class OutcomeMapping:
    """Result of mapping one row's `events` value to a training outcome.

    Attributes:
        outcome_class: One of `CLASS_ORDER`, or None if the outcome cannot
            be reliably determined (no label is invented in that case).
        eligible_for_training: Whether this event type alone (ignoring data
            completeness and rare-flag checks) supports a training label.
        exclusion_reason: Reason string if `eligible_for_training` is False,
            else None.
    """

    outcome_class: str | None
    eligible_for_training: bool
    exclusion_reason: str | None


def map_outcome_class(events: str | None) -> OutcomeMapping:
    """Map a single Statcast `events` value to a Version 0.1 outcome class.

    Conservative by design: fielder's-choice and field-error events do not
    reliably encode which base the batter-runner reached in public Statcast
    data, so they are returned with `outcome_class=None` and
    `eligible_for_training=False` rather than a guessed label.

    Args:
        events: The raw Statcast `events` string (may be None/NaN).

    Returns:
        An `OutcomeMapping` describing the result.
    """
    if events is None or (isinstance(events, float) and pd.isna(events)):
        return OutcomeMapping(None, False, REASON_NOT_ELIGIBLE_EVENT)

    if events not in ELIGIBLE_EVENTS_V0_1:
        return OutcomeMapping(None, False, REASON_NOT_ELIGIBLE_EVENT)

    if events in _UNAMBIGUOUS_OUT_EVENTS:
        return OutcomeMapping("out", True, None)

    if events in _DIRECT_HIT_EVENTS:
        return OutcomeMapping(_DIRECT_HIT_EVENTS[events], True, None)

    if events == "fielders_choice":
        return OutcomeMapping(None, False, REASON_AMBIGUOUS_FIELDERS_CHOICE)

    if events == "field_error":
        return OutcomeMapping(None, False, REASON_AMBIGUOUS_FIELD_ERROR)

    raise AssertionError(f"Unhandled eligible event '{events}' -- update eligibility.py")


def _contains_any(text: object, keywords: tuple[str, ...]) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def compute_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Version 0.1 eligibility flags and outcome labels for a table.

    Adds the following columns (see README.md / CLAUDE.md for definitions):
        - `is_eligible`: event type is in `ELIGIBLE_EVENTS_V0_1`.
        - `outcome_class`: mapped label in `CLASS_ORDER`, or null if
          undetermined.
        - `missing_contact_data`: True if a required contact field is null.
        - `external_interference`, `unmodeled_park_rule`,
          `player_rule_violation`, `walkoff_truncation`: conservative,
          best-effort text-based flags. These default to False rather than
          aggressively inferring rare situations from ambiguous text, and
          are known-incomplete in Version 0.1 (see README limitations).
        - `eligible_for_training`: True only if the event type is
          unambiguous, required contact data is present, and no rare-event
          flag is set.
        - `exclusion_reason`: reason `eligible_for_training` is False, else
          null.

    Args:
        df: A DataFrame containing at least an `events` column. A
            `description` column is used, if present, for the best-effort
            text flags.

    Returns:
        A copy of `df` with the eligibility/outcome columns added.
    """
    if "events" not in df.columns:
        raise ValueError("compute_eligibility requires an 'events' column")

    out = df.copy()
    description = (
        out["description"]
        if "description" in out.columns
        else pd.Series([None] * len(out), index=out.index)
    )

    mappings = [map_outcome_class(e) for e in out["events"]]
    out["is_eligible"] = out["events"].isin(ELIGIBLE_EVENTS_V0_1)
    out["outcome_class"] = pd.array([m.outcome_class for m in mappings], dtype="string")
    outcome_eligible = pd.Series([m.eligible_for_training for m in mappings], index=out.index)
    outcome_reason = pd.Series([m.exclusion_reason for m in mappings], index=out.index)

    missing_required = pd.Series(False, index=out.index)
    for field in REQUIRED_CONTACT_FIELDS:
        if field in out.columns:
            missing_required = missing_required | out[field].isna()
        else:
            missing_required = pd.Series(True, index=out.index)
    out["missing_contact_data"] = missing_required

    out["external_interference"] = description.apply(
        lambda text: _contains_any(text, _INTERFERENCE_KEYWORDS)
    )
    out["unmodeled_park_rule"] = description.apply(
        lambda text: _contains_any(text, _PARK_RULE_KEYWORDS)
    )
    out["player_rule_violation"] = description.apply(
        lambda text: _contains_any(text, _RULE_VIOLATION_KEYWORDS)
    )
    out["walkoff_truncation"] = description.apply(
        lambda text: _contains_any(text, _WALKOFF_KEYWORDS)
    )

    rare_flag_hit = (
        out["external_interference"]
        | out["unmodeled_park_rule"]
        | out["player_rule_violation"]
        | out["walkoff_truncation"]
    )

    eligible_for_training = outcome_eligible & ~missing_required & ~rare_flag_hit

    reason = outcome_reason.copy()
    reason = reason.where(~missing_required, REASON_MISSING_CONTACT_DATA)
    reason = reason.where(~out["external_interference"], REASON_EXTERNAL_INTERFERENCE)
    reason = reason.where(~out["unmodeled_park_rule"], REASON_UNMODELED_PARK_RULE)
    reason = reason.where(~out["player_rule_violation"], REASON_PLAYER_RULE_VIOLATION)
    reason = reason.where(~out["walkoff_truncation"], REASON_WALKOFF_TRUNCATION)
    reason = reason.where(~eligible_for_training, None)

    out["eligible_for_training"] = eligible_for_training
    out["training_exclusion_reason"] = reason
    return out


# ---------------------------------------------------------------------------
# Version 0.7A: outfield-opportunity eligibility
# ---------------------------------------------------------------------------
#
# A FURTHER restriction of `eligible_for_training` rows to those where an
# OUTFIELDER (not an infielder) had the fielding opportunity -- needed
# because the Version 0.7A opportunity model's target is specifically
# "would an average MLB OUTFIELDER convert this into an out", not any
# fielder. Kept here (not duplicated in a models/ or scoring/ module) per
# this file's "centralize eligibility logic" mandate.

#: `bb_type` values in scope for the outfield-opportunity model. `popup` is
#: EXCLUDED: verified against real 2024 data, popups are assigned an infield
#: `hit_location` (1-6) 99.5% of the time (8,756 of 8,803) -- they are
#: overwhelmingly an infield play type, not an outfield one, in this
#: dataset. `ground_ball` is excluded because it was never airborne toward
#: the outfield.
OUTFIELD_AIR_BALL_TYPES: frozenset[str] = frozenset({"fly_ball", "line_drive"})

#: Statcast `hit_location` codes assigned to outfield positions (7=LF,
#: 8=CF, 9=RF) vs. infield positions (1=P, 2=C, 3=1B, 4=2B, 5=3B, 6=SS).
OUTFIELD_HIT_LOCATIONS: frozenset[int] = frozenset({7, 8, 9})
INFIELD_HIT_LOCATIONS: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6})

#: The Statcast column identifying which player was playing each outfield
#: position on a given play (see `mlb_luck_score.data.download_statcast` --
#: `fielder_7`/`fielder_8`/`fielder_9` give the CURRENT player ID at that
#: position for that specific pitch, not just a season-average roster spot).
OUTFIELD_POSITION_FIELDER_COLUMN: dict[int, str] = {7: "fielder_7", 8: "fielder_8", 9: "fielder_9"}

#: Fallback mapping from `spray_sector` (`mlb_luck_score.data.
#: clean_batted_balls`) to an outfield `hit_location` code, used ONLY when
#: `hit_location` itself is missing -- verified against real 2024 data, this
#: is overwhelmingly home runs (4,980 of 5,164 missing-hit_location fly
#: balls) and deep doubles that were never fielded, so no real Statcast
#: fielding credit exists to look up; a coarse position estimate from spray
#: direction is the best available public proxy. Groups `left`/`left_center`
#: to LF and `right_center`/`right` to RF -- the SAME grouping already used
#: by `mlb_luck_score.data.clean_batted_balls._PULL_SECTORS_FOR_RIGHTY`, for
#: consistency with the rest of the codebase, not a new convention.
SPRAY_SECTOR_TO_OUTFIELD_POSITION: dict[str, int] = {
    "left": 7,
    "left_center": 7,
    "center": 8,
    "right_center": 9,
    "right": 9,
}

REASON_NOT_OUTFIELD_AIR_BALL = "not_outfield_air_ball_bb_type"
REASON_INFIELD_CREDITED_HIT_LOCATION = "infield_credited_hit_location"
REASON_BASE_TRAINING_INELIGIBLE = "base_training_ineligible"

#: Position-assignment provenance -- NEVER confuse the spray-direction
#: fallback with a real Statcast-credited fielding assignment.
POSITION_SOURCE_HIT_LOCATION = "hit_location"
POSITION_SOURCE_SPRAY_SECTOR_PROXY = "spray_sector_proxy"


def add_outfield_opportunity_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """Add Version 0.7A outfield-opportunity eligibility, position, and fielder-identity columns.

    Must be called AFTER `compute_eligibility` (uses `eligible_for_training`)
    and `mlb_luck_score.data.clean_batted_balls.add_spray_features` (uses
    `spray_sector`, only as a fallback -- see `SPRAY_SECTOR_TO_OUTFIELD_
    POSITION`).

    Adds:
        - `outfield_opportunity_eligible`: True only if `eligible_for_
          training` is True, `bb_type` is in `OUTFIELD_AIR_BALL_TYPES`, and
          `hit_location` is either an outfield code (7/8/9) or missing (the
          home-run/deep-double case -- see module comment). A row with an
          INFIELD-credited `hit_location` (an infielder made the play) is
          excluded -- the opportunity was never an outfielder's to begin
          with.
        - `outfield_opportunity_exclusion_reason`: reason string if not
          eligible, else None.
        - `assigned_outfield_position`: 7/8/9 (nullable `Int64`), or null if
          not outfield-opportunity-eligible.
        - `assigned_outfield_position_source`: `"hit_location"` (real
          Statcast fielding credit) or `"spray_sector_proxy"` (coarse
          fallback for missing-hit_location rows) -- always check this
          before treating `assigned_outfield_position` as measured data.
        - `responsible_outfielder_id`: the player ID from `fielder_7`/
          `fielder_8`/`fielder_9` corresponding to `assigned_outfield_
          position`, or null if unavailable/not eligible. This IS a real
          Statcast-recorded identity regardless of `assigned_outfield_
          position_source` (the fielder columns are always accurate for who
          was playing that position; only the POSITION ASSIGNMENT itself is
          sometimes a proxy).

    Args:
        df: A DataFrame with `bb_type`, `eligible_for_training`, and
            ideally `hit_location`, `spray_sector`, `fielder_7`,
            `fielder_8`, `fielder_9` (any missing column degrades
            gracefully -- see returns below -- rather than raising).

    Returns:
        A copy of `df` with the above columns added.
    """
    if "bb_type" not in df.columns or "eligible_for_training" not in df.columns:
        raise ValueError(
            "add_outfield_opportunity_eligibility requires 'bb_type' and "
            "'eligible_for_training' columns -- run compute_eligibility first."
        )

    out = df.copy()
    base_eligible = out["eligible_for_training"].astype(bool)
    is_air_ball_type = out["bb_type"].isin(OUTFIELD_AIR_BALL_TYPES)

    if "hit_location" in out.columns:
        hit_location_numeric = pd.to_numeric(out["hit_location"], errors="coerce")
    else:
        hit_location_numeric = pd.Series(float("nan"), index=out.index)
    has_of_hit_location = hit_location_numeric.isin(OUTFIELD_HIT_LOCATIONS)
    has_if_hit_location = hit_location_numeric.isin(INFIELD_HIT_LOCATIONS)
    hit_location_missing = hit_location_numeric.isna()

    outfield_opportunity_eligible = (
        base_eligible & is_air_ball_type & (has_of_hit_location | hit_location_missing)
    )
    out["outfield_opportunity_eligible"] = outfield_opportunity_eligible

    reason = pd.Series(None, index=out.index, dtype=object)
    reason = reason.where(base_eligible, REASON_BASE_TRAINING_INELIGIBLE)
    reason = reason.where(~(base_eligible & ~is_air_ball_type), REASON_NOT_OUTFIELD_AIR_BALL)
    reason = reason.where(
        ~(base_eligible & is_air_ball_type & has_if_hit_location),
        REASON_INFIELD_CREDITED_HIT_LOCATION,
    )
    reason = reason.where(~outfield_opportunity_eligible, None)
    out["outfield_opportunity_exclusion_reason"] = reason

    assigned_position = pd.Series(pd.NA, index=out.index, dtype="Int64")
    position_source = pd.Series(None, index=out.index, dtype=object)

    from_hit_location_mask = outfield_opportunity_eligible & has_of_hit_location
    assigned_position.loc[from_hit_location_mask] = hit_location_numeric.loc[
        from_hit_location_mask
    ].astype("Int64")
    position_source.loc[from_hit_location_mask] = POSITION_SOURCE_HIT_LOCATION

    from_proxy_mask = outfield_opportunity_eligible & hit_location_missing
    if "spray_sector" in out.columns and from_proxy_mask.any():
        proxy_position = out.loc[from_proxy_mask, "spray_sector"].map(
            SPRAY_SECTOR_TO_OUTFIELD_POSITION
        )
        assigned_position.loc[from_proxy_mask] = proxy_position.astype("Int64")
        position_source.loc[from_proxy_mask] = POSITION_SOURCE_SPRAY_SECTOR_PROXY

    out["assigned_outfield_position"] = assigned_position
    out["assigned_outfield_position_source"] = position_source

    responsible_fielder = pd.Series(pd.NA, index=out.index, dtype="object")
    for position, fielder_col in OUTFIELD_POSITION_FIELDER_COLUMN.items():
        if fielder_col not in out.columns:
            continue
        position_mask = assigned_position == position
        responsible_fielder.loc[position_mask] = out.loc[position_mask, fielder_col]
    out["responsible_outfielder_id"] = responsible_fielder

    return out


# ---------------------------------------------------------------------------
# Version 0.8: infield-opportunity eligibility
# ---------------------------------------------------------------------------
#
# A narrow restriction identifying fair GROUND balls fielded by an INFIELDER
# (pitcher/1B/2B/SS/3B) where the batter-runner is the unambiguous, sole out
# opportunity. Built INDEPENDENTLY of Version 0.1's `eligible_for_training`/
# `map_outcome_class` above -- reusing them would be WRONG here: Version 0.1
# groups `force_out`/`double_play`/`grounded_into_double_play`/
# `fielders_choice_out`/`sac_bunt` into outcome_class "out" because the
# BATTER's own extra-base value is zero either way (a contact-luck/run-value
# question), and separately marks `field_error` INELIGIBLE because the
# batter's resulting BASE can't be determined (a hit-classification
# question). Version 0.8 asks a different, narrower question -- "was the
# BATTER-RUNNER physically retired on this specific play" -- for which
# `field_error` is a perfectly clean "not retired" case (no base
# classification needed, just safe-vs-out) and `force_out`/`fielders_choice`
# -type events are NOT batter-runner outs at all (a preceding runner is the
# one retired; the batter himself typically reaches first safely) and must
# be excluded, not folded into "out." See README.md "Infield opportunity
# (Version 0.8)" for the real 2021-2024 row counts by exclusion category and
# CLAUDE.md/AGENTS.md for why this is a NEW eligibility question, not a bug
# fix to Version 0.1's.

#: `events` values where the batter-runner's fate is CLEAN and unambiguous:
#: retired (`field_out`) or safely reached base, INCLUDING on an error
#: (`field_error` -- the defense failed to complete the conversion; see
#: `add_infield_opportunity_eligibility`'s `reached_on_error` column). Never
#: use the scorer's error/hit classification as a model INPUT -- only as this
#: eligibility/labeling step and the separate `reached_on_error` reporting
#: indicator (task Phase 3).
INFIELD_RETIRED_EVENT = "field_out"
INFIELD_SAFE_EVENTS: frozenset[str] = frozenset({"single", "double", "triple", "field_error"})
INFIELD_CLEAN_OUTCOME_EVENTS: frozenset[str] = (
    frozenset({INFIELD_RETIRED_EVENT}) | INFIELD_SAFE_EVENTS
)

#: `events` values EXCLUDED from the primary Version 0.8 model because the
#: defense's target was not (only) the batter-runner: lead-runner force
#: plays, fielder's choices, double-play attempts, and sacrifice bunts/flies.
#: Verified against real 2021-2024 ground balls: these are the ONLY event
#: values, alongside `INFIELD_CLEAN_OUTCOME_EVENTS`, that occur at all for
#: `bb_type == "ground_ball"` (11 total; no `triple_play`, no null `events`).
INFIELD_EXCLUDED_STRATEGIC_EVENTS: frozenset[str] = frozenset(
    {
        "force_out",
        "fielders_choice",
        "fielders_choice_out",
        "double_play",
        "grounded_into_double_play",
        "sac_fly_double_play",
        "sac_bunt",
        "sac_fly",
    }
)

#: Statcast `hit_location` codes assigned to infield positions -- pitcher(1),
#: catcher(2, essentially never a real ground-ball fielder of interest but
#: kept for completeness), 1B(3), 2B(4), 3B(5), SS(6). Reuses the SAME
#: `INFIELD_HIT_LOCATIONS` frozenset already defined above for the Version
#: 0.7A outfield section (this file's single source of truth for the
#: position-code mapping) rather than redefining it.
#:
#: Verified against real 2021-2024 ground balls: 16% (34,659 of 214,032) have
#: an OUTFIELD-credited `hit_location` (7/8/9) -- the ball got through the
#: infield entirely before any fielder touched it. These are excluded: no
#: infielder ever had a genuine opportunity on that specific play.
INFIELD_POSITION_FIELDER_COLUMN: dict[int, str] = {
    1: "pitcher",  # no separate `fielder_1` column -- the pitcher IS fielder 1
    2: "fielder_2",
    3: "fielder_3",
    4: "fielder_4",
    5: "fielder_5",
    6: "fielder_6",
}

#: Keyword heuristics for the free-text `des` field -- the ONLY source for
#: these flags in public Statcast data (no dedicated boolean column exists
#: for any of them). Narrow and case-insensitive, matching this file's
#: existing `_INTERFERENCE_KEYWORDS`-style convention. Verified against real
#: 2021-2024 ground balls (see README.md for exact counts):
#:   - "bunt": catches non-sacrifice bunts too (2,364 of 214,032 ground balls
#:     have "bunt" in `des` with an `events` value OTHER than `sac_bunt` --
#:     bunt singles, bunt outs, bunt force outs -- none of which are tagged
#:     any other way in this schema).
#:   - "interference"/"obstruction": catches BOTH directions -- a batter
#:     ruled out for "batter interference" (recorded as an ordinary
#:     `field_out`, NOT a real defensive conversion: a rules-violation
#:     penalty, not fielding) and a hit awarded extra bases on "fan
#:     interference" (recorded as an ordinary `single`/`double`/`triple`, NOT
#:     a real defensive failure: an external event). Both must be excluded
#:     regardless of which `events` bucket they land in -- a pure
#:     `events`-value filter would silently miss both. "Catcher interference"
#:     cannot co-occur with a batted ball by construction (Statcast's
#:     `catcher_interf` event has no batted-ball trajectory at all --
#:     verified zero co-occurrence with `bb_type == "ground_ball"`), so it
#:     never reaches this check in practice, but the keyword still catches it
#:     if `des` ever mentions it.
#:   - "appeal"/"rundown": verified ZERO real 2021-2024 ground-ball rows
#:     match either -- kept as a documented, checkable exclusion category per
#:     the task's explicit list, not because it fires today.
_BUNT_KEYWORD = "bunt"
_INTERFERENCE_OR_OBSTRUCTION_KEYWORDS = ("interference", "obstruction")
_APPEAL_KEYWORD = "appeal"
_RUNDOWN_KEYWORDS = ("rundown", "run down")
#: INFORMATIONAL ONLY -- reviewed/overturned plays are NOT excluded (see
#: `add_infield_opportunity_eligibility`'s `reviewed_or_overturned` column
#: docstring for why), just counted and reported per task Phase 1's audit
#: ("reviewed or overturned outcomes").
_REVIEW_KEYWORDS = ("challenged", "overturned", " review")

REASON_NOT_GROUND_BALL = "not_ground_ball_bb_type"
REASON_BUNT_EXCLUDED = "bunt_excluded"
REASON_INTERFERENCE_OR_OBSTRUCTION = "interference_or_obstruction"
REASON_APPEAL_PLAY = "appeal_play"
REASON_RUNDOWN_PLAY = "rundown_play"
REASON_EXCLUDED_STRATEGIC_PLAY = "excluded_strategic_play"
REASON_AMBIGUOUS_INFIELD_EVENT = "ambiguous_or_unmapped_event"
REASON_MISSING_INFIELD_POSITION = "missing_responsible_infield_position"
REASON_OUTFIELD_CREDITED_HIT_LOCATION = "outfield_credited_hit_location"
REASON_MISSING_INFIELD_CONTACT_DATA = "missing_required_contact_data"


def _contains_keyword(text: object, keyword: str) -> bool:
    return isinstance(text, str) and keyword in text.lower()


def _contains_any_keyword(text: object, keywords: tuple[str, ...]) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def add_infield_opportunity_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """Add Version 0.8 infield-opportunity eligibility, label, and identity columns.

    Does NOT require `compute_eligibility` to have been run first -- unlike
    the Version 0.7A outfield section, this builds its own eligibility
    directly from raw `events`/`bb_type`/`hit_location`/`des`, since Version
    0.1's `eligible_for_training` answers a different question (see module
    comment above).

    Adds:
        - `infield_opportunity_eligible`: True only if ALL of: `bb_type ==
          "ground_ball"`; not bunt/interference-or-obstruction/appeal/
          rundown (per `des` keyword checks); `events` is in
          `INFIELD_CLEAN_OUTCOME_EVENTS` (not an excluded strategic-play
          event, and not some other unmapped event); `hit_location` is a
          real infield code (1-6); `launch_speed`/`launch_angle` are both
          present.
        - `infield_opportunity_exclusion_reason`: reason string if not
          eligible, else None. Precedence (first applicable reason wins,
          checked in this order): not-ground-ball, bunt, interference/
          obstruction, appeal, rundown, excluded-strategic-event, ambiguous
          -event, missing-position, outfield-credited-position,
          missing-contact-data.
        - `y_out`: 1 if `events == "field_out"`, 0 if in
          `INFIELD_SAFE_EVENTS`, else null. Task Phase 3's label -- a
          reached-on-error play gets `y_out = 0` (the defense did NOT
          complete the conversion), never `1`.
        - `reached_on_error`: True iff `events == "field_error"` --
          reporting/evaluation ONLY (task Phase 3: "Never use the scorer's
          error classification as a predictor"), never a model input.
        - `is_bunt`, `has_interference_or_obstruction`, `is_appeal_play`,
          `is_rundown_play`: the individual `des`-keyword flags (see module
          comment for exact keywords and real-data counts).
        - `reviewed_or_overturned`: `des` mentions "challenged"/"overturned"/
          "review" -- INFORMATIONAL ONLY, never used to exclude a row.
          Statcast's `events`/`des` already reflect the FINAL, corrected
          ruling after any review, so the recorded label is not made less
          reliable by having been reviewed; excluding these would introduce
          an unprincipled selection bias (high-leverage/close plays are
          reviewed more often) with no data-quality justification. Reported
          per task Phase 1's audit request, not filtered on.
        - `assigned_infield_position`: 1-6 (nullable `Int64`, from
          `hit_location`), or null if not infield-opportunity-eligible.
        - `responsible_infielder_id`: the player ID from `pitcher` (position
          1 -- there is no separate `fielder_1` Statcast column) or
          `fielder_2`.."fielder_6"` corresponding to `assigned_infield_
          position`, or null if unavailable/not eligible. Individual
          -defender identity, for POST-HOC evaluation only -- never a model
          input (task Phase 4).

    Args:
        df: A DataFrame with `events`, `bb_type`, `hit_location`,
            `launch_speed`, `launch_angle`, and ideally `des`, `pitcher`,
            `fielder_2`..`fielder_6` (any missing optional column degrades
            gracefully rather than raising).

    Returns:
        A copy of `df` with the above columns added.
    """
    required = ("events", "bb_type", "hit_location", "launch_speed", "launch_angle")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"add_infield_opportunity_eligibility requires column(s) {missing}")

    out = df.copy()
    des = out["des"] if "des" in out.columns else pd.Series([None] * len(out), index=out.index)

    is_ground_ball = out["bb_type"] == "ground_ball"
    is_bunt = des.apply(lambda text: _contains_keyword(text, _BUNT_KEYWORD))
    has_interference_or_obstruction = des.apply(
        lambda text: _contains_any_keyword(text, _INTERFERENCE_OR_OBSTRUCTION_KEYWORDS)
    )
    is_appeal_play = des.apply(lambda text: _contains_keyword(text, _APPEAL_KEYWORD))
    is_rundown_play = des.apply(lambda text: _contains_any_keyword(text, _RUNDOWN_KEYWORDS))
    reviewed_or_overturned = des.apply(lambda text: _contains_any_keyword(text, _REVIEW_KEYWORDS))

    out["is_bunt"] = is_bunt
    out["has_interference_or_obstruction"] = has_interference_or_obstruction
    out["is_appeal_play"] = is_appeal_play
    out["is_rundown_play"] = is_rundown_play
    out["reviewed_or_overturned"] = reviewed_or_overturned

    is_clean_event = out["events"].isin(INFIELD_CLEAN_OUTCOME_EVENTS)
    is_excluded_strategic_event = out["events"].isin(INFIELD_EXCLUDED_STRATEGIC_EVENTS)

    hit_location_numeric = pd.to_numeric(out["hit_location"], errors="coerce")
    has_infield_hit_location = hit_location_numeric.isin(INFIELD_HIT_LOCATIONS)
    has_outfield_hit_location = hit_location_numeric.isin(OUTFIELD_HIT_LOCATIONS)
    hit_location_missing = hit_location_numeric.isna()

    missing_contact_data = out["launch_speed"].isna() | out["launch_angle"].isna()

    infield_opportunity_eligible = (
        is_ground_ball
        & ~is_bunt
        & ~has_interference_or_obstruction
        & ~is_appeal_play
        & ~is_rundown_play
        & is_clean_event
        & has_infield_hit_location
        & ~missing_contact_data
    )
    out["infield_opportunity_eligible"] = infield_opportunity_eligible

    reason = pd.Series(None, index=out.index, dtype=object)
    reason = reason.where(is_ground_ball, REASON_NOT_GROUND_BALL)
    reason = reason.where(~(is_ground_ball & is_bunt), REASON_BUNT_EXCLUDED)
    reason = reason.where(
        ~(is_ground_ball & ~is_bunt & has_interference_or_obstruction),
        REASON_INTERFERENCE_OR_OBSTRUCTION,
    )
    reason = reason.where(
        ~(is_ground_ball & ~is_bunt & ~has_interference_or_obstruction & is_appeal_play),
        REASON_APPEAL_PLAY,
    )
    reason = reason.where(
        ~(
            is_ground_ball
            & ~is_bunt
            & ~has_interference_or_obstruction
            & ~is_appeal_play
            & is_rundown_play
        ),
        REASON_RUNDOWN_PLAY,
    )
    clean_base = (
        is_ground_ball
        & ~is_bunt
        & ~has_interference_or_obstruction
        & ~is_appeal_play
        & ~is_rundown_play
    )
    reason = reason.where(
        ~(clean_base & is_excluded_strategic_event), REASON_EXCLUDED_STRATEGIC_PLAY
    )
    reason = reason.where(
        ~(clean_base & ~is_excluded_strategic_event & ~is_clean_event),
        REASON_AMBIGUOUS_INFIELD_EVENT,
    )
    reason = reason.where(
        ~(clean_base & is_clean_event & hit_location_missing), REASON_MISSING_INFIELD_POSITION
    )
    reason = reason.where(
        ~(clean_base & is_clean_event & has_outfield_hit_location),
        REASON_OUTFIELD_CREDITED_HIT_LOCATION,
    )
    reason = reason.where(
        ~(clean_base & is_clean_event & has_infield_hit_location & missing_contact_data),
        REASON_MISSING_INFIELD_CONTACT_DATA,
    )
    reason = reason.where(~infield_opportunity_eligible, None)
    out["infield_opportunity_exclusion_reason"] = reason

    y_out = pd.Series(pd.NA, index=out.index, dtype="Int64")
    y_out.loc[out["events"] == INFIELD_RETIRED_EVENT] = 1
    y_out.loc[out["events"].isin(INFIELD_SAFE_EVENTS)] = 0
    y_out.loc[~infield_opportunity_eligible] = pd.NA
    out["y_out"] = y_out

    reached_on_error = (out["events"] == "field_error") & infield_opportunity_eligible
    out["reached_on_error"] = reached_on_error

    assigned_position = pd.Series(pd.NA, index=out.index, dtype="Int64")
    assigned_position.loc[infield_opportunity_eligible] = hit_location_numeric.loc[
        infield_opportunity_eligible
    ].astype("Int64")
    out["assigned_infield_position"] = assigned_position

    responsible_fielder = pd.Series(pd.NA, index=out.index, dtype="object")
    for position, fielder_col in INFIELD_POSITION_FIELDER_COLUMN.items():
        if fielder_col not in out.columns:
            continue
        position_mask = assigned_position == position
        responsible_fielder.loc[position_mask] = out.loc[position_mask, fielder_col]
    out["responsible_infielder_id"] = responsible_fielder

    return out


@dataclass(frozen=True)
class InfieldExclusionCounts:
    """Row counts by exclusion category and season -- task Phase 2's required report."""

    by_season: dict[int, dict[str, int]]
    total: dict[str, int]


def summarize_infield_exclusions(df: pd.DataFrame) -> InfieldExclusionCounts:
    """Row counts by `infield_opportunity_exclusion_reason` (plus "eligible"), per season and overall.

    Args:
        df: A DataFrame already passed through `add_infield_opportunity_
            eligibility`, with a `season` column.

    Returns:
        `InfieldExclusionCounts` -- `by_season[season][reason_or_"eligible"]
        = count`, and the same totals across all seasons in `.total`.
    """
    working = df.copy()
    label = working["infield_opportunity_exclusion_reason"].astype(object)
    label = label.where(~working["infield_opportunity_eligible"].astype(bool), "eligible")
    working["_label"] = label

    total: dict[str, int] = {
        str(k): int(v) for k, v in working["_label"].value_counts(dropna=False).items()
    }
    by_season: dict[int, dict[str, int]] = {}
    if "season" in working.columns:
        for season_value in working["season"].dropna().unique():
            group = working[working["season"] == season_value]
            by_season[int(season_value)] = {
                str(k): int(v) for k, v in group["_label"].value_counts(dropna=False).items()
            }

    return InfieldExclusionCounts(by_season=by_season, total=total)


# ---------------------------------------------------------------------------
# Version 0.9: batter-runner advancement eligibility and labeling
# ---------------------------------------------------------------------------
#
# The TARGET for this phase -- the batter-runner's final base at the end of
# the continuous play arising from their own batted ball -- is NOT a
# Statcast column. `events` records only the batter's HIT TYPE (single/
# double/triple/home_run), which is the base they were credited with on the
# batted ball itself, not necessarily where they ended up: a real, verified
# 2021-2024 example is a batter credited with a "single" who is later thrown
# out stretching for second (`events` stays "single"; `des` narrates the
# out), or a batter credited with a "double"/"triple" who scores anyway on a
# subsequent throwing/fielding error (`events` stays "double"/"triple"; only
# `des` reveals the extra advancement). This is why this label MUST be
# reconstructed from the free-text `des` field, not read off a controlled
# column, unlike every other outcome-class mapping in this file.
#
# The parsing method (validated against the real, already-downloaded
# 2021-2024 dataset before being written here -- see README.md "Batter
# -runner advancement (Version 0.9)" for the exact real coverage numbers):
#
#   1. Every `des` string for an eligible play begins with the batter's OWN
#      name, followed by a hit-type verb ("<Name> singles on a...", "<Name>
#      reaches on a fielding error by...", etc.) -- extracting the name from
#      this leading clause via regex succeeds on essentially 100% of real
#      eligible rows and requires no external batter-ID-to-name lookup.
#   2. A small number of real rows (roughly 0.8% of the eligible population)
#      have a review/challenge PREAMBLE before that leading clause (e.g.
#      "Blue Jays challenged (tag play), call on the field was overturned:
#      <Name> singles..."), sometimes doubled ("X challenged (...): X
#      challenged (...): <Name> singles..."). An EARLIER version of this
#      parser did not strip this preamble, which silently corrupted the
#      extracted batter name (it absorbed the whole preamble) and therefore
#      silently mislabeled these rows as `held_at_first`/`advanced_to_
#      second`/etc. instead of their true, often more-advanced or retired,
#      outcome -- caught only by comparing label counts before and after
#      adding `_strip_review_preamble` (`retired_while_advancing` moved from
#      734 to 887 real rows once fixed). `reviewed_or_overturned` is kept as
#      an INFORMATIONAL flag (same convention as Version 0.8's identically
#      -named column) -- these rows are NOT excluded, since Statcast's `des`
#      already reflects the final, corrected ruling.
#   3. With the batter's name in hand, the REST of `des` is searched for
#      that exact name (regex-escaped, so punctuation in real names --
#      periods, apostrophes, hyphens, suffixes -- is handled automatically)
#      against four specific clause patterns: retired while advancing
#      ("<Name> out at 2nd/3rd/home..."), error-driven advancement ("<Name>
#      advances to 2nd/3rd/home, on a throwing/fielding error by..."),
#      simple advancement ("<Name> to 2nd/3rd."), and scoring ("<Name>
#      scores."). If NONE of these match, the batter is assumed to have
#      stayed at the hit-implied base (verified correct against a manual
#      stratified sample -- see README.md).
#   4. Every batter-name occurrence in the remainder of `des` MUST be
#      accounted for by one of these four patterns; if the name appears
#      again in a form none of them recognize, the row is `unsupported_
#      play_sequence`, NEVER silently defaulted to "stayed at the
#      hit-implied base". Genuinely contradictory matches (e.g. both
#      "retired" and "scores" for the same name) are `ambiguous`. Both,
#      like every other unresolved status, are EXCLUDED from the eligible/
#      labeled population, never force-labeled -- see `add_advancement_
#      eligibility`.
#
# KNOWN, ACCEPTED LIMITATION: if two DIFFERENT physical players share an
# EXACT full name within the same play's `des` text (astronomically rare --
# not observed in the real 2021-2024 validation sample), this parser cannot
# disambiguate them from text alone and could attribute one player's clause
# to the other. This is documented here rather than defended against with
# extra machinery, matching this module's general philosophy of flagging
# genuine ambiguity as `ambiguous`/`unsupported_play_sequence` rather than
# guessing -- if this ever manifests in a future real-data validation pass,
# treat it as a `parse_failure`-class bug, not a modeling nuance.
#
# LABELING DESIGN NOTE on `INSIDE_THE_PARK_HOME_RUN`: this label is used for
# EVERY case where the batter-runner reaches home plate during the
# continuous play -- both a genuine inside-the-park home run (`events ==
# "home_run"` with "inside-the-park" in `des`) AND the rarer case of a
# batter credited with a lesser hit (single/double/triple) who nonetheless
# scores via a subsequent fielding/throwing error (`events` stays e.g.
# "triple"; real 2021-2024 example: "Jose Iglesias triples (2)... Jose
# Iglesias scores. Throwing error by second baseman Javier Baez."). Both are
# the SAME terminal state from the batter-runner's own perspective (reached
# home safely, ball never left the field of play) even though they have
# DIFFERENT `events` values -- `is_true_inside_the_park_home_run` is kept as
# a separate informational column so the two real phenomena remain
# distinguishable for anyone who needs that finer distinction. Plain
# over-the-fence home runs (`events == "home_run"` without "inside-the-park"
# in `des`) are explicitly EXCLUDED from the eligible population (`REASON_
# TRIVIAL_NO_ADVANCEMENT_OPPORTUNITY`): once a ball clears the fence, the
# batter-runner's advancement to home is certain and entirely
# defense-independent -- there is no genuine "opportunity" for a model to
# estimate, and including these rows would only add trivial, uninformative
# mass to the target distribution.
#
# ELIGIBILITY SCOPE for this first Version 0.9 candidate (per the task):
# fair OUTFIELD air balls (REUSES `OUTFIELD_AIR_BALL_TYPES` -- the SAME
# `fly_ball`/`line_drive` scope Version 0.7A already established, not a new
# convention) where the batter-runner safely reaches at least first
# (`events` in `ADVANCEMENT_SAFE_EVENTS`). Infield hits are explicitly
# DEFERRED (per the task: "overthrows and hurried throws complicate
# attribution") -- ground balls are excluded here entirely, not silently
# included. Preexisting-runner advancement is NOT modeled by this parser at
# all -- only the BATTER's own final base.

#: Reuses `OUTFIELD_AIR_BALL_TYPES` verbatim -- the same bb_type scope
#: Version 0.7A already established for "this was an outfield play."
ADVANCEMENT_ELIGIBLE_BB_TYPES: frozenset[str] = OUTFIELD_AIR_BALL_TYPES

#: `events` values where the batter-runner is known to have reached base
#: safely (on the batted ball itself, before any parsing) -- `field_error`
#: is included because "reaches on a throwing/fielding error" is a clean
#: safe-arrival case for this labeling question (contrast with Version 0.1's
#: `map_outcome_class`, which leaves `field_error` unmapped because the
#: batter's resulting BASE for RUN-VALUE purposes can't be reliably
#: determined there -- a different question than "did they reach base,"
#: which `des` answers directly for this phase).
ADVANCEMENT_SAFE_EVENTS: frozenset[str] = frozenset(
    {"single", "double", "triple", "home_run", "field_error"}
)

HELD_AT_FIRST = "held_at_first"
ADVANCED_TO_SECOND = "advanced_to_second"
ADVANCED_TO_THIRD = "advanced_to_third"
INSIDE_THE_PARK_HOME_RUN = "inside_the_park_home_run"
RETIRED_WHILE_ADVANCING = "retired_while_advancing"
ADVANCEMENT_LABELS: tuple[str, ...] = (
    HELD_AT_FIRST,
    ADVANCED_TO_SECOND,
    ADVANCED_TO_THIRD,
    INSIDE_THE_PARK_HOME_RUN,
    RETIRED_WHILE_ADVANCING,
)

#: Task safeguard #1's required parse-status vocabulary.
PARSE_STATUS_UNAMBIGUOUS = "parsed_unambiguous"
PARSE_STATUS_AMBIGUOUS = "ambiguous"
PARSE_STATUS_NO_BATTER_MATCH = "no_batter_match"
PARSE_STATUS_UNSUPPORTED_SEQUENCE = "unsupported_play_sequence"
PARSE_STATUS_PARSE_FAILURE = "parse_failure"

#: Bumped whenever the parsing RULES below change -- stored on every row
#: (task safeguard #2) so any future re-run can tell which parser version
#: produced a given label without re-deriving it from git history.
ADVANCEMENT_PARSER_VERSION = "v1"

REASON_NOT_ADVANCEMENT_SAFE_EVENT = "not_advancement_safe_event"
REASON_TRIVIAL_NO_ADVANCEMENT_OPPORTUNITY = "trivial_no_advancement_opportunity"
REASON_UNRESOLVED_DES_PARSE = "unresolved_des_parse"

_KIND_INSIDE_PARK_HR = "inside_the_park_hr"
_KIND_TRIVIAL_HR = "trivial_hr"
_KIND_HIT = "hit"
_KIND_HIT_IMMEDIATE_ERROR_ADVANCE = "hit_immediate_error_advance"

#: Ordered: more specific phrases first (e.g. "hits an inside-the-park
#: grand slam" before "hits a grand slam" before "homers") so the FIRST
#: pattern that matches is always the most specific real one, never a
#: coincidental prefix match. Verified against every real `des` verb
#: phrasing observed in the 2021-2024 eligible-scope population (see
#: README.md).
_LEADING_VERB_PATTERNS: tuple[tuple[str, str, int | None, bool], ...] = (
    # (regex, kind, floor_base, reach_on_error)
    (r"hits an inside-the-park grand slam", _KIND_INSIDE_PARK_HR, None, False),
    (r"hits an inside-the-park home run", _KIND_INSIDE_PARK_HR, None, False),
    (r"hits a grand slam", _KIND_TRIVIAL_HR, None, False),
    (r"hits a home run", _KIND_TRIVIAL_HR, None, False),
    (r"homers", _KIND_TRIVIAL_HR, None, False),
    (r"hits a ground-rule double", _KIND_HIT, 2, False),
    (r"doubles", _KIND_HIT, 2, False),
    (r"triples", _KIND_HIT, 3, False),
    (r"singles", _KIND_HIT, 1, False),
    (r"reaches on an? [a-z]+(?: [a-z]+)? error", _KIND_HIT, 1, True),
    # No separate "reaches on error" clause -- the error-driven advance IS
    # the leading clause (e.g. "X advances to 2nd, on a fielding error by
    # Y."). `floor_base` is resolved from the captured base group instead.
    (
        r"advances to (?:2nd|3rd|home), on an? [a-z]+(?: [a-z]+)? error",
        _KIND_HIT_IMMEDIATE_ERROR_ADVANCE,
        None,
        True,
    ),
)
_COMPILED_LEADING_PATTERNS: tuple[tuple[re.Pattern[str], str, int | None, bool], ...] = tuple(
    (re.compile(rf"^(?P<name>.*?) (?P<verb>{regex})"), kind, floor_base, reach_on_error)
    for regex, kind, floor_base, reach_on_error in _LEADING_VERB_PATTERNS
)

_BASE_WORD_TO_NUM: dict[str, int] = {"1st": 1, "2nd": 2, "3rd": 3, "home": 4}
#: Used only by `summarize_advancement_parse_coverage`'s multi-throw-relay
#: proxy -- NOT part of the label-parsing regexes above.
_FIELDING_POSITION_WORDS = (
    r"(?:pitcher|catcher|first baseman|second baseman|third baseman|shortstop"
    r"|left fielder|center fielder|right fielder)"
)
_NUM_TO_LABEL: dict[int, str] = {
    1: HELD_AT_FIRST,
    2: ADVANCED_TO_SECOND,
    3: ADVANCED_TO_THIRD,
    4: INSIDE_THE_PARK_HOME_RUN,
}

#: Strips a leading MLB Gameday review/challenge preamble (e.g. "Blue Jays
#: challenged (tag play), call on the field was overturned: ", "Umpire
#: reviewed (home run), call on the field was upheld: ", or the terser
#: "Cubs challenged (home-plate collision): ") so it never contaminates the
#: batter-name extraction below. See module comment for why this matters.
_REVIEW_PREAMBLE_RE = re.compile(
    r"^.*?(?:challenged?|reviewed) \([^)]*\)(?:, call on the field was (?:upheld|overturned))?: "
)
#: Real doubled-challenge plays exist (a play challenged on two separate
#: grounds); capped at 5 iterations purely as a runaway-loop guard, not a
#: real-data-derived limit.
_MAX_REVIEW_PREAMBLE_STRIPS = 5


def _strip_review_preamble(des: str) -> tuple[str, bool]:
    """Strip (possibly repeated) review/challenge preambles from `des`.

    Returns:
        `(stripped_des, reviewed_or_overturned)`.
    """
    stripped = False
    for _ in range(_MAX_REVIEW_PREAMBLE_STRIPS):
        m = _REVIEW_PREAMBLE_RE.match(des)
        if not m:
            break
        des = des[m.end() :]
        stripped = True
    return des, stripped


@dataclass(frozen=True)
class AdvancementParseResult:
    """Everything `parse_batter_advancement_des` determines for one `des` string."""

    label: str | None
    parse_status: str
    matched_clause: str | None
    failure_reason: str | None
    hit_type_implied_floor_base: int | None
    advancement_caused_by_error: bool
    is_true_inside_the_park_home_run: bool
    reviewed_or_overturned: bool


def parse_batter_advancement_des(des: object) -> AdvancementParseResult:
    """Parse one play's `des` text into a Version 0.9 batter-runner advancement label.

    See the "Version 0.9" module comment above for the full method and its
    real-data validation. NEVER force-labels an ambiguous or unrecognized
    play -- `label` is `None` unless `parse_status ==
    PARSE_STATUS_UNAMBIGUOUS`.

    Args:
        des: The play's free-text description (expected to be a non-empty
            `str`; anything else yields `PARSE_STATUS_NO_BATTER_MATCH`).

    Returns:
        An `AdvancementParseResult`. `matched_clause` and `failure_reason`
        are populated for auditability (task safeguard #2) regardless of
        outcome.
    """
    if not isinstance(des, str) or not des.strip():
        return AdvancementParseResult(
            None,
            PARSE_STATUS_NO_BATTER_MATCH,
            None,
            "empty_or_non_string_des",
            None,
            False,
            False,
            False,
        )

    try:
        des_for_matching, reviewed_or_overturned = _strip_review_preamble(des)

        m = None
        kind: str | None = None
        floor_base: int | None = None
        reach_on_error = False
        for compiled, k, fb, roe in _COMPILED_LEADING_PATTERNS:
            m = compiled.match(des_for_matching)
            if m:
                kind, floor_base, reach_on_error = k, fb, roe
                break
        if m is None or kind is None:
            return AdvancementParseResult(
                None,
                PARSE_STATUS_NO_BATTER_MATCH,
                None,
                "leading_clause_not_matched",
                None,
                False,
                False,
                reviewed_or_overturned,
            )

        name = m.group("name").strip()
        if not name:
            return AdvancementParseResult(
                None,
                PARSE_STATUS_NO_BATTER_MATCH,
                None,
                "empty_batter_name",
                None,
                False,
                False,
                reviewed_or_overturned,
            )

        if kind == _KIND_INSIDE_PARK_HR:
            return AdvancementParseResult(
                INSIDE_THE_PARK_HOME_RUN,
                PARSE_STATUS_UNAMBIGUOUS,
                m.group(0),
                None,
                None,
                False,
                True,
                reviewed_or_overturned,
            )

        if kind == _KIND_TRIVIAL_HR:
            return AdvancementParseResult(
                None,
                PARSE_STATUS_UNAMBIGUOUS,
                m.group(0),
                REASON_TRIVIAL_NO_ADVANCEMENT_OPPORTUNITY,
                None,
                False,
                False,
                reviewed_or_overturned,
            )

        if kind == _KIND_HIT_IMMEDIATE_ERROR_ADVANCE:
            base_match = re.search(r"advances to (2nd|3rd|home)", m.group(0))
            assert base_match is not None  # guaranteed by the pattern that matched
            floor_base = _BASE_WORD_TO_NUM[base_match.group(1)]
        assert floor_base is not None  # every remaining kind sets it

        # Scan the remainder of `des` for every clause mentioning this exact
        # batter name (kind in {_KIND_HIT, _KIND_HIT_IMMEDIATE_ERROR_ADVANCE}).
        name_re = re.escape(name)
        remainder = des_for_matching[m.end() :]

        retired_re = re.compile(
            rf"\b{name_re} (?:is |was )?(?:out|thrown out|tagged out|caught) at (1st|2nd|3rd|home)"
        )
        error_advance_re = re.compile(
            rf"\b{name_re} advances to (2nd|3rd|home), on an? [a-z]+(?: [a-z]+)? error"
        )
        simple_advance_re = re.compile(rf"\b{name_re} to (2nd|3rd)\b")
        scores_re = re.compile(rf"\b{name_re} scores\b")

        retired_match = retired_re.search(remainder)
        error_matches = list(error_advance_re.finditer(remainder))
        simple_matches = list(simple_advance_re.finditer(remainder))
        scores_match = scores_re.search(remainder)

        total_name_mentions = len(re.findall(rf"\b{name_re}\b", remainder))
        consumed_mentions = (
            (1 if retired_match else 0)
            + len(error_matches)
            + len(simple_matches)
            + (1 if scores_match else 0)
        )
        # Real 2021-2024 `des` text also reports an error as its OWN trailing
        # sentence ("... Randy Arozarena scores. Fielding error by third
        # baseman Jose Ramirez."), not always tied to a "<name> advances to
        # Nth, on error by" clause specifically -- this flag is informational
        # only (never a model input), so a play-level "an error happened
        # somewhere in this continuous play" signal is what matters, not
        # attributing it to one specific base-advance.
        standalone_error_clause = bool(re.search(r"\berror by\b", remainder))
        advancement_caused_by_error = (
            reach_on_error or bool(error_matches) or standalone_error_clause
        )

        if retired_match:
            if scores_match:
                return AdvancementParseResult(
                    None,
                    PARSE_STATUS_AMBIGUOUS,
                    remainder,
                    "retired_and_scores_both_matched",
                    floor_base,
                    advancement_caused_by_error,
                    False,
                    reviewed_or_overturned,
                )
            return AdvancementParseResult(
                RETIRED_WHILE_ADVANCING,
                PARSE_STATUS_UNAMBIGUOUS,
                retired_match.group(0),
                None,
                floor_base,
                advancement_caused_by_error,
                False,
                reviewed_or_overturned,
            )

        if consumed_mentions < total_name_mentions:
            return AdvancementParseResult(
                None,
                PARSE_STATUS_UNSUPPORTED_SEQUENCE,
                remainder,
                "unaccounted_batter_name_mention",
                floor_base,
                advancement_caused_by_error,
                False,
                reviewed_or_overturned,
            )

        max_base = floor_base
        matched_text: str | None = None
        if scores_match:
            max_base = 4
            matched_text = scores_match.group(0)
        for mm in error_matches + simple_matches:
            base = _BASE_WORD_TO_NUM.get(mm.group(1))
            if base is not None and base > max_base:
                max_base = base
                matched_text = mm.group(0)

        if max_base < floor_base:
            return AdvancementParseResult(
                None,
                PARSE_STATUS_AMBIGUOUS,
                remainder,
                "advanced_base_below_floor",
                floor_base,
                advancement_caused_by_error,
                False,
                reviewed_or_overturned,
            )

        return AdvancementParseResult(
            _NUM_TO_LABEL[max_base],
            PARSE_STATUS_UNAMBIGUOUS,
            matched_text or m.group(0),
            None,
            floor_base,
            advancement_caused_by_error,
            False,
            reviewed_or_overturned,
        )

    except Exception as exc:  # noqa: BLE001 - defensive catch-all, never crash a batch parse
        return AdvancementParseResult(
            None, PARSE_STATUS_PARSE_FAILURE, None, str(exc), None, False, False, False
        )


def add_advancement_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """Add Version 0.9 batter-runner advancement eligibility, label, and parse-audit columns.

    Args:
        df: A DataFrame with `bb_type`, `events`, and `des`.

    Returns:
        A copy of `df` with:
            - `advancement_eligible`: True only if `bb_type` is in
              `ADVANCEMENT_ELIGIBLE_BB_TYPES`, `events` is in
              `ADVANCEMENT_SAFE_EVENTS`, and `des` parses with
              `PARSE_STATUS_UNAMBIGUOUS`.
            - `advancement_exclusion_reason`: reason string if not eligible,
              else `None`.
            - `batter_final_base`: one of `ADVANCEMENT_LABELS` for eligible
              rows, else `None`. THIS IS THE TARGET LABEL -- never a model
              input (task safeguard #7).
            - `hit_type_implied_floor_base`: 1/2/3 (nullable `Int64`) for
              rows that reached the des-parsing stage with a real hit-type
              floor, else null.
            - `advancement_caused_by_error`, `is_true_inside_the_park_home_
              run`, `reviewed_or_overturned`: informational booleans (see
              module comment) -- never model inputs.
            - `advancement_parse_status`, `advancement_matched_clause`,
              `advancement_parse_failure_reason`, `advancement_parser_
              version`: full audit trail (task safeguard #2) for every row
              that reached the parsing stage, regardless of outcome.
    """
    required = ("bb_type", "events", "des")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"add_advancement_eligibility requires column(s) {missing}")

    out = df.copy()

    is_air_ball = out["bb_type"].isin(ADVANCEMENT_ELIGIBLE_BB_TYPES)
    is_safe_event = out["events"].isin(ADVANCEMENT_SAFE_EVENTS)
    parse_scope = is_air_ball & is_safe_event

    n = len(out)
    label = pd.Series([None] * n, index=out.index, dtype=object)
    floor_base = pd.Series(pd.NA, index=out.index, dtype="Int64")
    caused_by_error = pd.Series(False, index=out.index, dtype=bool)
    is_true_itp_hr = pd.Series(False, index=out.index, dtype=bool)
    reviewed_or_overturned = pd.Series(False, index=out.index, dtype=bool)
    parse_status = pd.Series([None] * n, index=out.index, dtype=object)
    matched_clause = pd.Series([None] * n, index=out.index, dtype=object)
    failure_reason = pd.Series([None] * n, index=out.index, dtype=object)
    parser_version = pd.Series([None] * n, index=out.index, dtype=object)

    scoped_idx = out.index[parse_scope]
    for idx in scoped_idx:
        result = parse_batter_advancement_des(out.at[idx, "des"])
        label.at[idx] = result.label
        floor_base.at[idx] = result.hit_type_implied_floor_base
        caused_by_error.at[idx] = result.advancement_caused_by_error
        is_true_itp_hr.at[idx] = result.is_true_inside_the_park_home_run
        reviewed_or_overturned.at[idx] = result.reviewed_or_overturned
        parse_status.at[idx] = result.parse_status
        matched_clause.at[idx] = result.matched_clause
        failure_reason.at[idx] = result.failure_reason
        parser_version.at[idx] = ADVANCEMENT_PARSER_VERSION

    advancement_eligible = parse_scope & (parse_status == PARSE_STATUS_UNAMBIGUOUS) & label.notna()

    reason = pd.Series([None] * n, index=out.index, dtype=object)
    reason = reason.where(is_air_ball, REASON_NOT_OUTFIELD_AIR_BALL)
    reason = reason.where(~(is_air_ball & ~is_safe_event), REASON_NOT_ADVANCEMENT_SAFE_EVENT)
    reason = reason.where(
        ~(parse_scope & (failure_reason == REASON_TRIVIAL_NO_ADVANCEMENT_OPPORTUNITY)),
        REASON_TRIVIAL_NO_ADVANCEMENT_OPPORTUNITY,
    )
    reason = reason.where(
        ~(
            parse_scope
            & (failure_reason != REASON_TRIVIAL_NO_ADVANCEMENT_OPPORTUNITY)
            & (parse_status != PARSE_STATUS_UNAMBIGUOUS)
        ),
        REASON_UNRESOLVED_DES_PARSE,
    )
    reason = reason.where(~advancement_eligible, None)

    out["advancement_eligible"] = advancement_eligible
    out["advancement_exclusion_reason"] = reason
    out["batter_final_base"] = label.where(advancement_eligible, None)
    out["hit_type_implied_floor_base"] = floor_base
    out["advancement_caused_by_error"] = caused_by_error
    out["is_true_inside_the_park_home_run"] = is_true_itp_hr
    out["reviewed_or_overturned"] = reviewed_or_overturned
    out["advancement_parse_status"] = parse_status
    out["advancement_matched_clause"] = matched_clause
    out["advancement_parse_failure_reason"] = failure_reason
    out["advancement_parser_version"] = parser_version

    return out


@dataclass(frozen=True)
class AdvancementExclusionCounts:
    """Row counts by `advancement_exclusion_reason` (plus "eligible") -- task safeguard #3's
    required "report their counts."
    """

    by_season: dict[int, dict[str, int]]
    total: dict[str, int]


def summarize_advancement_exclusions(df: pd.DataFrame) -> AdvancementExclusionCounts:
    """Row counts by `advancement_exclusion_reason`, per season and overall.

    Args:
        df: A DataFrame already passed through `add_advancement_eligibility`,
            with a `season` column.

    Returns:
        `AdvancementExclusionCounts` -- `by_season[season][reason_or_
        "eligible"] = count`, and the same totals across all seasons in
        `.total`.
    """
    working = df.copy()
    label = working["advancement_exclusion_reason"].astype(object)
    label = label.where(~working["advancement_eligible"].astype(bool), "eligible")
    working["_label"] = label

    total: dict[str, int] = {
        str(k): int(v) for k, v in working["_label"].value_counts(dropna=False).items()
    }
    by_season: dict[int, dict[str, int]] = {}
    if "season" in working.columns:
        for season_value in working["season"].dropna().unique():
            group = working[working["season"] == season_value]
            by_season[int(season_value)] = {
                str(k): int(v) for k, v in group["_label"].value_counts(dropna=False).items()
            }

    return AdvancementExclusionCounts(by_season=by_season, total=total)


def summarize_advancement_parse_coverage(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Parse-status counts broken out by `batter_final_base` label (task safeguard #4).

    Reports coverage/status separately for each of `ADVANCEMENT_LABELS` (the
    successfully-labeled rows) plus one `"reached_on_error"` breakdown (rows
    with `advancement_caused_by_error` True, cutting across labels) and one
    `"multi_clause_or_multi_throw"` breakdown (rows whose full `des` mentions
    at least THREE distinct fielding positions in a "to <position>" chain --
    the initial fielder plus at least two subsequent relay throws, e.g. "...
    left fielder Dominic Smith to catcher Patrick Mazeika to second baseman
    Jose Peraza." A `>=2` threshold would just mean "there was one throw
    after the initial fielding," true of nearly every retired-while
    -advancing play; `>=3` isolates genuinely multi-throw relay sequences --
    verified against real 2021-2024 data: 643 of 107,721 parsed-scope rows).

    Args:
        df: A DataFrame already passed through `add_advancement_eligibility`.

    Returns:
        `{group_label: {parse_status_or_reason: count}}`. Groups for
        successful labels are counted among `advancement_eligible` rows
        only (all `PARSE_STATUS_UNAMBIGUOUS` by construction); the reported
        `parse_status` counts for the FULL parsed-scope population (task
        safeguard #4's "coverage") are also included under the
        `"__all_parsed_scope__"` key.
    """
    working = df[
        df["bb_type"].isin(ADVANCEMENT_ELIGIBLE_BB_TYPES)
        & df["events"].isin(ADVANCEMENT_SAFE_EVENTS)
    ].copy()

    report: dict[str, dict[str, int]] = {
        "__all_parsed_scope__": {
            str(k): int(v)
            for k, v in working["advancement_parse_status"].value_counts(dropna=False).items()
        }
    }
    for label_value in ADVANCEMENT_LABELS:
        subset = working[working["batter_final_base"] == label_value]
        report[label_value] = {"n_rows": int(len(subset))}

    reached_on_error = working[working["advancement_caused_by_error"].astype(bool)]
    report["reached_on_error"] = {"n_rows": int(len(reached_on_error))}

    relay_throw_count = working["des"].astype(str).str.count(rf"\bto {_FIELDING_POSITION_WORDS}\b")
    multi_throw = working[relay_throw_count >= 3]
    report["multi_clause_or_multi_throw"] = {"n_rows": int(len(multi_throw))}

    return report
