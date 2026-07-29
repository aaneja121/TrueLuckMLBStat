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
