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
