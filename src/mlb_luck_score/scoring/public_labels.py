"""Contact Luck v0.12 Phase 6: centralized public language.

Every public-facing string this project uses to describe the Contact Luck
score lives HERE and only here -- no other module should hardcode a
definition, a positive/negative-value explanation, or the retrospective
-limitation sentence. Centralizing this is what makes it possible to audit
public language for banned phrasing in one place (`check_text_for_banned_
phrases`) rather than trusting every call site to remember the rules.

This module defines language only -- it computes nothing and touches no
model, ledger, or aggregation logic.
"""

from __future__ import annotations

import re

CONTACT_LUCK_RUNS_LABEL = "Contact Luck Runs"
CONTACT_LUCK_RUNS_DEFINITION = (
    "The number of runs by which a batter's observed eligible batted-ball outcomes were "
    "more or less favorable than the model expected."
)

CONTACT_LUCK_RUNS_PER_100_LABEL = "Contact Luck Runs per 100"
CONTACT_LUCK_RUNS_PER_100_DEFINITION = "Contact Luck Runs normalized to 100 eligible batted balls."

POSITIVE_VALUE_LABEL = "Positive value"
POSITIVE_VALUE_DEFINITION = "More favorable realized outcomes than expected."

NEGATIVE_VALUE_LABEL = "Negative value"
NEGATIVE_VALUE_DEFINITION = "Less favorable realized outcomes than expected."

RETROSPECTIVE_LIMITATION = (
    "Contact Luck is a retrospective description of realized outcomes, not a measure of "
    "stable batting talent and not a projection of future performance."
)

#: Phase 7's leaderboard produces two directions -- the task explicitly asks
#: that the unfavorable one NOT be called "worst players."
MOST_FAVORABLE_LEADERBOARD_LABEL = "Most favorable realized luck"
LEAST_FAVORABLE_LEADERBOARD_LABEL = "Least favorable outcomes relative to expectation"

#: Interval-interpretation labels are descriptive only -- see `mlb_luck_score.
#: scoring.public_score_schema.classify_interval`. Never phrase these as a
#: significance claim (see BANNED_PHRASES).
INTERVAL_INTERPRETATION_LABELS: dict[str, str] = {
    "entirely_above_zero": "Interval entirely above zero",
    "overlaps_zero": "Interval overlaps zero",
    "entirely_below_zero": "Interval entirely below zero",
}

#: A component's status label must always accompany any displayed value --
#: see Phase 5. These reuse mlb_luck_score.scoring.component_confidence's own
#: MODEL_STATUS_* vocabulary verbatim as keys.
COMPONENT_STATUS_DISPLAY_LABELS: dict[str, str] = {
    "calibrated": "Calibrated",
    "calibrated_with_limited_subgroup_evidence": "Calibrated (limited subgroup evidence)",
    "provisional": "Provisional -- not yet fully validated",
    "not_calibrated": "Not calibrated",
    "unavailable": "Unavailable for this play",
}

#: Phrases this project's public-facing text must never use. Each entry
#: documents WHY. `defense-independent` is conditionally allowed (the task's
#: own wording: "unless technically justified for the specific field") --
#: `check_text_for_banned_phrases` still flags it so a reviewer consciously
#: decides that exception case by case, rather than it slipping through
#: silently; it is not auto-excluded from the scan.
BANNED_PHRASES: tuple[str, ...] = (
    "deserved hits",
    "true talent luck",
    "guaranteed regression",
    "should have produced",
    "defense-independent",
    "statistically significant player",
)

_BANNED_PHRASE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (phrase, re.compile(re.escape(phrase), re.IGNORECASE)) for phrase in BANNED_PHRASES
)


def check_text_for_banned_phrases(text: str) -> list[str]:
    """Return every `BANNED_PHRASES` entry found in `text` (case-insensitive),
    in `BANNED_PHRASES` order. Empty list means the text is clean.
    """
    return [phrase for phrase, pattern in _BANNED_PHRASE_PATTERNS if pattern.search(text)]


def assert_no_banned_phrases(text: str, *, context: str = "") -> None:
    """Raise `ValueError` if `text` contains any `BANNED_PHRASES` entry."""
    found = check_text_for_banned_phrases(text)
    if found:
        location = f" in {context}" if context else ""
        raise ValueError(f"Banned phrase(s) found{location}: {found}")
