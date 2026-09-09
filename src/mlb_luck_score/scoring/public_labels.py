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

#: The same limitation, said for the PITCHER surface. `RETROSPECTIVE_
#: LIMITATION` above names batting talent, which is the correct caveat on a
#: hitter page and simply the wrong noun on a pitcher one -- so the pitcher
#: routes get their own sentence rather than a reworded shared one that
#: would be vague on both. The two are deliberately separate constants: a
#: surface picks the one that describes it, and neither is a default for
#: the other. The claim is identical in force -- retrospective, not talent,
#: not a forecast -- and the sign convention it describes is the frozen one
#: (positive = the realized outcome was more favorable to the pitcher than
#: the contact predicted).
PITCHER_RETROSPECTIVE_LIMITATION = (
    "Pitcher Contact Luck is retrospective. It describes how favorable or unfavorable the "
    "outcomes on a pitcher's contact were relative to what that contact predicted. It is "
    "not a measure of stable pitching talent or of future performance."
)

#: The pitcher surface's PROVENANCE line, replacing the development-prototype
#: banner it carried while the route was local-only. It says three things a
#: reader of a published historical board is owed and cannot infer: which
#: season these numbers are, that they come from the same frozen model the
#: hitter leaderboard uses, and that a season total is a record rather than
#: a forecast. Worded to be true on BOTH pitcher surfaces -- it sits on the
#: board and on a single pitcher's card, so it says nothing in the plural. `{season}` is filled with the season the surface publishes --
#: never hardcoded in a template.
#:
#: What it deliberately does NOT say: that the season is "current", that the
#: numbers are updated, or anything comparative about the pitcher. The
#: retrospective caveat proper is `PITCHER_RETROSPECTIVE_LIMITATION`; this
#: is provenance, not framing, and the two are separate so neither has to
#: carry the other's job.
PITCHER_SEASON_PROVENANCE = (
    "{season} season, measured on the same frozen scoring model as the Contact Luck "
    "hitter leaderboard. It is a record of what happened, not a projection."
)

#: The EXPOSURE disclosure the pitcher surface owes its reader.
#:
#: Contact Luck's `pitcher_primary` threshold set (>=450 eligible BBE)
#: describes starting pitchers: no reliever-season in 2021-2024 reached it,
#: the highest with >=50 appearances being 311 BBE. `CONTEXT.md` records
#: that any surface showing this metric must say so, and the reason is that
#: the omission reads as a judgement when it is arithmetic -- a reliever
#: does not face enough batted balls, which is a fact about usage and not
#: about the pitcher.
#:
#: The boards themselves rank on a 60-BBE DISPLAY minimum and never use the
#: threshold set at all, which is precisely why this has to be said out
#: loud: a reader who has met Contact Luck on the hitter side arrives
#: carrying its qualification rules, and nothing on the page would
#: otherwise tell them those rules are not the ones in force here.
PITCHER_EXPOSURE_SCOPE = (
    "Contact Luck's 450 batted-ball threshold set describes starting pitchers: no "
    "reliever season from 2021 to 2024 reached it, the highest with at least 50 "
    "appearances being 311. Relievers are absent from that threshold by exposure, not by "
    "choice -- they do not face enough batted balls to clear it. The boards here do not "
    "use that threshold; they rank on the batted balls each pitcher actually allowed."
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
