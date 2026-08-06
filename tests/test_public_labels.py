"""Tests for the Contact Luck v0.12 centralized public language (Phase 6/9).

No test touches the network.
"""

from __future__ import annotations

import pytest

from mlb_luck_score.scoring.public_labels import (
    BANNED_PHRASES,
    CONTACT_LUCK_RUNS_DEFINITION,
    CONTACT_LUCK_RUNS_LABEL,
    CONTACT_LUCK_RUNS_PER_100_DEFINITION,
    CONTACT_LUCK_RUNS_PER_100_LABEL,
    LEAST_FAVORABLE_LEADERBOARD_LABEL,
    MOST_FAVORABLE_LEADERBOARD_LABEL,
    NEGATIVE_VALUE_DEFINITION,
    POSITIVE_VALUE_DEFINITION,
    RETROSPECTIVE_LIMITATION,
    assert_no_banned_phrases,
    check_text_for_banned_phrases,
)


def test_required_language_constants_match_task_wording_exactly():
    assert CONTACT_LUCK_RUNS_LABEL == "Contact Luck Runs"
    assert CONTACT_LUCK_RUNS_DEFINITION == (
        "The number of runs by which a batter's observed eligible batted-ball outcomes were "
        "more or less favorable than the model expected."
    )
    assert CONTACT_LUCK_RUNS_PER_100_LABEL == "Contact Luck Runs per 100"
    assert (
        CONTACT_LUCK_RUNS_PER_100_DEFINITION
        == "Contact Luck Runs normalized to 100 eligible batted balls."
    )
    assert POSITIVE_VALUE_DEFINITION == "More favorable realized outcomes than expected."
    assert NEGATIVE_VALUE_DEFINITION == "Less favorable realized outcomes than expected."


def test_retrospective_limitation_is_present_and_worded_correctly():
    assert RETROSPECTIVE_LIMITATION == (
        "Contact Luck is a retrospective description of realized outcomes, not a measure of "
        "stable batting talent and not a projection of future performance."
    )
    assert "retrospective" in RETROSPECTIVE_LIMITATION.lower()
    assert "projection" in RETROSPECTIVE_LIMITATION.lower()


def test_least_favorable_label_never_says_worst_players():
    assert "worst" not in LEAST_FAVORABLE_LEADERBOARD_LABEL.lower()
    assert "least favorable" in LEAST_FAVORABLE_LEADERBOARD_LABEL.lower()
    assert "worst" not in MOST_FAVORABLE_LEADERBOARD_LABEL.lower()


@pytest.mark.parametrize(
    "phrase",
    [
        "deserved hits",
        "true talent luck",
        "guaranteed regression",
        "should have produced",
        "defense-independent",
        "statistically significant player",
    ],
)
def test_banned_phrase_list_contains_every_required_entry(phrase):
    assert phrase in BANNED_PHRASES


def test_check_text_for_banned_phrases_is_case_insensitive():
    text = "This batter clearly DESERVED HITS this season."
    found = check_text_for_banned_phrases(text)
    assert "deserved hits" in found


def test_check_text_for_banned_phrases_returns_empty_for_clean_text():
    text = "Contact Luck Runs per 100 describes realized outcomes, not future performance."
    assert check_text_for_banned_phrases(text) == []


def test_assert_no_banned_phrases_raises_on_violation():
    with pytest.raises(ValueError, match="guaranteed regression"):
        assert_no_banned_phrases("Expect guaranteed regression next year.", context="test caption")


def test_assert_no_banned_phrases_does_not_raise_on_clean_text():
    assert_no_banned_phrases(RETROSPECTIVE_LIMITATION)  # must not raise


def test_required_public_strings_are_themselves_free_of_banned_phrases():
    for text in (
        CONTACT_LUCK_RUNS_DEFINITION,
        CONTACT_LUCK_RUNS_PER_100_DEFINITION,
        POSITIVE_VALUE_DEFINITION,
        NEGATIVE_VALUE_DEFINITION,
        RETROSPECTIVE_LIMITATION,
        MOST_FAVORABLE_LEADERBOARD_LABEL,
        LEAST_FAVORABLE_LEADERBOARD_LABEL,
    ):
        assert check_text_for_banned_phrases(text) == []
