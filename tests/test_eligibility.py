from __future__ import annotations

import pandas as pd

from mlb_luck_score.eligibility import (
    ELIGIBLE_EVENTS_V0_1,
    REASON_AMBIGUOUS_FIELD_ERROR,
    REASON_AMBIGUOUS_FIELDERS_CHOICE,
    REASON_MISSING_CONTACT_DATA,
    REASON_NOT_ELIGIBLE_EVENT,
    compute_eligibility,
    map_outcome_class,
)


def test_direct_hit_events_map_to_matching_class():
    for events in ("single", "double", "triple", "home_run"):
        mapping = map_outcome_class(events)
        assert mapping.outcome_class == events
        assert mapping.eligible_for_training is True
        assert mapping.exclusion_reason is None


def test_unambiguous_out_events_map_to_out():
    for events in (
        "field_out",
        "force_out",
        "grounded_into_double_play",
        "double_play",
        "fielders_choice_out",
        "sac_bunt",
        "sac_fly",
        "sac_fly_double_play",
    ):
        mapping = map_outcome_class(events)
        assert mapping.outcome_class == "out"
        assert mapping.eligible_for_training is True


def test_ambiguous_events_do_not_invent_a_label():
    fc = map_outcome_class("fielders_choice")
    assert fc.outcome_class is None
    assert fc.eligible_for_training is False
    assert fc.exclusion_reason == REASON_AMBIGUOUS_FIELDERS_CHOICE

    err = map_outcome_class("field_error")
    assert err.outcome_class is None
    assert err.eligible_for_training is False
    assert err.exclusion_reason == REASON_AMBIGUOUS_FIELD_ERROR


def test_excluded_events_are_not_eligible():
    for events in ("strikeout", "walk", "hit_by_pitch", "catcher_interf", None):
        mapping = map_outcome_class(events)
        assert mapping.outcome_class is None
        assert mapping.eligible_for_training is False
        assert mapping.exclusion_reason == REASON_NOT_ELIGIBLE_EVENT


def test_eligible_events_v0_1_excludes_strikeouts_walks_and_friends():
    excluded = {
        "strikeout",
        "walk",
        "hit_by_pitch",
        "catcher_interf",
        "caught_stealing_2b",
        "pickoff_1b",
        "balk",
        "wild_pitch",
        "passed_ball",
    }
    assert excluded.isdisjoint(ELIGIBLE_EVENTS_V0_1)


def test_compute_eligibility_flags_missing_contact_data():
    df = pd.DataFrame(
        {
            "events": ["single", "single"],
            "launch_speed": [95.0, None],
            "launch_angle": [10.0, None],
        }
    )
    out = compute_eligibility(df)
    assert out["missing_contact_data"].tolist() == [False, True]
    assert not bool(out.loc[1, "eligible_for_training"])
    assert out.loc[1, "training_exclusion_reason"] == REASON_MISSING_CONTACT_DATA


def test_compute_eligibility_requires_events_column():
    import pytest

    with pytest.raises(ValueError, match="events"):
        compute_eligibility(pd.DataFrame({"launch_speed": [1.0]}))


def test_compute_eligibility_is_eligible_matches_event_set():
    df = pd.DataFrame({"events": list(ELIGIBLE_EVENTS_V0_1) + ["strikeout", "walk"]})
    out = compute_eligibility(df)
    assert out["is_eligible"].tolist() == [True] * len(ELIGIBLE_EVENTS_V0_1) + [False, False]
