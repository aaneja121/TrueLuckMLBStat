from __future__ import annotations

import pandas as pd

from mlb_luck_score.eligibility import (
    ELIGIBLE_EVENTS_V0_1,
    POSITION_SOURCE_HIT_LOCATION,
    POSITION_SOURCE_SPRAY_SECTOR_PROXY,
    REASON_AMBIGUOUS_FIELD_ERROR,
    REASON_AMBIGUOUS_FIELDERS_CHOICE,
    REASON_BASE_TRAINING_INELIGIBLE,
    REASON_INFIELD_CREDITED_HIT_LOCATION,
    REASON_MISSING_CONTACT_DATA,
    REASON_NOT_ELIGIBLE_EVENT,
    REASON_NOT_OUTFIELD_AIR_BALL,
    add_outfield_opportunity_eligibility,
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


# ---------------------------------------------------------------------------
# Version 0.7A: outfield-opportunity eligibility
# ---------------------------------------------------------------------------


def _opportunity_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bb_type": ["fly_ball", "line_drive", "ground_ball", "fly_ball", "fly_ball", "popup"],
            "eligible_for_training": [True, True, True, True, False, True],
            "hit_location": [8, 3, 6, None, 8, 2],
            "spray_sector": ["center", "center", "center", "right", "center", "center"],
            "fielder_7": [111, 111, 111, 111, 111, 111],
            "fielder_8": [222, 222, 222, 222, 222, 222],
            "fielder_9": [333, 333, 333, 333, 333, 333],
        }
    )


def test_outfield_hit_location_makes_row_eligible_with_real_position():
    out = add_outfield_opportunity_eligibility(_opportunity_df())
    row = out.iloc[0]  # fly_ball, hit_location=8 (CF), eligible_for_training=True
    assert row["outfield_opportunity_eligible"] is True or bool(
        row["outfield_opportunity_eligible"]
    )
    assert row["outfield_opportunity_exclusion_reason"] is None
    assert row["assigned_outfield_position"] == 8
    assert row["assigned_outfield_position_source"] == POSITION_SOURCE_HIT_LOCATION
    assert row["responsible_outfielder_id"] == 222


def test_infield_credited_hit_location_excluded():
    out = add_outfield_opportunity_eligibility(_opportunity_df())
    row = out.iloc[1]  # line_drive, hit_location=3 (1B)
    assert not bool(row["outfield_opportunity_eligible"])
    assert row["outfield_opportunity_exclusion_reason"] == REASON_INFIELD_CREDITED_HIT_LOCATION
    assert pd.isna(row["assigned_outfield_position"])


def test_ground_ball_excluded_as_not_outfield_air_ball():
    out = add_outfield_opportunity_eligibility(_opportunity_df())
    row = out.iloc[2]
    assert not bool(row["outfield_opportunity_eligible"])
    assert row["outfield_opportunity_exclusion_reason"] == REASON_NOT_OUTFIELD_AIR_BALL


def test_popup_excluded_as_not_outfield_air_ball():
    out = add_outfield_opportunity_eligibility(_opportunity_df())
    row = out.iloc[5]
    assert not bool(row["outfield_opportunity_eligible"])
    assert row["outfield_opportunity_exclusion_reason"] == REASON_NOT_OUTFIELD_AIR_BALL


def test_missing_hit_location_falls_back_to_spray_sector_proxy():
    out = add_outfield_opportunity_eligibility(_opportunity_df())
    row = out.iloc[3]  # fly_ball, hit_location missing, spray_sector="right"
    assert bool(row["outfield_opportunity_eligible"])
    assert row["assigned_outfield_position"] == 9  # right -> RF
    assert row["assigned_outfield_position_source"] == POSITION_SOURCE_SPRAY_SECTOR_PROXY
    assert row["responsible_outfielder_id"] == 333


def test_base_training_ineligible_row_excluded():
    out = add_outfield_opportunity_eligibility(_opportunity_df())
    row = out.iloc[4]  # eligible_for_training=False
    assert not bool(row["outfield_opportunity_eligible"])
    assert row["outfield_opportunity_exclusion_reason"] == REASON_BASE_TRAINING_INELIGIBLE


def test_add_outfield_opportunity_eligibility_requires_columns():
    import pytest

    with pytest.raises(ValueError, match="bb_type"):
        add_outfield_opportunity_eligibility(pd.DataFrame({"eligible_for_training": [True]}))
    with pytest.raises(ValueError, match="eligible_for_training"):
        add_outfield_opportunity_eligibility(pd.DataFrame({"bb_type": ["fly_ball"]}))


def test_missing_fielder_columns_degrade_gracefully():
    df = pd.DataFrame(
        {
            "bb_type": ["fly_ball"],
            "eligible_for_training": [True],
            "hit_location": [8],
        }
    )
    out = add_outfield_opportunity_eligibility(df)
    assert bool(out.iloc[0]["outfield_opportunity_eligible"])
    assert pd.isna(out.iloc[0]["responsible_outfielder_id"])
