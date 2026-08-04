from __future__ import annotations

import pandas as pd

from mlb_luck_score.eligibility import (
    ELIGIBLE_EVENTS_V0_1,
    POSITION_SOURCE_HIT_LOCATION,
    POSITION_SOURCE_SPRAY_SECTOR_PROXY,
    REASON_AMBIGUOUS_FIELD_ERROR,
    REASON_AMBIGUOUS_FIELDERS_CHOICE,
    REASON_BASE_TRAINING_INELIGIBLE,
    REASON_BUNT_EXCLUDED,
    REASON_EXCLUDED_STRATEGIC_PLAY,
    REASON_INFIELD_CREDITED_HIT_LOCATION,
    REASON_INTERFERENCE_OR_OBSTRUCTION,
    REASON_MISSING_CONTACT_DATA,
    REASON_MISSING_INFIELD_CONTACT_DATA,
    REASON_MISSING_INFIELD_POSITION,
    REASON_NOT_ELIGIBLE_EVENT,
    REASON_NOT_GROUND_BALL,
    REASON_NOT_OUTFIELD_AIR_BALL,
    REASON_OUTFIELD_CREDITED_HIT_LOCATION,
    add_infield_opportunity_eligibility,
    add_outfield_opportunity_eligibility,
    compute_eligibility,
    map_outcome_class,
    summarize_infield_exclusions,
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


# ---------------------------------------------------------------------------
# Version 0.8: infield-opportunity eligibility
# ---------------------------------------------------------------------------


def _infield_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021, 2021],
            "bb_type": [
                "ground_ball",  # 0: clean field_out
                "ground_ball",  # 1: clean single
                "ground_ball",  # 2: reached on error
                "ground_ball",  # 3: force out -- excluded strategic
                "ground_ball",  # 4: bunt single -- excluded (des keyword, not events)
                "ground_ball",  # 5: batter interference recorded as field_out -- excluded
                "fly_ball",  # 6: not a ground ball at all
                "ground_ball",  # 7: outfield-credited hit_location -- excluded
                "ground_ball",  # 8: missing hit_location -- excluded
                "ground_ball",  # 9: missing launch_speed -- excluded
                "ground_ball",  # 10: reviewed/overturned but otherwise clean -- STILL eligible
                "ground_ball",  # 11: double_play -- excluded strategic
            ],
            "events": [
                "field_out",
                "single",
                "field_error",
                "force_out",
                "single",
                "field_out",
                "field_out",
                "field_out",
                "field_out",
                "field_out",
                "single",
                "double_play",
            ],
            "des": [
                "Routine grounder to short.",
                "Line drive single up the middle.",
                "Reaches on a throwing error by the shortstop.",
                "Grounds into a force out, second baseman to shortstop.",
                "Singles on a bunt ground ball to third baseman.",
                "Grounds out, catcher to first baseman. Batter out on batter interference.",
                None,
                "Routine grounder that gets through.",
                "Routine grounder.",
                "Routine grounder.",
                "Challenged (play at 1st), call on the field was overturned: singles.",
                "Grounds into a double play, second baseman to shortstop to first baseman.",
            ],
            "hit_location": [6, 4, 6, 4, 5, 2, 8, 8, None, 6, 6, 4],
            "launch_speed": [
                85.0,
                90.0,
                80.0,
                88.0,
                40.0,
                70.0,
                95.0,
                85.0,
                82.0,
                None,
                91.0,
                87.0,
            ],
            "launch_angle": [-5.0, 2.0, 1.0, -3.0, -1.0, 0.0, 25.0, -4.0, -2.0, 5.0, -6.0, -2.0],
            "pitcher": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "fielder_2": [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212],
            "fielder_3": [301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312],
            "fielder_4": [401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412],
            "fielder_5": [501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512],
            "fielder_6": [601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612],
        }
    )


def test_infield_clean_field_out_is_eligible_and_retired():
    out = add_infield_opportunity_eligibility(_infield_df())
    row = out.iloc[0]
    assert bool(row["infield_opportunity_eligible"])
    assert row["infield_opportunity_exclusion_reason"] is None
    assert row["y_out"] == 1
    assert not bool(row["reached_on_error"])
    assert row["assigned_infield_position"] == 6
    assert row["responsible_infielder_id"] == 601  # fielder_6 for this row


def test_infield_clean_single_is_eligible_and_safe():
    out = add_infield_opportunity_eligibility(_infield_df())
    row = out.iloc[1]
    assert bool(row["infield_opportunity_eligible"])
    assert row["y_out"] == 0
    assert not bool(row["reached_on_error"])
    assert row["assigned_infield_position"] == 4


def test_infield_field_error_is_eligible_safe_and_reached_on_error():
    out = add_infield_opportunity_eligibility(_infield_df())
    row = out.iloc[2]
    assert bool(row["infield_opportunity_eligible"])
    assert row["y_out"] == 0
    assert bool(row["reached_on_error"])


def test_infield_force_out_excluded_as_strategic():
    out = add_infield_opportunity_eligibility(_infield_df())
    row = out.iloc[3]
    assert not bool(row["infield_opportunity_eligible"])
    assert row["infield_opportunity_exclusion_reason"] == REASON_EXCLUDED_STRATEGIC_PLAY
    assert pd.isna(row["y_out"])


def test_infield_bunt_single_excluded_via_des_keyword_not_events():
    out = add_infield_opportunity_eligibility(_infield_df())
    row = out.iloc[4]
    assert not bool(row["infield_opportunity_eligible"])
    assert row["infield_opportunity_exclusion_reason"] == REASON_BUNT_EXCLUDED
    assert bool(row["is_bunt"])


def test_infield_batter_interference_excluded_despite_field_out_event():
    """A real gap the task's audit flagged: 'batter interference' is recorded
    as an ordinary field_out in Statcast's `events` -- only `des` reveals it.
    """
    out = add_infield_opportunity_eligibility(_infield_df())
    row = out.iloc[5]
    assert not bool(row["infield_opportunity_eligible"])
    assert row["infield_opportunity_exclusion_reason"] == REASON_INTERFERENCE_OR_OBSTRUCTION
    assert bool(row["has_interference_or_obstruction"])


def test_infield_non_ground_ball_excluded():
    out = add_infield_opportunity_eligibility(_infield_df())
    row = out.iloc[6]
    assert not bool(row["infield_opportunity_eligible"])
    assert row["infield_opportunity_exclusion_reason"] == REASON_NOT_GROUND_BALL


def test_infield_outfield_credited_hit_location_excluded():
    out = add_infield_opportunity_eligibility(_infield_df())
    row = out.iloc[7]
    assert not bool(row["infield_opportunity_eligible"])
    assert row["infield_opportunity_exclusion_reason"] == REASON_OUTFIELD_CREDITED_HIT_LOCATION


def test_infield_missing_hit_location_excluded():
    out = add_infield_opportunity_eligibility(_infield_df())
    row = out.iloc[8]
    assert not bool(row["infield_opportunity_eligible"])
    assert row["infield_opportunity_exclusion_reason"] == REASON_MISSING_INFIELD_POSITION


def test_infield_missing_contact_data_excluded():
    out = add_infield_opportunity_eligibility(_infield_df())
    row = out.iloc[9]
    assert not bool(row["infield_opportunity_eligible"])
    assert row["infield_opportunity_exclusion_reason"] == REASON_MISSING_INFIELD_CONTACT_DATA


def test_infield_reviewed_play_stays_eligible_but_is_flagged_informationally():
    out = add_infield_opportunity_eligibility(_infield_df())
    row = out.iloc[10]
    assert bool(row["infield_opportunity_eligible"])
    assert row["infield_opportunity_exclusion_reason"] is None
    assert bool(row["reviewed_or_overturned"])


def test_infield_double_play_excluded_as_strategic():
    out = add_infield_opportunity_eligibility(_infield_df())
    row = out.iloc[11]
    assert not bool(row["infield_opportunity_eligible"])
    assert row["infield_opportunity_exclusion_reason"] == REASON_EXCLUDED_STRATEGIC_PLAY


def test_infield_responsible_fielder_uses_pitcher_column_for_position_1():
    df = pd.DataFrame(
        {
            "bb_type": ["ground_ball"],
            "events": ["field_out"],
            "des": ["Comebacker to the mound."],
            "hit_location": [1],
            "launch_speed": [70.0],
            "launch_angle": [-2.0],
            "pitcher": [999],
        }
    )
    out = add_infield_opportunity_eligibility(df)
    assert out.iloc[0]["responsible_infielder_id"] == 999


def test_add_infield_opportunity_eligibility_requires_columns():
    import pytest

    with pytest.raises(ValueError, match="events"):
        add_infield_opportunity_eligibility(pd.DataFrame({"bb_type": ["ground_ball"]}))


def test_summarize_infield_exclusions_counts_by_season_and_total():
    out = add_infield_opportunity_eligibility(_infield_df())
    report = summarize_infield_exclusions(out)
    assert report.total["eligible"] == 4  # rows 0, 1, 2, 10
    assert report.total[REASON_EXCLUDED_STRATEGIC_PLAY] == 2  # rows 3, 11
    assert 2021 in report.by_season
    assert report.by_season[2021] == report.total
