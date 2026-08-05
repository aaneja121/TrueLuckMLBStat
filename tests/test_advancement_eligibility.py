"""Tests for the Contact Luck v0.9 batter-runner advancement labeling.

No test touches the network -- synthetic `des` strings only, mirroring real
phrasing patterns observed in the (offline, already-downloaded) 2021-2024
dataset. The target label here is reconstructed from free text, so this
file specifically exercises the parser's edge cases per the task's explicit
safeguards: name collisions, substitutions, punctuation variation, errors,
trailing advancement clauses, multiple runners, and retired-while-advancing
language.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlb_luck_score.eligibility import (
    ADVANCED_TO_SECOND,
    ADVANCED_TO_THIRD,
    ADVANCEMENT_PARSER_VERSION,
    HELD_AT_FIRST,
    INSIDE_THE_PARK_HOME_RUN,
    PARSE_STATUS_AMBIGUOUS,
    PARSE_STATUS_NO_BATTER_MATCH,
    PARSE_STATUS_UNAMBIGUOUS,
    PARSE_STATUS_UNSUPPORTED_SEQUENCE,
    REASON_NOT_ADVANCEMENT_SAFE_EVENT,
    REASON_NOT_OUTFIELD_AIR_BALL,
    REASON_TRIVIAL_NO_ADVANCEMENT_OPPORTUNITY,
    REASON_UNRESOLVED_DES_PARSE,
    RETIRED_WHILE_ADVANCING,
    add_advancement_eligibility,
    parse_batter_advancement_des,
    summarize_advancement_exclusions,
    summarize_advancement_parse_coverage,
)

# ---------------------------------------------------------------------------
# parse_batter_advancement_des: direct unit tests
# ---------------------------------------------------------------------------


def test_single_with_no_further_mention_is_held_at_first():
    r = parse_batter_advancement_des(
        "Trevor Story singles on a sharp line drive to center fielder Leody Taveras."
    )
    assert r.label == HELD_AT_FIRST
    assert r.parse_status == PARSE_STATUS_UNAMBIGUOUS
    assert r.hit_type_implied_floor_base == 1
    assert not r.advancement_caused_by_error


def test_ordinary_double_with_no_further_mention_is_advanced_to_second():
    r = parse_batter_advancement_des(
        "Taylor Walls doubles (4) on a fly ball to left fielder Chas McCormick."
    )
    assert r.label == ADVANCED_TO_SECOND
    assert r.hit_type_implied_floor_base == 2


def test_ordinary_triple_with_no_further_mention_is_advanced_to_third():
    r = parse_batter_advancement_des(
        "Pete Alonso triples (1) on a fly ball to right fielder Kris Bryant."
    )
    assert r.label == ADVANCED_TO_THIRD
    assert r.hit_type_implied_floor_base == 3


def test_ground_rule_double_is_advanced_to_second():
    r = parse_batter_advancement_des(
        "Starling Marte hits a ground-rule double (5) on a line drive to right field."
    )
    assert r.label == ADVANCED_TO_SECOND


# -- trailing advancement clauses --------------------------------------------


def test_single_plus_extra_base_is_advanced_to_second():
    r = parse_batter_advancement_des(
        "Zach McKinstry singles on a ground ball to right fielder Juan Soto. "
        "Max Muncy scores. AJ Pollock scores. Gavin Lux to 3rd. Zach McKinstry to 2nd."
    )
    assert r.label == ADVANCED_TO_SECOND
    assert "Zach McKinstry to 2nd" in r.matched_clause


def test_double_plus_extra_base_is_advanced_to_third():
    r = parse_batter_advancement_des(
        "Lourdes Gurriel Jr. doubles (1) on a line drive to left fielder Brad Miller. "
        "Teoscar Hernandez scores. Lourdes Gurriel Jr. to 3rd."
    )
    assert r.label == ADVANCED_TO_THIRD


def test_batter_scores_from_a_single_is_inside_the_park_home_run():
    r = parse_batter_advancement_des(
        "Randy Arozarena singles on a ground ball to center fielder Oscar Mercado. "
        "Yandy Diaz scores. Wander Franco scores. Randy Arozarena scores. "
        "Fielding error by third baseman Jose Ramirez. Throwing error by third baseman Jose Ramirez."
    )
    assert r.label == INSIDE_THE_PARK_HOME_RUN
    assert r.advancement_caused_by_error
    assert not r.is_true_inside_the_park_home_run


# -- errors -------------------------------------------------------------------


def test_reaches_on_fielding_error_with_no_further_mention_is_held_at_first():
    r = parse_batter_advancement_des(
        "Ty France reaches on a fielding error by second baseman Tony Kemp."
    )
    assert r.label == HELD_AT_FIRST
    assert r.advancement_caused_by_error
    assert r.hit_type_implied_floor_base == 1


def test_reaches_on_throwing_error_then_advances_is_advanced_to_second():
    r = parse_batter_advancement_des(
        "Bryan Reynolds reaches on a fielding error by center fielder Rafael Ortega. "
        "Bryan Reynolds to 2nd."
    )
    assert r.label == ADVANCED_TO_SECOND
    assert r.advancement_caused_by_error


def test_leading_clause_itself_is_an_error_advance():
    # No separate "reaches on error" clause -- the error IS the leading clause.
    r = parse_batter_advancement_des(
        "Freddie Freeman advances to 2nd, on a fielding error by right fielder Hunter Renfroe."
    )
    assert r.label == ADVANCED_TO_SECOND
    assert r.advancement_caused_by_error
    assert r.hit_type_implied_floor_base == 2


def test_genuine_inside_the_park_home_run_is_flagged_true():
    r = parse_batter_advancement_des(
        "Yuli Gurriel hits an inside-the-park home run (2) on a line drive to left field."
    )
    assert r.label == INSIDE_THE_PARK_HOME_RUN
    assert r.is_true_inside_the_park_home_run
    assert not r.advancement_caused_by_error


def test_plain_over_the_fence_home_run_is_trivial_and_unlabeled():
    r = parse_batter_advancement_des("J.D. Martinez homers (3) on a fly ball to right field.")
    assert r.label is None
    assert r.parse_status == PARSE_STATUS_UNAMBIGUOUS
    assert r.failure_reason == REASON_TRIVIAL_NO_ADVANCEMENT_OPPORTUNITY


def test_grand_slam_is_trivial_and_unlabeled():
    r = parse_batter_advancement_des(
        "George Springer hits a grand slam (22) to center field. Teoscar Hernandez scores. "
        "Santiago Espinal scores. Danny Jansen scores."
    )
    assert r.label is None
    assert r.failure_reason == REASON_TRIVIAL_NO_ADVANCEMENT_OPPORTUNITY


def test_inside_the_park_grand_slam_is_labeled_not_trivial():
    r = parse_batter_advancement_des(
        "Raimel Tapia hits an inside-the-park grand slam (5) on a fly ball to center field. "
        "Lourdes Gurriel Jr. scores. Santiago Espinal scores. Danny Jansen scores."
    )
    assert r.label == INSIDE_THE_PARK_HOME_RUN
    assert r.is_true_inside_the_park_home_run


# -- retired-while-advancing language ------------------------------------------


def test_thrown_out_at_second_is_retired_while_advancing():
    r = parse_batter_advancement_des(
        "Mookie Betts singles on a line drive to center fielder Aaron Hicks. "
        "Mookie Betts out at 2nd on the throw, center fielder Aaron Hicks to second baseman Adam Frazier."
    )
    assert r.label == RETIRED_WHILE_ADVANCING


def test_thrown_out_at_third_from_a_double_is_retired_while_advancing():
    r = parse_batter_advancement_des(
        "Elly De La Cruz doubles (36) on a fly ball to center fielder Pete Crow-Armstrong. "
        "Elly De La Cruz out at 3rd on the throw, center fielder Pete Crow-Armstrong to third baseman Isaac Paredes."
    )
    assert r.label == RETIRED_WHILE_ADVANCING


def test_thrown_out_at_home_from_a_triple_is_retired_while_advancing():
    r = parse_batter_advancement_des(
        "Lane Thomas triples (1) on a sharp fly ball to right fielder Hunter Renfroe. "
        "Lane Thomas out at home on the throw, right fielder Hunter Renfroe to second baseman Kolten Wong to catcher Omar Narvaez."
    )
    assert r.label == RETIRED_WHILE_ADVANCING


def test_thrown_out_variant_phrasing_tagged_out():
    r = parse_batter_advancement_des(
        "Some Batter singles on a line drive to left fielder A B. "
        "Some Batter is tagged out at 2nd, left fielder A B to shortstop C D."
    )
    assert r.label == RETIRED_WHILE_ADVANCING


# -- punctuation variation in real names ---------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "J.D. Martinez",
        "Jackie Bradley Jr.",
        "Ke'Bryan Hayes",
        "D.J. LeMahieu",
        "Michael A. Taylor",
        "J. P. Crawford",
        "Ronald Acuna Jr.",
    ],
)
def test_names_with_punctuation_parse_correctly(name):
    r = parse_batter_advancement_des(
        f"{name} singles on a line drive to left fielder Someone Else."
    )
    assert r.label == HELD_AT_FIRST
    assert r.parse_status == PARSE_STATUS_UNAMBIGUOUS


@pytest.mark.parametrize(
    "name",
    ["J.D. Martinez", "Jackie Bradley Jr.", "Ke'Bryan Hayes", "D.J. LeMahieu"],
)
def test_names_with_punctuation_parse_correctly_with_trailing_advance(name):
    r = parse_batter_advancement_des(
        f"{name} singles on a ground ball to left fielder Someone Else. {name} to 2nd."
    )
    assert r.label == ADVANCED_TO_SECOND


# -- review/challenge preambles ------------------------------------------------


def test_review_preamble_is_stripped_and_flagged_informational():
    r = parse_batter_advancement_des(
        "Blue Jays challenged (tag play), call on the field was overturned: "
        "Giancarlo Stanton singles on a line drive to left fielder Lourdes Gurriel. "
        "Giancarlo Stanton out at 2nd on the throw, left fielder Lourdes Gurriel to second baseman Marcus Semien."
    )
    assert r.label == RETIRED_WHILE_ADVANCING
    assert r.reviewed_or_overturned


def test_doubled_review_preamble_is_fully_stripped():
    r = parse_batter_advancement_des(
        "White Sox challenged (tag play), call on the field was upheld: "
        "White Sox challenged (home-plate collision), call on the field was upheld: "
        "Seby Zavala singles on a line drive to left fielder Austin Meadows."
    )
    assert r.label == HELD_AT_FIRST
    assert r.reviewed_or_overturned


def test_terse_review_preamble_without_call_on_the_field_clause():
    r = parse_batter_advancement_des(
        "Cubs challenged (home-plate collision): Cubs challenge (tag play): "
        "Rafael Ortega singles on a line drive to right fielder Brendan Donovan."
    )
    assert r.label == HELD_AT_FIRST
    assert r.reviewed_or_overturned


def test_non_reviewed_play_is_not_flagged_reviewed():
    r = parse_batter_advancement_des(
        "Trevor Story singles on a sharp line drive to center fielder Leody Taveras."
    )
    assert not r.reviewed_or_overturned


# -- multiple runners: only the BATTER's own clause should resolve the label --


def test_other_runners_advancing_do_not_affect_batter_label():
    r = parse_batter_advancement_des(
        "Corey Seager singles on a sharp line drive to right fielder Tommy Pham. "
        "Leody Taveras to 3rd. Marcus Semien to 2nd."
    )
    # Neither trailing clause mentions Corey Seager -- he stayed at first.
    assert r.label == HELD_AT_FIRST


def test_batter_name_that_is_a_substring_of_another_runners_name_does_not_false_match():
    # "Will Smith" must not be matched by a word-boundary-unsafe regex against
    # a longer name that happens to contain it as a substring.
    r = parse_batter_advancement_des(
        "Will Smith singles on a line drive to left fielder Someone Else. Willson Smithson to 2nd."
    )
    assert r.label == HELD_AT_FIRST


# -- name collisions / ambiguity -----------------------------------------------


def test_contradictory_retired_and_scores_is_ambiguous():
    r = parse_batter_advancement_des(
        "Some Batter singles on a line drive to left fielder A B. "
        "Some Batter out at 2nd on the throw, left fielder A B to shortstop C D. "
        "Some Batter scores."
    )
    assert r.parse_status == PARSE_STATUS_AMBIGUOUS
    assert r.label is None


# -- unsupported / unparseable sequences ---------------------------------------


def test_unrecognized_trailing_mention_of_batter_is_unsupported_not_guessed():
    r = parse_batter_advancement_des(
        "Carlos Santana doubles (33) on a line drive to left fielder Ian Happ. "
        "Blake Perkins scores. Carlos Santana to 1st."
    )
    assert r.parse_status == PARSE_STATUS_UNSUPPORTED_SEQUENCE
    assert r.label is None
    assert r.matched_clause is not None


def test_completely_unmatchable_des_is_no_batter_match():
    r = parse_batter_advancement_des("Rain delay. Game resumes.")
    assert r.parse_status == PARSE_STATUS_NO_BATTER_MATCH
    assert r.label is None


def test_missing_des_is_no_batter_match_not_a_crash():
    r = parse_batter_advancement_des(None)
    assert r.parse_status == PARSE_STATUS_NO_BATTER_MATCH
    assert r.label is None

    r2 = parse_batter_advancement_des(float("nan"))
    assert r2.parse_status == PARSE_STATUS_NO_BATTER_MATCH


def test_empty_string_des_is_no_batter_match():
    r = parse_batter_advancement_des("   ")
    assert r.parse_status == PARSE_STATUS_NO_BATTER_MATCH


# ---------------------------------------------------------------------------
# add_advancement_eligibility: DataFrame-level tests
# ---------------------------------------------------------------------------


def _advancement_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2021] * 8,
            "bb_type": [
                "line_drive",  # 0: eligible single, held at first
                "line_drive",  # 1: eligible, advanced to second
                "ground_ball",  # 2: excluded, not outfield air ball
                "fly_ball",  # 3: excluded, not a safe event (field_out)
                "fly_ball",  # 4: excluded, trivial home run
                "fly_ball",  # 5: excluded, unresolved parse
                "line_drive",  # 6: eligible, retired while advancing
                "fly_ball",  # 7: eligible, reaches on error
            ],
            "events": [
                "single",
                "single",
                "single",
                "field_out",
                "home_run",
                "double",
                "single",
                "field_error",
            ],
            "des": [
                "Trevor Story singles on a sharp line drive to center fielder Leody Taveras.",
                "Zach McKinstry singles on a ground ball to right fielder Juan Soto. Zach McKinstry to 2nd.",
                "Someone singles on a ground ball to shortstop Someone Else.",
                "Someone flies out to center fielder Someone Else.",
                "J.D. Martinez homers (3) on a fly ball to right field.",
                "Carlos Santana doubles (33) on a line drive to left fielder Ian Happ. Blake Perkins scores. Carlos Santana to 1st.",
                "Mookie Betts singles on a line drive to center fielder Aaron Hicks. Mookie Betts out at 2nd on the throw, center fielder Aaron Hicks to second baseman Adam Frazier.",
                "Ty France reaches on a fielding error by second baseman Tony Kemp.",
            ],
        }
    )


def test_eligible_rows_get_correct_labels():
    out = add_advancement_eligibility(_advancement_df())
    assert out.loc[0, "advancement_eligible"]
    assert out.loc[0, "batter_final_base"] == HELD_AT_FIRST
    assert out.loc[1, "batter_final_base"] == ADVANCED_TO_SECOND
    assert out.loc[6, "batter_final_base"] == RETIRED_WHILE_ADVANCING
    assert out.loc[7, "batter_final_base"] == HELD_AT_FIRST
    assert out.loc[7, "advancement_caused_by_error"]


def test_non_air_ball_is_excluded():
    out = add_advancement_eligibility(_advancement_df())
    assert not out.loc[2, "advancement_eligible"]
    assert out.loc[2, "advancement_exclusion_reason"] == REASON_NOT_OUTFIELD_AIR_BALL


def test_non_safe_event_is_excluded():
    out = add_advancement_eligibility(_advancement_df())
    assert not out.loc[3, "advancement_eligible"]
    assert out.loc[3, "advancement_exclusion_reason"] == REASON_NOT_ADVANCEMENT_SAFE_EVENT


def test_trivial_home_run_is_excluded():
    out = add_advancement_eligibility(_advancement_df())
    assert not out.loc[4, "advancement_eligible"]
    assert out.loc[4, "advancement_exclusion_reason"] == REASON_TRIVIAL_NO_ADVANCEMENT_OPPORTUNITY


def test_unresolved_parse_is_excluded_never_guessed():
    out = add_advancement_eligibility(_advancement_df())
    assert not out.loc[5, "advancement_eligible"]
    assert out.loc[5, "advancement_exclusion_reason"] == REASON_UNRESOLVED_DES_PARSE
    assert out.loc[5, "batter_final_base"] is None


def test_parser_version_recorded_for_every_parsed_row():
    out = add_advancement_eligibility(_advancement_df())
    parsed_rows = out[
        out["bb_type"].isin(["line_drive", "fly_ball"])
        & out["events"].isin(["single", "double", "triple", "home_run", "field_error"])
    ]
    assert (parsed_rows["advancement_parser_version"] == ADVANCEMENT_PARSER_VERSION).all()


def test_missing_required_columns_raises():
    with pytest.raises(ValueError, match="requires column"):
        add_advancement_eligibility(pd.DataFrame({"bb_type": ["line_drive"]}))


def test_summarize_advancement_exclusions_counts():
    out = add_advancement_eligibility(_advancement_df())
    counts = summarize_advancement_exclusions(out)
    assert counts.total["eligible"] == 4
    assert counts.total[REASON_NOT_OUTFIELD_AIR_BALL] == 1
    assert counts.total[REASON_NOT_ADVANCEMENT_SAFE_EVENT] == 1
    assert counts.total[REASON_TRIVIAL_NO_ADVANCEMENT_OPPORTUNITY] == 1
    assert counts.total[REASON_UNRESOLVED_DES_PARSE] == 1
    assert counts.by_season[2021]["eligible"] == 4


def test_summarize_advancement_parse_coverage_reports_all_required_groups():
    out = add_advancement_eligibility(_advancement_df())
    coverage = summarize_advancement_parse_coverage(out)
    for label in (HELD_AT_FIRST, ADVANCED_TO_SECOND, ADVANCED_TO_THIRD, RETIRED_WHILE_ADVANCING):
        assert label in coverage
    assert "reached_on_error" in coverage
    assert "multi_clause_or_multi_throw" in coverage
    assert coverage["reached_on_error"]["n_rows"] == 1
    assert coverage["__all_parsed_scope__"][PARSE_STATUS_UNAMBIGUOUS] >= 4
