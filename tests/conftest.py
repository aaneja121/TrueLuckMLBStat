"""Shared synthetic fixtures. No test in this suite requires network access."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _row(
    game_pk: int,
    at_bat_number: int,
    pitch_number: int,
    game_date: str,
    events: str | None,
    *,
    launch_speed: float | None = 90.0,
    launch_angle: float | None = 15.0,
    hit_distance_sc: float | None = 250.0,
    hc_x: float | None = 130.0,
    hc_y: float | None = 150.0,
    bb_type: str | None = "line_drive",
    stand: str | None = "R",
    venue: str | None = "Synthetic Park",
    description: str | None = None,
    batter: int = 1,
) -> dict:
    return {
        "game_pk": game_pk,
        "game_date": game_date,
        "at_bat_number": at_bat_number,
        "pitch_number": pitch_number,
        "batter": batter,
        "pitcher": 999,
        "player_name": "Test Player",
        "home_team": "AAA",
        "away_team": "BBB",
        "venue": venue,
        "inning": 3,
        "inning_topbot": "Top",
        "outs_when_up": 1,
        "on_1b": None,
        "on_2b": None,
        "on_3b": None,
        "events": events,
        "description": description,
        "launch_speed": launch_speed,
        "launch_angle": launch_angle,
        "hit_distance_sc": hit_distance_sc,
        "hit_location": 7,
        "hc_x": hc_x,
        "hc_y": hc_y,
        "bb_type": bb_type,
        "stand": stand,
        "p_throws": "R",
        "if_fielding_alignment": "Standard",
        "of_fielding_alignment": "Standard",
        "estimated_ba_using_speedangle": 0.5,
        "estimated_woba_using_speedangle": 0.6,
        "woba_value": 1.0,
        "delta_home_win_exp": 0.01,
        "bat_speed": 70.0,
        "sprint_speed": 27.0,
    }


@pytest.fixture
def raw_statcast_rows() -> list[dict]:
    """Synthetic raw Statcast-shaped rows spanning eligible/ineligible events."""
    return [
        _row(1, 1, 1, "2022-04-05", "single", hc_x=140.0, hc_y=160.0),
        _row(1, 2, 1, "2022-04-05", "double", hc_x=90.0, hc_y=120.0),
        _row(1, 3, 1, "2022-04-05", "triple", hc_x=170.0, hc_y=100.0),
        _row(1, 4, 1, "2022-04-05", "home_run", hc_x=125.0, hc_y=50.0),
        _row(1, 5, 1, "2022-04-05", "field_out"),
        _row(2, 1, 1, "2023-05-10", "strikeout", launch_speed=None, launch_angle=None),
        _row(2, 2, 1, "2023-05-10", "walk", launch_speed=None, launch_angle=None),
        _row(2, 3, 1, "2023-05-10", "field_error"),
        _row(2, 4, 1, "2023-05-10", "fielders_choice"),
        _row(2, 5, 1, "2023-05-10", "fielders_choice_out"),
        _row(3, 1, 1, "2024-06-01", "single", launch_speed=None),  # missing contact data
        _row(3, 2, 1, "2024-06-01", "single"),
        _row(3, 2, 1, "2024-06-01", "single"),  # exact duplicate of the row above
        _row(3, 3, 1, "not-a-date", "double"),  # malformed date
    ]


@pytest.fixture
def raw_statcast_df(raw_statcast_rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(raw_statcast_rows)


# ---------------------------------------------------------------------------
# Version 0.11 shared fixture: a full Version 0.10 ledger (reusing test_
# attribution_ledger.py's synthetic-data builders and model-training steps
# verbatim, not duplicating them) with synthetic batter identifiers assigned
# on top, plus its Version 0.11 confidence/aggregation layers. Session
# -scoped -- training four models is the expensive part, and nothing in the
# Version 0.11 test suite mutates this fixture's outputs.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def v011_full_ledger() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from mlb_luck_score.eligibility import (
        add_advancement_eligibility,
        add_infield_opportunity_eligibility,
        add_outfield_opportunity_eligibility,
        compute_eligibility,
    )
    from mlb_luck_score.features.build_contact_features import (
        INFIELD_OPPORTUNITY_TARGET_COLUMN,
        add_advancement_contact_probability_features,
        add_advancement_features,
        add_opportunity_features_by_domain,
    )
    from mlb_luck_score.models.train_advancement_model import train_advancement_model
    from mlb_luck_score.models.train_contact_model import predict_proba_ordered, train_model
    from mlb_luck_score.models.train_opportunity_model import train_opportunity_model
    from mlb_luck_score.scoring.attribution_ledger import build_attribution_ledger
    from test_attribution_ledger import (  # noqa: PLC0415 -- deferred, test-only import
        ADVANCEMENT_CATEGORICAL,
        ADVANCEMENT_NUMERIC,
        _synthetic_ground_ball_df,
        _synthetic_outfield_df,
    )

    outfield_df = _synthetic_outfield_df()
    ground_df = _synthetic_ground_ball_df()
    df = pd.concat([outfield_df, ground_df], ignore_index=True)

    rng = np.random.default_rng(0)
    df["batter"] = rng.integers(1000, 1010, size=len(df))

    df = compute_eligibility(df)
    df = add_outfield_opportunity_eligibility(df)
    df = add_infield_opportunity_eligibility(df)
    df = add_advancement_eligibility(df)
    df = add_opportunity_features_by_domain(df)
    df = add_advancement_features(df)

    contact_elig = df[df["eligible_for_training"].fillna(False)]
    contact_trained = train_model(contact_elig, class_weight=None)
    contact_feature_cols = contact_trained.numeric_features + contact_trained.categorical_features
    contact_proba_all = predict_proba_ordered(contact_trained, df[contact_feature_cols])
    df = add_advancement_contact_probability_features(df, contact_proba_all)

    outfield_elig = df[df["outfield_opportunity_eligible"].astype(bool)]
    outfield_trained = train_opportunity_model(outfield_elig, class_weight=None)

    infield_elig = df[df["infield_opportunity_eligible"].astype(bool)]
    infield_trained = train_opportunity_model(
        infield_elig,
        numeric_features=[
            "launch_speed",
            "launch_angle",
            "spray_angle_approx",
            "hit_distance_sc",
            "sprint_speed",
            "outs_when_up",
            "on_1b_occupied",
        ],
        categorical_features=[
            "stand",
            "if_fielding_alignment",
            "assigned_infield_position",
            "surface_type",
        ],
        class_weight=None,
        target_column=INFIELD_OPPORTUNITY_TARGET_COLUMN,
    )

    advancement_elig = df[df["advancement_eligible"].astype(bool)]
    advancement_trained = train_advancement_model(
        advancement_elig,
        numeric_features=ADVANCEMENT_NUMERIC,
        categorical_features=ADVANCEMENT_CATEGORICAL,
    )

    ledger = build_attribution_ledger(
        df,
        contact_trained,
        outfield_trained=outfield_trained,
        infield_trained=infield_trained,
        advancement_trained=advancement_trained,
        outfield_confidence_status="calibrated",
        infield_confidence_status="calibrated_with_limited_subgroup_evidence",
        advancement_confidence_status="calibrated_with_limited_subgroup_evidence",
    )

    from mlb_luck_score.scoring.component_confidence import build_play_level_confidence

    confidence = build_play_level_confidence(
        df,
        ledger,
        outfield_model_status="calibrated",
        infield_model_status="calibrated_with_limited_subgroup_evidence",
        advancement_model_status="calibrated_with_limited_subgroup_evidence",
    )
    return df, ledger, confidence
