"""Tests for the Contact Luck v0.7C gated outfield opportunity/execution report.

No test touches the network -- synthetic gated-outfield-shaped fixtures only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.features.build_contact_features import (
    NEAR_WALL_CATEGORICAL_FEATURES,
    NEAR_WALL_NUMERIC_FEATURES,
    OPPORTUNITY_CATEGORICAL_FEATURES,
    OPPORTUNITY_NUMERIC_FEATURES,
)
from mlb_luck_score.models.train_contact_model import train_model
from mlb_luck_score.models.train_opportunity_model import train_opportunity_model
from mlb_luck_score.scoring.gated_outfield_report import (
    STATUS_AVAILABLE_NEAR_WALL_CALIBRATED,
    STATUS_AVAILABLE_OPEN_FIELD,
    STATUS_PROVISIONAL_NEAR_WALL,
    STATUS_UNAVAILABLE_GEOMETRY_FALLBACK,
    build_gated_outfield_report,
)


def _rows(n: int, *, gate_kind: str, rng: np.random.Generator) -> list[dict]:
    rows = []
    for i in range(n):
        converted = i % 2
        outcome_class = "out" if converted else "single"
        base = {
            "game_pk": 1000 + i,
            "launch_speed": 95.0 + rng.normal(0, 2),
            "launch_angle": 25.0 + rng.normal(0, 2),
            "spray_angle_approx": rng.uniform(-20, 20),
            "hit_distance_sc": 300.0 + rng.normal(0, 10),
            "bb_type": "fly_ball",
            "stand": "R",
            "venue": "Synthetic Park",
            "estimated_hang_time_s": 4.0 + rng.normal(0, 0.3),
            "landing_x_ft": rng.normal(0, 10),
            "landing_y_ft": 300.0,
            "wall_distance_in_spray_direction": 350.0,
            "absolute_distance_to_wall": 50.0,
            "wall_height_in_spray_direction": 10.0,
            "projected_distance_to_wall_margin": -50.0,
            "of_fielding_alignment": "Standard",
            "assigned_outfield_position": "8",
            "wall_segment_label": "CF",
            "outcome_class": outcome_class,
            "converted_to_out": converted,
        }
        if gate_kind == "open_field":
            base["has_park_geometry"] = True
            base["geometry_uncertain"] = "False"
            base["near_wall_20ft"] = "False"
        elif gate_kind == "near_wall":
            base["has_park_geometry"] = True
            base["geometry_uncertain"] = "False"
            base["near_wall_20ft"] = "True"
        else:  # fallback
            base["has_park_geometry"] = False
            base["geometry_uncertain"] = "False"
            base["near_wall_20ft"] = "False"
        rows.append(base)
    return rows


@pytest.fixture
def gated_setup():
    rng = np.random.default_rng(21)
    all_rows = (
        _rows(60, gate_kind="open_field", rng=rng)
        + _rows(60, gate_kind="near_wall", rng=rng)
        + _rows(10, gate_kind="fallback", rng=rng)
    )
    df = pd.DataFrame(all_rows)

    contact_trained = train_model(df, class_weight=None)
    open_field_trained = train_opportunity_model(
        df,
        class_weight=None,
        numeric_features=OPPORTUNITY_NUMERIC_FEATURES,
        categorical_features=OPPORTUNITY_CATEGORICAL_FEATURES,
    )
    near_wall_trained = train_opportunity_model(
        df,
        class_weight=None,
        numeric_features=NEAR_WALL_NUMERIC_FEATURES,
        categorical_features=NEAR_WALL_CATEGORICAL_FEATURES,
    )
    return df, contact_trained, open_field_trained, near_wall_trained


def test_report_preserves_row_count_and_order(gated_setup):
    df, contact_trained, open_field_trained, near_wall_trained = gated_setup
    report = build_gated_outfield_report(
        df,
        contact_trained,
        open_field_trained,
        near_wall_trained,
        near_wall_specialist_calibrated=False,
    )
    assert len(report) == len(df)
    assert report.index.equals(df.index)


def test_status_distribution_matches_gate(gated_setup):
    df, contact_trained, open_field_trained, near_wall_trained = gated_setup
    report = build_gated_outfield_report(
        df,
        contact_trained,
        open_field_trained,
        near_wall_trained,
        near_wall_specialist_calibrated=False,
    )
    counts = report["execution_availability_status"].value_counts()
    assert counts[STATUS_AVAILABLE_OPEN_FIELD] == 60
    assert counts[STATUS_PROVISIONAL_NEAR_WALL] == 60
    assert counts[STATUS_UNAVAILABLE_GEOMETRY_FALLBACK] == 10


def test_near_wall_calibrated_flag_changes_status_label(gated_setup):
    df, contact_trained, open_field_trained, near_wall_trained = gated_setup
    report = build_gated_outfield_report(
        df,
        contact_trained,
        open_field_trained,
        near_wall_trained,
        near_wall_specialist_calibrated=True,
    )
    counts = report["execution_availability_status"].value_counts()
    assert counts[STATUS_AVAILABLE_NEAR_WALL_CALIBRATED] == 60
    assert STATUS_PROVISIONAL_NEAR_WALL not in counts


def test_fallback_rows_have_nan_opportunity_and_execution(gated_setup):
    df, contact_trained, open_field_trained, near_wall_trained = gated_setup
    report = build_gated_outfield_report(
        df,
        contact_trained,
        open_field_trained,
        near_wall_trained,
        near_wall_specialist_calibrated=False,
    )
    fallback = report[
        report["execution_availability_status"] == STATUS_UNAVAILABLE_GEOMETRY_FALLBACK
    ]
    assert fallback["p_out_opportunity"].isna().all()
    assert fallback["defensive_execution"].isna().all()
    assert fallback["batter_favorable_defensive_circumstance"].isna().all()
    # Ground truth and contact-model expectation ARE still reported.
    assert fallback["converted_to_out"].notna().all()
    assert fallback["residual_contact_luck_runs"].notna().all()


def test_open_field_and_near_wall_rows_have_real_scores(gated_setup):
    df, contact_trained, open_field_trained, near_wall_trained = gated_setup
    report = build_gated_outfield_report(
        df,
        contact_trained,
        open_field_trained,
        near_wall_trained,
        near_wall_specialist_calibrated=False,
    )
    scored = report[report["execution_availability_status"] != STATUS_UNAVAILABLE_GEOMETRY_FALLBACK]
    assert scored["p_out_opportunity"].notna().all()
    assert (scored["p_out_opportunity"] >= 0).all() and (scored["p_out_opportunity"] <= 1).all()
    assert scored["defensive_execution"].notna().all()


def test_report_includes_game_pk(gated_setup):
    df, contact_trained, open_field_trained, near_wall_trained = gated_setup
    report = build_gated_outfield_report(
        df,
        contact_trained,
        open_field_trained,
        near_wall_trained,
        near_wall_specialist_calibrated=False,
    )
    assert "game_pk" in report.columns
