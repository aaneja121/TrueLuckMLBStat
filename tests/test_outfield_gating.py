"""Tests for the Contact Luck v0.7C outfield-opportunity gating logic.

No test touches the network -- synthetic geometry-shaped fixtures only.
"""

from __future__ import annotations

import pandas as pd

from mlb_luck_score.models.outfield_gating import (
    GATE_CONTACT_ONLY_FALLBACK,
    GATE_NEAR_WALL,
    GATE_OPEN_FIELD,
    assign_outfield_opportunity_gate,
)


def _row(**overrides) -> dict:
    base = {
        "has_park_geometry": True,
        "geometry_uncertain": "False",
        "wall_distance_in_spray_direction": 350.0,
        "near_wall_20ft": "False",
    }
    base.update(overrides)
    return base


def test_reliable_geometry_open_field_gate():
    df = pd.DataFrame([_row()])
    gate = assign_outfield_opportunity_gate(df)
    assert gate.iloc[0] == GATE_OPEN_FIELD


def test_reliable_geometry_near_wall_gate():
    df = pd.DataFrame([_row(near_wall_20ft="True")])
    gate = assign_outfield_opportunity_gate(df)
    assert gate.iloc[0] == GATE_NEAR_WALL


def test_missing_geometry_falls_back():
    df = pd.DataFrame([_row(has_park_geometry=False)])
    gate = assign_outfield_opportunity_gate(df)
    assert gate.iloc[0] == GATE_CONTACT_ONLY_FALLBACK


def test_geometry_uncertain_falls_back_even_if_near_wall():
    # near_wall_20ft=True but geometry_uncertain=True -- unreliable wall
    # distance means we can't trust the near-wall flag itself either.
    df = pd.DataFrame([_row(geometry_uncertain="True", near_wall_20ft="True")])
    gate = assign_outfield_opportunity_gate(df)
    assert gate.iloc[0] == GATE_CONTACT_ONLY_FALLBACK


def test_missing_wall_distance_falls_back():
    df = pd.DataFrame([_row(wall_distance_in_spray_direction=None)])
    gate = assign_outfield_opportunity_gate(df)
    assert gate.iloc[0] == GATE_CONTACT_ONLY_FALLBACK


def test_no_geometry_column_at_all_falls_back():
    df = pd.DataFrame({"launch_speed": [95.0]})
    gate = assign_outfield_opportunity_gate(df)
    assert gate.iloc[0] == GATE_CONTACT_ONLY_FALLBACK


def test_stringified_boolean_handled_correctly_not_naive_astype():
    # "False" naively cast via bool("False") is True in plain Python -- this
    # must NOT be misread as near_wall=True. See _bool_mask's docstring.
    df = pd.DataFrame([_row(near_wall_20ft="False")])
    gate = assign_outfield_opportunity_gate(df)
    assert gate.iloc[0] == GATE_OPEN_FIELD


def test_vectorized_over_multiple_rows_matches_real_world_proportions():
    rows = [
        _row(),  # open_field
        _row(near_wall_20ft="True"),  # near_wall
        _row(has_park_geometry=False),  # fallback
    ]
    df = pd.DataFrame(rows)
    gate = assign_outfield_opportunity_gate(df)
    assert gate.tolist() == [GATE_OPEN_FIELD, GATE_NEAR_WALL, GATE_CONTACT_ONLY_FALLBACK]
