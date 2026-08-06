"""Tests for the Contact Luck v0.12 public score table assembly (Phase 2-5/9).

No test touches the network -- uses the shared `v012_public_score_artifacts`
fixture (`tests/conftest.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.scoring.public_score_schema import (
    INTERVAL_INTERPRETATION_VALUES,
    REQUIRED_PUBLIC_SCORE_COLUMNS,
)
from mlb_luck_score.scoring.public_score_table import (
    DEFAULT_DEVELOPMENT_PERCENTILE_DESIGN,
    build_public_score_table,
    compute_development_percentile,
    describe_components,
)
from mlb_luck_score.scoring.qualification import STATUS_QUALIFIED


@pytest.fixture(scope="module")
def public_score_table(v012_public_score_artifacts):
    return build_public_score_table(v012_public_score_artifacts)


def test_public_score_table_has_every_required_column(public_score_table):
    for col in REQUIRED_PUBLIC_SCORE_COLUMNS:
        assert col in public_score_table.columns


def test_batter_name_is_always_null(public_score_table):
    # No reviewed batter-id-to-name join exists in this codebase.
    assert public_score_table["batter_name"].isna().all()


def test_total_and_rate_exactly_preserve_version_011_values(
    v012_public_score_artifacts, public_score_table
):
    player_season = v012_public_score_artifacts.player_season
    merged = public_score_table.merge(
        player_season.rename(columns={"batter": "batter_id"}),
        on=["batter_id", "season"],
        suffixes=("_public", "_v011"),
    )
    np.testing.assert_allclose(
        merged["total_contact_luck_runs"].to_numpy(),
        merged["total_observed_minus_expected_runs"].to_numpy(),
    )
    pd.testing.assert_series_equal(
        merged["contact_luck_runs_per_100"],
        merged["observed_minus_expected_per_100"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        merged["lower_95_interval"],
        merged["observed_minus_expected_per_100_ci_low"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        merged["upper_95_interval"],
        merged["observed_minus_expected_per_100_ci_high"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        merged["qualification_status_public"],
        merged["qualification_status_v011"],
        check_names=False,
    )


def test_official_rank_eligible_matches_qualified_status(public_score_table):
    is_qualified = public_score_table["qualification_status"] == STATUS_QUALIFIED
    assert (public_score_table["official_rank_eligible"] == is_qualified).all()


def test_interval_interpretation_is_always_a_known_value_or_null(public_score_table):
    values = public_score_table["interval_interpretation"].dropna().unique()
    assert set(values) <= set(INTERVAL_INTERPRETATION_VALUES)


def test_interval_interpretation_computed_for_non_qualified_rows_too(public_score_table):
    # Phase 3 explicitly forbids suppressing interval interpretation for
    # rows that are not officially ranked.
    non_qualified_with_interval = public_score_table[
        (public_score_table["qualification_status"] != STATUS_QUALIFIED)
        & public_score_table["lower_95_interval"].notna()
    ]
    if len(non_qualified_with_interval) > 0:
        assert non_qualified_with_interval["interval_interpretation"].notna().all()


def test_point_estimate_and_interval_fields_are_always_paired(public_score_table):
    has_rate = public_score_table["contact_luck_runs_per_100"].notna()
    has_lower = public_score_table["lower_95_interval"].notna()
    has_upper = public_score_table["upper_95_interval"].notna()
    assert (has_rate == has_lower).all()
    assert (has_rate == has_upper).all()


def test_component_status_reason_codes_present_for_every_row(public_score_table):
    assert (
        public_score_table["component_status_reason_codes"]
        .apply(lambda d: isinstance(d, dict))
        .all()
    )


def test_model_version_map_has_all_four_components(public_score_table):
    for record in public_score_table["model_version"]:
        assert set(record.keys()) == {
            "contact",
            "outfield_defense",
            "infield_defense",
            "advancement",
        }
        assert record["contact"] == "baseline_v02"


def test_describe_components_returns_four_named_components(public_score_table):
    row = public_score_table.iloc[0]
    components = describe_components(row)
    names = {c["component_name"] for c in components}
    assert names == {"contact", "unexplained_residual", "defensive_execution", "advancement"}
    for component in components:
        assert "total_runs" in component
        assert "per_100" in component
        assert "status" in component


def test_development_percentile_is_bounded(public_score_table):
    percentile = compute_development_percentile(public_score_table)
    valid = percentile.dropna()
    if len(valid) > 0:
        assert (valid >= 0).all()
        assert (valid <= 100).all()


def test_development_percentile_is_rank_based_not_a_linear_rescaling():
    # The defining "non-additive" property: percentile depends ONLY on rank
    # order, not on the magnitude of the underlying metric -- three widely
    # -spaced values (1, 100, 1000) produce EVENLY spaced percentiles,
    # proving this is not a linear/additive rescaling of the runs value.
    table = pd.DataFrame(
        {
            "batter_id": [1, 2, 3],
            "season": [2024, 2024, 2024],
            "qualification_status": [STATUS_QUALIFIED] * 3,
            "contact_luck_runs_per_100": [1.0, 100.0, 1000.0],
        }
    )
    percentile = compute_development_percentile(table)
    assert percentile.tolist() == pytest.approx([100 / 3, 200 / 3, 100.0])


def test_development_percentile_only_covers_qualified_rows_in_reference_seasons(public_score_table):
    percentile = compute_development_percentile(public_score_table)
    non_qualified_mask = public_score_table["qualification_status"] != STATUS_QUALIFIED
    assert percentile[non_qualified_mask].isna().all()


def test_development_percentile_design_preserves_required_fields():
    design = DEFAULT_DEVELOPMENT_PERCENTILE_DESIGN
    assert design.reference_seasons
    assert design.reference_population
    assert design.transformation_version
    assert design.tie_behavior
    assert design.out_of_range_handling


def test_development_percentile_not_included_in_default_table(public_score_table):
    assert "development_percentile" not in public_score_table.columns
    assert "percentile" not in public_score_table.columns
