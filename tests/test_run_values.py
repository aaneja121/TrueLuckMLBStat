from __future__ import annotations

import pytest

from mlb_luck_score.config import CLASS_ORDER
from mlb_luck_score.scoring.run_values import (
    DEFAULT_RUN_VALUE_MAP,
    SEASON_WOBA_CONSTANTS,
    compute_default_run_value_map,
    compute_season_run_values,
)


def test_annual_run_values_match_manual_linear_weights_formula():
    for season, constants in SEASON_WOBA_CONSTANTS.items():
        table = compute_season_run_values(season)
        assert table["out"] == pytest.approx((0.0 - constants.league_woba) / constants.woba_scale)
        assert table["single"] == pytest.approx(
            (constants.single - constants.league_woba) / constants.woba_scale
        )
        assert table["double"] == pytest.approx(
            (constants.double - constants.league_woba) / constants.woba_scale
        )
        assert table["triple"] == pytest.approx(
            (constants.triple - constants.league_woba) / constants.woba_scale
        )
        assert table["home_run"] == pytest.approx(
            (constants.home_run - constants.league_woba) / constants.woba_scale
        )


def test_2021_annual_run_values_exact():
    table = compute_season_run_values(2021)
    assert table["out"] == pytest.approx(-0.2597187758478081)
    assert table["single"] == pytest.approx(0.4673283705541769)
    assert table["double"] == pytest.approx(0.7675765095119933)
    assert table["triple"] == pytest.approx(1.0372208436724566)
    assert table["home_run"] == pytest.approx(1.4003308519437552)


def test_2024_annual_run_values_exact():
    table = compute_season_run_values(2024)
    assert table["out"] == pytest.approx(-0.249597423510467)
    assert table["single"] == pytest.approx(0.46054750402576494)
    assert table["double"] == pytest.approx(0.7600644122383252)
    assert table["triple"] == pytest.approx(1.030595813204509)
    assert table["home_run"] == pytest.approx(1.400966183574879)


def test_unknown_season_raises_clear_error():
    with pytest.raises(ValueError, match="No FanGraphs Guts constants configured"):
        compute_season_run_values(1999)


def test_fixed_2021_2024_average_matches_expected_values():
    # Expected values from the published task specification, verified to
    # match a from-scratch recomputation from the source constants.
    expected = {
        "out": -0.254916,
        "single": 0.463266,
        "double": 0.763026,
        "triple": 1.033068,
        "home_run": 1.400288,
    }
    for cls, value in expected.items():
        assert DEFAULT_RUN_VALUE_MAP[cls] == pytest.approx(value, abs=1e-5)


def test_default_run_value_map_is_arithmetic_mean_of_all_four_seasons():
    tables = [compute_season_run_values(s) for s in (2021, 2022, 2023, 2024)]
    for cls in CLASS_ORDER:
        manual_mean = sum(t[cls] for t in tables) / 4
        assert DEFAULT_RUN_VALUE_MAP[cls] == pytest.approx(manual_mean)


def test_compute_default_run_value_map_recomputes_not_hardcoded():
    # Averaging just one season should differ from the 4-season average,
    # proving the default table is genuinely computed, not a hardcoded copy.
    single_season = compute_default_run_value_map(seasons=(2022,))
    assert single_season != DEFAULT_RUN_VALUE_MAP
    assert single_season["home_run"] == pytest.approx(compute_season_run_values(2022)["home_run"])


def test_run_value_map_covers_all_class_order_outcomes():
    assert set(DEFAULT_RUN_VALUE_MAP.keys()) == set(CLASS_ORDER)


def test_out_run_value_is_negative_and_home_run_is_largest():
    assert DEFAULT_RUN_VALUE_MAP["out"] < 0
    assert DEFAULT_RUN_VALUE_MAP["home_run"] == max(DEFAULT_RUN_VALUE_MAP.values())
    ordered = [DEFAULT_RUN_VALUE_MAP[c] for c in CLASS_ORDER]
    assert ordered == sorted(ordered)
