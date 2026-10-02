"""Contact Forecast R1: the paired, globally batter-clustered bootstrap.

The claims worth pinning are: a resampled batter brings ALL of his rows with
him, across every season, cutoff and horizon; the multiplicity-weighted fast
path is exactly the explicit resample; the macro-average is recomputed inside
each replicate rather than assembled afterwards; and the whole thing is
deterministic given its seed.

Synthetic data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast import bootstrap as bs
from forecast.forecast_config import PRIMARY_COMPARISON

REFERENCE, CHALLENGER = PRIMARY_COMPARISON


def make_frame(
    *,
    n_batters: int = 30,
    seasons: tuple[int, ...] = (2023, 2024),
    cutoffs: tuple[int, ...] = (100, 150),
    horizons: tuple[int, ...] = (50, 100),
    seed: int = 4,
) -> pd.DataFrame:
    """A paired frame with the real study's repeated-hitter structure."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for batter in range(n_batters):
        skill = rng.normal(4.0, 5.0)
        for season in seasons:
            for cutoff in cutoffs:
                for horizon in horizons:
                    rows.append(
                        {
                            "batter": batter,
                            "season": season,
                            "cutoff": cutoff,
                            "horizon": horizon,
                            "target": skill + rng.normal(0.0, 7.0),
                            REFERENCE: skill + rng.normal(0.0, 6.0),
                            CHALLENGER: skill + rng.normal(0.0, 3.0),
                        }
                    )
    return pd.DataFrame(rows)


def prepared_frame(frame: pd.DataFrame) -> bs._PreparedFrame:
    return bs._prepare(
        frame,
        target="target",
        reference=REFERENCE,
        challenger=CHALLENGER,
        cluster_column="batter",
    )


# --------------------------------------------------------------------------
# The cluster really is the batter, globally
# --------------------------------------------------------------------------


def test_a_resampled_batter_brings_every_one_of_his_rows() -> None:
    frame = make_frame()
    prepared = prepared_frame(frame)
    for rows in prepared.cluster_rows:
        batters = set(frame.iloc[rows]["batter"])
        assert len(batters) == 1
        # All of that batter's seasons, cutoffs and horizons travel together.
        assert len(rows) == int((frame["batter"] == batters.pop()).sum())
        assert len(rows) == 8


def test_clusters_partition_the_frame_exactly_once() -> None:
    frame = make_frame()
    prepared = prepared_frame(frame)
    covered = np.concatenate(prepared.cluster_rows)
    assert sorted(covered.tolist()) == list(range(len(frame)))


def test_batter_seasons_are_not_the_cluster() -> None:
    """A (batter, season) cluster would give twice as many clusters as batters."""
    frame = make_frame()
    prepared = prepared_frame(frame)
    assert prepared.n_clusters == frame["batter"].nunique()
    assert prepared.n_clusters < frame.groupby(["batter", "season"]).ngroups


def test_multiplicity_weights_match_an_explicit_resample() -> None:
    """The fast path is checked against the literal, obvious construction."""
    frame = make_frame()
    prepared = prepared_frame(frame)
    n_groups = len(prepared.group_keys)
    rng = np.random.default_rng(0)

    for _ in range(5):
        drawn = rng.integers(0, prepared.n_clusters, prepared.n_clusters)
        multiplicities = np.bincount(drawn, minlength=prepared.n_clusters).astype(float)
        fast = bs._replicate_group_sums(prepared, multiplicities)

        index = bs._replicate_row_index(prepared.cluster_rows, drawn)
        slow_count = np.bincount(prepared.group_codes[index], minlength=n_groups)
        slow_abs = np.bincount(
            prepared.group_codes[index],
            weights=prepared.abs_reference[index],
            minlength=n_groups,
        )
        slow_sq = np.bincount(
            prepared.group_codes[index],
            weights=prepared.sq_challenger[index],
            minlength=n_groups,
        )
        assert np.allclose(fast["count"], slow_count)
        assert np.allclose(fast["abs_reference"], slow_abs)
        assert np.allclose(fast["sq_challenger"], slow_sq)


# --------------------------------------------------------------------------
# What the intervals are intervals on
# --------------------------------------------------------------------------


def test_point_estimate_matches_the_unresampled_paired_delta() -> None:
    frame = make_frame()
    result = bs.run_paired_bootstrap(
        frame,
        target="target",
        reference=REFERENCE,
        challenger=CHALLENGER,
        design=bs.BootstrapDesign(reps=50, seed=1),
    )
    cell = frame[(frame["cutoff"] == 100) & (frame["horizon"] == 100)]
    expected = float(
        np.abs(cell[CHALLENGER] - cell["target"]).mean()
        - np.abs(cell[REFERENCE] - cell["target"]).mean()
    )
    point = result["cells"]["K100_H100"]["pooled"]["delta_mae"]["point_estimate"]
    assert point == pytest.approx(expected)


def test_a_genuinely_better_challenger_gives_a_negative_interval() -> None:
    frame = make_frame(n_batters=120, seed=8)
    result = bs.run_paired_bootstrap(
        frame,
        target="target",
        reference=REFERENCE,
        challenger=CHALLENGER,
        design=bs.BootstrapDesign(reps=300, seed=2),
    )
    summary = result["cells"]["K100_H100"]["pooled"]["delta_mae"]
    assert summary["point_estimate"] < 0
    assert summary["ci_upper"] < 0
    assert summary["ci_crosses_zero"] is False
    assert summary["direction"] == "challenger_better"


def test_identical_predictors_give_a_degenerate_interval_at_zero() -> None:
    frame = make_frame()
    frame[CHALLENGER] = frame[REFERENCE]
    result = bs.run_paired_bootstrap(
        frame,
        target="target",
        reference=REFERENCE,
        challenger=CHALLENGER,
        design=bs.BootstrapDesign(reps=50, seed=3),
    )
    summary = result["cells"]["K100_H100"]["pooled"]["delta_mae"]
    assert summary["point_estimate"] == pytest.approx(0.0)
    assert summary["ci_lower"] == pytest.approx(0.0)
    assert summary["ci_upper"] == pytest.approx(0.0)
    assert summary["ci_crosses_zero"] is True


def test_the_macro_average_is_recomputed_inside_each_replicate() -> None:
    """Unbalanced seasons make pooled and macro differ, replicate by replicate."""
    frame = make_frame(n_batters=40, seasons=(2023, 2024), seed=6)
    # Thin 2023 out sharply so pooling and equal-weighting disagree.
    keep = (frame["season"] == 2024) | (frame["batter"] < 5)
    frame = frame[keep].reset_index(drop=True)
    frame.loc[frame["season"] == 2023, CHALLENGER] += 12.0

    result = bs.run_paired_bootstrap(
        frame,
        target="target",
        reference=REFERENCE,
        challenger=CHALLENGER,
        design=bs.BootstrapDesign(reps=200, seed=5),
    )
    cell = result["cells"]["K100_H100"]
    pooled = cell["pooled"]["delta_mae"]["point_estimate"]
    macro = cell["macro_average_of_season_deltas"]["delta_mae"]["point_estimate"]
    assert macro != pytest.approx(pooled)
    assert cell["macro_average_of_season_deltas"]["n_seasons"] == 2
    # The macro interval is its own interval, not a copy of the pooled one.
    assert cell["macro_average_of_season_deltas"]["delta_mae"]["ci_lower"] != pytest.approx(
        cell["pooled"]["delta_mae"]["ci_lower"]
    )


def test_every_season_gets_its_own_interval() -> None:
    frame = make_frame()
    result = bs.run_paired_bootstrap(
        frame,
        target="target",
        reference=REFERENCE,
        challenger=CHALLENGER,
        design=bs.BootstrapDesign(reps=50, seed=7),
    )
    by_season = result["cells"]["K100_H100"]["by_season"]
    assert set(by_season) == {"2023", "2024"}
    for summary in by_season.values():
        assert summary["delta_mae"]["ci_lower"] is not None


def test_every_cell_of_the_grid_is_reported() -> None:
    frame = make_frame()
    result = bs.run_paired_bootstrap(
        frame,
        target="target",
        reference=REFERENCE,
        challenger=CHALLENGER,
        design=bs.BootstrapDesign(reps=20, seed=9),
    )
    assert set(result["cells"]) == {"K100_H50", "K100_H100", "K150_H50", "K150_H100"}


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_the_bootstrap_is_deterministic_given_its_seed() -> None:
    frame = make_frame()
    design = bs.BootstrapDesign(reps=40, seed=123)
    first = bs.run_paired_bootstrap(
        frame, target="target", reference=REFERENCE, challenger=CHALLENGER, design=design
    )
    second = bs.run_paired_bootstrap(
        frame, target="target", reference=REFERENCE, challenger=CHALLENGER, design=design
    )
    assert first == second


def test_a_different_seed_gives_a_different_interval() -> None:
    frame = make_frame()
    first = bs.run_paired_bootstrap(
        frame,
        target="target",
        reference=REFERENCE,
        challenger=CHALLENGER,
        design=bs.BootstrapDesign(reps=40, seed=1),
    )
    second = bs.run_paired_bootstrap(
        frame,
        target="target",
        reference=REFERENCE,
        challenger=CHALLENGER,
        design=bs.BootstrapDesign(reps=40, seed=2),
    )
    assert (
        first["cells"]["K100_H100"]["pooled"]["delta_mae"]["ci_lower"]
        != second["cells"]["K100_H100"]["pooled"]["delta_mae"]["ci_lower"]
    )


def test_unpaired_nulls_are_refused_rather_than_dropped() -> None:
    frame = make_frame()
    frame.loc[0, CHALLENGER] = np.nan
    with pytest.raises(bs.ForecastBootstrapError, match="not paired"):
        bs.run_paired_bootstrap(
            frame,
            target="target",
            reference=REFERENCE,
            challenger=CHALLENGER,
            design=bs.BootstrapDesign(reps=5, seed=1),
        )


def test_a_missing_grouping_column_is_refused() -> None:
    frame = make_frame().drop(columns=["horizon"])
    with pytest.raises(bs.ForecastBootstrapError, match="missing column"):
        bs.run_paired_bootstrap(
            frame,
            target="target",
            reference=REFERENCE,
            challenger=CHALLENGER,
            design=bs.BootstrapDesign(reps=5, seed=1),
        )


def test_the_design_record_states_the_clustering_and_the_pairing() -> None:
    record = bs.BootstrapDesign(reps=2000, seed=1, alpha=0.05).as_record()
    assert record["cluster_column"] == "batter"
    assert record["paired"] is True
    assert "global" in record["clustering"]
    assert record["interval"] == "95% percentile interval"
    assert any("macro" in item for item in record["recomputed_inside_each_replicate"])


def test_the_sign_convention_travels_with_the_result() -> None:
    frame = make_frame()
    result = bs.run_paired_bootstrap(
        frame,
        target="target",
        reference=REFERENCE,
        challenger=CHALLENGER,
        design=bs.BootstrapDesign(reps=10, seed=1),
    )
    assert result["sign_convention"]["definition"].startswith("delta_metric =")
    assert result["reference"] == REFERENCE
    assert result["challenger"] == CHALLENGER
