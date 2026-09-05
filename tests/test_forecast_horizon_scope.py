"""Contact Forecast: the horizon scoping study's arithmetic and its guards.

Two pieces carry the study's conclusion and are pinned here: the theoretical
ceiling (how much of a target is even signal at a given horizon) and the
sample-size implication (which scales with 1/delta^2, and is what decides
whether a longer, sparser window pays for itself).

Synthetic inputs only.
"""

from __future__ import annotations

import numpy as np
import pytest

from forecast import horizon_scope as hs


def test_the_ceiling_rises_with_the_horizon() -> None:
    """Noise averages down as 1/H, so a longer target window is more learnable."""
    ceilings = [
        hs.theoretical_ceiling(between_variance=13.36, within_variance=2197.0, horizon=h)[
            "max_achievable_r_squared"
        ]
        for h in (50, 100, 200, 400)
    ]
    assert ceilings == sorted(ceilings)
    assert ceilings[0] < 0.30 < ceilings[2]


def test_the_ceiling_matches_the_closed_form() -> None:
    result = hs.theoretical_ceiling(between_variance=10.0, within_variance=1000.0, horizon=100)
    assert result["noise_variance"] == pytest.approx(10.0)
    assert result["max_achievable_r_squared"] == pytest.approx(0.5)


def test_a_noiseless_target_has_a_ceiling_of_one() -> None:
    result = hs.theoretical_ceiling(between_variance=5.0, within_variance=0.0, horizon=100)
    assert result["max_achievable_r_squared"] == pytest.approx(1.0)


def test_required_sample_scales_with_one_over_delta_squared() -> None:
    """The whole trade rests on this: doubling the effect quarters the sample."""
    differences = np.random.default_rng(0).normal(0.0, 1.5, 500)
    small = hs.required_sample_size(differences, delta=-0.05)
    doubled = hs.required_sample_size(differences, delta=-0.10)
    assert small is not None and doubled is not None
    assert doubled == pytest.approx(small / 4.0, rel=1e-9)


def test_required_sample_uses_the_observed_paired_spread() -> None:
    tight = hs.required_sample_size(np.full(100, 0.2) + np.linspace(-0.01, 0.01, 100), delta=-0.1)
    wide = hs.required_sample_size(np.random.default_rng(1).normal(0.0, 3.0, 100), delta=-0.1)
    assert tight is not None and wide is not None
    assert wide > tight


def test_a_zero_delta_has_no_finite_sample_size() -> None:
    assert hs.required_sample_size(np.array([1.0, 2.0, 3.0]), delta=0.0) is None


def test_the_scope_refuses_sealed_and_held_out_seasons() -> None:
    for season in (2025, 2026):
        with pytest.raises(hs.HorizonScopeError, match="development-only"):
            hs.run_scope(seasons=(2022, season))


def test_the_study_is_labelled_exploratory_not_confirmatory() -> None:
    report = hs.render_report(
        {
            "generated_at_utc": "now",
            "status": "EXPLORATORY SCOPING -- not a frozen result",
            "purpose": "scoping",
            "seasons": [2022, 2023, 2024],
            "frozen_elements_reused": {
                "ridge_alpha": 100.0,
                "alpha_source": "ridge freeze manifest",
            },
            "by_horizon": {},
        }
    )
    assert "EXPLORATORY SCOPING" in report
    assert "would still have to be prespecified" in report
    assert "2026 held out" in report
