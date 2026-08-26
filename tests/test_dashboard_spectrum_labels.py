"""Tests for `dashboard/spectrum_labels.py`'s deterministic Luck Spectrum
label-selection algorithm -- see that module's own docstring for the exact
rule. These are pure data-in/data-out tests (no rendering, no HTML/DOM), so
they cover the algorithm's actual decision logic directly rather than
inferring it from rendered markup.
"""

from __future__ import annotations

from spectrum_labels import label_tier, select_spectrum_labels


class TestWidelySeparatedValues:
    def test_labels_top_two_when_max_labels_is_two(self) -> None:
        result = select_spectrum_labels([8.0, 5.0, 2.0], max_labels=2)
        assert result == [0, 1, None]

    def test_labels_only_the_extreme_when_max_labels_is_one(self) -> None:
        result = select_spectrum_labels([8.0, 5.0, 2.0], max_labels=1)
        assert result == [0, None, None]


class TestClusteredValues:
    def test_labels_only_the_most_extreme_of_a_tight_cluster(self) -> None:
        # half_range = 1.59, 12% threshold = ~0.19 -- both 1.57 and 1.55
        # fall well under that distance from 1.59.
        result = select_spectrum_labels([1.59, 1.57, 1.55], max_labels=2)
        assert result == [0, None, None]

    def test_unordered_input_still_finds_the_extreme(self) -> None:
        result = select_spectrum_labels([1.55, 1.59, 1.57], max_labels=2)
        assert result == [None, 0, None]


class TestSinglePoint:
    def test_single_point_is_always_labeled(self) -> None:
        assert select_spectrum_labels([3.14], max_labels=2) == [0]

    def test_single_point_is_labeled_even_with_max_labels_one(self) -> None:
        assert select_spectrum_labels([-3.14], max_labels=1) == [0]


class TestEmptyOrNoCandidates:
    def test_empty_values_returns_empty(self) -> None:
        assert select_spectrum_labels([], max_labels=2) == []

    def test_max_labels_zero_labels_nothing(self) -> None:
        assert select_spectrum_labels([8.0, 5.0], max_labels=0) == [None, None]

    def test_negative_max_labels_labels_nothing(self) -> None:
        assert select_spectrum_labels([8.0, 5.0], max_labels=-1) == [None, None]


class TestNegativeValues:
    def test_applies_symmetrically_on_the_unfavorable_side(self) -> None:
        # Real homepage-shaped example: -6.41, -4.47, -4.28. half_range =
        # 6.41, 12% threshold ~= 0.77. -4.47 is 1.94 away from -6.41 (labeled);
        # -4.28 is only 0.19 away from -4.47, but max_labels=2 is already used.
        result = select_spectrum_labels([-6.41, -4.47, -4.28], max_labels=2)
        assert result == [0, 1, None]

    def test_clustered_negative_values_label_only_the_extreme(self) -> None:
        result = select_spectrum_labels([-1.55, -1.54], max_labels=2)
        assert result == [0, None]


class TestZeroHalfRange:
    def test_all_zero_values_label_only_one(self) -> None:
        # No meaningful separation is possible -- labeling several "+0.00"
        # points would be noise, not information.
        result = select_spectrum_labels([0.0, 0.0, 0.0], max_labels=2)
        assert result == [0, None, None]


class TestMinSeparationFractionParameter:
    def test_looser_threshold_admits_a_second_label(self) -> None:
        # 1.57 is ~1.3% below 1.59 -- fails the default 12% threshold but
        # passes a deliberately loose 1% one.
        result = select_spectrum_labels([1.59, 1.57], max_labels=2, min_separation_fraction=0.01)
        assert result == [0, 1]

    def test_stricter_threshold_can_suppress_the_second_label(self) -> None:
        result = select_spectrum_labels([8.0, 5.0], max_labels=2, min_separation_fraction=0.9)
        assert result == [0, None]


class TestLabelTier:
    def test_rank_zero_is_primary(self) -> None:
        assert label_tier(0) == "primary"

    def test_other_ranks_are_secondary(self) -> None:
        assert label_tier(1) == "secondary"
        assert label_tier(2) == "secondary"

    def test_none_is_unlabeled(self) -> None:
        assert label_tier(None) == "unlabeled"
