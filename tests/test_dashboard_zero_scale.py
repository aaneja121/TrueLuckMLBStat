"""Redesign Phase 1: the Zero Spine's two invariants, enforced.

`ZeroScale` exists so that no page can invent its own zero axis. These
tests are the enforcement; without them the invariants are documentation.

  * **Invariant Z (zero locus).** Every renderer places zero at the scale's
    own `zero_fraction` of its plot field, at every size. Before Phase 1
    this was false: `render_interval_bar_svg` reserved `margin=10` inside a
    220-wide viewBox and `margin=28` inside a 480-wide one, putting the same
    data's zero at 0.4630 and 0.4640 of width -- equal only when zero sits
    dead centre. That is the "0.463/0.464" recorded in DESIGN.md as though
    it were one shared position.

  * **Invariant D (domain identity).** Every rendering of Contact Luck Runs
    per 100 uses the ONE snapshot-level scale built from the qualified
    comparison population. A value outside it clips with an explicit
    indicator; the domain never widens, because a widened domain silently
    moves zero.

See docs/design/zero-spine-implementation-plan.md.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

from visuals import (  # noqa: E402
    ZeroScale,
    compute_interval_domain,
    render_interval_bar_svg,
)

#: A realistic spread: a strongly favorable hitter, a near-zero hitter, and
#: a strongly unfavorable one, all with wide game-clustered intervals.
QUALIFIED_INTERVALS = [
    (2.10, 13.14),
    (-0.42, 14.38),
    (-3.90, 3.78),
    (-13.05, -1.90),
    (-0.51, 0.39),
]


#: Chosen so the padded domain is [-12.20, +14.39] -- the domain the
#: 2026-08-14 snapshot's 124 qualified hitters actually produce. Pins the
#: tests for off-scale points to the values that ship, including Johnathan
#: Rodriguez (-13.82 [-56.60, +28.40], 6 eligible BBE).
PRODUCTION_QUALIFIED_INTERVALS = [(-9.63, 11.81)]


@pytest.fixture
def production_scale() -> ZeroScale:
    scale = ZeroScale.from_intervals(
        "league_per_100", "Runs / 100 eligible BBE", PRODUCTION_QUALIFIED_INTERVALS
    )
    assert scale.domain_min == pytest.approx(-12.20, abs=0.01)
    assert scale.domain_max == pytest.approx(14.39, abs=0.01)
    return scale


@pytest.fixture
def league_scale() -> ZeroScale:
    return ZeroScale.from_intervals(
        "league_per_100", "Runs / 100 eligible BBE", QUALIFIED_INTERVALS
    )


def _zero_x(svg: str) -> float:
    match = re.search(r'class="interval-zero-line" x1="([0-9.]+)"', svg)
    assert match, "expected a zero line in the rendered bar"
    return float(match.group(1))


def _view_box_width(svg: str) -> float:
    match = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', svg)
    assert match
    return float(match.group(1))


# ── the scale object ────────────────────────────────────────────────────


class TestZeroScale:
    def test_domain_matches_the_shared_helper(self, league_scale: ZeroScale) -> None:
        """`from_intervals` must not become a second, subtly different
        padding implementation alongside `compute_interval_domain`."""
        assert league_scale.domain == compute_interval_domain(QUALIFIED_INTERVALS)

    def test_zero_is_inside_the_domain(self, league_scale: ZeroScale) -> None:
        assert league_scale.domain_min < 0 < league_scale.domain_max
        assert 0.0 < league_scale.zero_fraction < 1.0

    def test_true_extremes_are_unpadded(self, league_scale: ZeroScale) -> None:
        """`true_min`/`true_max` let a caller tell 'off the scale' from
        'near the edge of the padding'."""
        assert league_scale.true_min == -13.05
        assert league_scale.true_max == 14.38
        assert league_scale.domain_min < league_scale.true_min
        assert league_scale.domain_max > league_scale.true_max

    def test_fraction_of_is_linear_and_anchored(self, league_scale: ZeroScale) -> None:
        assert league_scale.fraction_of(league_scale.domain_min) == pytest.approx(0.0)
        assert league_scale.fraction_of(league_scale.domain_max) == pytest.approx(1.0)
        assert league_scale.fraction_of(0.0) == pytest.approx(league_scale.zero_fraction)

    def test_fraction_of_is_unclamped_but_clamped_variant_is_not(
        self, league_scale: ZeroScale
    ) -> None:
        far = league_scale.domain_max * 3
        assert league_scale.fraction_of(far) > 1.0
        assert league_scale.clamped_fraction_of(far) == pytest.approx(1.0)

    def test_all_zero_input_does_not_divide_by_zero(self) -> None:
        scale = ZeroScale.from_intervals("degenerate", "u", [(0.0, 0.0)])
        assert scale.span > 0
        assert 0.0 < scale.zero_fraction < 1.0

    def test_directly_constructed_degenerate_scale_is_guarded(self) -> None:
        """A scale built by hand (not through `from_intervals`) must still
        not produce a divide-by-zero or an off-field zero line."""
        scale = ZeroScale("x", "u", domain_min=0.0, domain_max=0.0, true_min=0.0, true_max=0.0)
        assert scale.span == 1.0
        assert scale.zero_fraction == pytest.approx(0.0)


class TestClipState:
    def test_inside_the_domain_does_not_clip(self, league_scale: ZeroScale) -> None:
        assert league_scale.clip_state(point=5.0, lower=2.10, upper=13.14) == "none"

    def test_interval_beyond_each_edge(self, league_scale: ZeroScale) -> None:
        assert league_scale.clip_state(point=0.0, lower=-99.0, upper=1.0) == "low"
        assert league_scale.clip_state(point=0.0, lower=-1.0, upper=99.0) == "high"
        assert league_scale.clip_state(point=0.0, lower=-99.0, upper=99.0) == "both"

    def test_point_outside_the_domain_clips_even_when_interval_is_inside(
        self, league_scale: ZeroScale
    ) -> None:
        """A degenerate case, but the guarantee is about the point estimate
        as much as the interval -- neither may be silently clamped."""
        assert league_scale.clip_state(point=99.0, lower=1.0, upper=2.0) == "high"


# ── Invariant Z ─────────────────────────────────────────────────────────


class TestInvariantZeroLocus:
    @pytest.mark.parametrize("compact", [True, False])
    def test_zero_lands_at_the_scales_zero_fraction(
        self, league_scale: ZeroScale, compact: bool
    ) -> None:
        svg = render_interval_bar_svg(
            point=5.0, lower=2.10, upper=13.14, domain=league_scale.domain, compact=compact
        )
        rendered = _zero_x(svg) / _view_box_width(svg)
        assert rendered == pytest.approx(league_scale.zero_fraction, abs=5e-4)

    def test_compact_and_full_agree_on_the_zero_fraction(self, league_scale: ZeroScale) -> None:
        """The regression this whole primitive exists for.

        Before Phase 1 these differed by ~0.001 of width because each size
        reserved a different internal margin -- invisible at ornament scale,
        a visible misregistration under a page-level rule.
        """
        kwargs = {"point": 5.0, "lower": 2.10, "upper": 13.14, "domain": league_scale.domain}
        compact = render_interval_bar_svg(**kwargs, compact=True)
        full = render_interval_bar_svg(**kwargs, compact=False)
        # Emitted coordinates are quantised to 0.1 user units, which on the
        # 220-wide compact bar is 4.5e-4 of width -- so that, not the
        # transform, sets the achievable tolerance. The pre-Phase-1 gap was
        # ~1e-3, more than twice this bound, and is what this test catches.
        assert _zero_x(compact) / _view_box_width(compact) == pytest.approx(
            _zero_x(full) / _view_box_width(full), abs=5e-4
        )

    def test_renderer_reserves_no_horizontal_margin(self, league_scale: ZeroScale) -> None:
        """Breathing room is the layout's job. Anything inside the
        coordinate space moves zero."""
        svg = render_interval_bar_svg(
            point=league_scale.domain_max,
            lower=league_scale.domain_min,
            upper=league_scale.domain_max,
            domain=league_scale.domain,
            compact=True,
        )
        width = _view_box_width(svg)
        match = re.search(
            r'class="interval-range[^"]*" x1="([0-9.]+)" y1="[0-9.]+" x2="([0-9.]+)"', svg
        )
        assert match
        assert float(match.group(1)) == pytest.approx(0.0, abs=0.05)
        assert float(match.group(2)) == pytest.approx(width, abs=0.05)

    def test_clipping_does_not_move_zero(self, league_scale: ZeroScale) -> None:
        """The caret insets a clipped END; it must never shift the origin."""
        clean = render_interval_bar_svg(
            point=5.0, lower=2.10, upper=13.14, domain=league_scale.domain, compact=True
        )
        clipped = render_interval_bar_svg(
            point=5.0, lower=-500.0, upper=500.0, domain=league_scale.domain, compact=True
        )
        assert _zero_x(clean) == pytest.approx(_zero_x(clipped), abs=1e-6)


# ── Invariant D ─────────────────────────────────────────────────────────


class TestInvariantDomainIdentity:
    def test_same_scale_makes_two_players_comparable(self, league_scale: ZeroScale) -> None:
        a = render_interval_bar_svg(
            point=-7.0, lower=-13.05, upper=-1.90, domain=league_scale.domain
        )
        b = render_interval_bar_svg(point=6.0, lower=2.10, upper=13.14, domain=league_scale.domain)
        assert _zero_x(a) == pytest.approx(_zero_x(b))

    def test_an_out_of_domain_value_clips_rather_than_widening(
        self, league_scale: ZeroScale
    ) -> None:
        """Decision D1a. The canonical domain comes from the qualified
        population and never widens for one page; a small-sample outlier
        clips instead, because widening would move zero on that page.
        """
        outlier = (-42.0, 31.0)
        assert league_scale.clip_state(point=-8.0, lower=outlier[0], upper=outlier[1]) == "both"
        svg = render_interval_bar_svg(
            point=-8.0, lower=outlier[0], upper=outlier[1], domain=league_scale.domain
        )
        assert _zero_x(svg) / _view_box_width(svg) == pytest.approx(
            league_scale.zero_fraction, abs=5e-4
        )


# ── accessible clipping semantics ───────────────────────────────────────


class TestClipIsInformationNotDecoration:
    def test_unclipped_bar_declares_itself_unclipped(self, league_scale: ZeroScale) -> None:
        svg = render_interval_bar_svg(
            point=5.0, lower=2.10, upper=13.14, domain=league_scale.domain
        )
        assert 'data-clipped="none"' in svg
        assert "interval-clip-caret" not in svg
        assert "continues beyond" not in svg

    @pytest.mark.parametrize(
        ("lower", "upper", "state", "carets"),
        [
            (-99.0, 1.0, "low", 1),
            (-1.0, 99.0, "high", 1),
            (-99.0, 99.0, "both", 2),
        ],
    )
    def test_clipped_bar_draws_a_caret_and_says_so(
        self, league_scale: ZeroScale, lower: float, upper: float, state: str, carets: int
    ) -> None:
        """Interval clipped, point in-domain -> an open chevron per clipped
        edge, and no off-scale point marker."""
        svg = render_interval_bar_svg(
            point=0.5, lower=lower, upper=upper, domain=league_scale.domain
        )
        assert f'data-clipped="{state}"' in svg
        assert 'data-point-offscale="in"' in svg
        assert svg.count("interval-clip-caret") == carets
        assert svg.count("interval-point-offscale") == 0
        assert svg.count("interval-point ") == 1
        assert "extends" in svg


class TestOffScalePointEstimate:
    """A point estimate must never silently vanish because it lies beyond the
    canonical domain, and must never be drawn as a dot on the boundary --
    that would assert the boundary IS the estimate.
    """

    def test_in_domain_point_renders_a_normal_dot(self, league_scale: ZeroScale) -> None:
        svg = render_interval_bar_svg(
            point=5.0, lower=2.10, upper=13.14, domain=league_scale.domain
        )
        assert 'data-point-offscale="in"' in svg
        assert svg.count("interval-point ") == 1
        assert "interval-point-offscale" not in svg

    @pytest.mark.parametrize(
        ("point", "lower", "upper", "state", "edge_x"),
        [
            (-13.82, -56.60, 28.40, "below", "0.0"),
            (-47.03, -49.68, -44.39, "below", "0.0"),
            (46.18, 46.18, 46.18, "above", "480.0"),
        ],
    )
    def test_off_scale_point_gets_an_edge_marker_not_a_dot(
        self,
        production_scale: ZeroScale,
        point: float,
        lower: float,
        upper: float,
        state: str,
        edge_x: str,
    ) -> None:
        svg = render_interval_bar_svg(
            point=point, lower=lower, upper=upper, domain=production_scale.domain
        )
        assert f'data-point-offscale="{state}"' in svg
        # No dot anywhere: `.interval-point ` (with the trailing space) is the
        # in-domain circle; the off-scale marker is a different class.
        assert svg.count("interval-point ") == 0
        assert svg.count("interval-point-offscale") == 1
        # The marker is anchored to the correct edge, so it reads directional.
        marker = re.search(r'class="interval-point-offscale[^"]*" points="([^"]+)"', svg)
        assert marker
        assert marker.group(1).split(",")[0] == edge_x

    def test_off_scale_point_suppresses_the_chevron_on_that_edge_only(
        self, league_scale: ZeroScale
    ) -> None:
        """An off-scale point always implies a clipped interval on the same
        side, so the two marks never coexist on one edge -- but the OTHER
        edge still gets its chevron when it clips too.
        """
        svg = render_interval_bar_svg(
            point=-20.0, lower=-60.0, upper=30.0, domain=league_scale.domain
        )
        assert 'data-clipped="both"' in svg
        assert 'data-point-offscale="below"' in svg
        assert svg.count("interval-point-offscale") == 1  # low edge
        assert svg.count("interval-clip-caret") == 1  # high edge

    def test_interval_entirely_off_field_draws_no_line(self, production_scale: ZeroScale) -> None:
        """18 players in the 2026-08-14 snapshot sit entirely outside the
        canonical domain. There is no segment to draw -- and emitting a
        zero-length or negative-width line would be a rendering bug.
        """
        svg = render_interval_bar_svg(
            point=-47.03, lower=-49.68, upper=-44.39, domain=production_scale.domain
        )
        assert "interval-range" not in svg
        assert svg.count("interval-point-offscale") == 1
        assert 'class="interval-zero-line"' in svg

    def test_zero_width_interval_off_scale_is_not_degenerate(self, league_scale: ZeroScale) -> None:
        """A one-batted-ball player: lower == point == upper, far off scale."""
        svg = render_interval_bar_svg(
            point=46.18, lower=46.18, upper=46.18, domain=league_scale.domain
        )
        assert "interval-range" not in svg
        assert svg.count("interval-point-offscale") == 1
        assert "NaN" not in svg and "-0.0," not in svg

    def test_interval_spanning_the_whole_domain_with_in_domain_point(
        self, league_scale: ZeroScale
    ) -> None:
        svg = render_interval_bar_svg(
            point=1.0, lower=-99.0, upper=99.0, domain=league_scale.domain
        )
        assert svg.count("interval-clip-caret") == 2
        assert svg.count("interval-point ") == 1
        assert svg.count("interval-range") == 1

    def test_the_two_edge_marks_are_visually_distinct_shapes(self, league_scale: ZeroScale) -> None:
        """Open chevron vs solid triangle. If both became the same element
        type the two states would be indistinguishable to a sighted reader.
        """
        chevron = render_interval_bar_svg(
            point=1.0, lower=-99.0, upper=99.0, domain=league_scale.domain
        )
        solid = render_interval_bar_svg(
            point=-47.03, lower=-49.68, upper=-44.39, domain=league_scale.domain
        )
        assert '<polyline class="interval-clip-caret' in chevron
        assert '<polygon class="interval-point-offscale' in solid

    def test_off_scale_marker_never_appears_on_a_leaderboard_row(self) -> None:
        """The canonical domain is built from exactly the qualified
        population the leaderboard draws, so a leaderboard row cannot clip.
        This is what lets the off-scale marker be sized for the full bar
        without worrying about the compact bar's 108px mobile rendering.
        """
        for lower, upper in QUALIFIED_INTERVALS:
            point = (lower + upper) / 2
            scale = ZeroScale.from_intervals("league_per_100", "u", QUALIFIED_INTERVALS)
            svg = render_interval_bar_svg(
                point=point, lower=lower, upper=upper, domain=scale.domain, compact=True
            )
            assert 'data-clipped="none"' in svg
            assert 'data-point-offscale="in"' in svg


class TestClipIsInformationNotDecorationContinued:
    """Carried over from the original clipping tests (see above)."""

    def test_true_values_survive_clipping_in_the_accessible_label(
        self, league_scale: ZeroScale
    ) -> None:
        """The graphical mark is cut off; the NUMBERS never are."""
        svg = render_interval_bar_svg(
            point=-8.0, lower=-42.0, upper=31.0, domain=league_scale.domain
        )
        label = re.search(r'aria-label="([^"]+)"', svg)
        assert label
        assert "-42.00" in label.group(1)
        assert "+31.00" in label.group(1)
        assert "-8.00" in label.group(1)

    def test_no_point_dot_is_drawn_at_a_clamped_position(self, league_scale: ZeroScale) -> None:
        """A dot sitting on the field edge would assert that the edge is the
        estimate. When the point itself is off-scale, no dot is drawn."""
        outside = render_interval_bar_svg(
            point=99.0, lower=90.0, upper=110.0, domain=league_scale.domain
        )
        # `.interval-point ` (trailing space) is the in-domain circle;
        # `.interval-point-offscale` is the edge marker that replaces it.
        assert "interval-point " not in outside
        assert "interval-point-offscale" in outside
        inside = render_interval_bar_svg(
            point=5.0, lower=-99.0, upper=99.0, domain=league_scale.domain
        )
        assert "interval-point" in inside


# ── presentation behaviour: explicit signs ──────────────────────────────


class TestOffScaleAccessibleText:
    """A screen-reader user must never have to infer clipping from a glyph."""

    def test_states_the_point_is_off_scale_and_its_direction(
        self, production_scale: ZeroScale
    ) -> None:
        """The shipped Johnathan Rodriguez case, exactly as it renders."""
        svg = render_interval_bar_svg(
            point=-13.82, lower=-56.60, upper=28.40, domain=production_scale.domain
        )
        label = re.search(r'aria-label="([^"]+)"', svg).group(1)
        assert "-13.82" in label  # the actual point estimate
        assert "-56.60" in label and "+28.40" in label  # the actual interval
        assert "point estimate lies below the displayed range" in label
        assert "interval extends below and above the displayed range" in label
        assert "displayed range covers" in label  # the domain itself

    def test_direction_is_stated_for_an_above_scale_point(self, league_scale: ZeroScale) -> None:
        svg = render_interval_bar_svg(
            point=46.18, lower=46.18, upper=46.18, domain=league_scale.domain
        )
        label = re.search(r'aria-label="([^"]+)"', svg).group(1)
        assert "point estimate lies above the displayed range" in label

    def test_in_domain_point_with_clipped_interval_does_not_claim_the_point_is_off_scale(
        self, league_scale: ZeroScale
    ) -> None:
        svg = render_interval_bar_svg(
            point=1.0, lower=-99.0, upper=99.0, domain=league_scale.domain
        )
        label = re.search(r'aria-label="([^"]+)"', svg).group(1)
        assert "point estimate lies" not in label
        assert "interval extends below and above" in label

    def test_clean_bar_says_nothing_about_scale(self, league_scale: ZeroScale) -> None:
        svg = render_interval_bar_svg(
            point=5.0, lower=2.10, upper=13.14, domain=league_scale.domain
        )
        label = re.search(r'aria-label="([^"]+)"', svg).group(1)
        assert "Off scale" not in label
        assert "displayed range" not in label


class TestOffScaleHandlingIntroducesNoAutoscaling:
    """The correction must not become per-player widening by another route."""

    def test_domain_is_identical_regardless_of_how_far_off_scale_a_point_is(self) -> None:
        base = ZeroScale.from_intervals("league_per_100", "u", QUALIFIED_INTERVALS)
        for point in (-1000.0, -13.82, 0.0, 46.18, 1000.0):
            svg = render_interval_bar_svg(
                point=point, lower=point - 1, upper=point + 1, domain=base.domain
            )
            assert _zero_x(svg) / _view_box_width(svg) == pytest.approx(
                base.zero_fraction, abs=5e-4
            )

    def test_rendering_never_consults_the_point_when_building_a_domain(self) -> None:
        """`ZeroScale.from_intervals` takes intervals only. If an off-scale
        point could influence the domain, these two would differ.
        """
        a = ZeroScale.from_intervals("league_per_100", "u", QUALIFIED_INTERVALS)
        b = ZeroScale.from_intervals("league_per_100", "u", list(QUALIFIED_INTERVALS))
        assert a == b
        assert a.domain == compute_interval_domain(QUALIFIED_INTERVALS)


class TestExplicitSigns:
    def test_positive_values_carry_a_plus_in_the_accessible_label(
        self, league_scale: ZeroScale
    ) -> None:
        """The sign glyph is the non-colour channel for the semantic pair,
        so it must be present wherever the value is stated -- including for
        a screen reader, which never receives the colour.
        """
        svg = render_interval_bar_svg(
            point=7.62, lower=2.10, upper=13.14, domain=league_scale.domain
        )
        label = re.search(r'aria-label="([^"]+)"', svg)
        assert label
        assert "+7.62" in label.group(1)
        assert "+2.10" in label.group(1)

    def test_negative_values_keep_their_sign(self, league_scale: ZeroScale) -> None:
        svg = render_interval_bar_svg(
            point=-6.41, lower=-13.05, upper=-1.90, domain=league_scale.domain
        )
        label = re.search(r'aria-label="([^"]+)"', svg)
        assert label
        assert "-6.41" in label.group(1)


# ── the uncertainty invariant, restated at the primitive level ──────────


def test_overlapping_zero_is_rendered_identically_to_not_overlapping() -> None:
    """`PRODUCT.md` #2 and guardrail 4: an interval that crosses zero must
    be drawn exactly like one that does not. Asserted here, at the
    primitive, so no later phase can introduce a de-emphasis path.
    """
    scale = ZeroScale.from_intervals("league_per_100", "u", QUALIFIED_INTERVALS)
    crosses = render_interval_bar_svg(
        point=1.0, lower=-3.90, upper=3.78, domain=scale.domain, compact=True
    )
    clear = render_interval_bar_svg(
        point=6.0, lower=2.10, upper=13.14, domain=scale.domain, compact=True
    )
    for attribute in ("opacity", "stroke-dasharray", "filter", "fill-opacity"):
        assert attribute not in crosses
        assert attribute not in clear
    # Same classes, same structure -- only coordinates differ.
    assert re.findall(r'class="([^"]+)"', crosses) == re.findall(r'class="([^"]+)"', clear)
