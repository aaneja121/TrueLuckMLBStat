"""Numeric correctness tests for the interval-bar geometry in
`dashboard/visuals.py`.

These tests verify the actual PIXEL POSITIONS the SVG renders -- not merely
that the expected elements exist -- by parsing the `x`/`cx` attributes out
of the rendered markup and checking them against an independently computed
expected position (`_expected_x`, deliberately re-deriving the linear
transform rather than importing it from `visuals.py`, so a bug in the
transform itself can't hide from its own test).

Regression coverage for the specific bug this test file was added for: two
different SVGs rendered against the SAME `domain` must place `value=0` at
the SAME transformed position, and each SVG's own zero/lower/point/upper
must sit at proportionally correct positions relative to that one shared
domain -- see `TestSharedDomainAcrossRows` and `TestFourSyntheticExamples`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from visuals import (
    compute_interval_domain,
    render_demo_field_svg,
    render_interval_bar_svg,
    render_simulator_field_svg,
)

COMPACT_WIDTH, COMPACT_MARGIN = 220, 10


def _expected_x(value: float, domain: tuple[float, float], *, width: int, margin: int) -> float:
    """Independently re-derives the linear map `render_interval_bar_svg`
    is supposed to implement: `margin + (value - domain_min) / span * inner_w`.
    """
    domain_min, domain_max = domain
    inner_w = width - 2 * margin
    return margin + (value - domain_min) / (domain_max - domain_min) * inner_w


def _parse_positions(svg: str) -> dict[str, float]:
    zero_x = float(re.search(r'interval-zero-line" x1="([\d.-]+)"', svg).group(1))
    lower_x = float(re.search(r'interval-range[^"]*" x1="([\d.-]+)"', svg).group(1))
    upper_x = float(
        re.search(r'interval-range[^"]*" x1="[\d.-]+" y1="[\d.-]+" x2="([\d.-]+)"', svg).group(1)
    )
    point_x = float(re.search(r'interval-point[^"]*" cx="([\d.-]+)"', svg).group(1))
    return {"zero": zero_x, "lower": lower_x, "point": point_x, "upper": upper_x}


class TestComputeIntervalDomain:
    def test_domain_always_includes_zero(self) -> None:
        lo, hi = compute_interval_domain([(3.0, 9.0)])
        assert lo < 0.0 < hi

    def test_domain_spans_min_lower_and_max_upper_across_all_intervals(self) -> None:
        intervals = [(-10.0, -4.0), (3.0, 9.0), (-3.0, 8.0)]
        lo, hi = compute_interval_domain(intervals)
        # Padded outward, so strictly beyond the raw extent, on the correct sides.
        assert lo < -10.0
        assert hi > 9.0

    def test_domain_is_symmetric_padding_fraction_of_span(self) -> None:
        intervals = [(-10.0, -4.0), (3.0, 9.0)]
        lo, hi = compute_interval_domain(intervals, pad_fraction=0.1)
        raw_lo, raw_hi = -10.0, 9.0
        span = raw_hi - raw_lo
        assert lo == pytest.approx(raw_lo - span * 0.1)
        assert hi == pytest.approx(raw_hi + span * 0.1)


class TestFourSyntheticExamples:
    """The exact synthetic examples requested: verify ordering and
    proportional positioning NUMERICALLY, using one shared domain across
    all four (mirroring how the real leaderboard shares one domain across
    every row in the same view).
    """

    EXAMPLES = {
        "negative_ci": {"score": -7.0, "lower": -10.0, "upper": -4.0},
        "zero_score_crosses": {"score": 0.0, "lower": -5.0, "upper": 5.0},
        "positive_ci": {"score": 6.0, "lower": 3.0, "upper": 9.0},
        "asymmetric_crosses": {"score": 2.0, "lower": -3.0, "upper": 8.0},
    }

    @pytest.fixture(scope="class")
    def domain(self) -> tuple[float, float]:
        return compute_interval_domain([(v["lower"], v["upper"]) for v in self.EXAMPLES.values()])

    @pytest.fixture(scope="class")
    def rendered(self, domain: tuple[float, float]) -> dict[str, dict[str, float]]:
        out = {}
        for name, v in self.EXAMPLES.items():
            svg = render_interval_bar_svg(
                point=v["score"], lower=v["lower"], upper=v["upper"], domain=domain, compact=True
            )
            out[name] = _parse_positions(svg)
        return out

    def test_positions_match_independently_computed_transform(
        self, domain: tuple[float, float], rendered: dict[str, dict[str, float]]
    ) -> None:
        for name, v in self.EXAMPLES.items():
            pos = rendered[name]
            assert pos["zero"] == pytest.approx(
                _expected_x(0.0, domain, width=COMPACT_WIDTH, margin=COMPACT_MARGIN), abs=0.1
            )
            assert pos["lower"] == pytest.approx(
                _expected_x(v["lower"], domain, width=COMPACT_WIDTH, margin=COMPACT_MARGIN),
                abs=0.1,
            )
            assert pos["point"] == pytest.approx(
                _expected_x(v["score"], domain, width=COMPACT_WIDTH, margin=COMPACT_MARGIN),
                abs=0.1,
            )
            assert pos["upper"] == pytest.approx(
                _expected_x(v["upper"], domain, width=COMPACT_WIDTH, margin=COMPACT_MARGIN),
                abs=0.1,
            )

    def test_negative_ci_entirely_left_of_zero(self, rendered: dict[str, dict[str, float]]) -> None:
        pos = rendered["negative_ci"]
        assert pos["lower"] < pos["upper"] < pos["zero"]
        assert pos["lower"] < pos["point"] < pos["upper"]

    def test_positive_ci_entirely_right_of_zero(
        self, rendered: dict[str, dict[str, float]]
    ) -> None:
        pos = rendered["positive_ci"]
        assert pos["zero"] < pos["lower"] < pos["upper"]
        assert pos["lower"] < pos["point"] < pos["upper"]

    def test_ci_containing_zero_crosses_zero_line(
        self, rendered: dict[str, dict[str, float]]
    ) -> None:
        for name in ("zero_score_crosses", "asymmetric_crosses"):
            pos = rendered[name]
            assert pos["lower"] < pos["zero"] < pos["upper"]

    def test_point_estimate_sits_at_correct_proportional_location_within_ci(
        self, rendered: dict[str, dict[str, float]]
    ) -> None:
        # abs=0.01: the rendered SVG rounds each coordinate to 0.1px, and
        # the narrowest CI here spans only ~25px, so up to ~0.4% fractional
        # error from rounding alone is expected and not a real defect.
        for name, v in self.EXAMPLES.items():
            pos = rendered[name]
            value_fraction = (v["score"] - v["lower"]) / (v["upper"] - v["lower"])
            pixel_fraction = (pos["point"] - pos["lower"]) / (pos["upper"] - pos["lower"])
            assert pixel_fraction == pytest.approx(value_fraction, abs=0.01)

    def test_relative_distances_are_proportional_to_underlying_values(
        self, rendered: dict[str, dict[str, float]]
    ) -> None:
        """Cross-example check: the negative-CI example's score sits
        exactly halfway through its CI in value-space
        ((-7-(-10))/(-4-(-10)) = 3/6 = 0.5), and the positive-CI example's
        score sits exactly halfway through its CI too ((6-3)/(9-3) = 0.5)
        -- both must land at the SAME pixel-fraction, even though their raw
        numbers (and which side of zero they're on) differ, because both
        examples share one linear domain.
        """
        neg_ci = rendered["negative_ci"]
        pos_ci = rendered["positive_ci"]
        neg_fraction = (neg_ci["point"] - neg_ci["lower"]) / (neg_ci["upper"] - neg_ci["lower"])
        pos_fraction = (pos_ci["point"] - pos_ci["lower"]) / (pos_ci["upper"] - pos_ci["lower"])
        # abs=0.01: same 0.1px-rounding allowance as the test above.
        assert neg_fraction == pytest.approx(pos_fraction, abs=0.01)
        assert neg_fraction == pytest.approx(0.5, abs=0.01)


class TestSharedDomainAcrossRows:
    def test_two_bars_rendered_against_the_same_domain_place_zero_identically(self) -> None:
        domain = compute_interval_domain([(-10.0, -4.0), (3.0, 9.0)])
        svg_a = render_interval_bar_svg(point=-7.0, lower=-10.0, upper=-4.0, domain=domain)
        svg_b = render_interval_bar_svg(point=6.0, lower=3.0, upper=9.0, domain=domain)
        assert _parse_positions(svg_a)["zero"] == _parse_positions(svg_b)["zero"]

    def test_independently_normalized_per_row_domains_would_disagree(self) -> None:
        """Negative control: proves the shared-domain test above is actually
        exercising something real. If each bar were normalized to its OWN
        interval (the bug this whole task is about), their zero positions
        would NOT agree -- confirming the assertion above is a meaningful
        invariant, not a tautology of the rendering code.
        """
        domain_a = compute_interval_domain([(-10.0, -4.0)])
        domain_b = compute_interval_domain([(3.0, 9.0)])
        svg_a = render_interval_bar_svg(point=-7.0, lower=-10.0, upper=-4.0, domain=domain_a)
        svg_b = render_interval_bar_svg(point=6.0, lower=3.0, upper=9.0, domain=domain_b)
        assert _parse_positions(svg_a)["zero"] != _parse_positions(svg_b)["zero"]


class TestDemoFieldSvg:
    """Regression coverage for the SMIL `<animateMotion>` removal (Version
    1.3.0 QA fix): real-browser verification showed `<animateMotion>`
    animates via a transform that never touches `cx`/`cy` and does not
    reliably restart after `fill="freeze"`, so `static/demo.js` now drives
    the ball with `requestAnimationFrame` instead -- these tests lock in
    the markup contract that JS depends on.
    """

    def test_svg_contains_no_smil_animate_motion_element(self) -> None:
        svg = render_demo_field_svg(
            example_id="hard_contact_out",
            spray_angle_deg=-5.9,
            hit_distance_ft=413.0,
            favorable=False,
        )
        assert "<animateMotion" not in svg

    def test_ball_circle_carries_start_control_and_end_coordinates(self) -> None:
        svg = render_demo_field_svg(
            example_id="weak_contact_single",
            spray_angle_deg=46.5,
            hit_distance_ft=171.0,
            favorable=True,
        )
        for attr in (
            "data-start-x",
            "data-start-y",
            "data-control-x",
            "data-control-y",
            "data-end-x",
            "data-end-y",
        ):
            assert re.search(rf'{attr}="[\d.-]+"', svg), f"missing {attr} on the ball circle"

    def test_ball_starts_at_home_plate_matching_the_dashed_preview_path(self) -> None:
        svg = render_demo_field_svg(
            example_id="hard_contact_out",
            spray_angle_deg=0.0,
            hit_distance_ft=300.0,
            favorable=False,
        )
        cx = re.search(r'class="demo-ball [^"]*" r="7" cx="([\d.-]+)"', svg)
        cy = re.search(r'cy="([\d.-]+)"', svg)
        start_x = re.search(r'data-start-x="([\d.-]+)"', svg)
        start_y = re.search(r'data-start-y="([\d.-]+)"', svg)
        path_d = re.search(r'class="demo-ball-path" d="M ([\d.-]+),([\d.-]+)', svg)
        assert cx and cy and start_x and start_y and path_d
        assert cx.group(1) == start_x.group(1) == path_d.group(1)
        assert cy.group(1) == start_y.group(1) == path_d.group(2)

    def test_hard_contact_and_weak_contact_produce_visibly_different_endpoints(self) -> None:
        """The two demo examples must animate to CLEARLY different
        illustrative destinations -- a hard, deep line drive vs. a short,
        shallow bloop -- not the same spot with different labels.
        """
        hard = render_demo_field_svg(
            example_id="hard_contact_out",
            spray_angle_deg=-5.9,
            hit_distance_ft=413.0,
            favorable=False,
        )
        weak = render_demo_field_svg(
            example_id="weak_contact_single",
            spray_angle_deg=46.5,
            hit_distance_ft=171.0,
            favorable=True,
        )
        hard_end_y = float(re.search(r'data-end-y="([\d.-]+)"', hard).group(1))
        weak_end_y = float(re.search(r'data-end-y="([\d.-]+)"', weak).group(1))
        # Smaller y == further from home plate (SVG y grows downward) -- the
        # deep, hard-hit example must land meaningfully closer to the fence.
        assert hard_end_y < weak_end_y - 50

    def test_favorable_flag_selects_the_correct_sign_class(self) -> None:
        favorable_svg = render_demo_field_svg(
            example_id="weak_contact_single",
            spray_angle_deg=46.5,
            hit_distance_ft=171.0,
            favorable=True,
        )
        unfavorable_svg = render_demo_field_svg(
            example_id="hard_contact_out",
            spray_angle_deg=-5.9,
            hit_distance_ft=413.0,
            favorable=False,
        )
        assert 'class="demo-ball demo-ball-favorable"' in favorable_svg
        assert "demo-ball-unfavorable" not in favorable_svg
        assert 'class="demo-ball demo-ball-unfavorable"' in unfavorable_svg
        assert re.search(r'class="demo-ball demo-ball-favorable"\s', unfavorable_svg) is None


class TestSimulatorFieldSvg:
    """`render_simulator_field_svg` (Version 1.3.1 field-view addition) is a
    STATIC skeleton -- `static/demo_simulator.js` owns the ball/path
    coordinates from then on, driven live by slider state. These tests lock
    in the markup contract that script depends on, and confirm this
    function cannot regress `render_demo_field_svg`'s own output (they
    share no code).
    """

    def test_svg_contains_no_smil_animate_motion_element(self) -> None:
        svg = render_simulator_field_svg()
        assert "<animateMotion" not in svg

    def test_ball_and_path_start_as_empty_placeholders_at_home_plate(self) -> None:
        svg = render_simulator_field_svg()
        path_d = re.search(r'class="simulator-field-path"[^>]*\sd="([^"]*)"', svg)
        assert path_d is not None
        assert path_d.group(1) == ""
        ball = re.search(r'class="simulator-field-ball"[^>]*\scx="([\d.-]+)"\scy="([\d.-]+)"', svg)
        assert ball is not None
        home = re.search(r'data-home-x="([\d.-]+)"\sdata-home-y="([\d.-]+)"', svg)
        assert home is not None
        assert ball.group(1) == home.group(1)
        assert ball.group(2) == home.group(2)

    def test_svg_exposes_the_same_coordinate_constants_the_walkthrough_uses(self) -> None:
        # `static/demo_simulator.js` reads these data-* attributes rather
        # than hardcoding a second copy of visuals.py's field constants --
        # this locks the exposed values to the walkthrough's own diagram so
        # the two visuals share one coordinate system.
        simulator_svg = render_simulator_field_svg()
        walkthrough_svg = render_demo_field_svg(
            example_id="hard_contact_out",
            spray_angle_deg=-5.9,
            hit_distance_ft=413.0,
            favorable=False,
        )
        home_x = re.search(r'data-home-x="([\d.-]+)"', simulator_svg).group(1)
        home_y = re.search(r'data-home-y="([\d.-]+)"', simulator_svg).group(1)
        start_x = re.search(r'data-start-x="([\d.-]+)"', walkthrough_svg).group(1)
        start_y = re.search(r'data-start-y="([\d.-]+)"', walkthrough_svg).group(1)
        assert home_x == start_x
        assert home_y == start_y
        assert re.search(r'data-fence-radius="([\d.-]+)"', simulator_svg) is not None
        assert re.search(r'data-max-spray-deg="([\d.-]+)"', simulator_svg) is not None

    def test_arc_height_and_depth_use_different_scalars_in_the_companion_js(self) -> None:
        """This SVG's ball/path are driven entirely by `static/demo_simulator.js`
        (see that module's `computeFieldVisualState`), so the actual numeric
        monotonicity guarantee lives there and was verified via real-browser
        QA, not here. This test locks in the STRUCTURAL contract that bug
        depended on breaking: an earlier version computed arc height as
        `Math.min(homeY, endY) - liftPx`, coupling it to the depth-dependent
        endpoint and (because depth uses a bell curve peaking at 45 degrees)
        making an 80-87 degree popup render with a LOWER arc than a
        45-60 degree hit. The fix anchors arc height purely from `homeY`,
        with `liftPx` a linear (not bell-shaped) function of launch angle --
        this asserts that contract holds in the shipped source, so a future
        edit can't silently reintroduce the coupling.
        """
        js_path = (
            Path(__file__).resolve().parent.parent / "dashboard" / "static" / "demo_simulator.js"
        )
        source = js_path.read_text()

        control_y_line = re.search(r"var controlY = ([^\n;]+);", source)
        assert control_y_line is not None, "expected a single `controlY = ...` assignment"
        assert "endY" not in control_y_line.group(1), (
            "arc height (controlY) must not depend on the depth-derived endpoint -- "
            f"found: {control_y_line.group(1)!r}"
        )
        assert control_y_line.group(1).strip() == "field.homeY - liftPx"

        arc_height_fn = re.search(r"function arcHeightFactor\(laDeg\) \{[^}]*\}", source)
        assert arc_height_fn is not None
        assert "Math.sin" not in arc_height_fn.group(0), (
            "arc height must be a monotonic (not bell-shaped) function of launch angle"
        )

        depth_shape_fn = re.search(r"function laDepthShapeFactor\(laDeg\) \{[^}]*\}", source)
        assert depth_shape_fn is not None
        assert "Math.sin" in depth_shape_fn.group(0), (
            "depth may still use the bell-shaped shaping curve -- only arc height may not"
        )

    def test_aria_label_disclaims_reconstruction_and_names_no_specific_play(self) -> None:
        svg = render_simulator_field_svg()
        assert "not a reconstruction" in svg
        assert "hard_contact_out" not in svg
        assert "weak_contact_single" not in svg

    def test_output_is_deterministic(self) -> None:
        assert render_simulator_field_svg() == render_simulator_field_svg()
