"""Contact Luck v1.2: hand-rolled inline SVG for the interval bar and the
season-to-date trend chart.

Rendered at BUILD TIME from already-computed snapshot values (never
recomputes a score/interval) so the site needs no client-side charting
dependency. Colors are applied via CSS classes only (see `static/style.css`
for the actual hex values, drawn from the validated diverging blue/red pair)
-- this module never inlines a hex color, so light/dark theming lives in one
place.

Direction (blue for a non-negative point estimate, red for negative) is
based on the SIGN OF THE POINT ESTIMATE ONLY, applied identically whether or
not the 95% interval crosses zero. Per the Version 1.2 spec: an interval
that overlaps zero must never be rendered as faded, muted, or otherwise
visually different from one that doesn't -- overlap-with-zero is not a
good/bad or confident/unconfident signal here, so this module has no code
path that varies opacity, size, or style by `interval_interpretation`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from content import TrendPoint

__all__ = [
    "compute_interval_domain",
    "render_demo_field_svg",
    "render_interval_bar_svg",
    "render_trend_chart_svg",
]


def compute_interval_domain(
    intervals: Sequence[tuple[float, float]], *, pad_fraction: float = 0.12
) -> tuple[float, float]:
    """A shared x-axis domain for a set of (lower, upper) intervals, always
    including zero and padded a bit so bars/markers near the edge aren't
    clipped.
    """
    lows = [lo for lo, _ in intervals] + [0.0]
    highs = [hi for _, hi in intervals] + [0.0]
    lo, hi = min(lows), max(highs)
    span = hi - lo or 1.0
    pad = span * pad_fraction
    return lo - pad, hi + pad


def render_interval_bar_svg(
    *,
    point: float,
    lower: float,
    upper: float,
    domain: tuple[float, float],
    compact: bool = False,
) -> str:
    """A horizontal point-estimate-plus-95%-interval bar with a visible zero
    line. `compact=True` is the small leaderboard-row size; `compact=False`
    is the larger player-detail-page size with an axis label under the zero
    line. Exact values are always in the `<title>`/`aria-label`, never ONLY
    in the title -- the point, the interval line, and the zero line are all
    drawn directly on the bar itself.
    """
    width, height = (220, 32) if compact else (480, 72)
    margin = 10 if compact else 28
    inner_w = width - 2 * margin
    domain_min, domain_max = domain
    if domain_max <= domain_min:
        domain_max = domain_min + 1.0

    def x(value: float) -> float:
        return margin + (value - domain_min) / (domain_max - domain_min) * inner_w

    mid_y = height / 2
    zero_x = x(0.0)
    lower_x = x(max(lower, domain_min))
    upper_x = x(min(upper, domain_max))
    point_x = x(min(max(point, domain_min), domain_max))
    sign_class = "interval-positive" if point >= 0 else "interval-negative"
    size_class = "interval-bar-compact" if compact else "interval-bar-full"

    parts = [
        f'<svg class="interval-bar {size_class}" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="xMidYMid meet" role="img" aria-label="Point estimate '
        f"{point:.2f} runs per 100 eligible batted balls; 95 percent interval "
        f'{lower:.2f} to {upper:.2f}">',
        f"<title>{point:.2f} runs/100 (95% interval: {lower:.2f} to {upper:.2f})</title>",
        f'<line class="interval-zero-line" x1="{zero_x:.1f}" y1="4" '
        f'x2="{zero_x:.1f}" y2="{height - 4}"></line>',
        f'<line class="interval-range {sign_class}" x1="{lower_x:.1f}" y1="{mid_y:.1f}" '
        f'x2="{upper_x:.1f}" y2="{mid_y:.1f}"></line>',
        f'<circle class="interval-point {sign_class}" cx="{point_x:.1f}" cy="{mid_y:.1f}" '
        f'r="{4 if compact else 6}"></circle>',
    ]
    if not compact:
        parts.append(
            f'<text class="interval-axis-label" x="{zero_x:.1f}" y="{height - 6:.1f}" '
            f'text-anchor="middle">0</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_trend_chart_svg(
    points: Sequence[TrendPoint], *, width: int = 640, height: int = 220
) -> str:
    """A line (point estimates) plus a shaded band (95% interval) across
    stored historical snapshots only. X positions are TRUE calendar
    positions (not evenly-spaced ordinal slots) -- a gap like the missing
    2026-08-07 snapshot shows up as a longer segment between 08-06 and
    08-08, never as a fabricated point. Each marker's `<title>` and the
    date label under it show the real snapshot date, and a
    `retrospective_backfill` point gets a distinct marker class
    (`trend-point-backfill`) so it can never be mistaken for a genuine
    contemporaneous observation.
    """
    if not points:
        return '<p class="trend-empty">No historical snapshots available yet.</p>'
    if len(points) == 1:
        only = points[0]
        return (
            '<p class="trend-empty">Only one snapshot available so far '
            f"({only.data_through_date}) -- a trend needs at least two.</p>"
        )

    dates = [date.fromisoformat(p.data_through_date) for p in points]
    x_min, x_max = min(dates), max(dates)
    x_span = (x_max - x_min).days or 1

    values = [p.contact_luck_runs_per_100 for p in points]
    lowers = [p.lower_95_interval for p in points]
    uppers = [p.upper_95_interval for p in points]
    y_min = min(lowers + [0.0])
    y_max = max(uppers + [0.0])
    y_span = (y_max - y_min) or 1.0
    y_pad = y_span * 0.12
    y_min -= y_pad
    y_max += y_pad
    y_span = y_max - y_min

    margin_left, margin_right, margin_top, margin_bottom = 44, 16, 16, 28
    inner_w = width - margin_left - margin_right
    inner_h = height - margin_top - margin_bottom

    def xpix(d: date) -> float:
        return margin_left + (d - x_min).days / x_span * inner_w

    def ypix(value: float) -> float:
        return margin_top + (1 - (value - y_min) / y_span) * inner_h

    zero_y = ypix(0.0)
    band_upper = " ".join(
        f"{xpix(d):.1f},{ypix(u):.1f}" for d, u in zip(dates, uppers, strict=True)
    )
    band_lower = " ".join(
        f"{xpix(d):.1f},{ypix(low):.1f}"
        for d, low in zip(reversed(dates), reversed(lowers), strict=True)
    )
    line_pts = " ".join(f"{xpix(d):.1f},{ypix(v):.1f}" for d, v in zip(dates, values, strict=True))

    parts = [
        f'<svg class="trend-chart" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="xMidYMid meet" role="img" aria-label="Contact Luck Runs '
        f'per 100 trend across {len(points)} stored snapshots">',
        f'<line class="trend-zero-line" x1="{margin_left}" y1="{zero_y:.1f}" '
        f'x2="{width - margin_right}" y2="{zero_y:.1f}"></line>',
        f'<text class="interval-axis-label" x="{margin_left - 6}" y="{zero_y:.1f}" '
        f'text-anchor="end" dominant-baseline="middle">0</text>',
        f'<polygon class="trend-band" points="{band_upper} {band_lower}"></polygon>',
        f'<polyline class="trend-line" points="{line_pts}"></polyline>',
    ]
    for d, v, point in zip(dates, values, points, strict=True):
        sign_class = "interval-positive" if v >= 0 else "interval-negative"
        marker_class = (
            "trend-point-backfill"
            if point.snapshot_type == "retrospective_backfill"
            else "trend-point"
        )
        cx, cy = xpix(d), ypix(v)
        parts.append(
            f'<circle class="{marker_class} {sign_class}" cx="{cx:.1f}" cy="{cy:.1f}" r="5">'
            f"<title>{point.data_through_date}: {v:.2f} runs/100 (95% interval: "
            f"{point.lower_95_interval:.2f} to {point.upper_95_interval:.2f}); "
            f"{point.eligible_batted_balls} eligible BBE, {point.qualification_status}</title>"
            f"</circle>"
        )
        parts.append(
            f'<text class="trend-date-label" x="{cx:.1f}" y="{height - margin_bottom + 16}" '
            f'text-anchor="middle">{d.strftime("%b %-d")}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


#: Version 1.3 demo field diagram: a deliberately schematic, illustrative
#: fan shape (foul lines + an arc standing in for "the outfield wall"), NOT
#: a scaled reconstruction of any real park's geometry, real ball
#: trajectory physics, or real/assumed defender positioning -- see
#: CLAUDE.md's demo constraints #9/#10. The template pairs this SVG with an
#: explicit "illustrative, not a reconstruction" caption; this module has no
#: code path that claims otherwise (no fielder markers, no distance-to-wall
#: labels, no physics-derived hang time).
_DEMO_FIELD_HOME_X = 150.0
_DEMO_FIELD_HOME_Y = 250.0
_DEMO_FIELD_FENCE_RADIUS = 190.0
_DEMO_FIELD_MAX_DISTANCE_FT = 420.0
_DEMO_FIELD_MAX_SPRAY_DEG = 45.0


def _demo_ball_endpoint(spray_angle_deg: float, hit_distance_ft: float) -> tuple[float, float]:
    clamped_spray = max(-_DEMO_FIELD_MAX_SPRAY_DEG, min(_DEMO_FIELD_MAX_SPRAY_DEG, spray_angle_deg))
    clamped_distance = max(0.0, min(_DEMO_FIELD_MAX_DISTANCE_FT, hit_distance_ft))
    radius_px = clamped_distance / _DEMO_FIELD_MAX_DISTANCE_FT * _DEMO_FIELD_FENCE_RADIUS
    angle_rad = math.radians(clamped_spray)
    x = _DEMO_FIELD_HOME_X + radius_px * math.sin(angle_rad)
    y = _DEMO_FIELD_HOME_Y - radius_px * math.cos(angle_rad)
    return x, y


def render_demo_field_svg(
    *,
    example_id: str,
    spray_angle_deg: float,
    hit_distance_ft: float,
    favorable: bool,
    width: int = 300,
    height: int = 280,
) -> str:
    """An illustrative field diagram with a ball-flight path for the `/demo/`
    page.

    The ball's motion is driven entirely by `static/demo.js` (a small
    `requestAnimationFrame` loop interpolating `cx`/`cy` along the same
    quadratic curve drawn below), not SVG SMIL (`<animateMotion>`) -- an
    earlier version used `<animateMotion>`, but real-browser verification
    (see the task that replaced it) showed its `fill="freeze"` end state
    does not reliably restart on a second `beginElement()` call, and its
    motion is applied as an independent transform that never touches
    `cx`/`cy`, making the element's rendered position impossible to
    verify/reset from the attributes alone. The circle's `data-start-*`/
    `data-control-*`/`data-end-*` attributes are the SAME coordinates used
    to draw the dashed preview path below (`example_id` is no longer
    embedded in any element id -- there is nothing left to target) --
    single source of truth for both the static preview line and the actual
    animation, so they can never drift apart. A reduced-motion viewer is
    placed directly at `data-end-x`/`data-end-y` with no animation at all.

    `favorable` (whether this example's `contact_luck_runs >= 0`) becomes
    a `demo-ball-favorable`/`demo-ball-unfavorable` class on the ball
    circle -- see `static/style.css`, which gates the SIGNED fill color
    behind the SAME `.demo-card.demo-reality-revealed` ancestor class
    `static/demo.js` already adds/removes at Stage 3 ("Reality") for the
    probability-row highlight (see `render_demo_field_svg`'s caller and
    that CSS rule's docstring). This function needs no JS changes and adds
    none here -- the class is present in the markup from first paint (the
    same pattern already used for the probability-row observed-outcome
    highlight), but has NO visual effect until that ancestor class is
    added, so it never leaks the outcome's favorable/unfavorable sign
    during Stages 1-2. This is Contact Luck's sign (favorable/unfavorable
    relative to expectation), NOT the raw hit/out outcome class -- a future
    example with an unfavorable hit or a mildly unfavorable out must still
    color correctly by this same rule, since it only ever looks at the
    sign of `contact_luck_runs`.
    """
    home_x, home_y = _DEMO_FIELD_HOME_X, _DEMO_FIELD_HOME_Y
    end_x, end_y = _demo_ball_endpoint(spray_angle_deg, hit_distance_ft)
    control_x = (home_x + end_x) / 2
    control_y = min(home_y, end_y) - 60
    path_d = (
        f"M {home_x:.1f},{home_y:.1f} Q {control_x:.1f},{control_y:.1f} {end_x:.1f},{end_y:.1f}"
    )

    left_foul_x = home_x + _DEMO_FIELD_FENCE_RADIUS * math.sin(
        math.radians(-_DEMO_FIELD_MAX_SPRAY_DEG)
    )
    left_foul_y = home_y - _DEMO_FIELD_FENCE_RADIUS * math.cos(
        math.radians(-_DEMO_FIELD_MAX_SPRAY_DEG)
    )
    right_foul_x = home_x + _DEMO_FIELD_FENCE_RADIUS * math.sin(
        math.radians(_DEMO_FIELD_MAX_SPRAY_DEG)
    )
    right_foul_y = home_y - _DEMO_FIELD_FENCE_RADIUS * math.cos(
        math.radians(_DEMO_FIELD_MAX_SPRAY_DEG)
    )

    diamond_half = 16.0
    infield_pts = (
        f"{home_x:.1f},{home_y - diamond_half:.1f} "
        f"{home_x + diamond_half:.1f},{home_y - 2 * diamond_half:.1f} "
        f"{home_x:.1f},{home_y - 3 * diamond_half:.1f} "
        f"{home_x - diamond_half:.1f},{home_y - 2 * diamond_half:.1f}"
    )

    sign_class = "demo-ball-favorable" if favorable else "demo-ball-unfavorable"

    parts = [
        f'<svg class="demo-field" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'aria-label="Illustrative field diagram (not a reconstruction of the actual play)" '
        f'data-example-id="{example_id}">',
        f'<path class="demo-field-fence" d="M {left_foul_x:.1f},{left_foul_y:.1f} '
        f"A {_DEMO_FIELD_FENCE_RADIUS:.0f},{_DEMO_FIELD_FENCE_RADIUS:.0f} 0 0 1 "
        f'{right_foul_x:.1f},{right_foul_y:.1f}"></path>',
        f'<line class="demo-field-foul-line" x1="{home_x:.1f}" y1="{home_y:.1f}" '
        f'x2="{left_foul_x:.1f}" y2="{left_foul_y:.1f}"></line>',
        f'<line class="demo-field-foul-line" x1="{home_x:.1f}" y1="{home_y:.1f}" '
        f'x2="{right_foul_x:.1f}" y2="{right_foul_y:.1f}"></line>',
        f'<polygon class="demo-field-infield" points="{infield_pts}"></polygon>',
        f'<path class="demo-ball-path" d="{path_d}"></path>',
        f'<circle class="demo-ball {sign_class}" r="7" cx="{home_x:.1f}" cy="{home_y:.1f}" '
        f'data-start-x="{home_x:.1f}" data-start-y="{home_y:.1f}" '
        f'data-control-x="{control_x:.1f}" data-control-y="{control_y:.1f}" '
        f'data-end-x="{end_x:.1f}" data-end-y="{end_y:.1f}"></circle>',
        "</svg>",
    ]
    return "".join(parts)
