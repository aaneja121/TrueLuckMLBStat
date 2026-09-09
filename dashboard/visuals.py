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
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from content import TrendPoint

__all__ = [
    "TrendFigure",
    "ZeroScale",
    "build_trend_figure",
    "compute_interval_domain",
    "render_demo_field_svg",
    "render_interval_bar_svg",
    "render_simulator_field_svg",
]

#: Redesign Phase 1 (Zero Spine): every zero-centred instrument in the
#: product registers zero at the SAME fraction of its own plot field, so a
#: single page-level rule can descend through figures whose domains differ
#: and always mean "zero". See `docs/design/zero-spine-implementation-plan.md`
#: § Invariant Z. `CL_ZERO_FRACTION_PROPERTY` is the CSS custom property the
#: build writes onto `<html>` once, from the canonical `league_per_100`
#: scale, so CSS and Python can never disagree about where zero is.
CL_ZERO_FRACTION_PROPERTY = "--cl-zero"

ClipState = Literal["none", "low", "high", "both"]
PointState = Literal["in", "below", "above"]


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


@dataclass(frozen=True)
class ZeroScale:
    """The canonical zero-centred quantitative scale for one named quantity.

    Built ONCE per build and threaded to every renderer, so no page can
    invent its own axis. The product has exactly FOUR of these --
    `league_per_100`, `run_value`, `component_per_100`, and
    `pitcher_cumulative_runs`; adding a fifth is a design review, not a code
    change.

    The fourth was that review, made deliberately and written down rather
    than slipped in: the pitcher surface ranks a CUMULATIVE RUN TOTAL, which
    no existing scale measures, so drawing it on `league_per_100` would put
    a +20-run season off the end of a rate axis and drawing it on
    `run_value` would put a season total on a single-play axis. The full
    argument, and the price the exception pays in the template, is at its
    construction site in `build.py`.

    Two invariants this class exists to make mechanically true:

    * **Invariant Z (zero locus).** `zero_fraction` is where zero sits in a
      plot field, and EVERY renderer places zero there exactly -- so a rule
      drawn at that fraction registers against every mark on the page.
      Renderers therefore contribute no horizontal margin of their own
      (see `render_interval_bar_svg`); breathing room is the layout's job.
    * **Invariant D (domain identity).** Every rendering of Contact Luck
      Runs per 100 uses the ONE snapshot-level `league_per_100` scale, built
      from the qualified comparison population. No per-page recomputation,
      no per-chart autoscaling. A value outside the domain CLIPS with an
      explicit indicator (`clip_state`); the domain never widens to swallow
      it, because a widened domain silently moves zero.

    `true_min`/`true_max` are the unpadded observed extremes the scale was
    built from, kept so a caller can tell "this is off the scale" from
    "this is near the edge of the padding".
    """

    name: str
    unit_label: str
    domain_min: float
    domain_max: float
    true_min: float
    true_max: float

    @classmethod
    def from_intervals(
        cls,
        name: str,
        unit_label: str,
        intervals: Sequence[tuple[float, float]],
        *,
        pad_fraction: float = 0.12,
    ) -> ZeroScale:
        lo, hi = compute_interval_domain(intervals, pad_fraction=pad_fraction)
        lows = [low for low, _ in intervals] + [0.0]
        highs = [high for _, high in intervals] + [0.0]
        return cls(
            name=name,
            unit_label=unit_label,
            domain_min=lo,
            domain_max=hi,
            true_min=min(lows),
            true_max=max(highs),
        )

    @classmethod
    def from_values(
        cls,
        name: str,
        unit_label: str,
        values: Sequence[float],
        *,
        zero_fraction: float,
        pad_fraction: float = 0.12,
    ) -> ZeroScale:
        """A scale over plain scalar values whose zero lands at a GIVEN
        fraction of the field.

        This is the Invariant Z machinery for the product's non-league
        scales (Phase 4 introduces the first one, `component_per_100`).
        Invariant D does not apply to them -- a component decomposition is
        a per-player quantity and must be allowed its own domain, or every
        player's components would be four marks huddled against one edge.
        Invariant Z still does: the component bars have to sit under the
        SAME amber rule the hero figure sits under, or the rule stops
        meaning "zero" halfway down the page.

        So the domain is built from the values, padded, and then ONE side is
        extended until `(0 - domain_min) / span` equals `zero_fraction`
        exactly. Extending is always safe -- it only ever adds empty scale
        beyond the data, never crops a value out of the field.

        The price of the exception is paid in the template, not here: a
        figure on any scale other than `league_per_100` MUST render visible
        tick labels carrying its unit (`docs/design/zero-spine-implementation
        -plan.md` § 1.2, the anti-fake-alignment guard). A figure that
        borrows the shared zero while hiding its own domain is precisely the
        deception the guard exists to prevent.
        """
        if not 0.0 < zero_fraction < 1.0:
            raise ValueError(f"zero_fraction must be strictly between 0 and 1, got {zero_fraction}")
        finite = [float(v) for v in values]
        lo = min(finite + [0.0])
        hi = max(finite + [0.0])
        span = (hi - lo) or 1.0
        pad = span * pad_fraction
        lo -= pad
        hi += pad
        # Both sides are now strictly signed (lo < 0 < hi) because 0.0 was
        # injected before padding, so neither ratio below can divide by zero.
        needed_hi = -lo * (1.0 - zero_fraction) / zero_fraction
        if needed_hi >= hi:
            hi = needed_hi
        else:
            lo = -hi * zero_fraction / (1.0 - zero_fraction)
        return cls(
            name=name,
            unit_label=unit_label,
            domain_min=lo,
            domain_max=hi,
            true_min=min(finite + [0.0]),
            true_max=max(finite + [0.0]),
        )

    @property
    def domain(self) -> tuple[float, float]:
        """The raw (min, max) tuple, for renderers that still take one."""
        return (self.domain_min, self.domain_max)

    @property
    def span(self) -> float:
        span = self.domain_max - self.domain_min
        # `compute_interval_domain` always injects 0.0 and then pads, so a
        # non-positive span is unreachable for a scale built through
        # `from_intervals`. Guarded anyway: a directly-constructed
        # degenerate scale must not produce a divide-by-zero or an
        # off-field zero line.
        return span if span > 0 else 1.0

    @property
    def zero_fraction(self) -> float:
        """Where zero sits, as a fraction of the plot field. THE number."""
        return (0.0 - self.domain_min) / self.span

    def fraction_of(self, value: float) -> float:
        """Unclamped position of `value`. May fall outside [0, 1]."""
        return (value - self.domain_min) / self.span

    def clamped_fraction_of(self, value: float) -> float:
        return min(max(self.fraction_of(value), 0.0), 1.0)

    def contains(self, value: float) -> bool:
        return self.domain_min <= value <= self.domain_max

    def point_state(self, point: float) -> PointState:
        """Where the POINT ESTIMATE sits relative to the canonical domain.

        Separate from `clip_state` because the two carry different meanings
        and get different marks. An interval running off the edge says "the
        uncertainty extends further than we can draw"; a point estimate off
        the edge says "the estimate itself is not on this scale", which is a
        much stronger statement and must never be rendered as a dot sitting
        on the boundary -- that would assert the boundary IS the estimate.

        Note that `point_state != "in"` always implies `clip_state` is
        clipped on the same side, since `lower <= point <= upper`. The two
        markers therefore never need to coexist on one edge, which is what
        keeps the combination unambiguous.
        """
        if point < self.domain_min:
            return "below"
        if point > self.domain_max:
            return "above"
        return "in"

    def clip_state(self, *, point: float, lower: float, upper: float) -> ClipState:
        """Which end(s) of this mark fall outside the canonical domain.

        Used to draw an explicit continuation caret. The graphical mark is
        cut off; the NUMBERS never are -- callers keep rendering the true
        point estimate and interval in text, so a clipped endpoint can never
        be mistaken for the real one.
        """
        low_clipped = lower < self.domain_min or point < self.domain_min
        high_clipped = upper > self.domain_max or point > self.domain_max
        if low_clipped and high_clipped:
            return "both"
        if low_clipped:
            return "low"
        if high_clipped:
            return "high"
        return "none"


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

    **Invariant Z (redesign Phase 1).** The plot field is the FULL viewBox
    width: this function contributes no horizontal margin of its own, so
    `zero_x / width` is exactly the scale's zero fraction at every size.
    It previously reserved `margin=10` inside a 220-wide box and `margin=28`
    inside a 480-wide one, which put the same data's zero line at 0.4630 and
    0.4640 of width respectively -- equal only when zero sits dead centre.
    That ~0.1% divergence is the "0.463/0.464" recorded in `DESIGN.md` as if
    it were one shared position; it is two. Horizontal breathing room is now
    the layout's job (CSS padding on the containing cell), never the
    coordinate space's, because anything inside the coordinate space moves
    zero. See `docs/design/zero-spine-implementation-plan.md` § 0.1.

    **Clipping (decision D1a).** The canonical `league_per_100` domain comes
    from the QUALIFIED comparison population, and never widens per page --
    a widened domain silently moves zero. A value outside it is therefore
    drawn cut off at the field edge with an explicit continuation caret and
    a `data-clipped` attribute, and the accessible label states both the
    true numbers and the fact that the mark extends beyond the displayed
    range. The point dot is NOT drawn at a clamped position: a dot sitting
    on the edge would assert that the edge is the estimate. The caret is
    information, not decoration.
    """
    width, height = (220, 32) if compact else (480, 72)
    domain_min, domain_max = domain
    if domain_max <= domain_min:
        domain_max = domain_min + 1.0
    span = domain_max - domain_min

    def x(value: float) -> float:
        return (value - domain_min) / span * width

    # Only a CLIPPED end is inset, and only far enough to seat the caret;
    # an unclipped end sits at its true position. Neither affects `zero_x`,
    # which is always the domain's own zero fraction of the full width.
    caret_w = 7.0 if compact else 10.0
    caret_h = 4.0 if compact else 6.0
    point_r = 4 if compact else 6
    # The off-scale point marker is drawn at the POINT's scale, not the
    # line's: same visual weight as the dot it stands in for, so it reads as
    # "the estimate, pushed off the edge" rather than as a bigger caret.
    point_mark_w = point_r * 2.4
    point_mark_h = point_r * 1.5
    mid_y = height / 2
    zero_x = x(0.0)
    sign_class = "interval-positive" if point >= 0 else "interval-negative"
    size_class = "interval-bar-compact" if compact else "interval-bar-full"

    point_state = "below" if point < domain_min else "above" if point > domain_max else "in"
    low_clipped = lower < domain_min or point < domain_min
    high_clipped = upper > domain_max or point > domain_max
    clip_state = (
        "both"
        if low_clipped and high_clipped
        else "low"
        if low_clipped
        else "high"
        if high_clipped
        else "none"
    )

    # The portion of the interval that actually lands inside the field. A
    # small-sample interval can sit ENTIRELY outside the canonical domain
    # (18 such players in the 2026-08-14 snapshot), in which case there is no
    # segment to draw and only the off-scale point marker is rendered --
    # never a zero-length or negative-width line.
    visible_lo = max(lower, domain_min)
    visible_hi = min(upper, domain_max)

    # Accessible text carries every fact the glyphs carry, plus the domain
    # itself, so a screen-reader user never has to infer clipping from a
    # shape they cannot see.
    direction = {"low": "below", "high": "above", "both": "below and above"}.get(clip_state)
    notes = []
    if point_state != "in":
        notes.append(
            f"the point estimate lies {point_state} the displayed range"
        )
    if clip_state != "none":
        notes.append(f"the 95 percent interval extends {direction} the displayed range")
    clip_note = ""
    if notes:
        clip_note = (
            f". Off scale: {'; and '.join(notes)}. The displayed range covers "
            f"{domain_min:+.2f} to {domain_max:+.2f}; the figures above are the actual values"
        )

    parts = [
        f'<svg class="interval-bar {size_class}" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="xMidYMid meet" role="img" data-clipped="{clip_state}" '
        f'data-point-offscale="{point_state}" '
        f'aria-label="Point estimate {point:+.2f} runs per 100 eligible batted balls; '
        f'95 percent interval {lower:+.2f} to {upper:+.2f}{clip_note}">',
        f"<title>{point:+.2f} runs/100 (95% interval: {lower:+.2f} to "
        f"{upper:+.2f}){clip_note}</title>",
        f'<line class="interval-zero-line" x1="{zero_x:.1f}" y1="4" '
        f'x2="{zero_x:.1f}" y2="{height - 4}"></line>',
    ]

    if visible_hi > visible_lo:
        lower_x = caret_w if low_clipped else x(visible_lo)
        upper_x = width - caret_w if high_clipped else x(visible_hi)
        parts.append(
            f'<line class="interval-range {sign_class}" x1="{lower_x:.1f}" '
            f'y1="{mid_y:.1f}" x2="{upper_x:.1f}" y2="{mid_y:.1f}"></line>'
        )

    if point_state == "in":
        parts.append(
            f'<circle class="interval-point {sign_class}" cx="{x(point):.1f}" '
            f'cy="{mid_y:.1f}" r="{point_r}"></circle>'
        )

    def edge_mark(side: str) -> str:
        """One mark per clipped edge, encoding the stronger of the two facts.

        Two shapes, deliberately different in BOTH fill and scale so they
        stay apart at a glance:

          * an OPEN outward chevron at the interval line's weight --
            "the interval continues past here; the estimate is inside".
          * a SOLID outward triangle at the point marker's scale, carrying
            the same `--surface` ring the in-domain dot has --
            "the POINT ESTIMATE itself lies beyond this edge".

        The solid marker wins where both apply, because an off-scale point
        already implies an off-scale interval on that side (see
        `ZeroScale.point_state`) and drawing both would be mud. The
        accessible label states both facts regardless of which is drawn.

        An off-scale point marker can only ever appear on the full-size
        player bar: the leaderboard's domain is built from exactly the
        qualified population it draws, so a leaderboard row cannot clip.
        Legibility at the compact bar's mobile size is therefore not a
        constraint here (asserted in tests/test_dashboard_zero_scale.py).
        """
        outward = -1 if side == "low" else 1
        edge = 0.0 if side == "low" else float(width)
        if point_state == ("below" if side == "low" else "above"):
            base = edge - outward * point_mark_w
            return (
                f'<polygon class="interval-point-offscale {sign_class}" points="'
                f"{edge:.1f},{mid_y:.1f} {base:.1f},{mid_y - point_mark_h:.1f} "
                f'{base:.1f},{mid_y + point_mark_h:.1f}"></polygon>'
            )
        tip_x = edge + outward * 1.0
        back_x = edge - outward * (caret_w - 1.0)
        return (
            f'<polyline class="interval-clip-caret {sign_class}" points="'
            f"{back_x:.1f},{mid_y - caret_h:.1f} {tip_x:.1f},{mid_y:.1f} "
            f'{back_x:.1f},{mid_y + caret_h:.1f}"></polyline>'
        )

    if low_clipped:
        parts.append(edge_mark("low"))
    if high_clipped:
        parts.append(edge_mark("high"))

    if not compact:
        parts.append(
            f'<text class="interval-axis-label" x="{zero_x:.1f}" y="{height - 6:.1f}" '
            f'text-anchor="middle">0</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


#: Candidate y-axis steps for the season trend, ascending. A player whose
#: whole season sits inside a half-run band still has to get three real
#: tick VALUES (gate G2), which is why the list starts below 0.1 -- an
#: axis labelled only with its endpoints is the "two ticks" failure G2
#: exists to catch.
_TREND_Y_STEPS: tuple[float, ...] = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0,
)

#: The most x-axis date labels the figure ever prints. Every point keeps its
#: exact date in its own accessible name and in the figure's data table --
#: this caps only how many are drawn UNDER the axis, so the labels stay
#: legible at 320px without a second, mobile-only label set to maintain.
_TREND_MAX_X_LABELS = 5


@dataclass(frozen=True)
class TrendFigure:
    """Geometry for the season-to-date trend, as percentages of a plot box.

    **Why this is not one SVG.** The baseline drew the whole figure --
    marks, axis labels and dates -- inside a 640x220 viewBox that rendered
    at 350x120 on a phone, a 0.55 scale factor that put its 10px labels at
    ~5.5 CSS px. `docs/design/guardrails.md` anti-pattern 11 bans SVG text
    below 12 CSS px after viewBox scaling, and no amount of re-sizing that
    viewBox fixes the class of bug: any text inside a scaled coordinate
    space is one narrow breakpoint away from being unreadable again.

    So the text leaves the coordinate space. This object carries plain CSS
    percentages; the template lays the axis labels, the date labels and the
    point markers out as ordinary HTML positioned against a `position:
    relative` plot box, at real, unscaled font sizes. `marks_svg` holds the
    only two things that genuinely need a coordinate space -- the interval
    band polygon and the point-estimate polyline -- and nothing else, so it
    contains no text at all and can be `aria-hidden`.

    **The declared exception is still declared.** `y_min`/`y_max` are this
    player's own, not `league_per_100` (decision D2), and `league_band`
    exists to pay for that: it shades the part of this player's vertical
    range that the leaderboard's canonical domain covers, so the reader can
    see how much of the league range is in view.
    """

    marks_svg: str
    y_ticks: list[dict[str, Any]]
    x_labels: list[dict[str, Any]]
    points: list[dict[str, Any]]
    zero_top: str
    league_band: dict[str, Any] | None
    unit_label: str
    y_min: float
    y_max: float
    summary: str


def _trend_pct(fraction: float) -> str:
    return f"{fraction * 100:.4f}%"


def _trend_y_ticks(y_min: float, y_max: float) -> list[float]:
    """At least three real tick VALUES spanning the player's own range.

    Picks the step that yields the most ticks without exceeding six, and
    falls back to the densest available step if even the finest cannot
    reach three -- zero is always among them, because every step divides it.
    """
    best: list[float] = []
    for step in _TREND_Y_STEPS:
        start = math.ceil(y_min / step) * step
        values: list[float] = []
        value = start
        while value <= y_max + 1e-9:
            values.append(0.0 if abs(value) < 1e-9 else value)
            value += step
        if 3 <= len(values) <= 6:
            return values
        if len(values) <= 6 and len(values) > len(best):
            best = values
        if len(values) > 6 and not best:
            best = values[:6]
    return best


def build_trend_figure(
    points: Sequence[TrendPoint],
    *,
    league_scale: ZeroScale,
) -> TrendFigure | None:
    """The season-to-date trend, re-authored for Phase 4 / gate G2.

    Returns `None` when there is nothing to draw (no snapshots, or the one
    snapshot case -- a trend needs two points, and the page prints a
    sentence saying so rather than an empty axis).

    X positions are TRUE calendar positions, not evenly-spaced ordinal
    slots, so the missing 2026-08-07 snapshot shows up as a longer segment
    between 08-06 and 08-08 and never as a fabricated point. That behaviour
    predates this rewrite and is preserved deliberately.

    **DECLARED EXCEPTION to Invariant D (decision D2, approved 2026-08-26;
    rendered sign-off is gate G2).** The y-domain is PER PLAYER, not the
    canonical `league_per_100` scale. This figure answers "has this
    player's own number moved?", which is a different question from "where
    does this player sit in the league?", and forcing the league domain
    onto it would render almost every player's season as a flat line in the
    middle of an empty chart -- consistent, and useless.

    The exception is paid for here rather than asserted:

      * zero is always drawn, always labelled, and always inside the range
        (0.0 is injected into the extremes before padding);
      * `y_ticks` carries at least three real values, not just endpoints;
      * `unit_label` is rendered ON the figure, not only in prose;
      * `league_band` shades the portion of this view that the leaderboard's
        canonical domain covers, so the two scales are visibly different
        rather than silently different;
      * every label is HTML at an unscaled size (see `TrendFigure`);
      * `summary` and the per-point entries state value, date and unit.
    """
    if len(points) < 2:
        return None

    dates = [date.fromisoformat(p.data_through_date) for p in points]
    x_min, x_max = min(dates), max(dates)
    x_span = (x_max - x_min).days or 1

    values = [p.contact_luck_runs_per_100 for p in points]
    lowers = [p.lower_95_interval for p in points]
    uppers = [p.upper_95_interval for p in points]
    y_min = min(lowers + [0.0])
    y_max = max(uppers + [0.0])
    y_pad = ((y_max - y_min) or 1.0) * 0.12
    y_min -= y_pad
    y_max += y_pad
    y_span = (y_max - y_min) or 1.0

    def xf(d: date) -> float:
        return (d - x_min).days / x_span

    def yf(value: float) -> float:
        """Top-down fraction: 0 is the top of the plot box, 1 the bottom."""
        return 1.0 - (value - y_min) / y_span

    band_upper = " ".join(
        f"{xf(d) * 100:.4f},{yf(u) * 100:.4f}" for d, u in zip(dates, uppers, strict=True)
    )
    band_lower = " ".join(
        f"{xf(d) * 100:.4f},{yf(low) * 100:.4f}"
        for d, low in zip(reversed(dates), reversed(lowers), strict=True)
    )
    line_pts = " ".join(
        f"{xf(d) * 100:.4f},{yf(v) * 100:.4f}" for d, v in zip(dates, values, strict=True)
    )
    # A 0-100 box stretched to the plot's real aspect by
    # `preserveAspectRatio="none"`: the band is a fill (distortion is
    # meaningless on it) and the line carries `vector-effect` so its 2px
    # weight survives the stretch. Nothing that must keep its shape -- no
    # dot, and above all no glyph -- is inside this coordinate space.
    marks_svg = (
        '<svg class="trend-marks" viewBox="0 0 100 100" preserveAspectRatio="none" '
        'aria-hidden="true" focusable="false">'
        f'<polygon class="trend-band" points="{band_upper} {band_lower}"></polygon>'
        f'<polyline class="trend-line" points="{line_pts}" '
        'vector-effect="non-scaling-stroke"></polyline>'
        "</svg>"
    )

    tick_values = _trend_y_ticks(y_min, y_max)
    # One precision for the whole axis, chosen from the smallest gap in it.
    # A column mixing "+12.5" and "+0.00" is exactly the misalignment tabular
    # figures exist to prevent, and it reads as two different quantities.
    gaps = [abs(b - a) for a, b in zip(tick_values, tick_values[1:], strict=False)]
    digits = 2 if (gaps and min(gaps) < 1.0) else 1
    y_ticks = [
        {
            "value": tick,
            "label": f"{tick:+.{digits}f}".replace("-", "−"),
            "top": _trend_pct(yf(tick)),
            "is_zero": abs(tick) < 1e-9,
        }
        for tick in tick_values
    ]

    step = max(1, math.ceil((len(points) - 1) / (_TREND_MAX_X_LABELS - 1)))
    label_indexes = set(range(0, len(points), step)) | {0, len(points) - 1}
    x_labels = [
        {
            "label": dates[i].strftime("%b %-d"),
            "left": _trend_pct(xf(dates[i])),
        }
        for i in sorted(label_indexes)
    ]

    view_points = []
    for d, value, point in zip(dates, values, points, strict=True):
        view_points.append(
            {
                "date": point.data_through_date,
                "date_label": d.strftime("%B %-d, %Y"),
                "value": value,
                "lower": point.lower_95_interval,
                "upper": point.upper_95_interval,
                "eligible_batted_balls": point.eligible_batted_balls,
                "qualification_status": point.qualification_status,
                "is_backfill": point.snapshot_type == "retrospective_backfill",
                "left": _trend_pct(xf(d)),
                "top": _trend_pct(yf(value)),
                "favorable": value >= 0,
            }
        )

    # The leaderboard's canonical range, as up to two reference rules.
    #
    # Only a boundary that genuinely falls INSIDE this player's view gets a
    # rule. Clamping them to the plot edges instead -- the obvious
    # implementation -- draws a rule at the bottom of a chart whose lowest
    # value is nowhere near the league minimum, which asserts something
    # false about where that boundary is. Two rules, one, or none is the
    # honest set, and `rules_drawn` lets the caption say which case it is,
    # because "no rule because the whole view is inside the range" and "no
    # rule because there is no range" look identical on the chart.
    band_lo = max(league_scale.domain_min, y_min)
    band_hi = min(league_scale.domain_max, y_max)
    league_band: dict[str, Any] | None = None
    if band_hi > band_lo:
        rules = [
            {"value": value, "top": _trend_pct(yf(value))}
            for value in (league_scale.domain_max, league_scale.domain_min)
            if y_min < value < y_max
        ]
        coverage = (band_hi - band_lo) / y_span
        league_band = {
            "rules": rules,
            "rules_drawn": len(rules),
            # A measured share, not a boolean. The boolean this replaced was
            # false for a range covering 99.9% of the view -- true by 0.02
            # runs, and read as "most of this chart is off the league scale".
            "coverage_pct": f"{coverage * 100:.0f}%",
            "domain_min": league_scale.domain_min,
            "domain_max": league_scale.domain_max,
        }

    def signed(value: float) -> str:
        # A true U+2212 minus, applied per NUMBER. Never over a whole
        # sentence: `.replace("-", ...)` across prose would also rewrite
        # every hyphen in it.
        return f"{value:+.2f}".replace("-", "−")

    summary = (
        f"Contact Luck at {len(points)} stored snapshots, "
        f"{dates[0].strftime('%B %-d')} to {dates[-1].strftime('%B %-d, %Y')}. "
        f"It starts at {signed(values[0])}, ends at {signed(values[-1])}, and ranges from "
        f"{signed(min(values))} to {signed(max(values))} {league_scale.unit_label}. "
        # The scale is a declared exception to Invariant D (gate G2), so the
        # text equivalent has to say whose scale it is; a reader who cannot
        # see the axis would otherwise assume the leaderboard's.
        f"The vertical scale runs {signed(y_min)} to {signed(y_max)}, set by this player's "
        "own values and not the leaderboard's range."
    )

    return TrendFigure(
        marks_svg=marks_svg,
        y_ticks=y_ticks,
        x_labels=x_labels,
        points=view_points,
        zero_top=_trend_pct(yf(0.0)),
        league_band=league_band,
        unit_label=league_scale.unit_label,
        y_min=y_min,
        y_max=y_max,
        summary=summary,
    )


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


def render_simulator_field_svg(*, width: int = 300, height: int = 280) -> str:
    """A static field-diagram SKELETON for the "Try It Yourself" simulator
    (Version 1.3.1), reusing the same schematic fan-shaped chrome (fence
    arc, foul lines, infield diamond) as `render_demo_field_svg` above, at
    the SAME `_DEMO_FIELD_*` coordinates -- so the two visuals read as one
    consistent field, not two different diagrams.

    This function deliberately does NOT share code with
    `render_demo_field_svg` (the small amount of fence/foul-line/infield
    geometry below is duplicated, not factored into a shared helper) so
    that nothing here can ever change that function's own output -- the
    v1.3.0 walkthrough's rendering must stay provably untouched by this
    v1.3.1 addition.

    Unlike `render_demo_field_svg`, this SVG carries NO baked-in ball
    position: it renders a `simulator-field-path`/`simulator-field-ball`
    placeholder sitting at home plate with an empty path, and
    `static/demo_simulator.js` is the sole owner of both elements'
    coordinates from then on -- driven live by the selected exit
    velocity/launch angle/reference play, via a deterministic heuristic
    that is explicitly NOT a physics simulation (see that script's
    module docstring). This mirrors `render_demo_field_svg`'s own
    single-source-of-truth pattern (there, `data-start/control/end-*`
    attributes feed a `requestAnimationFrame` loop instead of SMIL), just
    with the JS computing the coordinates itself instead of reading them
    off the markup, since here they must change continuously as the user
    drags a slider rather than play once on reveal.
    """
    home_x, home_y = _DEMO_FIELD_HOME_X, _DEMO_FIELD_HOME_Y

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

    parts = [
        f'<svg class="demo-field simulator-field" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'aria-label="Illustrative field view driven by the selected exit velocity and '
        f'launch angle (not a reconstruction of physical ball flight)" '
        f'data-role="simulator-field-svg" '
        f'data-home-x="{home_x:.1f}" data-home-y="{home_y:.1f}" '
        f'data-fence-radius="{_DEMO_FIELD_FENCE_RADIUS:.1f}" '
        f'data-max-spray-deg="{_DEMO_FIELD_MAX_SPRAY_DEG:.1f}">',
        f'<path class="demo-field-fence" d="M {left_foul_x:.1f},{left_foul_y:.1f} '
        f"A {_DEMO_FIELD_FENCE_RADIUS:.0f},{_DEMO_FIELD_FENCE_RADIUS:.0f} 0 0 1 "
        f'{right_foul_x:.1f},{right_foul_y:.1f}"></path>',
        f'<line class="demo-field-foul-line" x1="{home_x:.1f}" y1="{home_y:.1f}" '
        f'x2="{left_foul_x:.1f}" y2="{left_foul_y:.1f}"></line>',
        f'<line class="demo-field-foul-line" x1="{home_x:.1f}" y1="{home_y:.1f}" '
        f'x2="{right_foul_x:.1f}" y2="{right_foul_y:.1f}"></line>',
        f'<polygon class="demo-field-infield" points="{infield_pts}"></polygon>',
        '<path class="simulator-field-path" data-role="simulator-field-path" d=""></path>',
        f'<circle class="simulator-field-ball" data-role="simulator-field-ball" r="7" '
        f'cx="{home_x:.1f}" cy="{home_y:.1f}"></circle>',
        "</svg>",
    ]
    return "".join(parts)
