# Data visualization

Figures, the shared domain, and mobile re-authoring. Entry point: `DESIGN.md`.
Read with `docs/design/color.md` (mark vs. text grades) and the global `dataviz` skill —
but note that skill's palette is a **chart** palette and must never be adopted as the UI
palette (`docs/design/color.md` § The finding this system exists to fix).

Every visualization answers one named question. Follows the `dataviz` skill: thin marks,
≥ 8 px point markers, 2 px lines, recessive grid, text in ink tokens (never in a series
colour), a 2 px surface ring where marks overlap.

| Figure | Question | Form |
|---|---|---|
| **Interval strip** | Where does this player sit, and how sure are we? | Zero-anchored strip on the **league-wide fixed domain**; 2 px interval line + ≥ 8 px point dot; sign colours the mark |
| **Season trend** | Has it changed over the season? | Line + dots at true snapshot dates; **gaps never interpolated**; zero always drawn and labelled, ≥ 3 real y tick values, and the league range as dashed reference rules. **Declared exception to shared scale** — see below |
| **Component decomposition** | Why this score? | Signed bars from zero on `component_per_100`, one table row per component carrying label, bar, value and model status together |
| **Outcome probabilities** | What did the model expect? | Horizontal bars with the **recorded result marked in place**, so expected and actual are one figure |
| **Expected → observed** | How big was the gap? | A single zero-anchored gap figure, not three stacked text blocks |

---

## Rules

- **The domain is fixed per snapshot, league-wide — and this already works. Do not break
  it.** Verified: the zero line sits at 0.463 of width on the leaderboard strip
  (`viewBox 0 0 220 32`, `x=101.8`) and at 0.464 on every player page
  (`viewBox 0 0 480 72`, `x=222.6`), including unqualified players. A `+7.62` and a `−0.06`
  are therefore already visually comparable across routes — an uncommon and valuable
  property that the current styling squanders by rendering the strip as a small cell
  ornament. Per-chart or per-player auto-scaling is banned.
- **Charts re-author on mobile; they do not rescale.** The viewBox is chosen per breakpoint
  so SVG text renders at **≥ 12 CSS px**. Baseline defect: the trend chart's 640 × 220
  viewBox renders at 350 × 120 on mobile (scale 0.55), putting 10 px labels at ~5.5 CSS px.
  If a target size cannot hold labels, the figure changes form — trend becomes a sparkline
  with first/last values, or a two-row table.
- **Near-zero values.** Where |point estimate| < 0.10 the dot renders centred on the spine
  with its surface ring, the value is written with an explicit sign (`−0.06`), and the
  colour still follows the sign. The reader sees a dot on the spine with a wide interval —
  the shared axis conveys "small and uncertain" without breaking the sign-colour invariant
  and without any de-emphasis.
- **Every meaningful SVG is `role="img"` with an `aria-label` stating the value in words.**
  Purely decorative marks are `aria-hidden`. Baseline: interval bars and trend charts do
  this correctly; the play page's field diagram has no role, label, title, or `aria-hidden`.
- No gradients, glows, shadows, 3-D, dual axes, or pie charts. No number count-up animation
  on a statistic — an animated statistic is a misleading statistic.

Implementation note: SVG is generated at build time in `dashboard/visuals.py`, which emits
CSS classes only and **never a hex colour** (`ARCHITECTURE.md`). Keep it that way — the
tokens in `docs/design/color.md` are the single source of every mark colour.

---

## The season trend — how the exception is paid for (decision D2, gate G2)

The trend's y-domain is **per player**, not `league_per_100`. It answers "has this hitter's
own number moved?", which is a different question from "where do they sit in the league?",
and forcing the league domain onto it renders almost every season as a flat line in the
middle of an empty chart. The exception is declared, not silent, and the figure pays for it:

- zero is always drawn, always labelled, and always inside the range;
- at least **three real tick values**, not just the endpoints, at one precision for the
  whole axis (an axis mixing `+12.5` with `+0.00` reads as two quantities);
- the **unit is on the figure**, not only in surrounding prose;
- the **league range is drawn as two dashed reference rules**, and only where a boundary
  genuinely falls inside the view. Clamping a rule to the plot edge instead — the obvious
  implementation — draws a boundary somewhere it is not. The caption states the share of
  the view the range covers, and how many rules were drawn, so "no rule because the whole
  view is inside the range" cannot be confused with "no rule because there is no range";
- **nothing borrows the Zero Spine's vocabulary** — no `--cl-zero`, no amber rule, no
  shared axis chrome. The two scales are different and must not look alike.

**A fill loses this argument twice.** As ground, the league band disappears under the 95%
interval band (measured on a small-sample hitter whose interval covers most of the plot);
over the top, it washes the interval band out — and the interval is the data, so it wins.
Dashed rules sit above every fill and stay legible either way. Dashing is safe here
*because this is not an interval*: guardrail 17 forbids dashing an interval that crosses
zero, and a reference boundary is neither an interval nor a point estimate.

## Text never lives inside a scaled coordinate space

The baseline drew the whole trend — marks, axis labels and dates — inside a 640 × 220
viewBox that rendered at 350 × 120 on a phone, a 0.55 scale factor that put its 10 px
labels at ~5.5 CSS px. Re-sizing that viewBox does not fix the *class* of bug: any text
inside a scaled coordinate space is one narrow breakpoint away from being unreadable again.

So the text left the coordinate space. `visuals.build_trend_figure` returns plain CSS
percentages; the template lays axis labels, date labels and point markers out as ordinary
HTML against a `position: relative` plot box at real, unscaled sizes. The SVG holds only
the two things that genuinely need a coordinate space — the interval band polygon and the
point-estimate polyline — and therefore contains no text at all and is `aria-hidden`. The
line keeps its 2 px weight through `vector-effect="non-scaling-stroke"`; the dots are HTML,
outside the SVG, so they stay round under `preserveAspectRatio="none"`.

Measured after the rewrite: every label is 12 px at 1440 / 1180 / 768 / 390 / 320. Below
360 px the interior date labels are dropped and the endpoints kept — an axis is
orientation, and every point's exact date is still carried by its own marker and by the
Snapshot values table.

## A second scale must declare its own domain

`component_per_100` is a per-player domain, padded by `ZeroScale.from_values` until its
zero lands on the shared `--cl-zero`. Padding only ever *extends* a scale beyond the data;
it never crops a value out of the field.

The price is the **anti-fake-alignment guard**: any figure on a scale other than
`league_per_100` must render visible tick labels carrying its unit. A figure that borrows
the shared zero while hiding its own domain is exactly the deception the guard exists to
prevent. The component decomposition prints its ticks directly above the first bar, and
its section lede states the domain in words.

## The `run_value` scale (Explore, Phase 5)

The third registered scale. One snapshot-level domain over **every published play**, read
off the shard walk `explore_content.load_explore_catalog` already performs, then padded by
`ZeroScale.from_values` until its zero lands on the shared `--cl-zero`. Explore draws every
hitter's plays on it, so two hitters' plays are comparable and a one-play hitter is not
stretched across the whole field.

It pays the anti-fake-alignment guard in full: its own tick labels, its own unit on the
figure, and its domain stated in words under the unit. The leaderboard hides `.cl-axis-unit`
at mobile because its column head carries the unit; Explore's column heads go visually
hidden there, so its unit is explicitly re-shown.

**The domain is computed at build time and shipped as data.** `explore.js` interpolates a
layout percentage against it and derives no domain of its own — a per-hitter autoscale
would put two hitters on two scales and move zero off the spine, which is decision D1a
applied to a client-rendered figure. A value outside the domain clips; the scale never
widens.

### Two figures on one page must share one coordinate space

Explore's first build drew the play distribution as a standalone `<figure>` above the
results table. Both were correct internally, and their zeros were 120 px apart, because a
full-width figure and a table column are two different percentage bases.

The fix is the leaderboard's own construction: the axis rail and the all-plays strip are
**rows inside the results table**, in the verdict column, so the ticks, the strip and every
row's mark are the same grid track and cannot disagree at any width. Measured at
1440/1280/768/390/320 — all three land on one x to three decimals.

The showcase list is a separate section and cannot be a table row, so it is aligned the
other way: its grid uses the same four leading tracks as the table (`32rem`, minus the
grid gap the table carries as cell padding) and the same reserved numeral track. One zero
on the page, not two that happen to be close.
