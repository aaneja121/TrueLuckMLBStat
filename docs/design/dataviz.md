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
| **Season trend** | Has it changed over the season? | Line + dots at true snapshot dates; **gaps never interpolated** (already correct — preserve); add a zero reference line and ≥ 3 y-axis labels |
| **Component decomposition** | Why this score? | Signed stacked bar on the **same zero axis**, with the exact-value table retained beneath |
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
