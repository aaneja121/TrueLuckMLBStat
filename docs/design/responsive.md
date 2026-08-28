# Responsive behavior

Breakpoints and the per-element behaviour matrix. Entry point: `DESIGN.md`.
Read with `docs/design/tables.md` (shedding order) and `docs/design/dataviz.md`
(charts re-author, they do not rescale).

Breakpoints: **≥ 1280** desktop · **1024–1279** laptop · **640–1023** narrow · **< 640**
mobile. (Consolidates the current 640 / 800 / 900 / 1440 set.)

These four tiers are the **default** every component starts from, and the vocabulary the
matrix below is written in. They are not a promise that every component's content fits
inside them — see § Content-driven component breakpoints.

| Element | Desktop | Laptop | Narrow | Mobile |
|---|---|---|---|---|
| **Navigation** (application header — **switches at 1100, not 1024**; see below) | Inline row, search inline | Inline row, search inline — field narrows, zone rules tighten, field label goes visually hidden | Identity + provenance / routes in one row / full-width search, as three ruled lines | Disclosure menu, 44 px targets, search as a full-screen sheet. **Never a two-row wrap.** |
| **Leaderboard** | 6 columns, merged verdict cell | Evidence tier merges to one sample cell | Three-line ruled row; spine segmented (`docs/design/tables.md`) | Three-line ruled row; axis rail and distribution strip drop below 768 |
| **Tables (general)** | `min-width` contract + visible scroll affordance | same | same | Re-authored form, not a squeeze |
| **Player header** | 8/4 split; score and interval share a baseline | same | Stacked; interval stays attached to the score | same |
| **Charts** | Full figure | Full figure | Re-authored viewBox | Simplified form (sparkline + endpoints) |
| **Play data** | 4-across stat line | 4-across | 2 × 2 | Two-line ruled list |
| **Methodology** | Sticky section index | Sticky index | Collapsed ToC | Collapsed ToC |
| **Status history** | Scrolling table with min-width | same | **Definition list per snapshot** (date as heading, fields as label/value pairs) | Definition list |
| **Search** | Inline combobox | Inline | Inline | Full-screen sheet |
| **Long names** | Full | Full | Full-width line | Full-width line; ellipsis at a word boundary only if still overflowing |

The status timestamp is reformatted at the source (`Aug 15, 2026, 13:35 UTC`, with the ISO
value in `<time datetime>`), which removes the unbreakable 32-character token entirely —
preferable to engineering a layout that survives it. This is design principle 8 (*format the
data rather than engineering around it*) applied.

Verification: `tests/test_dashboard_responsive_overflow.py` is the test most likely to be
tripped by work in this file (`ARCHITECTURE.md` § Tests & tooling). Rendered checks run at
~1440, ~1280 and ~390 via the global Playwright MCP. A component claiming a content-driven
breakpoint is checked at the boundary itself as well — for the header, at 1099 and 1100.

---

## Content-driven component breakpoints

The four tiers above are design guidance and stay that way: they keep unrelated components
changing shape at the same widths, which is what makes the whole page feel like one system
rather than a pile of independent widgets. Start every component there.

But a tier boundary is a guess about width, and a component is made of **content** —
labels, numerals, names — whose intrinsic width is a fact. **When measurement shows the
declared tier cannot hold a component's contract, the component gets its own breakpoint and
the measurement is written down here.** The alternatives are worse: shrinking type until it
fits is compressed desktop, and letting the layout overflow is a defect.

Three rules govern this, and the third is the one that keeps the system from dissolving:

1. **The tier is the default.** A component departs from it only with a measured reason.
2. **Only that component moves.** A content-driven breakpoint is local. It never becomes the
   new global tier, and it never silently drags a neighbouring component with it.
3. **Measured and documented, never chosen.** The number comes from a rendered measurement
   at a real width, and both the number and the measurement land in the table below. A
   breakpoint picked because it "looked about right" is not one of these.

| Component | Breakpoint | Tier it departs from | Measured reason |
|---|---|---|---|
| **Application header** (`.site-header-inner`) | **1100** | Laptop/narrow boundary at 1024 | The five route labels measure **493 px** together at their normal size. Alongside the wordmark, the search field and the "Data through" provenance, the one-row contract overran a 1024 viewport by **13 px** — and that was *after* narrowing the field to 140 px, tightening the zone rules to 16 px and hiding the field's visible label. The routes are the one thing in the header that must not shrink, so the row structure changes instead. Verified: 0 overflow across 8 routes × 10 widths × 2 themes. |

The header is the only component with a content-driven breakpoint today. Adding a second
means adding a row to that table, with its own measurement.

---

## Mobile homepage hierarchy

Below 768 the homepage re-ranks its preamble. The rows were never the problem — the
accumulated bands in front of them were. Measured at 390 px, the first hitter moved from
**574 px to 427 px** from the top, and 390×844 went from 2 whole rows visible to 4 (with a
fifth part-visible).

Nothing here was bought with type, targets or row height, all of which are unchanged:

| Band | Before | After | What changed |
|---|---|---|---|
| Page top air | 36 px | 20 px | `main` top padding `--sp-5` → `--sp-3` at mobile. The header already ends in its own padding; a second 24 px band was doubled space. |
| Lede | 153 px | 133 px | The disclosure **shares the title's band** instead of owning one below it, and its target came **up** from 32 px to the 44 px floor while doing so. |
| Scale caption | 48 px | 0 px | Moved **below** the table (see below). |
| Ranking tabs | 79 px | 44 px | Two stacked full-width tabs → one row of two, each at 44 px (they were 40 px). |
| Sort rail | 75 px | 46 px | Two wrapped rows of 11 px labels — the second holding "Sample" alone — → one row, each control at 44 px (they were 24.5 px). |
| Filter | 30 px | 44 px | Raised **to** the `--target-min` floor — the last control on the page under it. Its 14 px comes back out of the margins either side (below). |

Three rules govern how this was done, and they are the reusable part:

1. **Re-order prose, never controls.** `.home-leaderboard` is a wrapper that is
   `display: contents` at every width except mobile, where it becomes a flex column so the
   axis caption can take `order` behind the table. It contains no focusable content, so
   visual order and focus order cannot diverge. The lede's disclosure — which *is*
   focusable — was **not** re-ordered for the same reason; it was moved onto a band that
   already existed. A CSS `order` that moves a control past a table is a WCAG 2.4.3 defect,
   not a layout technique.
2. **Shorten what is seen, never what is announced.** The mobile ranking tabs read "Most
   favorable" / "Least favorable". The rest of each frozen `public_labels` string is in
   `.ranking-tab-rest`, **clipped, never `display: none`**, so the accessible name is still
   "Most favorable realized luck" / "Least favorable outcomes relative to expectation" and
   the visible text is a leading substring of it (WCAG 2.5.3). Verified against the
   accessibility tree, not the DOM.
3. **A control short of the target floor grows; the space comes from spacing, not from
   other content.** The name filter was 30 px against a 44 px floor. An oversized hit area
   over a 30 px field was rejected: at 4 px of clearance it would have overlapped the sort
   buttons below it, and a tap landing on the wrong control is worse than a small one. The
   field itself is 44 px, and its 14 px is repaid by the margins on either side —
   `.ranking-tabs` bottom margin 8 → 4 px, `.leaderboard-controls` bottom margin 4 → 0 —
   because the field now carries that air as its own internal padding, so those margins were
   doubling what the control already provides.
4. **A caption that moves must be true in both places.** The scale caption used to end
   "every row below is drawn on it"; below the table that is false. It now reads "every row
   in this table is drawn on it", which holds above and below. Desktop copy changed by three
   words; desktop layout did not change at all.

**Why the caption moves at all:** above 768 it captions a visible tick rail and a 124-mark
distribution strip. Below 768 both are hidden (§ the matrix above), so pre-table it captions
nothing the reader can see, while still charging 48 px for the position directly in front of
the product. Below the table it sits with the leaderboard note, where the rest of the
table's apparatus already lives.

**Desktop is unchanged, and this was measured rather than assumed.** `tbody` top, lede
height, tab height and sort-rail height are byte-identical before and after at 768, 800,
900, 1100, 1280 and 1440.

**Known floor:** at 390×844 the first row starts at 427 px and the mobile row is 104 px, so
the 4th row ends at 843.4 px against an 844 px viewport. **Four whole rows is true by 0.6 px
and should not be leaned on** — anything that renders the header a pixel taller makes it
three, which is the correct trade under the priority order (target size first, density last)
and not a regression. A fifth row needs the row itself to shrink, which is out of bounds. At
320×700 the first row moved 597 px → 448 px, and 0 whole rows became 2.

---

## Player page — the mobile transformation

Structural, not a shrink. Three things change shape rather than size at ≤ 767 px.

**1. The headline numeral stops being anchored at `--cl-pt`.** On desktop the score is
printed *at* its own mark: the numeral band and the plot are siblings in one block, so both
resolve `--cl-pt` against the same width, and the block's anchor — not its position —
changes near an edge (`start` below 0.18, `end` above 0.82, `center` between), which is
what keeps a 44 px numeral on the field for an off-scale hitter. At 390 the field is ~340 px
and the numeral block ~200 px, so no anchor survives near an edge; the number goes
left-aligned instead. Nothing is lost — at this width the dot is a couple of centimetres
away regardless.

**2. The hero keeps the full axis and the whole league.** It does not simplify, because it
is the product. Measured at 390: the mark row, the 124-hitter distribution and the labelled
tick rail all render, and the amber rule descends all three.

**3. The component table sheds its status column into a line under each row.** A four-column
table cannot hold a usable coordinate space at 390. The row becomes a three-area grid —
`label value` / `plot` / `status` — using grid areas rather than floats or inline-blocks,
because the whitespace between table cells is real text and it is what breaks an
inline-block row. **The status is never dropped**: Phase 4 requires every displayed
component value to carry one.

Because the bar then spans the whole row, the row box *is* the coordinate space (measured:
field and row both 16 → 374 at 390). That lets the spine move from the field to the
`tbody`, which is what keeps it one continuous rule — per-field segments would be five
22 px ticks separated by the label and status lines between them, which is precisely the
"stray ticks instead of a rule" failure `docs/design/tables.md` rule 2 is drawing.

**The sample line becomes a 2 × 2 grid.** Four items on one ruled line needs ~520 px; at
390 it wrapped into three ragged rows with dividers landing mid-air.

**Trend labels.** The plot drops to 200 px tall. Below 360 px the interior date labels are
dropped and the endpoints kept — five labels collide in the 235 px plot a 320 px viewport
leaves. `:first-child`/`:last-child` rather than an `:nth-child` rhythm, so the first and
last snapshot survive for any label count (they range from two to five).

Verified at 390 × 844 and 320 × 700, light and dark: no horizontal overflow, every axis
label 12 px, every operable control ≥ 44 px.

## Explore — the mobile transformation (Phase 5)

Structural, not a shrink.

- **The picker** goes full width and keeps its own visible label. It does **not** become
  the header's full-screen sheet: that pattern exists to rescue a cramped header strip,
  and Explore's picker is page content with its listbox directly below it.
- **The controls** stack to one per line, each keeping its label and a 44 px target.
- **A play row** becomes a three-area grid (identity + result / contact + expected /
  verdict). The verdict stays one statement on its own line, field and numeral adjacent.
- **The axis rail and the all-plays strip stay on screen**; only the column heads go
  visually hidden. See `docs/design/tables.md` for why.
- **A showcase item** becomes head + value / detail + value / plot, with the same reserved
  numeral track.

Measured at 390 and 320: no horizontal overflow, tick labels 12 px, and the axis zero, the
strip spine and every row spine on one x to three decimals.
