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
