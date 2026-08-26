# Responsive behavior

Breakpoints and the per-element behaviour matrix. Entry point: `DESIGN.md`.
Read with `docs/design/tables.md` (shedding order) and `docs/design/dataviz.md`
(charts re-author, they do not rescale).

Breakpoints: **≥ 1280** desktop · **1024–1279** laptop · **640–1023** narrow · **< 640**
mobile. (Consolidates the current 640 / 800 / 900 / 1440 set.)

| Element | Desktop | Laptop | Narrow | Mobile |
|---|---|---|---|---|
| **Navigation** | Inline row, search inline | Inline row, search inline | Routes in one row, search promoted to a full-width field | Disclosure menu, 44 px targets, search as a full-screen sheet. **Never a two-row wrap.** |
| **Leaderboard** | 6 columns, merged verdict cell | Evidence tier merges to one sample cell | Two-line ruled row | Two-line ruled row |
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
~1440, ~1280 and ~390 via the global Playwright MCP.
