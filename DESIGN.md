# DESIGN.md — Contact Luck visual and product design

Canonical source of truth for how Contact Luck looks and behaves. Product intent lives in
`PRODUCT.md`; terminology in `CONTEXT.md`; code layout in `ARCHITECTURE.md`. This file
governs the dashboard only — it never overrides a product principle or a research rule.

This is a router, like `CLAUDE.md`. The thesis, the principles and the reference world are
below in full because they apply to every decision. Everything else — measured values,
tokens, contrast ratios, component rules — lives in a focused file under `docs/design/`.
**Load only what your task needs.**

Baseline evidence for every claim in this system: `outputs/figures/ui_baseline_2026-08-26/`
(gitignored; see its `README.md`). Build under audit: dashboard v1.4.1, snapshot
`2026-08-14`, commit `05dbf24`.

---

## Load this for your task

| Doing | Read |
|---|---|
| **Anything at all** | `docs/design/guardrails.md` — 18 anti-patterns + the preserved product invariants. Non-negotiable, and short. |
| Page structure, page order, copy placement, identity, nav, search | `docs/design/information-architecture.md` |
| Type families, sizes, numeric setting | `docs/design/typography.md` |
| Any colour, token, or contrast question | `docs/design/color.md` |
| Spacing, measures, grid, radii, borders, "should this be a card?" | `docs/design/layout.md` |
| The leaderboard or any table | `docs/design/tables.md` |
| Any chart, strip, or SVG figure | `docs/design/dataviz.md` |
| Hover / focus / selected / sorted / loading / empty / error, motion | `docs/design/interaction.md` |
| Breakpoint behaviour | `docs/design/responsive.md` |
| Contrast, semantics, keyboard, labels, targets | `docs/design/accessibility.md` |

Whole-system work (a new page, a design direction) reads this file plus `guardrails.md`
plus the two or three files the page actually touches — not all ten.

---

## Visual thesis

Contact Luck should read like a scorekeeper's sheet that a statistician has audited: a
ruled, fixed-column page where every number is entered against a **shared zero line**, and
every claim carries its uncertainty and its provenance in the same breath. The interface's
job is to let a reader see a whole league's luck as one shape, then walk that shape down to
a single batted ball without ever losing the scale it was measured on. Numbers are set like
a register — tabular, signed, aligned. The prose that says what the numbers do *not* mean
is set like a journal, in a serif that asks to be read rather than skimmed. Nothing floats:
sections are separated by rules and space, not by boxes. One warm accent marks what the
frozen snapshot made official; blue and red are spent only on the sign of a score, never on
decoration.

The product's best idea is already in the DOM and is currently hidden by its styling: every
leaderboard row's interval SVG is drawn on a **common domain with the zero line at a fixed
x** (verified: `x1="101.8"` on all 124 rows). That column is already a strip plot of the
league. The redesign's central move is to stop treating it as a table-cell ornament and
make it the spine of the whole system.

---

## Design principles

1. **One zero line, everywhere.** Every signed Contact Luck quantity — leaderboard row,
   player hero, component decomposition, play-level gap — is drawn against the same
   zero-anchored scale with the same domain. *Consequence:* per-player or per-chart
   auto-scaling is banned; a snapshot fixes one league-wide domain and every strip uses it.
2. **Uncertainty is a rendered fact, never a verdict.** An interval that crosses zero is
   drawn exactly like one that does not. *Consequence:* no fading, muting, dashing,
   greying, italicising, or "not significant" affordance is ever attached to
   `overlaps_zero`. This overrides any visual argument for de-emphasis.
3. **Rules and space group things; boxes do not.** *Consequence:* a page section never
   becomes a bordered rounded rectangle. The three admissible cards are defined in
   `docs/design/layout.md`.
4. **Numbers get the typography budget; prose gets the reading budget.** *Consequence:* two
   type families with different jobs, tabular lining figures as a primitive, and a prose
   measure that is narrower than the data measure on the same page.
5. **Every structural device encodes something true.** *Consequence:* a divider means
   "different tier of information," an accent means "this is the frozen official record,"
   an uppercase eyebrow means "this labels a measured quantity." Decoration applied
   uniformly to every heading is banned.
6. **Density is earned, not avoided.** This is an analytics product; a table of 124 rows is
   the feature. *Consequence:* whitespace is spent separating tiers, never padding a box.
   Compression inside a tier is a win, not a compromise.
7. **A control that cannot be operated does not exist.** *Consequence:* keyboard
   reachability, a programmatic name, and a visible focus state are properties of the
   primitive, not a later audit pass. `display: none` on an operable control is banned.
8. **Format the data rather than engineering around it.** *Consequence:* when a raw string
   breaks a layout (an ISO timestamp, a `snake_case` status code), the fix is a display
   format and a scoped break rule — not a global `overflow-wrap` escape hatch.

---

## Reference world

Three references, each contributing exactly one thing. Do not add a fourth, and do not
blend them into a pastiche.

| Reference | What is borrowed | What is explicitly not borrowed |
|---|---|---|
| **The baseball scorebook** (printed scorekeeping sheet) | The ruled fixed-column frame; the running tally read down a column; the discipline that every mark means exactly one thing | Hand-drawn marks, diamond glyphs as decoration, pencil/paper texture, novelty scorekeeping symbols |
| **The statistical abstract / league register** | Numeric typography, dense ruled tables, and an apparatus register (footnotes, status codes, provenance) that is present but typographically subordinate | Century-old ornament, all-serif setting, justified columns |
| **The measuring instrument** (a zero-anchored gauge face) | The shared zero spine, tick language, and the idea that the reader reads *position against a scale* rather than a decorated cell | Skeuomorphic dials, bezels, needles, LED/dot-matrix faces |

Prohibited references: Baseball Savant's visual system, ESPN/broadcast chrome, team
colours or logos, stadium photography, stitched-baseball motifs, any "novelty baseball"
theme.

---

## Standing decisions

Settled; do not re-litigate. Reasoning and measured values are in the linked file.

1. Grotesk for everything measured; **serif for everything argued** — a deliberate inversion
   of the usual pairing. → `typography.md`
2. **Two grades per semantic colour** — text ≥ 4.5:1, mark ≥ 3:1. Root cause of every
   contrast failure: a chart palette was adopted as a UI palette. → `color.md`
3. The amber accent means **"the frozen official record"**, and is removed on user re-sort.
   → `color.md`, `tables.md`
4. **The verdict cell is one cell** — point estimate and interval on the shared zero axis.
   → `tables.md`
5. **Three admissible cards**; everything else gets rules and space. Card sets cap at six.
   → `layout.md`
6. **Four column tiers with a declared shedding order**; Identity and Verdict never shed.
   → `tables.md`
7. **Format the data rather than engineering around it.** → principle 8, `responsive.md`
8. Three radii, a 9-step type scale, an 8 px spacing scale, two measures per page.
   → `layout.md`, `typography.md`
9. **Identity is name-first; no layout may depend on imagery.** Headshots and team context
   are permitted if and when the data model supports them, and must degrade cleanly to
   name-only. → `information-architecture.md` § Player identity

The one remaining open item is the **web-font budget** (~60–80 KB for two self-hosted
families), which the owner has not ruled on. It blocks nothing: the grotesk/serif split
holds with system faces if it is declined. → `typography.md` § Font budget.
