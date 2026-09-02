# Three redesign directions — 2026-08-26

Phase 3 of the redesign. **Proposals only. Nothing here is implemented, and no application
code, CSS, template, JS, or asset has been modified.** Awaiting the owner's decision.

Governed by `DESIGN.md` and `docs/design/guardrails.md`; product semantics from
`PRODUCT.md` and `CONTEXT.md`. Baseline evidence: `outputs/figures/ui_baseline_2026-08-26/`.

---

## A note on the reference world

The brief offered *sports editorial* and *baseball broadcast information design* as candidate
lineages. **Both are prohibited by `DESIGN.md` § Reference world**, which fixes exactly three
references — baseball scorebook, statistical abstract, measuring instrument — and explicitly
bans "ESPN/broadcast chrome" and sports-site cliché. Rather than break the design system to
manufacture variety, distinctness here comes from **which of the three approved references
dominates**:

| Direction | Dominant | Subordinate | Architecture it produces |
|---|---|---|---|
| **1 · The Register** | Scorebook | abstract, instrument | Horizontal ruled grid; fixed column positions shared across every route |
| **2 · The Zero Spine** | Measuring instrument | scorebook, abstract | A vertical axis at a fixed x that the whole site registers against |
| **3 · The Annual** | Statistical abstract | scorebook, instrument | Two-column document with a marginal apparatus |

This is a stronger source of separation than a fourth reference would have been: it forces
three genuinely different **page architectures**, not three skins on one layout.

---

## What all three hold constant

Stated once so the three sections below can be read as pure differences. Every direction
preserves, without exception:

- **Two independent official rankings**, both beginning at #1, both labelled with their
  frozen `public_labels` strings. Never "worst players."
- **Sign colour follows the point estimate's sign only.** An interval crossing zero renders
  identically to one that does not — no fading, muting, dashing, or "not significant" cue.
- **The league-wide shared interval domain** (zero at 0.463/0.464 of width), identical
  across the leaderboard and every player page. No per-chart auto-scaling.
- **Intervals ship beside every point estimate**, never behind a toggle.
- **Unqualified players** keep a page, a score, and an interval; receive no rank; are never
  de-emphasised; their status chip is neutral.
- **"Sorted view — not the official Contact Luck ranking"** on any re-sort, never by default;
  the amber official accent leaves the rank column; `aria-sort` on the header; a polite live
  region announces the change; the Rank column always restores the frozen order.
- **Retrospective, not predictive** framing in the lead, per-page callouts, and footer.
- **Explore behaviour**, play-level drilldown, the "Show all 12" one-way reveal, the empty
  state that hides table and count together, and the "Play not found" route back.
- **Dark mode**, and every requirement in `docs/design/accessibility.md`.
- **The dashboard displays; it never computes.** All three directions are renderings of
  numbers already in the snapshot. Where a direction adds a figure, that figure is generated
  at build time in `dashboard/visuals.py` from values the snapshot already contains.
- Three radii, the 9-step type scale, the 8 px spacing scale, two measures per page, and the
  two-grade colour system from `docs/design/color.md`.

All three also use the same two families (`IBM Plex Sans` / `Sans Condensed` / `Mono` with
`Source Serif 4`, or the system grotesk/serif fallback if the font budget is declined). They
differ enormously in the **ratio** in which those families are used — which is a typographic
decision, not a token change.

---
---

# Direction 1 · The Register

## 1. Name

**The Register.**

## 2. One-sentence concept

The entire site is one continuous ruled scoring sheet on a fixed column grid, where the same
four column positions — identity, verdict, evidence, apparatus — hold the same x coordinate
on every route, so learning the leaderboard teaches you the whole product.

## 3. Reference lineage

**Scorebook-dominant.**

- **Borrowed from the scorebook:** the ruled fixed-column frame; the running tally read down
  a column; the discipline that every mark means exactly one thing; the sense of a *sheet*
  rather than a screen — content sits directly on the page ground with rules doing all
  grouping, and nothing is contained in a box.
- **Borrowed from the statistical abstract:** numeric typography and a footnote register at
  the foot of each table.
- **Borrowed from the instrument:** only the zero spine itself.
- **Explicitly NOT borrowed:** hand-drawn marks, diamond glyphs, pencil/paper texture, ruled
  notebook backgrounds, K/6-4-3 scorekeeping notation, any faux-print artefact. There is no
  paper simulation anywhere — the reference is the scorebook's *grammar*, not its surface.

## 4. Why it fits Contact Luck

Contact Luck is an **additive** metric in real run units, with a fixed four-tier column
structure that recurs on every surface. A register is the form that additive, tiered,
signed quantities have taken for two centuries. It is also the only direction that makes the
product's four column tiers (`docs/design/tables.md`) into a *site-wide* contract rather than
a table-local one — which is the highest-leverage move available given that the leaderboard
is the product.

## 5. First 5 seconds

A ruled masthead line reading `CONTACT LUCK` with the routes as tab-stops and
`DATA THROUGH AUG 14` inked amber at the right. Under it, one serif sentence at reading
width. Under that, two tab-stops sitting on the table's top rule — *Most favorable realized
luck* / *Least favorable outcomes relative to expectation* — and then, immediately, rows. The
reader sees a signed column of numbers with decimal points aligned and a continuous amber
line running down the middle of the interval column. They understand within five seconds that
this is a ranked record, that the numbers are signed, and that the amber line is zero.

## 6. Homepage composition

- **Header:** one ruled line, 56 px. Wordmark in condensed grotesk, 14 px, uppercase,
  +0.06em. Routes as evenly spaced tab-stops. Search as a recessed 220 px field at the right.
  `DATA THROUGH AUG 14` right-aligned, amber. A 1 px rule closes it. No logo, no icon set.
- **Product explanation:** `h1` at 30 px, then **one** serif sentence at 68ch, then a single
  muted line — "Retrospective, not a projection. How it works →". Total ~140 px, replacing
  the baseline's 786 px of preamble.
- **Favorable / unfavorable:** a tab pair rendered as two ruled tab-stops sitting *on* the
  table's top rule, like the two facing pages of a scorebook. Real `<button role="tab">`,
  `aria-selected`, 2 px amber underline on the active one.
- **Filters:** a single ruled row above the header row, each input with a visible 11 px
  uppercase label (never a placeholder-as-label). `124 qualified hitters` right-aligned.
- **Above the fold at 1440 × 900:** masthead, lead, tabs, filter row, column heads, and
  **~11 data rows**.
- **Intentionally subordinate:** the example-play proof strip moves *below* the table as a
  three-entry ruled footer band; the extended explainer moves to `/demo/`.

```
┌───────────────────────────────────────────────────────────── 1440 ──┐
│ CONTACT LUCK   Leaderboard Explore How-it-works Methodology Data     │
│                                      [ search        ]  DATA THRU ▪  │
├──────────────────────────────────────────────────────────────────────┤
│ Contact Luck                                                         │
│ How much better or worse a hitter's batted-ball results have been    │
│ than the contact itself was worth, in runs per 100 eligible BBE.     │
│ Retrospective, not a projection. How it works →                      │
│                                                                      │
│ ┌ Most favorable realized luck ┐ ┌ Least favorable outcomes ┐        │
│ ══════════════════════════════════════════════════════════════════   │
│ MIN BBE [    ]  MIN GAMES [    ]  PLAYER [        ]  124 qualified   │
│ ─────────────────────────────────────────────────────────────────    │
│ RK  PLAYER              RUNS/100  (95% INTERVAL)   TOT  BBE   G      │
│ ─────────────────────────────────────────────────────────────────    │
│  1  Pete Crow-Armstrong   +7.62   │   ├───●──┤     24.5  321  120    │
│  2  ...                   +6.98   │  ├────●───┤    22.1  298  118    │
│  3  ...                   +6.41   │   ├──●─┤      19.8  355  121    │
│                                   ▲ amber zero spine, full height    │
└──────────────────────────────────────────────────────────────────────┘
```

## 7. Leaderboard

**Six columns at fixed x, 36 px rows, hairline rule between, no zebra fills.**

| Column | Width | Treatment |
|---|---|---|
| **RK** | 48 | Tabular, amber ink while the view is official; ink-primary once user-sorted |
| **PLAYER** | 260 | 14 px/500, never breaks mid-word (`overflow-wrap` is scoped away from names) |
| **RUNS / 100 (95% INTERVAL)** | 420 | **One merged verdict cell** |
| **TOT RUNS** | 90 | 1 dp, tabular |
| **BBE** | 80 | integer |
| **G** | 70 | integer |

- **The verdict cell** splits internally into a fixed 90 px numeric sub-column (right-aligned,
  18 px/600, explicit sign, U+2212 minus, so **every decimal point aligns down 124 rows**)
  and a 330 px strip on the league-wide fixed domain. The zero spine is a continuous 2 px
  amber rule running the **full height of the table body**, not a per-row tick.
- **Interval width** sheds entirely (Derived tier) — it is recoverable from the strip, and
  its absence fixes the baseline's silent clipping at every desktop width. The table declares
  a `min-width` contract and shows a visible scroll affordance.
- **Official vs. sorted:** re-sorting removes the amber from RK, shows
  "Sorted view — not the official Contact Luck ranking" as a ruled line directly under the
  header row, sets `aria-sort`, and announces via a polite live region. Clicking RK restores
  the frozen order.
- **Unqualified players** never appear here (they are not ranked) but appear in global search
  with a neutral `Reported, not ranked` entry in the rank position — same type, same weight.
- **Density:** the highest of the three. 124 rows in ~4,500 px; the table is meant to be
  scrolled fast and read as a column.

**Mobile (390):** a two-line ruled row. Rank in a 28 px gutter; name at 16 px/600 owning a
full-width line (measured: the longest name in the 628-player set, "Christian
Encarnacion-Strand", is 224 px against 310 px available, so mid-word breaks become
structurally impossible); value right-aligned on the same line; second line carries
`321 BBE · 120 G` and the strip at the same fractional zero position. Filters collapse to one
disclosure row. Routes move to a disclosure sheet as a ruled list with 44 px targets.

## 8. Player page

The reading sequence is delivered by **the same column grid as the leaderboard** — the score
lands at exactly the x where the reader's eye already was.

1. **Who** — name as a masthead entry at 30 px condensed 600, `batter_id` and data-through in
   the apparatus register beneath, muted, 12 px.
2. **What** — the signed figure at 44 px in the verdict column's x position.
3. **How extreme** — both official ranks written as answers on one ruled line:
   `#3 of 124 most favorable · #122 of 124 least favorable`. Not pills.
4. **How uncertain** — the interval strip sits **immediately beneath the number, sharing its
   column**, one type step down in brackets. Fixes the baseline defect where the strip was
   orphaned in a full-width band.
5. **Why** — component decomposition as a ruled sub-ledger: four rows (contact, unexplained
   residual, defensive execution, advancement), each with its value in the verdict column and
   its **status code in the apparatus column**, footnoted at the table foot.
6. **Which plays** — a ruled "Plays" band linking into Explore, filtered to this hitter via
   the URL. Closes the baseline dead-end.
7. **What next** — both rankings, methodology, that hitter's plays.

**Supporting stats** (Total Contact Luck Runs, Eligible BBE, Scored Games, Provisional share)
become **one ruled box-score line**, not four bordered tiles. **Provisional/small-sample:**
provisional share is a value in that line with its own footnote; an unqualified hitter shows
`Reported, not ranked` where the ranks line would be, in identical type.

**Trend** is a ruled sparkline strip aligned to the same grid, with true snapshot dates and
gaps never interpolated.

## 9. Explore

A **register of plays**, not a form with cards under it. The hitter picker is a real combobox
(`role="combobox"`, `aria-expanded`, `aria-controls`, `aria-activedescendant`; Up/Down, Enter,
**Escape closes**) that **writes the selection into the URL**, making a hitter's plays
linkable from their player page.

Showcase plays lead as up to six cards — the one admissible card use (repeated instances of
one object type scanned across). "Show all 12" reveals a **ruled list**, not six more cards,
per the six-card cap. The hitter's own plays render as a ruled play ledger with the same
verdict column x. Filters are a ruled row with visible labels. Empty state hides table and
count together.

## 10. Individual play page

A **play ledger**, read top to bottom in the required order.

- **CONTACT** — a 4-across ruled stat line: EV (mph), Launch angle (°), Batted-ball type,
  Spray (approximate). The word *approximate* is in the label, not a footnote, per
  `CONTEXT.md`.
- **EXPECTATION** — the frozen model's five outcome classes as a five-row ruled probability
  ledger (out / single / double / triple / home run), each with a horizontal bar on a shared
  baseline. The **recorded result's row carries a caret in the apparatus column**, so
  expected and actual are one figure rather than two blocks.
- **ACTUAL RESULT** — the recorded result set as a register entry, distinct from the
  run-value accounting beneath it.
- **RUN VALUE GAP** — three entries in the verdict column's x: `Expected RV`, `Observed RV`,
  and the gap, ruled off like a balance.
- **CONTACT LUCK CONTRIBUTION** — the resulting per-play figure, signed, in the same column.
- **Field diagram** — demoted from its baseline position of greatest prominence to a small
  labelled figure in the apparatus column, `role="img"` with an `aria-label` (the baseline has
  no role, label, title, or `aria-hidden` at all).
- **Sensitivity** (showcase plays only) — a ruled sub-table whose values update in place; the
  counterfactual figure appears **in the same verdict column x** as the real one, so the
  comparison is a column read.

## 11. Methodology

A **definitions ledger**: term · definition · where it appears in the product, on the same
ruled grid, with a sticky section index at the left. Prose passages run at the 68ch prose
measure in serif. It reads as an authoritative reference sheet rather than a paper, and it is
never a prerequisite — the leaderboard's lead links to `/demo/`, not here.

## 12. Status / provenance

The lead answers freshness in one line:
`Current through August 14, 2026 · genuine prospective snapshot · built Aug 15, 13:35 UTC`.
The snapshot history becomes a proper ruled ledger with a `min-width` contract, **dates
formatted at the source** (`Aug 15, 2026, 13:35 UTC` with the ISO value in `<time datetime>`)
— which removes the unbreakable 32-character token that produced the baseline's 31 px
columns, 682 px rows, and 8,363 px page. Reason codes sit in mono in the apparatus column
behind `<details>`. Below 1024 the table becomes a definition list per snapshot.

## 13. Demo

The two compared batted balls become **two ledger columns side by side sharing the same
fixed rows** — a true comparison sheet, which is what the demo has always wanted to be. The
ball-flight animation plays as a figure above (permitted motion: it teaches the metric). The
counterfactual simulator is the second admissible card — an interactive instrument with a
defined input surface.

## 14. Typography

- **Brand/display:** condensed grotesk, 600, uppercase, +0.06em. Sparing — wordmark and `h1`.
- **Measured/numeric:** the dominant surface. Tabular lining figures applied once as a `.num`
  primitive. Explicit signs, U+2212 minus, fixed precision per quantity.
- **Prose:** serif, but **rationed** — the single lead sentence, methodology, and the honesty
  callouts. This is the direction where the serif appears least.
- **Tables:** 14 px/400 body, 11 px uppercase condensed heads.
- **Metadata:** 12 px grotesk in `--ink-muted`, never below 12 px.

Ratio: roughly 85% grotesk / 15% serif. The most clerical of the three.

## 15. Color

The most restrained use of colour of the three. **Amber is rule ink** — the zero spine, the
official rank column, the active tab-stop's underline, the data-through badge, the focus
ring. **Sign colours appear only on the numeral and its dot**, never on a rule, border, or
label. Neutrals carry everything else via the three-plane tonal system (page < recessed <
surface). `--accent-field` green is confined to field diagrams.

**Dark mode:** the sheet inverts to a dark ledger — `#0d0c0a` ground, rules at `#322f29`,
amber at `#e0aa57`. Because the rules do the grouping, dark mode is a straight token swap
with no re-composition.

## 16. Geometry

- **Radius:** `0` on tables, rules, sections, charts, figures; `4px` on inputs and buttons;
  `999px` on status pills only.
- **Borders:** hairline 1 px only; the two exceptions are the 2 px focus ring and the 2 px
  zero spine.
- **Rules do all grouping.** 48 px between sections desktop / 32 px mobile, delimited by a
  hairline.
- **Page edges:** content sits in the 1200 px data measure on the page ground; the masthead
  rule and section rules run the **full measure**, not the full viewport.
- **What does NOT get a card:** page sections, callouts, stat readouts, charts, figures,
  tables, page headers, the player hero, the component decomposition, the trend, the status
  history. **Only two things get a card:** Explore's showcase play instances (capped at six)
  and the demo simulator.

## 17. Player imagery

**Recommendation: no headshots.** A register has no portraits, and adding one would be the
only element on the page that is not a mark or a number. The direction's whole argument is
that the fixed column grid is the identity system; a portrait column would break the grid's
claim that every column position means one thing. Identity is name-first everywhere, which
this direction already satisfies structurally.

## 18. Data visualization

- **Shared-zero intervals:** a 2 px sign-coloured line with a ≥8 px dot and a 2 px surface
  ring, on the fixed league domain, in the merged verdict cell. Near-zero values (|est| <
  0.10) render centred on the spine with an explicit sign — small and uncertain reads from
  the shared axis, with no de-emphasis.
- **Trends:** a ruled sparkline aligned to the grid; true dates, gaps never interpolated,
  zero reference line, ≥3 y labels. Re-authored (not rescaled) on mobile so SVG text stays
  ≥12 CSS px — the baseline's 640×220 viewBox at 350×120 put labels at ~5.5 px.
- **Decomposition:** a signed ruled sub-ledger with the exact-value table retained.
- **Probabilities:** five bars on a shared baseline with the recorded result marked in place.
- **Comparisons:** always a column read on the shared grid — the demo's two plays, the
  sensitivity counterfactual, both rankings.

## 19. Desktop feel (1440)

A pale, unboxed sheet. One thin rule under a masthead of small uppercase type. Two short
paragraphs of serif at a narrow measure, floating left, with a large empty right field —
deliberately, because the table below claims that width. Then a hard horizontal rule, two
tab-stops sitting on it, and a dense block of ruled rows filling the rest of the screen: a
column of aligned signed decimals at the left of the verdict field, and to their right a
vertical amber thread with short blue and red ticks radiating either side of it, growing
shorter and crossing more often as the eye travels down. No boxes, no shadows, no fills, no
icons. It looks like a document that was set, not a UI that was assembled.

## 20. Mobile feel (390)

Not a stack — a **re-authored row**. Each hitter becomes a two-line ruled entry: rank and
name and value on line one, evidence and interval strip on line two, hairline between
entries. The six-column table does not exist at this width; it has been replaced. The
masthead compresses to wordmark + search + menu, with routes in a ruled disclosure sheet at
44 px per target. The status history is a definition list, not a table. Charts change form
rather than shrinking.

## 21. Signature design move

**The persistent verdict column.** The x position of the Runs/100 figure is identical on the
leaderboard, the player page, the play page, the demo comparison, and the sensitivity
counterfactual. Navigate from a row to that player's page and the headline number appears
exactly where your eye already was. It is a register's central discipline — one column, one
meaning, everywhere — and it is not decorative: it is what makes cross-route comparison
physically effortless.

### Anti-generic test

| Generic failure | How this direction forecloses it |
|---|---|
| SaaS dashboard / KPI tiles | There are no tiles. The four player stats are one ruled box-score line. |
| Endless rounded cards | Exactly two card uses site-wide, both justified by `layout.md`. |
| Generic hero | The hero is one sentence and 140 px tall; row 11 is above the fold. |
| Excessive whitespace | Whitespace only between tiers; 36 px rows, 124 of them. |
| Arbitrary pills | One pill use: qualification status. |
| Random gradients / glass / neon | No gradient, shadow, glow, or blur anywhere; depth is tonal. |
| Arbitrary iconography | Zero icons. Structure is carried by rules and type. |
| Equal visual weight for every statistic | Four explicit column tiers with a declared shedding order. |
| Decorative charts | Two figure types, each answering one named question. |
| Sports clichés | No team colour, logo, diamond glyph, or broadcast chrome. |
| **The swap test** | Replace "Contact Luck" with a crypto/finance term and the fixed verdict column carrying a signed point estimate *and its game-clustered interval as one cell on a league-wide shared domain* has nothing to attach to. The grid exists because the metric has four tiers and one zero. |

## 22. Risks

- **Too dense / reads as a spreadsheet.** The real risk. Guard: the serif lead, 48 px section
  rules, and generous vertical space *between* tiers; the density lives strictly inside a
  tier. If it starts feeling like an export, the fix is more air between sections, never
  fewer rows.
- **Too austere / cold.** Guard: the serif prose and the demo's animation carry warmth; the
  amber spine gives the page one deliberate note of colour.
- **Gimmicky.** Low risk — the only conceit is the shared column, which is functional.
  Guard: never *label* the grid; it should be felt, not announced.
- **Too technical.** Guard: `/demo/` remains the conversion path and is linked from the lead.
- **Too sports-broadcast.** Structurally impossible here.
- **Too generic.** Guard: if a bordered box appears anywhere outside the two admissible card
  uses, the direction has been lost.

## 23. Implementation complexity

**MEDIUM.**

- `static/style.css` restructures around tokens and a fixed grid — large but mechanical.
- `templates/_macros.html` changes for the merged verdict cell; real tab buttons replacing
  the `display:none` radio pattern; `<th aria-sort><button>`; labelled filter inputs; a skip
  link in `base.html`.
- `visuals.py` re-authors the interval strip at two widths (desktop cell, mobile row) and
  adds the continuous spine as a table-level element rather than a per-row tick.
- Mobile two-line row is a new macro, not a media query on the existing table.
- `tests/test_dashboard_responsive_overflow.py` and `test_dashboard_isolation.py` are the
  two most likely to be tripped; the `min-width` contract should get a test.
- **No new data.** Every value already exists in the snapshot.

---
---

# Direction 2 · The Zero Spine

## 1. Name

**The Zero Spine.**

## 2. One-sentence concept

The shared zero axis stops being a thing inside charts and becomes the building's structure —
one continuous vertical rule at a fixed x that every signed quantity on the site is registered
against, so the leaderboard *is* a strip plot of the league rather than a table containing one.

## 3. Reference lineage

**Measuring-instrument-dominant.**

- **Borrowed from the instrument:** the zero-anchored spine; tick language; the premise that
  the reader reads *position against a scale* rather than a decorated cell; the discipline
  that a scale, once fixed, never moves.
- **Borrowed from the scorebook:** the ruled row and the fixed reading order.
- **Borrowed from the abstract:** the footnote register and the numeric setting.
- **Explicitly NOT borrowed:** skeuomorphic dials, bezels, needles, gauge faces, LED or
  dot-matrix type, tick marks used ornamentally, anything that looks like an instrument
  *panel*. The instrument contributes a measuring discipline, not a machine aesthetic. There
  is no bevel, no housing, no chrome.

## 4. Why it fits Contact Luck

`DESIGN.md` records that the product's best idea is already in the DOM and hidden by its
styling: every leaderboard row's interval SVG is already drawn on a common domain with zero
at a fixed x (`x1="101.8"` on all 124 rows), and the player pages use the same domain
(0.464). **The column is already a strip plot of the league.** This direction is the literal
execution of that finding. It is also the only direction in which the metric's two hardest
truths — that the scale is shared, and that most intervals cross zero — are visible without
being read.

## 5. First 5 seconds

A thin header, then a real horizontal axis with ticks at −8 −4 **0** +4 +8, and beneath the 0
tick an amber line dropping down the page. Immediately under the axis, all 124 qualified
hitters appear as one row of dots on that axis — a visible league distribution, clustered
near the spine with a few outliers pushed right and left. Below it, the ranked rows begin, on
the same axis. The reader understands, without reading a word: there is a zero, most players
are near it, some are far, and blue is on one side and red on the other.

## 6. Homepage composition

- **Header:** a thin rule with the wordmark left and routes right; the **spine descends from
  a tick in the header**, and `DATA THROUGH AUG 14` sits at the spine's head in amber, the
  way an instrument carries its scale label. Search sits at the right.
- **Product explanation:** `h1`, one serif sentence, and the retrospective line — the same
  140 px budget as Direction 1. Positioned to the **left** of the spine, so prose occupies
  unfavorable territory and does not straddle zero.
- **Favorable / unfavorable:** a tab pair of real buttons whose selection **pivots the table
  around the spine** — the axis and the spine hold still, ranks re-origin at the negative end,
  and rows re-order. The frozen `public_labels` strings label each; each begins at #1; row
  re-ordering is permitted motion (position change is the payload), disabled under
  `prefers-reduced-motion`.
- **League distribution figure:** one build-time strip of 124 dots on the shared axis, sitting
  directly beneath the axis header as "row zero". It answers a real question — *what does the
  league look like and where do the extremes sit* — and is the 10-second answer from the
  reading budget delivered as a figure. Hovering a dot highlights its row; it is not a hero.
- **Above the fold at 1440 × 900:** header, lead, tabs, axis, distribution strip, and
  **~8 data rows**.
- **Intentionally subordinate:** evidence statistics move to a muted right rail; the proof
  strip moves below the table.

```
┌───────────────────────────────────────────────────────────── 1440 ──┐
│ contactluck    Leaderboard  Explore  How it works  Methodology  Data │
│                                        [ search ]     DATA THRU ▪    │
├──────────────────────────────────────────────────────────────────────┤
│ Contact Luck                          ▪                              │
│ How much better or worse a hitter's   ▪                              │
│ results were than the contact was     ▪                              │
│ worth. Retrospective, not a           ▪                              │
│ projection.                           ▪                              │
│                                       ▪                              │
│ [Most favorable] [Least favorable]    ▪                              │
│         −8      −4       0      +4      +8        ← shared axis      │
│  league  ·· ···:::∷∷∷▪∷∷∷:::··· ·  ·                  BBE     G      │
│  ─────────────────────────────────▪──────────────────────────────    │
│   1  Pete Crow-Armstrong          ▪  ├────●───┤ +7.62   321   120    │
│   2  ...                          ▪ ├─────●──┤  +6.98   298   118    │
│   3  ...                          ▪  ├───●─┤   +6.41    355   121    │
│                                   ▲ the spine: one x, every route    │
└──────────────────────────────────────────────────────────────────────┘
```

## 7. Leaderboard

**The chart is the table.** Three zones: a 300 px identity gutter, a 720 px plot field with
the spine at 0.463 of it, and a 180 px evidence rail.

- **Rank** — tabular, amber while official. **Player** — 15 px/500 in the gutter.
- **The verdict is the entire row.** A 2 px sign-coloured interval line with a 9 px dot and a
  2 px surface ring, drawn on the league-wide fixed domain. **The value is printed at the
  outboard end of the interval**, 14 px tabular with an explicit sign.
- **Trade-off, stated plainly:** values float at their axis position, so decimal points do
  **not** align down the column. You lose column-scanning of the numeral; you gain a directly
  readable distribution of 124 players. This is the sharpest difference between Directions 1
  and 2 and is the core of the choice.
- **Supporting statistics** (`321 BBE · 120 G`) sit in the evidence rail, right-aligned,
  muted, 12 px. Total runs is available in the rail at ≥1280 and merges into the sample cell
  below that.
- **Uncertainty** is the row's length. An interval crossing the spine is drawn identically to
  one that does not — this direction makes that invariant *load-bearing* rather than merely
  observed.
- **Official vs. sorted:** re-sorting removes amber from RK and shows the "Sorted view" line;
  the axis and spine never move, which is precisely the reassurance a re-sort needs.
- **Density:** medium. 44 px rows, because each row carries a plot. 124 rows in ~5,500 px.
- **Filters** sit in the identity gutter's width so they never straddle the spine.

**Mobile (390):** the spine survives, which is the whole point. Name on line one; line two is
a 310 px plot field with the spine at 143 px, the interval drawn on it, and the value at a
fixed right-aligned position (the one concession — at this width a floating value collides).
A **sticky 24 px axis strip** pins to the top of the list while scrolling, so the spine always
has a visible anchor. That is a structural transformation, not a stack.

## 8. Player page

1. **Who** — name at 30 px above the axis; `batter_id` and data-through muted beneath.
2–4. **What / how extreme / how uncertain, as one figure.** The hero is a full-width instance
   of the same axis carrying **the league distribution in gridline ink** with *this player's*
   interval and dot drawn over it at double weight in sign colour, and the score printed at
   the dot at 44 px. One glance answers all three questions: the value, the position against
   everyone, and the width of the doubt. Both official ranks are written beneath as answers:
   `#3 of 124 most favorable · #122 of 124 least favorable`.
5. **Why** — the strongest decomposition of the three: contact, unexplained residual,
   defensive execution and advancement as four signed bars **on the same spine**, stacked
   vertically, each labelled with its status code at the outboard end. They visibly sum to the
   headline. The exact-value table is retained beneath.
6. **Which plays** — a link into Explore pre-filtered to this hitter via the URL.
7. **What next** — both rankings, methodology, plays.

**Trend** is the one place the axis rotates: time on x, runs/100 on y, with **the same domain
on the y-axis** and a horizontal zero line, framed by its caption as the same scale turned on
its side. True snapshot dates; gaps never interpolated.

**Supporting stats** are a ruled line in the evidence rail. **Provisional share** is a value
with a status footnote. **Unqualified** hitters get the identical hero — the same axis, the
same league distribution behind them, their dot and interval at full weight — and simply no
rank line. Their status chip is neutral. This is the direction in which an unqualified player
is least de-emphasised, because the axis treats everyone identically by construction.

## 9. Explore

A discovery instrument. The hitter combobox (full ARIA semantics, Escape closes,
**URL-addressable**) sits in the identity gutter. Once a hitter is chosen, **their individual
plays appear as dots on a run-value axis** — the same grammar, one level down — so a season's
worth of batted balls is a distribution you can see before you filter it. Filters narrow the
distribution live; sorting re-orders the list beneath. Showcase plays lead as up to six cards
with "Show all 12" revealing a ruled list. Empty state hides table and count together.

## 10. Individual play page

The run-value axis is the hero, which finally puts the gap where the answer belongs.

- **CONTACT** — EV, launch angle, batted-ball type, spray (approximate) as a compact stat line
  in the gutter.
- **EXPECTATION** — the five outcome probabilities as horizontal bars on a shared left
  baseline, with **the recorded outcome's bar carrying a solid marker in place**.
- **ACTUAL RESULT** — the recorded result named beside that marker.
- **RUN VALUE GAP** — **the hero figure**: Expected RV and Observed RV as two ticks on a
  run-value axis with the span between them shaded and labelled. The gap is a *measured
  distance*, which is exactly what it is.
- **CONTACT LUCK CONTRIBUTION** — the signed per-play figure printed at the end of that span,
  on the same axis.
- **Field diagram** — a small labelled figure in the rail, `role="img"` with an `aria-label`,
  its caption stating that spray angle is approximate.
- **Sensitivity** (showcase only) — dragging EV or launch angle **slides the dot along the
  axis**. The movement is the information, which is the only justification this design system
  accepts for motion. Respects `prefers-reduced-motion` by stepping to discrete values.

## 11. Methodology

Prose at the 68ch measure; the spine degrades to a plain left rule on pages with no signed
quantity — it appears only where something is registered against it, never as decoration.
Each defined signed quantity (Contact Luck Runs, per-100, expected RV) carries a **miniature
inline axis** beside its definition, so the methodology teaches the same grammar the product
uses. Sticky section index; collapsed ToC below 1024.

## 12. Status / provenance

The best `/status/` of the three. The lead answers freshness in one line. The snapshot history
becomes **a freshness timeline** — snapshots as ticks on a date axis, which makes the real
gaps in coverage (e.g. the intentional 2026-08-07 gap) *visible information* rather than a
missing table row. Per-snapshot detail sits beneath as a definition list with formatted dates
(`Aug 15, 2026, 13:35 UTC`, ISO in `<time datetime>`) and reason codes in mono behind
`<details>`. The broken 31 px-column table is not fixed; it is replaced.

## 13. Demo

The two real batted balls become **two dots animating to their positions on one shared
run-value axis** — the existing ball-flight animation resolves into the axis, which is the
single clearest way to teach what the metric measures. The counterfactual simulator is an
admissible card: moving EV/launch angle slides the expected-RV tick and the gap re-shades
live.

## 14. Typography

- **Brand/display:** grotesk at regular width, not condensed — the spine already supplies the
  vertical tension, so the type stays calm.
- **Measured/numeric:** the largest numerals of the three (the player hero at `--fs-800`),
  because a value printed at a position must read instantly. Tabular figures, explicit signs,
  U+2212.
- **Prose:** serif for the lead, the honesty callouts, and methodology.
- **Tables:** minimal — the evidence rail is 12 px muted grotesk; there are few table cells to
  set.
- **Metadata:** 12 px floor; axis tick labels are **text** and held to the 4.5:1 text grade,
  not the 3:1 mark grade.

Ratio: roughly 75% grotesk / 25% serif, with uppercase reserved to axis ticks and column
heads only.

## 15. Color

The most colour of the three — but never a non-semantic hue. Every interval line is
sign-coloured, so the plot field carries blue on one side of the spine and red on the other,
and the page's colour *is* the data. **Amber is reserved to the spine, the rank column, the
data-through label at the spine's head, and the focus ring** — nothing else, because the
spine's amber must stay unambiguous. Marks use the mark grade (≥3:1); every numeral and axis
label uses the text grade (≥4.5:1). Field green stays inside field diagrams.

**Dark mode:** reads as an instrument face. Ground `#0d0c0a`, gridlines `#322f29`, and the
spine at `#e0aa57` becomes the brightest element on the page — the strongest dark theme of
the three, because a lit scale on a dark ground is the reference's natural state.

## 16. Geometry

- **Radius:** `0` on the plot field, axes, rules, and figures; `4px` on inputs/buttons;
  `999px` on status pills only.
- **Borders:** hairline; exceptions are the 2 px focus ring and the 2 px spine. The plot field
  has **no border at all** — it is defined by the axis above it and the spine through it.
- **Grouping:** by the axis and by rules. Nothing is boxed.
- **Page edges:** the spine runs the full height of the content column, edge to edge
  vertically, giving the page a strong asymmetric structure — the only direction of the three
  that is deliberately asymmetric.
- **What does NOT get a card:** everything except Explore's showcase instances and the demo
  simulator. In particular the player hero, the distribution figure, the decomposition, the
  trend, and the play page's gap figure are **never** boxed — boxing a figure that shares a
  global axis would visually break the claim that the axis is global.

## 17. Player imagery

**Recommendation: no headshots on any data surface.** A portrait beside a dot on a shared axis
competes with the mark for the same attention, and the axis's authority depends on nothing
else in the plot field asking to be looked at. The one place imagery would be admissible is
the **play page's context header**, where there is no shared-axis field — and even there it is
optional and non-structural. Identity is name-first throughout.

## 18. Data visualization

This direction *is* the visualization language.

- **Shared-zero intervals:** the primitive. 2 px line, ≥8 px dot, 2 px surface ring, fixed
  league domain, sign colours the mark. Near-zero (|est| < 0.10) sits centred on the spine
  with an explicit sign — the shared axis conveys "small and uncertain" with no de-emphasis.
- **Trends:** the same domain rotated; zero reference line; true dates; gaps never
  interpolated; re-authored viewBox on mobile so text stays ≥12 CSS px, degrading to a
  sparkline with endpoint values if labels will not fit.
- **Decomposition:** four signed bars on the shared spine, summing visibly to the total.
- **Probabilities:** horizontal bars on a shared baseline with the recorded outcome marked in
  place.
- **Comparisons:** always two marks on one axis — never two charts side by side.
- Every figure is `role="img"` with an `aria-label` stating the value in words; decorative
  marks are `aria-hidden`.

## 19. Desktop feel (1440)

An unusually quiet page with one very strong vertical gesture. Thin rule at top, small type,
a short serif paragraph sitting left of centre. Then a real measuring axis with numbered ticks
spanning most of the width, and from its zero a single amber line descending all the way to
the footer. Under the axis, a fine spray of 124 dots — dense at the middle, thinning to
singletons at both ends. Then rows: names down the left in calm grey type, and out in the
field, short horizontal blue and red bars at varying distances from the amber line, each with
a dot and a small number at its tip. The eye reads the field before it reads the names. There
is no box on the screen. It looks like a measurement, not a report.

## 20. Mobile feel (390)

The axis becomes a **sticky 24 px strip** at the top of the scrolling list, carrying the tick
labels and the spine's origin, so the spine has a fixed anchor no matter where you are in 124
rows. Each hitter is a two-line entry: name on line one; on line two a 310 px plot field with
the spine at 143 px, the interval drawn across it, and the value right-aligned at the row's
end. The evidence rail is gone; `321 BBE · 120 G` merges under the name at 12 px. The player
page hero keeps the full axis and the league distribution behind it — it is the one element
that must not simplify, because it is the product. Trend degrades to a sparkline with
first/last values.

## 21. Signature design move

**The spine.** One 2 px amber vertical rule at a fixed x, descending from a tick in the header
through the axis, through the distribution strip, through all 124 rows, through the component
decomposition on a player page, and through the run-value gap on a play page — never moving
between routes or breakpoints. It is the shared-domain invariant made into architecture: the
reason a `+7.62` and a `−0.06` are comparable across the whole site is that they are literally
measured against the same line.

### Anti-generic test

| Generic failure | How this direction forecloses it |
|---|---|
| SaaS dashboard / KPI tiles | There is no tile and no summary row; the summary is the distribution figure. |
| Endless rounded cards | Two card uses site-wide; boxing a figure is explicitly banned here. |
| Generic hero | The above-fold figure is the 10-second answer, clickable into rows, ~120 px tall. |
| Excessive whitespace | The plot field is full of data; whitespace is the axis's margin. |
| Arbitrary pills | One pill use: qualification status. |
| Random gradients / glass / neon | None. Dark mode is a lit scale, not a neon panel — no glow, no bloom. |
| Arbitrary iconography | Zero icons; ticks are labelled scale marks, not decoration. |
| Equal visual weight | Weight is literally position on a scale — the strongest possible hierarchy. |
| Decorative charts | Every figure shares one domain and answers one named question. |
| Sports clichés | No team colour, logo, or broadcast chrome; the instrument reference bans gauge skeuomorphism. |
| **The swap test** | The composition is meaningless without a *signed* metric on a *shared league-wide domain* with *intervals*. A crypto or AI dashboard has no fixed zero to build a page around. This is the least portable of the three. |

## 22. Risks

- **Gimmicky** — the spine becoming decorative on pages with no signed quantity. Guard: the
  rule that the spine appears **only** where something is registered against it; on
  `/methodology/` it degrades to a plain left rule.
- **Too sparse** — a strip-plot row carries less text than a table row. Guard: the evidence
  rail is not optional at ≥1280; `321 BBE · 120 G` is never dropped, only merged.
- **Loss of column scanning** — floating values break decimal alignment. Guard: accept it
  knowingly (it is the direction's central trade), keep tabular figures so widths are stable,
  and keep a sortable Total-runs column in the rail for readers who scan numerically.
- **Too technical / instrument-cold.** Guard: serif prose, the demo animation, and generous
  type sizes on the player hero.
- **Too broadcast-like.** Guard: no bezels, dials, needles, glows, or panel chrome — the
  banned half of the instrument reference is the enforcement.
- **Too dense at the extremes** — dots overlap near zero in the distribution figure. Guard:
  the 2 px surface ring on every mark, plus a build-time jitter or beeswarm layout computed
  from existing values.

## 23. Implementation complexity

**HIGH.** The most ambitious of the three, and the most engineering in `visuals.py`.

- New build-time figures: the league distribution strip, the shared axis header, the
  decomposition bar set, the run-value gap figure, the status freshness timeline.
- A **shared-domain layout contract** must be enforced in one place and consumed by every
  figure — the domain is already fixed per snapshot, but it becomes a first-class build-time
  object rather than an implicit property.
- The spine must be a page-level element aligned to a plot field across independent
  components, at every breakpoint. This is the hardest CSS in any of the three.
- The pivot interaction (tab switch re-origins ranks and re-orders rows) is client-side
  presentation over pre-computed data — no computation, but real JS.
- Mobile sticky axis and per-breakpoint viewBox re-authoring.
- **No new data.** Every value already exists in the snapshot; the distribution figure is a
  rendering of the 124 scores already on the page.

---
---

# Direction 3 · The Annual

## 1. Name

**The Annual.**

## 2. One-sentence concept

The site is a statistical annual with a scholarly apparatus: a wide data column carrying
numbered tables and figures, and a persistent right margin in which every definition, status
code, caveat and provenance note sits at the exact vertical position of the claim it
qualifies.

## 3. Reference lineage

**Statistical-abstract-dominant.**

- **Borrowed from the abstract:** numeric typography; dense ruled tables set to proper
  statistical-table conventions (top rule, header rule, bottom rule, **no interior verticals,
  no zebra**); numbered tables and figures with captions; and an apparatus register —
  footnotes, status codes, provenance — that is present but typographically subordinate.
- **Borrowed from the scorebook:** the fixed column order within each table.
- **Borrowed from the instrument:** the shared zero domain, rendered as a restrained 3 px
  gauge rule beneath the numerals rather than as a chart.
- **Explicitly NOT borrowed:** century-old ornament, drop caps, rules with finials,
  all-serif setting, justified columns, faux-paper grounds, small-caps everything, or any
  antiquarian styling. The reference is a *modern* statistical annual — the apparatus, not
  the patina.

## 4. Why it fits Contact Luck

This product's defining constraint is that **every number needs a caveat attached to it** —
the interval, the qualification status, the component status codes, the provisional share,
the "approximate" on spray angle, the retrospective framing. The baseline handles this by
stacking disclaimers into callout boxes, which is why the leaderboard is buried under 786 px
of preamble. A marginal apparatus solves it structurally: the caveat sits *beside* the claim
instead of *before* it. This is the only direction that makes statistical honesty a
typographic system rather than a sequence of interruptions.

## 5. First 5 seconds

A masthead with an edition line — `2026 season · data through August 14 · snapshot v1.1` —
over a contents rule. Then a short serif paragraph, and immediately beside it in the margin,
in small type, the definition of Runs / 100 and the words *Retrospective. Not a projection.*
Then a caption — **Table 1. Most favorable realized luck. Qualified hitters, 2026 season to
date. n = 124.** — and the table. The reader understands they are looking at a published
record with its terms defined in view, and that someone has been careful.

## 6. Homepage composition

- **Header:** wordmark in the display setting, an edition line beneath a hairline, and routes
  as a plain contents line. **No section numbering on navigation** — routes are not a
  sequence, so numbering them would be decoration. Numbering is reserved for tables and
  figures, which are genuinely cited.
- **Product explanation:** `h1`, one serif sentence at 68ch in the data column. The rest of
  the explainer moves to the **margin** as three short registered notes (definition,
  retrospective framing, unit). This is how the 786 px of baseline preamble is removed without
  removing any of its content — it moves sideways, not away.
- **Favorable / unfavorable:** a tab pair of real buttons above Table 1; switching changes the
  table caption with it, so the caption always states which record is on screen.
- **Leaderboard entry point:** Table 1, with a proper caption.
- **Table 1a — "Reported, not ranked."** The sub-threshold hitters, in identical typography,
  **ordered alphabetically** (never by score — small samples are reported, never ranked),
  behind a one-way disclosure. Margin note states the threshold (≥200 eligible BBE, ≥100
  scored games). This turns the unqualified-player invariant from a thing the design must
  avoid breaking into a **feature the design delivers**: those players are currently reachable
  only by search.
- **Above the fold at 1440 × 900:** masthead, lead, margin notes, tabs, Table 1's caption and
  **~9 rows**.
- **Intentionally subordinate:** everything in the margin, by construction — it is set two
  steps down in muted ink.

```
┌───────────────────────────────────────────────────────────── 1440 ──┐
│ CONTACT LUCK                                                         │
│ 2026 season · data through August 14 · snapshot v1.1                 │
│ Leaderboard · Explore · How it works · Methodology · Data & status    │
│ ─────────────────────────────────────────────────────────────────    │
│                                                                      │
│ Contact Luck                              ┊  RUNS / 100              │
│ How much better or worse a hitter's       ┊  Contact Luck Runs       │
│ batted-ball results have been than the    ┊  normalized to 100       │
│ contact itself was worth, in runs.        ┊  eligible batted balls.  │
│                                           ┊                          │
│ [Most favorable] [Least favorable]        ┊  Retrospective.          │
│                                           ┊  Not a projection.       │
│ Table 1. Most favorable realized luck.    ┊                          │
│ Qualified hitters, 2026 to date. n = 124. ┊                          │
│ ═════════════════════════════════════════ ┊  Most intervals cross    │
│ RK  HITTER            RUNS/100   TOT  BBE ┊  zero. One that does is  │
│ ───────────────────────────────────────── ┊  shown exactly as one    │
│  1  Pete Crow-Armstrong  +7.62   24.5 321 ┊  that does not.          │
│                          [+2.1,+13.1]     ┊                          │
│                          ▂▂▂▂▂▪▂▂▂▂       ┊                          │
│  2  ...                  +6.98   22.1 298 ┊                          │
│                                    margin ┘                          │
└──────────────────────────────────────────────────────────────────────┘
```

## 7. Leaderboard

**A properly typeset statistical table**, 32 px rows, top rule / header rule / bottom rule,
**no interior vertical rules, no zebra striping** — the conventions that make dense numeric
tables readable in print.

| Column | Treatment |
|---|---|
| **RK** | Tabular, amber while official |
| **HITTER** | 14 px/500; a superscript letter marks a hitter whose provisional share is high, keyed to a margin footnote |
| **RUNS / 100 [95% interval]** | The merged verdict cell: estimate at 16 px/600 with sign, the interval beneath in brackets one step down, and **beneath both a 3 px gauge rule** on the league-wide domain with a 5 px dot |
| **TOT** · **BBE** · **G** | Tabular, right-aligned |

- **The interval strip is the quietest of the three** and the typography is the loudest. The
  numbers carry the argument; the gauge rule confirms position at a glance. This is the exact
  inverse of Direction 2 and a real choice, not a shade of one.
- **Official vs. sorted:** amber leaves RK; **the caption changes** to
  "Table 1. Sorted view — not the official Contact Luck ranking." A caption that states what
  the table currently *is* is stronger than a badge floating beside it, and it is the same
  mechanism a published table would use. `aria-sort` and a polite live region as always.
- **Filters** sit above the caption with visible labels, styled as query terms, and the caption
  reports the resulting n — `n = 41 of 124` — so the table always declares its own population.
- **Density:** medium-low at 32 px rows, but the margin means no vertical space is spent on
  callouts; net above-fold content is comparable to Direction 1.

**Mobile (390):** the margin **collapses into the flow as numbered footnotes** rendered
directly beneath the element they annotate — the classic sidenote-to-footnote transformation,
not a stack. Rows become two-line ruled entries with the name at 16 px/600 owning a full line.
The caption stays, because at 390 px a caption stating "n = 124, qualified hitters" is doing
more work than it does at 1440.

## 8. Player page

An **entry in the annual**, and the most narratively legible of the three — with one honest
constraint: **the dashboard never computes, so there is no generated per-player prose.**
Storytelling here comes from sequence, captions, and apparatus, never from sentences a
template invented about a hitter.

1. **Who** — an entry head: an optional 48 px square portrait, the name in the display
   setting, and `batter_id` / data-through in the apparatus register beneath. The head is a
   fixed-height ruled band whose text layout is **identical with or without the portrait**.
2. **What** — the headline figure at 44 px with its sign.
3. **How extreme** — both official ranks as an answer line beneath, and in the margin the
   note that ranks come from the point estimate alone and that interval endpoints never
   reorder anyone.
4. **How uncertain** — the interval in brackets directly beneath the figure, sharing its
   left edge, with the gauge rule under it on the league domain.
5. **Why** — **Table 2. Component decomposition.** Contact, unexplained residual, defensive
   execution, advancement, each with its value and a **footnote marker keyed to a margin note
   carrying the status code** (`calibrated`, `provisional`, …). Every displayed component
   value carries its status, as `CONTEXT.md` requires — and here that requirement is met by
   the page's native structure rather than by extra chrome.
6. **Which plays** — a "Selected plays" section linking into Explore pre-filtered by URL.
7. **What next** — both rankings, methodology, plays.

**Trend** is **Figure 1. Contact Luck per 100, by snapshot date.** — captioned, with the
caption stating that gaps are real and not interpolated. **Supporting stats** are a ruled
stat line, not four tiles. **Provisional share** is a value in that line with a margin note
giving its meaning. **Unqualified hitters** get an entry head reading
`Reported, not ranked` where the rank line sits, in identical type, with a margin note stating
the threshold — the most dignified treatment of the three, because "reported, not ranked" is a
publishing status, not a warning.

## 9. Explore

A **catalogue**, not a form. The showcase leads as up to six captioned figures — each a real
play with a caption naming the hitter, the date, and the break — so the reader browses
examples rather than filling in a field. Beneath, the hitter combobox (full ARIA semantics,
Escape closes, **URL-addressable**) is framed as a catalogue lookup with the current selection
stated in a caption: `Plays for Pete Crow-Armstrong · 321 eligible batted balls`. Filters
restate the caption's n as they narrow. Results are a typeset table with the same verdict-cell
treatment as Table 1. "Show all 12" reveals a ruled list. Empty state hides table and count
together.

## 10. Individual play page

Set as a worked example — the form a statistical annual uses to show its method on one case.

- **CONTACT** — **Table 3.** EV, launch angle, batted-ball type, spray angle. The spray row
  carries a footnote marker; the margin note says the value is a directional approximation
  derived from Statcast visualization coordinates, never an exact measurement.
- **EXPECTATION** — **Table 4. Frozen model outcome probabilities.** Five rows with bars on a
  shared baseline; the recorded outcome's row is set in bold with a footnote marker.
- **ACTUAL RESULT** — the recorded result named in the entry head, distinct from the run-value
  accounting below it.
- **RUN VALUE GAP** — a **run-value accounting block** set like a small balance sheet:
  Expected RV, Observed RV, and the difference on a rule, in tabular figures. A balance is
  precisely what `observed − expected` is.
- **CONTACT LUCK CONTRIBUTION** — the signed result beneath the rule, at emphasis.
- **Field diagram** — **Figure 2**, small, captioned, `role="img"` with an `aria-label`, in
  the margin column where illustrative material belongs. This is the clearest demotion of the
  baseline's over-prominent 260 px unlabelled diagram.
- **Sensitivity** (showcase only) — **Figure 3. Sensitivity of expected run value to contact
  inputs**, whose caption updates with the slider so the reader always has the current
  counterfactual stated in words as well as marks.

## 11. Methodology

This direction's home ground, and therefore the place its main risk lives. Guard: the
methodology page is genuinely scholarly — numbered sections, a sticky table of contents,
sidenotes, linkable definitions — **and the rest of the site stays terse.** The annual's
apparatus is *thin* on the leaderboard (three margin notes) and *thick* here. The reading
budget on `/` is unchanged, and the lead links to `/demo/`, never here. Rigour is available
in one click and imposed on no one.

## 12. Status / provenance

An editorial note plus two captioned tables. The lead states freshness in a sentence.
**Table 5. Snapshot history.** replaces the broken admin grid: proper table typography, a
`min-width` contract, dates formatted at source (`Aug 15, 2026, 13:35 UTC`, ISO in
`<time datetime>`) — which alone removes the unbreakable 32-character token behind the
baseline's 31 px columns and 8,363 px page. Snapshot type and reason codes are footnote
markers into the margin. **Table 6. Component model status.** carries the calibration states.
Below 1024 both become definition lists per snapshot.

## 13. Demo

A **worked example**, which is what the demo already is — this direction just names it
properly. Two real batted balls as **Figure 4** and **Figure 5**, each captioned with what it
demonstrates, with the ball-flight animation running in the figure and margin commentary
tracking the stages. The counterfactual simulator is an admissible card (an instrument with a
defined input surface) captioned as **Figure 6**, with its caption restating the current
inputs. The demo's existing strength — that it teaches by showing two real plays — is
preserved exactly and given an apparatus.

## 14. Typography

The most typographically driven of the three, and the only one where the **serif does the
most work**.

- **Brand/display:** a genuine display setting for the masthead and entry heads — serif or
  condensed grotesk at large size with tight tracking, used only there.
- **Measured/numeric:** grotesk with tabular lining figures, confined almost entirely to
  tables and headline figures.
- **Prose:** serif at 68ch for leads, methodology, and all margin notes — the margin is serif
  at `--fs-200`, which is what makes it read as apparatus rather than as UI chrome.
- **Tables:** 14 px grotesk body, 11 px uppercase condensed heads, captions in **serif italic**
  at `--fs-200` above each table.
- **Metadata:** the apparatus register — `--ink-muted`, 12 px floor, footnote markers as
  superscript letters (never numbers, which would collide with the tabular figures).

Ratio: roughly 45% grotesk / 55% serif. The inversion in `docs/design/typography.md` —
grotesk for what is measured, serif for what is argued — is most visible here, because this
direction has the most argument on screen.

## 15. Color

The quietest colour and the loudest type. **Sign colours appear on the numeral and the 3 px
gauge dot only** — never on the interval brackets, never on a rule. **Amber** marks the
running-head rule, the official rank column, and the focus ring. The margin is set entirely in
`--ink-muted` at the 4.5:1 text grade, which is the direction's single most important contrast
requirement, since the apparatus is small type doing real work and must never fall to a mark
grade. Field green stays in field diagrams.

**Dark mode:** reads as a well-set dark reading surface — ground `#0d0c0a`, margin ink
`#898781` (4.62–5.44 in dark, passing), rules `#322f29`. The margin needs explicit attention
in dark mode because small muted serif is the first thing to fail; it is held to the text
grade on all three dark surfaces.

## 16. Geometry

- **Radius:** `0` on tables, rules, figures; `4px` on inputs/buttons; `999px` on status pills
  only.
- **Borders:** hairline. Tables use **only** the three horizontal rules of statistical table
  convention — no interior verticals, no cell borders, no outer box.
- **Grouping:** by the two-column grid and by captions. A caption above and a rule below is
  the grouping device; a border is not.
- **Page edges:** the data column and margin are contained to ~1040 px combined; the
  running-head rule spans the full measure. The margin gutter is the page's defining edge.
- **Full-width vs contained:** figures may span the data column; nothing spans the margin.
- **What does NOT get a card:** page sections, callouts, stat readouts, charts, tables, page
  headers, the player entry head, the decomposition, the trend, and — importantly — **the
  margin notes**, which are the greatest temptation in this direction. A margin note is set
  type in the gutter, never a tinted box. Cards remain: showcase play instances (≤6) and the
  demo simulator.

## 17. Player imagery

> **Superseded in part.** The owner asked for portraits in the leaderboard's Player
> column, which this direction recommended against. They shipped there — 34×38 px,
> `object-fit: contain`, never in the player entry head. The recommendation's *reasons*
> were kept: fixed reservation, no schema change, and a row that is complete without the
> image. See `information-architecture.md` § "Decision on record: leaderboard portraits".

**Recommendation: restrained headshots — 48 px square, in the player entry head only.** This
is the one direction where a portrait is native to the reference: a biographical entry in a
statistical annual carries a plate. Constraints: never on the leaderboard, never in a table
cell, never larger than 48 px, always muted to sit with the apparatus rather than compete with
the figure. **The entry head is a fixed-height ruled band whose type layout is byte-identical
with or without the image**, so the page is visually complete when imagery is missing —
satisfying the name-first rule in `docs/design/information-architecture.md`. No team or
position data is added and no schema changes; if `batter_id`-keyed headshots are never
adopted, this direction loses nothing.

## 18. Data visualization

Figures are numbered, captioned, and rationed — the annual's discipline is that a figure must
earn a number.

- **Shared-zero intervals:** a 3 px gauge rule with a 5 px dot on the league-wide domain,
  beneath the numerals. Restrained by design; the numbers lead. Near-zero values sit on the
  spine position with an explicit sign and no de-emphasis.
- **Trends:** Figure 1, captioned, zero reference line, ≥3 y labels, true dates, gaps never
  interpolated, re-authored on mobile so text stays ≥12 CSS px.
- **Decomposition:** primarily **Table 2** with values and status codes; a signed bar column
  inside the table on the same zero domain, not a separate chart.
- **Probabilities:** Table 4 with in-cell bars and the recorded outcome in bold.
- **Comparisons:** two captioned figures sharing an axis and an explicit caption stating what
  is being compared.

## 19. Desktop feel (1440)

A published page. A masthead with an edition line, a hairline, a contents row. Below, a
narrow serif paragraph on the left and, set apart across a gutter, a column of small serif
notes in muted ink — visibly secondary, visibly deliberate. Then an italic caption, a heavy
top rule, small uppercase column heads, a lighter rule, and rows of aligned figures with
bracketed intervals beneath them and a faint horizontal gauge rule under each pair, the dots
drifting left and right across the rows. No box anywhere; no fills; the only strong marks are
the table's three rules. It looks like a page from a reference volume that was designed this
year rather than a UI that adopted a serif.

## 20. Mobile feel (390)

The defining transformation: **the margin column does not stack — it dissolves into numbered
footnotes** placed immediately after the element each note annotates, in the same muted serif
at 12 px. Nothing is lost and nothing is deferred to the page bottom. Rows become two-line
ruled entries with the name owning a full 16 px/600 line. Captions survive and become more
valuable, since a caption declaring `n = 124, qualified hitters` replaces context the wide
layout carried spatially. The masthead compresses to wordmark + edition line + menu; the
contents line becomes a disclosure list with 44 px targets. `/status/` tables become
definition lists.

## 21. Signature design move

**The margin apparatus.** A persistent column in which every definition, status code, caveat,
threshold and provenance note is set at the exact vertical position of the claim it qualifies
— so "most intervals cross zero" sits beside the intervals, the qualification threshold sits
beside "Reported, not ranked", and each component's calibration status sits beside its value.
The product's central obligation, statistical honesty, becomes typographic infrastructure
instead of a sequence of disclaimer boxes the reader must get past. It is also the mechanism
that removes 786 px of baseline preamble without deleting a single word of it.

### Anti-generic test

| Generic failure | How this direction forecloses it |
|---|---|
| SaaS dashboard / KPI tiles | No tiles; supporting stats are a ruled line, and the margin absorbs what a dashboard would tile. |
| Endless rounded cards | Two card uses; margin notes are explicitly never boxed. |
| Generic hero | The lead is one sentence with its definitions beside it; Table 1's caption is ~9 rows down. |
| Excessive whitespace | The gutter is not whitespace — it is occupied by the apparatus. |
| Arbitrary pills | One pill use: qualification status. Footnote markers replace what would otherwise be badges. |
| Random gradients / glass / neon | None; the only marks are rules, type, and a 3 px gauge. |
| Arbitrary iconography | Zero icons; footnote letters carry what icons would. |
| Equal visual weight | Three explicit registers — figure, table, apparatus — each with its own size, face and ink. |
| Decorative charts | A figure must earn a number and a caption stating its question. |
| Sports clichés | No team colour, logo, or broadcast chrome; antiquarian ornament is banned too. |
| **The swap test** | The margin exists to carry *component calibration status, provisional share, qualification thresholds, and interval interpretation*. Strip those and the apparatus has nothing to hold — the layout would collapse to an ordinary article. |

## 22. Risks

- **Too editorial / too academic** — the largest risk of the three. Guard: the apparatus is
  **thin** on `/` (three notes) and thick only on `/methodology/`; the lead is one sentence;
  `/demo/` stays the conversion path. If the margin on the homepage exceeds three registered
  notes, the direction has drifted.
- **The margin becomes a dumping ground.** Guard: margin content is restricted to exactly
  three registered kinds — definition, status, provenance. Anything else belongs in the data
  column or on `/methodology/`.
- **Too sparse** — a two-column grid can leave the data column narrow. Guard: the data column
  holds the full 780 px table measure; the margin is 220 px and never wider.
- **Precious / antiquarian.** Guard: no drop caps, no small caps as body, no ornament, no
  faux-paper ground, modern faces only.
- **Too technical.** Guard: numbering is only ever on tables and figures, never on navigation
  or sections of the homepage.
- **Too generic** — a serif and a sidebar is a common blog layout. Guard: the margin must
  vertically register with its anchor (a note that merely floats near the top of a section is
  the failure mode), and the tables must use true statistical-table conventions rather than
  bordered cells.

## 23. Implementation complexity

**MEDIUM-HIGH.**

- The **margin/anchor registration** is the hard part: notes must align vertically with the
  element they annotate, survive re-flow, and collapse to inline footnotes below 1024. This is
  a well-understood CSS problem (grid with named rows, or absolutely positioned notes keyed to
  anchors) but it needs real care and a test.
- **Table 1a** ("Reported, not ranked", alphabetical) needs a new view-model in `content.py`
  listing sub-threshold hitters. The data already exists — global search already reaches all
  628 players — so this is projection, **not computation**, and the `dashboard/` isolation
  boundary is untouched.
- Table/figure numbering must be stable and build-time generated, not hand-maintained.
- Standard work shared with the other directions: real tab buttons, `<th aria-sort><button>`,
  labelled inputs, skip link, combobox semantics, `min-width` contracts, formatted timestamps.
- `visuals.py` changes are the **smallest** of the three — the gauge rule is a simplification
  of the existing interval strip.

---
---

# Comparative analysis

| | **1 · The Register** | **2 · The Zero Spine** | **3 · The Annual** |
|---|---|---|---|
| **Contact Luck specificity** | High — the fixed column grid encodes the metric's four tiers | **Highest** — meaningless without a signed metric on a shared domain | High — the apparatus exists to carry status codes and intervals |
| **Information density** | **Highest** (36 px rows, 6 columns) | Medium (44 px rows, plot per row) | Medium-low (32 px rows, but no callout overhead) |
| **Analytical credibility** | High — reads as a record | High — reads as a measurement | **Highest** — reads as a published, sourced result |
| **Visual distinctiveness** | Medium-high — austere and unusual, but ledgers exist elsewhere | **Highest** — no other analytics site is built on one axis | High — distinctive, though serif+margin is a known idiom |
| **Leaderboard strength** | **Strongest for scanning** — aligned decimals down 124 rows | **Strongest for comprehension** — the league as one shape | Strongest for *interpretation* — every row carries its terms |
| **Player-page storytelling** | Weakest — consistent but flat | **Strongest** — one figure answers questions 2–4 at once | Strong — sequence and apparatus, no generated prose |
| **Mobile potential** | Strong — two-line ruled row is a clean re-author | Strong but hardest — sticky axis is the enabling trick | **Strongest** — margin→footnote is a genuine transformation |
| **Reliance on imagery** | **None** (recommended: no headshots) | **None** on data surfaces | **Restrained** — optional 48 px plate, never structural |
| **Implementation complexity** | **MEDIUM** | **HIGH** | **MEDIUM-HIGH** |
| **Primary risk** | Reads as a spreadsheet | The spine becomes decorative where nothing is measured | The apparatus turns the product academic |

**Where the real decision lies.** Directions 1 and 2 disagree about the leaderboard's job.
The Register optimises for **scanning a ranked record** — decimal points align down 124 rows
and the strip confirms position. The Zero Spine optimises for **seeing a distribution** —
values float at their axis position, so you read the league's shape before any individual
number. That trade (column alignment vs. distributional legibility) is the sharpest
either/or in this set. Direction 3 declines that trade and instead optimises for **a reader
who needs the terms in view**, spending its budget on apparatus rather than on either
alignment or plot.

## Design recommendation

If asked, I would build **Direction 2 · The Zero Spine**, with one reservation.

It is the direction with the strongest claim under `PRODUCT.md` and `DESIGN.md`
simultaneously. `DESIGN.md` explicitly names the shared-domain interval column as "the
product's best idea… currently hidden by its styling," and identifies making it "the spine of
the whole system" as *the redesign's central move* — Direction 2 is that sentence executed
literally. It is also the direction that best serves the product's two hardest honesty
obligations: "uncertainty is presented, never used as a visual verdict" becomes structural
rather than enforced, and unqualified players are treated identically by construction rather
than by discipline. Its swap test is the strongest of the three: the composition cannot be
transplanted to a product without a fixed zero. And it turns `/status/` from an admin table
into a freshness timeline that makes real coverage gaps visible.

**The reservation:** it is HIGH complexity, and it trades away decimal-column scanning — a
real loss for the analyst audience in `PRODUCT.md`'s group 3, who scan numerically. If that
trade is unacceptable, **Direction 1 is the safer strong answer**: it keeps every scanning
affordance, costs meaningfully less to build, and its persistent verdict column is a genuine
idea rather than a fallback. And if the site's biggest problem is judged to be that
first-time readers do not understand the caveats — a defensible reading of a metric whose
core message is "most of this is not distinguishable from zero" — **Direction 3 is the
correct answer and the other two are not close**, because it is the only one that puts the
caveat beside the claim.

**This is your decision.** I am not choosing, not merging, and not implementing.
