# DESIGN.md — Contact Luck visual and product design

Canonical source of truth for how Contact Luck looks and behaves. Product intent lives in
`PRODUCT.md`; terminology in `CONTEXT.md`; code layout in `ARCHITECTURE.md`. This file
governs the dashboard only — it never overrides a product principle or a research rule.

Baseline evidence for every claim here: `outputs/figures/ui_baseline_2026-08-26/`
(gitignored; see its `README.md`).

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
   becomes a bordered rounded rectangle. The three admissible cards are defined below.
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

## Information hierarchy

### Home / leaderboard

The reading budget:

| Time | The reader must have | Delivered by |
|---|---|---|
| **2 s** | "This ranks hitters by how much better or worse their batted-ball results were than expected, in runs." | The page title, one lead sentence, and the top rows of the ranked table — all visible at once |
| **10 s** | Who sits at the extremes, by how much, and how uncertain each one is | The shared-zero interval strip read down the column as a distribution |
| **30 s** | There are two rankings; the unit is per 100 eligible batted balls; most intervals cross zero; rows are clickable, sortable and filterable | The tab pair, the column head, the strip, the row affordance |

**Hard budget (the rule with teeth):** the **first data row must be visible at
1024 × 768**. Measured baseline: the first row currently sits at y = 900 with the table
header at y = 853, so **exactly one row is above the fold at 1440 × 900 and zero at
1180 × 820 and 390 × 844**. 786 px of preamble precedes the table at every desktop width.
This budget forces the explainer to a single lead sentence with the rest deferred, and
moves the example-play "proof strip" below the table or onto `/demo/`.

### Player page

The page answers seven questions in this order, and the layout must make the order legible
without headings for the first four:

1. **Who is this?** — name, and the identity context the data supports (see *Player
   identity*).
2. **What is their Contact Luck?** — the signed per-100 figure, the largest number on the
   page.
3. **How extreme is it?** — position on the league-wide zero axis, plus both official
   ranks. Rank is currently a small pill; it is an answer, not a tag.
4. **How uncertain is it?** — the interval, rendered *attached to* the number it qualifies.
   Baseline defect: the interval strip is orphaned in a full-width band below the header
   card, disconnected from the score.
5. **Why did they get that score?** — component decomposition with status labels.
6. **Which plays contributed?** — **currently missing.** The player page's only outbound
   links are the leaderboard, methodology and demo. See *Navigation & search*.
7. **What next?** — that player's plays, the two rankings, the methodology.

### Explore

Discovery, not a search form. Order: the editorial showcase (what a big break looks like) →
a named hitter's plays. The player selector is a **combobox that changes the page's
address**, so a hitter's plays are linkable from their player page and from a shared URL.
The current selection is client-only and unaddressable, which is why question 6 above has
no answer.

### Play page

The sequence must make expected-vs-actual immediately understandable:
**what was hit → what the model expected → what happened → the gap.** The gap is the
answer and belongs above the fold. Baseline defect: a 260 px unlabelled field diagram
occupies the position of greatest prominence while the run-value accounting sits last.

### Methodology

Rigorous without dominating: prose measure, a section index, and definitions that a reader
can link to. It is the destination for depth, never a prerequisite for using the product.

### Status

Provenance and freshness, not an admin console. The lead answers "how current is this and
is it trustworthy" in one line; the snapshot history is the archive beneath it.

### Demo

The conversion path for a confused visitor: two real batted balls explained visually, then
a counterfactual instrument. Structurally the strongest page in the product today —
protect it.

---

## Typography

### Roles

Two families, split along the product's own fault line: **grotesk for everything measured,
serif for everything argued.** The serif is not a display face — it carries the careful,
hedged prose the product cannot afford to have skimmed ("retrospective, not a projection").
This inverts the usual serif-display/sans-body pairing on purpose.

| Role | Family | Notes |
|---|---|---|
| Brand / page title | Grotesk, condensed cut, 600 | Sparing: wordmark and `h1` only |
| Column heads, eyebrows, status codes | Grotesk, condensed, 600, uppercase, +0.06em | Labels a measured quantity |
| UI / navigation / controls | Grotesk, 400–500 | |
| **Numbers and tables** | Grotesk, tabular lining figures | The product's primary typographic surface |
| Body / explanation / methodology | **Serif**, 400 | Prose measure, comfortable at length |
| Metadata, captions, footnotes | Grotesk, 400, `--ink-muted` | Never below 12 px |
| Identifiers (`play_id`, reason codes, hashes) | Mono | Identifiers only — **never statistics** |

Recommended: **IBM Plex Sans / Plex Sans Condensed / Plex Mono** (true tabular figures,
instrument register, open licence, not Inter) with **Source Serif 4** for prose. Self-hosted
`woff2` subsets. The site currently loads **zero** web fonts and makes zero external
requests; budget two families at ~60–80 KB total, and ship a system fallback stack that
preserves the grotesk/serif split (`ui-sans-serif` / `ui-serif`). If the font budget is
rejected, the split still holds using system faces.

### Scale

Replaces ~26 ad-hoc sizes, of which five sat inside a 0.06 rem band (0.82/0.84/0.85/0.86/
0.88 rem) — differences no reader perceives and every alignment breaks on.

| Token | px | Use |
|---|---|---|
| `--fs-100` | 11 | Column heads, status codes (uppercase only) |
| `--fs-200` | 12 | Metadata, captions, footnotes — **floor for any text** |
| `--fs-300` | 14 | Table body, UI labels |
| `--fs-400` | 16 | Body prose |
| `--fs-500` | 18 | Lead prose, emphasised table figures |
| `--fs-600` | 22 | Section heads |
| `--fs-700` | 30 | Page title |
| `--fs-800` | 44 | Player hero score |
| `--fs-900` | 64 | Reserved: the single largest number on a page |

No step may be added between two existing steps. Mobile: `--fs-700`/`800`/`900` step down
one; `--fs-400` never drops below 16 px; `--fs-100` never drops below 11 px.

Mobile `h1` is capped at **24 px**, not 26. Measured: "Christian Encarnacion-Strand" is
404 px at 30 px/600 and 350 px at 26 px/600 against 350 px of available width at 390 px —
26 px lands exactly on the limit with no margin for a wider face or a longer future name.

Line height: 1.15 display, 1.3 tables, 1.6 prose. Letter-spacing: `-0.01em` at `--fs-700`
and above, `+0.06em` on uppercase micro-labels, `0` everywhere else.

### Numeric typography

- `font-variant-numeric: tabular-nums lining-nums` is a **primitive** (`.num`), applied
  once, not re-declared per component (currently declared ad hoc in 10 places).
- Contact Luck values always carry an explicit sign: `+7.62`, `−6.41`, `±0.00`. The sign is
  the **non-colour channel** for the semantic pair.
- Use the true minus **U+2212**, never a hyphen, so `−6.41` and `+7.62` align in a column.
- Precision is fixed per quantity, never per context: per-100 → 2 dp; total runs → 1 dp;
  counts → integer; probabilities → 1 dp percent.
- Intervals render one scale step below their point estimate, in brackets, on the same unit.

---

## Color

### The finding this system exists to fix

The site adopted the `dataviz` skill's **chart palette as its UI palette**. Those hexes are
specified as *mark* and *axis* colours (3:1 floor); the site uses them as *text* at 12–17 px
(4.5:1 required). Measured against all three light surfaces (`--page #f8f6f0`,
`--surface #fefdfa`, `--surface-recessed #f1ede1`):

| Token | Light contrast (page / surface / recessed) | Dark contrast |
|---|---|---|
| muted `#898781` | 3.32 / 3.53 / **3.07** | 5.44 / 4.62 / 5.11 ✔ |
| favorable `#2a78d6` | 4.09 / 4.34 / **3.77** | 5.37 / 4.56 / 5.05 ✔ |
| unfavorable `#e34948` | 3.66 / 3.89 / **3.38** | 6.05 / 5.14 / 5.69 ✔ |
| accent `#b06f16` | 3.79 / 4.02 / **3.50** | 9.36 / 7.95 / 8.80 ✔ |

**Dark mode already passes AA on every token; only the light theme fails.** That is the
signature of a dark theme that was designed and a light theme that inherited chart hexes
untouched.

### The rule

**Every semantic colour has two grades: a text grade (≥ 4.5:1 on all three surfaces of its
theme) and a mark grade (≥ 3:1).** Text never uses a mark grade.

### Tokens

| Role | Light — text | Light — mark | Dark — text | Dark — mark |
|---|---|---|---|---|
| Favorable (positive sign) | `#256bbf` (4.56–5.25) | `#2a78d6` | `#3987e5` (4.56–5.37) | `#3987e5` |
| Unfavorable (negative sign) | `#d22120` (4.51–5.19) | `#e34948` | `#e66767` (5.14–6.05) | `#e66767` |
| Official / accent | `#975f13` (4.53–5.21) | — | `#e0aa57` (7.95–9.36) | — |
| Ink primary | `#0b0b0b` | — | `#ffffff` | — |
| Ink secondary | `#52514e` | — | `#c3c2b7` | — |
| Ink muted (incl. **chart axis labels**) | `#6d6b66` (4.55–5.23) | — | `#898781` (4.62–5.44) | — |
| Page / surface / recessed | `#f8f6f0` / `#fefdfa` / `#f1ede1` | | `#0d0c0a` / `#211e19` / `#17140f` | |
| Gridline / baseline / hairline border | `#e3dfd3` / `#c3c2b7` / `rgba(11,11,11,.12)` | | `#322f29` / `#3a3834` / `rgba(255,255,255,.12)` | |

Neutral / uncertainty: the interval line takes the **same** semantic colour as its point
estimate (never a separate "uncertainty grey," which would read as a hedge). Uncertainty is
carried by *length on the shared axis*, not by hue.

### Rules

- **The diverging pair is reserved for the sign of a point estimate.** No decorative
  element, border, illustration, link, or state may use blue or red. (`PRODUCT.md` §3.)
- **One non-semantic accent: amber, meaning "the frozen official record."** It marks the
  active nav item, the "Data through" badge, the official-rank column, the zero spine and
  the focus ring. When a table is user-sorted, the accent is **removed** from the rank
  column — the departure from official order is signalled by absence plus the explicit
  "Sorted view" label, never by a second colour.
- **`--accent-field` (green) is demoted to an illustration-only token.** It may appear
  inside field diagrams and nowhere else. It currently colours card top-borders on the home
  and demo pages, competing with the semantic pair on the same visual device.
- **Colour is never the only signal.** Every coloured value ships with an explicit sign
  glyph and, where the value is a headline, the word Favorable / Unfavorable.
- Links: ink primary with a 1 px underline at 0.12em offset; the accent is not a link
  colour. This removes the last non-semantic use of blue.

---

## Layout

- **Two measures on the same page.** Data measure `1200px` max; **prose measure `68ch`**.
  Explanatory prose never runs the full data width (the leaderboard explainer currently
  runs 1140 px and stands 227 px tall).
- **Spacing scale, 8 px base:** 4 / 8 / 12 / 16 / 24 / 32 / 48 / 64. Nothing off-scale.
- **Section rhythm:** 48 px desktop / 32 px mobile between sections, delimited by a hairline
  rule. Headings sit tight to their content (12 px), loose from what precedes (48 px).
- **Whitespace philosophy:** spend it between tiers, not inside boxes. A 260 px diagram
  centred in a 1140 px column with empty gutters (play page) is wasted space, not breathing
  room; a 12 px gap between a number and the interval that qualifies it is a missing
  relationship, not compression.
- **Grid:** 12 columns ≥ 1024. The leaderboard takes the full data measure. The player page
  is an 8/4 split (score + evidence / context and trend), not a stack of full-width bands.

---

## Geometry

- **Radius:** exactly three values. `0` (tables, rules, sections, charts, figures), `4px`
  (inputs, buttons, cards), `999px` (**status pills only** — qualification status, and the
  Favorable/Unfavorable badge). Baseline has **eleven** distinct radii.
- **Borders:** hairline `1px` only. The two exceptions are the 2 px focus ring and the 2 px
  zero spine. No 2 px decorative borders, no coloured top-borders on cards.
- **No shadows on data.** No drop shadow, glow, or gradient on any mark, card, or surface.
  Depth comes from the three-plane tonal system (page < recessed control < surface), which
  already exists and works.

### When should something be a card?

A card is admissible in exactly three cases:

1. **A repeated instance of one object type in a set the reader scans across** — showcase
   plays, search results, the two compared plays on `/demo/`. The boundary tells you where
   one instance ends and the next begins.
2. **An interactive instrument with a defined input surface** — the counterfactual
   simulator. The boundary marks "everything inside here responds to you."
3. Nothing else.

**It is not a card when** it is a section of a page, a callout or aside, a stat readout, a
chart or figure, a table, or a page header. Those are grouped by a rule and space. Baseline:
18 separate rules hand-paint the surface+border+radius card, and 31 `stat-*` / 13 `card` /
11 `callout` class uses produce the same box on every route.

**Stat tiles become a stat line.** The player page's four bordered tiles (Total Contact Luck
Runs, Eligible BBE, Scored Games, Provisional share) are one stat line, not four objects.
Render them as a ruled box-score line: four borders removed, scannability gained, vertical
space recovered for content that is currently below the fold.

**A card set is capped at six.** Beyond six instances the set becomes a compact ruled list.
Consequence: `/explore/`'s "Show all 12" reveals a **list**, not six more cards — the
one-way reveal behaviour is preserved, the box proliferation is not.

---

## Leaderboards and tables

First-class product surfaces. Columns are not eight equals; they are four tiers.

| Tier | Columns | Behaviour |
|---|---|---|
| **Identity** | Rank, Player | Never sheds, never truncates mid-word |
| **Verdict** | Runs / 100 **+ its interval, as one cell** | Never sheds; the interval is never behind a toggle |
| **Evidence** | Total runs, Eligible BBE, Scored Games | Compresses before it sheds |
| **Derived** | Interval width | Sheds first — it is recoverable from the interval already shown |

### Rules

1. **The verdict cell is one cell.** The point estimate and its interval strip share a cell
   on the shared zero axis. They are one statement. This removes the widest column pair and
   turns the column into a readable distribution.
2. **The zero spine is a full-height rule**, continuous down the table body — not a stray
   per-row tick, which is how it currently reads.
3. **Every scrollable table declares a `min-width` contract.** `.overflow-x` never ships
   without one, and a scrollable table always shows a **visible affordance** (edge fade +
   persistent scrollbar). Baseline: the same inert `.overflow-x { overflow-x: auto }` with no
   min-width produces opposite failures in two tables — the leaderboard silently clips
   "Interval width" at *every* desktop width (table 1158 px inside a 1100 px box), and the
   status history silently crushes two columns to 31 px with one character per line
   (682 px row height, 8363 px page).
4. **Shedding is ranked and declared, never ad-hoc.** Derived sheds; then Evidence
   compresses to one cell (`321 BBE · 120 G`); Identity and Verdict never shed.
5. **Sorted state has three simultaneous expressions:** `aria-sort` on the header, the
   visible "Sorted view — not the official Contact Luck ranking" label, and removal of the
   official accent from the rank column. The Rank column always restores the frozen order.
6. **The two rankings are a tab pair of real buttons** (`role="tablist"`, `aria-selected`),
   matching the pattern `/demo/` already implements correctly. Each ranking begins at #1 and
   is labelled with its frozen `public_labels` string; neither is "worst."
7. **Comparison across the two rankings** is supported on the player page, where both
   official ranks are shown as answers ("#1 favorable · #124 unfavorable"), not as tags.

### Mobile leaderboard

Not "fewer columns" — a different row structure. A two-line ruled row:

```
 1   Pete Crow-Armstrong                        +7.62
     321 BBE · 120 G          |———●————|   (shared zero axis)
```

The name owns a full-width line, so mid-word breaks become structurally impossible.

Measured against the **full 628-player set**, not just the 124 qualified: the longest name
is "Christian Encarnacion-Strand" (28 characters). At 390 px the name line has **310 px**
available (390 − 40 page padding − 28 rank − 12 gap). That name measures 195 px at
14 px/500 and **224 px at 16 px/600** — so the row can afford `--fs-400` at 600 weight for
the name with 86 px of headroom, and every name in the set fits on one line. Set the name
at 16 px/600.

Baseline defect being replaced: a 79 px Player column plus a site-wide
`overflow-wrap: anywhere` on `body` yields "Pete Crow-Armstro / ng" and
"Kyle Schwarb / er".

---

## Data visualization

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

### Rules

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

---

## Player identity

Player identity supports scanning and storytelling; it is not decoration.

- **The name is the identity primitive** and gets the strongest treatment available in its
  context. It never breaks mid-word and never truncates unless the container genuinely
  cannot hold it, in which case it ellipsises at a word boundary with a `title`.
- **Consistent across surfaces:** leaderboard row, player page hero, Explore selection, play
  page header, and search result all present the name identically (same face, same casing,
  same weight relationship to its context).
- **`batter_id` is shown as apparatus, not identity** — footnote register, `--ink-muted`,
  alongside the data-through date.
- **Team, position and headshots do not exist in the data model today.** Verified: the
  snapshot content model and the Explore shards carry only `batter_id` and `batter_name`.
  `CONTEXT.md` notes `batter_id` is the key for "headshot lookups," but no headshot code,
  asset, or request exists, and the site currently makes **zero** external requests. Adding
  either is a product and privacy decision with a real cost — **see the open question at
  the end of this file.** Until it is decided, identity is name-first and the design must
  not depend on imagery.

---

## Navigation and search

- **Global header:** wordmark, five routes, global player search, "Data through" badge. The
  header is `position: static` today and should stay that way on long pages only if a
  back-to-top affordance exists — the status page is 8363 px tall with no route back to nav.
- **Skip link is the first focusable element on every page.** None exists today; the
  leaderboard has 135 focusables before the footer.
- **Search is a real combobox** on every surface: `role="combobox"`, `aria-expanded`,
  `aria-controls`, `aria-activedescendant`; results are a `listbox` of `option`s; Up/Down
  navigate; Enter selects; **Escape closes** (it currently does not); outside click closes
  (it currently does). One implementation, used by both the global search and Explore's
  hitter picker — they differ today for no reason.
- **The play-level path must be navigable in both directions.** A player page links to that
  player's plays; Explore's selection is reflected in the URL so that link can exist and be
  shared. This is the product's stated value proposition ("it goes all the way down to the
  play") and it currently dead-ends: the player page's only outbound links are the
  leaderboard, methodology, and demo.

---

## Interaction

| State | Treatment |
|---|---|
| **Hover** | Row: recessed surface tint. Link: underline thickens. **No lift, no shadow, no scale.** |
| **Focus** | 2 px accent ring, 2 px offset, plus a contrasting outer hairline so it reads on any surface. Never removed, never replaced by a background change alone. |
| **Selected** | Tab: accent underline + `aria-selected`. Sorted column: direction glyph + `aria-sort`. Both carry a text or shape channel, never colour alone. |
| **Sorted** | The three simultaneous expressions in *Leaderboards and tables* §5, plus a polite live region announcing the new order. |
| **Loading** | A labelled placeholder in the destination's own shape. No spinners on a static site; no skeleton shimmer. |
| **Empty** | States what matched nothing and what to change. Preserve "No plays match these filters." — it hides the table *and* the result count, which is correct. |
| **Error** | Says what happened and offers the way out. Preserve the "Play not found" state and its link back to Explore. |
| **Disclosure** | `<details>`/`<summary>` for technical reason codes. One-way reveals ("Show all 12") hide their trigger after use — preserve. |

---

## Responsive behavior

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
preferable to engineering a layout that survives it.

---

## Accessibility

Hard requirements, satisfied by the primitives rather than by a later pass.

1. **Contrast.** Text ≥ 4.5:1 against **its own** surface — page, card, *and* recessed
   control surfaces are all checked, because the recessed plane is the worst case. Marks
   ≥ 3:1. Two grades per semantic colour (see *Color*). SVG axis labels are text and are
   held to the text grade.
2. **Non-colour semantics.** Every Contact Luck value carries an explicit sign glyph; every
   headline value also carries the word Favorable / Unfavorable.
3. **Tabs are buttons.** `role="tablist"` / `role="tab"` / `aria-selected`, keyboard
   operable. The `display:none` radio + `<label>` pattern is banned — it removes the
   leaderboard's ranking switch from both the accessibility tree and the tab order.
4. **Sortable headers are buttons.** `<th aria-sort><button>`; sort changes announced via a
   polite live region.
5. **Every input has a programmatic label.** `<label>` or `aria-label`. A placeholder is
   never a label. Two identical unlabelled filter inputs currently ship on the leaderboard.
6. **Combobox semantics** as specified in *Navigation and search*.
7. **Skip link** first in the tab order on every page.
8. **Focus** always visible, per *Interaction*.
9. **Touch targets ≥ 44 × 44** for anything tapped. Nav links are 90 × 25 px today.
10. **Meaningful SVG is labelled;** decorative SVG is `aria-hidden`.
11. **One `<h1>` per page**, no skipped heading levels, landmarks on every route.
12. `prefers-reduced-motion: reduce` disables all non-essential motion.

Preserve what already passes: `lang="en"`, no duplicate ids, no `<img>` (so no alt debt),
correct landmarks, and a consistent visible focus ring.

---

## Motion

Motion is allowed only where the movement *is* the information.

- **Permitted:** the `/demo/` ball-flight animation (it teaches the metric); row re-ordering
  on sort (position change is the payload); the one-way reveal of a hidden set.
- **Prohibited:** entrance/scroll-reveal animation, hover lift or scale on cards, parallax,
  skeleton shimmer, ambient background motion, and number count-ups.
- Durations 120–200 ms, ease-out. Everything above respects `prefers-reduced-motion`.

---

## Anti-patterns

Explicitly prohibited. Each is drawn from the measured baseline.

1. **A decorative accent applied uniformly to every `<h2>`.** `h2::before` currently gives
   every section the same amber tick regardless of importance — decoration masquerading as
   structure.
2. **Boxing a page section** in a bordered rounded rectangle so it "looks like a section."
3. **Nudging a font size** by 0.02 rem instead of using a scale step. No new sizes.
4. **Adding a border-radius value.** Three exist; that is the set.
5. **`overflow-wrap: anywhere` on `body`** or on any ancestor of prose or player names.
   Scope it to the elements that actually hold unbreakable identifiers (`code`, reason
   codes, snapshot labels).
6. **`.overflow-x` without a `min-width` contract and a visible scroll affordance.**
7. **`display: none` on a control that must remain operable.**
8. **A placeholder used as a label.**
9. **Reusing blue or red for anything that is not the sign of a point estimate.**
10. **A third or fourth non-semantic hue.** One accent; the field green is illustration-only.
11. **SVG text below 12 CSS px** after viewBox scaling.
12. **Count-up / odometer animation on a statistic.**
13. **Gradients, glows, or drop shadows on data marks or surfaces.**
14. **A stat rendered as a bordered tile** when it belongs to a stat line.
15. **A hero section whose only job is to restate the page title** before the actual data.
16. **Emoji as UI iconography.**
17. **Any visual treatment that makes an interval crossing zero look weaker** — fading,
    muting, dashing, italics, a "not significant" marker. This is a product invariant, not a
    style preference.
18. **Ranking or leaderboarding an individual component**, or any phrasing on the
    `BANNED_PHRASES` list.

---

## Preserved product invariants

The redesign must not change any of these. They are behaviour, not styling.

- Two independent official rankings, both starting at #1, both using their frozen
  `public_labels` strings. Never "worst players."
- Official rank semantics: competition ranking on `contact_luck_runs_per_100` alone;
  interval endpoints never reorder anyone; the Rank column always restores the frozen order.
- "Sorted view — not the official Contact Luck ranking" appears on any user re-sort and
  never on the default view.
- Sign colour follows the **point estimate's sign only**, identically whether or not the
  interval crosses zero.
- Unqualified players keep a page, a score, and an interval, receive no rank, and are never
  visually de-emphasised. Their status chip is neutral, not a warning.
- Intervals ship next to every point estimate, never behind a toggle.
- **The league-wide shared interval domain**, identical on the leaderboard and every player
  page (zero at 0.463/0.464 of width), so scores stay comparable across routes.
- Explore's empty state hides the table and the result count together.
- The "Play not found" state and its route back to Explore.
- Retrospective / not-predictive framing in the hero, per-page callouts, and footer.
- Dark mode.
- The one-way "Show all 12" reveal (its trigger hides after use).
- Every existing useful interaction — sort, filter, search, tab, disclosure, simulator —
  unless a deliberately designed superior replacement is specified here.
- The dashboard displays; it never computes.

---

## Open question for the product owner

**Player imagery and team context.** Team, position, and headshots are absent from the data
model. Adding MLB's static headshot CDN would introduce the site's first external request
and its first third-party dependency; adding team/position requires a new field in the
snapshot content model. Both would materially strengthen scanning and storytelling on the
leaderboard and player pages. Until this is decided, the system is name-first and must not
depend on imagery.
