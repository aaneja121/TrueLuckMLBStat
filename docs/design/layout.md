# Layout and geometry

Measures, spacing, grid, radii, borders, and the card rule. Entry point: `DESIGN.md`.
Read with `docs/design/responsive.md` (how these hold across breakpoints).

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
  The **pitcher card** follows the same rule as of 2026-09-09: ≥ 1024 the hero is a grid —
  numeral left, the sentence that gives it its sign beside it, the figure across the full
  measure. It was a stack of narrow bands (a 720 px figure in a 1152 px column), which is
  the wasted-space shape named above, not a measure problem: `--measure-data` already gives
  every route 1152 px of usable content at ≥ 1248. **The data measure is not per-route** —
  `.page-shell` is worn by the header, main and footer alike, so widening it for one
  surface would move the wordmark between routes.

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

---

## When should something be a card?

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
