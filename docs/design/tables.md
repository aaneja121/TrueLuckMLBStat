# Leaderboards and tables

Column tiers, the verdict cell, shedding order, and the mobile row. Entry point:
`DESIGN.md`. Read with `docs/design/responsive.md`, `docs/design/typography.md`
(tabular figures) and `docs/design/accessibility.md` (tabs, sortable headers).

Leaderboards and tables are first-class product surfaces. Columns are not eight equals;
they are four tiers.

| Tier | Columns | Behaviour |
|---|---|---|
| **Identity** | Rank, Player | Never sheds, never truncates mid-word |
| **Verdict** | Runs / 100 **+ its interval, as one cell** | Never sheds; the interval is never behind a toggle |
| **Evidence** | Total runs, Eligible BBE, Scored Games | Compresses before it sheds |
| **Derived** | Interval width | Sheds first — it is recoverable from the interval already shown |

---

## Rules

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

---

## Value placement — settled at V1 (gate G3)

**Decision: V1.** One fixed-width numeral box after the plot field, right-aligned at a
constant x for all 124 rows. Sign is carried three ways — the explicit `+` / `−` (U+2212)
glyph, the mark's colour, and the mark's side of the spine.

**V2 was built, rendered, and rejected.** V2 aligned unfavorable numerals at a constant x on
the field's left margin and favorable numerals at a constant x on its right, making the
alignment itself a sign cue. It is genuinely more elegant *under the default sort*, where the
rows within a ranking are almost all one sign. It fails under any mixed-sign sort: re-sorting
by BBE interleaves the signs, the numerals alternate between two columns down the page, and
exact-value scanning — the thing the numeral exists for — degrades exactly when it matters
most. Value position must not depend on sort order (§ 1.6 of
`docs/design/zero-spine-implementation-plan.md`), and V2 makes value *column* depend on it.

The division of labour is the reason V1 wins rather than a taste call:

| Read | Carried by |
|---|---|
| Where this hitter sits in the league, and how wide the uncertainty is | The Zero Spine field — position against a shared scale |
| What the number actually is, comparable down the column | The fixed numeral column — one x, tabular figures, decimals aligned |

V2 asks the spine's coordinate space to do the numeral's job too. V1 keeps the two reads on
two devices.

**Consequences, now enforced by `tests/test_dashboard_leaderboard.py`:**

- The plot field is the **first track** of the verdict cell at every width. Nothing is
  reserved to its left, so no row's field can start further right than another's — which is
  the failure mode that would silently destroy the shared percentage basis the spine
  registers against.
- No development-only variant switch ships. The `?value-placement=` query parameter, its
  `localStorage` key, the `--lb-lead` reserved track and the `data-sign` row attribute all
  existed only to serve V2 and are gone.
- Neither variant ever placed a numeral at an interval endpoint, and that stays banned: the
  numeral's x is a constant, never a function of `--cl-lo` / `--cl-hi` / `--cl-pt`.

---

## Mobile leaderboard

Not "fewer columns" — a different row structure. As shipped, a **three-line** ruled row, one
job per line:

```
 1   Pete Crow-Armstrong
     |————●————|                              +7.62
     321 BBE · 120 G
```

This is a declared departure from the two-line sketch this file originally carried, which put
the numeral on the name's line and the field on the evidence line. That sketch splits the
verdict across two lines and contradicts rule 1 above — the point estimate and its interval
are one statement in one cell. The stronger rule wins: value and field stay adjacent on one
line. Every property the two-line row was buying survives, because the name still owns a
full-width line, so mid-word breaks remain structurally impossible.

### Zero registration is segmented on mobile — intentionally

**The invariant is: every verdict field registers zero at the same x.** It holds at 390: the
row grid gives every field the same content column, so the spine segments all land on one x
and read as a single rule down the page.

**The invariant is not: the amber rule must be physically unbroken from the first row to the
last.** At mobile the spine is drawn per field and bleeds over the row's vertical padding, so
consecutive verdict lines abut — but the name line and the evidence line sit between them and
carry no spine. The rule is therefore *segmented*: continuous through the data, interrupted by
identity and apparatus.

That is the correct behaviour, not a defect to fix. Drawing the spine through the player-name
line to buy literal continuity would put a measurement rule through a text field that has no
position on the scale — a mark that means nothing where it is drawn, which design principle 5
prohibits. **Do not restore literal continuity at mobile.** Above 768 the question does not
arise: the axis rail and the distribution strip are present, the rows are one line each, and
the spine runs unbroken from the tick labels to the last hitter.

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

## The component decomposition (player page, Phase 4)

A table, not a card set: it carries four facts per row — label, bar, value, model status —
and a table is the structure that ties them together for assistive technology without any
extra wiring. `<th scope="row">` on the component name does the association.

The baseline shipped these as **two disconnected tables**, one listing values and one
listing statuses, with nothing tying a row in either to a row in the other. Phase 4 merged
them, which required joining the snapshot's per-MODEL status (`contact`,
`outfield_defense`, `infield_defense`, `advancement`) to the per-VALUE decomposition a
reader sees: defensive execution is one value produced by two models, and appears in the
technical disclosure as two entries.

Three rules the decomposition adds to the ones above:

1. **The plot cell carries no padding of its own.** It *is* the coordinate space; the
   `padding-right` every other cell uses would move zero off the spine. The three text
   columns are fixed (`--pl-component-*`) precisely so the plot column is stable across
   rows.
2. **The axis rail sits between the column heads and the first bar**, not above the heads.
   An axis separated from its marks by a row of text stops reading as their axis.
3. **A bar is not an interval.** It runs from zero to the value and carries no point
   marker, because a component value has no interval in the snapshot — drawing it with the
   interval's vocabulary would claim one.

The **Total Contact Luck** row is drawn on the same component scale as the parts, which is
what makes "the components sum to the headline" visible rather than asserted.
