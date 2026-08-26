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

## Mobile leaderboard

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
