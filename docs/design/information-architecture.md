# Information architecture

What each page must deliver, in what order, plus player identity and the navigation/search
model. Entry point: `DESIGN.md`. Read with `PRODUCT.md` (intent and copy) and
`docs/design/layout.md`.

---

## Home / leaderboard

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

---

## Player page

The page answers seven questions in this order, and the layout must make the order legible
without headings for the first four:

1. **Who is this?** — name, and the identity context the data supports (see *Player
   identity* below).
2. **What is their Contact Luck?** — the signed per-100 figure, the largest number on the
   page.
3. **How extreme is it?** — position on the league-wide zero axis, plus both official
   ranks. Rank is currently a small pill; it is an answer, not a tag.
4. **How uncertain is it?** — the interval, rendered *attached to* the number it qualifies.
   Baseline defect: the interval strip is orphaned in a full-width band below the header
   card, disconnected from the score.
5. **Why did they get that score?** — component decomposition with status labels.
6. **Which plays contributed?** — a link to that hitter's plays.
7. **What next?** — that player's plays, the two rankings, the methodology.

**Built in Phase 4.** The hero is one figure carrying all of questions 2–4 at once: the
number printed *at* its own mark, this hitter's interval and dot at double weight, the
whole qualified league beneath it in gridline ink, and the shared axis under both — one
amber rule descending all three, because they are three siblings in one containing block
and therefore share one percentage basis. Measured: the hero and the leaderboard row for
the same hitter carry byte-identical `--cl-lo` / `--cl-pt` / `--cl-hi`.

The guard against the hero becoming a KPI panel is that there is **no box, no border and
no fill** anywhere in it. Rank is written out as an answer rather than set in a pill, and
sits at `--fs-500` against the score's `--fs-800`, so it never overpowers the estimate it
describes. The four sample figures are a ruled stat line, not four tiles.

**The reason a hitter holds no official rank is specific to their status.** The baseline
told every unqualified hitter they had "not yet reached the minimum eligible-batted-ball
threshold", which is false for three of the four statuses — a `provisionally_qualified`
hitter can hold *more* eligible batted balls than the #1 ranked one, and is unranked
because of where the value came from, not how much of it there is.

---

## Explore

Discovery, not a search form. Order: the editorial showcase (what a big break looks like) →
a named hitter's plays. The player selector is a **combobox that changes the page's
address**, so a hitter's plays are linkable from their player page and from a shared URL.
Phase 4 added the inbound half — Explore reads `?batter=<id>` and preselects — but the
selection is still not written back to the URL, so a hitter chosen *inside* Explore is
still unaddressable. That half is Phase 5's.

## Play page

The sequence must make expected-vs-actual immediately understandable:
**what was hit → what the model expected → what happened → the gap.** The gap is the
answer and belongs above the fold. Baseline defect: a 260 px unlabelled field diagram
occupies the position of greatest prominence while the run-value accounting sits last.

## Methodology

Rigorous without dominating: prose measure, a section index, and definitions that a reader
can link to. It is the destination for depth, never a prerequisite for using the product.

## Status

Provenance and freshness, not an admin console. The lead answers "how current is this and
is it trustworthy" in one line; the snapshot history is the archive beneath it.

## Demo

The conversion path for a confused visitor: two real batted balls explained visually, then
a counterfactual instrument. Structurally the strongest page in the product today —
protect it.

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

### Imagery and team context — permitted, but never depended on

**Rule: identity is name-first, and no layout may depend on imagery or team context.**
Every surface must be complete and correct with nothing but `batter_name` — the name-only
rendering is the design, not a fallback state.

Headshots, team, and position are **permitted if and when the data model supports them**,
and must degrade cleanly to name-only: adding or removing one changes what is inside a
row or hero, never the row's structure, height, alignment, or column contract.

Current state (verified): the snapshot content model and the Explore shards carry only
`batter_id` and `batter_name`. `CONTEXT.md` notes `batter_id` is the key for "headshot
lookups," but no headshot code, asset, or request exists, and the site makes **zero**
external requests today. So anyone adding imagery is also making two decisions that this
design system does not make for them, and that must be raised explicitly at that time:

- a headshot CDN would be the site's **first external request and first third-party
  dependency** — a product and privacy decision with a real cost;
- team/position requires a **new field in the snapshot content model**, which is a research
  half change, not a dashboard change (`RESEARCH_RULES.md`, and `CLAUDE.md` rule 6 —
  the dashboard displays, it never computes).

Neither of those blocks any design work. Design to the name-first rule and the question
stays open without holding anything up.

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
  play") and the baseline dead-ended: the player page's only outbound links were the
  leaderboard, methodology, and demo.

  **Half-built in Phase 4.** The player page emits `/explore/?batter=<id>`, and Explore
  reads that parameter on load and preselects the hitter. The link is emitted **only** when
  that hitter is actually in the published Play Explorer catalog, so it can never land on
  an empty selection; where they are absent the section still offers a route onward
  (`/demo/`) rather than dead-ending. Writing the selection *back* into the URL — so a
  hitter chosen inside Explore is linkable and shareable — remains Phase 5's work.
