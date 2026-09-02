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

Discovery, not a search form. The player selector is a **combobox that changes the page's
address**, so a hitter's plays are linkable from their player page and from a shared URL.

**Built in Phase 5.** The order this file originally specified — showcase first, then a
named hitter's plays — was **inverted**, and the reason is Phase 4's own link. A large
share of arrivals now come from a player page's `/explore/?batter=<id>`, with a hitter
already chosen; for those readers a twelve-play editorial set above the results is
obstruction, and Phase 3's lesson (the product arrives early) applies to this route too.
The order is now **picker → the selected hitter's plays → showcase**. With nothing
selected the prompt is one line, so the showcase is still the first substantive content a
cold visitor meets. DOM order, visual order and tab order are the same and no JavaScript
reorders anything.

The URL contract is complete: selecting pushes `?batter=<id>`, `popstate` re-derives the
selection from the address, and an unknown or malformed id clears to the prompt rather
than producing a broken selected state. **The address is the single source of truth** —
a click only changes the URL, and `applyUrl()` is the one path into a selection. That is
what keeps Back and Forward honest and stops the URL from ever describing a hitter who is
not on screen.

## Play page

The sequence must make expected-vs-actual immediately understandable:
**what was hit → what the model expected → what happened → the gap.** The gap is the
answer and belongs above the fold. Baseline defect: a 260 px unlabelled field diagram
occupies the position of greatest prominence while the run-value accounting sits last.

**Built in Phase 6**, with the verdict leading, as on every other route: name and result,
the Contact Luck numeral, then the one figure that produces it, then a sentence stating the
arithmetic in words. The evidence follows in causal order: the contact, then what the model
expected of that contact, then sensitivity where it exists.

The field diagram was kept rather than dropped, because it carries play-specific evidence
the numbers do not. It is now subordinate (below the gap figure, capped at 300 px, hidden
entirely when the play carries no spray angle or hit distance rather than drawing a ball at
home plate) and captioned with what it is and is not.

**Both routes out live in one place**, at full target size: back to *this hitter* in Play
Explorer (`/explore/?batter=<id>`, never a cold Explore) and on to their season page. That
closes the loop the product's value proposition depends on: leaderboard row → player page →
that hitter in Explore → one play → back.

## Methodology

Rigorous without dominating: prose measure, a section index, and definitions that a reader
can link to. It is the destination for depth, never a prerequisite for using the product.

**Built in Phase 7** as eleven ruled sections — Definition, Inputs, Expected run value,
Observed run value, Contact Luck, Aggregation, Uncertainty, Qualification, Interpretation,
Limitations, Version and provenance — each with a stable `id`, under a numbered index that
is sticky at ≥ 1024 and a collapsed table of contents below it. Every section id is a link
target, and `/status/` uses one: its qualification breakdown links to `#qualification`
rather than to the top of the page.

Two things the rewrite is careful about. The **per-play / per-100 confusion** is named and
refused directly, because it is the one reading error this metric invites: a leaderboard
figure is a season rate, and the page says a hitter at `+7.62` did not gain 7.62 runs on a
batted ball. And the **five qualification statuses** are given in their real precedence
order with what each one means, which is Phase 4's finding applied to the reference page:
three of the four unranked statuses are not about sample size at all.

The page also **scopes its own claims**. An early draft said nothing about the batter
reaches the model; `sprint_speed` is an input to the infield-opportunity feature set, so
the claim is scoped to the contact model and the exception is named.

## Status

Provenance and freshness, not an admin console. The lead answers "how current is this and
is it trustworthy" in one line; the snapshot history is the archive beneath it.

**Built in Phase 7.** The lead is one sentence carrying the data-through date, the snapshot
name, the integrity result and the two counts. Under it: the published snapshot as a ruled
record, what is published, the models, the sources, then the history.

Amber is spent on exactly two things — the snapshot this build was rendered from, and its
data-through date — so the accent still means "the frozen official record" on a page where
almost every row is a published snapshot.

Three things the baseline got wrong and this page does not. The snapshot's three model
vocabularies (`model_versions` per model, `component_model_status` per component,
`model_selection_winners` per component plus `near_wall_specialist`) are **joined into one
row per component**, so a status sits with the version it belongs to; the baseline printed
three disconnected lists. A **status value outside the documented vocabulary** is shown as
recorded, in the identifier register, rather than reworded into a label this site chose:
the current snapshot records `outfield: "False"`, and the dashboard displays, it does not
reinterpret. And a **date with no valid stored snapshot is stated**, not skipped, so an
intentional gap (2026-08-07) cannot read as a rendering bug. No cause is asserted, because
the page does not know one.

## Demo

The conversion path for a confused visitor: two real batted balls explained visually, then
a counterfactual instrument. Structurally the strongest page in the product today —
protect it.

**Built in Phase 7**, and the protection held: the four-stage reveal, the
`requestAnimationFrame` ball flight, the `demo-reality-revealed` gate and the simulator's
whole state model are unchanged. What changed is what the page spends before the reader
reaches them. The lede is one line sharing a band with the Play control, and the six-line
note about what the two plays are moved **below** the plays it describes.

Where a quantity here is the same quantity a play page shows, it is now drawn the same way:
outcome probabilities take Phase 6's neutral bar on a full-width track with the recorded
result marked in text, and the payoff is a signed numeral with the word under it. What is
deliberately **not** borrowed is the run-value gap figure. The demo's examples are 2024
fixture plays and are not in the published `run_value` domain, so drawing them on that axis
would assert an alignment that is not true. The gap is stated as two labelled values and a
signed difference instead — which also keeps the demo visibly simpler than the play page,
which is the correct hierarchy for a teaching route.

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

Current state: the snapshot content model and the Explore shards carry only `batter_id`
and `batter_name`. `CONTEXT.md` notes `batter_id` is the key for "headshot lookups."
Anyone adding imagery is also making two decisions that this design system does not make
for them, and that must be raised explicitly at that time:

- a headshot CDN would be the site's **first external request and first third-party
  dependency** — a product and privacy decision with a real cost;
- team/position requires a **new field in the snapshot content model**, which is a research
  half change, not a dashboard change (`RESEARCH_RULES.md`, and `CLAUDE.md` rule 6 —
  the dashboard displays, it never computes).

### Decision on record: leaderboard portraits (owner-requested)

The first of those two was **taken**, at the owner's request, for the leaderboard's Player
column only. What it means and what it does not:

- **The request.** `img.mlbstatic.com`, the standardized MLB "silo" headshot keyed by
  `batter_id` (a square, transparent-background head-and-shoulders portrait, framed
  identically for every player). This is the site's **first and only third-party request**;
  it is `preconnect`ed and declared on the leaderboard route alone, so no other page pays
  for it, and it is lazy, so a reader who never scrolls fetches a handful of images.
- **Still name-first.** The portrait box is a fixed 34×38 px reservation. A missing, failed
  or slow portrait changes no row height, no name x, and no column width, and every row is
  complete and correct with `batter_name` alone.
- **Framing is a property of the fit, not a crop.** `object-fit: contain` with
  `object-position: center bottom` on a square source: hairline, face, chin and jaw cannot
  be clipped on any axis, at any width. `cover` is prohibited here — it fills by cropping,
  and on this source it takes the chin first. No circle mask.
- **Three fallbacks, in order.** The CDN's own neutral silhouette for a player it has no
  photo of; the player's initials, set as type in the same box, if the request fails; and
  the name itself, which was never dependent on either.
- **Density held.** Rows grew 37 → 40 px at 1440 and 104 → 111 px at 390, both inside the
  44 px desktop ceiling `tables.md` sets.
- **Nowhere else.** The player page, Explore, and the play page still make no external
  request and still contain no `<img>`; their tests assert it.

The second decision (team/position) remains open and still blocks nothing.

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

  **Built across Phases 4 and 5.** The player page emits `/explore/?batter=<id>` **only**
  when that hitter is actually in the published Play Explorer catalog, so it can never land
  on an empty selection; where they are absent the section still offers a route onward
  (`/demo/`) rather than dead-ending. Phase 5 added the outbound half, so a hitter chosen
  inside Explore is linkable and shareable, and gave every result row a path to its play.
  The loop closes: leaderboard row → player page → that hitter in Explore → one play →
  back.

  **One combobox implementation now ships** (`window.ContactLuck.createCombobox` in
  `app.js`). The header and Explore differ only in what an option looks like and what
  selecting one does: the header navigates, Explore changes the address. Everything else —
  roles, `aria-expanded`, `aria-activedescendant`, Up/Down/Home/End, Enter, Escape, outside
  click, the announced result count, the non-selectable empty state, diacritic-insensitive
  matching — is the shared factory. **Explore does not use the header's mobile full-screen
  sheet**, deliberately: the sheet exists because the header field is a cramped strip
  competing with five routes, whereas Explore's picker is full-width page content with its
  listbox directly beneath it, and a modal there would add a state without removing a
  problem.
