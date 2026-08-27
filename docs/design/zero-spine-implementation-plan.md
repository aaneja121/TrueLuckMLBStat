# Zero Spine — implementation plan

**Selected direction:** Direction 2 · The Zero Spine (`docs/design/directions-2026-08-26.md`).
**Status: plan only. No frontend code has been modified.**

Governed by `DESIGN.md`, `docs/design/guardrails.md`, `PRODUCT.md`, `CONTEXT.md`.
Companion decision surface (diagrams, gates, order): the published artifact linked from the
session that produced this file.

---

## 0. What code inspection changed

Four findings from reading `visuals.py`, `build.py`, `_macros.html`, `style.css` and the
dashboard tests. Each one changes the plan; none was visible from the docs alone.

### 0.1 The zero position is *not* currently identical across renderings

`render_interval_bar_svg` uses `margin = 10` inside a 220-wide viewBox (compact) and
`margin = 28` inside a 480-wide viewBox (full). With the same domain and the same data:

```
compact zero fraction = (10 + 200·f) / 220
full    zero fraction = (28 + 424·f) / 480     where f = (0 − min)/(max − min)
```

These are equal only at `f = 0.5`. At the measured `f ≈ 0.4593` they render **0.4630 and
0.4640** — which is exactly the "0.463/0.464" recorded in `DESIGN.md` as if it were one
shared position. It is two positions that differ by ~0.1% of width.

At today's ornament scale nobody sees it. Under Zero Spine, where a page-level rule must sit
on the marks, a 0.1% divergence is a visible misregistration and a violation of the brief's
guardrail 3 ("do not fake visual alignment"). **The primitive must remove SVG-internal
horizontal margin entirely** — see §1.2.

### 0.2 The domain already widens per page, and the widening is deliberate

`build.py:153` computes `player_domain` as one padding pass over
`qualified_intervals + this player's own interval`. For a qualified player that equals the
leaderboard domain. **For an unqualified player it widens**, so their spine sits at a
different fraction than the leaderboard's. The existing code comment documents this as
intentional (it prevents clipping).

Under Zero Spine that is no longer acceptable — it is the one case where the site would draw
two different zeros. §1.1 proposes one snapshot-level scale plus an explicit clip affordance,
and this is **decision D1** below because it changes what an unqualified player's bar looks
like.

### 0.3 An existing test asserts an anti-pattern the design system bans

`tests/test_dashboard_responsive_overflow.py::test_body_rule_sets_overflow_wrap_anywhere`
asserts that `body { overflow-wrap: anywhere }` is present in `style.css`.
`docs/design/guardrails.md` anti-pattern 5 bans exactly that rule, and it is the root cause of
"Pete Crow-Armstro / ng" on mobile.

**The test and the design system are in direct conflict.** Phase 1 must rewrite that test to
assert the *scoped* replacement instead. This is a deliberate, called-out reversal of an
existing guard, not a silent deletion — see approval gate **G1**.

### 0.4 The verdict is currently two `<td>`s, and no value carries a sign

`_macros.html` renders `<td class="numeric">{{ "%.2f"|format(row.score) }}</td>` and a
separate `<td class="interval-col">`. So: the merge into one verdict cell is a real template
change, and **positive values ship with no `+` today** — the explicit-sign requirement
(the non-colour channel for the semantic pair) is new behaviour, not a restyle.

Also confirmed present and needing Phase 3 work: `.overflow-x` with no `min-width`;
`<th data-sort-key>` with a click handler and no `role`/`tabindex`/`aria-sort`; the filter
input labelled only by `placeholder`; `.interval-bar-compact { margin: 0 auto }` (centring
that makes a shared spine x impossible to guarantee from the cell alone).

---

## 1. Zero Spine primitive architecture

The primitive exists so that **no page can invent its own zero axis.** One scale object built
once per build, two renderers over it, two invariants enforced by tests.

### 1.1 Canonical quantitative domain

A frozen build-time object, constructed once in `build.py` and threaded everywhere:

```
ZeroScale
  name           "league_per_100" | "run_value" | "component_per_100" | …
  unit_label     "Runs / 100 eligible BBE"
  domain_min     padded lower bound   (always ≤ 0)
  domain_max     padded upper bound   (always ≥ 0)
  zero_fraction  (0 − domain_min) / (domain_max − domain_min)
  true_min       unpadded observed minimum   (for clip detection)
  true_max       unpadded observed maximum
  ticks          [(value, fraction), …]      explicit, for any non-league scale
```

**The product has exactly three scales, and no fourth may be created:**

| Scale | Unit | Where | Domain source |
|---|---|---|---|
| `league_per_100` | Runs / 100 | Leaderboard rows, player hero, any cross-player comparison | One snapshot-level domain, built once |
| `run_value` | Runs | Play page gap figure, Explore play distribution, `/demo/` | One snapshot-level domain over scored plays |
| `component_per_100` | Runs / 100 | Component decomposition on a player page | Per-player: that player's total + components |

**Decision D1 — the `league_per_100` population.** Two options:

- **D1a (recommended).** Build the domain from the **qualified** population only (as today's
  `qualified_domain` does), and give any value outside it an explicit **clip caret** rather
  than a silent clamp. Unqualified player pages then use the *identical* domain, and the
  spine is genuinely one x everywhere. Cost: an unqualified hitter with a wild small-sample
  interval will now visibly clip where today the page silently widened to fit them.
- **D1b.** Build the domain from **all scored players**. Nothing ever clips, but small-sample
  intervals are much wider than qualified ones, so the domain stretches and all 124 qualified
  bars compress toward the spine — the leaderboard loses resolution to serve ~500 pages.

D1a preserves the leaderboard, which is the product. The clip caret is more honest than a
silently rescaled page. **This needs your sign-off before Phase 1** because it changes what an
unqualified player sees.

### 1.2 Zero-position calculation — the two invariants

> **Invariant Z (zero locus).** Every scale renders zero at the same fraction of its plot
> field, exposed once as the CSS custom property `--cl-zero`. Different figures may carry
> different *domains*; they never carry different *zeros*.
>
> **Invariant D (domain identity).** Every rendering of Contact Luck Runs per 100 uses the one
> snapshot-level `league_per_100` domain. No per-page recomputation, no per-chart autoscale.

Invariant Z is what lets one continuous rule descend a page through figures whose domains
differ (the run-value gap on a play page sits under the same spine as a per-100 mark), and it
is achieved by **padding the shorter side of every scale until `zero_fraction` equals the
constant** `--cl-zero`. Invariant D is the stronger guarantee, and it applies only to the
per-100 quantity — where the brief's guardrail 3 actually bites.

**The anti-fake-alignment guard:** any figure whose scale is *not* `league_per_100` **must**
render visible tick labels with its unit. The spine says "this is zero"; the ticks say "this
is the scale." A figure that shares the zero but hides its domain is exactly the deception
guardrail 3 forbids.

To make Invariant Z mechanically true, the primitive removes SVG-internal horizontal margin
(§0.1). Breathing room comes from the *layout* — the containing element's padding — never
from the coordinate space. Test: for every size variant and every scale,
`rendered_zero_x / field_width == zero_fraction` to within 0.05%.

### 1.3 The reusable primitive — two renderers, one scale

**Renderer A — `.cl-scale`, a CSS layout primitive.** Used wherever a mark sits inside flowing
layout: leaderboard rows, the player hero, component bars. **Not SVG.**

- `--cl-zero` is set once, on `<html>` in `base.html`, from the build context.
- Each row carries build-computed percentages as inline custom properties:
  `style="--pt:63.1%; --lo:48.2%; --hi:78.4%"`.
- The plot field is `position: relative`; the interval is an absolutely positioned element
  from `--lo` to `--hi`; the dot is a fixed-size element at `--pt` with `translateX(-50%)`
  and a 2 px `--surface` ring.
- **The spine is a single `::before` on the list container**, at `left: var(--cl-zero)`,
  spanning the container's full height — not one rule per row.

This choice is load-bearing, not stylistic. Because every row's plot field shares one
containing block and one percentage basis, a single continuous rule aligns with all 124 rows
**by construction**, at any width, including mobile. Per-row SVG cannot give that: each SVG
scales independently and `margin: 0 auto` centring (today's `.interval-bar-compact`) makes the
absolute x depend on cell width. It also removes 124 inline SVGs per page.

**Renderer B — `visuals.render_scale_figure(...)`, build-time SVG.** Used where a figure needs
axes, ticks, or many marks at once: the league distribution strip, the axis header, the trend,
the run-value gap. Consumes the same `ZeroScale`.

A test asserts A and B place zero identically for the same scale.

**Where a percentage is computed:** in Python, at build time, from values already in the
snapshot. Templates print it. `PRODUCT.md` §5 forbids deriving a *score, rank, interval or
probability* in a template or in JS — a layout percentage is none of those, and no scoring
code is imported. `tests/test_dashboard_isolation.py` stays green.

### 1.4 Point-estimate and interval positioning

- `pt = (point − domain_min) / span`, likewise `lo`, `hi`.
- Dot ≥ 8 px (per `docs/design/dataviz.md`), fixed pixel size so it never distorts, 2 px
  surface ring so it reads where it overlaps the spine or another mark.
- Interval: 2 px, sign-coloured, spanning `lo`→`hi`. **No minimum rendered length** — a
  minimum would visually overstate uncertainty, which is a statistical-honesty defect, not a
  polish item. At a 720 px field over a ~30 runs/100 span, 1 run/100 ≈ 24 px, so real
  intervals are never invisible.
- Sign colour follows the **point estimate only**. The interval takes the same colour as its
  point estimate — never a separate "uncertainty grey."

### 1.5 Clipping behaviour

Today: silent clamp (`min(max(point, domain_min), domain_max)`).

Replacement: when `point`, `lower` or `upper` falls outside the domain, draw to the boundary
and terminate with a **clip caret** (a small outward triangle), set
`data-clipped="low|high|both"`, and keep the exact value in the visible numeral and the
accessible text. A reader must never see a bar end at the edge and believe that is the value.

### 1.6 Text / value collision — retaining exact-value scanning

This is the brief's guardrail 2, and the original Direction 2 write-up got it wrong: it put
the value at the interval's outboard tip, which reads beautifully when the table is sorted by
score and **collapses the moment the user re-sorts by BBE**, scattering values across the
row's width exactly when scanning matters most. Value position must not depend on sort order.

**Recommended (V1): a trailing fixed-width numeric box, right-aligned at a constant x.**
Decimals align down all 124 rows. Sign is carried three ways — the explicit `+`/`−` glyph
(U+2212), the mark's colour, and the mark's side of the spine.

This is *not* the Register's architecture, and the difference is hierarchy, not decoration:

| | The Register | Zero Spine V1 |
|---|---|---|
| Primary read | The numeral (18 px/600, mid-row) | The plot field |
| The other element | A 330 px strip confirming position | A 14 px numeral confirming magnitude |
| The spine | A feature of one table column | A page-level rule from the header axis down |
| Field share of row | ~28% | ~65% |

**Alternative (V2), for the Phase 3 visual review: two aligned value margins.** Favorable
values right-align at the field's right edge; unfavorable values left-align at the left edge.
Two aligned columns, and the alignment itself becomes a sign cue. Elegant under the default
sort; noisier under a re-sort, when values alternate sides. Worth seeing rendered before
choosing. **Approval gate G3.**

Optional polish either way: on row hover/focus, a hairline leader from dot to numeral.

### 1.7 Very-near-zero behaviour

`|point| < 0.10`: the dot centres on the spine, its surface ring keeping it legible over the
2 px rule (marks paint above the spine). The value is written with an explicit sign — `−0.06`,
never `0.06`. Colour still follows the sign. **No de-emphasis of any kind.** The reader sees a
dot on the spine with a wide interval; the shared axis conveys "small and uncertain" without
any of the treatments guardrail 4 forbids.

Edge case for Phase 8: a point estimate of `−0.001` rounds to `−0.00`. Display the rounded
magnitude with the sign of the **unrounded** value, so `−0.00` is a legitimate rendering.
Exactly `0.0` keeps today's `point >= 0` rule and renders `+0.00` in the favorable colour —
unchanged behaviour, deliberately not revisited here.

### 1.8 Extreme values

Handled by §1.5's clip caret. Additionally the **league distribution strip** needs a
build-time beeswarm/jitter layout so ~124 dots clustered near zero remain individually
visible; overlap is resolved by the 2 px surface ring plus vertical offset, never by
transparency (which would read as de-emphasis).

### 1.9 Responsive transformation

The brief permits abandoning verticality on mobile. **We do not need to** — and that is a
direct dividend of the CSS primitive:

| Breakpoint | Spine | Structure |
|---|---|---|
| ≥ 1280 desktop | Continuous vertical rule, one `::before` on the table body | 300 px identity gutter · 720 px field · 180 px evidence rail |
| 1024–1279 | Same | Evidence tier merges to one sample cell |
| 640–1023 | Same | Identity gutter narrows; evidence merges under the name |
| < 640 mobile | **Still one continuous vertical rule** — every row's plot field shares the same percentage basis, so a single container-level rule registers against all of them | Two-line ruled row; name owns line one; plot field is line two at full width; a **sticky 24 px axis strip** pins the ticks and the zero origin to the top of the list |

The axis header (ticks + labels) is what changes form: full ticks on desktop, a compact
sticky strip on mobile. The spine itself never moves.

### 1.10 Accessible text equivalent

Renderer A produces no SVG, so accessibility moves to text — which is better:

- The visible numeral carries the point estimate with its sign.
- A visually-hidden span per row carries the full statement:
  *"Point estimate +7.62 runs per 100 eligible batted balls; 95% interval +2.10 to +13.14;
  interval does not cross zero."* The last clause is `interval_interpretation`, which
  `CONTEXT.md` defines as purely descriptive — it is honest as text and must never become a
  visual treatment.
- The plot field is `aria-hidden="true"` (it duplicates text already present).
- Renderer B figures keep `role="img"` + `aria-label` (today's interval bar and trend already
  do this correctly — preserve), and the distribution strip additionally gets a `<details>`
  data table, because a 124-mark figure cannot be summarised in one label.

### 1.11 Dark mode

Tokens split into text/mark grades in Phase 1 (`docs/design/color.md`). In dark the spine
(`#e0aa57`) becomes the brightest element on the page — the reference's natural state, a lit
scale on a dark ground. Marks use mark grades (≥3:1); **axis tick labels are text** and use
the text grade (≥4.5:1) on all three surfaces. The dot's ring is `--surface` in both themes.

Note: `style.css`'s `:root` comment currently documents `--accent-amber` as *decorative /
wayfinding only, never semantic*. Zero Spine promotes amber to structural meaning — "the
frozen official record," and the zero locus. That comment must be rewritten in Phase 1, and
the promotion recorded, so a future reader does not treat the spine as ornament.

### 1.12 The trend chart — a declared exception

`render_trend_chart_svg` computes `y_min`/`y_max` from that player's own snapshots. That is
per-player autoscale, which Invariant D forbids for per-100 quantities.

Forcing the league domain onto it would make almost every player's trend a flat line in the
middle of an empty chart — technically consistent, practically useless.

**Recommendation (decision D2):** keep the per-player y-domain, and pay for the exception
explicitly — always draw and label the zero line, render ≥3 explicit y tick values with the
unit, and add a faint `league_per_100` range band behind the plot so the reader can see how
much of the league range this view covers. The trend is then visibly its own scale rather than
silently a different one. **This is a deliberate exception to shared-scale and needs your
sign-off (G2).**

---

## 2. Implementation phases

Each phase runs the full loop and stops:
**implement → test → render → desktop inspect → mobile inspect → design critique →
accessibility check → fix → re-render → verify → STOP.** No phase begins automatically.

Common to every phase — stated once rather than repeated sixteen times:

- **Automated:** `make check` green (`ruff format`, `ruff check`, `mypy`, `pytest`; ruff/mypy
  cover `src` + `tests` only, so dashboard correctness is pytest's job).
  `test_dashboard_isolation.py` must stay green at every step.
- **Manual browser:** the globally configured Microsoft Playwright MCP, at ~1440, ~1280 and
  ~390. Rebuild per `ARCHITECTURE.md` (`generate_production_explorer_artifacts.py
  --skip-sensitivity`, then `build.py --explore-artifacts-dir …`) — `dist/` is gitignored and
  may be stale from another branch.
- **Accessibility floor:** visible focus on every interactive element, no console errors,
  no duplicate ids, one `<h1>`, landmarks intact, `prefers-reduced-motion` respected.

---

### Phase 1 — Design foundation

1. **Objective.** Establish the vocabulary — tokens, scale, type, spacing, primitives — with
   no page redesigned yet.
2. **User-visible outcome.** Deliberately near-zero. Light-theme contrast rises to AA; long
   player names stop breaking mid-word. Nothing is recomposed.
3. **Routes affected.** All, incidentally. None redesigned.
4. **Files.** `static/style.css` (token block, `:root` + dark block, the `body` rule);
   `visuals.py` (add `ZeroScale`, refactor `compute_interval_domain` behind it);
   `build.py` (construct the three scales once); `templates/base.html` (emit `--cl-zero`);
   `tests/test_dashboard_visuals.py`, `tests/test_dashboard_responsive_overflow.py`.
5. **New primitives.** `ZeroScale`; `--cl-zero`; the `.cl-scale` CSS primitive and its
   sub-parts (field, spine, interval, dot, clip caret); `.num` tabular primitive; two-grade
   colour tokens (`--fav-text`/`--fav-mark`, `--unfav-text`/`--unfav-mark`, `--amber-text`);
   the 9-step type scale; the 8 px spacing scale; three radii; focus-ring primitive; skip-link
   styles.
6. **Must survive.** Every rendered page keeps working. Sign colour from the point estimate
   only. Dark mode. The existing interval bar keeps rendering (it is refactored onto
   `ZeroScale`, not replaced yet). No score, rank or interval changes value.
7. **Accessibility.** Two colour grades land here so no later phase inherits a failing token.
   Focus-ring primitive defined. Skip-link styles ready for Phase 2.
8. **Desktop verification.** Every route renders unchanged in composition; computed
   `--cl-zero` matches `ZeroScale.zero_fraction`; contrast sampled on all three light surfaces
   and all three dark surfaces.
9. **Mobile verification.** Player names no longer break mid-word on the leaderboard and
   player pages; nothing else regresses.
10. **Edge cases.** `domain_max <= domain_min` (single-player or degenerate snapshot); a
    snapshot where every interval is identical; the unqualified-player domain question (D1).
11. **Automated tests.** New: zero-fraction identity across every size variant and scale
    (Invariant Z); `ZeroScale` construction including the degenerate case; a token-grade test
    asserting no mark-grade hex is used as a text colour. **Rewritten:**
    `test_body_rule_sets_overflow_wrap_anywhere` → asserts `overflow-wrap: anywhere` is
    **absent** from `body` and present only on the scoped identifier classes.
12. **Manual interactions.** Toggle OS light/dark; tab through an unchanged page to confirm
    the new focus ring; resize 1440→390.
13. **Main design risk.** Doing too much — quietly restyling pages under cover of "tokens."
    Guard: the diff touches the token block, the `body` rule and primitive definitions; no
    page-level selector is recomposed.
14. **Main engineering risk.** The `overflow-wrap` reversal breaking a layout that has been
    silently relying on it (e.g. `snake_case` status codes on `/status/`). Guard: scope the
    rule to `code`, reason codes and snapshot labels in the same commit, and check `/status/`
    before and after.
15. **Completion criteria.** `make check` green with the rewritten test; Invariant Z test
    passing; all four failing light tokens at ≥4.5:1 on all three light surfaces; no visual
    diff on any route other than name-wrapping.
16. **Dependencies.** None. Blocked only on decisions **D1** and **D2**.

---

### Phase 2 — Application shell

1. **Objective.** The frame every page sits in: header, axis-aware page grid, navigation,
   search, footer, skip link.
2. **User-visible outcome.** A thin ruled header with the wordmark left and routes right; the
   data-through badge at the spine's head in amber; a real skip link; search that closes on
   Escape; a mobile navigation that never wraps to two rows.
3. **Routes affected.** All (`base.html` is the shell).
4. **Files.** `templates/base.html`; `static/style.css` (header, footer, page grid, nav);
   `static/app.js` (global search → combobox semantics); `tests/test_dashboard_build.py`.
5. **New primitives.** The page grid with its declared plot-field column; the combobox
   pattern (one implementation, to be reused by Explore in Phase 5); the skip link; the
   disclosure nav.
6. **Must survive.** Global player search reaching all players including unqualified ones;
   outside-click-closes (works today); the data-through badge; the footer's
   retrospective/not-predictive framing; five routes.
7. **Accessibility.** Skip link first in tab order on every page (none exists today; the
   leaderboard has 135 focusables before the footer). Search as a real combobox:
   `role="combobox"`, `aria-expanded`, `aria-controls`, `aria-activedescendant`; results a
   `listbox` of `option`s; Up/Down navigate; Enter selects; **Escape closes** (it does not
   today). Nav targets ≥44×44 (90×25 today).
8. **Desktop verification.** Header at 1440/1280; search open/closed/keyboard-driven; skip
   link visible on first Tab; focus ring on every nav item.
9. **Mobile verification.** Nav as a disclosure at 390 — **never a two-row wrap**; 44 px
   targets; search as a full-screen sheet; skip link reachable.
10. **Edge cases.** A search query matching nothing; a query matching an unqualified player
    (must appear, neutrally, marked *Reported, not ranked* — never suppressed); very long
    player names in results; JS disabled (nav must still work).
11. **Automated tests.** Every emitted route contains a skip link as the first focusable;
    header/footer markup assertions; combobox ARIA attributes present in rendered HTML.
12. **Manual interactions.** Tab from page load → skip link → main; type in search → arrow
    down → Enter; press Escape; click outside; open the mobile disclosure and tab through it.
13. **Main design risk.** The header becoming a generic SaaS app bar. Guard: it is a thin
    rule with a tick, not a bar with a background; no icon set; no rounded search pill.
14. **Main engineering risk.** Combobox keyboard handling regressing the existing search,
    which works today. Guard: build the new pattern behind the same `data-role` hooks and
    verify each existing behaviour before adding new ones.
15. **Completion criteria.** Skip link on every route; full combobox semantics including
    Escape; no two-row nav at any width ≥320; `make check` green.
16. **Dependencies.** Phase 1 (tokens, focus primitive, `--cl-zero`).

---

### Phase 3 — Homepage + leaderboard  ← the phase that matters

1. **Objective.** Build the Zero Spine leaderboard: shared axis header, league distribution
   strip, continuous spine, merged verdict row, accessible sorting and filtering.
2. **User-visible outcome.** A real measuring axis with ticks; beneath it all 124 qualified
   hitters as one distribution; then ranked rows whose intervals radiate from a single amber
   rule, each with its exact value right-aligned at a constant x. The clipped "Interval width"
   column is gone.
3. **Routes affected.** `/`.
4. **Files.** `templates/index.html`, `templates/_macros.html`; `static/app.js`;
   `static/style.css`; `build.py` (`_leaderboard_view_rows` → percentages instead of per-row
   SVG; distribution-strip figure); `visuals.py` (axis header, distribution strip);
   `tests/test_dashboard_build.py`, `test_dashboard_content.py`, `test_dashboard_visuals.py`,
   `test_dashboard_responsive_overflow.py`.
5. **New primitives.** Axis header with ticks; league distribution strip (build-time beeswarm);
   the two-line mobile row; the sticky mobile axis strip; real tab-pair; sortable header
   button; the sorted-view state trio.
6. **Must survive.** Two independent official rankings, both from #1, both with frozen
   `public_labels` strings, never "worst." Competition ranking on
   `contact_luck_runs_per_100` alone; interval endpoints never reorder anyone; Rank restores
   frozen order. Client-side sort and filter. Row links to the player page. Intervals never
   behind a toggle. An interval crossing zero rendered identically to one that does not.
7. **Accessibility.** Tabs become real buttons (`role="tablist"`/`role="tab"`/`aria-selected`)
   — today they are `display:none` radios, absent from the a11y tree *and* the tab order.
   Sortable headers become `<th aria-sort><button>`; sort changes announced via a polite live
   region. Filter inputs get real `<label>`s (two identical unlabelled inputs ship today).
   Distribution strip gets `role="img"` + label + a `<details>` data table.
8. **Desktop verification (1440, 1280).** Axis ticks aligned to the spine; spine continuous
   through the whole table body; ~8 rows above the fold at 1440×900; decimals aligned down all
   124 rows; official → sorted → restored; both ranking tabs; filter narrowing; **no
   horizontal clipping at any desktop width** (the baseline clips a 1158 px table in a 1100 px
   box at *every* desktop width).
9. **Mobile verification (390).** Two-line row; name on its own line without mid-word breaks;
   spine still continuous down the list; sticky axis strip pinned while scrolling; 44 px
   targets; sort and filter operable.
10. **Edge cases.** A value that clips (caret shown, exact value legible); a near-zero dot
    sitting on the spine; the longest name in the 628-player set ("Christian
    Encarnacion-Strand", 28 chars); ties sharing a rank with the next value skipping; a filter
    matching nothing; sorting by a column then switching ranking tabs.
11. **Automated tests.** Percentages emitted match `ZeroScale` for every row; every row's
    plot field shares one percentage basis; `.overflow-x` carries a `min-width` contract;
    verdict is one cell; every value carries an explicit sign and U+2212 for negatives; tab
    and header ARIA present; no `display:none` on an operable control.
12. **Manual interactions.** Click each ranking tab; keyboard-drive the tablist; sort by each
    column via keyboard and confirm `aria-sort` and the live-region announcement; type in the
    filter; re-sort and confirm the accent leaves the rank column and the "Sorted view — not
    the official Contact Luck ranking" label appears; click Rank to restore.
13. **Main design risk.** Density loss — a plot per row is taller than a text row, and the
    page could drift toward a sparse art piece. Guard: 44 px rows are the ceiling, not a
    target; the evidence rail is not optional at ≥1280; count rows above the fold at 1440×900
    and treat fewer than 8 as a failure.
14. **Main engineering risk.** Spine registration across the header axis, the distribution
    strip and the table body — three independently laid-out components that must share one
    percentage basis. Guard: all three are children of one grid column whose width defines the
    basis; a test asserts the computed left offset of all three is identical.
15. **Completion criteria.** All of 8–12 verified in a real browser at three widths; the
    clipping defect gone; `make check` green; screenshots approved (**G3, G4**).
16. **Dependencies.** Phases 1 and 2. Decisions D1 and D3 (V1 vs V2 value placement).

---

### Phase 4 — Player page

1. **Objective.** One figure that answers *what / how extreme / how uncertain* at once, then
   the decomposition on the same zero.
2. **User-visible outcome.** A full-width instance of the league axis carrying the whole
   league in gridline ink, with this hitter's interval and dot drawn over it at double weight
   and the score printed at the dot; both official ranks written as answers; components as
   signed bars summing visibly to the headline.
3. **Routes affected.** `/players/<batter_id>/`.
4. **Files.** `templates/player.html`; `build.py` (`_player_view`); `visuals.py` (hero figure,
   component bars, trend revisions); `static/style.css`;
   `tests/test_dashboard_content.py`, `test_dashboard_build.py`, `test_dashboard_visuals.py`.
5. **New primitives.** The hero axis figure (league distribution as ground); the component bar
   set on `component_per_100` with explicit ticks; the ruled stat line replacing four tiles.
6. **Must survive.** Score, interval, total runs, eligible BBE, scored games, provisional
   share, component values **each with their status code**, and the trend. Unqualified players
   keep page, score and interval, receive no rank, and are **never** de-emphasised; their
   status chip stays neutral. "Scored Games" keeps saying it is not official MLB games played.
7. **Accessibility.** Hero figure `role="img"` with a label stating value, interval and both
   ranks. Component bars labelled with value **and** status. Trend keeps its current correct
   labelling. Every component status is programmatically associated with its value.
8. **Desktop verification.** Hero at 1440/1280; spine aligned with the leaderboard's (open
   both and compare); a qualified player, an unqualified player, a near-zero player, a
   maximally favorable and maximally unfavorable player.
9. **Mobile verification.** Hero keeps the full axis and league ground — it must not simplify,
   because it is the product; components stack while keeping their shared zero; trend degrades
   to a sparkline with first/last values if labels cannot hold ≥12 CSS px.
10. **Edge cases.** Unqualified player whose interval clips under D1a; a player with one
    snapshot (trend refuses to draw — preserve); the intentional 2026-08-07 gap (never
    interpolated); provisional share of 0 and of 1; a component that is `unavailable`.
11. **Automated tests.** Hero and leaderboard row for the same player produce identical
    `--pt`; component bars sum to the headline within rounding; every displayed component
    value carries a status; unqualified players emit no rank.
12. **Manual interactions.** Navigate leaderboard → player and confirm the mark lands at the
    same x; keyboard-reach the plays link; expand any `<details>` reason codes.
13. **Main design risk.** The hero becoming a giant KPI panel. Guard: no box, no border, no
    fill — the figure is a mark on an axis, and the number sits *at the mark*.
14. **Main engineering risk.** Rendering the league distribution behind ~500 player pages
    (build time, page weight). Guard: generate the distribution geometry **once** and reuse the
    identical markup across pages; measure build time before and after.
15. **Completion criteria.** Cross-route spine registration verified by measurement, not by
    eye; unqualified and near-zero states reviewed; `make check` green; screenshots approved
    (**G5**).
16. **Dependencies.** Phases 1–3 (the leaderboard defines the axis this page must match).

---

### Phase 5 — Explore

1. **Objective.** A discovery instrument — a hitter's season of batted balls as a distribution
   you can see before you filter it.
2. **User-visible outcome.** Showcase plays lead; choosing a hitter puts *their* plays as dots
   on a run-value axis; filters narrow the visible distribution live; the selection is in the
   URL and therefore linkable.
3. **Routes affected.** `/explore/`, and the new inbound link from `/players/<id>/`.
4. **Files.** `templates/explore.html`; `static/explore.js`; `explore_content.py`;
   `static/style.css`; `tests/test_dashboard_explore_*.py`.
5. **New primitives.** The `run_value` scale applied to a play set; the shared combobox from
   Phase 2 reused (the two search implementations differ today for no reason); URL state.
6. **Must survive.** The sharded fetch pattern — `players.json` up front, then one
   `players/<batter_id>.json` (the old monolithic index hit Cloudflare's 25 MiB per-asset
   limit; **do not reintroduce a single large index**). The empty state that hides the table
   *and* the result count together. The one-way "Show all 12" reveal whose trigger hides after
   use. Fail-closed behaviour when the Play Explorer is disabled.
7. **Accessibility.** One combobox implementation with full semantics and Escape. Filters
   labelled. The play distribution `role="img"` + label + a data table. Results table sortable
   with `aria-sort`. 44 px targets.
8. **Desktop verification.** Showcase; select a hitter; filter; sort; clear; deep-link a
   selection by URL and confirm it restores.
9. **Mobile verification.** Combobox as a full-screen sheet; distribution readable at 390;
   "Show all 12" revealing a **ruled list, not six more cards**; filters operable.
10. **Edge cases.** A hitter with very few plays; filters matching nothing; a bad
    `?player=` value in the URL; Explorer disabled (bare `build.py`); slow shard fetch.
11. **Automated tests.** URL round-trip for the selection; empty state hides table and count;
    reveal trigger hides after use; no monolithic index emitted; ARIA present.
12. **Manual interactions.** Type-ahead → arrow → Enter → Escape; copy the URL, reload,
    confirm the same hitter; player page → Explore link.
13. **Main design risk.** Reverting to a generic filter bar over a card grid. Guard: the card
    cap is six and applies here; results are a ruled table; filters are a ruled row with
    visible labels, not a pill bar.
14. **Main engineering risk.** URL state colliding with the existing client-side selection
    logic in a 470-line IIFE. Guard: make URL the single source of truth and derive the
    in-memory selection from it, never the reverse.
15. **Completion criteria.** Selection linkable; sharding intact; empty state and reveal
    preserved; `make check` green; screenshots approved (**G6**).
16. **Dependencies.** Phases 1, 2 (combobox), 4 (the inbound link).

---

### Phase 6 — Play page

1. **Objective.** Make the run-value gap the answer, and put a zero-centred instrument only
   where one is genuinely meaningful.
2. **User-visible outcome.** Contact → expectation → actual → gap, in that order, with the gap
   as a measured distance on a run-value axis above the fold.
3. **Routes affected.** `/plays/?id=<play_id>`.
4. **Files.** `templates/play.html`; `static/play.js`; `static/showcase_whatif.js`;
   `visuals.py` (gap figure); `static/style.css`; `tests/test_dashboard_explore_*.py`.
5. **New primitives.** The run-value gap figure (two ticks + shaded span); the probability bar
   set with the recorded outcome marked in place.
6. **Must survive.** EV, launch angle, batted-ball type, recorded result, the frozen model's
   five-class probabilities, Expected RV → observed RV, the resulting Contact Luck, and the
   sensitivity sliders on showcase plays. The **"Play not found"** state and its route back to
   Explore. Spray angle described as approximate. The field diagram's "illustrative, not a
   reconstruction" caption.
7. **Accessibility.** The field diagram gets `role="img"` and an `aria-label` — **it currently
   has neither, nor a `<title>` nor `aria-hidden`** (confirmed in `play.html:39`). Sliders
   labelled, keyboard-operable, and stepping to discrete values under
   `prefers-reduced-motion`. Probability bars labelled with value and outcome.
8. **Desktop verification.** A favorable play, an unfavorable play, a near-zero play, a
   showcase play with sensitivity, and the not-found state.
9. **Mobile verification.** Gap figure legible at 390; probability bars readable; sliders
   thumb-operable at 44 px; field diagram demoted, not dominant.
10. **Edge cases.** Missing `?id`; malformed `play_id`; a play whose `game_pk` shard is
    absent; a play with no sensitivity grid; probabilities summing to 1 within rounding.
11. **Automated tests.** Field-diagram SVG carries a role and label; not-found state renders
    with its link; the gap figure's zero sits at `--cl-zero`; run-value scale ticks present.
12. **Manual interactions.** Load a showcase play, drag both sliders, confirm the dot slides
    along the axis and the caption updates; load a bad id; return to Explore.
13. **Main design risk.** Mechanically forcing the league per-100 spine onto a page whose
    quantities are run values. Guard: **this page uses the `run_value` scale with visible
    ticks**, and nothing on it claims the per-100 domain. EV, launch angle and probabilities
    are *not* zero-centred and get no spine at all.
14. **Main engineering risk.** `play.js` (243 lines) and `showcase_whatif.js` (186) both write
    to the same DOM regions. Guard: keep their ownership boundaries exactly as they are;
    change markup and CSS, not the ownership split.
15. **Completion criteria.** Gap above the fold at all three widths; field diagram labelled;
    sensitivity working; not-found preserved; `make check` green; screenshots approved (**G7**).
16. **Dependencies.** Phases 1, 2, 5.

---

### Phase 7 — Demo + Methodology + Status

1. **Objective.** Protect the demo, keep methodology rigorous but subordinate, and replace the
   status table outright.
2. **User-visible outcome.** The demo unchanged in substance, its two plays resolving onto one
   shared run-value axis. Methodology as a linkable reference with a sticky index. `/status/`
   as a freshness timeline over a readable history.
3. **Routes affected.** `/demo/`, `/methodology/`, `/status/`.
4. **Files.** `templates/demo.html`, `methodology.html`, `status.html`; `static/demo.js`,
   `demo_simulator.js`; `content.py` (`build_status_page_data` — date formatting);
   `visuals.py` (freshness timeline); `tests/test_dashboard_demo_*.py`.
5. **New primitives.** The freshness timeline (a **date** axis — explicitly *not* the zero
   spine, since dates have no meaningful zero); the definition list that replaces the history
   table below 1024.
6. **Must survive.** The `/demo/` ball-flight animation and its `requestAnimationFrame`
   ownership (SMIL was removed for documented reasons — do not reintroduce it); the
   counterfactual simulator; the `demo-reality-revealed` gating that prevents the outcome's
   sign leaking during stages 1–2; per-component model status; snapshot type and precedence;
   fail-closed snapshot handling.
7. **Accessibility.** Sticky index keyboard-reachable; timeline `role="img"` + label + data
   table; `<details>` for reason codes; all timestamps in `<time datetime>`.
8. **Desktop verification.** Demo walkthrough end to end; simulator; methodology index; status
   at 1440 — the history must not exceed a normal page height (**8,363 px today**).
9. **Mobile verification.** Demo animation and simulator at 390; methodology ToC collapsed;
   status history as a **definition list per snapshot**, not a crushed table.
10. **Edge cases.** One snapshot only; the intentional 2026-08-07 gap (must render as a
    visible gap, never interpolated); an `invalid_incomplete_snapshot`; a
    `retrospective_backfill` marker (distinct class today — preserve); reduced motion.
11. **Automated tests.** Timestamps formatted at source with ISO in `<time datetime>`; no
    unbreakable 32-char token in the history markup; gap preserved in the timeline; demo
    reveal-gating classes intact.
12. **Manual interactions.** Run the demo twice (confirm the animation restarts); drag the
    simulator sliders; jump via the methodology index; scroll the status page end to end.
13. **Main design risk.** Over-designing the demo, which is the strongest page in the product
    today. Guard: change its type, spacing and axis only — do not restructure the walkthrough.
14. **Main engineering risk.** Touching `demo.js`'s animation, which has a documented history
    of subtle failure. Guard: treat `demo.js` as read-only unless a rendering bug forces
    otherwise, and re-verify the restart behaviour explicitly.
15. **Completion criteria.** `/status/` under a normal page height at all three widths with no
    31 px columns; demo verified twice through; `make check` green; screenshots approved
    (**G8**).
16. **Dependencies.** Phases 1, 2, 6 (`run_value` scale).

---

### Phase 8 — Cross-product polish

1. **Objective.** Make the eight routes read as one system, and close every accessibility and
   edge-case item.
2. **User-visible outcome.** Consistent rhythm, states and typography across the product; no
   remaining contrast, focus, target-size or labelling failure.
3. **Routes affected.** All.
4. **Files.** `static/style.css` (consolidation and dead-rule removal); any template with a
   residual inconsistency; all dashboard tests.
5. **New primitives.** None — this phase removes duplicates rather than adding.
6. **Must survive.** Everything. This phase changes no behaviour.
7. **Accessibility.** Full sweep: contrast on all six surfaces, keyboard path through every
   route, a11y tree per route, target sizes, reduced motion, heading order, landmarks.
8. **Desktop verification.** All eight routes at 1440 and 1280, both themes.
9. **Mobile verification.** All eight routes at 390, both themes.
10. **Edge cases.** The longest name at every breakpoint; clipped extremes; exact zero;
    `−0.00`; a filter matching nothing; a play not found; one snapshot; Explorer disabled;
    JS disabled; reduced motion; 200% browser zoom.
11. **Automated tests.** Breakpoint consolidation (640/1024/1280 only — 640/800/900/1440
    today); no radius outside the three; no font size off the scale; no `.overflow-x` without
    `min-width`; no SVG text below 12 CSS px after scaling.
12. **Manual interactions.** A full keyboard-only pass through every route; a full pass at 390;
    both themes.
13. **Main design risk.** Treating this phase as the accessibility phase. Guard: a11y is
    delivered in Phases 1–7; this phase *verifies*, and any a11y finding here is a defect in
    an earlier phase.
14. **Main engineering risk.** Regression from CSS consolidation late in the sequence. Guard:
    remove rules only with a rendered before/after at three widths.
15. **Completion criteria.** The anti-vibecode checklist from
    `docs/design/directions-2026-08-26.md` passes on every route; every baseline defect in the
    handoff closed or explicitly deferred with a reason; `make check` green.
16. **Dependencies.** Phases 1–7.

---

## 3. Files and components index

| Area | Files | Heaviest phase |
|---|---|---|
| Scale + figures | `dashboard/visuals.py` (397 lines) | 1, 3, 4 |
| Scale construction, view-models | `dashboard/build.py` (736) | 1, 3, 4 |
| View-models | `dashboard/content.py` (424), `explore_content.py` (770) | 4, 5, 7 |
| Shell | `templates/base.html` (63) | 2 |
| Leaderboard | `templates/_macros.html` (40), `index.html` (91) | 3 |
| Pages | `player.html` (137), `explore.html` (133), `play.html` (165), `demo.html` (211), `methodology.html` (80), `status.html` (86) | 4–7 |
| Styles | `static/style.css` (1,764) | every |
| Behaviour | `app.js` (149), `explore.js` (470), `play.js` (243), `demo.js` (163), `demo_simulator.js` (334), `showcase_whatif.js` (186) | 2, 3, 5, 6 |
| Tests | `test_dashboard_{build,build_cli,content,visuals,snapshot_discovery,isolation,responsive_overflow,explore_*,demo_*}.py` | every |

**Build-time projections proposed** (all from values already in the snapshot; no schema
change, no new field, no scoring code imported):

1. Per-row scale percentages replacing per-row inline SVG (Phase 3).
2. The league distribution strip's beeswarm geometry (Phase 3).
3. The snapshot-history freshness timeline's date positions (Phase 7).

**Explicitly not doing:** no scoring/model change, no retraining, no methodology change, no
schema change, no team/position fields, no frontend framework, no new runtime dependency, no
npm, no bundler.

---

## 4. Key risks

| # | Risk | Phase | Mitigation |
|---|---|---|---|
| R1 | **Spine misregistration** across independently laid-out components | 3, 4 | One percentage basis from one grid column; Invariant Z test; measured (not eyeballed) cross-route comparison |
| R2 | **Density loss** — the instrument turns the product sparse | 3 | 44 px row ceiling; ≥8 rows above the fold at 1440×900 treated as a hard gate; evidence rail non-optional ≥1280 |
| R3 | **Exact-value scanning degrades** | 3 | Fixed-x trailing numeric box (V1); sort-independent by construction; G3 review of V1 vs V2 |
| R4 | **The spine turns decorative** on pages with no zero-centred quantity | 6, 7 | Rule: no spine without a registered quantity; `/status/` uses a date axis; EV/LA/probabilities get none |
| R5 | **Fake alignment** — figures sharing a zero but not a domain | all | Any non-`league_per_100` figure must render visible ticks and its unit |
| R6 | **Unqualified players clip** under D1a | 1, 4 | Explicit clip caret + exact value in text; reviewed at G5 |
| R7 | **Build time / page weight** from a league distribution on ~500 pages | 4 | Generate geometry once, reuse identical markup; measure before/after |
| R8 | **Regressing working behaviour** — demo animation, Explore sharding, search | 5, 6, 7 | Named must-survive lists per phase; treat `demo.js` as read-only; sharding test |
| R9 | **`overflow-wrap` reversal** breaking `/status/` codes | 1 | Scope to `code`/reason codes/snapshot labels in the same commit; verify `/status/` before and after |
| R10 | **Scope creep into the research half** | all | `test_dashboard_isolation.py` green at every step; no `mlb_luck_score` import under `dashboard/` |

---

## 5. Approval gates

Decisions needed **before Phase 1 starts:**

- **D1** — league domain population: qualified-only + clip caret (**recommended**), or
  all-scored + compression. §1.1.
- **D2** — the trend chart's declared exception to shared scale. §1.12.

Visual approvals — each is a stop, with rendered screenshots at 1440 / 1280 / 390 in both
themes:

| Gate | After | What you are approving |
|---|---|---|
| **G1** | Phase 1 | The `overflow-wrap` test reversal, and the two-grade token set |
| **G2** | Phase 1 | The trend's declared exception, rendered |
| **G3** | Phase 3 | **V1 vs V2 value placement** — the single most consequential visual choice |
| **G4** | Phase 3 | The leaderboard as a whole: axis, distribution strip, spine, density |
| **G5** | Phase 4 | The player hero, and the unqualified/near-zero/clipped states |
| **G6** | Phase 5 | Explore as a discovery instrument rather than a filter form |
| **G7** | Phase 6 | The play page's run-value instrument and what gets no spine |
| **G8** | Phase 7 | `/status/` replacement and the untouched demo |
| **G9** | Phase 8 | Final anti-vibecode review across all eight routes |

---

## 6. Order of implementation

```
D1, D2  →  Phase 1  →  Phase 2  →  Phase 3  →  Phase 4  →  Phase 5  →  Phase 6  →  Phase 7  →  Phase 8
decisions  foundation   shell      leaderboard   player     explore      play      demo/meth/status  polish
           G1, G2                   G3, G4        G5          G6           G7            G8            G9
```

Phase 3 is the pivot: it is where the direction either works or does not, and it is the
earliest point at which the concept can be judged on rendered evidence rather than argument.
Phases 4–7 are largely independent of one another once Phase 3 sets the axis, so if scheduling
demands it, 5 and 6 could swap. **Nothing after Phase 1 should start before Phase 3 has been
seen and approved**, because a rejected leaderboard invalidates the axis every later phase
registers against.
