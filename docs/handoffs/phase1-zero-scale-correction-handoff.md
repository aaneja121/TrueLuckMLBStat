# Handoff — Phase 1 off-scale point-estimate correction, 2026-08-27

## PROJECT

Contact Luck (https://contactluck.com). Canonical references — do not re-derive:
`CLAUDE.md` · `PRODUCT.md` · `CONTEXT.md` · `ARCHITECTURE.md` · `DESIGN.md` ·
`docs/design/guardrails.md` · `docs/design/zero-spine-implementation-plan.md`.

## BRANCH

`ui-redesign-v2`. Selected redesign direction: **Zero Spine**
(`docs/design/directions-2026-08-26.md`, Direction 2).

## TASK

Redesign Phase 1 (design foundation) was approved conceptually, with one required
primitive-level correction before commit or Phase 2: **an off-scale point estimate must get
an explicit graphical state rather than silently disappearing.** Decisions D1a (one canonical
qualified-player domain, no per-player widening, no autoscaling) and D2 (trend keeps its
player-specific y-domain as a declared exception) remain approved and unchanged.

## CURRENT STATE

The correction is **implemented and internally consistent**. All modules import, all nine
templates parse, and the targeted test suites pass. Uncommitted; nothing has been pushed.
**Phase 2 has not been started.**

## FILES MODIFIED

Uncommitted working-tree changes (all under `dashboard/` and `tests/`):

- `dashboard/visuals.py` — `ZeroScale` (+ `point_state`, `clip_state`), margin-free plot
  field, off-scale point marker, open clip chevron, accessible off-scale label.
- `dashboard/build.py` — one `league_scale` built per build; player pages use it (D1a);
  `--cl-zero` in base context; `format_signed` / `signed` Jinja filter; `MINUS_SIGN`.
- `dashboard/templates/base.html` — emits `--cl-zero` on `<html>`.
- `dashboard/templates/player.html` — off-scale / clipped explanatory note.
- `dashboard/templates/_macros.html`, `index.html`, `demo.html` — `signed` filter applied.
- `dashboard/static/style.css` — two-grade tokens, type/spacing/radius scales, scoped
  wrapping, `.cl-scale*`, `.num`, skip link, focus ring, `.interval-clip-caret`,
  `.interval-point-offscale`, `.interval-clip-note`.
- `dashboard/static/{play,explore,demo_simulator,showcase_whatif}.js` — U+2212 in
  `formatSigned` (2 lines each).
- `tests/test_dashboard_zero_scale.py` — **new**, 42 tests.
- `tests/test_dashboard_{visuals,build,demo_page,responsive_overflow}.py` — updated.

## IMPLEMENTATION COMPLETED

**Off-scale point estimate.** Two edge marks, deliberately distinct in fill *and* scale:

- **Open outward chevron** (`polyline`, `.interval-clip-caret`, line weight) — the interval
  continues past this edge; the point estimate is still inside.
- **Solid outward triangle** (`polygon`, `.interval-point-offscale`, point-marker scale,
  `--surface` ring) — the point estimate itself lies beyond this edge.

A dot is never drawn at a clamped position. `data-clipped` and `data-point-offscale`
attributes expose both states. Structural fact the design relies on, verified against the
snapshot: an off-scale point *always* implies its interval clips on that side (guaranteed by
`lower <= point <= upper`, 0 violations in 628 players), so the two marks never coexist on one
edge; the solid marker wins and the label states both facts.

Handled explicitly: interval clipped low / high / both; point clipped low / high; interval
clipped with point in-domain; point and interval clipped on the same side; interval spanning
the whole domain; **interval entirely off-field** (18 such players — no line is drawn at all);
zero-width interval off-scale (1-BBE players).

Rendered distribution across 628 player pages: 30 `below`, 13 `above`, 585 `in`.
**The leaderboard never clips** (0 markers) — its domain is built from exactly the qualified
population it draws.

## TESTS COMPLETED

```
tests/test_dashboard_zero_scale.py
tests/test_dashboard_visuals.py
tests/test_dashboard_responsive_overflow.py     67 passed
tests/test_dashboard_build.py
tests/test_dashboard_demo_page.py               49 passed
```

Covers: point below / above / in-domain; interval clipped with in-domain point; point +
interval clipped; both ends clipped; interval entirely off-field; zero-width off-scale
interval; the two marks being distinct element types; accessible off-scale description
(value, interval, direction, domain); canonical domain unchanged regardless of how far
off-scale a point is; and that off-scale handling introduces no per-player autoscaling.

## TESTS NOT YET RUN

- **`make check` in full** — not re-run after the off-scale correction. It passed
  (1802 passed, 8 skipped) at the end of the pre-correction Phase 1 work. The dashboard
  subsets above have passed since; `ruff format`, `ruff check` and `mypy` have **not** been
  run against the corrected `visuals.py` / `build.py` / test file.
- The remaining ~1,700 non-dashboard research tests are untouched by this diff but unverified
  since the correction.

## VISUAL STATES VERIFIED

Verified **before** the off-scale correction (still valid — the correction changed only the
edge marks and the note text):

- Invariant Z in the live DOM at 1440: 248 bars, zero-fraction spread 0, all at screen
  x 697.17.
- Light contrast: 0 failures, all 6 text tokens and 5 mark tokens on all 3 light surfaces.
  Dark contrast from the shipped CSSOM: 0 failures.
- Mobile 390: 0 mid-word name breaks (81 wraps, all at a space or hyphen), verified by
  per-character Range measurement.
- Overflow sweep: 7 routes × 4 widths (320/375/390/768) with all `<details>` open —
  0 overflowing.
- Leaderboard 1440 + 390 light; clipped player page (671286) 1440 light **and** dark.

## VISUAL STATES NOT YET VERIFIED

**None of the off-scale renderings have been looked at in a browser.** Specifically:

- an ordinary qualified player (e.g. `/players/677951/`)
- an unqualified player whose interval clips but whose point is in-domain (id `694673`)
- a player whose point is **below** the domain (extreme: id `703492`, `-47.03
  [-49.68, -44.39]`, 2 BBE — whole interval off-field, no line drawn)
- a player whose point is **above** the domain (extreme: id `682989`, `+46.18` with a
  zero-width interval, 1 BBE)
- the Johnathan Rodríguez case, id `671286`, `-13.82 [-56.60, +28.40]`, 6 BBE
- an interval spanning the whole domain (id `671155`)
- all of the above at **mobile 390**, and in **dark mode** (the new
  `.interval-point-offscale` `--surface` ring is theme-dependent and unseen in dark)

## KNOWN ISSUES

- The accessible label uses ASCII hyphen for negatives while visible text uses U+2212. This is
  deliberate (assistive-technology compatibility vs. column alignment) but is **not yet
  documented** in `docs/design/typography.md`.
- `--cl-zero` (45.8946%) vs SVG-rendered zero (45.909%) differ by ~0.1 px from 0.1-unit
  coordinate rounding. **Accepted by the owner as harmless**; disappears when the CSS
  primitive replaces SVG rows in Phase 3. Do not spend time on it.
- 43 player pages now show an off-scale point marker instead of a dot. This is the intended
  consequence of D1a and was approved; the correction is what makes it legible.

## CURRENT FAILURES

None in the suites that have been run.

## DO NOT REPEAT

- Do not re-derive the baseline audit, defect list, contrast numbers or CSS inventory —
  `docs/handoffs/ui-redesign-2026-08-26.md` and `outputs/figures/ui_baseline_2026-08-26/`.
- Do not restore per-player domain widening or any autoscaling. D1a is final.
- Do not force the league domain onto the trend chart. D2 is final; the exception is
  documented in `render_trend_chart_svg`'s docstring.
- Do not re-litigate the `overflow-wrap` test replacement — approved.
- Do not pull in deferred work: link colour (Phase 2), `.cl-scale` route adoption (Phase 3),
  numeric-primitive consolidation (Phase 8), amber structural spine (Phase 3/4).
- Playwright note: screenshots land in the repo root, not the CWD — locate with
  `find "$HOME" -name '*.png' -mmin -5`. Delete `.playwright-mcp/` afterwards; it is untracked.

## EXACT NEXT ACTION

Finish Phase 1 verification, then stop. In order:

1. **Run `make check` in full.** If `ruff format` rewrites `dashboard/visuals.py`,
   `dashboard/build.py` or `tests/test_dashboard_zero_scale.py`, accept the reformat and
   re-run. Fix any `mypy` finding on the new `PointState` / `ClipState` literals.
2. **Rebuild and serve** (`ARCHITECTURE.md` § Rebuilding):
   ```
   SNAP=2026-08-14
   .venv/bin/python dashboard/build.py --explore-artifacts-dir "outputs/explorer_build/$SNAP"
   cd dashboard/dist && python3 -m http.server 8000   # may already be running
   ```
3. **Render and inspect the six player pages listed under VISUAL STATES NOT YET VERIFIED**, at
   **1440 and 390**, in **light and dark**. For each, confirm: the exact numeric value is
   legible and unambiguous; the edge marker reads as a direction, not as a value; the open
   chevron and the solid triangle are distinguishable from one another; and the explanatory
   note matches what is drawn.
4. **Confirm no new overflow** — re-run the 7-route × 4-width sweep with `<details>` open.
5. **Confirm dark mode** for `.interval-point-offscale`: its `stroke: var(--surface)` ring must
   still separate the marker from the interval line on the dark ground.
6. **Report** against the owner's seven requested items: implementation chosen for off-scale
   points, files changed, tests added/changed, rendered edge cases inspected, accessibility
   behaviour, confirmation that G2 moved to Phase 4, final Phase 1 test status.
7. **G2 relocation is still outstanding as a document edit.** The owner approved moving the
   trend-chart rendered-exception gate from Phase 1 to Phase 4;
   `docs/design/zero-spine-implementation-plan.md` still lists G2 under Phase 1 in both the
   Phase 1 section and the § 5 gate table. Update both, with the Phase 4 gate verifying:
   player-specific y-domain retained, zero always visible, ≥3 meaningful y ticks, units
   visible, league-range context band, mobile labels ≥ 12 CSS px, accessible description, and
   no false implication that the trend shares the leaderboard scale.
8. **Then STOP for approval. Do not begin Phase 2** (application shell).
