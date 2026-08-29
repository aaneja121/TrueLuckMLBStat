# ARCHITECTURE.md -- repository map

Where things live and how data flows, so an agent can find code without re-crawling the
repo. See `PRODUCT.md` for intent, `CONTEXT.md` for terminology, `RESEARCH_RULES.md` for
the modeling/data-safety rules, `README.md` (184 KB) for full version history.

## Two halves

| | Research half | Presentation half |
|---|---|---|
| Code | `src/mlb_luck_score/`, `prospective/`, `evaluation/`, `demo/`, `scripts/`, `notebooks/` | `dashboard/` |
| Job | download → clean → train → score → snapshot | read a snapshot → render static HTML |
| Rules | `RESEARCH_RULES.md` (read in full) | `PRODUCT.md` + this file |
| Enforced boundary | — | `tests/test_dashboard_isolation.py`: **no module under `dashboard/` may import `mlb_luck_score`, training, scoring, or download code** |

`dashboard/dashboard_config.py` re-declares every path it needs rather than importing
from `prospective/` — the boundary is enforced by having nothing to import.

## Data flow (production)

```
completed MLB slate
  → prospective/run_v1_1_2026_scoring.py --data-through YYYY-MM-DD
      → immutable snapshot: outputs/prospective/v1_1/<date>/ + artifacts/prospective/v1_1/<date>/
  → scripts/generate_production_explorer_artifacts.py  → outputs/explorer_build/<snapshot>/
  → dashboard/build.py [--explore-artifacts-dir …]     → dashboard/dist/
  → wrangler pages deploy dashboard/dist --project-name=contact-luck   (Cloudflare Pages)
```

`scripts/publish_snapshot.sh --data-through YYYY-MM-DD` is the **only** orchestration
entry point and chains all of the above.
`.github/workflows/publish-prospective.yml` triggers it daily at 13:00 UTC
(real deploys gated on the repo variable `PROSPECTIVE_AUTO_DEPLOY=true`).

## `dashboard/` internals

**Stack:** Python 3 + Jinja2 → static HTML; hand-written CSS; vanilla ES5-style IIFE JS.
No framework, no bundler, no npm, no runtime dependency. SVG charts are generated at
build time in Python.

| File | Role |
|---|---|
| `build.py` | Entry point. Renders every page, copies static assets + JSON, writes `dist/data/dashboard_build_manifest.json`. `dist/` is wiped and rebuilt each run. |
| `dashboard_config.py` | Paths, `DASHBOARD_VERSION`, `SITE_URL` (`https://contactluck.com`). |
| `snapshot_data.py` | The **only** place snapshots are discovered, integrity-checked, classified, and ranked by precedence. Fails closed. |
| `content.py` | Snapshot JSON → page view-models (leaderboard rows, player detail, trend, status). Recomputes nothing. |
| `explore_content.py` | Loads/validates the sharded Play Explorer artifacts (`players.json`, `players/<batter_id>.json`, `games/<game_pk>.json`, `explore-metadata.json`, `showcase.json`, `showcase-sensitivity/<play_id>.json`). Also reports `contact_luck_min`/`contact_luck_max` across every published play, which is the `run_value` domain Explore is drawn on. |
| `demo_content.py`, `demo_counterfactual_content.py` | View-models for `/demo/` and its counterfactual grid. |
| `visuals.py` | The `ZeroScale` domain object, hand-rolled inline SVG (interval bars), and `build_trend_figure`, which returns the season trend as CSS **percentages** plus a marks-only SVG — the trend's text is HTML, never inside a scaled viewBox. Emits CSS classes only — **never a hex color**. |
| `templates/` | `base.html` (shell, header nav, global search, footer), `_macros.html` (leaderboard table), `index.html`, `player.html`, `explore.html`, `play.html`, `demo.html`, `methodology.html`, `status.html`. |
| `static/style.css` | ~1760 lines, single stylesheet. `:root` tokens + a `prefers-color-scheme: dark` block. Breakpoints: 640 / 800 / 900 / 1440. |
| `static/*.js` | `app.js` (leaderboard sort/filter + global player search), `explore.js`, `play.js`, `demo.js`, `demo_simulator.js`, `showcase_whatif.js`. All presentational; each is an independent IIFE with no shared state. |
| `explore_fixture/`, `demo_fixture.json`, `demo_counterfactual_grid.json` | Committed development fixtures. |
| `dist/` | Build output. **Gitignored** — may be stale relative to source. |

### Routes emitted

`/` · `/players/<batter_id>/` · `/explore/` · `/plays/` (one shell; reads `?id=<play_id>`
client-side) · `/demo/` · `/methodology/` · `/status/` · `/static/*` · `/og-image.png` ·
`/data/dashboard_build_manifest.json`

The Play Explorer is **fail-closed**: a bare `build.py` builds it *disabled*. Pass
`--explore-artifacts-dir dashboard/explore_fixture` for local work.

### Client-side data loading

`/explore/` fetches only `players.json` up front, then one `players/<batter_id>.json`
after a hitter is selected. `/plays/` fetches only the one `games/<game_pk>.json` it
needs. This sharding exists because the old monolithic index hit Cloudflare Pages'
25 MiB per-asset limit — do not reintroduce a single large index.

## Current redesign baseline (resolved)

The active branch **`ui-redesign-v2`** is based on the current **live cream/amber
site** — that is the intended and correct starting point. The dark "broadcast/Statcast"
redesign on `backup-pre-skill-redesign` was **deliberately archived** before this process
began: do not restore or inherit it wholesale. A specific idea from it may be inspected
selectively later, but the new design must be derived from this baseline, `PRODUCT.md`,
`CONTEXT.md`, the rendered product, and the design audit — which is complete, and whose
conclusions are now `DESIGN.md` + `docs/design/`.

## Rebuilding `dashboard/dist/`

`dist/` is gitignored and can be stale (it may hold a build from another branch). It has
**no clean-working-tree guard** — only `prospective/run_v1_1_2026_scoring.py` has one —
so it can be rebuilt with uncommitted docs present.

**Production-faithful rebuild from the newest local snapshot** (steps 5–6 of
`scripts/publish_snapshot.sh`; no snapshot generation, no archive, no deploy, no network):

```bash
SNAP=2026-08-14   # newest local snapshot
.venv/bin/python scripts/generate_production_explorer_artifacts.py \
    --data-through "$SNAP" \
    --output-dir "outputs/explorer_build/$SNAP" \
    --skip-sensitivity
.venv/bin/python dashboard/build.py --explore-artifacts-dir "outputs/explorer_build/$SNAP"
cd dashboard/dist && python3 -m http.server 8000
```

The generator is read-only against the snapshot (integrity-checked, then projected into an
ephemeral `outputs/explorer_build/` dir). **`--skip-sensitivity` matters**: without it the
generator *retrains the frozen contact model* to build showcase "What if?" grids. Drop the
flag only when you actually need that module, and say so first.

`--explore-artifacts-dir dashboard/explore_fixture` is the lighter alternative — committed
development fixture data, fine for structural work, not representative of production scale.
A bare `build.py` with no flag builds the Play Explorer **disabled**.

Local snapshots are present through **2026-08-14**.

### Check on Explore after a production build

Three things cannot be verified against the committed fixture (53 hitters, 56 plays, at
most **two plays per hitter**) and must be looked at the first time `/explore/` is built
from real Explorer artifacts:

1. **The `run_value` domain.** It is read from the min and max per-play Contact Luck across
   every published play, so production values set it. Confirm the axis ticks are sensible
   and that the bulk of plays are not crushed into the middle by one extreme.
2. **The picker at production catalog size** (hundreds of hitters, not 53): matching speed
   and the eight-result cap.
3. **The results table at real per-hitter play counts** (hundreds of rows, not two): row
   density, and that the page stays responsive while filtering and sorting client-side.

And on `/plays/`, which draws on the same domain:

4. **Open several production play ids** from Explore rows and from showcase rows. Confirm
   the gap figure's two marks land where the printed values say, and that a play whose
   expected and observed values nearly coincide still renders as a ring around a dot.
5. **Sensitivity availability.** The committed fixture ships zero interactive showcase
   plays, so the "What if?" module has never been rendered from committed data. On a
   production build, open a play with `interactive_available: true` and confirm the section
   reveals, the sliders move only *Expected RV, this contact*, and the play's own Contact
   Luck and observed run value do not change.
6. **Location completeness.** The field diagram hides itself when a play has no spray angle
   or hit distance. Check the share of production plays that render it; if it is very low,
   the section is worth revisiting.
7. **The full loop**, on production ids: player page → `/explore/?batter=<id>` → a play →
   back, and the play page's own back link carrying the same hitter.

Nothing about this is a blocker for a fixture build; it is what the fixture cannot tell you.

### Open data-integrity questions — resolve before deployment

Neither is a rendering defect. In both cases the dashboard is displaying the frozen
artifact correctly, which is exactly why they have to be answered in the data rather than
in the templates. **Do not change scoring or renderer behaviour to make either go away.**

1. **Observed run value vs. outcome class (found in Phase 6, unresolved).** Some fixture
   plays carry an `observed_run_value` that matches a *different* outcome class's run-value
   table entry than the one the play's `outcome_class` names. Two explanations are live and
   they are not distinguishable from the dashboard: it is legitimate whole-play run-value
   behaviour, because observed run value counts baserunner advancement and state changes
   and therefore need not equal the recorded result's table value; or it is a
   fixture/artifact-generation defect. Settle it against the scoring ledger on a production
   build before shipping, not by assumption.

2. **`component_model_status.outfield` is the string `"False"` (found in Phase 7).** The
   snapshot records a value that is not one of `component_confidence`'s five documented
   statuses, so no public label exists for it. `/status/` shows it verbatim in the
   identifier register rather than reinterpreting it, which is the honest display, but the
   underlying value is almost certainly a serialization slip upstream. Check what the
   outfield component's status is meant to be and fix it at the source; the status page
   will then label it like every other component with no dashboard change.

### Check on the supporting routes after a production build (Phase 7)

3. **The snapshot history at real length.** The local set is 10 snapshots. Confirm the
   record list stays readable at production length, and that the `is-current` marker still
   lands on exactly one row.
4. **Snapshot-type coverage.** No `retrospective_backfill` and no
   `invalid_incomplete_snapshot` exists locally, so neither type label has ever rendered.
   Invalid snapshots are excluded from the history by construction; confirm that is still
   what you want a reader to see, or that the count of excluded ones is surfaced.
5. **Gap rows.** The gap marker is exercised by the real 2026-08-07 hole. Confirm it reads
   as an absence rather than an error, and that a longer run of missing dates collapses to
   one row with a date range.

## Tests & tooling

- `make check` = `ruff format` + `ruff check` + `mypy` + `pytest`.
  **`ruff`/`mypy` target `src` and `tests` only — not `dashboard/`.** Dashboard
  correctness is covered by pytest alone.
- Dashboard tests: `tests/test_dashboard_build.py`, `test_dashboard_build_cli.py`,
  `test_dashboard_content.py`, `test_dashboard_visuals.py`,
  `test_dashboard_snapshot_discovery.py`, `test_dashboard_isolation.py`,
  `test_dashboard_responsive_overflow.py`, `test_dashboard_explore_*.py`,
  `test_dashboard_play_page.py`, `test_dashboard_player_page.py`,
  `test_dashboard_demo_*.py`, plus `tests/dashboard_snapshot_fixtures.py`.
  `test_dashboard_responsive_overflow.py` and `test_dashboard_isolation.py` are the two
  most likely to be tripped by UI work.
- The suite is **fully offline**. Never add a test needing network or real Statcast data.
- **Browser verification uses the globally configured Microsoft Playwright MCP** — the
  canonical browser capability for this project. Do not add a repo-local Playwright
  install, browser dependency, or dev-server skill.

## Docs

`README.md` (full research history) · `PRODUCT.md` · `CONTEXT.md` · `RESEARCH_RULES.md` ·
`CLAUDE.md` (router for Claude) · `AGENTS.md` (equivalent router for Codex and other
cross-agent tooling).

`DESIGN.md` (repo root) is the canonical design system and is itself a router: it holds the
visual thesis, the eight design principles, the reference world, and the standing
decisions, then points into `docs/design/` — `guardrails.md` (anti-patterns + preserved
invariants; load for any dashboard change), `information-architecture.md`, `typography.md`,
`color.md`, `layout.md`, `tables.md`, `dataviz.md`, `interaction.md`, `responsive.md`,
`accessibility.md`. Load only the files a task needs.

`docs/handoffs/` holds dated session handoffs for multi-session work.
