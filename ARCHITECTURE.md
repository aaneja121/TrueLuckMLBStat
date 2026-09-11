# ARCHITECTURE.md -- repository map

Where things live and how data flows, so an agent can find code without re-crawling the
repo. See `PRODUCT.md` for intent, `CONTEXT.md` for terminology, `RESEARCH_RULES.md` for
the modeling/data-safety rules, `README.md` (184 KB) for full version history.

## Two halves

| | Research half | Presentation half |
|---|---|---|
| Code | `src/mlb_luck_score/`, `prospective/`, `evaluation/`, `replication/`, `demo/`, `scripts/`, `notebooks/` | `dashboard/` |
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

### When the season ends

The loop stops on its own. `prospective.prospective_ingestion.assert_data_through_date_
has_completed_games` refuses a `--data-through` date on which no game was completed, so
every date after the final regular-season game (and every league-wide off day, and any
fully postponed slate) is declined instead of re-scoring identical data into a new
snapshot and a new write-once archive key. The scoring entry point exits **3** for this
case, `publish_snapshot.sh` treats exit 3 as a clean stop (exit 0, nothing archived,
built, or deployed), and the daily workflow therefore goes quiet rather than red. Any
other non-zero exit is still a genuine failure.

A second, explicit layer sits beside it: `prospective_config.PROSPECTIVE_2026_SEASON_
END_DATE` (2026-09-27, maintainer-cited) is cross-checked against the live schedule by
`assert_data_through_date_agrees_with_recorded_season_end`, which fails loudly (exit 2)
if real games completed after the recorded finale -- a makeup game, or a stale
constant. It is **not** a date cutoff: dates with no completed games are handled by the
schedule-derived guard above, so a post-finale makeup can never be silently dropped.
See `RESEARCH_RULES.md` for the full rationale.

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
| `pitcher_season_content.py` | Loads/validates the committed pitcher fixture (`pitcher_season_fixture.json`), fail-closed on schema version, on a failed batter-side self-check, and on an unknown role bucket. Recomputes nothing. Mirrors the surface's public strings from `public_labels`. |
| `explore_content.py` | Loads/validates the sharded Play Explorer artifacts (`players.json`, `players/<batter_id>.json`, `games/<game_pk>.json`, `explore-metadata.json`, `showcase.json`, `showcase-sensitivity/<play_id>.json`). Also reports `contact_luck_min`/`contact_luck_max` across every published play, which is the `run_value` domain Explore is drawn on. |
| `demo_content.py`, `demo_counterfactual_content.py` | View-models for `/demo/` and its counterfactual grid. |
| `visuals.py` | The `ZeroScale` domain object, hand-rolled inline SVG (interval bars), and `build_trend_figure`, which returns the season trend as CSS **percentages** plus a marks-only SVG — the trend's text is HTML, never inside a scaled viewBox. Emits CSS classes only — **never a hex color**. |
| `templates/` | `base.html` (shell, header nav, global search, footer), `_macros.html` (leaderboard + pitcher-board tables, season switcher), `index.html`, `player.html`, `explore.html`, `play.html`, `demo.html`, `methodology.html`, `status.html`, `pitchers.html`, `pitcher.html`, `pitchers_redirect.html`. |
| `static/style.css` | ~4800 lines, single stylesheet. `:root` tokens + a `prefers-color-scheme: dark` block. Breakpoints: 640 / 800 / 900 / 1440. |
| `static/*.js` | `app.js` (leaderboard sort/filter + global player search), `explore.js`, `play.js`, `demo.js`, `demo_simulator.js`, `showcase_whatif.js`. All presentational; each is an independent IIFE with no shared state. |
| `explore_fixture/`, `demo_fixture.json`, `demo_counterfactual_grid.json` | Committed development fixtures. |
| `dist/` | Build output. **Gitignored** — may be stale relative to source. |

### Routes emitted

`/` · `/players/<batter_id>/` · `/explore/` · `/plays/` (one shell; reads `?id=<play_id>`
client-side) · `/demo/` · `/methodology/` · `/status/` · `/pitchers/` (redirect stub) ·
`/pitchers/<season>/` · `/pitchers/<season>/<pitcher_id>/` · `/static/*` ·
`/og-image.png` · `/data/dashboard_build_manifest.json`

It is a **season fixture, not a snapshot** -- deliberately not named for one.
Snapshots here are dated, discovered, integrity-checked and replaced as a season
progresses (`snapshot_data.py`); this is one completed season, committed, undated
and never updated, sharing none of that contract.

The pitcher routes are **season-scoped** and pass **two independent gates**:

1. `--pitcher-season-fixture PATH` must be passed, or no pitcher route and no nav entry
   is emitted at all. Fail-closed like the Play Explorer, so a bare `build.py` publishes
   nothing; `scripts/publish_snapshot.sh` passes it explicitly, on one named line, exactly
   as it passes `--explore-artifacts-dir`. The flag is **repeatable**, once per season.
2. Each fixture's season must appear in `dashboard_config.PITCHER_PUBLIC_SEASONS` (**2024**
   only) or the build **fails outright** — no partial site. Which seasons may be published
   is a maintainer decision recorded in `RESEARCH_RULES.md` ("Public launch of the 2024
   pitcher surface"), never a consequence of a fixture existing.

`/pitchers/` itself is a `noindex` redirect stub pointing at the newest published season,
so a season-less link never 404s while the season board stays the one canonical URL. The
season switcher (`_macros.html`) renders only when more than one season is built.

### Global player search

One index over **both** published surfaces, built in `dashboard/build.py` and
rendered into `#player-index-data` on every page; `app.js` only matches and
renders it. Entries are people rather than batters (`name`/`mlbam_id`, the same
key `headshot_url` uses) and each carries a `kind` and a build-time `label`
("Hitter", "Pitcher · 2024"). A person holding both a hitter page and a pitcher
page produces **two labelled results**, never one silently chosen.

The pitcher half is derived from the datasets that already passed the
`PITCHER_PUBLIC_SEASONS` gate, so an unauthorized season cannot be searchable:
it never becomes a dataset, and the build fails first. There is no second season
list anywhere in the search path.

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

## Feature-only publish: building from an already-archived snapshot

`scripts/publish_snapshot.sh --build-from-existing-snapshot --data-through <DATE>`

Ships a **presentation** change against the exact hitter state already in production,
without re-scoring anything.

**Why it has to exist.** A snapshot's `manifest.json` records `repository_commit`,
`generated_at`, `retrieval_timestamps` and a hash of all 32 frozen inputs. Re-scoring a
date that is *already archived* therefore produces a different manifest the moment the
repository has moved on — **even when every scored value is identical** — and
`archive_snapshot.py --sync-history` then correctly refuses to reconcile the fresh copy
with the archived one. That refusal is the archive guard working, not a bug to route
around. The fix is to stop re-scoring an already-archived date.

Stages (5, versus the normal path's 7):

```
ensure_frozen_inputs.py                  (dev parquet + detail JSONs; R2 read only if missing)
  → archive_snapshot.py --sync-history   (READ-ONLY restore of missing snapshots)
  → verify_local_snapshot_integrity.py   (post-condition: every file matches its own
                                          integrity_hashes.json, or the build stops)
  → generate_production_explorer_artifacts.py
  → dashboard/build.py --explore-artifacts-dir … --pitcher-season-fixture …
  → STOP
```

It is **incapable** of the three dangerous operations, and refuses rather than silently
downgrading a request for any of them: it never invokes the scoring script, never invokes
the archive **write** (only `--sync-history`), and never deploys. `--skip-deploy` and
`--skip-archive` are both **refused** as redundant/meaningless. Proven by
`tests/test_publish_snapshot_orchestration.py::TestFeatureOnlyBuildFromExistingSnapshot`,
which asserts the exact five-stage sequence and that the normal scheduled path still
scores and archives.

In CI: `workflow_dispatch` with `build_from_existing_snapshot: true`. It requires an
explicit `data_through` (it must not guess which archived snapshot to build from) and
refuses to combine with `deploy`. The dist is uploaded as
`dashboard-dist-<data_through>` for inspection.

## Promoting a built artifact to production

`.github/workflows/promote-dashboard-artifact.yml` (manual dispatch only).

Deploys an **already-built, already-inspected** artifact by source run id. It builds
nothing: it downloads `dashboard-dist-<date>` from a named run, gates it, and runs
`wrangler pages deploy` on that exact directory.

Splitting "build an artifact" from "promote that artifact" is what makes the thing
inspected and the thing shipped provably the same object rather than the same by
assumption — the failure mode behind the stale-`dist` incident.

The gate is `scripts/verify_dashboard_artifact.py`, which refuses unless:

- the manifest's `data_through_date` and `repository_commit` match what the operator
  typed, **and** the source run's head SHA matches the same commit (all three agree);
- the artifact agrees with **itself** — Explore metadata, the manifest and the files on
  disk all describe one snapshot, and the declared counts match the pages actually built.
  An operator can only confirm what they already believe; these catch a build that is
  wrong in a way nobody thought to type in;
- every published pitcher season is in `PITCHER_PUBLIC_SEASONS` and has a route, and no
  route exists for a season that is not;
- nothing is truncated (required files present, no zero-byte files);
- **the artifact is not older than what production is already serving.** Unreadable live
  manifest ⇒ refuse, because a rollback cannot then be ruled out. `allow_rollback` exists
  for a deliberate revert and defaults off.

It is given no R2 credentials and references no scoring, archive, sync or build script —
its safety is structural, asserted by
`tests/test_verify_dashboard_artifact.py::TestThePromotionWorkflowCannotDoAnythingElse`.
It shares the publish loop's `publish-prospective` concurrency group, so a promotion
queues behind an in-flight publish instead of racing it to the same Pages project.

## Rebuilding `dashboard/dist/`

`dist/` is gitignored and can be stale (it may hold a build from another branch). It has
**no clean-working-tree guard** — only `prospective/run_v1_1_2026_scoring.py` has one —
so it can be rebuilt with uncommitted docs present.

**Production-faithful rebuild from the newest local snapshot** (steps 5–6 of
`scripts/publish_snapshot.sh`; no snapshot generation, no archive, no deploy, no network):

```bash
SNAP=2026-09-01   # newest local snapshot
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

Local snapshots run from **2026-08-05** through **2026-09-01** — 27 dated snapshots
under `artifacts/prospective/v1_1/`, plus a `2026-08-08__refreshed` variant beside the
2026-08-08 one. 2026-08-07 is absent by design and shows on `/status/` as a gap row, so
the set is 27 snapshots over 28 dates.

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

3. **The snapshot history at real length.** The local set is 27 snapshots. Confirm the
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
  **`ruff`/`mypy` target `src` and `tests` only — not `dashboard/`, `evaluation/`,
  `prospective/`, `demo/`, or `replication/`.** Those are plain script directories,
  importable in tests via `pyproject.toml`'s pytest `pythonpath`; lint and typecheck them
  explicitly (`ruff format/check replication`, `mypy replication`). Dashboard correctness
  is covered by pytest alone.
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
