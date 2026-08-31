# Phase 8 Interrupted-Run Handoff

PROJECT
Contact Luck / TrueLuckMLBStat

BRANCH
`ui-redesign-v2`, HEAD `508fd57` ("Redesign Contact Luck supporting routes").
Phases 1–7 are committed. Nothing from Phase 8 is committed.

TASK
Phase 8 — final production hardening. The run was interrupted by an API
streaming error partway through the end-to-end flow checks (§15). This file
records exactly what had been established so the run resumes rather than
restarts.

CURRENT WORKING TREE
Ten modified files plus one new test file, all uncommitted:

```
 M .gitignore
 M dashboard/build.py
 M dashboard/static/app.js
 M dashboard/static/play.js
 M dashboard/static/showcase_whatif.js
 M dashboard/static/style.css
 M dashboard/templates/base.html
 M prospective/prospective_scoring.py
 M tests/test_dashboard_explore_build.py
 M tests/test_dashboard_play_page.py
?? tests/test_phase8_production_hardening.py
```

`dashboard/dist/` is gitignored and currently holds a **fixture** build
(`--explore-artifacts-dir dashboard/explore_fixture`), generated
2026-08-30 16:52 UTC. It is NOT a production-faithful build.

---

## COMPLETED

- **§3 Blocker A — observed run value.** RESOLVED — EXPECTED BEHAVIOR.
- **§4 Blocker B — outfield `"False"`.** RESOLVED — BUG FIXED upstream
  (fix lands on the next snapshot; see UNRESOLVED for the immutability
  consequence).
- **§5 Global footer timestamp.** Done, one canonical formatter.
- **§6 Dead CSS cleanup.** Done, proven-unused only.
- **§7 JavaScript hygiene.** `PLAY_ID_PATTERN` consolidated.
- **§8 Tool scratch hygiene.** `.playwright-mcp/` gitignored and verified.
- **§9 Copy audit.** Clean across 46 rendered routes.
- **§11 Colour-semantics audit.** One real violation found and fixed.
- **§13 Responsive matrix.** 200 checks, complete.
- **§14 Accessibility audit.** 9 route states, complete.
- **§16 Link/route integrity.** 634 pages, complete.
- **§17 Metadata integrity.** 634 pages, complete.

## VERIFIED

**Tests already green (do not re-run to "confirm", only as part of the
final `make check`):**
- `tests/test_phase8_production_hardening.py` — 11 passed (new file).
- Whole dashboard + phase8 selection — **658 passed, 1541 deselected**,
  after every code change listed below.
- Baseline before Phase 8 was 2180 passed / 8 skipped. Phase 8 adds 11
  tests, so the expected full-suite count is **2191 passed / 8 skipped**.
  **`make check` has NOT been run since the Phase 8 edits.**

**Responsive sweep (§13) — complete, 200 checks.**
10 routes/states × 10 widths (1440/1280/1180/1100/1024/900/768/640/390/320)
× 2 themes. Routes covered: home, explore-cold, explore-selected,
player-qualified, play-valid, play-diverging, play-not-found, demo,
methodology, status.
Result: **zero horizontal overflow, zero text below 11 px, zero desktop nav
two-row wrap.** One target-size finding (see FINDINGS).

**Accessibility audit (§14) — complete, 9 route states at 1440.**
Zero duplicate ids, zero dangling ARIA, exactly one visible `<h1>` per state
(including "Play not found"), no heading-level skips, all landmarks present,
skip link first focusable on every route, zero unlabelled inputs, 8
`aria-sort` headers + 8 sort controls on the leaderboard, combobox present
per route, zero unlabelled SVG, zero images.

**Link/metadata crawl (§16, §17) — complete, 634 built pages.**
Zero broken links, zero broken fragments, zero `localhost` links, **zero
external hosts of any kind**. All 44 player→`/explore/?batter=` links
resolve and every linked batter id is present in the published catalog.
Zero metadata gaps (`<title>`, description, og:title, og:url, og:image,
`lang`, viewport) on all 634 pages.

**Copy audit (§9) — complete, 46 rendered routes** (all six route types plus
40 player pages), against visible `<main>` text, entity-unescaped:
zero em dashes inside prose, zero `BANNED_PHRASES`, zero AI-assistant tics,
no caveat repeated more than twice.

**Colour-semantics audit (§11) — complete, stylesheet-wide.**
`--fav-*`/`--unfav-*` appear in 14 rules each, every one sign-bearing.
`--diverging-positive/negative` only in `:root` and `.interval-positive/
negative` (SVG mark classes). Field green now appears only in `:root` and
the two field-diagram blocks. Amber's 31 rules are all frozen-record, zero
spine, or focus.

**Blocker A investigation — traced through code, not the UI.**
- `dashboard/explore_fixture` has **1 of 56** plays where
  `observed_run_value != DEFAULT_RUN_VALUE_MAP[outcome_class]`:
  `747198-56-4`, Kerry Carpenter, `outcome_class="triple"`,
  `observed_run_value=-0.25491574127584554`, `expected=+1.2963`,
  `contact_luck=-1.5513`, `p_home_run=0.9227`, 421 ft at 105.5 mph.
- `play_ledger_schema.py` Version 2.0 notes document the distinction
  explicitly: `outcome_class` is **Rc** (Statcast `events`, the officially
  recorded contact result); `observed_run_value` is **Rf**
  (`ledger["observed_final_run_value"]`, the full accounting including
  realized advancement). Version 1.0 wrongly used Rc and FAILED
  reconciliation against `public_score` for **261 of 630** batters.
- Mechanism located: `attribution_ledger.py` sets
  `rf.loc[mask] = batter_final_base.map(ADVANCEMENT_BASE_VALUE).map(
  FINAL_BASE_RUN_VALUE_MAP)`, and `FINAL_BASE_RUN_VALUE_MAP[0]` **is**
  `DEFAULT_RUN_VALUE_MAP["out"]` — the map deliberately reuses the same
  constants re-indexed by final base. `ADVANCEMENT_BASE_VALUE
  ["retired_while_advancing"] = 0.0`.
- So bit-exact equality with another class's constant is the **expected**
  signature of a legitimate Rf, not a bug signature. The initial "exact
  constant match looks like a mapping bug" hypothesis was tested and
  **disproved**.
- Existing upstream coverage confirms the scenario is anticipated:
  `tests/test_advancement_eligibility.py::
  test_thrown_out_at_home_from_a_triple_is_retired_while_advancing`, plus
  `test_attribution_ledger.py::
  test_final_base_run_value_map_reuses_default_run_value_map_exactly` and
  `::test_final_result_surprise_diverges_from_contact_result_surprise_on_a_
  deterministic_advancement_row`.
- The play is a near-certain home run (92.3%) that stayed in the park for a
  triple with the batter-runner thrown out at home. Contact Luck −1.55.

**Blocker B investigation — traced through code, not the UI.**
- `"False"` occurs **exactly twice** in the whole 2026-08-14 snapshot
  (`component_status_summary.json`, `coverage_and_schema_report.json`) and
  reaches **no** per-player field. Per-player `outfield_defense` statuses
  are `not_calibrated` / `provisional` / `unavailable` — all valid.
- Root cause: `_read_existing_gate_status` (in
  `evaluation/run_v1_final_evaluation.py`) returns `str(node)`. The three
  components read **different key paths with different types**: infield and
  advancement read `gate_summary.overall_status` (a status string), outfield
  reads `validation_summary.per_candidate.measured_contact_only_v07.
  passes_basic_validation` — a **boolean**. `str(False)` → `"False"`.
- Verified against the real gate file: `outputs/tables/
  opportunity_model_comparison_detail.json` has **no** `gate_summary` at
  all; `passes_basic_validation: false` is its only verdict field, so the
  key path is right and the *type* is the defect.
- The canonical mapping already exists and already names this exact input:
  `component_confidence.normalize_model_status` maps `"false"*` →
  `not_calibrated`, is idempotent on valid statuses, and never guesses
  `calibrated`. `tests/test_component_confidence.py::
  test_normalize_model_status_maps_true_false_strings` already covers it.
  **No semantics were invented.**

**Play-page JS behaviour re-verified in a real browser after the
consolidation:** valid id renders (`+0.49 runs`), malformed id shows
not-found **and issues no game-shard fetch**, missing id shows not-found,
`gamePkFromPlayId('744821-35-3') === '744821'`, malformed and empty → null.

## CHANGES ALREADY MADE

| File | Purpose |
|---|---|
| `prospective/prospective_scoring.py` | **Blocker B fix.** Routes all three `component_model_status` values through `normalize_model_status`, and adds `component_model_status_raw` so the unnormalized gate readings stay recorded and the mapping stays auditable. |
| `dashboard/build.py` | Adds `build_timestamp_display` via the existing `_display_timestamp` — one canonical formatter shared with `/status/`. |
| `dashboard/templates/base.html` | Footer build timestamp becomes `<time datetime="{iso}">{reader form}</time>`. |
| `dashboard/static/app.js` | Adds `PLAY_ID_PATTERN` + `gamePkFromPlayId`, exported on `window.ContactLuck`. |
| `dashboard/static/play.js` | Uses the shared helper with a local fallback; drops its own copy of the pattern. |
| `dashboard/static/showcase_whatif.js` | Same consolidation; drops its duplicate pattern and the "duplicated, not shared" note. |
| `dashboard/static/style.css` | Dead-code removal (net −97 lines) and one colour fix. Removed: `.cl-scale-list`(+`::before`), `.interval-clip-note`, `.player-hero-figure`(+2 sign variants), `.player-hero-name`, `.player-meta`, `.player-name-row`(+` h1`), `.player-orientation`(+` a`), `.showcase-whatif-section`, `.terminology-tabs`, `.terminology-tab`, `.trend-zero-line`, `.hide-mobile`, and four orphaned tokens (`--diverging-mid`, `--band-opacity`, `--accent-amber-soft`, `--accent-field-soft`) in both themes. Updated the stale "RETAINED BUT NO LONGER USED HERE" note. **`.trend-line` stroke `--accent-field` → `--text-secondary`.** |
| `.gitignore` | Ignores `.playwright-mcp/`. |
| `tests/test_dashboard_play_page.py` | Two assertions re-aimed from `PLAY_ID_PATTERN.exec` / `if (!match)` at the contract (`gamePkFromPlayId` / `if (!gamePk)` before `fetch(`). |
| `tests/test_dashboard_explore_build.py` | Same two re-aimings in `TestPlayPageRouting`. |
| `tests/test_phase8_production_hardening.py` | **New, 11 tests.** Blocker A (fixture still has a diverging row; every observed value is a real `FINAL_BASE_RUN_VALUE_MAP` value; the self-referential identity holds; a diverging row survives the build verbatim; both pages explain the distinction). Blocker B (boolean → `not_calibrated`; normalization idempotent; `prospective_scoring.py` normalizes and keeps the raw; `/status/` still shows a recorded value verbatim). Footer timestamp on every route, ISO only in `datetime`. |

**Deliberately kept, do not "clean up":**
- `.player-header` — `tests/test_dashboard_player_page.py` slices the
  stylesheet on it as a block boundary.
- `.no-break-words` — unused but an intentional documented defensive
  utility in the wrapping contract.
- `.is-backfill` (player.html) and `.is-filtered-out` (explore.js) — both
  live; they only looked dead to a naive selector scan.

## FINDINGS

1. **Leaderboard target size at ≥768 px.** Measured on `/`:
   `.leaderboard-filter-input` 132×**30**, `.sort-button` 48×**25**.
   At ≤640 both are 44 px, because Phase 3's documented remediation
   (`docs/design/responsive.md` § Mobile homepage hierarchy) was mobile-only.
   Both clear WCAG 2.5.8 AA (24×24) but miss the project's own stricter
   44 px rule (`docs/design/accessibility.md` #9). Raising them at desktop
   costs leaderboard density, which Phase 3 deliberately optimized.
   **Not a WCAG AA failure; a deviation from the internal rule.**
   → classify POST-SHIP unless the owner rules otherwise.
2. **`.trend-line` was field green.** `docs/design/color.md` demotes
   `--accent-field` to "field diagrams and nowhere else", and a season trend
   is not a field diagram — it was the fifth colour the `.trend-band`
   comment directly above it is careful to avoid. **Fixed** to
   `--text-secondary`. Rendered verification of the trend chart in both
   themes is still outstanding.
3. **No favicon exists.** No `<link rel="icon">` and no icon asset in the
   repo, so production genuinely 404s `/favicon.ico`. The only image asset
   is `og-image.png` (1200×627), wrong shape for an icon. Per §18, adding
   one would be inventing branding. → BACKLOG.
4. **Orphaned comment in `play.js` lines 30–34** still describes the removed
   `PLAY_ID_PATTERN`. Cosmetic residue from the consolidation, not yet
   tidied. → tidy on resume.
5. `src/mlb_luck_score/scoring/run_season_aggregation.py` (development path)
   and `evaluation/run_v1_final_evaluation.py` (**sealed** V1.0 evaluation)
   contain the same boolean-key gate read. The sealed file was deliberately
   left untouched. The development twin is unfixed and ships nothing.
   → POST-SHIP.
6. **The 2026-08-14 snapshot cannot be regenerated.** `RESEARCH_RULES.md`
   § Namespaces: each `--data-through` date is an immutable directory,
   never overwritten; a rerun with changed code raises
   `SnapshotConflictError` by design. So the Blocker B fix takes effect on
   the **next** snapshot date, and the shipped `/status/` will keep showing
   `False` for this snapshot — correctly and honestly, in the identifier
   register. This is provenance working as designed, not an outstanding bug.

## UNRESOLVED

Phase 8 items not finished:

- **§15 End-to-end flows — INTERRUPTED MID-CHECK.** Flow A partially run;
  Flows B–H not completed. Two specific loose ends from the partial run:
  - Flow A: the #1 favorable hitter (Pete Crow-Armstrong) rendered **no**
    `/explore/?batter=` link. This is *expected* fixture-limited behaviour
    (Phase 5 emits the link only for hitters in the published catalog; the
    fixture has 53) — but it was **not yet confirmed** that he is absent
    from `explore_fixture/players.json` and that the documented `/demo/`
    fallback route renders instead. Confirm before calling Flow A passed.
  - Flow B: the sort click landed on the **Rank** button, which restores
    official order, so no sorted-view state was produced. Must be re-run
    against a **non-rank** sort (Sample/BBE) to exercise `data-sorted`, the
    "Sorted view — not the official Contact Luck ranking" label, `aria-sort`,
    the polite live region, and amber removal from the rank column.
  - Flows C–H (global search → player; Explore keyboard selection → filters
    → sort → play; demo sliders/outcome/reset; methodology anchors; status
    mobile history; play-not-found → Explore) not started. Note the demo
    simulator flow was already exercised in Phase 7, but not in Phase 8's
    build.
- **§10 Cross-site visual consistency** — not started.
- **§12 ZeroScale invariants** — not started (league_per_100 zero fraction
  shared by leaderboard/player hero; one canonical run_value domain across
  Explore/play; component_per_100 own ticks; trend exception).
- **§18 Console/network cleanliness** — only partially observed. No
  `pageerror` and no console errors seen during the flow attempt, but no
  systematic per-route pass was recorded. Known-expected fixture 404s:
  `/explore/showcase-sensitivity/*.json` (fixture ships zero interactive
  plays) and `/favicon.ico`.
- **§19 Build performance / page weight** — not measured.
- **§20 Production-faithful build** — not attempted. Requires
  `scripts/generate_production_explorer_artifacts.py --data-through
  2026-08-14 --skip-sensitivity` then `dashboard/build.py
  --explore-artifacts-dir outputs/explorer_build/2026-08-14`
  (`ARCHITECTURE.md`). No `play_ledger.parquet` was found locally, so
  whether this can run at all is **unverified**.
- **§21 Production Explorer checks** — blocked on §20.
- **§22 Status production checks** — partially satisfiable: the real
  snapshot set exists locally (10 snapshots, incl. the
  `2026-08-08__refreshed` corrected pair and the intentional 2026-08-07
  gap), but `retrospective_backfill` and `invalid_incomplete_snapshot`
  labels have no local instance and cannot be verified.
- **§23 `make check`** — **NOT RUN since any Phase 8 edit.** This is the
  single most important outstanding verification.
- **§24 Final screenshot set** — not captured.
  Target dir `outputs/figures/phase8_production_candidate_2026-08-29/` does
  not exist yet. (Note the brief's date; today is 2026-08-30.)
- **§25 Anti-vibecode review** — not started.
- **§26 Security/static-site sanity** — partially done via the link crawl
  (no localhost, no external hosts, no absolute filesystem paths seen in
  the crawl), but no explicit secrets/debug-output scan.
- **§27 Repository hygiene** — partially done (`.gitignore`, scratch
  verified not dirtying the tree). Remaining: confirm no screenshots
  tracked, no backup files, no production-blocker TODOs, docs updated.
- **§29/§30 Ship classification and final report** — not produced.

## DO NOT REPEAT

- The Blocker A code trace. It is settled: **RESOLVED — EXPECTED
  BEHAVIOR**, with the mechanism, the constants and the pre-existing
  upstream tests all identified above.
- The Blocker B code trace. It is settled: **RESOLVED — BUG FIXED**
  upstream, using the project's own canonical normalizer.
- The 200-check responsive sweep (§13).
- The 9-state accessibility audit (§14).
- The 634-page link and metadata crawl (§16, §17).
- The 46-route copy audit (§9).
- The stylesheet-wide colour audit (§11).
- The dead-selector audit (§6) — 404 selectors scanned against built HTML,
  templates and JS; 16 candidates triaged, 12 removed, 4 kept for stated
  reasons.
- The play-page JS browser re-verification after the consolidation.
- Any repository reconnaissance, baseline audit, design-direction work, or
  Phase 1–7 reasoning.

## PHASE 8 CHECKPOINT VALIDATED

**2026-08-30, 13:26 EDT (17:26 UTC).**

`make check` → **2191 passed, 8 skipped** (3:45). Exactly the predicted
count: 2180 prior baseline + 11 new Phase 8 tests. The 8 skips are the
pre-existing GNU-date ones in `tests/test_resolve_data_through_date.py`,
unchanged from the Phase 7 baseline.

**One defect found and fixed during validation.** `git diff --check`
reported `dashboard/static/style.css:4727: new blank line at EOF` — trailing
whitespace left behind by the Phase 8 dead-code removal. Stripped to a
single terminating newline; brace balance re-verified at 0; `make check`
re-run afterwards and still **2191 passed, 8 skipped**. `git diff --check`
is now clean, and no merge markers exist anywhere in the diff.

**A gap in the validation premise, corrected.** `make check` does **not**
lint `prospective/`. The Makefile runs `ruff format src tests`,
`ruff check src tests` and `mypy src`, so the one Phase 8 change outside
`dashboard/` — `prospective/prospective_scoring.py` — is covered by *no*
stage of `make check`. It was therefore checked explicitly and separately:

```
ruff format --check prospective/prospective_scoring.py  -> 1 file already formatted
ruff check          prospective/prospective_scoring.py  -> All checks passed!
mypy                prospective/prospective_scoring.py  -> Success: no issues found
python -c "import prospective_scoring"                  -> imports cleanly
```

Anyone re-validating this diff must run those four commands by hand;
`make check` alone does not exercise that file.

**Modified-file set at checkpoint** (10 modified, 2 untracked):

```
 M .gitignore                              M dashboard/static/style.css
 M dashboard/build.py                      M dashboard/templates/base.html
 M dashboard/static/app.js                 M prospective/prospective_scoring.py
 M dashboard/static/play.js                M tests/test_dashboard_explore_build.py
 M dashboard/static/showcase_whatif.js     M tests/test_dashboard_play_page.py
?? tests/test_phase8_production_hardening.py
?? docs/handoffs/phase8-interrupted-run-handoff.md
```

`git diff --stat`: 10 files, +112 / −135. No stray files, no screenshots, no
scratch directories, no backup files.

### Blocker classifications, confirmed at checkpoint

**Blocker A — observed run value: RESOLVED — EXPECTED BEHAVIOR.**
`outcome_class` is Rc (the officially recorded contact result, from Statcast
`events`); `observed_run_value` is Rf (the full accounting including
realized baserunner advancement). A row where they disagree is the system
working. Regression-covered by
`tests/test_phase8_production_hardening.py::TestObservedRunValueIsWholePlay`.

**Blocker B — `component_model_status.outfield`: RESOLVED — BUG FIXED.**
A boolean gate reading (`passes_basic_validation`) was being stringified
into a field documented as carrying the `MODEL_STATUS_*` vocabulary. Fixed
upstream in `prospective/prospective_scoring.py` using the project's own
canonical `normalize_model_status`; no semantics invented. Regression-
covered by `::TestComponentModelStatusVocabulary`.

### Two consequences of Blocker B that must be carried to ship

1. **The immutable 2026-08-14 snapshot still preserves its historical raw
   `"False"` value, and that is correct.** Dated snapshots are never
   overwritten (`RESEARCH_RULES.md` § Namespaces; a changed-code rerun
   raises `SnapshotConflictError` by design). The fix changes what is
   *generated*, never what was already recorded. `/status/` will keep
   displaying `False` verbatim for this snapshot, in the identifier
   register — honest provenance, not an outstanding bug. A dashboard-side
   test pins that display behaviour so it is not "tidied" away later.

2. **The next normally generated snapshot must be checked before ship.**
   The first snapshot produced after this fix should carry
   `component_model_status.outfield = "not_calibrated"` (matching what the
   per-play layer has always recorded) plus a new
   `component_model_status_raw.outfield = "False"`. Confirm both on
   `/status/` and in the snapshot JSON. Until that has been observed once,
   the fix is verified only by unit and structural tests, never end to end.

### Remaining Phase 8 work

1. Visual consistency audit completion (§10)
2. ZeroScale invariant verification (§12)
3. Console/network sweep (§18)
4. Build/page-weight measurement (§19)
5. Production-faithful build investigation (§20)
6. Production Explorer checks where possible (§21)
7. Production Status checks (§22)
8. Final compact screenshots (§24)
9. Anti-vibecode audit (§25)
10. Security/repository hygiene (§26, §27)
11. Final blocker / post-ship / backlog classification (§29)
12. SHIP / DO NOT SHIP report (§30)

Also still open from FINDINGS, to be folded into the final classification:
the ≥768 px leaderboard target sizes (POST-SHIP candidate), the missing
favicon (BACKLOG candidate), the orphaned `play.js` comment at lines 30–34,
and the unfixed development twin of the gate-read in
`run_season_aggregation.py`.

Also still unverified from §15: Flow A's withheld Explore link for a hitter
absent from the fixture catalog, Flow B with a **non-rank** sort, and Flows
C–H.

**This checkpoint is safe to commit.** Full suite green, lint/typecheck
green including the file `make check` does not cover, whitespace clean, no
unrelated files.

---

## EXACT NEXT ACTION

**Superseded by PHASE 8 CHECKPOINT VALIDATED above.** `make check` is green
at 2191/8, the `prospective/` file has been linted and typechecked by hand,
and the tree is clean. This checkpoint is safe to commit.

Resume at §15 Flow B with a **non-rank** sort (Sample/BBE) to exercise
`data-sorted`, the "Sorted view" label, `aria-sort`, the polite live region
and amber removal from the rank column. Then Flow A (confirm the withheld
Explore link plus the `/demo/` fallback), then Flows C–H, then the remaining
numbered work listed in the checkpoint.
