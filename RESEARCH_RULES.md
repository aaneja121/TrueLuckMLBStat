# RESEARCH_RULES.md -- modeling, scoring, and data-safety rules

**Read this file in full before touching anything under `src/mlb_luck_score/`,
`prospective/`, `evaluation/`, `demo/`, `scripts/`, or any notebook.** These rules were
previously the body of `CLAUDE.md`; the text below is unchanged. `CLAUDE.md` is now a
short router that points here so purely presentational work (`dashboard/`) does not have
to load the whole research rulebook — but nothing here is optional for research work.

Read `README.md` for project scope and version history; this file is about *how to work
here safely*, not *what the project is*.

## The single most important rule

> **Never tune, iterate, or select features using 2025 results.**

2025 is the untouched final-test season (see `mlb_luck_score.config.FINAL_TEST_SEASONS`
and `assert_seasons_allowed`). It is enforced in code -- any command that touches a
season list must call `assert_seasons_allowed(...)` and must not pass
`allow_final_evaluation=True` except for a genuine, intentional, one-time final
evaluation that the user has explicitly asked for. Do not add a new code path
(notebook cell, script, ad hoc analysis) that reads 2025 data during normal
development. If you are ever unsure whether something counts as "tuning" on 2025,
treat it as tuning and ask first.

This applies equally to `mlb_luck_score.data.download_development_data` and
`mlb_luck_score.data.clean_development_data` (the full 2021-2024 dataset workflow): both
call `assert_seasons_allowed`, and the downloader has no configured date range for 2025
in `mlb_luck_score.config.MLB_REGULAR_SEASON_DATE_RANGES` as a second layer of
protection. If you ever add a new season to that dict or to `DEVELOPMENT_SEASONS`,
2025 must never be one of them.

### The one narrow exception: the sealed Version 1.0 final evaluation

2025 remains permanently prohibited for development, tuning, iteration, and feature
selection -- nothing below weakens that. It may enter this repository's workflow
**exactly once**, through the dedicated, sealed Version 1.0 final-evaluation entry point
(`evaluation/run_v1_final_evaluation.py`), for the single genuine final evaluation the
user has explicitly asked for. Once that run completes and is sealed, its 2025 output is
evaluation data, not development data -- see the Version 1.0 post-evaluation sealing rule
in that module's docstring for what a permitted defect-fix rerun looks like versus a
forbidden post-hoc retune. This exception is intentionally narrow and operational, not a
general loosening:

1. **Never** add 2025 to `MLB_REGULAR_SEASON_DATE_RANGES`, `DEVELOPMENT_SEASONS`, any
   development-data downloader (`download_development_data`, `clean_development_data`,
   or any future equivalent), or any general-purpose CLI flag. These stay exactly as
   documented above, unconditionally.
2. The frozen 2025 regular-season date range is defined **locally**, inside the Version
   1.0 evaluation code, under an explicit name such as `FINAL_EVALUATION_2025_DATE_RANGE`
   -- never in `mlb_luck_score.config` and never re-exported through normal development
   configuration.
3. The dedicated entry point may call `download_statcast_range` (or an equivalent
   low-level fetch) directly for that range, but only after it has itself verified, in
   order: a clean working tree, a frozen commit hash recorded, a pre-evaluation manifest
   built and validated, every frozen artifact hash verified against that manifest, no
   prior sealed final evaluation already exists, and explicit final-evaluation
   authorization is active for that run.
4. `assert_seasons_allowed(..., allow_final_evaluation=True)` remains the enforcement
   point, but that authorization must be reachable **only** from the dedicated Version
   1.0 orchestration path -- no development runner, notebook, or general script may set
   `allow_final_evaluation=True`.
5. 2025 raw data and every downstream artifact from this run live in a namespace
   separate from development caches (e.g. `data/final_evaluation/2025/`,
   `outputs/final_evaluation/v1/`, `artifacts/final_evaluation/v1/`) -- never mixed into
   `data/raw`, `data/interim`, `data/processed`, or `artifacts/` where an ordinary
   development runner could discover it by accident.
6. This ingestion must record provenance: the exact start/end dates fetched, the data
   source, a retrieval timestamp, hashes of the raw files, row counts, date coverage, and
   any missing dates -- the same evidentiary bar as the park-geometry/weather provenance
   rules elsewhere in this file.
7. Fail-fast guards (see `tests/` for the Version 1.0 evaluation-guard suite) must prove:
   development downloaders still reject 2025; development runners cannot locate or
   process the final-evaluation cache; only the dedicated entry point can authorize 2025;
   the final evaluation cannot be re-run after sealing without the explicit
   defect-fix-archival path; and building or testing the orchestration code itself never
   triggers real 2025 access (synthetic fixtures only, exactly as for every other module).
8. The complete evaluation protocol (manifest, system checks, distribution-shift
   analysis, report assembly, sealing) must be built and tested against synthetic data,
   then committed with a clean working tree, before any real 2025 download occurs. That
   committed commit hash is what the manifest records as the pre-evaluation commit.

If you are ever asked to extend this exception -- to run a second final evaluation, to
loosen any guard above, or to make 2025 reachable from a second code path -- treat that
as a new, separate decision requiring the user's explicit sign-off, not a natural
extension of this one.

### A second sealed 2025 evaluation IS authorized -- for the pitcher replication ONLY

Version 0.14 (`replication/`) pre-registers a one-time held-out 2025 replication of the
pitcher findings: the questions, the estimators, and the classification rule were frozen
and hashed before any 2025 data was opened. **Freezing the specification was not the
sign-off.** On 2026-09-08 the maintainer gave that sign-off explicitly, as the separate
decision the paragraph above requires. It is recorded in
`replication/pitcher_replication_authorization.py`.

The authorization is deliberately narrow:

- **Scope.** "One-time held-out 2025 full-season replication of Pitcher Contact Luck
  under the frozen Version 0.14 specification." Exactly one evaluation, for the
  preregistered questions only.
- **Bound by hash.** It names the authorized `freeze_content_hash` AND
  `spec_content_hash`. `resolve_authorization` refuses to apply it to any other freeze,
  so amending the specification silently voids it and a new sign-off is required.
- **One-time.** Enforced by `assert_no_replication_outputs_exist`: once the replication
  namespace holds output, readiness fails and a rerun is refused.
- **Recorded after the freeze, on purpose.** The frozen spec's
  `AUTHORIZATION_STATUS["second_sealed_2025_evaluation_authorized"]` still reads False
  because it records the state at freeze time. Do not "fix" it -- that flag is the
  evidence the questions preceded the sign-off, and editing it would invalidate the
  freeze.

It does NOT authorize: changing the metric, denominator, interval estimator,
resolving-power or split-half methodology, the role/display rules, or the primary and
secondary quantities; adding metrics after seeing 2025; selecting subsets; retuning
thresholds; redesigning the pitcher UI before the result is reported and frozen; reading
2026; touching Contact Forecast; or deploying. The full withheld list is
`AUTHORIZATION_EXCLUSIONS`.

**No other research line may read 2025 on the strength of this.** Neither this
authorization nor the Version 1.0 final evaluation extends to any other use; each needs
its own explicit sign-off, asked for in the moment.

Everything in the numbered list above (isolated namespaces, a locally-defined date range,
a dedicated entry point, provenance recording, fail-fast guards, synthetic-data testing
before real access, a clean committed tree) applies to this run unchanged. `replication/`
still may not pass `allow_final_evaluation=True` from anywhere but its own dedicated
entry point, and `assert_ready_for_2025` still requires a clean tree, a non-provisional
freeze, intact 2025 protection, an empty replication namespace, and a validating freeze.

### Public launch of the 2024 pitcher surface IS authorized -- for 2024 ONLY

The pitcher replication authorization above explicitly withholds "deploying anything".
That withholding is about the replication: reaching a `REPLICATED` classification does not
by itself license publishing anything. **Publishing is a separate maintainer decision**,
and on 2026-09-09 the maintainer took it, for the 2024 pitcher surface and nothing else.

What is authorized:

- Publishing `/pitchers/<season>/` and `/pitchers/<season>/<pitcher_id>/` for the **2024**
  season, from the committed development fixture
  (`dashboard/pitcher_season_fixture.json`), through the ordinary
  `scripts/publish_snapshot.sh` path.

Applied in exactly one place: **`dashboard_config.PITCHER_PUBLIC_SEASONS`**. That tuple is
the gate. `dashboard/build.py` refuses -- hard, with no site produced -- to render any
pitcher season absent from it, so a season cannot reach the public site because a fixture
for it exists, because a flag was passed twice, or because someone assumed.

What this does NOT authorize, each needing its own explicit sign-off asked for in the
moment:

1. **Publishing 2025.** It remains sealed final-evaluation data. Its replication output is
   question-level agreement, not a per-pitcher season table, so publishing 2025 would mean
   a *new computation on sealed data* -- a second sealed-2025 use, which the paragraphs
   above already say requires its own decision.
2. **Publishing 2021-2023.** They are `TRAIN_SEASONS`. The contact model is fitted on them,
   so a board over them would present in-sample fitted values as measurement. This is a
   scientific objection, not a governance one, and it does not go away with a sign-off.
3. **Publishing 2026.** Prospective and unopened for this line.
4. Anything else the replication authorization already withholds. Widening
   `PITCHER_PUBLIC_SEASONS` is a governance edit, not a configuration change, and the
   reason for each season added belongs here before the tuple changes.

`replication/pitcher_replication_authorization.py` is **not** amended by this. Its
exclusion list is a statement of fact at authorization time, and editing it to say
deploying is now fine would destroy the evidence of what was authorized when.

## Version 1.1: prospective 2026 scoring

Version 1.1 (`prospective/run_v1_1_2026_scoring.py`) applies the exact frozen Version
1.0 system -- contact model, outfield/infield opportunity models, advancement model,
attribution ledger, confidence framework, qualification thresholds, and public-score
contract -- to 2026 season-to-date data, producing immutable dated snapshots. It is
**scoring-only**: nothing in Versions 0.2-1.0 is modified, retrained, recalibrated, or
retuned by this version, and none of the season-list/model-selection guards those
versions rely on are loosened by anything below.

1. **2025 remains sealed final-evaluation data.** It cannot enter development, and it
   cannot enter Version 1.1 either -- the prospective runner may perform ONE optional,
   read-only integrity check that the Version 1.0 seal (`artifacts/final_evaluation/
   v1/seal.json`) still exists and parses as a well-formed `EvaluationSeal`
   (`prospective.prospective_manifest.verify_v1_seal_unchanged`), and must never read
   2025 data as a model input. That check is non-fatal by design -- 2026 scoring does
   not depend on 2025 data at all, so a seal problem is a finding for a human reviewer,
   never a reason to block a 2026 snapshot.
2. **2026 is prospective scoring data**, not development data. See `mlb_luck_score.
   config.PROSPECTIVE_SEASONS`. It stays out of `MLB_REGULAR_SEASON_DATE_RANGES` and
   `DEVELOPMENT_SEASONS` -- exactly like 2025's omission, this is a second layer of
   defense (alongside namespace isolation) against a development runner accidentally
   discovering or training on 2026 rows.
3. **Version 1.1 may score new 2026 observations but may never tune against them.** No
   code path in `prospective/` may retrain, recalibrate, reselect a feature, or
   re-choose a qualification/calibration threshold based on what a 2026 result looks
   like. The four component models are retrained FRESH each snapshot run, but only on
   `TRAIN_SEASONS` (2021-2023, unchanged) -- identical to how Version 1.0 trained for
   the 2025 final evaluation (`prospective.prospective_scoring.train_and_score_2026`
   mirrors `evaluation.run_v1_final_evaluation.train_and_score_2025` exactly). This
   codebase does not persist trained model objects between runs (see `evaluation.
   v1_final_evaluation_manifest`'s "What 'frozen artifact' means here"), so "frozen
   model" means the frozen SOURCE CODE and frozen `TRAIN_SEASONS` data, re-run
   deterministically (fixed seeds) each time -- never a literal cached model file.
4. **Any future model change requires**, before it may touch a real 2026 observation:
   a separately named development version (e.g. Version 1.2), development using
   2021-2024 only (2025 and 2026 both stay untouched during that development), and an
   explicit freeze date recorded before evaluating any 2026 observation made after that
   date. A change developed AFTER 2026 data has already been produced by Version 1.1
   must not be tuned against, or selected using, any 2026 result already observed.
5. **Previously observed 2026 data cannot later be presented as untouched validation
   data.** Once a prospective snapshot has scored a given `--data-through` date, that
   slice of 2026 is no longer a pristine, never-analyzed season for any future
   development decision -- treat it the same way CLAUDE.md's "Version 0.7 is now
   FROZEN" rule treats repeatedly-inspected 2024 near-wall results: informative for
   monitoring, but not evidence for a NEW model/feature/threshold decision without
   explicit maintainer sign-off that the season is being treated as non-pristine for
   that specific change.

### Namespaces and the dedicated entry point

Every 2026 raw/derived file lives under `data/prospective/2026/`, `outputs/
prospective/v1_1/`, and `artifacts/prospective/v1_1/` (`prospective.prospective_
config`) -- never `data/raw`, `data/processed`, `outputs/tables`, or `artifacts/`, so no
development runner can discover 2026 data by accident. The prospective runner refuses
(`SealedNamespaceAccessError`) any path resolving inside `data/final_evaluation/2025`,
`outputs/final_evaluation/v1`, or `artifacts/final_evaluation/v1`.

`prospective/run_v1_1_2026_scoring.py` exposes exactly three flags: `--data-through`
(required), `--snapshot-label` (optional), and `--force-redownload` (refreshes only the
shared, mutable 2026 raw cache before building a NEW snapshot -- never touches an
already-completed snapshot directory). It deliberately exposes NO model-selection,
calibration, feature-selection, or threshold-tuning flag; the qualification threshold
set is hardcoded to the same `"primary"` default Version 0.12 uses.

Unlike Version 1.0's one-time sealed evaluation, this tool is meant to run repeatedly
across a season, so it uses a different immutability mechanism instead of a seal
-and-defect-fix-acknowledgement ceremony: each `--data-through` date gets its own
immutable snapshot directory (never overwritten); a rerun with identical inputs and
code is accepted as a deterministic no-op (`SnapshotManifest.
deterministic_content_hash()`, which deliberately excludes wall-clock fields); a rerun
that differs raises `SnapshotConflictError` rather than silently overwriting. This
holds regardless of `--force-redownload` -- that flag only refreshes the shared,
mutable raw cache under `data/prospective/2026/` before a NEW snapshot is built; it has
no code path that can write into an already-completed snapshot's `outputs/`/`artifacts/`
directory (see `tests/test_prospective_force_redownload_immutability.py`).

**A dirty working tree BLOCKS every real snapshot run** (`prospective.
prospective_ingestion.run_prospective_guards`, reusing Version 1.0's `assert_clean_
working_tree` unchanged, raising `WorkingTreeNotCleanError`) -- this is a hard block,
not merely a recorded flag. A snapshot generated from uncommitted code could differ
from the commit its own manifest names, making it impossible to reproduce exactly. This
covers BOTH a dirty tracked file (staged or unstaged) AND an untracked-but-not-ignored
file (a stray new `.py`/config/test file counts) -- `git status --porcelain` treats
both as dirty, and reports neither for a properly gitignored path. Files under
`data/prospective/2026/`, `outputs/prospective/v1_1/`, and `artifacts/prospective/v1_1/`
are gitignored (see `.gitignore`'s "Version 1.1 prospective 2026 scoring namespace"
section) specifically so a prior snapshot's own output files can never falsely dirty
the tree for a later run. (`SnapshotManifest.working_tree_clean`/`prospective_manifest.
is_working_tree_clean` remain a separate, non-raising low-level recorder -- the actual
enforcement is in `run_prospective_guards`, one layer up; see `tests/
test_prospective_working_tree_guard.py`.)

**A `--data-through` date with any game not yet final is refused**
(`prospective.prospective_ingestion.assert_data_through_date_is_complete`, checked via
the MLB Stats API `/schedule` endpoint's per-game status before ingestion). A postponed
or cancelled game never happened, so it is excluded from the "must be final"
requirement (recorded separately for provenance, never silently dropped). A SUSPENDED
game is the one case that needed an explicit rule: it has real partial play but no
final, reconciled outcome, so it is deliberately treated exactly like an in-progress
game -- it blocks the date, exactly like "In Progress"/"Warmup"/"Scheduled"/any
unrecognized status (fail-safe: an unrecognized status is never assumed final). This
guard FAILS rather than silently walking backward to "the preceding fully completed
date" -- the caller must explicitly choose an earlier `--data-through` and rerun; see
`tests/test_prospective_date_completeness_guard.py`.

**A `--data-through` date that completed NO games is refused**
(`prospective.prospective_ingestion.assert_data_through_date_has_completed_games`,
raising `NoCompletedGamesToScoreError`). This is the guard that knows a season can
END. Nothing else in the pipeline does: `check_data_through_date_completeness` treats
a date with zero scheduled games as trivially complete (correctly -- there is nothing
unfinished about a day with no baseball), both schedule fetches filter `gameType="R"`
so the postseason is invisible here, and the coverage contract cannot report a missing
completed date on a date that completed none. Without this guard every day after the
final regular-season game re-scored byte-identical data into a NEW snapshot directory
and a NEW write-once archive key -- one duplicate per day, indefinitely -- and, with
scheduled deploys enabled, advanced the public `data_through_date` onto days no
baseball was played. A fully postponed slate is refused for the same reason: a
postponed game never happened, so that date carries no new play either.

It takes the result `assert_data_through_date_is_complete` already returned rather
than re-fetching, so it costs no extra network request, and it runs BEFORE ingestion
so the off-season does not re-download a full season every morning to discover there
is nothing to do. Like its sibling it FAILS rather than silently walking backward to
the last date that did have games. `run_v1_1_2026_scoring.main` maps it to exit code
**3**, distinct from the generic failure code 2, and `scripts/publish_snapshot.sh`
treats exit 3 as a clean stop (exit 0, nothing archived/built/deployed) so the daily
scheduled loop does not report a failure every morning of the off-season. See
`tests/test_prospective_no_completed_games_guard.py` and
`tests/test_publish_snapshot_orchestration.py::TestNothingToScoreIsACleanStop`.

`DataThroughDateCompletenessResult.completed_games_on_date` is a derived PROPERTY, not
a dataclass field, on purpose: `to_dict()` is `asdict()` and flows into every snapshot
manifest's `schema_checks`, which are content-hashed to tell an idempotent rerun from a
conflict. A new field there would change the recorded shape of every already-archived
snapshot and turn a legitimate rerun of an archived date into a `SnapshotConflictError`.
Do not promote it to a field.

**The verified season-END date is a CROSS-CHECK, not a cutoff.**
`PROSPECTIVE_2026_SEASON_END_DATE = date(2026, 9, 27)` records the regular-season
finale (Baltimore Orioles at New York Yankees), under the same rule as the opening
date: a maintainer-provided citation of the official MLB schedule, with `_SOURCE` and
`_VERIFIED_AT` updated together and never one without the others. It was corroborated
against the MLB Stats API `/schedule` endpoint, which returned no `gameType=R` games
after 2026-09-27 through 2026-11-15 -- corroboration is NOT the citation, and this
repository's tooling still never fetches or guesses a schedule of its own accord.

`assert_data_through_date_agrees_with_recorded_season_end` raises
`RecordedSeasonEndDateStaleError` ONLY when real games completed after that date. It is
deliberately not a cutoff, and the reason is the whole point: a rainout makeup played
after the finale is a genuine regular-season game whose plays belong in the season
totals, and a blind cutoff would drop it silently and permanently. Dates that completed
no games are none of this guard's business -- the ordinary off-season case belongs to
the schedule-derived guard above and its quiet exit 3. So the constant can only ever
cause a LOUD failure (exit 2) saying the recorded fact disagrees with observed play,
never a silent skip. Do not "simplify" it into a date cutoff.

The real 2026 season-opening date is VERIFIED:
`prospective.prospective_config.PROSPECTIVE_2026_SEASON_START_DATE = date(2026, 3, 25)`,
`PROSPECTIVE_2026_SEASON_START_VERIFIED = True`. Source: MLB's official 2026
championship-season schedule -- the season began Wednesday, March 25, 2026, with
Opening Night (New York Yankees at San Francisco Giants); the official schedule
confirms that game was played. This was confirmed via an explicit maintainer-provided
citation of the official MLB schedule (verified-at date recorded in
`PROSPECTIVE_2026_SEASON_START_VERIFIED_AT`, alongside the source citation in
`PROSPECTIVE_2026_SEASON_START_SOURCE`) -- Claude Code did not independently fetch or
cross-check a live schedule source for this date (this repository's tooling never
generates or guesses a schedule URL on its own). If this date is ever wrong or needs
revision for a future season, update the date, the source citation, and the
verification date together -- never change one without the others, following the exact
precedent `FINAL_EVALUATION_2025_DATE_RANGE` sets for the Tokyo Series exception.

### Player name resolution is presentation-only

`prospective.prospective_player_names.apply_player_name_overlay` fills the `batter_name`
column the frozen public-score schema already defines but every earlier version leaves
null (via `mlb_luck_score.data.download_player_names.fetch_player_names`, the public MLB
Stats API `/people` endpoint). It runs strictly AFTER `build_public_score_table`/
`assign_official_ranks` and is asserted, at runtime, to change no column other than
`batter_name` -- it never becomes a model feature, never affects a score, rank, or
qualification status, and an unresolved name stays null with a reason code recorded in a
separate `name_resolution_report.json` sidecar, never a new column on the frozen schema.

**v1.1.1 presentation patch**: the real 2026-08-05 snapshot (immutable, never modified by
this patch) shipped with `favorable_leaderboard.json`/`unfavorable_leaderboard.json`
showing `batter_name = null` for every row even though `public_score.json` had resolved
names -- the leaderboards were sliced off `public_score_table` BEFORE the name overlay
ran, so they never picked up the later-applied names. Purely a presentation bug: scores,
ranks, intervals, and qualification status were never affected. Fixed in
`prospective/run_v1_1_2026_scoring.py` by moving the overlay before the leaderboard
slicing, so `public_score.*` and both leaderboard exports are always sliced from the SAME
already-overlaid table. See `tests/test_prospective_leaderboard_name_propagation.py`.

**v1.1.2 operational correctness fix: coverage-aware raw Statcast caching.** The
2026-08-06 snapshot and the first 2026-08-08 attempt (both immutable, never modified by
this patch) silently scored STALE data: `data/prospective/2026/statcast_2026_regular_
season.parquet` was downloaded once on 2026-08-06 and reused for every later run because
the old guard (`ingest_2026_raw_statcast`) only checked whether the raw cache file
EXISTED, never whether its coverage actually reached the requested `--data-through` date.
The fix -- do NOT reintroduce a "does the file exist" check anywhere in this pipeline
without pairing it with the coverage check below:

- Every raw Statcast cache write now also writes a provenance sidecar
  (`prospective.prospective_ingestion.RawStatcastCacheProvenance`, persisted next to the
  parquet file as `*.provenance.json`): `requested_start_date`, `requested_end_date`,
  `observed_min_game_date`, `observed_max_game_date`, `observed_game_dates` (the FULL set,
  not just min/max -- see below for why), `retrieved_at`, `row_count`, `sha256`.
- `evaluate_raw_statcast_cache` is the single decision point for reuse vs. refresh. It
  refuses to reuse the cache (refreshing instead) if: the raw file or its provenance
  sidecar is missing; the provenance sidecar is malformed; the sidecar's recorded sha256
  no longer matches the actual raw file; or ANY completed MLB game date (checked via a
  fresh MLB Stats API `/schedule` range fetch, `fetch_schedule_game_statuses_range`) from
  `requested_start_date` through `requested_end_date` is absent from the cache's own
  `observed_game_dates`. This is deliberately NOT `max(game_date)` comparison alone --
  comparing only the max/tail date is exactly what let the real incident happen: a later
  date being present can hide an earlier internal gap. A date with zero completed games
  (an off day, the All-Star break, or a postponed/cancelled game) is never treated as
  missing. `--force-redownload` still forces a refresh unconditionally, but per the task
  that produced this fix, it is NOT required for ordinary forward-moving `--data-through`
  requests -- the cache now refreshes itself automatically when its verified coverage
  doesn't reach the request. Cache AGE is never consulted for validity, only coverage.
- `assert_scoring_dataset_satisfies_coverage_contract` adds a SECOND, fully independent
  fail-fast check immediately before scoring (in `run_v1_1_2026_scoring.run_prospective_
  snapshot`, right before `train_and_score_2026`), re-derived directly from the actual
  `scoring_df` with its own fresh schedule fetch. It never reads or trusts the earlier
  `CacheCoverageValidation` result -- this is intentional so a bug (or an incorrectly
  mocked cache decision, e.g. in a test) upstream can never silently let stale data reach
  the model. If you ever add another data source with similar cache-then-reuse semantics,
  apply the SAME two-layer pattern (a coverage-provenance-based reuse decision, PLUS an
  independent final assertion on the actual data about to be used) -- do not go back to
  "does the file exist."
- Every snapshot manifest now records: whether the raw Statcast cache was reused or
  refreshed, its cached provenance before that decision, the final raw-data provenance,
  the full coverage-validation result (including any missing completed game dates found),
  and the final pre-scoring coverage check's own result (`coverage_and_schema_report` in
  `run_v1_1_2026_scoring.py`, which flows into the manifest's `schema_checks`).

See `tests/test_prospective_statcast_cache_coverage.py` for the regression tests.

## Avoid target leakage

Never use `events`, `outcome_class`, `description`, `estimated_ba_using_speedangle`,
`estimated_woba_using_speedangle`, `woba_value`, `delta_home_win_exp`, or any other
column computed from or encoding the play's result as a model input feature. The
enforced list lives in `mlb_luck_score.config.LEAKAGE_COLUMNS`; the check is
`mlb_luck_score.features.build_contact_features.assert_no_leakage`. If you add a new
feature, check whether it's post-outcome before wiring it in, and prefer running the
leakage check over trusting your own judgment.

## Keep eligibility logic centralized

All fair-batted-ball eligibility rules, outcome-class mapping, and training-exclusion
reasons live in `mlb_luck_score/eligibility.py` and nowhere else. Do not duplicate the
eligible-event list, the outcome mapping, or the ambiguous-event handling in another
module, script, or notebook -- import from `eligibility.py`. If the rules need to
change, change them there and update its docstrings and `README.md`/this file together.

## Never silently redefine the Luck Score

The current (Version 0.2) default is `compute_raw_contact_luck_runs`
(`mlb_luck_score.scoring.contact_luck`, using the fixed run-value table in
`mlb_luck_score.scoring.run_values.DEFAULT_RUN_VALUE_MAP`) and
`compute_empirical_public_score` (`mlb_luck_score.scoring.empirical_score`, using a
`ReferenceScoreArtifact` built by `mlb_luck_score.models.build_reference_score`). The
Version 0.1 ordinal functions (`mlb_luck_score.scoring.raw_luck.compute_raw_luck`,
`mlb_luck_score.scoring.public_score.raw_luck_to_public_score`) are LEGACY -- kept only
for backward compatibility and explicit Version-0.1-vs-0.2 comparison, never as the
default for new work. All of these have exact, documented formulas. Do not change a
formula, the default run-value/value map, or the class ordering without: (1) updating
every docstring that states the formula, (2) updating the corresponding tests
(`tests/test_contact_luck.py`, `tests/test_run_values.py`, `tests/test_empirical_score.py`,
`tests/test_scoring.py`), and (3) calling out the change clearly to the user as a
redefinition, not a bug fix. Silent redefinition breaks comparability across any results
already produced. If you ever add a new run-value source (e.g. a different season range
or a context-aware model), version it explicitly (`scoring_version` in
`ReferenceScoreArtifact`) rather than overwriting `DEFAULT_RUN_VALUE_MAP` in place.

## Never use class_weight="balanced" (or similar) for the probability baseline

`train_model`'s default `class_weight=None` is load-bearing, not arbitrary. It was
changed FROM `class_weight="balanced"` after that setting was confirmed (via a controlled
comparison on real 2021-2024 data, isolating `class_weight` as the only variable changed)
to cause severe probability miscalibration -- e.g. rows called ~54% likely to be a triple
were observed to be one ~3.6% of the time. See "Model comparison and probability
calibration" in README.md and `mlb_luck_score.models.compare_models` for the exact numbers
and methodology. `class_weight="balanced"` is preserved ONLY as the explicitly-labeled
`VARIANT_CLASS_BALANCED` comparison model (`mlb_luck_score.models.train_contact_model`) --
never use its output, or any other class-reweighting/oversampling scheme, to produce
probabilities for the Contact Luck score, no matter how good its accuracy or per-class
recall looks. Recall/accuracy and calibration are different questions -- a model can
excel at one while failing the other. If you ever change the default `class_weight` or
add a new reweighting scheme, verify calibration with
`mlb_luck_score.models.compare_models.run_comparison` first and report the actual ECE/log
loss numbers, not just accuracy.

## Do not compute or publish Luck Scores against real data with a miscalibrated model

`mlb_luck_score.scoring.raw_luck` requires well-calibrated probabilities to mean anything
(`expected_value = sum(p(outcome) * value(outcome))`) -- a model that ranks outcomes
correctly but assigns systematically wrong probabilities produces biased raw-luck values
even though its classification metrics look fine. Before computing real (non-synthetic)
raw-luck or public-score values, check the current model's calibration via `make
compare-models` / notebook 03 and report the actual numbers. If a change makes
calibration meaningfully worse, do not fold it into the default without calling that out
explicitly.

## Never silently adopt a candidate model variant as the default

`park_aware_v03_candidate` (`mlb_luck_score.models.compare_park_aware`, adds `venue_id` to
the Version 0.2 `baseline_v02` features) is a documented CANDIDATE, not the default --
`train_model`'s default feature set does not include `venue_id`. The same is true of the
Version 0.4 park-geometry candidates (`geometry_only_v04_candidate` and
`venue_plus_geometry_v04_candidate`, `mlb_luck_score.models.compare_geometry_aware`):
despite a large, bootstrap-confirmed log-loss and near-wall-calibration improvement on real
2021-2024 data, `recommend_geometry_adoption` reports `recommend_adopt_any_v04_candidate:
False` because a reliably-sampled venue (loanDepot park) shows a material calibration
regression -- see "Park geometry (Version 0.4)" in README.md for the full numbers. The
Version 0.5 weather candidates (`weather_basic_v05_candidate` and `weather_vector_v05_
candidate`, `mlb_luck_score.models.compare_weather_aware`) are an even sharper illustration of
why: they pass ALL 8 automatable adoption criteria (`recommend_adopt_any_v05_candidate: True`)
with a real, bootstrap-confirmed (if tiny) log-loss improvement, but a direct physical
-plausibility check -- the one criterion the rule deliberately never automates -- found the
per-play weather attribution weak and partly wrong-signed (see "Weather and air density
(Version 0.5)" in README.md), so neither has been adopted either. Version 0.5.1
(`mlb_luck_score.models.compare_weather_variants`) went further and made the physical
-plausibility check itself a checkable, automated gate (controlled-perturbation directional
checks, see `mlb_luck_score.models.weather_perturbation`) combined via AND with the usual
statistical criteria -- `density_only_v051_candidate` and `density_anomaly_v051_candidate`
don't even clear bootstrap significance on real 2024 data, and `components_only_v051_candidate`
does but fails the perturbation checks (backwards density AND wind direction) -- see "Weather
correction (Version 0.5.1)" in README.md. `recommend_adopt_any_v051_candidate: False`;
`baseline_v02` remains the default. Version 0.6 (`mlb_luck_score.models.
compare_alignment_aware`) confirms the same pattern for defensive alignment: `alignment_
interactions_v06` shows a real, bootstrap-confirmed aggregate improvement, yet fails
adoption on both a material `bb_type_ground_ball` subgroup regression and a
backwards-signed controlled-perturbation check (see "Watch for confounding-by-indication"
above and "Alignment-aware positioning (Version 0.6)" in README.md) --
`recommend_adopt_any_v06_candidate: False`; `baseline_v02` remains the default. This
pattern generalizes: any future comparison variant (more park factors, exact defender
positioning/execution, etc.) stays a candidate, reported with its exact metrics via
`recommend_*`-style rule-based logic, until a maintainer explicitly decides to adopt it. A
rule-based recommendation is a starting point for judgment (read the
full by-venue/by-subgroup table yourself -- a real, noteworthy regression can exist below a
conservative automated threshold, and criteria that are inherently a judgment call, like
"is the model learning physically plausible effects", are deliberately NOT automated at
all), never a substitute for it. If a candidate is adopted, give it a distinct
`scoring_version` in its `ReferenceScoreArtifact` (see `mlb_luck_score.models.
build_reference_score`) rather than overwriting an earlier version's artifact file.

## Never commit datasets, secrets, virtual environments, or model artifacts

`.gitignore` already excludes `.venv/`, `.env`, `data/raw/*`, `data/interim/*`,
`data/processed/*`, `artifacts/*`, and `outputs/{figures,tables}/*` (keeping only
`.gitkeep` placeholders). Before committing, run `git status` and double-check nothing
under those paths, and nothing that looks like an API key or credential, is staged.
`mlb_luck_score/data/game_metadata_overrides.py` (like `mlb_luck_score/scoring/
run_values.py` and `mlb_luck_score/data/park_geometry.py`) is a deliberate exception: it is
small, hand-reviewed reference data written as source code, not a downloaded dataset, so it
IS tracked in git. Every entry in it must cite a documented, verifiable `source_note` --
never add or edit an entry without one.

## Park-geometry reference data must document provenance and review status

Every record in `mlb_luck_score.data.park_geometry.PARK_GEOMETRY_POINTS` must have a
non-empty `source_name`/`source_reference`/`source_accessed_date` (enforced at import time
by `validate_geometry_points`) and an honest `review_status`. Every record added so far by
an AI coding agent uses `REVIEW_STATUS_AGENT_SOURCED` ("agent_sourced_pending_human_
review") -- do not upgrade a record's `review_status` to imply human review has happened
unless a human maintainer actually did it. Never fabricate a wall distance, height, or
configuration date -- if a source is ambiguous or conflicting (e.g. a park's exact pre-
renovation dimensions can't be pinned down), document the ambiguity in `notes` and either
use the best-supported figure with a caveat or leave the venue/era without geometry
(`geometry_status` will correctly report it as unavailable downstream -- see
`mlb_luck_score.data.join_park_geometry`) rather than guessing. Never add geometry for a
venue in `TEMPORARY_OR_SPECIAL_VENUE_IDS` (temporary/neutral-site venues, including the
Field of Dreams sentinel `venue_id=-1`) without the same level of verified, cited evidence
required for any other venue -- the current default is to leave them without geometry, and
that is a deliberate choice, not a gap to casually fill in.

## Never fabricate weather, roof status, or indoor climate conditions

`mlb_luck_score.data.venue_environment.VENUE_ENVIRONMENTS` (station/elevation/timezone
reference data) follows the same provenance rules as park geometry above -- every record
needs a `source_note`, and weather-station coordinates/elevation must be independently
verified against the actual data source (see that module's docstring), not guessed.
Historical weather itself (`mlb_luck_score.data.download_historical_weather`) comes from
exactly two sources -- the MLB Stats API's own per-game `weather` field and the public Iowa
Environmental Mesonet ASOS archive -- and nothing else; never substitute current/live
weather for a historical game, and never invent an hourly observation to fill a gap.
`mlb_luck_score.data.build_game_weather.classify_roof_status` never infers a closed roof
from precipitation or any other condition text -- only an explicit `"Roof Closed"` string
does that; everything else defaults to `roof_status_unknown` rather than a guess. For
`retractable_roof_closed`/`fixed_indoor`/`roof_status_unknown` games, `effective_
temperature_c`/`effective_relative_humidity_pct`/`effective_pressure_hpa`/
`air_density_kg_m3` MUST stay null (no reviewed indoor-climate assumption exists in this
repository) -- only `effective_wind_speed_mps=0.0` is set, because a closed roof physically
blocking outdoor wind is a certainty, not an assumption. Do not add an indoor-climate
default (e.g. "72F, 50% humidity") without first adding genuinely reviewed, cited
per-venue indoor climate-control data -- until then, leave it unavailable.

## Weather attribution must stay separate from raw contact luck

`mlb_luck_score.scoring.weather_attribution.compute_weather_attribution`
(`weather_run_value_effect = expected_run_value_actual_environment -
expected_run_value_standard_environment`) is a DIFFERENT quantity from `mlb_luck_score.
scoring.contact_luck.compute_raw_contact_luck_runs` (`raw_contact_luck_runs = actual_run_
value - expected_run_value`, using the selected baseline model's own prediction). Never add
them together -- see `weather_attribution`'s module docstring for exactly why that would
double-count weather. Building and testing the weather-attribution module does not itself
constitute using it for real scores; per the task's "only after a weather-aware candidate
passes validation" rule, treat its output as informational infrastructure, not a validated
headline number, until a maintainer has actually adopted a weather-aware candidate.

## When generating counterfactual/standardized feature rows, only use categories the model has seen

`mlb_luck_score.features.build_contact_features.generate_standardized_environment_rows` was
verified to produce physically backwards predictions (Coors Field's thin actual air scoring
WORSE than a denser standardized reference) when an earlier version set
`weather_match_quality` to a synthetic `"standardized"` label the model never saw in
training -- `OneHotEncoder(handle_unknown="ignore")` silently encodes any unseen category as
all zeros, a pattern the fitted model never learned to interpret, rather than raising an
error. Any function that builds a counterfactual/synthetic feature row for prediction must
only assign categorical values that genuinely occur in real training data (see that
function's fix and `tests/test_compare_weather_aware.py::
test_standardized_environment_categorical_overrides_are_realistic_categories` for the
regression test). This class of bug is easy to miss because it fails silently -- no
exception, no NaN, just a quietly wrong prediction -- so treat any new synthetic/
counterfactual row generator with the same suspicion and verify its categorical overrides
against real data before trusting its output.

## Never adopt a model on a statistically significant log-loss delta alone

`mlb_luck_score.models.compare_weather_variants.recommend_variant_adoption` (Version 0.5.1)
requires every candidate to pass CONTROLLED-PERTURBATION directional checks
(`mlb_luck_score.models.weather_perturbation`) in addition to the usual log-loss/bootstrap/ECE/
venue-regression checks, combined with AND. This exists because Version 0.5 found a real,
bootstrap-significant log-loss improvement whose per-play attribution was physically backwards
-- large real-world sample sizes (n=122,132 here) make it easy for a tiny, physically-meaningless
effect to clear a bootstrap-significance bar. When adding a new model candidate/adoption rule,
do not treat "the confidence interval excludes zero" as sufficient justification by itself --
prefer an explicit, checkable physical/behavioral test (a monotonicity check, a known-effect
sanity check like Coors Field, a controlled perturbation) alongside the statistical one, exactly
as `weather_perturbation.check_directional_effect` does.

## Any "fit on train, apply to all" statistic must never see validation data

`mlb_luck_score.data.join_weather_features.compute_venue_air_density_baseline` (each venue's
"normal" air density, used by `density_anomaly_v051_candidate`) is computed ONLY from
`TRAIN_SEASONS` rows, then applied unchanged to both training and validation rows -- exactly
like fitting a `StandardScaler` on train and transforming both splits with it. Recomputing this
kind of per-group baseline from the full dataset (including validation seasons) would leak
validation-season information into the "baseline" itself, silently making the validation
evaluation optimistic. Any future per-venue/per-group reference statistic (a mean, a baseline,
a normalization constant) must follow this same pattern -- fit once on `TRAIN_SEASONS` only, and
say so explicitly in its docstring (see `compute_venue_air_density_baseline`'s docstring and
`tests/test_join_weather_features.py::test_venue_baseline_computed_from_training_seasons_only`).

## Watch for confounding-by-indication in any feature reflecting a human/strategic decision

Version 0.6's `alignment_interactions_v06` (`mlb_luck_score.models.compare_alignment_aware`)
found a real, bootstrap-confirmed aggregate log-loss and ECE improvement, yet its controlled
-perturbation check showed "Strategic" outfield alignment predicting MORE pull-side doubles
than "Standard" -- backwards from the physical expectation that a strategic shift exists to
prevent exactly that. This is very plausibly not a code bug: alignment (like a shift, a pitch
call, a defensive substitution, or any other in-game decision made by a human in response to
the SAME conditions the model is trying to predict) is not randomly assigned -- teams shift
more against batters already known to be extra-base threats, so a model trained on
observational data can learn "this alignment co-occurs with this outcome" rather than "this
alignment causes this outcome." A material subgroup regression can point at the same root
cause from a different angle (see `bb_type_ground_ball`'s regression in the same candidate).
When evaluating ANY future feature that reflects a strategic choice rather than a fixed
physical fact (park geometry and weather are physical facts; alignment, positioning, and
similar decisions are not), treat a backwards-signed controlled-perturbation result as a
likely confounding signal first, not immediately as a bug to "fix" by relaxing the check --
see "Alignment-aware positioning (Version 0.6)" in README.md for the full real-data writeup.

## pandas' `pd.NA` sentinel breaks more than `SimpleImputer` -- audit raw comparisons too

The nullable-dtype `pd.NA` footgun documented elsewhere in this file (`SimpleImputer`
raising `TypeError: boolean value of NA is ambiguous`) is not sklearn-specific. Version
0.7A's `mlb_luck_score.models.compare_opportunity_models.
compute_defender_subgroup_calibration` hit the SAME error from a completely different code
path: a raw numpy `fielder_arr == fielder_id` equality comparison on an object-dtype array
that still contained `pd.NA` elements (`pd.NA == x` returns `pd.NA`, not `False`, and numpy
cannot coerce that to a boolean mask). The fix was the same pattern used elsewhere for
stringified booleans (`compare_geometry_aware._bool_mask`): normalize `pd.NA` to plain
Python `None` (via `series.where(series.notna(), None)`) BEFORE any raw `==`/`bool()`
operation, since `None == x` returns an ordinary `False` with no ambiguity. When writing
ANY code that does raw equality comparisons, boolean masking, or truthiness checks on a
pandas Series/array that may contain nulls -- not just when feeding a column into
`SimpleImputer` -- check whether nulls are `pd.NA` (nullable dtype) or plain `np.nan`
(float dtype); only the former raises this class of error, and only a real audit of the
specific dtype in play catches it, not just following the `SimpleImputer` pattern by rote.

## A model with no prior baseline needs an ABSOLUTE quality bar, not a relative one

Every comparison module through Version 0.6 (`compare_park_aware`, `compare_geometry_
aware`, `compare_weather_aware`, `compare_weather_variants`, `compare_alignment_aware`)
evaluates a CANDIDATE against `baseline_v02` or `selected_production_baseline` -- a prior
production model that already exists. Version 0.7A's `measured_contact_only_v07`
(`mlb_luck_score.models.compare_opportunity_models`) has no such baseline: it is a
genuinely NEW binary opportunity-difficulty model, not a replacement for anything already
in production. `summarize_opportunity_validation` therefore checks calibration against a
FIXED absolute ECE threshold (`MATERIAL_ECE_ABSOLUTE_THRESHOLD`), not a delta relative to
a baseline's own calibration -- there is nothing to take a delta against. This is why that
module reports `passes_basic_validation`, not `recommend_adopt`: "adopt" implies replacing
something, and there is nothing here to replace. Do not force a genuinely new model
category through the relative-comparison adoption-rule pattern just for consistency with
earlier versions -- when a future phase (e.g. infield pickup/throwing models) introduces
another model with no prior baseline, use the SAME absolute-quality-bar pattern instead of
inventing a placeholder baseline to diff against.

## Any override of a raw feature must recompute every feature DERIVED from it

This exact bug class has now been caught THREE times in this repository, in three
different modules, which is why it gets its own rule rather than staying a per-module
docstring note. `mlb_luck_score.features.build_contact_features.
generate_standardized_environment_rows` (Version 0.5) and `generate_typical_alignment_rows`
(Version 0.6) both document it; Version 0.7C's near-wall controlled-perturbation checks
(`mlb_luck_score.models.compare_near_wall_models`) hit it a third time, in a NEW form:
overriding `wall_distance_in_spray_direction` without recomputing `projected_distance_to_
wall_margin`/`absolute_distance_to_wall` (both DERIVED from it via `mlb_luck_score.data.
join_park_geometry`'s `margin = hit_distance_sc - wall_distance_in_spray_direction`)
produced a spurious backwards-signed perturbation result -- the model was correctly reading
a genuinely self-contradictory row (raw wall distance said "close," the stale derived
margin still said "far"), not learning something wrong. The same issue applies to
`estimated_hang_time_s`, which is itself DERIVED from `launch_speed`/`launch_angle` via
`mlb_luck_score.data.outfield_physics.estimate_hang_time_seconds` -- overriding hang time
directly (rather than overriding the genuinely independent launch angle and recomputing
hang time from it) would repeat the same mistake one feature removed. Before writing ANY
new counterfactual/perturbation/override logic, explicitly enumerate every feature DERIVED
from the one being overridden (grep for where it's computed) and recompute all of them in
the same override step -- do not assume a single `override_columns`-style column-by-column
substitution is safe just because it worked for a case with no derived dependents.

## Subgroup ECE against a fixed absolute threshold does not transfer across population sizes

Verified in Version 0.7C (`mlb_luck_score.models.compare_near_wall_models`, real 2024
data): the SAME `MATERIAL_ECE_ABSOLUTE_THRESHOLD` (0.05) that works well for open-field
-scale subgroups (tens of thousands of rows) flags 28 of 30 reliably-sampled venues in the
near-wall-specific final comparison, where each venue subgroup is only ~200-330 rows --
ECE estimated from a sample that small has enough sampling variance that many venues could
cross 0.05 even under a genuinely well-calibrated model, not because they are actually
worse. Separately, restricting `open_field_v07`'s ALREADY-VALIDATED predictions (Version
0.7A, blended ECE 0.017771) to just its own open-field-gated subset gave ECE 0.063025 for
the EXACT SAME model -- independently confirmed NOT a bug: aggregate ECE across a mixed
population is not a simple weighted decomposition of subgroup ECEs, and restricting to (or
mixing) subpopulations can shift the aggregate figure in either direction purely from bin
-composition effects, unrelated to genuine miscalibration. When applying an existing
absolute-quality threshold to a NEW, smaller subgroup population, report the result
honestly (do not hide or silently pass a real-looking failure) but also do not treat it as
equivalent evidence to the same threshold applied at the original population's scale --
note the sample-size caveat explicitly, as `compare_near_wall_models` does, rather than
either recalibrating the threshold ad hoc or treating every crossing as a confirmed defect.

## Measured miscalibration vs. insufficient evidence are DIFFERENT findings -- never conflate them

Version 0.7D (`mlb_luck_score.models.compare_near_wall_calibration_gate`) replaces the rule
above's fixed-ECE-threshold pass/fail with a game_pk-clustered bootstrap confidence interval
per subgroup/venue, and a three-way status instead of a boolean:

  - `calibrated`: adequate outcome support (plays/games/minority-class count all above a
    documented minimum) AND the ECE confidence interval sits entirely at or below the
    material threshold, with no credible paired regression against `open_field_v07` on the
    same rows.
  - `not_calibrated`: CREDIBLE evidence of a problem -- an ENTIRE confidence interval on the
    wrong side of a line (either the ECE CI entirely above threshold, or the paired
    specialist-vs-baseline log-loss delta CI entirely above zero), never a bare point
    estimate.
  - `insufficient_evidence`: EITHER raw support is below the documented minimum, OR support
    is nominally adequate but the confidence interval straddles the threshold -- the data
    genuinely cannot tell "calibrated" from "not calibrated" yet.

This resolved the previous rule's own documented weakness with a REAL, unexpectedly sharp
result on real 2024 data: applying this to the near-wall specialist showed 6 subgroups/
venues genuinely `calibrated`, but 19 (mostly individual venues, ~250-330 rows each, plus
`wall_height_medium`/`wall_height_tall`) are CREDIBLY `not_calibrated` -- their ECE
confidence intervals hold up entirely above 0.05 under bootstrap resampling, even though
the SAME rows show the specialist is a credible, large improvement over `open_field_v07`
(paired log-loss delta CIs entirely negative). This DISPROVES the earlier "probably just
sampling noise" hypothesis for a meaningful fraction of these venues -- some of that
apparent miscalibration is real, not an artifact of small-sample ECE variance, even though
the specialist is still comparatively much better than the alternative. The remaining 20
subgroups/venues are honestly `insufficient_evidence` (mostly smaller venues and the
`near_wall_5ft`/`spray_sector_left`/`spray_sector_left_center`/`spray_sector_right`/
`opportunity_time_q1_shortest`/`opportunity_time_q2` subgroups) -- NOT reported as passing.
See "Version 0.7D: calibration-gate correction" in README.md for the full table.

`near_wall_specialist_calibrated` is `True` ONLY if every subgroup/venue is EITHER
`calibrated` or has adequate evidence with nothing `not_calibrated` -- if some groups are
`insufficient_evidence` but none are credibly `not_calibrated`, the overall status is the
separately-labeled `calibrated_with_limited_subgroup_evidence`, which STILL does not flip
`near_wall_specialist_calibrated` to `True` (near-wall rows stay `provisional_near_wall`) --
"insufficient evidence" must never be reported as a pass, per the task that produced this
correction. When adding ANY future confidence-interval-based gate, follow the SAME pattern:
a credible finding requires an entire CI on the wrong side of a line, and "we don't have
enough data to tell" is its own honestly-labeled outcome, never silently merged into either
"pass" or "fail".

## Version 0.7 is now FROZEN -- 2024 is development validation, not an untouched test set

The near-wall design line (Versions 0.7A-0.7D, `mlb_luck_score.models.
compare_near_wall_models`/`compare_near_wall_calibration_gate`) has now had its checks,
required-check set, and calibration-gating rule redesigned MULTIPLE times specifically in
response to what real 2024 results looked like each time (0.7C's launch-angle perturbation
check was replaced after its result looked backwards on 2024 data; 0.7D's subgroup-gating
rule was replaced after the SAME 2024 venue results looked like a threshold artifact). This
is fundamentally different from `FINAL_TEST_SEASONS` (2025, never touched at all) -- 2024
has been iterated against repeatedly, so it no longer functions as an untouched validation
season FOR THIS DESIGN LINE, even though `mlb_luck_score.config.VALIDATION_SEASONS`/
`CALIBRATION_EVAL_SEASONS` still formally include it. Per the task that produced Version
0.7D: **do not make any further near-wall model, feature, perturbation-check, or
calibration-gate change that is justified by, or tuned against, 2024 near-wall results.**
If a future near-wall change is genuinely needed, either (a) evaluate it against a season
range not yet used for near-wall design decisions, or (b) get explicit maintainer sign-off
to treat 2024 as non-pristine for that specific change, and say so plainly in the
docstring/README -- do not silently repeat the pattern this rule documents. This does NOT
freeze anything else in the codebase (weather, geometry, alignment, contact model, etc.
proceed under their own existing rules above).

## Confidence must never dampen the score

The Version 0.1 confidence report
(`mlb_luck_score.scoring.confidence`) is descriptive data-completeness reporting only. Do
not use it to pull `raw_luck` or the public score toward zero, and do not present it as a
statistical uncertainty estimate -- it isn't one.

## Use conservative scientific language

Do not describe the model as "accurate," "validated," or "calibrated" without pointing to
the specific evaluation numbers that support the claim, computed on real held-out
(non-2025) data. Running calibration or evaluation code successfully is not evidence that
the results are good -- report the actual numbers and let them speak. Prefer "provisional,"
"preliminary," and "Version 0.1 research placeholder" over stronger language, matching the
existing docstrings.

## Document assumptions and provisional choices

Every provisional constant (the ordinal value map, the public-score scale, the spray-angle
formula, the eligible-event list) already has a docstring explaining it is a Version 0.1
choice, not a validated result. When you add a new one, do the same -- future readers
(human or agent) should be able to tell a documented placeholder from a validated result at
a glance.

## Prefer small, reviewable changes

This is a research prototype under active iteration. Prefer focused diffs over broad
refactors; don't restructure multiple modules in one change unless asked.

## Inspect existing work before editing

Before modifying a module, read it (and its tests) in full. Before running any git
operation, run `git status` first -- do not assume the working tree is clean.

## Avoid destructive Git operations

Never run `git push --force`, `git reset --hard`, `git checkout -- <path>` /
`git restore` over uncommitted work, `git clean -f`, or `git branch -D` unless the user
explicitly asks for that specific action in that specific moment. Prefer creating a new
commit over amending an existing one.

## Ask before pushing, publishing, or changing external resources

Never `git push`, never create or modify a git remote, never create cloud resources, and
never touch CI/CD configuration without the user explicitly asking first in that
conversation. Downloading Statcast data requires internet access -- that's expected and
fine, but always say so before running a download. The full development dataset download
(`make download-development-data`, all four 2021-2024 seasons) is large and can take a
long time (see README.md "Full development dataset" for storage/runtime estimates) --
implement and test that workflow with synthetic data first, then show the user the exact
command and its storage/runtime considerations, and get explicit approval before actually
running it against the network.

## Before finishing any change

Run, in this order, and fix failures before reporting done (or explain clearly why you
couldn't):

```bash
.venv/bin/python -m ruff format src tests   # or: make format
.venv/bin/python -m ruff check src tests    # or: make lint
.venv/bin/python -m mypy src                # or: make typecheck
.venv/bin/python -m pytest                  # or: make test
```

All four are bundled in `make check`. The test suite must remain fully offline --
never add a test that requires network access or real Statcast data.
