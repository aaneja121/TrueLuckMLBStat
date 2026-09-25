# Operations

How a Contact Luck snapshot is produced, verified, archived and deployed: the daily
GitHub Actions loop, the single orchestration entry point, the frozen input bundle,
durable Cloudflare R2 archival, history sync, credentials, and recovery.

Companion documents: [`RESEARCH_LOG.md`](RESEARCH_LOG.md) (model development history),
[`DATA.md`](DATA.md) (inputs, licensing, reproducibility).

## Version 1.2: dashboard deployment and operations

Version 1.2 (`dashboard/`) is a read-only, static-site presentation layer over Version
1.1's immutable prospective snapshots -- see CLAUDE.md and `dashboard/snapshot_data.py`'s
module docstring for the full snapshot-selection/precedence rules and integrity
guarantees. This section documents the existing production workflow: generate a
snapshot, rebuild the dashboard, deploy the static output. It changes no model,
prospective-scoring, snapshot, or dashboard logic.

**1. Generate the newest snapshot only after the requested MLB slate is fully
complete.** Version 1.1's own `assert_data_through_date_is_complete` guard already
refuses an incomplete date (see "Version 1.1: prospective 2026 scoring" in
[`RESEARCH_LOG.md`](RESEARCH_LOG.md)), so this
is enforced, not just a convention:

```bash
.venv/bin/python prospective/run_v1_1_2026_scoring.py \
    --data-through YYYY-MM-DD
```

**2. `--force-redownload` is not part of the normal workflow.** The prospective
runner's raw Statcast cache is coverage-aware -- it refreshes itself automatically
whenever its verified coverage doesn't reach the requested `--data-through` date (see
"v1.1.2 operational correctness fix" in [`RESEARCH_LOG.md`](RESEARCH_LOG.md)). Only pass
`--force-redownload` if there is
a specific reason to force a cache refresh regardless of coverage.

**3. Verify the snapshot completed successfully** by checking its manifest
(`artifacts/prospective/v1_1/<snapshot>/manifest.json`) reports:
- the requested `data_through_date`
- `observed_date_coverage` actually reaching that date
- `schema_checks.raw_statcast_cache_coverage_validation.decision` (`reused` or
  `refreshed`) and its `reason`
- `schema_checks.final_pre_scoring_coverage_check.missing_completed_game_dates` is empty
- `output_hashes`/`frozen_artifact_hashes` are present -- the same information
  `dashboard/snapshot_data.py` independently re-verifies (via `integrity_hashes.json`)
  before the dashboard will ever display the snapshot; see "Integrity validation" in
  that module's docstring

**4. Build the dashboard:**

```bash
.venv/bin/python dashboard/build.py
```

**5. The dashboard automatically resolves the newest valid preferred snapshot** -- no
flag is needed to point it at a specific date. Historical snapshots remain immutable; a
corrected/refreshed snapshot for an earlier date never affects which snapshot is newest
overall (see `dashboard/snapshot_data.py`'s "Snapshot precedence" docstring section).

**6. Local preview:**

```bash
cd dashboard/dist
python3 -m http.server 8000
```

**7. Production artifact:** `dashboard/dist/` -- a plain static site (HTML/CSS/JS plus a
couple of small JSON payloads). This entire directory can be deployed to a static host
as-is.

**8. Production architecture:**

```
completed MLB slate
    -> immutable prospective snapshot (prospective/run_v1_1_2026_scoring.py)
    -> dashboard/build.py
    -> dashboard/dist/
    -> static hosting
```

**9. Operational guarantees:**
- Production hosting never trains or scores models -- `dashboard/dist/` is static
  output; nothing in it executes Python.
- The dashboard never downloads Statcast -- see `tests/test_dashboard_isolation.py` for
  the structural check that no module under `dashboard/` imports scoring, training, or
  download code.
- Sealed 2025 evaluation data is never used as live product data -- the dashboard only
  reads `outputs/prospective/v1_1/`/`artifacts/prospective/v1_1/`.
- Missing historical snapshot dates are allowed and are never interpolated -- a gap
  (e.g. a day with no snapshot run) simply has no entry in the trend or snapshot
  history; see `dashboard/visuals.py`'s trend-chart docstring.
- Corrected snapshots follow the dashboard's deterministic precedence rules -- a later,
  differently-labeled snapshot for the same `--data-through` date supersedes an earlier
  one for display (both remain on disk, unmodified); a retrospective-backfill snapshot
  (if one is ever produced by a future, separate mechanism) never supersedes a genuine
  or corrected one for the same date.

**Deploying.** The dashboard is hosted on Cloudflare Pages (project `contact-luck`),
deployed via `wrangler pages deploy dashboard/dist --project-name=contact-luck`.
`scripts/publish_snapshot.sh` (below) wraps the whole snapshot-to-deploy chain behind
one command and remains the only orchestration entry point -- nothing reproduces its
commands elsewhere.

### Operational entry point: `scripts/publish_snapshot.sh`

```bash
scripts/publish_snapshot.sh --data-through YYYY-MM-DD [options]
```

Runs, in order, and stops at the first failure: `prospective/run_v1_1_2026_scoring.py
--data-through <date>` -> `dashboard/build.py --explore-artifacts-dir <ephemeral dir>
--pitcher-season-fixture dashboard/pitcher_season_fixture.json` -> (unless
`--skip-deploy`) a confirmation
prompt -> `wrangler pages deploy dashboard/dist --project-name=contact-luck`. It
duplicates none of Version 1.1's guards (clean working tree, date completeness, coverage
validation, conflict detection) -- it only calls the existing entry points and reports
their exit codes. Flags: `--snapshot-label`, `--project-name` (default `contact-luck`),
`--skip-deploy` (build only, no deploy -- the dry-run path), `--yes` (skip the
interactive confirmation, required for any non-interactive/CI invocation), `--help`.

**Manual publication.** Run the command above directly from a clean working tree once a
date's MLB slate is fully complete. Omit `--skip-deploy` to deploy for real (you'll be
asked to confirm unless `--yes` is also passed).

### Scheduled publication: `.github/workflows/publish-prospective.yml`

A GitHub Actions workflow triggers `scripts/publish_snapshot.sh` on a schedule, so
publishing doesn't require a human to run the command by hand every day. It is a thin
trigger only -- it builds the same `.venv` the script expects
(`python -m venv .venv && .venv/bin/pip install -e ".[dashboard]"`), calls the script as
a single atomic step, and stops there. It never reimplements or bypasses any Version 1.1
guard.

- **Manual trigger**: GitHub -> Actions -> "Publish prospective snapshot" -> "Run
  workflow". Inputs: `data_through` (optional -- blank auto-resolves yesterday in
  America/New_York) and `deploy` (checkbox, default OFF -- manual runs default to a dry
  run, matching the scheduled default below).
- **Schedule**: once daily at `13:00 UTC` (`0 13 * * *`). GitHub Actions cron is fixed
  UTC and does not shift for daylight saving: `13:00 UTC` is `09:00 America/New_York`
  during EDT (roughly mid-March to early November -- most of the season) and `08:00`
  during EST. Either is a conservative morning buffer after even a late West Coast
  extra-inning game; the cron time only needs to land "safely after games usually end,"
  not be exact, because the actual `--data-through` date is resolved separately (next
  point) and a slate that somehow isn't complete yet is rejected by Version 1.1's own
  date-completeness guard rather than silently scored partial.
- **Date resolution**: `scripts/resolve_data_through_date.sh` resolves "yesterday in
  America/New_York" via the runner's tzdata (`TZ="America/New_York" date --date="...
  yesterday"`), not a fixed UTC offset -- a run firing at, say, 02:00 UTC can still be
  evening of the previous day in New York, and a naive UTC-calendar-date approach would
  be off by one. Extracted into its own script specifically so this logic has direct
  test coverage (`tests/test_resolve_data_through_date.py`, Linux-only -- see that
  script's header for why) independent of running the workflow or real prospective
  scoring. It also fails loudly (rather than silently returning a wrong date) if the
  `tzdata` package is missing -- discovered during development that a bare
  `TZ=America/New_York` conversion against a missing zoneinfo file does not error, it
  just silently fails to convert.
- **Current mode: `PROSPECTIVE_AUTO_DEPLOY` is set to `true` -- scheduled runs perform
  real archive writes and real deploys.** The rollout gate this variable provides was
  exercised as designed before being flipped: scheduled/dry-run cycles were inspected
  (via the uploaded `dashboard/dist/` and snapshot-manifest artifacts) to confirm the
  season-to-date trend charts showed the full historical series before any unattended
  real deploy was allowed to happen -- see "Durable archival"/"Frozen input bundle
  portability" below for that verification work, and the "known gap" note there for what
  is still not automated. With the variable unset or anything other than `true`,
  scheduled runs fall back to `scripts/publish_snapshot.sh --data-through "$DATE"
  --skip-archive --skip-deploy --yes` -- snapshot generation, the read-only history sync,
  and the dashboard rebuild happen for real, but nothing is written to R2 and nothing is
  deployed. `publish_snapshot.sh` refuses `--skip-archive` without `--skip-deploy` (see
  the flag table above), so this workflow only ever produces one of two states: both
  flags, or neither.
- **Toggling the mode**: set the repository variable `PROSPECTIVE_AUTO_DEPLOY` (GitHub ->
  Settings -> Secrets and variables -> Actions -> Variables) to `true` for real scheduled
  deploys, or to anything else (or unset it) to fall back to scheduled dry runs. That is
  the *only* change needed either direction -- the workflow YAML does not change. A manual
  run can independently opt into a real deploy any time via the `deploy` checkbox,
  regardless of this variable.
- **Required GitHub secrets.** `CLOUDFLARE_API_TOKEN` and the actual Cloudflare Pages
  deploy are only consumed when an actual deploy happens (scope the token to Pages edit
  access for this project only, not full account access). The four `R2_*` credentials
  (`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, plus `CLOUDFLARE_ACCOUNT_ID` reused as
  `R2_ACCOUNT_ID`), in contrast, are consumed on **every** run now, dry or not -- history
  sync always reads from R2, even in dry-run mode. Set under GitHub -> Settings -> Secrets
  and variables -> Actions -> Secrets. Never committed anywhere in this repository.
- **Concurrency**: all runs share a single `publish-prospective` concurrency group with
  `cancel-in-progress: false` -- a second trigger while one is in flight queues rather
  than racing it or cancelling a possibly-mid-deploy run.
- **When a guard rejects a run**: the workflow step fails with `scripts/
  publish_snapshot.sh`'s own exit code and error message (e.g. a dirty working tree, an
  incomplete `--data-through` date, or a snapshot that already exists with different
  inputs) -- exactly as it would locally. The workflow does not retry (a retry could race
  or attempt to bypass Version 1.1's own immutability/conflict handling) and never falls
  back to an earlier date on its own.

### Frozen input bundle portability: `scripts/ensure_frozen_inputs.py`

**The problem, in full.** Version 1.1's snapshot manifest freezes its inputs by hashing
every path in `evaluation.v1_final_evaluation_manifest.FROZEN_ARTIFACT_RELATIVE_PATHS` --
the SAME 32-path list the sealed Version 1.0 evaluation uses (`prospective.
prospective_manifest.build_snapshot_manifest` calls `build_artifact_hashes(repo_root,
relative_paths=FROZEN_ARTIFACT_RELATIVE_PATHS)` directly, never a second,
prospective-specific list). 27 of those 32 are `src/mlb_luck_score/*.py` source files --
already git-tracked, present on any fresh checkout automatically. The remaining FOUR are
gitignored, local-only data/detail files (CLAUDE.md "Never commit datasets") that have
only ever existed on a maintainer's own long-lived machine:

| Local path | Produced by |
|---|---|
| `data/processed/cleaned_development_data_with_sprint_speed.parquet` | the Version 0.x data-cleaning pipeline |
| `outputs/tables/opportunity_model_comparison_detail.json` | `mlb_luck_score.models.compare_opportunity_models` |
| `outputs/tables/infield_opportunity_detail.json` | `mlb_luck_score.models.compare_infield_opportunity` |
| `outputs/tables/advancement_detail.json` | `mlb_luck_score.models.compare_advancement_models` |

The first scheduled GitHub Actions dry-run failed on the parquet alone (fixed first, as an
initially parquet-only version of this module). The SECOND scheduled dry-run then got all
the way through downloading/cleaning 2026 data and fitting/selecting the frozen component
models, only to fail building the snapshot manifest on the remaining three -- exposing
that the portability gap was never just the parquet. `scripts/ensure_frozen_inputs.py`
now covers the complete, audited set as one versioned bundle, not three more one-off
patches.

**The fix, and why it never regenerates anything.** `scripts/ensure_frozen_inputs.py`
runs as the very first stage of `scripts/publish_snapshot.sh`, before scoring, and
ensures EVERY bundle entry in order: reuse the local file if it's already there AND its
sha256 matches a hash pinned in source (never trusted blindly), otherwise fetch it from a
dedicated, immutable Cloudflare R2 object and verify the same pinned hash before writing
anything to disk. Processing is fail-fast -- the first entry that can't be satisfied stops
the whole stage immediately, and no later entry is attempted. There is no third option for
any entry: the module contains no import of, or call into, any cleaning/feature
-engineering/model-comparison/Statcast-download code (see `tests/
test_ensure_frozen_inputs.py::TestNoRegenerationFallback`'s structural AST check), so
"regenerate it" is not a code path that exists here, not just a policy this script happens
to follow.

- **Storage layout: one bundle, relative-path-preserving keys.** The three detail JSONs
  use keys that mirror their local relative path exactly, e.g.
  `frozen-inputs/v1/outputs/tables/opportunity_model_comparison_detail.json`. The parquet
  keeps its already-live key unchanged (`frozen-inputs/v1/cleaned_development_data_with_
  sprint_speed.parquet`, flat) rather than being migrated to match -- that object was
  already uploaded and independently verified against real R2 before this bundle existed,
  and re-keying it would silently orphan a working, already-verified production object for
  no functional benefit. Every key lives under the SAME R2 bucket `scripts/
  archive_snapshot.py` uses (so CI needs no new secrets), under a deliberately separate
  top-level prefix (`frozen-inputs/` vs. `prospective/`) so none of this bundle can ever be
  discovered by, or confused with, `list_archived_snapshots()`/`--sync-history`'s per-date
  snapshot enumeration. The `v1` segment versions the ARCHIVED INPUT BUNDLE itself; a
  legitimately different frozen input (a new Version 0.x data-cleaning or model-selection
  run) would become a `frozen-inputs/v2/...` key with its own pinned hash, never an
  in-place overwrite.
- **The pinned hashes**, each independently cross-checked against `outputs/
  final_evaluation/v1/v1_final_report.json`'s own `manifest.artifact_hashes` entry for that
  same path (recorded when the real, sealed Version 1.0 evaluation actually ran against
  it) -- every entry agrees exactly:

  | Local path | SHA256 | Size |
  |---|---|---|
  | `data/processed/cleaned_development_data_with_sprint_speed.parquet` | `f791415d218334aa578fd8104e7c3aec5f52716931fc69287aa2ee05b12443f8` | 111,445,297 bytes |
  | `outputs/tables/opportunity_model_comparison_detail.json` | `a694f6f65f0f94b7ed30fd785144a363d5f075a4a465f6a5b9a0eda2237a1d38` | 49,619 bytes |
  | `outputs/tables/infield_opportunity_detail.json` | `2b879a72af11618b8d9f8939d900120c325c20990698261d7f4dcbb95b8141f0` | 85,434 bytes |
  | `outputs/tables/advancement_detail.json` | `8826bae37c6b51489098018e95154d47ec0e1f05cd7c52bd29a633b20d01e222` | 18,174 bytes |

- **Local-first, zero network dependency in the common case.** The R2 client is
  constructed AT MOST ONCE per `ensure_frozen_inputs.py` invocation, shared across every
  bundle entry that actually needs one, and never constructed at all if every local file
  is already present and valid -- true of a maintainer's own machine, false of a fresh CI
  runner (before the one-time seed) or any runner missing part of the bundle.
- **Fails loudly, never silently, on every anomaly, for every entry**: a local file that
  exists but doesn't match its pinned hash (never redownloaded or overwritten -- investigate
  by hand), a missing R2 object, or a downloaded object whose hash doesn't match the pin
  (discarded, never written to disk looking like a verified artifact). See
  `FrozenInputLocalHashMismatchError` / `FrozenInputMissingRemoteObjectError` /
  `FrozenInputRemoteHashMismatchError`.
- **The bundle can never silently drift out of sync with the real frozen-artifact list.**
  `tests/test_ensure_frozen_inputs.py::TestBundleCoversEveryGitignoredFrozenArtifact`
  cross-references every path in `FROZEN_ARTIFACT_RELATIVE_PATHS` against `git
  check-ignore` and asserts every gitignored one is present in `FROZEN_INPUT_BUNDLE` --
  exactly the audit that would have caught today's gap automatically before it ever
  reached a real dry-run, and the guard that keeps a FUTURE frozen artifact from repeating
  it.
- **The one-time seeding upload (`--upload`) is a separate action**, never invoked by
  `ensure_frozen_input_bundle()` or `scripts/publish_snapshot.sh` -- run by hand, once per
  entry that hasn't been seeded yet, with explicit human intent, mirroring `scripts/
  archive_snapshot.py`'s own write-once philosophy (refuses to upload a local file that
  doesn't match the pin, refuses to overwrite an existing, DIFFERENT R2 object, re-verifies
  each upload by fetching it back). Safe to re-run over an already-fully-seeded bundle --
  every entry simply no-ops.
- **This stage always runs**, even under `--skip-archive --skip-deploy`, because scoring
  and manifest generation need the complete bundle regardless of what happens to the
  snapshot's output afterward -- the same reasoning that makes history sync always run.
  See `tests/test_publish_snapshot_orchestration.py::TestFailuresBlockLaterStages::
  test_ensure_frozen_inputs_failure_prevents_everything_after_it` for the orchestration
  proof that a failure here blocks scoring (and everything after it), exactly like a
  scoring failure would.

### Durable archival: `scripts/archive_snapshot.py`

**The problem.** `outputs/prospective/v1_1/`/`artifacts/prospective/v1_1/` are gitignored
by design (see "Version 1.1" in [`RESEARCH_LOG.md`](RESEARCH_LOG.md)) -- correct for a
snapshot generated on a
maintainer's own machine, but a snapshot generated by a GitHub Actions run exists only
on that run's disposable runner and vanishes when the job ends. `scripts/
archive_snapshot.py` copies a completed local snapshot's files, byte for byte, into a
durable Cloudflare R2 bucket, so an official snapshot survives runner destruction.

**Pipeline ordering: score -> archive -> sync history -> build -> deploy.** Archival runs
immediately after scoring, history sync runs immediately after that, and the dashboard is
never built (let alone deployed) if EITHER fails. This is deliberate: the raw scoring
output is the precious, comparatively irreplaceable artifact (re-scoring an old date
depends on the same historical Statcast data still being fetchable, which is not
guaranteed indefinitely), while the dashboard build is cheap and already iterated on
constantly. Archiving first means an official snapshot is durable even if a LATER stage
breaks, and it means "the public site must never deploy if durable archival failed"
falls directly out of `scripts/publish_snapshot.sh`'s existing sequential
`set -euo pipefail` structure -- no special-cased check was needed. This ordering
guarantee has a dedicated regression test (`tests/test_publish_snapshot_orchestration.py`)
that runs the real script against fake `.venv/bin/python`/`npx` executables and asserts,
by inspecting what was actually invoked, that a failed archive OR a failed history sync
genuinely prevents the build and deploy steps from running -- not just an argument that
`set -e` ought to guarantee it.

**`--skip-archive` and `--skip-deploy` are independently controllable**, specifically so
real R2 archival can be validated on its own before production deploys are enabled.
History sync is NOT gated by either flag -- it always runs, because it only ever READS
from R2 (never writes), so even a full dry run exercises the real CI dashboard-build
behavior (see "History sync" below for why that matters):

| Flags | Behavior |
|---|---|
| (neither) | `score -> archive -> sync history -> build -> deploy` |
| `--skip-deploy` | `score -> archive -> sync history -> build -> stop` (archives for real, doesn't deploy) |
| `--skip-archive --skip-deploy` | `score -> sync history (read-only) -> build -> stop` (writes to neither R2 nor Cloudflare Pages) |
| `--skip-archive` alone | **refused** -- the script will not deploy a dashboard built from a snapshot that wasn't just durably archived; there is no override flag for this |

Scoring, history sync, and the dashboard build happen for real in every row above; only
the archive WRITE and the deploy are ever skipped.

**History sync (why it exists, and why the dashboard itself stays filesystem-only).** A
GitHub Actions runner starts from a fresh git checkout -- `outputs/prospective/v1_1/`/
`artifacts/prospective/v1_1/` are gitignored, so a CI job that just scored today's date
has ONLY today's snapshot locally. Without more, `dashboard/build.py`'s season-to-date
trend charts would show a single point on every CI-built dashboard, no matter how much
history is sitting in R2 -- `dashboard/snapshot_data.py` itself was deliberately never
changed to fetch from R2 directly (see its own module docstring's "read-only boundary");
it still only ever reads local disk. Instead, `scripts/archive_snapshot.py --sync-history
--season <year>` runs as its own pipeline stage, between archiving and the dashboard
build, and repopulates local disk from R2 before the dashboard ever sees it:

- `list_archived_snapshots(season=...)` enumerates every candidate `<snapshot_dir_name>`
  under `prospective/<season>/` in the archive (never a hardcoded date list). A key only
  becomes a candidate at all if its name has the SHAPE of a real snapshot directory
  (`YYYY-MM-DD` or `YYYY-MM-DD__<label>`, a genuine calendar date) -- a key that reaches
  the right depth but isn't shaped like a snapshot (some unrelated object under the same
  prefix) is silently ignored, never reported. Every candidate that DOES pass the shape
  check then gets a completeness determination: complete only if `artifacts/
  integrity_hashes.json` exists, parses, AND every file it references is also actually
  present in the archive.
- `sync_missing_snapshots(season=...)` restores every COMPLETE archived snapshot missing
  locally, using the exact same `restore_snapshot()` hash-verification machinery. A
  `<snapshot_dir_name>` that already exists locally is either an identical, safe no-op
  (compares `integrity_hashes.json` AND re-hashes every referenced local file) or a
  `HistorySyncConflictError` -- partial/corrupt/differing local snapshots are NEVER
  auto-repaired or overwritten. A candidate that passed the shape check but failed
  completeness raises `HistorySyncIncompleteArchiveError` and stops the whole sync
  immediately -- a broken OFFICIAL snapshot entry is never silently skipped, only a
  genuinely unrelated key (one that never became a candidate) is ignored without blocking
  anything else.
- The representative end-to-end regression test (`tests/test_archive_history_sync.py`)
  builds a scenario with several historical snapshots that exist ONLY in the archive plus
  a freshly-scored local "today" snapshot -- exactly what a real CI runner sees -- and
  confirms `dashboard/snapshot_data.discover_snapshots()` sees only today's point BEFORE
  sync, and the full, correctly-precedented date history AFTER it.

**Archive layout** mirrors the existing two-namespace local contract exactly, for every
file actually present (enumerated dynamically, never a hardcoded filename list, so a
future Version 1.1 output file is archived automatically):

```
prospective/<season>/<snapshot_dir_name>/outputs/<every file from
    outputs/prospective/v1_1/<snapshot_dir_name>/>
prospective/<season>/<snapshot_dir_name>/artifacts/<every file from
    artifacts/prospective/v1_1/<snapshot_dir_name>/>
```

`<season>` comes from the snapshot's own `manifest.json` (`prospective_season`), and
`<snapshot_dir_name>` is `YYYY-MM-DD` or `YYYY-MM-DD__<label>` -- identical to the local
directory-naming convention `run_v1_1_2026_scoring.py`/`dashboard/snapshot_data.py`
already use, so a corrected/refreshed snapshot (e.g. `2026-08-08__refreshed`) archives as
its own separate, additional entry, exactly mirroring how it exists locally.

**Write-once.** `artifacts/integrity_hashes.json` (already written by Version 1.1) is
reused directly as the identity anchor -- no second hashing scheme was invented. If the
archive has no entry yet for a `<snapshot_dir_name>`, every local file is uploaded, then
immediately re-fetched and compared to confirm the upload actually took. If an entry
already exists, its `integrity_hashes.json` is compared (as parsed JSON, not raw bytes)
against the local one: identical -> no-op (safe to rerun); different -> `ArchiveConflictError`,
and the archive is NEVER silently overwritten.

**Recovery -- two related but distinct tools.** `scripts/archive_snapshot.py --restore
--data-through YYYY-MM-DD --season 2026 [--snapshot-label LABEL]` is the explicit,
on-demand reverse operation for ONE known snapshot -- downloads it back into the normal
local paths, verifies every file against the archived hashes, and refuses (never silently
overwriting) if different local files already exist at that path. `--sync-history` (above)
is the bulk analog: every missing snapshot for a season, discovered automatically rather
than named one at a time, which is what the automated pipeline actually runs. **Either
way, this is the only way R2 data reaches local disk** -- the dashboard (`dashboard/
snapshot_data.py`) and Version 1.1's own guards are UNCHANGED by this: they still only
ever read local disk. Nothing auto-restores from R2 outside these two explicit calls; if
a future version wants the dashboard itself to depend on R2 at runtime, that is a real
architectural change (a new runtime dependency on remote state) and deserves its own
explicit decision -- not something either tool does quietly.

**Cloudflare R2 configuration.**
- A bucket dedicated to this archive (`contact-luck-prospective-archive`) -- created and in
  active use (see below); nothing in this codebase creates one automatically, so a future
  fork/redeploy still needs this step done by hand.
- An **R2 API token** scoped to Object Read & Write on that one bucket only (Cloudflare
  dashboard -> R2 -> Manage R2 API Tokens) -- deliberately narrower than the general
  `CLOUDFLARE_API_TOKEN` used for Pages deploys, and never full account/admin access.
- Required GitHub secrets: `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` (from that R2 API
  token). `R2_ACCOUNT_ID` is NOT a separate secret -- R2 shares the same Cloudflare
  account as Pages, so the workflow reuses the existing `CLOUDFLARE_ACCOUNT_ID` secret.
- Required GitHub repository variable: `PROSPECTIVE_ARCHIVE_BUCKET` (the bucket name --
  not sensitive, so a variable rather than a secret, matching `PROSPECTIVE_AUTO_DEPLOY`).
- Implementation: `boto3` (the `archive` extra, `pip install -e ".[archive]"`) against
  R2's S3-compatible endpoint (`https://<account_id>.r2.cloudflarestorage.com`) -- a
  library rather than a CLI specifically so the write-once/identity-comparison logic
  could be unit-tested against a plain in-memory fake
  (`tests/test_archive_snapshot.py`) without contacting real Cloudflare services or
  needing a mocking library.
- The bucket (`contact-luck-prospective-archive`) has been created and is in active use.
  It was first validated with a real R2 integration test (archive, remote hash
  verification, idempotent-rerun check, and a restore into an isolated temp directory that
  passed the real dashboard integrity validator) using the existing, already-official
  `2026-08-09` snapshot -- no new date was scored to test this, and nothing was deployed.
  Every pre-existing local snapshot (`2026-08-05`, `2026-08-06`, `2026-08-08`,
  `2026-08-08__refreshed`, `2026-08-09`) was subsequently archived for real, and history
  sync was independently verified against the real bucket with real multi-date history --
  restoring all five into a fully empty simulated fresh-runner directory and confirming
  the dashboard's own precedence logic picks `2026-08-08__refreshed` for that date's trend
  point (both Aug 8 variants stay archived for auditability; Aug 7 correctly has no
  entry).

**Known gap this does not solve**: every CI run still starts with a cold local
Statcast/game-metadata cache (`data/prospective/2026/` is gitignored and ephemeral on the
runner, same as before archival existed) -- durable archival of the SCORED OUTPUT does not
change that every scheduled run currently re-downloads the season-to-date raw data from
scratch. Worth solving eventually, but kept out of scope here deliberately so this change
stays focused on output durability/auditability.
