#!/usr/bin/env bash
#
# Contact Luck operational loop: ensure every gitignored frozen input
# artifact is present, generate a new Version 1.1 prospective snapshot for
# a completed MLB date, durably archive it, repopulate any OTHER historical
# snapshots missing on this machine from the durable archive, generate Play
# Explorer browser artifacts from that SAME snapshot, rebuild the Version
# 1.4 static dashboard (with the Explorer wired in) from the complete local
# history, and deploy the result to Cloudflare Pages.
#
#     completed MLB slate
#         -> scripts/ensure_frozen_inputs.py         (frozen input bundle:
#                                                      the 2021-2024 dev
#                                                      parquet + 3 model
#                                                      -comparison detail
#                                                      JSONs. Each: reuse
#                                                      local if hash-valid,
#                                                      else fetch from R2
#                                                      and verify)
#         -> prospective/run_v1_1_2026_scoring.py   (immutable snapshot)
#         -> scripts/archive_snapshot.py             (durable R2 archive)
#         -> scripts/archive_snapshot.py --sync-history
#                                                     (repopulate missing
#                                                      local history from R2)
#         -> scripts/generate_production_explorer_artifacts.py
#                                                     (Play Explorer browser
#                                                      artifacts, projected
#                                                      from THIS run's own
#                                                      local canonical
#                                                      outputs -- no
#                                                      re-download, no
#                                                      rescoring -- written
#                                                      to an ephemeral build
#                                                      dir, never into the
#                                                      canonical snapshot)
#         -> dashboard/build.py --explore-artifacts-dir <that ephemeral dir>
#                                                     (dashboard/dist/)
#         -> wrangler pages deploy                  (static hosting)
#
# WHY THE FROZEN-INPUT STAGE EXISTS: a GitHub Actions runner starts from a
# clean git checkout with no data/processed/ or outputs/tables/ cache at
# all. Version 1.1's manifest freezes its inputs by hashing every path in
# evaluation.v1_final_evaluation_manifest.FROZEN_ARTIFACT_RELATIVE_PATHS --
# 27 of those 32 paths are git-tracked source files (present on any fresh
# checkout automatically), but 4 are gitignored, local-only data/detail
# files (CLAUDE.md "Never commit datasets") that have only ever existed on
# a maintainer's own machine: the 2021-2024 development parquet, plus
# outputs/tables/{opportunity_model_comparison,infield_opportunity,
# advancement}_detail.json. The first scheduled dry-run failed on the
# parquet alone; the second got past scoring and failed building the
# manifest, needing the remaining three. scripts/ensure_frozen_inputs.py
# closes the COMPLETE gap (all four, audited against the real frozen
# -artifact list -- see that script's own module docstring): for each, it
# reuses the local file if present and hash-valid, otherwise fetches it
# from a dedicated, immutable R2 object (a SEPARATE prefix from the
# prospective snapshot archive below) and verifies its sha256 before use --
# it never regenerates or substitutes any of them. This stage ALWAYS runs,
# even under --skip-archive --skip-deploy, because scoring/manifest
# generation always needs the full bundle regardless of what happens to
# the output afterward -- exactly the same reasoning that makes history
# sync always run.
#
# WHY HISTORY SYNC EXISTS: a GitHub Actions runner starts from a fresh git
# checkout -- outputs/prospective/v1_1/ and artifacts/prospective/v1_1/ are
# gitignored, so a CI job that just scored today's date has ONLY today's
# snapshot locally. Without a sync step, dashboard/build.py's season-to-date
# trend charts would show a single point on every CI-built dashboard, no
# matter how much history is sitting in R2. The sync stage restores every
# OTHER archived snapshot that's missing locally before the dashboard is
# built, so a CI-built dashboard sees the same complete history a
# maintainer's own long-lived machine would already have on disk. It never
# overwrites an existing local snapshot -- see scripts/archive_snapshot.py's
# "History sync" docstring section for the exact no-op/conflict rules.
#
# ORDERING: the frozen-input check happens BEFORE scoring (scoring cannot
# proceed without it), archive happens BEFORE history sync, which happens
# BEFORE Explorer artifact generation, which happens BEFORE the dashboard
# build, and no later stage ever runs if ANY earlier stage fails. This is
# deliberate -- the raw scoring output is the precious, hard-to-reproduce
# artifact (re-scoring depends on the same historical Statcast data still
# being fetchable later); the dashboard build is comparatively cheap and
# frequently-iterated. Archiving first means an official snapshot survives
# runner destruction even if something goes wrong in a LATER stage, and it
# means "the public site must never deploy if durable archival failed" is
# satisfied by simple sequential ordering under `set -e`, not by any
# special-cased check. Explorer artifact generation reads ONLY this run's
# own already-scored local outputs (`outputs/prospective/v1_1/<snapshot>/`)
# -- it never re-downloads Statcast or rescores, and it writes to an
# ephemeral build directory under `outputs/explorer_build/`, never into the
# canonical snapshot directory the archive step already wrote to R2.
#
# This script performs no modeling/scoring/build/archival logic of its
# own -- it only invokes the existing, already-guarded entry points and
# stops if any of them fail. In particular it does NOT bypass, duplicate,
# or relax:
#   - the frozen-input check's local-hash-mismatch / missing-remote-object /
#     remote-hash-mismatch guards (never a silent regeneration or a
#     mismatched file left in place)
#   - the prospective runner's clean-working-tree guard (commit or stash
#     your changes first; this script will not do that for you)
#   - the prospective runner's data-through-date-completeness guard
#   - the archive's write-once/identity verification (never silently
#     overwrites a differently-coded snapshot already in R2)
#   - the history sync's never-overwrite-a-local-snapshot guarantee
#   - the dashboard's snapshot integrity/precedence rules
#   - the Explorer generator's own canonical-source-integrity and
#     play_ledger_version=="2.0" fail-closed checks (scripts/
#     generate_production_explorer_artifacts.py)
#
# --skip-archive and --skip-deploy are INDEPENDENTLY controllable, so real
# R2 archival can be validated on its own before production deploys are
# enabled -- see README.md "Durable archival" for the rollout plan this
# supports. The frozen-input check and history sync ALWAYS run (even with
# both skip flags) so a dry run still validates the real CI scoring/
# dashboard-build behavior -- both only ever READ from R2 in the common
# case (the frozen-input check writes to R2 only via the separate, never
# automatically invoked --upload seeding action), so this doesn't
# compromise "dry run touches no external WRITE path." Explorer artifact
# generation and the dashboard build ALWAYS run regardless of these flags
# (same reasoning as history sync -- generation reads only this run's own
# local outputs, no R2 write involved). Their combinations:
#
#   (neither flag)                  ensure -> score -> archive -> sync -> explore -> build -> deploy
#   --skip-deploy                   ensure -> score -> archive -> sync -> explore -> build -> stop
#   --skip-archive --skip-deploy    ensure -> score -> sync (read-only) -> explore -> build -> stop
#   --skip-archive (alone)          REFUSED -- see below.
#
# --skip-archive without --skip-deploy is deliberately refused: this
# script will not deploy a dashboard built from a snapshot that was not
# just durably archived. There is currently no override flag for this --
# if you have a genuinely compelling reason to deploy without archiving,
# that is a decision to make explicitly (e.g. by archiving separately via
# scripts/archive_snapshot.py first), not something this script does
# quietly.
#
# Usage:
#   scripts/publish_snapshot.sh --data-through YYYY-MM-DD [options]
#
# Options:
#   --data-through DATE     Required. Passed straight through to the
#                           prospective runner. Its year is also used as
#                           the --season for archive/history-sync (Version
#                           1.1 is single-season; see scripts/
#                           archive_snapshot.py if that ever changes).
#   --snapshot-label LABEL  Optional. Passed straight through (e.g. for a
#                           deliberate corrected/refreshed rerun of a date
#                           that already has a snapshot).
#   --project-name NAME     Cloudflare Pages project name. Default:
#                           contact-luck
#   --archive-bucket NAME   R2 bucket for durable snapshot archival, history
#                           sync, AND the frozen-input bundle check (same
#                           bucket, separate key prefixes -- see
#                           scripts/ensure_frozen_inputs.py). Default:
#                           $R2_BUCKET_NAME if set, else
#                           contact-luck-prospective-archive.
#   --skip-archive          Do not archive to R2. Refused unless
#                           --skip-deploy is ALSO passed (see above).
#   --skip-deploy           Rebuild the dashboard but do not deploy to
#                           Cloudflare Pages. Archival still happens
#                           unless --skip-archive is also passed. History
#                           sync always happens regardless. Preview
#                           locally with:
#                             cd dashboard/dist && python3 -m http.server 8000
#   --yes                   Skip the interactive deploy confirmation
#                           prompt (for non-interactive use).
#   -h, --help              Show this help and exit.
#
# R2 credentials: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY as
# environment variables -- ALWAYS required now, even for a full dry run
# (--skip-archive --skip-deploy), because history sync always reads from
# R2. The frozen-input bundle check (scripts/ensure_frozen_inputs.py) only
# actually NEEDS these if at least one bundle entry (the development
# parquet or one of the three model-comparison detail JSONs) is missing
# locally -- true on every fresh CI runner, false on a maintainer's own
# machine that already has the full bundle, where this stage never
# touches R2 at all. See scripts/archive_snapshot.py's module docstring
# and README.md's "Durable archival" section for exactly what these need
# to be.
#
# Example:
#   scripts/publish_snapshot.sh --data-through 2026-08-10

set -euo pipefail

usage() {
  # '2,$p' (not a hardcoded end line) -- the header comment block above
  # this function IS the entire usage text, from the line after the
  # shebang through the last comment line before real code starts. A
  # hardcoded upper bound silently truncates this output the next time the
  # header grows (as it just did for the frozen-input stage) without ever
  # failing loudly, so don't reintroduce one.
  grep '^#' "${BASH_SOURCE[0]}" | sed -n '2,$p' | sed 's/^# \{0,1\}//'
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
DATA_THROUGH=""
SNAPSHOT_LABEL=""
PROJECT_NAME="contact-luck"
ARCHIVE_BUCKET="${R2_BUCKET_NAME:-contact-luck-prospective-archive}"
SKIP_ARCHIVE=0
SKIP_DEPLOY=0
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-through)
      DATA_THROUGH="$2"
      shift 2
      ;;
    --snapshot-label)
      SNAPSHOT_LABEL="$2"
      shift 2
      ;;
    --project-name)
      PROJECT_NAME="$2"
      shift 2
      ;;
    --archive-bucket)
      ARCHIVE_BUCKET="$2"
      shift 2
      ;;
    --skip-archive)
      SKIP_ARCHIVE=1
      shift
      ;;
    --skip-deploy)
      SKIP_DEPLOY=1
      shift
      ;;
    --yes)
      ASSUME_YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$DATA_THROUGH" ]]; then
  echo "error: --data-through YYYY-MM-DD is required" >&2
  usage
  exit 1
fi

if [[ "$SKIP_ARCHIVE" -eq 1 && "$SKIP_DEPLOY" -ne 1 ]]; then
  echo "error: --skip-archive cannot be combined with a real deploy." >&2
  echo "       This script refuses to deploy a dashboard built from a snapshot that was" >&2
  echo "       not just durably archived. Pass --skip-deploy too if you want to skip" >&2
  echo "       both (ensure -> score -> build -> stop), or drop --skip-archive to archive" >&2
  echo "       normally (ensure -> score -> archive -> build -> deploy)." >&2
  exit 1
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "error: $PYTHON not found -- run 'make setup' first" >&2
  exit 1
fi

SNAPSHOT_DIR_NAME="$DATA_THROUGH"
if [[ -n "$SNAPSHOT_LABEL" ]]; then
  SNAPSHOT_DIR_NAME="${DATA_THROUGH}__${SNAPSHOT_LABEL}"
fi

SEASON="${DATA_THROUGH:0:4}"
EXPLORER_BUILD_DIR="$PROJECT_ROOT/outputs/explorer_build/$SNAPSHOT_DIR_NAME"

echo "==> [1/7] Ensuring the frozen input bundle (dev parquet + 3 detail JSONs) is present and verified"
R2_BUCKET_NAME="$ARCHIVE_BUCKET" "$PYTHON" scripts/ensure_frozen_inputs.py

echo "==> [2/7] Generating prospective snapshot for --data-through $DATA_THROUGH"
SNAPSHOT_ARGS=(--data-through "$DATA_THROUGH")
if [[ -n "$SNAPSHOT_LABEL" ]]; then
  SNAPSHOT_ARGS+=(--snapshot-label "$SNAPSHOT_LABEL")
fi
"$PYTHON" prospective/run_v1_1_2026_scoring.py "${SNAPSHOT_ARGS[@]}"

if [[ "$SKIP_ARCHIVE" -eq 1 ]]; then
  echo "==> [3/7] Skipping durable archive (--skip-archive)."
else
  echo "==> [3/7] Archiving snapshot $SNAPSHOT_DIR_NAME to R2 bucket '$ARCHIVE_BUCKET'"
  ARCHIVE_ARGS=(--data-through "$DATA_THROUGH")
  if [[ -n "$SNAPSHOT_LABEL" ]]; then
    ARCHIVE_ARGS+=(--snapshot-label "$SNAPSHOT_LABEL")
  fi
  R2_BUCKET_NAME="$ARCHIVE_BUCKET" "$PYTHON" scripts/archive_snapshot.py "${ARCHIVE_ARGS[@]}"
fi

echo "==> [4/7] Syncing missing historical snapshots from R2 (read-only) for season $SEASON"
R2_BUCKET_NAME="$ARCHIVE_BUCKET" "$PYTHON" scripts/archive_snapshot.py --sync-history --season "$SEASON"

echo "==> [5/7] Generating Play Explorer browser artifacts from snapshot $SNAPSHOT_DIR_NAME"
EXPLORE_ARGS=(--data-through "$DATA_THROUGH" --output-dir "$EXPLORER_BUILD_DIR")
if [[ -n "$SNAPSHOT_LABEL" ]]; then
  EXPLORE_ARGS+=(--snapshot-label "$SNAPSHOT_LABEL")
fi
"$PYTHON" scripts/generate_production_explorer_artifacts.py "${EXPLORE_ARGS[@]}"

echo "==> [6/7] Rebuilding the dashboard"
"$PYTHON" dashboard/build.py --explore-artifacts-dir "$EXPLORER_BUILD_DIR"

if [[ "$SKIP_DEPLOY" -eq 1 ]]; then
  echo "==> [7/7] Skipping deploy (--skip-deploy)."
  echo "    Preview locally with: cd dashboard/dist && python3 -m http.server 8000"
  exit 0
fi

GIT_HEAD="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo
echo "Snapshot:      $SNAPSHOT_DIR_NAME"
echo "Data through:  $DATA_THROUGH"
echo "Dashboard:     dashboard/dist"
echo "Project:       $PROJECT_NAME"
echo "Git HEAD:      $GIT_HEAD"
echo

if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p "Deploy now? [y/N] " reply
  case "$reply" in
    [yY]|[yY][eE][sS])
      ;;
    *)
      echo "Aborted -- dashboard was rebuilt but not deployed."
      exit 0
      ;;
  esac
fi

echo "==> [7/7] Deploying dashboard/dist to Cloudflare Pages project '$PROJECT_NAME'"
npx wrangler pages deploy dashboard/dist --project-name="$PROJECT_NAME" --commit-dirty=true
