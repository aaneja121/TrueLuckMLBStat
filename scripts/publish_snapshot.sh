#!/usr/bin/env bash
#
# Contact Luck operational loop: generate a new Version 1.1 prospective
# snapshot for a completed MLB date, rebuild the Version 1.2 static
# dashboard from it, and deploy the result to Cloudflare Pages.
#
#     completed MLB slate
#         -> prospective/run_v1_1_2026_scoring.py   (immutable snapshot)
#         -> dashboard/build.py                     (dashboard/dist/)
#         -> wrangler pages deploy                  (static hosting)
#
# This script performs no modeling/scoring/build logic of its own -- it
# only invokes the existing, already-guarded entry points and stops if any
# of them fail. In particular it does NOT bypass, duplicate, or relax:
#   - the prospective runner's clean-working-tree guard (commit or stash
#     your changes first; this script will not do that for you)
#   - the prospective runner's data-through-date-completeness guard
#   - the dashboard's snapshot integrity/precedence rules
#
# Usage:
#   scripts/publish_snapshot.sh --data-through YYYY-MM-DD [options]
#
# Options:
#   --data-through DATE     Required. Passed straight through to the
#                           prospective runner.
#   --snapshot-label LABEL  Optional. Passed straight through (e.g. for a
#                           deliberate corrected/refreshed rerun of a date
#                           that already has a snapshot).
#   --project-name NAME     Cloudflare Pages project name. Default:
#                           contact-luck
#   --skip-deploy           Generate the snapshot and rebuild the
#                           dashboard, but do not deploy. Useful as a dry
#                           run -- preview locally with:
#                             cd dashboard/dist && python3 -m http.server 8000
#   --yes                   Skip the interactive deploy confirmation
#                           prompt (for non-interactive use).
#   -h, --help              Show this help and exit.
#
# Example:
#   scripts/publish_snapshot.sh --data-through 2026-08-10

set -euo pipefail

usage() {
  grep '^#' "${BASH_SOURCE[0]}" | sed -n '2,44p' | sed 's/^# \{0,1\}//'
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
DATA_THROUGH=""
SNAPSHOT_LABEL=""
PROJECT_NAME="contact-luck"
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

if [[ ! -x "$PYTHON" ]]; then
  echo "error: $PYTHON not found -- run 'make setup' first" >&2
  exit 1
fi

echo "==> [1/3] Generating prospective snapshot for --data-through $DATA_THROUGH"
SNAPSHOT_ARGS=(--data-through "$DATA_THROUGH")
if [[ -n "$SNAPSHOT_LABEL" ]]; then
  SNAPSHOT_ARGS+=(--snapshot-label "$SNAPSHOT_LABEL")
fi
"$PYTHON" prospective/run_v1_1_2026_scoring.py "${SNAPSHOT_ARGS[@]}"

echo "==> [2/3] Rebuilding the dashboard"
"$PYTHON" dashboard/build.py

if [[ "$SKIP_DEPLOY" -eq 1 ]]; then
  echo "==> [3/3] Skipping deploy (--skip-deploy)."
  echo "    Preview locally with: cd dashboard/dist && python3 -m http.server 8000"
  exit 0
fi

SNAPSHOT_DIR_NAME="$DATA_THROUGH"
if [[ -n "$SNAPSHOT_LABEL" ]]; then
  SNAPSHOT_DIR_NAME="${DATA_THROUGH}__${SNAPSHOT_LABEL}"
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

echo "==> [3/3] Deploying dashboard/dist to Cloudflare Pages project '$PROJECT_NAME'"
npx wrangler pages deploy dashboard/dist --project-name="$PROJECT_NAME" --commit-dirty=true
