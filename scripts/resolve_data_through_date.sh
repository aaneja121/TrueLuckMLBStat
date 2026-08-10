#!/usr/bin/env bash
#
# Resolves "the previous MLB calendar date in America/New_York" -- used by
# .github/workflows/publish-prospective.yml when no explicit --data-through
# date is supplied (scheduled runs, or a manual run with the date input
# left blank).
#
# Extracted into its own script, rather than inlined in the workflow YAML,
# specifically so this logic can be exercised directly by
# tests/test_resolve_data_through_date.py without running the workflow,
# the prospective scoring pipeline, or a real Cloudflare deploy.
#
# WHY America/New_York AND NOT UTC:
# MLB's "calendar day" runs on America/New_York wall-clock time, not UTC --
# a late West Coast game can still be in progress well past midnight UTC,
# and a run that fires in the early UTC morning can actually be evening of
# the PREVIOUS day in New York. Resolving "yesterday" via the system tzdata
# (rather than a fixed UTC offset) makes this correct across the EST/EDT
# daylight-saving boundary automatically -- see the test suite for the
# specific boundary cases this was verified against.
#
# GNU DATE ONLY: this script uses GNU coreutils `date --date=... yesterday`
# syntax, as provided by the `ubuntu-latest` GitHub Actions runner. It does
# NOT work with BSD/macOS `date`.
#
# TZDATA MUST BE PRESENT: during development, testing this against a bare
# `ubuntu:24.04` container (which, unlike a real Ubuntu install/GH Actions
# runner, does NOT ship the `tzdata` package by default) produced silently
# WRONG dates -- `TZ="America/New_York"` with no matching zoneinfo file
# doesn't error, it just fails to convert, so a stale/inconsistent result
# comes back with no indication anything was wrong. Verified correct with
# `tzdata` actually installed, across two different coreutils versions
# (9.4 -- matching Ubuntu 24.04 -- and 9.7). To fail loudly instead of
# repeating that mistake, this script checks the zoneinfo file exists
# before relying on it.
#
# Usage:
#   scripts/resolve_data_through_date.sh
#       -> yesterday in America/New_York, relative to the real current time.
#
#   REFERENCE_NOW="2026-03-09 12:00" scripts/resolve_data_through_date.sh
#       -> yesterday in America/New_York, relative to a FIXED reference
#          instant instead of the real current time. Exists purely so tests
#          can pin specific instants (e.g. either side of a DST boundary)
#          and assert a deterministic result -- REFERENCE_NOW is never set
#          by the workflow itself.

set -euo pipefail

# Overridable only so tests/test_resolve_data_through_date.py can point
# this at a path that doesn't exist and assert the guard actually fires --
# never set in real use (the workflow never sets it), so production always
# checks the real zoneinfo file.
ZONEINFO_NEW_YORK="${ZONEINFO_NEW_YORK:-/usr/share/zoneinfo/America/New_York}"

if [[ ! -e "$ZONEINFO_NEW_YORK" ]]; then
  echo "error: $ZONEINFO_NEW_YORK not found -- the tzdata package is" >&2
  echo "missing, so a TZ=America/New_York conversion cannot be trusted (it" >&2
  echo "fails silently rather than erroring). Install tzdata." >&2
  exit 1
fi

REFERENCE_NOW="${REFERENCE_NOW:-now}"

TZ="America/New_York" date --date="${REFERENCE_NOW} yesterday" +%Y-%m-%d
