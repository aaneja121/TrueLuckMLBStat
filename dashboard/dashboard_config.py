"""Contact Luck v1.2: shared path/constant configuration for the dashboard.

The dashboard is a READ-ONLY presentation layer over immutable Version 1.1
prospective snapshots (see CLAUDE.md "Version 1.1: prospective 2026
scoring"). This module intentionally imports nothing from `prospective/`,
`evaluation/`, or `src/mlb_luck_score` -- every path/constant it needs is
redefined here so the dashboard's read-only boundary is enforced by having
nothing scoring-related to import, not merely by convention. See
`tests/test_dashboard_isolation.py` for the structural check that
verifies no module under `dashboard/` imports model-training, prospective
-scoring-orchestration, final-evaluation, or Statcast-download code.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Same namespace Version 1.1 writes to (`prospective.prospective_config`) --
# duplicated here rather than imported, per this module's docstring.
PROSPECTIVE_OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "prospective" / "v1_1"
PROSPECTIVE_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "prospective" / "v1_1"

DASHBOARD_SOURCE_ROOT = Path(__file__).resolve().parent
DASHBOARD_TEMPLATES_DIR = DASHBOARD_SOURCE_ROOT / "templates"
DASHBOARD_STATIC_DIR = DASHBOARD_SOURCE_ROOT / "static"
DASHBOARD_DIST_DIR = DASHBOARD_SOURCE_ROOT / "dist"

DASHBOARD_VERSION = "1.3.1"

# Committed, hand-reviewed reference data (like `mlb_luck_score.data.
# game_metadata_overrides`) produced ONCE, offline, by `demo/
# build_demo_fixture.py` -- see that module's docstring. Dashboard code only
# ever READS this file; nothing under `dashboard/` regenerates it, and
# `tests/test_dashboard_isolation.py` enforces that no module here can even
# import the generator (which itself imports `mlb_luck_score`).
DEMO_FIXTURE_PATH = DASHBOARD_SOURCE_ROOT / "demo_fixture.json"

# Version 1.3.1: the "Try It Yourself" counterfactual EV/launch-angle grid --
# same committed-reference-data convention as DEMO_FIXTURE_PATH above,
# produced ONCE, offline, by `demo/build_counterfactual_grid.py`. Dashboard
# code only ever reads and copies this file (into dist/demo/ for the
# browser to fetch once) -- never regenerates or recomputes it.
DEMO_COUNTERFACTUAL_GRID_PATH = DASHBOARD_SOURCE_ROOT / "demo_counterfactual_grid.json"

# Reserved snapshot-label value for a future, separately-built retrospective
# -backfill mechanism (not yet implemented as of this writing -- see the
# 2026-08-07 gap discussion). An ordinary rerun of
# `prospective/run_v1_1_2026_scoring.py --snapshot-label
# retrospective_backfill` would misuse this reserved name; enforcing that is
# outside the dashboard's read-only scope (it belongs in the Version 1.1
# ingestion/orchestration code if ever added there). The dashboard only
# relies on the convention to CLASSIFY a snapshot it finds on disk -- see
# `classify_snapshot_type()` in `snapshot_data.py`.
RETROSPECTIVE_BACKFILL_LABEL = "retrospective_backfill"

FROZEN_PUBLIC_TERMINOLOGY = {
    "favorable": "Most favorable realized luck",
    "unfavorable": "Least favorable outcomes relative to expectation",
}
