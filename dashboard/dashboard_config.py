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

DASHBOARD_VERSION = "1.4.1"

# Production canonical domain, used for absolute URLs in Open Graph/Twitter
# card metadata (`<meta property="og:url">`, `<meta property="og:image">`,
# etc.) -- those tags must be absolute, not relative, per the OG spec. Never
# the old `contact-luck.pages.dev` Cloudflare Pages preview domain.
SITE_URL = "https://contactluck.com"

# A committed, hand-reviewed real screenshot of the production homepage
# (cropped/resized only -- no AI-generated or reconstructed artwork), used
# as the Open Graph/Twitter-card social preview image. Copied verbatim to
# `dist/og-image.png` (see `build.py`) so it serves at `{SITE_URL}/og-image.png`,
# matching the `og:image`/`twitter:image` URLs above.
OG_IMAGE_PATH = DASHBOARD_SOURCE_ROOT / "og-image.png"

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

# Version 1.4.0 Phase 4 (sharded browser artifacts since Phase 4.2): the
# Play Explorer's committed, bounded, real-2024-data browser artifacts --
# same committed-reference-data convention as DEMO_FIXTURE_PATH/
# DEMO_COUNTERFACTUAL_GRID_PATH above, produced ONCE, offline, by `demo/
# build_play_explorer_fixture.py` (a projection-only generator that itself
# requires its INPUT canonical ledger to have `play_ledger_version ==
# "2.0"`, fail-closed) fed by `demo/build_play_explorer_dev_ledger.py`
# (which trains real models on real 2021-2023/2024 development data --
# local dev only, never 2025, never a prospective 2026 run). Dashboard code
# only ever reads and copies these files -- never regenerates, rescoring,
# or recomputes any value in them.
#
# Phase 4.2 replaced the single monolithic `search-index.json` (~29.3 MiB
# at full 2024 development-data scale -- over Cloudflare Pages' 25 MiB
# per-asset limit) with a small `players.json` catalog plus one
# `players/<batter_id>.json` file per batter, so the Explorer only ever
# fetches one hitter's plays at a time -- see `demo/
# build_play_explorer_fixture.py`'s module docstring.
EXPLORE_PLAYERS_PATH = DASHBOARD_SOURCE_ROOT / "explore_fixture" / "players.json"
EXPLORE_PLAYERS_DIR = DASHBOARD_SOURCE_ROOT / "explore_fixture" / "players"
EXPLORE_GAMES_DIR = DASHBOARD_SOURCE_ROOT / "explore_fixture" / "games"
# Phase 4.1: file-level provenance sidecar (never repeated per row) -- see
# `demo/build_play_explorer_fixture.build_explore_metadata`/`dashboard/
# explore_content.load_explore_metadata` for the SECOND, independent
# fail-closed `play_ledger_version`/`explorer_artifact_version` check this
# enables at build time, after the generator's own.
EXPLORE_METADATA_PATH = DASHBOARD_SOURCE_ROOT / "explore_fixture" / "explore-metadata.json"

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


# The committed Pitcher Contact Luck season fixture -- the published
# development fixture -- same committed-reference-data convention as
# DEMO_FIXTURE_PATH / EXPLORE_PLAYERS_PATH above, produced ONCE, offline,
# by `demo/build_pitcher_prototype_fixture.py` (which trains the frozen
# models on real 2021-2023 development data and scores 2024 -- local dev
# only, never 2025, never a prospective 2026 run). Dashboard code only ever
# READS this file; nothing under `dashboard/` regenerates or recomputes any
# value in it.
#
# Like EXPLORE_PLAYERS_PATH and unlike DEMO_FIXTURE_PATH, this path is NOT a
# default for `build.py`: the pitcher route is fail-closed and must be opted
# into with `--pitcher-season-fixture PATH`. `scripts/publish_snapshot.sh`
# now passes it explicitly, exactly as it already passes
# `--explore-artifacts-dir` -- so the surface is published deliberately, by
# one named line in the publish path, and an ad-hoc `build.py` run still
# emits no pitcher route at all.
PITCHER_SEASON_FIXTURE_PATH = DASHBOARD_SOURCE_ROOT / "pitcher_season_fixture.json"

#: THE PUBLICATION GATE for pitcher seasons. This tuple -- and nothing else
#: -- decides which pitcher seasons reach the public site.
#:
#: A season is in this tuple ONLY when the maintainer has explicitly
#: authorized publishing it, recorded in `RESEARCH_RULES.md` ("Public launch
#: of the 2024 pitcher surface"). It is never widened because a fixture for
#: another season happens to exist, because a season "looks ready", or
#: because the loader would accept it: the fixture is data, this is
#: permission, and they are deliberately two different things.
#:
#: Why the tuple rather than "whatever the fixture says": the fixture
#: carries its own `season` field, so without a gate, handing `build.py` a
#: different fixture would silently publish a different season. `build.py`
#: refuses to render any pitcher season absent from this tuple, which is
#: what makes "2024 only" a property of the build rather than a habit.
#:
#: 2025 is sealed final-evaluation data whose replication authorization
#: explicitly withholds deploying; 2021-2023 are `TRAIN_SEASONS`, so a board
#: on them would present in-sample fitted values as measurement; 2026 is
#: prospective and unopened. Each would need its own separate sign-off, and
#: publishing any of them is a research/governance decision, never a
#: dashboard one.
PITCHER_PUBLIC_SEASONS: tuple[int, ...] = (2024,)
