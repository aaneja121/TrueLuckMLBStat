"""Version 1.4.0 Phase 2: play-ledger scale/reconciliation measurement prototype.

READ-ONLY research/measurement tooling, not a committed-artifact generator
(unlike `demo/build_demo_fixture.py`/`demo/build_counterfactual_grid.py`,
this script writes nothing into `dashboard/` and produces no artifact any
other code reads). It exists to answer, with real numbers instead of
estimates, Phase 2's Decisions 2/6/7: candidate canonical/web-artifact
sizes, and whether a canonical play ledger reconciles exactly to the
existing player-season aggregates.

Trains the frozen contact model on `TRAIN_SEASONS` (2021-2023) and exports
a play ledger for `VALIDATION_SEASONS` (2024) -- approved development data
only, the same seasons this repository's Version 0.7 line has already used
for validation throughout its documented history (see CLAUDE.md's "Version
0.7 is now FROZEN" note: 2024 is validation-quality data for exactly this
kind of measurement, never 2025, never a 2026 rescore). Skips the
outfield/infield/advancement component models entirely -- `play_ledger_
schema`'s v1.4.0 contract has no component-attribution fields, so
`build_attribution_ledger` is called with only the contact model, matching
what a canonical ledger export actually needs.

Run: `.venv/bin/python demo/measure_play_ledger_scale.py`. Reads
`data/processed/cleaned_development_data.parquet` (local-only, gitignored,
never committed). Writes nothing outside a scratch temp directory (for the
size measurements themselves) and prints a report to stdout.
"""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlb_luck_score.config import TRAIN_SEASONS, VALIDATION_SEASONS  # noqa: E402
from mlb_luck_score.data.clean_batted_balls import build_event_id  # noqa: E402
from mlb_luck_score.eligibility import (  # noqa: E402
    add_advancement_eligibility,
    add_infield_opportunity_eligibility,
    add_outfield_opportunity_eligibility,
    compute_eligibility,
)
from mlb_luck_score.models.train_contact_model import train_model  # noqa: E402
from mlb_luck_score.scoring.aggregate_attribution import aggregate_to_batter_season  # noqa: E402
from mlb_luck_score.scoring.attribution_ledger import build_attribution_ledger  # noqa: E402
from mlb_luck_score.scoring.component_confidence import build_play_level_confidence  # noqa: E402
from mlb_luck_score.scoring.play_ledger_export import build_play_ledger  # noqa: E402
from mlb_luck_score.scoring.play_ledger_metadata import (  # noqa: E402
    build_play_ledger_metadata,
    validate_play_ledger_metadata,
)
from mlb_luck_score.scoring.play_ledger_schema import (  # noqa: E402
    PLAY_LEDGER_COLUMNS,
    resolved_rows,
    validate_play_ledger,
)

DEV_DATA_PATH = REPO_ROOT / "data" / "processed" / "cleaned_development_data.parquet"

#: The compact search-index field set (Decision 7) -- enough for browse/
#: filter/sort, not the full public play object. `batter_name` is
#: DELIBERATELY excluded (Phase 2 contract amendment, Section 3): it is a
#: presentation overlay, not a canonical field on `play_ledger`; a future
#: web search index would join it in at the presentation layer.
SEARCH_INDEX_COLUMNS: tuple[str, ...] = (
    "play_id",
    "game_pk",
    "game_date",
    "batter_id",
    "outcome_class",
    "launch_speed",
    "launch_angle",
    "contact_luck_runs",
    "expected_run_value",
)


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    if not DEV_DATA_PATH.exists():
        raise SystemExit(
            f"{DEV_DATA_PATH} not found -- this script requires the local, gitignored "
            "cleaned development parquet (2021-2024 approved development data). "
            "Never invented/downloaded here; see README.md's frozen-input-bundle section."
        )

    print(f"=== Loading {DEV_DATA_PATH.name} ===")
    full = pd.read_parquet(DEV_DATA_PATH)
    print(f"total rows (all development seasons): {len(full)}")

    train_df = full[full["season"].isin(TRAIN_SEASONS)].copy()
    score_season = VALIDATION_SEASONS[0]
    score_df = full[full["season"] == score_season].copy()
    print(f"TRAIN_SEASONS={TRAIN_SEASONS}: {len(train_df)} rows")
    print(f"scoring season={score_season} (VALIDATION_SEASONS[0]): {len(score_df)} rows")

    print("\n=== Eligibility + event_id ===")
    train_df = compute_eligibility(train_df)
    score_df = compute_eligibility(score_df)
    # Opportunity/advancement ELIGIBILITY flags only (cheap, no model
    # training) -- needed by build_play_level_confidence's column
    # requirements below; the corresponding component MODELS are
    # deliberately not trained (outfield_trained=None etc. passed to
    # build_attribution_ledger), since v1.4.0's play_ledger schema has no
    # component-attribution fields to populate from them.
    score_df = add_outfield_opportunity_eligibility(score_df)
    score_df = add_infield_opportunity_eligibility(score_df)
    score_df = add_advancement_eligibility(score_df)
    score_df["event_id"] = build_event_id(score_df)
    score_df["game_date"] = score_df["game_date"].astype(str)
    n_resolved = score_df["outcome_class"].notna().sum()
    n_ambiguous = score_df["is_eligible"].sum() - n_resolved
    print(f"scoring season is_eligible rows: {int(score_df['is_eligible'].sum())}")
    print(f"  resolved (outcome_class not null): {int(n_resolved)}")
    print(f"  ambiguous (is_eligible=True, outcome_class=None): {int(n_ambiguous)}")

    print("\n=== Training frozen contact model on TRAIN_SEASONS (contact model only -- ===")
    print("=== no outfield/infield/advancement; v1.4.0 schema has no component fields) ===")
    t0 = time.time()
    train_elig = train_df[train_df["eligible_for_training"].fillna(False)]
    contact_trained = train_model(train_elig, class_weight=None)
    print(f"trained in {time.time() - t0:.1f}s on {len(train_elig)} rows")

    print("\n=== Scoring + building the real attribution ledger ===")
    # Version 1.4.0 Phase 3: `ledger` now carries native p_* columns
    # (`attribution_ledger.py`'s own single predict_proba_ordered call,
    # never discarded) -- no second prediction call needed here anymore.
    ledger = build_attribution_ledger(score_df, contact_trained)
    confidence = build_play_level_confidence(score_df, ledger)

    print(
        "\n=== Decision 6: reconciliation against the REAL, unmodified aggregate_to_batter_season ==="
    )
    summary = aggregate_to_batter_season(score_df, ledger, confidence)
    play_ledger = build_play_ledger(score_df, ledger)
    validate_play_ledger(play_ledger)

    metadata = build_play_ledger_metadata(
        play_ledger,
        season=score_season,
        score_version="0.2",
        model_versions={"contact": contact_trained.variant},
    )
    validate_play_ledger_metadata(metadata, play_ledger)
    print(f"play_ledger_metadata: {json.dumps(metadata, indent=2)}")

    print(f"play_ledger total rows (all is_eligible, canonical): {metadata['row_count']}")
    print(f"play_ledger scored rows (is_scored=True): {metadata['scored_row_count']}")
    print(f"play_ledger unresolved rows (is_scored=False): {metadata['unresolved_row_count']}")
    assert metadata["row_count"] == len(play_ledger), "no eligible row may silently disappear"
    assert metadata["scored_row_count"] + metadata["unresolved_row_count"] == metadata["row_count"]
    resolved = resolved_rows(play_ledger)
    assert len(resolved) == metadata["scored_row_count"]

    per_batter_ledger_count = resolved.groupby("batter_id").size()
    per_batter_ledger_sum = resolved.groupby("batter_id")["contact_luck_runs"].sum()
    per_batter_published_count = summary.set_index("batter")["eligible_batted_balls"]
    per_batter_published_sum = summary.set_index("batter")["total_observed_minus_expected_runs"]

    joined_count = pd.DataFrame(
        {"ledger": per_batter_ledger_count, "published": per_batter_published_count}
    ).fillna(0)
    count_mismatches = joined_count[joined_count["ledger"] != joined_count["published"]]
    print(f"batters with eligible_batted_balls count MISMATCH: {len(count_mismatches)}")
    if len(count_mismatches):
        print(count_mismatches.head(10))

    joined_sum = pd.DataFrame(
        {"ledger": per_batter_ledger_sum, "published": per_batter_published_sum}
    ).fillna(0.0)
    sum_diff = (joined_sum["ledger"] - joined_sum["published"]).abs()
    sum_mismatches = joined_sum[sum_diff > 1e-6]
    print(f"batters with total Contact Luck sum MISMATCH (atol=1e-6): {len(sum_mismatches)}")
    if len(sum_mismatches):
        print(sum_mismatches.head(10))

    # Runs/100 reconciliation.
    per_batter_published_per100 = summary.set_index("batter")["observed_minus_expected_per_100"]
    reconstructed_per100 = 100.0 * joined_sum["ledger"] / joined_count["ledger"].replace(0, np.nan)
    per100_diff = (reconstructed_per100 - per_batter_published_per100).abs()
    per100_mismatches = per100_diff[per100_diff > 1e-6]
    print(f"batters with Runs/100 MISMATCH (atol=1e-6): {len(per100_mismatches.dropna())}")

    # Scored Games reconstruction -- Section 7 requires proving the actual
    # SET of scored game_pk values matches, not merely the set SIZE (a size
    # match alone could hide two batters each missing a different game).
    per_batter_ledger_games = resolved.groupby("batter_id")["game_pk"].nunique()
    per_batter_published_games = summary.set_index("batter")["games"]
    joined_games = pd.DataFrame(
        {"ledger": per_batter_ledger_games, "published": per_batter_published_games}
    ).fillna(0)
    games_mismatches = joined_games[joined_games["ledger"] != joined_games["published"]]
    print(f"batters with Scored Games count MISMATCH: {len(games_mismatches)}")

    ledger_game_sets = resolved.groupby("batter_id")["game_pk"].agg(lambda s: frozenset(s.tolist()))
    # aggregate_to_batter_season doesn't expose the raw game_pk SET directly
    # (only the count), so re-derive the "actual published scored-game
    # subset" independently from the SAME resolved-play definition it uses
    # internally (observed_contact_result_run_value.notna()), never trusting
    # the ledger's own resolved subset alone -- this is the cross-check.
    published_resolved_mask = ledger["observed_contact_result_run_value"].notna()
    published_scored_plays = score_df.loc[published_resolved_mask, ["batter", "game_pk"]]
    published_game_sets = published_scored_plays.groupby("batter")["game_pk"].agg(
        lambda s: frozenset(s.tolist())
    )
    joined_game_sets = pd.DataFrame({"ledger": ledger_game_sets, "published": published_game_sets})
    joined_game_sets["ledger"] = joined_game_sets["ledger"].apply(
        lambda v: v if isinstance(v, frozenset) else frozenset()
    )
    joined_game_sets["published"] = joined_game_sets["published"].apply(
        lambda v: v if isinstance(v, frozenset) else frozenset()
    )
    set_mismatches = joined_game_sets[joined_game_sets["ledger"] != joined_game_sets["published"]]
    print(f"batters with Scored Games EXACT SET mismatch (not just count): {len(set_mismatches)}")
    assert len(set_mismatches) == 0, (
        "the scored-game SET, not just its size, must reconcile exactly"
    )

    print("\n=== Decision 2/7: candidate serialization sizes (real season data) ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        parquet_path = tmp / "play_ledger.parquet"
        play_ledger.to_parquet(parquet_path, index=False)
        parquet_size = parquet_path.stat().st_size

        csv_path = tmp / "play_ledger.csv"
        play_ledger.to_csv(csv_path, index=False)
        csv_size = csv_path.stat().st_size

        # Compact JSON: array-of-arrays (no repeated field names), the
        # tightest realistic JSON representation.
        compact_json_path = tmp / "play_ledger_compact.json"
        records = play_ledger[list(PLAY_LEDGER_COLUMNS)].to_numpy().tolist()
        compact_json_path.write_text(json.dumps(records, separators=(",", ":"), default=str))
        compact_json_size = compact_json_path.stat().st_size

        # Verbose JSON: one dict per row (field names repeated every row) --
        # the naive representation this measurement recommends AGAINST.
        verbose_json_path = tmp / "play_ledger_verbose.json"
        verbose_json_path.write_text(play_ledger.to_json(orient="records", date_format="iso"))
        verbose_json_size = verbose_json_path.stat().st_size

        for label, path, size in (
            ("Parquet (typed, canonical)", parquet_path, parquet_size),
            ("CSV", csv_path, csv_size),
            ("Compact JSON (array-of-arrays)", compact_json_path, compact_json_size),
            ("Verbose JSON (one dict/row)", verbose_json_path, verbose_json_size),
        ):
            gz_size = len(gzip.compress(path.read_bytes()))
            print(
                f"  {label}: {_human_bytes(size)} raw, {_human_bytes(gz_size)} gzipped "
                f"({size / len(play_ledger):.0f} bytes/row raw)"
            )

        print("\n=== Decision 7: compact search-index size ===")
        index_df = play_ledger[list(SEARCH_INDEX_COLUMNS)]
        index_json_path = tmp / "search_index.json"
        index_records = index_df.to_numpy().tolist()
        index_json_path.write_text(json.dumps(index_records, separators=(",", ":"), default=str))
        index_size = index_json_path.stat().st_size
        index_gz = len(gzip.compress(index_json_path.read_bytes()))
        print(
            f"  search index ({len(SEARCH_INDEX_COLUMNS)} fields, {len(index_df)} rows): "
            f"{_human_bytes(index_size)} raw, {_human_bytes(index_gz)} gzipped "
            f"({index_size / len(index_df):.0f} bytes/row raw)"
        )

    print("\n=== Decision 7: per-game (game_pk) partitioning measurements ===")
    per_game_counts = play_ledger.groupby("game_pk").size()
    print(f"distinct games: {len(per_game_counts)}")
    print(
        f"records per game -- mean={per_game_counts.mean():.1f}, "
        f"median={per_game_counts.median():.1f}, "
        f"p95={per_game_counts.quantile(0.95):.1f}, "
        f"max={per_game_counts.max()}"
    )
    # Estimate per-game detail-file size using the measured bytes/row from
    # the compact JSON representation above.
    bytes_per_row = compact_json_size / len(play_ledger)
    per_game_bytes = per_game_counts * bytes_per_row
    print(
        f"estimated per-game detail file size (compact JSON) -- "
        f"mean={_human_bytes(per_game_bytes.mean())}, "
        f"median={_human_bytes(per_game_bytes.median())}, "
        f"p95={_human_bytes(per_game_bytes.quantile(0.95))}, "
        f"max={_human_bytes(per_game_bytes.max())}"
    )

    print("\n=== Decision 9: full-snapshot-scale extrapolation ===")
    print(
        f"this single VALIDATION_SEASONS season ({score_season}) has {len(play_ledger)} play rows"
    )
    print(f"canonical Parquet for this one season: {_human_bytes(parquet_size)}")
    print(
        "(compare to a 2026-to-date snapshot's real ~93,198-row scale measured directly "
        "against the live prospective pipeline's own real numbers in the Phase 2 report)"
    )


if __name__ == "__main__":
    main()
