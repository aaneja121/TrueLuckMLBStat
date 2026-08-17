"""Version 1.4.0 Phase 4: builds the committed, BOUNDED Play Explorer local
-development fixture from real 2021-2023/2024 development data.

This is a ONE-TIME (or occasionally-rerun) offline generation step, run
manually by a maintainer -- never by `dashboard/build.py`, never by CI, and
never as part of `make check`. It lives outside `dashboard/` for the exact
same reason `demo/build_demo_fixture.py`/`demo/build_counterfactual_grid.py`
do (see their docstrings and `tests/test_dashboard_isolation.py`): it trains
real frozen models, which `dashboard/` is structurally forbidden from ever
importing.

## What this script does NOT do

- Never touches 2025 (sealed final-evaluation data) or 2026 (prospective
  scoring) -- trains on `TRAIN_SEASONS` (2021-2023) and scores
  `VALIDATION_SEASONS[0]` (2024) only, exactly like `demo/
  measure_play_ledger_full_stack_reconciliation.py` (Phase 3.1), whose
  four-model training glue this script duplicates rather than imports
  (same established convention as `evaluation/run_v1_final_evaluation.py`/
  `prospective/prospective_scoring.py` duplicating each other's glue).
- Never runs `prospective/run_v1_1_2026_scoring.py` or archives/deploys
  anything.
- Never modifies model fitting, scoring formulas, or `public_score`
  aggregation -- this is a SCORING step (like a real prospective run would
  be), followed by a strictly PROJECTION-ONLY step (`demo/
  build_play_explorer_fixture.py`, imported and called directly here,
  never reimplemented) that turns the resulting canonical ledger into
  browser artifacts.

## Network access

Resolving real display names for the sampled batters calls
`mlb_luck_score.data.download_player_names.fetch_player_names`, which
hits the public MLB Stats API (`/people`) -- the SAME mechanism `prospective/
run_v1_1_2026_scoring.py` already uses in production, over a small (dozens)
number of batter IDs. This script prints a clear notice before doing so.

## Bounded sample

The full real 2024 canonical ledger (~124k rows) is never committed --
only a deliberately bounded, representative sample (the most favorable and
most unfavorable resolved plays, plus a fixed-seed random sample for
variety) is selected BEFORE the browser-artifact generator ever sees it, so
`dashboard/explore_fixture/` stays small and reviewable, matching the
`dashboard/demo_fixture.json` precedent. The generator itself (`demo/
build_play_explorer_fixture.py`) places no limit on ledger size -- it is
equally capable of processing a full canonical season ledger; only the
INPUT here is intentionally bounded.

Usage (requires the full 2021-2024 development dataset -- see README.md
"Full development dataset"; real 2021-2024 data is local-only, gitignored,
never committed; requires network access for name resolution only):

    .venv/bin/python demo/build_play_explorer_dev_ledger.py

Writes `dashboard/explore_fixture/players.json`,
`dashboard/explore_fixture/players/<batter_id>.json`,
`dashboard/explore_fixture/games/<game_pk>.json`, and
`dashboard/explore_fixture/explore-metadata.json` (Phase 4.2's sharded
layout -- see `demo/build_play_explorer_fixture.py`'s module docstring),
which ARE committed to git as reviewed reference data (same convention as
`dashboard/demo_fixture.json`).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "demo"))

import build_play_explorer_fixture as explorer_gen  # noqa: E402

from mlb_luck_score.config import TRAIN_SEASONS, VALIDATION_SEASONS  # noqa: E402
from mlb_luck_score.data.clean_batted_balls import build_event_id  # noqa: E402
from mlb_luck_score.data.download_player_names import fetch_player_names  # noqa: E402
from mlb_luck_score.eligibility import (  # noqa: E402
    add_advancement_eligibility,
    add_infield_opportunity_eligibility,
    add_outfield_opportunity_eligibility,
    compute_eligibility,
)
from mlb_luck_score.features.build_contact_features import (  # noqa: E402
    ADVANCEMENT_CONTACT_PROBABILITY_FEATURES,
    add_advancement_features,
    add_opportunity_features_by_domain,
)
from mlb_luck_score.models.compare_advancement_models import (  # noqa: E402
    fit_advancement_contact_model,
    get_advancement_rows,
    run_advancement_model_selection,
)
from mlb_luck_score.models.compare_infield_opportunity import (  # noqa: E402
    run_infield_model_selection,
)
from mlb_luck_score.models.train_contact_model import train_model  # noqa: E402
from mlb_luck_score.models.train_opportunity_model import train_opportunity_model  # noqa: E402
from mlb_luck_score.scoring.attribution_ledger import build_attribution_ledger  # noqa: E402
from mlb_luck_score.scoring.play_ledger_export import build_play_ledger  # noqa: E402
from mlb_luck_score.scoring.play_ledger_metadata import build_play_ledger_metadata  # noqa: E402
from mlb_luck_score.scoring.play_ledger_schema import (  # noqa: E402
    resolved_rows,
    validate_play_ledger,
)

DEV_DATA_PATH = REPO_ROOT / "data" / "processed" / "cleaned_development_data.parquet"
OUTPUT_DIR = REPO_ROOT / "dashboard" / "explore_fixture"

#: How many plays to keep from each end of the Contact Luck distribution,
#: plus how many more to keep as a fixed-seed random sample for variety.
N_MOST_FAVORABLE = 18
N_MOST_UNFAVORABLE = 18
N_RANDOM_SAMPLE = 20
RANDOM_SAMPLE_SEED = 20260101


def _apply_eligibility_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_eligibility(df)
    df = add_outfield_opportunity_eligibility(df)
    df = add_infield_opportunity_eligibility(df)
    df = add_advancement_eligibility(df)
    df = add_opportunity_features_by_domain(df)
    df = add_advancement_features(df)
    return df


def _select_bounded_sample(play_ledger: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, representative, BOUNDED subset of a full canonical
    ledger: the most favorable and most unfavorable resolved plays (good
    illustrative extremes for local screenshots/UI review), plus a fixed
    -seed random sample of otherwise-typical resolved plays for variety.
    Never includes unresolved rows (nothing to show for them in the
    Explorer -- see `build_play_explorer_fixture.py`'s module docstring).
    """
    scored = resolved_rows(play_ledger)
    by_luck = scored.sort_values("contact_luck_runs", kind="stable")
    most_unfavorable = by_luck.head(N_MOST_UNFAVORABLE)
    most_favorable = by_luck.tail(N_MOST_FAVORABLE)

    remaining = scored.drop(index=most_unfavorable.index.union(most_favorable.index))
    rng = np.random.default_rng(RANDOM_SAMPLE_SEED)
    if len(remaining) > N_RANDOM_SAMPLE:
        sample_positions = rng.choice(len(remaining), size=N_RANDOM_SAMPLE, replace=False)
        random_sample = remaining.iloc[np.sort(sample_positions)]
    else:
        random_sample = remaining

    combined = pd.concat([most_favorable, most_unfavorable, random_sample])
    combined = combined.drop_duplicates(subset=["play_id"]).sort_values(
        ["game_date", "game_pk", "play_id"]
    )
    return combined


def main() -> None:
    if not DEV_DATA_PATH.exists():
        raise SystemExit(
            f"{DEV_DATA_PATH} not found -- this script requires the local, gitignored "
            "cleaned development parquet (2021-2024 approved development data)."
        )

    print(f"=== Loading {DEV_DATA_PATH.name} ===")
    full = pd.read_parquet(DEV_DATA_PATH)
    train_df = full[full["season"].isin(TRAIN_SEASONS)].copy()
    score_season = VALIDATION_SEASONS[0]
    score_df = full[full["season"] == score_season].copy()

    print("=== Eligibility pipeline ===")
    dev_df = _apply_eligibility_pipeline(train_df)
    score_df = _apply_eligibility_pipeline(score_df)
    score_df["event_id"] = build_event_id(score_df)
    score_df["game_date"] = score_df["game_date"].astype(str)

    print("=== Training contact model ===")
    contact_train_df = dev_df[dev_df["eligible_for_training"].fillna(False)]
    contact_trained = train_model(contact_train_df, class_weight=None)

    print("=== Training outfield opportunity model (single candidate) ===")
    outfield_train_df = dev_df[dev_df["outfield_opportunity_eligible"].astype(bool)]
    outfield_trained = train_opportunity_model(outfield_train_df, class_weight=None)

    print("=== Training infield opportunity model (model selection) ===")
    infield_elig = dev_df[dev_df["infield_opportunity_eligible"].astype(bool)]
    infield_winner, _infield_metrics, infield_trained_by_candidate = run_infield_model_selection(
        infield_elig
    )
    infield_trained = infield_trained_by_candidate[infield_winner]

    print("=== Training advancement model (model selection) ===")
    advancement_contact_trained = fit_advancement_contact_model(dev_df)
    advancement_elig = get_advancement_rows(dev_df, advancement_contact_trained)
    advancement_winner, _advancement_metrics, advancement_trained_by_candidate = (
        run_advancement_model_selection(advancement_elig)
    )
    advancement_trained = advancement_trained_by_candidate[advancement_winner]

    print(f"=== Scoring {score_season} with all four models ===")
    prob_cols = list(ADVANCEMENT_CONTACT_PROBABILITY_FEATURES)
    for col in prob_cols:
        score_df[col] = np.nan
    score_advancement_elig = get_advancement_rows(score_df, advancement_contact_trained)
    score_df.loc[score_advancement_elig.index, prob_cols] = score_advancement_elig[prob_cols]

    ledger = build_attribution_ledger(
        score_df,
        contact_trained,
        outfield_trained=outfield_trained,
        infield_trained=infield_trained,
        advancement_trained=advancement_trained,
        outfield_confidence_status="calibrated",
        infield_confidence_status="calibrated_with_limited_subgroup_evidence",
        advancement_confidence_status="calibrated_with_limited_subgroup_evidence",
    )

    print("=== Building canonical v2.0 play ledger ===")
    full_play_ledger = build_play_ledger(score_df, ledger)
    validate_play_ledger(full_play_ledger)
    print(f"full canonical ledger rows: {len(full_play_ledger)}")

    print("=== Selecting bounded representative sample ===")
    bounded = _select_bounded_sample(full_play_ledger)
    print(f"bounded sample rows: {len(bounded)} across {bounded['game_pk'].nunique()} games")
    validate_play_ledger(bounded)

    metadata = build_play_ledger_metadata(
        bounded,
        season=score_season,
        score_version="0.2",
        model_versions={
            "contact": contact_trained.variant,
            "outfield": "measured_contact_only_v07 (single candidate, no selection)",
            "infield": infield_winner,
            "advancement": advancement_winner,
        },
    )

    batter_ids = sorted({int(b) for b in bounded["batter_id"].unique()})
    print(
        f"=== Resolving {len(batter_ids)} real batter name(s) via the public MLB Stats API "
        "(/people) -- network access, no credentials required ==="
    )
    names_df = fetch_player_names(batter_ids)
    names: dict[int, str] = {}
    for _, name_row in names_df.iterrows():
        full_name = name_row["full_name"]
        if full_name is not None:
            names[int(name_row["batter_id"])] = str(full_name)
    print(f"resolved {len(names)}/{len(batter_ids)} names")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ledger_path = tmp_path / "play_ledger.parquet"
        metadata_path = tmp_path / "play_ledger_metadata.json"
        names_path = tmp_path / "names.json"
        bounded.to_parquet(ledger_path, index=False)
        metadata_path.write_text(json.dumps(metadata))
        names_path.write_text(json.dumps({str(k): v for k, v in names.items()}))

        result = explorer_gen.generate_explorer_artifacts(
            play_ledger_path=ledger_path,
            play_ledger_metadata_path=metadata_path,
            output_dir=OUTPUT_DIR,
            names_path=names_path,
        )

    print(f"=== Wrote {OUTPUT_DIR} ===")
    print(result)


if __name__ == "__main__":
    main()
