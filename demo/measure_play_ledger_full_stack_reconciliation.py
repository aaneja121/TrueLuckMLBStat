"""Version 1.4.0 Phase 3.1: full four-model (contact + outfield + infield +
advancement) real-2024-data reconciliation -- the PERMANENT regression gate
for the Rc-vs-Rf class of error the real 2026-08-15 production snapshot
exposed.

## Why this script exists, distinct from demo/measure_play_ledger_scale.py

`measure_play_ledger_scale.py` (Phase 2) deliberately trains ONLY the
contact model -- by design, since v1.4.0's schema originally had no
component-attribution fields to populate from the other three. That
choice had an unintended side effect: `Rc - E0` (raw contact-luck) and
`Rf - E0` (the full telescoping-identity headline quantity) are
IDENTICAL whenever no defensive/advancement model runs, so a contact-only
reconciliation can never distinguish a correct Rf-based play-ledger
implementation from an incorrect Rc-based one -- exactly the bug that
reached the real 2026-08-15 production snapshot (261/630 batters
mismatched on Total Contact Luck Runs and Runs/100).

This script trains the SAME four component models real `train_and_score_
2026`/`train_and_score_2025` train (contact, outfield opportunity -- single
candidate, infield opportunity -- model-selected, advancement -- model-
selected), on real TRAIN_SEASONS (2021-2023) data, then scores real
VALIDATION_SEASONS (2024) data -- so real, nonzero defensive/advancement
execution contributions are genuinely exercised, the same way they are in
production. It deliberately does NOT import `evaluation.run_v1_final_
evaluation`/`prospective.prospective_scoring` (both target 2025/2026
specifically and are out of scope here) -- it duplicates their small
model-training glue instead, matching this codebase's own established
convention (see those two modules' own docstrings for why each duplicates
rather than imports the other).

Run: `.venv/bin/python demo/measure_play_ledger_full_stack_reconciliation.py`.
Reads `data/processed/cleaned_development_data.parquet` (local-only,
gitignored, never committed). Writes nothing outside a scratch temp
directory and prints a report to stdout. No 2025/2026 data touched.
"""

from __future__ import annotations

import sys
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
from mlb_luck_score.scoring.aggregate_attribution import aggregate_to_batter_season  # noqa: E402
from mlb_luck_score.scoring.attribution_ledger import build_attribution_ledger  # noqa: E402
from mlb_luck_score.scoring.component_confidence import build_play_level_confidence  # noqa: E402
from mlb_luck_score.scoring.play_ledger_export import build_play_ledger  # noqa: E402
from mlb_luck_score.scoring.play_ledger_metadata import (  # noqa: E402
    build_play_ledger_metadata,
    validate_play_ledger_metadata,
)
from mlb_luck_score.scoring.play_ledger_schema import (  # noqa: E402
    resolved_rows,
    validate_play_ledger,
)

DEV_DATA_PATH = REPO_ROOT / "data" / "processed" / "cleaned_development_data.parquet"


def _apply_eligibility_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_eligibility(df)
    df = add_outfield_opportunity_eligibility(df)
    df = add_infield_opportunity_eligibility(df)
    df = add_advancement_eligibility(df)
    df = add_opportunity_features_by_domain(df)
    df = add_advancement_features(df)
    return df


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
    print(f"TRAIN_SEASONS={TRAIN_SEASONS}: {len(train_df)} rows")
    print(f"scoring season={score_season} (VALIDATION_SEASONS[0]): {len(score_df)} rows")

    print("\n=== Eligibility pipeline (contact + outfield + infield + advancement) ===")
    dev_df = _apply_eligibility_pipeline(train_df)
    score_df = _apply_eligibility_pipeline(score_df)
    score_df["event_id"] = build_event_id(score_df)
    score_df["game_date"] = score_df["game_date"].astype(str)

    t0 = time.time()
    print("\n=== Training contact model ===")
    contact_train_df = dev_df[dev_df["eligible_for_training"].fillna(False)]
    contact_trained = train_model(contact_train_df, class_weight=None)
    print(f"  {time.time() - t0:.1f}s")

    t0 = time.time()
    print("=== Training outfield opportunity model (single candidate) ===")
    outfield_train_df = dev_df[dev_df["outfield_opportunity_eligible"].astype(bool)]
    outfield_trained = train_opportunity_model(outfield_train_df, class_weight=None)
    print(f"  {time.time() - t0:.1f}s")

    t0 = time.time()
    print("=== Training infield opportunity model (model selection) ===")
    infield_elig = dev_df[dev_df["infield_opportunity_eligible"].astype(bool)]
    infield_winner, _infield_metrics, infield_trained_by_candidate = run_infield_model_selection(
        infield_elig
    )
    infield_trained = infield_trained_by_candidate[infield_winner]
    print(f"  winner={infield_winner} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    print("=== Training advancement model (model selection) ===")
    advancement_contact_trained = fit_advancement_contact_model(dev_df)
    advancement_elig = get_advancement_rows(dev_df, advancement_contact_trained)
    advancement_winner, _advancement_metrics, advancement_trained_by_candidate = (
        run_advancement_model_selection(advancement_elig)
    )
    advancement_trained = advancement_trained_by_candidate[advancement_winner]
    print(f"  winner={advancement_winner} ({time.time() - t0:.1f}s)")

    print("\n=== Scoring 2024 with all four models ===")
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
    confidence = build_play_level_confidence(
        score_df,
        ledger,
        outfield_model_version="measured_contact_only_v07",
        infield_model_version=infield_winner,
        advancement_model_version=advancement_winner,
        outfield_model_status="calibrated",
        infield_model_status="calibrated_with_limited_subgroup_evidence",
        advancement_model_status="calibrated_with_limited_subgroup_evidence",
    )
    summary = aggregate_to_batter_season(score_df, ledger, confidence)

    print("\n=== Rc-vs-Rf divergence check (proves this configuration is NOT degenerate) ===")
    resolved_mask = ledger["baseline_expected_contact_run_value"].notna()
    diff = (
        ledger.loc[resolved_mask, "final_result_surprise"]
        - ledger.loc[resolved_mask, "contact_result_surprise"]
    ).abs()
    n_diverging = int((diff > 1e-9).sum())
    print(f"resolved rows: {int(resolved_mask.sum())}")
    print(f"rows where Rc - E0 != Rf - E0: {n_diverging}")
    if n_diverging == 0:
        raise SystemExit(
            "FATAL: zero diverging rows -- this run is degenerate (same failure class as a "
            "contact-only configuration) and cannot serve as a valid regression gate."
        )

    print("\n=== Building + validating the canonical play ledger (v2.0 contract) ===")
    play_ledger = build_play_ledger(score_df, ledger)
    validate_play_ledger(play_ledger)
    metadata = build_play_ledger_metadata(
        play_ledger,
        season=score_season,
        score_version="0.12.0",
        model_versions={
            "contact": contact_trained.variant,
            "outfield": "measured_contact_only_v07 (single candidate, no selection)",
            "infield": infield_winner,
            "advancement": advancement_winner,
        },
    )
    validate_play_ledger_metadata(metadata, play_ledger)
    print(f"play_ledger_version={metadata['play_ledger_version']}")
    print(
        f"row_count={metadata['row_count']} scored={metadata['scored_row_count']} "
        f"unresolved={metadata['unresolved_row_count']}"
    )

    print("\n=== FULL FOUR-MODEL RECONCILIATION (the permanent regression gate) ===")
    resolved = resolved_rows(play_ledger)
    per_batter_ledger_count = resolved.groupby("batter_id").size()
    per_batter_ledger_sum = resolved.groupby("batter_id")["contact_luck_runs"].sum()
    per_batter_ledger_games = resolved.groupby("batter_id")["game_pk"].nunique()

    per_batter_published_count = summary.set_index("batter")["eligible_batted_balls"]
    per_batter_published_sum = summary.set_index("batter")["total_observed_minus_expected_runs"]
    per_batter_published_per100 = summary.set_index("batter")["observed_minus_expected_per_100"]
    per_batter_published_games = summary.set_index("batter")["games"]

    joined_count = pd.DataFrame(
        {"ledger": per_batter_ledger_count, "published": per_batter_published_count}
    ).fillna(0)
    count_mismatches = joined_count[joined_count["ledger"] != joined_count["published"]]

    joined_sum = pd.DataFrame(
        {"ledger": per_batter_ledger_sum, "published": per_batter_published_sum}
    ).fillna(0.0)
    sum_diff = (joined_sum["ledger"] - joined_sum["published"]).abs()
    sum_mismatches = joined_sum[sum_diff > 1e-6]

    reconstructed_per100 = 100.0 * joined_sum["ledger"] / joined_count["ledger"].replace(0, np.nan)
    per100_diff = (reconstructed_per100 - per_batter_published_per100).abs()
    per100_mismatches = per100_diff[per100_diff > 1e-6]

    joined_games = pd.DataFrame(
        {"ledger": per_batter_ledger_games, "published": per_batter_published_games}
    ).fillna(0)
    games_mismatches = joined_games[joined_games["ledger"] != joined_games["published"]]

    print(f"A. BBE mismatches: {len(count_mismatches)}")
    if len(count_mismatches):
        print(count_mismatches.head(10))
    print(
        f"B. Total Contact Luck mismatches (atol=1e-6): {len(sum_mismatches)}"
        + (f" max abs diff={float(sum_diff.max()):.6f}" if len(sum_diff) else "")
    )
    if len(sum_mismatches):
        print(sum_mismatches.head(10))
    print(
        f"C. Runs/100 mismatches (atol=1e-6): {len(per100_mismatches.dropna())}"
        + (f" max abs diff={float(per100_diff.max()):.6f}" if len(per100_diff) else "")
    )
    print(f"D. Scored Games mismatches: {len(games_mismatches)}")
    if len(games_mismatches):
        print(games_mismatches.head(10))

    total_mismatches = (
        len(count_mismatches)
        + len(sum_mismatches)
        + len(per100_mismatches.dropna())
        + len(games_mismatches)
    )
    print(f"\nTOTAL MISMATCHES: {total_mismatches}")
    if total_mismatches:
        raise SystemExit("FAILED: full four-model reconciliation found mismatches -- see above.")
    print(
        "PASSED: 0 mismatches for all four checks (BBE, Total Contact Luck, Runs/100, Scored Games)."
    )


if __name__ == "__main__":
    main()
