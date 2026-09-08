"""Version 0.14: the frozen split-half reliability procedure for replication
question F.

Question F asks a SECONDARY structural question: **does pitcher Contact Luck
remain approximately non-persistent across within-season halves?** It is not
a success criterion on its own and cannot override the package-level
classification in `pitcher_replication_spec.CLASSIFICATION_RULE`. A near-zero
result CONFIRMS the retrospective framing -- Contact Luck is not claimed to
persist, and a quantity that does not persist is not supposed to be reliable.

## This is a PORT, not a new estimator

The accepted 2024 procedure is `mlb_luck_score.models.
evaluate_aggregation_stability.compute_split_half_reliability` (Version 0.11
Phase 6), whose split rules and inclusion threshold were fixed before any
real player-season number was examined. This module reuses that module's own
mask helpers and its own `MIN_ELIGIBLE_EACH_HALF` constant **by import**,
rather than restating them, so "the same design" holds by construction and
cannot drift into a second definition. The only difference is the grouping
key: `aggregate_to_pitcher_season` instead of `aggregate_to_batter_season`.

Nothing here was tuned against 2024. `reproduce_batter_side` exists to prove
that: it runs this module's own code path on the BATTER key and compares
against the committed Version 0.11 numbers.

## The design, restated exactly

- **Splits.** Two, both game-clustered:
  - `calendar` -- `game_date <= median(game_date)` vs. the rest.
  - `odd_even` -- `game_pk % 2 == 1` vs. the rest. A deterministic
    quasi-random split whose unit is the GAME, so both halves of a
    pitcher-season are made of whole games and no appearance is split.
- **Aggregation.** Each half is re-aggregated independently through the
  frozen pipeline, so each half's per-100 rate uses the same resolved
  eligible BBE denominator as a full season would.
- **Inclusion.** A pitcher-season needs `MIN_ELIGIBLE_EACH_HALF` (20)
  resolved eligible batted balls in BOTH halves. Imported, not restated.
- **Statistic.** Pearson and Spearman correlation of
  `observed_minus_expected_per_100` between the two halves.

## Why the pitcher sign convention does not matter here

`aggregate_to_pitcher_season` negates every signed column. A correlation is
invariant under a common sign flip applied to both variables, so the
reliability figure is identical whether it is computed on the pitcher or the
batter sign convention for a FIXED grouping. The pitcher/batter difference
in this module is entirely the grouping key, never the sign.

## Denominators and grouping are unchanged

`eligible_batted_balls` remains outcome-resolved rows only, and `games`
remains distinct games containing at least one resolved eligible batted ball
-- both inherited from `aggregate_to_batter_season` via
`aggregate_to_pitcher_season`, neither redefined here. See
`pitcher_replication_spec.DENOMINATOR`.

This module reads no season's data on its own: it operates on a
`SeasonAggregationArtifacts` the caller already produced.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Imported, never restated: reusing the accepted Version 0.11 split rules and
# inclusion threshold is what makes "the same design" a fact rather than a
# claim. The two mask helpers are private to that module, which is why they
# are imported explicitly and named here -- a copy would be free to drift.
from mlb_luck_score.models.evaluate_aggregation_stability import (
    MIN_ELIGIBLE_EACH_HALF,
    RANKING_METRIC,
    _half_season_masks,
    _odd_even_masks,
)
from mlb_luck_score.scoring.aggregate_attribution import aggregate_to_batter_season
from mlb_luck_score.scoring.pitching_contact_luck import aggregate_to_pitcher_season

SPLITS: tuple[str, ...] = ("calendar", "odd_even")

#: The carried-forward 2024 wording was "approximately zero at all pitcher
#: WORKLOADS", but the accepted Version 0.11 procedure reports ONE number per
#: split, at `MIN_ELIGIBLE_EACH_HALF`. Rather than invent a by-workload
#: reliability estimator, the sweep re-runs the SAME estimator at three fixed
#: values of its OWN existing `min_eligible_each_half` parameter. That is a
#: documented parameter of the accepted design, not a new statistic.
#:
#: Frozen here, before 2025 is opened, so the values cannot be chosen later
#: to suit a result. 20 is the accepted default and is the PRIMARY figure;
#: 50 and 100 are secondary context.
MIN_ELIGIBLE_EACH_HALF_SWEEP: tuple[int, ...] = (20, 50, 100)

#: The committed Version 0.11 Phase 6 batter-side 2024 result
#: (`outputs/tables/aggregation_stability_v011_report.json`). Recorded here
#: as the reproduction target for `reproduce_batter_side`, so a port that
#: silently changed the procedure would be caught rather than trusted.
BATTER_SIDE_2024_REFERENCE: dict[str, dict[str, Any]] = {
    "calendar": {
        "n_players_compared": 399,
        "pearson_r": 0.050297542421047836,
        "spearman_r": 0.07131881210564099,
    },
    "odd_even": {
        "n_players_compared": 487,
        "pearson_r": 0.09654304470474931,
        "spearman_r": 0.11889983530505965,
    },
}


class SplitHalfError(ValueError):
    """Raised when a split-half comparison cannot be computed."""


def _masks(scoring_df: pd.DataFrame, split: str) -> tuple[pd.Series, pd.Series]:
    if split not in SPLITS:
        raise SplitHalfError(f"Unknown split {split!r}; expected one of {SPLITS}")
    return _half_season_masks(scoring_df) if split == "calendar" else _odd_even_masks(scoring_df)


def _aggregate_subset(
    scoring_df: pd.DataFrame,
    ledger: pd.DataFrame,
    confidence: pd.DataFrame,
    mask: pd.Series,
    *,
    group: str,
) -> pd.DataFrame:
    """Re-aggregate one half through the frozen pipeline.

    Mirrors `evaluate_aggregation_stability._aggregate_subset` exactly,
    including the `event_id`-based confidence subset, and dispatches on the
    grouping key only.
    """
    idx = scoring_df.index[mask]
    subset_confidence = confidence[confidence["event_id"].isin(scoring_df.loc[idx, "event_id"])]
    aggregate = aggregate_to_pitcher_season if group == "pitcher" else aggregate_to_batter_season
    return aggregate(scoring_df.loc[idx], ledger.loc[idx], subset_confidence)


def compute_split_half_reliability(
    artifacts: Any,
    *,
    group: str = "pitcher",
    split: str = "calendar",
    metric: str = RANKING_METRIC,
    min_eligible_each_half: int = MIN_ELIGIBLE_EACH_HALF,
) -> dict[str, Any]:
    """Pearson/Spearman correlation of `metric` between two halves of each
    player-season's scored plays.

    Args:
        artifacts: A `SeasonAggregationArtifacts` (needs `scoring_df`,
            `ledger`, `confidence`). Produced by the caller; this function
            opens no data itself.
        group: `"pitcher"` (question F) or `"batter"` (the reproduction
            control).
        split: `"calendar"` or `"odd_even"`.
        metric: The per-100 rate column, frozen to `RANKING_METRIC`.
        min_eligible_each_half: Resolved eligible BBE required in BOTH
            halves, frozen to Version 0.11's `MIN_ELIGIBLE_EACH_HALF`.

    Returns:
        The split's identity, its inclusion rule, `n_players_compared`, and
        the two correlations (both `None` when fewer than two rows survive).

    Raises:
        SplitHalfError: on an unknown `split` or `group`.
    """
    if group not in ("pitcher", "batter"):
        raise SplitHalfError(f"Unknown group {group!r}; expected 'pitcher' or 'batter'")

    mask_a, mask_b = _masks(artifacts.scoring_df, split)
    summary_a = _aggregate_subset(
        artifacts.scoring_df, artifacts.ledger, artifacts.confidence, mask_a, group=group
    )
    summary_b = _aggregate_subset(
        artifacts.scoring_df, artifacts.ledger, artifacts.confidence, mask_b, group=group
    )

    # `aggregate_to_pitcher_season` keys on `pitcher`; the batter path keys
    # on `batter`. Both name the grouping column after the key itself.
    key = "pitcher" if group == "pitcher" else "batter"
    merged = summary_a.merge(summary_b, on=[key, "season"], suffixes=("_a", "_b"), how="inner")
    merged = merged[
        (merged["eligible_batted_balls_a"] >= min_eligible_each_half)
        & (merged["eligible_batted_balls_b"] >= min_eligible_each_half)
    ]

    n = len(merged)
    base: dict[str, Any] = {
        "group": group,
        "split": split,
        "metric": metric,
        "min_eligible_each_half": min_eligible_each_half,
        "n_players_compared": n,
    }
    if n < 2:
        return {**base, "pearson_r": None, "spearman_r": None}

    col_a, col_b = f"{metric}_a", f"{metric}_b"
    return {
        **base,
        "pearson_r": float(merged[col_a].corr(merged[col_b], method="pearson")),
        "spearman_r": float(merged[col_a].corr(merged[col_b], method="spearman")),
    }


def compute_all_splits(
    artifacts: Any, *, group: str = "pitcher", min_eligible_each_half: int = MIN_ELIGIBLE_EACH_HALF
) -> dict[str, Any]:
    """Both frozen splits for one grouping key."""
    return {
        split: compute_split_half_reliability(
            artifacts,
            group=group,
            split=split,
            min_eligible_each_half=min_eligible_each_half,
        )
        for split in SPLITS
    }


def compute_workload_sweep(artifacts: Any, *, group: str = "pitcher") -> dict[str, Any]:
    """The SAME estimator at each frozen value of its own
    `min_eligible_each_half` parameter -- the "at all pitcher workloads"
    check, without a second statistic. See `MIN_ELIGIBLE_EACH_HALF_SWEEP`.
    """
    return {
        f"min_eligible_each_half_{floor}": compute_all_splits(
            artifacts, group=group, min_eligible_each_half=floor
        )
        for floor in MIN_ELIGIBLE_EACH_HALF_SWEEP
    }


def reproduce_batter_side(artifacts: Any, *, tolerance: float = 1e-9) -> dict[str, Any]:
    """Run THIS module's code path on the batter key and compare against the
    committed Version 0.11 Phase 6 numbers.

    This is the evidence that the port did not change the procedure. It is
    not a test of the metric; it is a test of this module.

    Returns:
        Per split: the reproduced values, the reference values, the absolute
        differences, and whether every one is within `tolerance`. Plus a
        top-level `reproduces` flag.
    """
    results: dict[str, Any] = {"tolerance": tolerance, "splits": {}}
    all_match = True
    for split, reference in BATTER_SIDE_2024_REFERENCE.items():
        measured = compute_split_half_reliability(artifacts, group="batter", split=split)
        diffs = {
            "n_players_compared": measured["n_players_compared"] - reference["n_players_compared"],
            "pearson_r": (
                abs(measured["pearson_r"] - reference["pearson_r"])
                if measured["pearson_r"] is not None
                else None
            ),
            "spearman_r": (
                abs(measured["spearman_r"] - reference["spearman_r"])
                if measured["spearman_r"] is not None
                else None
            ),
        }
        matches = (
            diffs["n_players_compared"] == 0
            and diffs["pearson_r"] is not None
            and diffs["pearson_r"] <= tolerance
            and diffs["spearman_r"] is not None
            and diffs["spearman_r"] <= tolerance
        )
        all_match = all_match and matches
        results["splits"][split] = {
            "measured": measured,
            "reference": reference,
            "absolute_differences": diffs,
            "matches": matches,
        }
    results["reproduces"] = all_match
    return results


def build_question_f_report(artifacts: Any) -> dict[str, Any]:
    """The complete question-F payload: the pitcher result, both splits, and
    the batter-side reproduction control that validates the implementation.
    """
    reproduction = reproduce_batter_side(artifacts)
    return {
        "question": "F_persistence",
        "status": "secondary_structural_question",
        "is_a_success_criterion_on_its_own": False,
        "cannot_override_package_classification": True,
        "procedure": (
            "mlb_luck_score.models.evaluate_aggregation_stability."
            "compute_split_half_reliability, ported to the pitcher grouping key. Split "
            "rules and MIN_ELIGIBLE_EACH_HALF are imported from that module, not restated."
        ),
        "splits": SPLITS,
        "min_eligible_each_half": MIN_ELIGIBLE_EACH_HALF,
        "metric": RANKING_METRIC,
        "pitcher": compute_all_splits(artifacts, group="pitcher"),
        "pitcher_workload_sweep": compute_workload_sweep(artifacts, group="pitcher"),
        "workload_sweep_note": (
            "The sweep is the SAME estimator at three frozen values of its own "
            "min_eligible_each_half parameter, not a second statistic. The "
            "min_eligible_each_half=20 row is the primary figure and matches the "
            "accepted Version 0.11 default."
        ),
        "batter_side_reproduction": reproduction,
        "implementation_trustworthy": bool(reproduction["reproduces"]),
    }


def main() -> None:
    """Recompute the 2024 DEVELOPMENT question-F figures recorded in
    `pitcher_replication_spec`, so they are reproducible rather than merely
    asserted.

    2024 only. Season protection is inherited from `build_player_season_report`,
    which calls `assert_seasons_allowed` itself and exposes no flag this script
    could use to reach 2025. The parquet read happens HERE, inside `main`, never
    at import time -- importing this module opens nothing.
    """
    import json
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))

    from mlb_luck_score.config import TABLES_DIR
    from mlb_luck_score.scoring.run_season_aggregation import build_player_season_report

    dev_data = repo_root / "data/processed/cleaned_development_data_with_sprint_speed.parquet"
    if not dev_data.exists():
        raise SystemExit(
            f"{dev_data} not found -- this runner needs the local, gitignored 2021-2024 "
            "development parquet."
        )
    artifacts = build_player_season_report(dev_data, output_dir=TABLES_DIR)
    report = build_question_f_report(artifacts)
    print(json.dumps(report, indent=2, default=str))
    if not report["implementation_trustworthy"]:
        raise SystemExit(
            "Batter-side reproduction FAILED: this implementation does not reproduce the "
            "committed Version 0.11 Phase 6 numbers, so its pitcher figures are not "
            "trustworthy."
        )


if __name__ == "__main__":
    main()
