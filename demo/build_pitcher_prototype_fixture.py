"""Version 0.13.1: builds the committed Pitcher Contact Luck UI-prototype
fixture from real 2024 development data.

A ONE-TIME (or occasionally-rerun) offline generation step, run manually by
a maintainer -- never by `dashboard/build.py`, never by CI, and never as
part of `make check`. It lives outside `dashboard/` for exactly the reason
`demo/build_play_explorer_dev_ledger.py` and `demo/build_demo_fixture.py`
do (see their docstrings and `tests/test_dashboard_isolation.py`): it
trains real frozen models, which `dashboard/` is structurally forbidden
from importing.

## What this script does NOT do

- Never touches 2025 (sealed final-evaluation data) or 2026 (prospective
  scoring). It calls `mlb_luck_score.scoring.run_season_aggregation.
  build_player_season_report` UNCHANGED, which trains on `TRAIN_SEASONS`
  (2021-2023) and scores `VALIDATION_SEASONS` (2024), and which calls
  `assert_seasons_allowed` itself. This script exposes no flag that could
  reach a final-test or prospective season.
- Never trains a second copy of anything. The four component models are fit
  exactly once, by that frozen runner; every pitcher number here is a
  re-aggregation of the SAME scored plays, per `SeasonAggregationArtifacts`'
  documented contract. This is that contract's third consumer, after
  `evaluate_aggregation_stability` and `run_pitching_contact_luck`.
- Never redefines Contact Luck, its run values, or a threshold. The sign
  convention and the season arithmetic come from `mlb_luck_score.scoring.
  pitching_contact_luck` unchanged.
- Never resolves a name over the network. Raw Statcast's `player_name`
  column IS the pitcher's name (see README.md Version 0.13, "`batter_name`
  is not populated"), so pitcher identity is available locally and this
  script makes no API call at all.

## What it adds beyond the Version 0.13 season table

Three presentation inputs the season table does not carry, all of them
projections of already-scored values, never new scoring:

1. `player_name` per pitcher id, taken from the raw column.
2. A DESCRIPTIVE role bucket from BBE-per-appearance. This is not role
   metadata -- the repository has none. See `ROLE_*` below.
3. The single largest favorable and largest unfavorable batted ball of the
   pitcher's season, read straight off the per-play attribution ledger
   (`-final_result_surprise`, the pitcher's sign convention applied to the
   same per-play quantity the season total sums).

Usage (requires the local, gitignored 2021-2024 development parquet; no
network access):

    .venv/bin/python demo/build_pitcher_prototype_fixture.py

Writes `dashboard/pitcher_prototype_fixture.json` (committed reviewed
reference data, same convention as `dashboard/demo_fixture.json`) and
`outputs/tables/pitcher_prototype_research_report.json` (gitignored
research output backing the Version 0.13.1 README section).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "replication"))

# Version 0.14: the question-E estimators are RESEARCH code and live in
# `replication/`, so the replication freeze does not depend on this
# presentation-side generator for them. This script is now a consumer of
# that module, which is what guarantees the committed fixture and the
# replication are computed by the same functions rather than two copies.
from pitcher_replication_estimators import (  # noqa: E402
    PRACTICAL_WORKLOAD_FLOORS,
    rate_precision_report,
    resolving_power_report,
)

from mlb_luck_score.config import TABLES_DIR  # noqa: E402
from mlb_luck_score.scoring.pitching_contact_luck import (  # noqa: E402
    build_pitcher_season_table,
    verify_batter_side_reproduction,
)
from mlb_luck_score.scoring.run_season_aggregation import (  # noqa: E402
    build_player_season_report,
)

DEV_DATA_PATH = (
    REPO_ROOT / "data" / "processed" / "cleaned_development_data_with_sprint_speed.parquet"
)
FIXTURE_PATH = REPO_ROOT / "dashboard" / "pitcher_prototype_fixture.json"
RESEARCH_REPORT_PATH = REPO_ROOT / "outputs" / "tables" / "pitcher_prototype_research_report.json"

FIXTURE_VERSION = "0.1"
PROTOTYPE_SEASON = 2024

#: DESCRIPTIVE usage buckets read off the observed bimodal BBE-per-appearance
#: distribution. These are NOT roles: this repository has no authoritative
#: starter/reliever/closer metadata, and nothing here may be presented as
#: one. The gap between the two bounds is reported as `ambiguous` rather
#: than forced into either bucket.
ROLE_STARTER_LIKE_MIN_BBE_PER_APPEARANCE = 10.0
ROLE_RELIEVER_LIKE_MAX_BBE_PER_APPEARANCE = 8.0

#: PRESENTATION rule only, motivated by normalized-rate uncertainty at very
#: low exposure -- never a qualification, eligibility, or MLB threshold, and
#: never described as one. Below this, a pitcher-season keeps its full page
#: and its numbers but is left off the ranked board.
BOARD_DISPLAY_MINIMUM_BBE = 60

#: Display organization for the reliever-like board only, same caveat.
WORKLOAD_BAND_HIGH_MIN_BBE = 150


def _role_bucket(bbe: int, appearances: int) -> str:
    if appearances <= 0:
        return "ambiguous"
    per_appearance = bbe / appearances
    if per_appearance >= ROLE_STARTER_LIKE_MIN_BBE_PER_APPEARANCE:
        return "starter_like"
    if per_appearance <= ROLE_RELIEVER_LIKE_MAX_BBE_PER_APPEARANCE:
        return "reliever_like"
    return "ambiguous"


def _workload_band(bbe: int) -> str:
    if bbe >= WORKLOAD_BAND_HIGH_MIN_BBE:
        return "high"
    if bbe >= BOARD_DISPLAY_MINIMUM_BBE:
        return "moderate"
    return "below_board_minimum"


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


def _play_record(row: pd.Series) -> dict[str, Any]:
    """One batted ball, projected for display. Every value is READ, never
    recomputed: `pitching_contact_luck_runs` is the ledger's own per-play
    `final_result_surprise` on the pitcher's sign convention.
    """
    return {
        "play_id": str(row["event_id"]),
        "game_date": str(row["game_date"])[:10],
        "pitching_contact_luck_runs": round(float(row["pitching_contact_luck_runs"]), 4),
        "launch_speed_mph": _optional_float(row.get("launch_speed")),
        "launch_angle_deg": _optional_float(row.get("launch_angle")),
        "bb_type": None if pd.isna(row.get("bb_type")) else str(row["bb_type"]),
        "outcome_class": None if pd.isna(row.get("outcome_class")) else str(row["outcome_class"]),
    }


def build_fixture(
    scoring_df: pd.DataFrame,
    ledger: pd.DataFrame,
    confidence: pd.DataFrame,
    reproduction: dict[str, Any],
    player_season: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    table = build_pitcher_season_table(scoring_df, ledger, confidence)
    table = table[table["season"] == PROTOTYPE_SEASON].copy()

    # Per-play pitcher Contact Luck: the SAME per-play quantity the season
    # total sums, negated once, exactly as `SIGNED_VALUE_COLUMNS` negates
    # the season columns. Never a second definition.
    plays = scoring_df.loc[
        :, ["event_id", "pitcher", "season", "game_date", "launch_speed", "launch_angle", "bb_type"]
    ].copy()
    plays["outcome_class"] = scoring_df["outcome_class"]
    # Index-aligned, never positional: `ledger` is built on `scoring_df`'s own
    # index (`build_attribution_ledger`), so this joins row-for-row by identity.
    plays["pitching_contact_luck_runs"] = -ledger["final_result_surprise"]
    plays = plays[plays["season"] == PROTOTYPE_SEASON]
    resolved = plays[plays["pitching_contact_luck_runs"].notna()].copy()

    names = (
        scoring_df[scoring_df["season"] == PROTOTYPE_SEASON]
        .groupby("pitcher")["player_name"]
        .agg(lambda s: s.mode().iat[0] if len(s.mode()) else None)
    )

    by_pitcher = {int(pid): grp for pid, grp in resolved.groupby("pitcher")}

    pitchers: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        pitcher_id = int(row["pitcher"])
        bbe = int(row["eligible_batted_balls"])
        appearances = int(row["games"])
        total = float(row["total_observed_minus_expected_runs"])
        grp = by_pitcher.get(pitcher_id)

        favorable = unfavorable = None
        largest_abs_share = None
        if grp is not None and len(grp):
            favorable = _play_record(grp.loc[grp["pitching_contact_luck_runs"].idxmax()])
            unfavorable = _play_record(grp.loc[grp["pitching_contact_luck_runs"].idxmin()])
            largest_abs = float(grp["pitching_contact_luck_runs"].abs().max())
            if abs(total) > 1e-9:
                largest_abs_share = round(largest_abs / abs(total), 4)

        raw_name = names.get(pitcher_id)
        pitchers.append(
            {
                "pitcher_id": pitcher_id,
                "name": _display_name(raw_name),
                "role_bucket": _role_bucket(bbe, appearances),
                "workload_band": _workload_band(bbe),
                "cumulative_contact_luck_runs": round(total, 4),
                "eligible_batted_balls": bbe,
                "appearances": appearances,
                "bbe_per_appearance": round(bbe / appearances, 2) if appearances else None,
                "contact_luck_per_100": round(float(row["observed_minus_expected_per_100"]), 4),
                "per_100_ci_low": round(float(row["observed_minus_expected_per_100_ci_low"]), 4),
                "per_100_ci_high": round(float(row["observed_minus_expected_per_100_ci_high"]), 4),
                "cumulative_ci_low": round(float(row["observed_minus_expected_ci_low"]), 4),
                "cumulative_ci_high": round(float(row["observed_minus_expected_ci_high"]), 4),
                "largest_play_share_of_net": largest_abs_share,
                "largest_favorable_play": favorable,
                "largest_unfavorable_play": unfavorable,
            }
        )

    pitchers.sort(key=lambda p: -p["cumulative_contact_luck_runs"])

    fixture = {
        "pitcher_prototype_fixture_version": FIXTURE_VERSION,
        "season": PROTOTYPE_SEASON,
        "metric": "pitching_contact_luck",
        "sign_convention": (
            "pitching_contact_luck = expected run value - observed run value = "
            "-1 x batting_contact_luck. Positive means the realized outcome was more "
            "favorable to the pitcher than the contact itself predicted."
        ),
        "primary_quantity": "cumulative_contact_luck_runs",
        "secondary_quantity": "contact_luck_per_100",
        "development_only": True,
        "batter_side_reproduction": reproduction,
        "presentation_rules": {
            "starter_like_min_bbe_per_appearance": ROLE_STARTER_LIKE_MIN_BBE_PER_APPEARANCE,
            "reliever_like_max_bbe_per_appearance": ROLE_RELIEVER_LIKE_MAX_BBE_PER_APPEARANCE,
            "board_display_minimum_bbe": BOARD_DISPLAY_MINIMUM_BBE,
            "workload_band_high_min_bbe": WORKLOAD_BAND_HIGH_MIN_BBE,
            "role_buckets_are_descriptive_only": True,
            "board_minimum_is_not_qualification": True,
        },
        "pitchers": pitchers,
    }
    return fixture, build_research_report(pitchers, resolved, player_season)


def _display_name(raw: Any) -> str | None:
    """Statcast's `player_name` is `Last, First`; display order is First Last."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if "," not in text:
        return text
    last, first = text.split(",", 1)
    return f"{first.strip()} {last.strip()}"


def _hitter_totals_vs_rate(player_season: pd.DataFrame) -> dict[str, Any]:
    """The control that keeps the HITTER product unchanged: at the shipped
    qualified bar, ranking hitters by cumulative runs and by runs/100 give
    almost the same order, so the pitcher side's totals-first presentation
    is not an argument for changing the hitter leaderboard.

    The population is `qualification_status == "qualified"` -- the SHIPPED
    contract (`QUALIFICATION_THRESHOLD_SETS["primary"]`: >=200 resolved
    eligible BBE AND >=100 games, plus the component gates), not a bare
    `eligible_batted_balls >= 200` cut. Those two differ by 67 rows on 2024
    (216 vs 283) because the BBE-only cut drops `min_games`, and only the
    former answers a question about the board this project actually ships.
    `also_reported_bbe_only_cut` records the other one alongside it so the
    two can never again be mistaken for the same number.
    """
    qualified = player_season[player_season["qualification_status"] == "qualified"]
    out = _totals_vs_rate_stats(qualified)
    out["n_qualified_hitters"] = out.pop("n")
    out["population"] = 'qualification_status == "qualified" (shipped contract)'
    out["also_reported_bbe_only_cut"] = _totals_vs_rate_stats(
        player_season[player_season["eligible_batted_balls"] >= 200]
    ) | {"population": "eligible_batted_balls >= 200 only (drops min_games; NOT shipped)"}
    return out


def _totals_vs_rate_stats(rows: pd.DataFrame) -> dict[str, Any]:
    """Every total-vs-rate agreement statistic, on whatever population is
    handed in. Rank changes and overlaps use the FAVORABLE direction and the
    shipped `batter_id`-ascending tie-break, so they describe the same
    ordering `mlb_luck_score.scoring.leaderboard` would produce.
    """
    total = rows["total_observed_minus_expected_runs"]
    rate = rows["observed_minus_expected_per_100"]
    bbe = rows["eligible_batted_balls"]
    delta = (
        total.rank(ascending=False, method="min") - rate.rank(ascending=False, method="min")
    ).abs()

    def overlap(k: int, *, ascending: bool) -> int:
        by_total, by_rate = (
            set(rows.sort_values([key, "batter"], ascending=[ascending, True]).head(k)["batter"])
            for key in ("total_observed_minus_expected_runs", "observed_minus_expected_per_100")
        )
        return int(len(by_total & by_rate))

    return {
        "n": int(len(rows)),
        "bbe_min": int(bbe.min()),
        "bbe_max": int(bbe.max()),
        "bbe_median": float(bbe.median()),
        "bbe_max_over_min": round(float(bbe.max() / bbe.min()), 2),
        "pearson_total_vs_rate": round(float(total.corr(rate)), 4),
        "spearman_total_vs_rate": round(float(total.corr(rate, method="spearman")), 4),
        "median_rank_change": float(delta.median()),
        "p90_rank_change": float(np.percentile(delta, 90)),
        "max_rank_change": float(delta.max()),
        "top_10_overlap": overlap(10, ascending=False),
        "top_25_overlap": overlap(25, ascending=False),
        "bottom_10_overlap": overlap(10, ascending=True),
        "bottom_25_overlap": overlap(25, ascending=True),
        "pearson_abs_total_vs_bbe": round(float(total.abs().corr(bbe)), 4),
        "spearman_abs_total_vs_bbe": round(float(total.abs().corr(bbe, method="spearman")), 4),
        "pearson_signed_total_vs_bbe": round(float(total.corr(bbe)), 4),
        "spearman_signed_total_vs_bbe": round(float(total.corr(bbe, method="spearman")), 4),
    }


def build_research_report(
    pitchers: list[dict[str, Any]],
    resolved: pd.DataFrame,
    player_season: pd.DataFrame,
) -> dict[str, Any]:
    """Every quantitative claim the Version 0.13.1 README section makes,
    recomputed here from the same rows the fixture ships, so the prose can
    cite measured numbers rather than remembered ones.
    """
    frame = pd.DataFrame(pitchers)
    starters = frame[frame["role_bucket"] == "starter_like"]
    relievers = frame[frame["role_bucket"] == "reliever_like"]

    share = relievers["largest_play_share_of_net"].dropna()
    total = frame["cumulative_contact_luck_runs"]
    bbe = frame["eligible_batted_balls"]

    return {
        "season": PROTOTYPE_SEASON,
        "pitcher_season_rows": int(len(frame)),
        "resolved_play_rows": int(len(resolved)),
        "workload": {
            "max_bbe_any_pitcher": int(bbe.max()),
            "starter_like_n": int(len(starters)),
            "starter_like_median_bbe": float(starters["eligible_batted_balls"].median()),
            "starter_like_max_bbe": int(starters["eligible_batted_balls"].max()),
            "reliever_like_n": int(len(relievers)),
            "reliever_like_median_bbe": float(relievers["eligible_batted_balls"].median()),
            "reliever_like_max_bbe": int(relievers["eligible_batted_balls"].max()),
            "ambiguous_n": int((frame["role_bucket"] == "ambiguous").sum()),
            "reliever_like_seasons_reaching_300_bbe": int(
                (relievers["eligible_batted_balls"] >= 300).sum()
            ),
        },
        "workload_relationship": {
            "pearson_abs_total_vs_bbe": round(float(total.abs().corr(bbe)), 4),
            "spearman_abs_total_vs_bbe": round(float(total.abs().corr(bbe, method="spearman")), 4),
            "pearson_signed_total_vs_bbe": round(float(total.corr(bbe)), 4),
            "spearman_signed_total_vs_bbe": round(float(total.corr(bbe, method="spearman")), 4),
        },
        "reliever_like_single_play_dominance": {
            "n_with_defined_share": int(len(share)),
            "share_over_25_pct": round(float((share > 0.25).mean()), 4),
            "share_over_50_pct": round(float((share > 0.50).mean()), 4),
            "share_over_100_pct": round(float((share > 1.00).mean()), 4),
        },
        "board_population": {
            "board_display_minimum_bbe": BOARD_DISPLAY_MINIMUM_BBE,
            "starter_like_on_board": int(
                (starters["eligible_batted_balls"] >= BOARD_DISPLAY_MINIMUM_BBE).sum()
            ),
            "reliever_like_on_board": int(
                (relievers["eligible_batted_balls"] >= BOARD_DISPLAY_MINIMUM_BBE).sum()
            ),
            "below_board_minimum": int((bbe < BOARD_DISPLAY_MINIMUM_BBE).sum()),
        },
        "rate_precision_requirements": rate_precision_report(frame),
        "resolving_power_by_workload_floor": [
            resolving_power_report(frame, min_bbe=floor)
            for floor in (1, *PRACTICAL_WORKLOAD_FLOORS)
        ],
        # The >=1 floor's 1.94 is an ARTIFACT, not a finding: the bootstrap
        # resamples games within a pitcher-season, so a one-appearance season
        # has one game to resample, every replicate reproduces the point
        # estimate, and its measurement variance is recorded as zero. The
        # decomposition then books that row's (enormous) spread entirely as
        # signal. This pair is what makes that verifiable rather than asserted.
        "resolving_power_all_rows_excluding_zero_width": resolving_power_report(
            frame, min_bbe=1, drop_zero_width=True
        ),
        "single_appearance_rows": int((frame["appearances"] == 1).sum()),
        "hitter_totals_vs_rate_control": _hitter_totals_vs_rate(player_season),
    }


def main() -> None:
    if not DEV_DATA_PATH.exists():
        raise SystemExit(
            f"{DEV_DATA_PATH} not found -- this script requires the local, gitignored "
            "cleaned development parquet (2021-2024 approved development data)."
        )

    print(f"=== Scoring via the frozen batter pipeline ({DEV_DATA_PATH.name}) ===")
    # `outputs/tables`, exactly like `run_pitching_contact_luck`'s default.
    # `build_player_season_report` only READS this directory (the three
    # already-computed component gate verdicts at `opportunity_model_
    # comparison_detail.json`/`infield_opportunity_detail.json`/
    # `advancement_detail.json`); the writes in that module live in its
    # `main`, not in the function, so nothing here overwrites a table.
    # Passing a temp dir instead would silently downgrade every component
    # confidence status to "not_supplied" and change the provisional-share
    # columns relative to the committed Version 0.13 run.
    artifacts = build_player_season_report(DEV_DATA_PATH, output_dir=TABLES_DIR)

    print(f"reused {len(artifacts.scoring_df)} scored plays")

    reproduction = verify_batter_side_reproduction(
        artifacts.scoring_df, artifacts.ledger, artifacts.confidence
    )
    print(f"batter_side_reproduction={reproduction}")
    if not reproduction.get("reproduces", False):
        raise SystemExit(
            f"Batter-side reproduction FAILED ({reproduction}); pitcher values from this "
            "run are not trustworthy and this fixture must not be written."
        )

    fixture, report = build_fixture(
        artifacts.scoring_df,
        artifacts.ledger,
        artifacts.confidence,
        reproduction,
        artifacts.player_season,
    )

    FIXTURE_PATH.write_text(json.dumps(fixture, indent=1, default=str))
    RESEARCH_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESEARCH_REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))

    print(f"=== Wrote {FIXTURE_PATH} ({FIXTURE_PATH.stat().st_size / 1024:.0f} KiB) ===")
    print(f"=== Wrote {RESEARCH_REPORT_PATH} ===")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
