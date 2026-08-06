"""Contact Luck v0.12 Phase 2-5: assemble the public player-season table.

FREEZES Versions 0.2-0.11 completely. This module computes NOTHING new about
models, ledger components, confidence, or aggregation -- it only reads
`mlb_luck_score.scoring.run_season_aggregation.SeasonAggregationArtifacts`
(itself built from the frozen pipeline) and reshapes/renames it into the
`mlb_luck_score.scoring.public_score_schema.PUBLIC_SCORE_FIELDS` contract.
Official ranks are NOT assigned here -- see `mlb_luck_score.scoring.
leaderboard` for that (a deliberate separation: this module is pure schema
population, ranking is pure ranking policy).

Named `public_score_table.py` rather than the `public_score.py` the task
requested -- see `public_score_schema`'s module docstring for why (that name
is already taken by the genuinely live, tested, frozen Version 0.1 legacy
score mapping).

## The official metric (task decisions 1-3, 9, 10)

`contact_luck_runs_per_100` (Version 0.11's `observed_minus_expected_per_100`,
renamed for the public contract) is the ONE official score. `total_contact_
luck_runs` (Version 0.11's `total_observed_minus_expected_runs`) is reported
alongside it as the additive season total -- it remains reportable whenever
`eligible_batted_balls > 0` regardless of any component's provisional status,
because it is an EXACT observed-minus-expected accounting total; provisional
component status affects how the DECOMPOSITION should be read, not whether
the total exists. Positive = more favorable than expected; negative = less
favorable. Any percentile/index is secondary and non-additive (Phase 4).

## Intervals (Phase 3)

`lower_95_interval`/`upper_95_interval` are on the SAME per-100 scale as
`contact_luck_runs_per_100` (Version 0.11's `observed_minus_expected_per_100_
ci_low`/`_ci_high`) -- always populated together with the point estimate,
never behind a flag. `interval_interpretation` is descriptive only (`mlb_luck_
score.scoring.public_score_schema.classify_interval`) and is computed for
EVERY row with a non-null interval, including non-qualified ones -- Phase 3
explicitly forbids suppressing a row for crossing zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from mlb_luck_score.scoring.public_score_schema import (
    PUBLIC_SCORE_SCHEMA_VERSION,
    REQUIRED_PUBLIC_SCORE_COLUMNS,
    classify_interval,
    validate_public_score_table,
)
from mlb_luck_score.scoring.qualification import STATUS_QUALIFIED
from mlb_luck_score.scoring.run_season_aggregation import SeasonAggregationArtifacts

COMPONENT_NAMES: tuple[str, ...] = ("contact", "outfield_defense", "infield_defense", "advancement")


class PublicScoreTableError(ValueError):
    """Raised when the public score table cannot be assembled from its inputs."""


def _component_status_reason_codes_by_batter_season(
    scoring_df: pd.DataFrame, confidence: pd.DataFrame
) -> dict[tuple[object, object], dict[str, dict[str, list[str]]]]:
    """Per (batter, season): per component, the DISTINCT `model_status_
    confidence` values and the union of `reason_codes` seen across that
    player-season's plays -- not a single value, since a player's plays for
    one component (e.g. outfield defense) can legitimately span multiple
    domain statuses (open-field AND near-wall) within one season.
    """
    key = scoring_df[["event_id", "batter", "season"]]
    merged = confidence.merge(key, on="event_id", how="left")

    result: dict[tuple[object, object], dict[str, dict[str, list[str]]]] = {}
    for (batter_val, season_val), group in merged.groupby(["batter", "season"], sort=False):
        per_component: dict[str, dict[str, list[str]]] = {}
        for component_name, comp_group in group.groupby("component_name", sort=False):
            model_status_values = sorted(comp_group["model_status_confidence"].unique().tolist())
            reason_codes = sorted({code for codes in comp_group["reason_codes"] for code in codes})
            per_component[str(component_name)] = {
                "model_status_values": model_status_values,
                "reason_codes": reason_codes,
            }
        result[(batter_val, season_val)] = per_component
    return result


def _data_through_date_by_season(scoring_df: pd.DataFrame) -> dict[object, str]:
    game_date = pd.to_datetime(scoring_df["game_date"])
    return {
        season_val: game_date.loc[group_idx].max().date().isoformat()
        for season_val, group_idx in scoring_df.groupby("season", sort=False).groups.items()
    }


def build_model_version_map(report: dict[str, object]) -> dict[str, str]:
    """`{component_name: version_label}` for every `COMPONENT_NAMES` entry,
    read verbatim from `mlb_luck_score.scoring.run_season_aggregation`'s
    report -- `baseline_v02` for contact is the Version 0.2 ADOPTED default,
    never re-derived here.
    """
    winners = report.get("model_selection_winners", {})
    winners_dict = winners if isinstance(winners, dict) else {}
    return {
        "contact": "baseline_v02",
        "outfield_defense": str(winners_dict.get("outfield", "unknown")),
        "infield_defense": str(winners_dict.get("infield", "unknown")),
        "advancement": str(winners_dict.get("advancement", "unknown")),
    }


def build_public_score_table(artifacts: SeasonAggregationArtifacts) -> pd.DataFrame:
    """Build the public player-season table (schema-shaped, ranks NOT yet
    assigned -- see `mlb_luck_score.scoring.leaderboard.assign_official_
    ranks`) from a `SeasonAggregationArtifacts` bundle.
    """
    player_season = artifacts.player_season
    model_version_map = build_model_version_map(artifacts.report)
    reason_codes_map = _component_status_reason_codes_by_batter_season(
        artifacts.scoring_df, artifacts.confidence
    )
    data_through_by_season = _data_through_date_by_season(artifacts.scoring_df)
    generated_at = datetime.now(UTC).isoformat()

    is_qualified = player_season["qualification_status"] == STATUS_QUALIFIED
    lower = player_season["observed_minus_expected_per_100_ci_low"]
    upper = player_season["observed_minus_expected_per_100_ci_high"]

    out = pd.DataFrame(
        {
            "batter_id": player_season["batter"].astype("int64"),
            "batter_name": pd.array([None] * len(player_season), dtype="string"),
            "season": player_season["season"].astype("int64"),
            "games": player_season["games"].astype("int64"),
            "eligible_batted_balls": player_season["eligible_batted_balls"].astype("int64"),
            "total_contact_luck_runs": player_season["total_observed_minus_expected_runs"],
            "contact_luck_runs_per_100": player_season["observed_minus_expected_per_100"],
            "lower_95_interval": lower,
            "upper_95_interval": upper,
            "qualification_status": player_season["qualification_status"],
            "official_rank_eligible": is_qualified.to_numpy(),
            "official_rank_favorable": pd.array([pd.NA] * len(player_season), dtype="Int64"),
            "official_rank_unfavorable": pd.array([pd.NA] * len(player_season), dtype="Int64"),
            "total_contact_component_runs": player_season["total_contact_component_runs"],
            "total_unexplained_residual_component_runs": player_season[
                "total_unexplained_residual_component_runs"
            ],
            "total_defensive_execution_component_runs": player_season[
                "total_defensive_execution_component_runs"
            ],
            "total_advancement_component_runs": player_season["total_advancement_component_runs"],
            "contact_component_per_100": player_season["contact_component_per_100"],
            "unexplained_residual_component_per_100": player_season[
                "unexplained_residual_component_per_100"
            ],
            "defensive_execution_component_per_100": player_season[
                "defensive_execution_component_per_100"
            ],
            "advancement_execution_component_per_100": player_season[
                "advancement_execution_component_per_100"
            ],
            "share_of_value_from_provisional_components": player_season[
                "share_of_value_from_provisional_components"
            ],
            "defense_unavailable_play_count": player_season[
                "defense_unavailable_play_count"
            ].astype("int64"),
            "advancement_unavailable_play_count": player_season[
                "advancement_unavailable_play_count"
            ].astype("int64"),
        }
    )

    out["interval_interpretation"] = [
        classify_interval(lo, hi) for lo, hi in zip(lower.to_numpy(), upper.to_numpy(), strict=True)
    ]
    out["component_status_reason_codes"] = pd.Series(
        [
            reason_codes_map.get((batter_val, season_val), {})
            for batter_val, season_val in zip(
                player_season["batter"], player_season["season"], strict=True
            )
        ],
        index=out.index,
        dtype=object,
    )
    out["model_version"] = pd.Series(
        [dict(model_version_map) for _ in range(len(player_season))], index=out.index, dtype=object
    )
    out["score_version"] = PUBLIC_SCORE_SCHEMA_VERSION
    out["generated_at"] = generated_at
    out["data_through_date"] = [
        data_through_by_season.get(season_val, "") for season_val in player_season["season"]
    ]

    out = out[list(REQUIRED_PUBLIC_SCORE_COLUMNS)]
    validate_public_score_table(out)
    return out


def describe_components(row: pd.Series) -> list[dict[str, object]]:
    """Build the Phase 5 component-display records for ONE public-score row:
    one dict per additive component (contact / unexplained residual /
    defensive execution / advancement), each with its run total, per-100
    value, and status/reason codes -- for player-page or notebook display.
    Never implies a provisional component is equivalently validated to a
    calibrated one (the raw status/reason codes are passed through verbatim,
    not summarized into a single pass/fail).
    """
    reason_codes = row["component_status_reason_codes"]
    return [
        {
            "component_name": "contact",
            "total_runs": row["total_contact_component_runs"],
            "per_100": row["contact_component_per_100"],
            "status": reason_codes.get("contact", {}),
        },
        {
            "component_name": "unexplained_residual",
            "total_runs": row["total_unexplained_residual_component_runs"],
            "per_100": row["unexplained_residual_component_per_100"],
            "status": {},
        },
        {
            "component_name": "defensive_execution",
            "total_runs": row["total_defensive_execution_component_runs"],
            "per_100": row["defensive_execution_component_per_100"],
            "status": {
                "outfield_defense": reason_codes.get("outfield_defense", {}),
                "infield_defense": reason_codes.get("infield_defense", {}),
            },
        },
        {
            "component_name": "advancement",
            "total_runs": row["total_advancement_component_runs"],
            "per_100": row["advancement_execution_component_per_100"],
            "status": reason_codes.get("advancement", {}),
        },
    ]


@dataclass(frozen=True)
class DevelopmentPercentileDesign:
    """Phase 4: an OPTIONAL, NOT-adopted-by-default secondary display index.

    Preserves every field Phase 4 requires if this is ever computed:
    `reference_seasons`/`reference_population`/`transformation_version`/
    `tie_behavior`/`out_of_range_handling`. Default recommendation remains
    "do not adopt a separate index" -- `mlb_luck_score.scoring.
    run_public_score` never includes this in the default public table.
    """

    reference_seasons: tuple[int, ...]
    reference_population: str
    transformation_version: str
    tie_behavior: str
    out_of_range_handling: str


DEFAULT_DEVELOPMENT_PERCENTILE_DESIGN = DevelopmentPercentileDesign(
    reference_seasons=(2024,),
    reference_population="qualified_rows_in_reference_seasons",
    transformation_version="v0.12.0-candidate",
    tie_behavior="average_rank_percentile (pandas .rank(pct=True, method='average'))",
    out_of_range_handling="not_applicable_percentile_is_always_within_reference_population",
)


def compute_development_percentile(
    public_score_table: pd.DataFrame,
    *,
    metric: str = "contact_luck_runs_per_100",
    design: DevelopmentPercentileDesign = DEFAULT_DEVELOPMENT_PERCENTILE_DESIGN,
) -> pd.Series:
    """Compute the OPTIONAL candidate percentile described by `design`, among
    QUALIFIED rows only, in `design.reference_seasons`. NON-additive, NEVER
    substituted for `contact_luck_runs_per_100`, and NOT included in the
    default public table -- a caller must explicitly request this.

    Returns:
        A `float` Series (0-100), index-aligned with `public_score_table`,
        `NaN` for rows outside the reference population (non-qualified, or a
        season outside `design.reference_seasons`).
    """
    in_reference_population = (public_score_table["qualification_status"] == STATUS_QUALIFIED) & (
        public_score_table["season"].isin(design.reference_seasons)
    )
    percentile = pd.Series(float("nan"), index=public_score_table.index)
    if in_reference_population.any():
        percentile.loc[in_reference_population] = (
            public_score_table.loc[in_reference_population, metric].rank(pct=True, method="average")
            * 100.0
        )
    return percentile
