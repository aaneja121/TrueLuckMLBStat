"""Contact Luck v0.12 Phase 1: the versioned public player-season schema.

FREEZES Versions 0.2-0.11 completely -- this module defines a SCHEMA and its
VALIDATION only. It computes nothing, refits nothing, and does not touch any
model, ledger component, confidence rule, bootstrap procedure, or
qualification threshold. `mlb_luck_score.scoring.public_score_table` is the
module that actually POPULATES a table matching this schema, by reading
(never recomputing) `mlb_luck_score.scoring.run_season_aggregation`'s frozen
output.

Note on naming: the task that produced this module asked for `scoring/
public_score.py`, but that name is ALREADY TAKEN by the genuinely live,
tested, frozen Version 0.1 LEGACY tanh-based score mapping (`raw_luck_to_
public_score`, imported by `mlb_luck_score.models.predict_outcomes` and
covered by `tests/test_scoring.py`) -- overwriting it would destroy real,
in-use code, which CLAUDE.md explicitly protects ("kept only for backward
compatibility... do not use as default for new work" is a preservation
instruction, not permission to delete). This Version 0.12 assembly module is
named `public_score_table.py` instead; nothing about the legacy module was
touched.

## `batter_name` is NOT populated

The schema includes `batter_name` ("if available from a reviewed ID join")
because the task asks for it conditionally. It is NOT available in this
codebase: the raw Statcast `player_name` column in `data/processed/cleaned_
development_data_with_sprint_speed.parquet` is the PITCHER's name, not the
batter's (verified directly: for a sample `batter` id, `player_name` never
matches rows where that id appears as `pitcher`, and instead matches rows
where a KNOWN pitcher id appears as `pitcher`). There is no reviewed
batter-id-to-name join anywhere in this repository. `batter_name` is
therefore always `None` here -- explicitly left null rather than populated
with a silently wrong (pitcher) name, matching this codebase's standing rule
against fabricating reference data without a reviewed source.

## Column reference

See `PUBLIC_SCORE_FIELDS` for the authoritative, single source of truth for
names/types/nullability/descriptions -- do not duplicate this list elsewhere;
import it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PUBLIC_SCORE_SCHEMA_VERSION = "0.12.0"

QUALIFICATION_STATUSES_WITH_VALUES_PERMITTED: tuple[str, ...] = (
    "qualified",
    "provisionally_qualified",
    "small_sample",
    "insufficient_component_coverage",
)
#: `not_reportable` rows (zero eligible plays) may still appear in the public
#: table (per Phase 1's "fail validation on ... missing identifiers", NOT on
#: missing VALUES) -- but every numeric column legitimately collapses to
#: null/0 for them; see `validate_public_score_table`.


@dataclass(frozen=True)
class PublicScoreField:
    name: str
    dtype: str
    nullable: bool
    description: str


PUBLIC_SCORE_FIELDS: tuple[PublicScoreField, ...] = (
    PublicScoreField("batter_id", "int64", False, "Statcast batter identifier."),
    PublicScoreField(
        "batter_name",
        "string",
        True,
        "Batter display name, if available from a reviewed ID join. Always null in this "
        "codebase -- see module docstring.",
    ),
    PublicScoreField("season", "int64", False, "MLB season (2021-2024 only; 2025 is refused)."),
    PublicScoreField("games", "int64", False, "Distinct games with an eligible batted ball."),
    PublicScoreField(
        "eligible_batted_balls", "int64", False, "Count of resolved, ledger-eligible plays."
    ),
    PublicScoreField(
        "total_contact_luck_runs",
        "float64",
        False,
        "Additive season total (== Version 0.11's total_observed_minus_expected_runs).",
    ),
    PublicScoreField(
        "contact_luck_runs_per_100",
        "float64",
        True,
        "The OFFICIAL public metric: Contact Luck Runs per 100 eligible batted balls. Null "
        "only when eligible_batted_balls == 0.",
    ),
    PublicScoreField(
        "lower_95_interval",
        "float64",
        True,
        "Lower bound of the 95% game_pk-clustered sampling interval, on the SAME per-100 "
        "scale as contact_luck_runs_per_100.",
    ),
    PublicScoreField(
        "upper_95_interval", "float64", True, "Upper bound, same scale as lower_95_interval."
    ),
    PublicScoreField(
        "interval_interpretation",
        "string",
        True,
        "One of 'entirely_above_zero' / 'overlaps_zero' / 'entirely_below_zero' -- "
        "descriptive only, never a significance or skill claim.",
    ),
    PublicScoreField(
        "qualification_status",
        "string",
        False,
        "Frozen Version 0.11 status, reused verbatim: qualified / provisionally_qualified / "
        "small_sample / insufficient_component_coverage / not_reportable.",
    ),
    PublicScoreField(
        "official_rank_eligible",
        "bool",
        False,
        "True iff qualification_status == 'qualified' -- the ONLY rows that may carry an "
        "official rank.",
    ),
    PublicScoreField(
        "official_rank_favorable",
        "Int64",
        True,
        "Rank on the 'most favorable realized luck' leaderboard (1 = highest contact_luck_"
        "runs_per_100 among qualified rows). Null for non-qualified rows.",
    ),
    PublicScoreField(
        "official_rank_unfavorable",
        "Int64",
        True,
        "Rank on the 'least favorable realized luck' leaderboard (1 = lowest contact_luck_"
        "runs_per_100 among qualified rows). Null for non-qualified rows.",
    ),
    PublicScoreField(
        "total_contact_component_runs",
        "float64",
        True,
        "Additive component total -- see mlb_luck_score.scoring.aggregate_attribution.",
    ),
    PublicScoreField(
        "total_unexplained_residual_component_runs", "float64", True, "Additive component total."
    ),
    PublicScoreField(
        "total_defensive_execution_component_runs", "float64", True, "Additive component total."
    ),
    PublicScoreField(
        "total_advancement_component_runs", "float64", True, "Additive component total."
    ),
    PublicScoreField("contact_component_per_100", "float64", True, "Per-100 component rate."),
    PublicScoreField(
        "unexplained_residual_component_per_100", "float64", True, "Per-100 component rate."
    ),
    PublicScoreField(
        "defensive_execution_component_per_100", "float64", True, "Per-100 component rate."
    ),
    PublicScoreField(
        "advancement_execution_component_per_100", "float64", True, "Per-100 component rate."
    ),
    PublicScoreField(
        "share_of_value_from_provisional_components",
        "float64",
        True,
        "Bounded in [0, 1] -- see Version 0.11's aggregate_attribution docstring.",
    ),
    PublicScoreField(
        "defense_unavailable_play_count",
        "int64",
        False,
        "Count of eligible plays with no defense component available at all.",
    ),
    PublicScoreField(
        "advancement_unavailable_play_count",
        "int64",
        False,
        "Count of eligible plays with no advancement component available at all.",
    ),
    PublicScoreField(
        "component_status_reason_codes",
        "object",
        False,
        "dict[component_name, {'model_status_values': [...], 'reason_codes': [...]}] -- the "
        "DISTINCT statuses/reason codes this player-season's plays carried per component "
        "(see mlb_luck_score.scoring.component_confidence).",
    ),
    PublicScoreField(
        "model_version",
        "object",
        False,
        "dict[component_name, version_label] -- which frozen Version 0.2/0.7A/0.8/0.9 "
        "model/candidate produced this player-season's values.",
    ),
    PublicScoreField(
        "score_version", "string", False, "PUBLIC_SCORE_SCHEMA_VERSION at generation time."
    ),
    PublicScoreField(
        "generated_at", "string", False, "ISO 8601 UTC timestamp this row was generated."
    ),
    PublicScoreField(
        "data_through_date",
        "string",
        False,
        "Latest game_date among this row's season's scored plays (ISO 8601 date).",
    ),
)

REQUIRED_PUBLIC_SCORE_COLUMNS: tuple[str, ...] = tuple(f.name for f in PUBLIC_SCORE_FIELDS)

INTERVAL_ABOVE_ZERO = "entirely_above_zero"
INTERVAL_OVERLAPS_ZERO = "overlaps_zero"
INTERVAL_BELOW_ZERO = "entirely_below_zero"
INTERVAL_INTERPRETATION_VALUES: tuple[str, ...] = (
    INTERVAL_ABOVE_ZERO,
    INTERVAL_OVERLAPS_ZERO,
    INTERVAL_BELOW_ZERO,
)


class PublicScoreSchemaError(ValueError):
    """Raised when a public player-season table fails schema validation."""


def classify_interval(lower: float, upper: float) -> str | None:
    """`entirely_above_zero` / `overlaps_zero` / `entirely_below_zero`, or `None`
    if either bound is null (no interval available for this row).
    """
    if pd.isna(lower) or pd.isna(upper):
        return None
    if lower > 0:
        return INTERVAL_ABOVE_ZERO
    if upper < 0:
        return INTERVAL_BELOW_ZERO
    return INTERVAL_OVERLAPS_ZERO


def validate_public_score_table(df: pd.DataFrame) -> None:
    """Fail loudly on: missing columns, duplicate (batter_id, season) rows,
    missing identifiers, non-finite (inf/-inf) official values, or
    inconsistent qualification/ranking fields.

    Deliberately does NOT require every numeric column to be non-null --
    `not_reportable` (zero-eligible-play) rows legitimately have null
    rates/intervals, and non-qualified rows legitimately have null rank
    columns. Null is a valid, honest absence; infinity is not.
    """
    missing_cols = [c for c in REQUIRED_PUBLIC_SCORE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise PublicScoreSchemaError(f"Public score table is missing column(s): {missing_cols}")

    if df[["batter_id", "season"]].isna().any().any():
        raise PublicScoreSchemaError("batter_id/season must never be null")

    duplicated = df.duplicated(subset=["batter_id", "season"], keep=False)
    if duplicated.any():
        dupes = df.loc[duplicated, ["batter_id", "season"]].drop_duplicates()
        raise PublicScoreSchemaError(
            f"Duplicate (batter_id, season) row(s) found: {dupes.to_dict(orient='records')}"
        )

    finite_required_cols = (
        "total_contact_luck_runs",
        "contact_luck_runs_per_100",
        "lower_95_interval",
        "upper_95_interval",
    )
    for col in finite_required_cols:
        values = df[col].to_numpy(dtype=float)
        non_null = ~np.isnan(values)
        if non_null.any() and not np.isfinite(values[non_null]).all():
            raise PublicScoreSchemaError(f"{col} contains a non-finite (inf) value")

    if "qualification_status" not in df.columns:
        raise PublicScoreSchemaError("qualification_status column missing")

    is_qualified = df["qualification_status"] == "qualified"
    expected_eligible = is_qualified
    if not (df["official_rank_eligible"] == expected_eligible).all():
        raise PublicScoreSchemaError(
            "official_rank_eligible is inconsistent with qualification_status == 'qualified'"
        )

    for rank_col in ("official_rank_favorable", "official_rank_unfavorable"):
        has_rank = df[rank_col].notna()
        if (has_rank & ~df["official_rank_eligible"]).any():
            raise PublicScoreSchemaError(
                f"{rank_col} is populated for a row where official_rank_eligible is False"
            )

    if is_qualified.any():
        qualified_rates = df.loc[is_qualified, "contact_luck_runs_per_100"]
        if qualified_rates.isna().any():
            raise PublicScoreSchemaError(
                "qualified rows must have a non-null contact_luck_runs_per_100"
            )
        qualified_intervals = df.loc[is_qualified, ["lower_95_interval", "upper_95_interval"]]
        if qualified_intervals.isna().any().any():
            raise PublicScoreSchemaError(
                "qualified rows must have non-null lower_95_interval/upper_95_interval"
            )
