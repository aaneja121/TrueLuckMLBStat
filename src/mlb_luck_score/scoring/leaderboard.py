"""Contact Luck v0.12 Phase 2: ranking policy.

FREEZES Versions 0.2-0.11 completely, and does not alter `mlb_luck_score.
scoring.public_score_table`'s values -- this module only ASSIGNS rank numbers
on top of an already-built public score table, and only to rows the frozen
Version 0.11 qualification status marks `qualified`.

## Ranking method: COMPETITION ranking (chosen and documented before any real
   output was inspected)

Ties share the SAME rank, and the next distinct value's rank skips ahead by
the number of tied rows (pandas' `.rank(method="min")` -- e.g. two players
tied for 1st both get rank 1, the next player gets rank 3). This is the
convention most public sports leaderboards already use (golf, track medal
counts, ...) and avoids the confusing "two 1st-place ranks followed
immediately by a 2nd place" reading DENSE ranking would produce. `RANK_METHOD`
documents this choice; do not switch it without updating this docstring and
every dependent test.

## Deterministic tie-break for DISPLAY order (chosen before any real output
   was inspected)

Competition ranking assigns the SAME rank number to tied rows, but rows still
need a stable, deterministic DISPLAY order among themselves. `TIE_BREAK_
COLUMN` (`batter_id`, ascending) is that secondary sort key -- arbitrary in
the sense that no baseball meaning is claimed for it, but deterministic and
reproducible run to run.

## What this module NEVER does

- Never uses `lower_95_interval`/`upper_95_interval` to rank or reorder
  players -- ranking is on `contact_luck_runs_per_100` alone.
- Never assigns a rank to a row where `official_rank_eligible` is `False`
  (i.e. `qualification_status != "qualified"`).
- Never produces a component-level leaderboard of any kind -- there is
  deliberately NO ranking function here for the outfield-defense/infield
  -defense/advancement components (provisional near-wall, provisional
  infield, provisional advancement) or for any not-calibrated component.
  Component values may still be DISPLAYED (see `mlb_luck_score.scoring.
  public_score_table.describe_components`), just never ranked.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

RANK_METHOD: Literal["min"] = "min"  # standard competition ranking
TIE_BREAK_COLUMN = "batter_id"
SCORE_COLUMN = "contact_luck_runs_per_100"

FAVORABLE_DIRECTION = "favorable"
UNFAVORABLE_DIRECTION = "unfavorable"

LEADERBOARD_DISPLAY_COLUMNS: tuple[str, ...] = (
    "batter_id",
    "batter_name",
    "season",
    "contact_luck_runs_per_100",
    "lower_95_interval",
    "upper_95_interval",
    "interval_interpretation",
    "eligible_batted_balls",
    "games",
    "total_contact_luck_runs",
)


class LeaderboardError(ValueError):
    """Raised when leaderboard construction inputs are invalid."""


def assign_official_ranks(public_score_table: pd.DataFrame) -> pd.DataFrame:
    """Return a COPY of `public_score_table` with `official_rank_favorable`/
    `official_rank_unfavorable` populated for `official_rank_eligible` rows
    only (every other row keeps its null rank, exactly as `mlb_luck_score.
    scoring.public_score_table.build_public_score_table` left it).
    """
    if "official_rank_eligible" not in public_score_table.columns:
        raise LeaderboardError("public_score_table is missing official_rank_eligible")

    out = public_score_table.copy()
    eligible_mask = out["official_rank_eligible"].astype(bool)
    if eligible_mask.any():
        scores = out[SCORE_COLUMN].loc[eligible_mask]
        out.loc[eligible_mask, "official_rank_favorable"] = scores.rank(
            method=RANK_METHOD, ascending=False
        ).astype("Int64")
        out.loc[eligible_mask, "official_rank_unfavorable"] = scores.rank(
            method=RANK_METHOD, ascending=True
        ).astype("Int64")
    return out


def _leaderboard(
    public_score_table: pd.DataFrame, *, direction: str, top_n: int | None
) -> pd.DataFrame:
    if direction not in (FAVORABLE_DIRECTION, UNFAVORABLE_DIRECTION):
        raise LeaderboardError(f"Unknown direction {direction!r}")

    rank_col = (
        "official_rank_favorable"
        if direction == FAVORABLE_DIRECTION
        else "official_rank_unfavorable"
    )
    if rank_col not in public_score_table.columns:
        raise LeaderboardError(
            f"public_score_table is missing {rank_col} -- call assign_official_ranks first"
        )

    eligible = public_score_table[public_score_table["official_rank_eligible"].astype(bool)].copy()
    eligible = eligible.sort_values([rank_col, TIE_BREAK_COLUMN], ascending=[True, True])
    if top_n is not None:
        eligible = eligible.head(top_n)

    display_cols = [rank_col, *[c for c in LEADERBOARD_DISPLAY_COLUMNS if c in eligible.columns]]
    return eligible[display_cols].reset_index(drop=True)


def most_favorable_leaderboard(
    public_score_table: pd.DataFrame, *, top_n: int | None = None
) -> pd.DataFrame:
    """The "Most favorable realized luck" leaderboard (`mlb_luck_score.
    scoring.public_labels.MOST_FAVORABLE_LEADERBOARD_LABEL`): highest
    `contact_luck_runs_per_100` first, among `qualified` rows only. Requires
    `assign_official_ranks` to have already been called.
    """
    return _leaderboard(public_score_table, direction=FAVORABLE_DIRECTION, top_n=top_n)


def least_favorable_leaderboard(
    public_score_table: pd.DataFrame, *, top_n: int | None = None
) -> pd.DataFrame:
    """The "Least favorable outcomes relative to expectation" leaderboard
    (`mlb_luck_score.scoring.public_labels.LEAST_FAVORABLE_LEADERBOARD_
    LABEL`) -- NEVER call this "worst players" in any label or output.
    Lowest `contact_luck_runs_per_100` first, among `qualified` rows only.
    """
    return _leaderboard(public_score_table, direction=UNFAVORABLE_DIRECTION, top_n=top_n)
