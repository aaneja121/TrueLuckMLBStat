"""Contact Luck v1.1 Phase 6: presentation-only batter-ID-to-name overlay.

Fills the `batter_name` column the frozen public-score schema already
defines (`mlb_luck_score.scoring.public_score_schema`) but every earlier
version leaves null, using `mlb_luck_score.data.download_player_names.
fetch_player_names`. This is deliberately a PRESENTATION step, not a scoring
step:

  - runs strictly AFTER `mlb_luck_score.scoring.public_score_table.
    build_public_score_table` and `mlb_luck_score.scoring.leaderboard.
    assign_official_ranks` -- it never participates in model fitting,
    feature-building, ranking, or qualification.
  - `batter_id` remains the primary identifier; the join is BY batter_id
    ONLY, never by name (so a name collision or formatting difference can
    never merge two different players).
  - a name that cannot be resolved stays `None` -- never guessed or
    fabricated -- with a reason code recorded in the RETURNED report, not as
    a new column on the frozen public-score schema (which has no reason-code
    column and must not gain one here).
  - no score, interval, rank, or qualification column is ever touched: see
    `tests/test_prospective_player_names.py` for the structural checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

REASON_RESOLVED = "resolved"
REASON_NOT_FOUND_IN_SOURCE = "unresolved_batter_id_not_found_in_source"
REASON_SOURCE_RETURNED_NULL_NAME = "unresolved_source_returned_null_name"

#: Columns `apply_player_name_overlay` is allowed to change on the output
#: table relative to its input -- everything else must be byte-identical.
_OVERLAY_MUTABLE_COLUMNS = frozenset({"batter_name"})


class PlayerNameOverlayError(ValueError):
    """Raised when the name overlay cannot be safely applied."""


@dataclass(frozen=True)
class NameResolutionReport:
    source: str
    retrieved_at: str
    requested_batter_count: int
    resolved_batter_count: int
    unresolved_batter_count: int
    unresolved_batter_ids: list[int]
    reason_codes_by_batter_id: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_player_name_overlay(
    public_score_table: pd.DataFrame, names_df: pd.DataFrame
) -> tuple[pd.DataFrame, NameResolutionReport]:
    """Return `(overlaid_table, report)`. `overlaid_table` is a COPY of
    `public_score_table` with only `batter_name` changed; every other column
    -- including `contact_luck_runs_per_100`, both interval bounds, both
    official-rank columns, and `qualification_status` -- is guaranteed
    unchanged (asserted below, not just assumed).

    Args:
        public_score_table: Already-ranked frozen public score table (see
            module docstring for the required ordering).
        names_df: `mlb_luck_score.data.download_player_names.
            fetch_player_names`'s output (`batter_id`, `full_name`,
            `retrieved_at`, `source`).
    """
    if public_score_table.empty:
        report = NameResolutionReport(
            source=str(names_df["source"].iloc[0]) if len(names_df) else "unknown",
            retrieved_at=datetime.now(UTC).isoformat(),
            requested_batter_count=0,
            resolved_batter_count=0,
            unresolved_batter_count=0,
            unresolved_batter_ids=[],
            reason_codes_by_batter_id={},
        )
        return public_score_table.copy(), report

    before = public_score_table.copy()
    name_lookup = names_df.set_index("batter_id")["full_name"]

    requested_ids = before["batter_id"].astype("int64").tolist()
    reason_codes: dict[str, str] = {}
    resolved_names: list[str | None] = []
    for batter_id in requested_ids:
        if batter_id not in name_lookup.index:
            reason_codes[str(batter_id)] = REASON_NOT_FOUND_IN_SOURCE
            resolved_names.append(None)
            continue
        full_name = name_lookup.loc[batter_id]
        if pd.isna(full_name):
            reason_codes[str(batter_id)] = REASON_SOURCE_RETURNED_NULL_NAME
            resolved_names.append(None)
        else:
            reason_codes[str(batter_id)] = REASON_RESOLVED
            resolved_names.append(str(full_name))

    overlaid = before.copy()
    overlaid["batter_name"] = pd.array(resolved_names, dtype="string")

    unchanged_columns = [c for c in before.columns if c not in _OVERLAY_MUTABLE_COLUMNS]
    if not before[unchanged_columns].equals(overlaid[unchanged_columns]):
        raise PlayerNameOverlayError(
            "apply_player_name_overlay changed a column other than batter_name -- refusing to "
            "return a table where the name overlay could have affected scores or rankings."
        )

    resolved_count = sum(1 for code in reason_codes.values() if code == REASON_RESOLVED)
    unresolved_ids = sorted(
        int(bid) for bid, code in reason_codes.items() if code != REASON_RESOLVED
    )
    report = NameResolutionReport(
        source=str(names_df["source"].iloc[0]) if len(names_df) else "unknown",
        retrieved_at=(
            str(names_df["retrieved_at"].iloc[0])
            if len(names_df)
            else datetime.now(UTC).isoformat()
        ),
        requested_batter_count=len(requested_ids),
        resolved_batter_count=resolved_count,
        unresolved_batter_count=len(unresolved_ids),
        unresolved_batter_ids=unresolved_ids,
        reason_codes_by_batter_id=reason_codes,
    )
    return overlaid, report
