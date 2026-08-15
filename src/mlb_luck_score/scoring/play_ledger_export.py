"""Version 1.4.0 (Play Explorer foundation): canonical play-ledger exporter.

PROJECTS an already-scored per-play state into the `play_ledger_schema`
contract. Computes NOTHING new: every scoring-truth value (`observed_run_
value`, `expected_run_value`, `contact_luck_runs`, the probability vector)
is copied verbatim from `mlb_luck_score.scoring.attribution_ledger.
build_attribution_ledger`'s own output (`ledger`) -- this module never
calls a model, never re-derives an expected/observed run value, and never
reimplements the Contact Luck formula. See `play_ledger_schema.py`'s
module docstring for why `outcome_class`/`observed_run_value`/`contact_
luck_runs` are nullable together, and CLAUDE.md's "Never silently
redefine the Luck Score."

## Phase 3: pure projection, no second inference

Phase 2's exporter accepted a separate `contact_proba` parameter because
`build_attribution_ledger` did not yet expose the probability vector it
already computes internally -- that required a Phase-2-only validation
workaround (a second, deterministic `predict_proba_ordered` call in the
CALLER, never inside this module) to reproduce it for testing/measurement.

Phase 3 made `build_attribution_ledger` itself return `p_out`..`p_home_run`
natively (the SAME single `predict_proba_ordered` call it always made,
just no longer discarded -- see `attribution_ledger.py`'s module
docstring), nulled together with every other result-linked ledger column
by the SAME unresolved-nulling step. This exporter now reads those columns
directly off `ledger`, exactly like every other scoring field -- there is
no `contact_proba` parameter, no masking/reconstruction step, and no
second inference call anywhere in this module.
"""

from __future__ import annotations

import pandas as pd

from mlb_luck_score.config import CLASS_ORDER
from mlb_luck_score.data.clean_batted_balls import build_event_id
from mlb_luck_score.scoring.play_ledger_schema import PLAY_LEDGER_COLUMNS, validate_play_ledger

__all__ = ["PlayLedgerExportError", "build_play_ledger"]

_REQUIRED_DF_COLUMNS: tuple[str, ...] = (
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "game_date",
    "season",
    "batter",
    "stand",
    "launch_speed",
    "launch_angle",
    "bb_type",
    "spray_angle_approx",
    "hit_distance_sc",
)

#: Ledger columns this module reads verbatim -- never recomputes. Named
#: explicitly so a future ledger-column rename fails loudly here rather
#: than silently reading the wrong (or a stale, pre-rename) column.
_LEDGER_OBSERVED_COLUMN = "observed_contact_result_run_value"
_LEDGER_EXPECTED_COLUMN = "baseline_expected_contact_run_value"
_LEDGER_CONTACT_LUCK_COLUMN = "contact_result_surprise"
_LEDGER_PROBABILITY_COLUMNS: tuple[str, ...] = tuple(f"p_{cls}" for cls in CLASS_ORDER)


class PlayLedgerExportError(ValueError):
    """Raised when inputs to the play-ledger exporter are invalid or misaligned."""


def build_play_ledger(df: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """Project `df`/`ledger` into the canonical play-ledger schema.

    Args:
        df: The same per-play DataFrame passed to `build_attribution_
            ledger` (post-eligibility, one row per `is_eligible` play).
        ledger: `build_attribution_ledger(df, ...)`'s own output, same
            index as `df` -- including its native `p_out`..`p_home_run`
            columns (Phase 3; see module docstring).

    Returns:
        A DataFrame with exactly `PLAY_LEDGER_COLUMNS`, one row per input
        row, already validated via `play_ledger_schema.validate_play_
        ledger` before being returned. Carries only stable scoring
        identifiers (`play_id`, `game_pk`, `at_bat_number`, `pitch_number`,
        `game_date`, `season`, `batter_id`) -- NOT `batter_name`/
        `batter_team`/`opponent_team`, which are presentation overlays
        applied after scoring and are therefore not canonical scoring
        truth (Phase 2 contract amendment; see `play_ledger_schema.py`'s
        module docstring). A future browser artifact enriches rows with
        those labels at the presentation layer by joining the same
        overlays the rest of the site already uses -- never inside this
        function.

    Raises:
        PlayLedgerExportError: if `df`/`ledger` are not index-aligned, or
            `df`/`ledger` is missing a required column.
        PlayLedgerValidationError: if the projected result fails schema
            validation (propagated from `validate_play_ledger`).
    """
    if not df.index.equals(ledger.index):
        raise PlayLedgerExportError("df and ledger must share the identical index/row order")

    missing_df = [c for c in _REQUIRED_DF_COLUMNS if c not in df.columns]
    if missing_df:
        raise PlayLedgerExportError(f"df is missing required column(s): {missing_df}")
    missing_ledger = [
        c
        for c in (
            _LEDGER_OBSERVED_COLUMN,
            _LEDGER_EXPECTED_COLUMN,
            _LEDGER_CONTACT_LUCK_COLUMN,
            *_LEDGER_PROBABILITY_COLUMNS,
        )
        if c not in ledger.columns
    ]
    if missing_ledger:
        raise PlayLedgerExportError(f"ledger is missing required column(s): {missing_ledger}")

    play_id = df["event_id"] if "event_id" in df.columns else build_event_id(df)

    out = pd.DataFrame(index=df.index)
    out["play_id"] = play_id.astype("string")
    out["game_pk"] = df["game_pk"].astype("Int64")
    out["at_bat_number"] = df["at_bat_number"].astype("Int64")
    out["pitch_number"] = df["pitch_number"].astype("Int64")
    out["game_date"] = df["game_date"].astype("string")
    out["season"] = df["season"].astype("Int64")
    out["batter_id"] = df["batter"].astype("Int64")

    out["stand"] = df["stand"].astype("string")
    out["launch_speed"] = df["launch_speed"].astype("Float64")
    out["launch_angle"] = df["launch_angle"].astype("Float64")
    out["bb_type"] = df["bb_type"].astype("string")
    out["spray_angle_approx"] = df["spray_angle_approx"].astype("Float64")
    out["hit_distance_sc"] = df["hit_distance_sc"].astype("Float64")

    # `is_scored` mirrors `aggregate_attribution.aggregate_to_batter_
    # season`'s OWN denominator rule EXACTLY (`resolved = ledger[
    # "observed_contact_result_run_value"].notna()`) -- Phase 2.5
    # amendment: keyed on `observed_run_value.notna()` primarily, so it
    # directly mirrors the published `eligible_batted_balls` denominator's
    # own defining column. See play_ledger_schema.py's `_validate_is_
    # scored_matches_resolution` for the enforced invariant.
    out["is_scored"] = ledger[_LEDGER_OBSERVED_COLUMN].notna().astype("boolean")

    # Everything below is copied verbatim from the ALREADY-SCORED ledger --
    # never recomputed, never masked/reconstructed. `outcome_class` comes
    # from `df` (the same column `build_attribution_ledger` itself read to
    # decide resolution), not re-derived here. `p_*` are native ledger
    # columns since Phase 3 (`attribution_ledger.py`), already nulled
    # together with every other result-linked column by that module's own
    # unresolved-nulling step -- no `.where(...)` masking needed here.
    out["outcome_class"] = df["outcome_class"].astype("string")
    out["observed_run_value"] = ledger[_LEDGER_OBSERVED_COLUMN].astype("Float64")
    for cls in CLASS_ORDER:
        out[f"p_{cls}"] = ledger[f"p_{cls}"].astype("Float64")
    out["expected_run_value"] = ledger[_LEDGER_EXPECTED_COLUMN].astype("Float64")
    out["contact_luck_runs"] = ledger[_LEDGER_CONTACT_LUCK_COLUMN].astype("Float64")

    out = out[list(PLAY_LEDGER_COLUMNS)]
    validate_play_ledger(out)
    return out
