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

## Phase 3.1: `observed_run_value`/`contact_luck_runs` corrected to Rf - E0

The real 2026-08-15 canonical production snapshot's per-batter reconciliation
against `public_score` FAILED for 261/630 batters on Total Contact Luck and
Runs/100 (BBE and Scored Games matched exactly). Root cause: this exporter
was sourcing `observed_run_value`/`contact_luck_runs` from `ledger["observed_
contact_result_run_value"]`/`ledger["contact_result_surprise"]` -- the raw
Version 0.2 quantity `Rc - E0`, which `attribution_ledger.py`'s own module
docstring explicitly documents as NOT the same as the published headline
metric whenever a play has nonzero defensive/advancement execution
contribution. Phase 2's development-data reconciliation never caught this
because it deliberately used a contact-only ledger (no outfield/infield/
advancement models), where `Rc - E0` and `Rf - E0` coincide trivially.

Fixed by re-sourcing both fields from the ledger's `observed_final_run_
value`/`final_result_surprise` columns (`Rf`/`Rf - E0` -- the FULL
telescoping-identity quantity `aggregate_attribution.aggregate_to_batter_
season` itself sums into `total_contact_luck_runs`/`contact_luck_runs_per_
100`). This is a pure constant-rename in this module (see `_LEDGER_OBSERVED_
COLUMN`/`_LEDGER_CONTACT_LUCK_COLUMN` below) -- `final_result_surprise` was
added to `attribution_ledger.py` as the single new computed field this fix
required; nothing here recomputes it. `play_ledger_schema.PLAY_LEDGER_
VERSION` was bumped to `"2.0"` for this semantic change -- see that
module's docstring for the outcome_class/Rc-vs-Rf finding and the sealed
v1.0 2026-08-15 snapshot's disposition.
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
#:
#: Phase 3.1: `_LEDGER_OBSERVED_COLUMN`/`_LEDGER_CONTACT_LUCK_COLUMN` are
#: deliberately Rf-based (`observed_final_run_value`/`final_result_
#: surprise`), NOT Rc-based (`observed_contact_result_run_value`/`contact_
#: result_surprise`) -- see module docstring for the real-production
#: reconciliation failure this corrects. `_LEDGER_EXPECTED_COLUMN` (E0) is
#: unchanged; the two quantities being differenced always share the same
#: baseline.
_LEDGER_OBSERVED_COLUMN = "observed_final_run_value"
_LEDGER_EXPECTED_COLUMN = "baseline_expected_contact_run_value"
_LEDGER_CONTACT_LUCK_COLUMN = "final_result_surprise"
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
    # season`'s OWN denominator rule (`resolved = ledger["observed_contact_
    # result_run_value"].notna()`) -- Phase 3.1 note: `_LEDGER_OBSERVED_
    # COLUMN` is now the Rf column (`observed_final_run_value`), not the Rc
    # column that rule literally names, but `attribution_ledger.py` nulls
    # both together for every unresolved row (rf starts as rc.copy() and
    # is only ever overwritten for rows already guaranteed resolved), so
    # their null patterns are always identical -- `is_scored`'s boolean
    # values are unaffected by this rename. See play_ledger_schema.py's
    # `_validate_is_scored_matches_resolution` for the enforced invariant.
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
