"""Contact Luck v0.11 Phase 4: game_pk-clustered bootstrap uncertainty.

Produces sampling-variability intervals for each batter-season's additive
totals and per-100-eligible-play rates, CONDITIONAL ON the frozen Version
0.2-0.10 models -- this module never refits, retrains, or recalibrates
anything. Every bootstrap replicate reuses the EXACT play-level predictions
`mlb_luck_score.scoring.attribution_ledger.build_attribution_ledger` already
computed; only which PLAYS are included in a given replicate changes.

## What these intervals are -- and are NOT

These are SAMPLING-VARIABILITY intervals: "if this exact batter had faced
this exact set of games again, with the SAME model predictions, how much
would the total plausibly vary just from which games happened to occur?"
They do NOT include:

  - model-specification uncertainty (a differently-specified contact/
    opportunity/advancement model could give a materially different
    prediction for the same play -- these intervals hold the model fixed);
  - measurement uncertainty (Statcast's own launch-speed/angle/spray-angle
    measurement error);
  - labeling uncertainty (`des`-string parse ambiguity, hit-location
    crediting).

A SEPARATE, explicitly-named model-estimation-uncertainty analysis (e.g.
refitting each component model inside every bootstrap replicate) would be
required to capture the first of those, and is NOT implemented here per the
task's explicit instruction.

## Method: game_pk-clustered, percentile intervals

Resampling unit is the GAME, not the individual play -- every play from a
resampled game moves together, preserving within-game correlation (same
park, weather, opponent, day) exactly as `mlb_luck_score.models.
compare_opportunity_models.compute_opportunity_bootstrap` and every other
existing bootstrap in this codebase already does. For a given batter-season
with `k` distinct games, replicate `r` draws `k` games WITH REPLACEMENT from
THAT BATTER's own `k` games (not from the league-wide game pool -- the
question being asked is "how much would THIS PLAYER's own total vary,"
which only depends on the (in)stability of their own game-to-game sample,
not on other players' games) and sums the batter's plays across the drawn
games (a game drawn twice counts twice).

**Interval method: PERCENTILE**, chosen deliberately and documented BEFORE
any real player-season result was examined (see Phase 6/9's real-data run,
executed only after this module was written and its tests passed):

  - Consistency: every existing bootstrap CI in this codebase (`compare_
    opportunity_models`, `compare_near_wall_calibration_gate`, `compare_
    infield_opportunity`, `compare_advancement_models`) already uses the
    plain percentile method (`np.nanquantile` at `alpha/2`/`1-alpha/2`) --
    matching it avoids introducing a second, inconsistent CI convention.
  - BCa (bias-corrected and accelerated) requires a jackknife acceleration
    estimate PER batter-season PER metric -- at real-data scale (hundreds of
    batters) this multiplies the compute cost by roughly the average games
    -per-batter with no house precedent to justify the added complexity.
  - The "basic" (reflection) method is less standard in this codebase and no
    more robust than percentile for the sample sizes qualification rules
    (Phase 5) already gate on.
  - Documented LIMITATION: percentile intervals are known to undercover for
    small, skewed samples -- this is exactly why Phase 5's qualification
    rules use minimum sample-size thresholds rather than treating "the
    interval doesn't error" as evidence of adequate support.

## Additivity within (not across) replicates

Within a SINGLE replicate, the resampled totals are constructed from the
SAME resampled game indices for every metric, so
`observed_minus_expected_rep == contact_component_rep + unexplained_
residual_component_rep + defensive_execution_component_rep + advancement_
execution_component_rep` EXACTLY for that replicate (see
`verify_bootstrap_replicate_additivity`). The reported marginal percentile
bounds of different metrics are NOT expected to satisfy that same identity
against each other -- a sum's interval is not the sum of its parts'
intervals; this is a standard, well-known property of intervals on
correlated quantities, not a defect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlb_luck_score.scoring.aggregate_attribution import build_play_level_components

#: Default bootstrap replication count -- documented, not hidden. 1000
#: matches the order of magnitude used elsewhere in this codebase (the
#: existing opportunity/near-wall/infield bootstraps default to 500) while
#: giving slightly finer percentile resolution for the 2.5/97.5 tails.
DEFAULT_N_BOOTSTRAP_REPS = 1000

#: Deterministic default seed -- documented so a re-run reproduces bit
#: -identical intervals given the same input ledger.
DEFAULT_BOOTSTRAP_SEED = 42

DEFAULT_ALPHA = 0.05  # 95% interval

METRIC_COLUMNS: tuple[str, ...] = (
    "observed_minus_expected",
    "contact_component",
    "unexplained_residual_component",
    "defensive_execution_component",
    "advancement_execution_component",
)


class AggregationUncertaintyError(ValueError):
    """Raised when inputs to the bootstrap layer are invalid."""


@dataclass(frozen=True)
class BootstrapDesign:
    """Documents the exact design choices behind a bootstrap run -- recorded
    verbatim in every report this module produces so a reader never has to
    guess the method/seed/replication count after the fact.
    """

    method: str = "percentile"
    resampling_unit: str = "game_pk_within_batter_season"
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS
    seed: int = DEFAULT_BOOTSTRAP_SEED
    alpha: float = DEFAULT_ALPHA
    refits_models: bool = False


def _player_game_totals(
    df: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    batter_column: str,
    season_column: str,
    game_column: str,
) -> pd.DataFrame:
    if not df.index.equals(ledger.index):
        raise AggregationUncertaintyError("df and ledger must share the identical index/row order")

    components = build_play_level_components(ledger)
    resolved = ledger["observed_contact_result_run_value"].notna()

    working = pd.DataFrame(
        {
            batter_column: df[batter_column].to_numpy(),
            season_column: df[season_column].to_numpy(),
            game_column: df[game_column].to_numpy(),
            "eligible": resolved.to_numpy().astype(float),
        }
    )
    for col in METRIC_COLUMNS:
        working[col] = components[col].fillna(0.0).to_numpy()

    return (
        working.groupby([batter_column, season_column, game_column], sort=True, dropna=False)[
            ["eligible", *METRIC_COLUMNS]
        ]
        .sum()
        .reset_index()
    )


def bootstrap_batter_season_intervals(
    df: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = DEFAULT_ALPHA,
    batter_column: str = "batter",
    season_column: str = "season",
    game_column: str = "game_pk",
) -> pd.DataFrame:
    """Game_pk-clustered percentile bootstrap for every batter-season in `df`.

    Args:
        df: Same as `mlb_luck_score.scoring.aggregate_attribution.
            aggregate_to_batter_season`'s `df` -- `batter_column`/`season_
            column`/`game_column` present, same index/order as `ledger`.
        ledger: `build_attribution_ledger`'s output.
        n_reps / seed / alpha: See `BootstrapDesign`. Iteration is over
            `groupby(..., sort=True)`, and the SAME `numpy.random.Generator`
            (seeded once) is advanced in that deterministic (batter, season)
            order -- re-running with the same inputs and seed reproduces
            bit-identical intervals regardless of the input row order.

    Returns:
        One row per (batter, season) with `n_games`, `n_reps`, `seed`, and
        `f"{metric}_ci_low"`/`f"{metric}_ci_high"` for every `METRIC_
        COLUMNS` entry plus its per-100-eligible-play-rate counterpart
        (`f"{metric}_per_100_ci_low"`/`..._ci_high"`). A batter-season with
        zero games gets NaN bounds (see `verify_interval_width_shrinks_with_
        sample_size` for the expected width-vs-`n_games` relationship this
        should exhibit on real, well-behaved data).
    """
    if n_reps < 1:
        raise AggregationUncertaintyError(f"n_reps must be >= 1, got {n_reps}")

    game_totals = _player_game_totals(
        df,
        ledger,
        batter_column=batter_column,
        season_column=season_column,
        game_column=game_column,
    )
    metric_cols = ["eligible", *METRIC_COLUMNS]

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (batter_val, season_val), group in game_totals.groupby(
        [batter_column, season_column], sort=True
    ):
        values = group[metric_cols].to_numpy()
        k = values.shape[0]
        row: dict[str, object] = {
            batter_column: batter_val,
            season_column: season_val,
            "n_games": k,
            "n_reps": n_reps,
            "seed": seed,
        }
        if k == 0:
            for col in metric_cols[1:]:
                row[f"{col}_ci_low"] = float("nan")
                row[f"{col}_ci_high"] = float("nan")
                row[f"{col}_per_100_ci_low"] = float("nan")
                row[f"{col}_per_100_ci_high"] = float("nan")
            rows.append(row)
            continue

        idx = rng.integers(0, k, size=(n_reps, k))
        resampled = values[idx].sum(axis=1)  # shape (n_reps, len(metric_cols))
        eligible_rep = resampled[:, 0]

        for i, col in enumerate(metric_cols[1:], start=1):
            reps = resampled[:, i]
            row[f"{col}_ci_low"] = float(np.nanquantile(reps, alpha / 2))
            row[f"{col}_ci_high"] = float(np.nanquantile(reps, 1.0 - alpha / 2))

            with np.errstate(divide="ignore", invalid="ignore"):
                rate_reps = np.where(eligible_rep > 0, reps * 100.0 / eligible_rep, np.nan)
            if np.isfinite(rate_reps).any():
                row[f"{col}_per_100_ci_low"] = float(np.nanquantile(rate_reps, alpha / 2))
                row[f"{col}_per_100_ci_high"] = float(np.nanquantile(rate_reps, 1.0 - alpha / 2))
            else:
                row[f"{col}_per_100_ci_low"] = float("nan")
                row[f"{col}_per_100_ci_high"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


def verify_bootstrap_replicate_additivity(
    df: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    n_reps: int = 200,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    atol: float = 1e-9,
    batter_column: str = "batter",
    season_column: str = "season",
    game_column: str = "game_pk",
) -> bool:
    """Confirm that WITHIN every individual replicate (not across the reported
    marginal percentile bounds -- see module docstring), `observed_minus_
    expected_rep` equals the sum of the four component replicates exactly.

    Intended for tests/diagnostics, not for production reporting -- reruns
    the resampling with its own smaller `n_reps` for speed.
    """
    game_totals = _player_game_totals(
        df,
        ledger,
        batter_column=batter_column,
        season_column=season_column,
        game_column=game_column,
    )
    metric_cols = ["eligible", *METRIC_COLUMNS]
    rng = np.random.default_rng(seed)
    for _, group in game_totals.groupby([batter_column, season_column], sort=True):
        values = group[metric_cols].to_numpy()
        k = values.shape[0]
        if k == 0:
            continue
        idx = rng.integers(0, k, size=(n_reps, k))
        resampled = values[idx].sum(axis=1)
        lhs = resampled[:, metric_cols.index("observed_minus_expected")]
        rhs = (
            resampled[:, metric_cols.index("contact_component")]
            + resampled[:, metric_cols.index("unexplained_residual_component")]
            + resampled[:, metric_cols.index("defensive_execution_component")]
            + resampled[:, metric_cols.index("advancement_execution_component")]
        )
        if not np.allclose(lhs, rhs, atol=atol):
            return False
    return True
