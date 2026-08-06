"""Contact Luck v0.11 Phase 3: additive batter-season aggregation.

Aggregates ONLY quantities that are additive in run-value units, straight
from Version 0.10's `mlb_luck_score.scoring.attribution_ledger` output --
this module computes no new predictions and alters no Version 0.2-0.10
component. Every non-additive quantity (a percentile, a public composite
score) is explicitly out of scope -- see `assert_additive_only` below, which
`run_season_aggregation.py` calls before handing anything to this module.

## Splitting `unexplained_residual` into a reported "contact" component

Version 0.10's own accounting identity has exactly THREE additive terms on
the right-hand side (see `attribution_ledger`'s module docstring):

    Rf - E0 = unexplained_residual + defensive_execution_contribution
                                    + advancement_execution_contribution

There is no independently-additive "contact" term in that identity --
`contact_result_surprise` (`Rc - E0`) is explicitly documented as NOT
additive alongside `defensive_execution_contribution` (it already contains
that contribution as a sub-component; see attribution_ledger's docstring on
why adding both would double-count). The task nonetheless asks for a
reported "contact component" alongside defensive execution, advancement, and
an unexplained residual, reconciling to the same observed-minus-expected
total. This module satisfies that WITHOUT inventing any new double-count by
PARTITIONING the existing `unexplained_residual` column (never altering any
individual row's value) into two named buckets, by row category:

  - `contact_component`: `unexplained_residual` on rows where NEITHER a
    defensive-execution model NOR the advancement model applies at all
    (`component_eligibility_status` starts with `STATUS_DEFENSE_UNAVAILABLE`
    and does not contain `STATUS_ADVANCEMENT_MODELED`). On these rows,
    Version 0.10's own test suite already proves `unexplained_residual`
    collapses EXACTLY to `contact_result_surprise`
    (`test_no_opportunity_or_advancement_model_supplied_yields_pure_contact_
    ledger`) -- i.e. this bucket really is "the Version 0.2 contact-luck
    signal, unmixed with anything else."
  - `unexplained_residual_component`: `unexplained_residual` on every OTHER
    resolved row (a genuine defense and/or advancement model applies) --
    this is the residual model-information-update term the ledger's
    docstring describes ((Eo-E0) + (Ea-Rc)), unrelated to any REALIZED
    execution event.

By construction, `contact_component + unexplained_residual_component` equals
the FULL `unexplained_residual` column exactly (it is a row-mask partition,
not a recomputation), so the identity below reduces algebraically to Version
0.10's own, unchanged:

    season_observed_minus_expected
      = season_contact_component + season_unexplained_residual_component
        + season_defensive_execution_component + season_advancement_component
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_luck_score.scoring.attribution_ledger import (
    IDENTITY_ATOL,
    STATUS_ADVANCEMENT_MODELED,
    STATUS_CONTACT_RESULT_UNAVAILABLE,
    STATUS_CONTACT_RESULT_UNAVAILABLE_OTHER,
    STATUS_DEFENSE_UNAVAILABLE,
)
from mlb_luck_score.scoring.component_confidence import (
    COMPONENT_ADVANCEMENT,
    COMPONENT_INFIELD_DEFENSE,
    COMPONENT_OUTFIELD_DEFENSE,
    DATA_QUALITY_EXCLUDED,
    DATA_QUALITY_FALLBACK,
    DATA_QUALITY_MISSING,
    MODEL_STATUS_UNAVAILABLE,
)

#: Additive component/total columns this module ever sums or averages.
#: `assert_additive_only` rejects anything outside this set from being fed
#: into the aggregation as if it were additive.
ADDITIVE_LEDGER_COLUMNS: tuple[str, ...] = (
    "baseline_expected_contact_run_value",
    "observed_contact_result_run_value",
    "observed_final_run_value",
    "defensive_execution_contribution",
    "advancement_execution_contribution",
    "unexplained_residual",
)

#: Explicitly NON-additive quantities that must NEVER be summed/averaged --
#: percentiles, ratios, and any public composite/display score. Listed here
#: so `assert_additive_only` can name the exact thing it rejected rather than
#: just refusing an unrecognized column.
KNOWN_NON_ADDITIVE_COLUMNS: tuple[str, ...] = (
    "contact_result_surprise",  # Rc - E0, documented as non-additive (double-counts)
    "defensive_opportunity_probability",  # a probability, not a run value
    "component_eligibility_status",
    "component_confidence_status",
)

PER_100_DENOMINATOR = 100.0

_PROVISIONAL_MODEL_STATUSES = frozenset(
    {
        "provisional",
        "calibrated_with_limited_subgroup_evidence",
        "not_calibrated",
    }
)


class AggregationError(ValueError):
    """Raised when inputs to the aggregation layer are invalid or non-additive."""


def assert_additive_only(columns: list[str] | tuple[str, ...]) -> None:
    """Fail loudly if any requested column is a KNOWN non-additive quantity.

    Does not require every column to be pre-registered in `ADDITIVE_LEDGER_
    COLUMNS` (a caller may aggregate a legitimate new additive quantity), but
    DOES hard-reject anything in `KNOWN_NON_ADDITIVE_COLUMNS` -- a percentile,
    a probability, or a status string summed/averaged as if numeric.
    """
    bad = [c for c in columns if c in KNOWN_NON_ADDITIVE_COLUMNS]
    if bad:
        raise AggregationError(
            f"Refusing to aggregate known non-additive column(s) as if additive: {bad}"
        )


def _contact_only_mask(ledger: pd.DataFrame) -> pd.Series:
    status = ledger["component_eligibility_status"]
    is_defense_unavailable = status.str.startswith(STATUS_DEFENSE_UNAVAILABLE, na=False)
    is_advancement_modeled = status.str.contains(STATUS_ADVANCEMENT_MODELED, regex=False, na=False)
    return is_defense_unavailable & ~is_advancement_modeled


def _unresolved_mask(ledger: pd.DataFrame) -> pd.Series:
    return ledger["component_eligibility_status"].isin(
        [STATUS_CONTACT_RESULT_UNAVAILABLE, STATUS_CONTACT_RESULT_UNAVAILABLE_OTHER]
    )


def build_play_level_components(ledger: pd.DataFrame) -> pd.DataFrame:
    """Build the four PARTITIONED, per-row additive component columns described
    in this module's docstring (index-aligned with `ledger`).

    Returns:
        A DataFrame with `observed_minus_expected`, `contact_component`,
        `unexplained_residual_component`, `defensive_execution_component`,
        `advancement_execution_component` -- all `NaN` for unresolved rows
        (matching `ledger`'s own convention), summing/skipna-summing
        correctly to reproduce Version 0.10's identity exactly.
    """
    contact_only = _contact_only_mask(ledger)
    unresolved = _unresolved_mask(ledger)
    other_resolved = ~unresolved & ~contact_only

    observed_minus_expected = (
        ledger["observed_final_run_value"] - ledger["baseline_expected_contact_run_value"]
    )
    contact_component = ledger["unexplained_residual"].where(contact_only)
    unexplained_residual_component = ledger["unexplained_residual"].where(other_resolved)
    defensive_execution_component = ledger["defensive_execution_contribution"]
    advancement_execution_component = ledger["advancement_execution_contribution"]

    return pd.DataFrame(
        {
            "observed_minus_expected": observed_minus_expected,
            "contact_component": contact_component,
            "unexplained_residual_component": unexplained_residual_component,
            "defensive_execution_component": defensive_execution_component,
            "advancement_execution_component": advancement_execution_component,
        },
        index=ledger.index,
    )


def verify_play_level_partition_identity(
    ledger: pd.DataFrame, components: pd.DataFrame, *, atol: float = IDENTITY_ATOL
) -> pd.Series:
    """Confirm `contact_component + unexplained_residual_component` reproduces
    the FULL `unexplained_residual` column exactly (row-mask partition, not a
    recomputation) -- `pd.NA` for unresolved rows, boolean otherwise.
    """
    resolved = ledger["observed_contact_result_run_value"].notna()
    reconstructed = components["contact_component"].fillna(0.0) + components[
        "unexplained_residual_component"
    ].fillna(0.0)
    holds = pd.Series(pd.NA, index=ledger.index, dtype="boolean")
    holds.loc[resolved] = np.isclose(
        reconstructed.loc[resolved].to_numpy(),
        ledger.loc[resolved, "unexplained_residual"].to_numpy(),
        atol=atol,
    )
    return holds


def _defense_and_advancement_field_by_row(
    df: pd.DataFrame,
    confidence: pd.DataFrame,
    *,
    field: str,
    unavailable_value: str,
) -> tuple[pd.Series, pd.Series]:
    """Reduce the long-format confidence table's `field` column to one
    `defense`-component value and one `advancement`-component value PER ROW
    of `df` (aligned to `df.index`), picking whichever of outfield/infield
    defense is not `unavailable_value` (mutually exclusive by construction --
    see `mlb_luck_score.features.build_contact_features.
    add_opportunity_features_by_domain`).
    """
    wide = confidence.pivot(index="event_id", columns="component_name", values=field)
    for col in (COMPONENT_OUTFIELD_DEFENSE, COMPONENT_INFIELD_DEFENSE, COMPONENT_ADVANCEMENT):
        if col not in wide.columns:
            wide[col] = unavailable_value

    defense_value = wide[COMPONENT_OUTFIELD_DEFENSE].where(
        wide[COMPONENT_OUTFIELD_DEFENSE] != unavailable_value,
        wide[COMPONENT_INFIELD_DEFENSE],
    )
    advancement_value = wide[COMPONENT_ADVANCEMENT]

    event_id_order = df["event_id"]
    defense_by_row = pd.Series(defense_value.reindex(event_id_order).to_numpy(), index=df.index)
    advancement_by_row = pd.Series(
        advancement_value.reindex(event_id_order).to_numpy(), index=df.index
    )
    return defense_by_row, advancement_by_row


def _defense_and_advancement_status_by_row(
    df: pd.DataFrame, confidence: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    return _defense_and_advancement_field_by_row(
        df, confidence, field="model_status_confidence", unavailable_value=MODEL_STATUS_UNAVAILABLE
    )


def aggregate_to_batter_season(
    df: pd.DataFrame,
    ledger: pd.DataFrame,
    confidence: pd.DataFrame,
    *,
    batter_column: str = "batter",
    season_column: str = "season",
    game_column: str = "game_pk",
    atol: float = IDENTITY_ATOL,
) -> pd.DataFrame:
    """Aggregate the play-level attribution ledger to one row per (batter, season).

    Args:
        df: Prepared DataFrame with `event_id`, `batter_column`, `season_
            column`, `game_column` -- same index/order as `ledger`.
        ledger: `mlb_luck_score.scoring.attribution_ledger.
            build_attribution_ledger`'s output.
        confidence: `mlb_luck_score.scoring.component_confidence.
            build_play_level_confidence`'s output (long format).
        atol: Tolerance for `verify_season_identity`'s own recheck (see that
            function) -- the SAME default as `attribution_ledger.
            IDENTITY_ATOL`, documented rather than silently reused.

    Returns:
        One row per (batter, season) with eligible-play/game counts, the
        four additive component totals plus their sum, per-100-eligible
        -play rates, pathway/provisional/unavailable counts, and the
        provisional-value share. See module docstring for the exact
        component-partition definitions.
    """
    if not df.index.equals(ledger.index):
        raise AggregationError("df and ledger must share the identical index/row order")

    required = (batter_column, season_column, game_column, "event_id")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AggregationError(f"aggregate_to_batter_season requires column(s) {missing} on df")

    components = build_play_level_components(ledger)
    resolved = ledger["observed_contact_result_run_value"].notna()

    defense_status_by_row, advancement_status_by_row = _defense_and_advancement_status_by_row(
        df, confidence
    )
    defense_provisional = defense_status_by_row.isin(_PROVISIONAL_MODEL_STATUSES)
    advancement_provisional = advancement_status_by_row.isin(_PROVISIONAL_MODEL_STATUSES)
    defense_unavailable = defense_status_by_row == MODEL_STATUS_UNAVAILABLE
    advancement_unavailable = advancement_status_by_row == MODEL_STATUS_UNAVAILABLE

    defense_quality_by_row, advancement_quality_by_row = _defense_and_advancement_field_by_row(
        df, confidence, field="row_data_quality", unavailable_value=DATA_QUALITY_EXCLUDED
    )
    defense_missing_input = defense_quality_by_row.isin(
        {DATA_QUALITY_MISSING, DATA_QUALITY_FALLBACK}
    )
    advancement_missing_input = advancement_quality_by_row.isin(
        {DATA_QUALITY_MISSING, DATA_QUALITY_FALLBACK}
    )
    any_missing_input = defense_missing_input | advancement_missing_input
    any_component_available = ~defense_unavailable | ~advancement_unavailable

    provisional_signed_value = components["defensive_execution_component"].where(
        defense_provisional, 0.0
    ).fillna(0.0) + components["advancement_execution_component"].where(
        advancement_provisional, 0.0
    ).fillna(0.0)
    provisional_value = provisional_signed_value.abs()
    total_abs_value = (
        components["contact_component"].fillna(0.0).abs()
        + components["unexplained_residual_component"].fillna(0.0).abs()
        + components["defensive_execution_component"].fillna(0.0).abs()
        + components["advancement_execution_component"].fillna(0.0).abs()
    )

    working = pd.DataFrame(
        {
            batter_column: df[batter_column].to_numpy(),
            season_column: df[season_column].to_numpy(),
            game_column: df[game_column].to_numpy(),
            "resolved": resolved.to_numpy(),
            "pathway": ledger["component_eligibility_status"].to_numpy(),
            "observed_minus_expected": components["observed_minus_expected"].to_numpy(),
            "contact_component": components["contact_component"].to_numpy(),
            "unexplained_residual_component": components[
                "unexplained_residual_component"
            ].to_numpy(),
            "defensive_execution_component": components["defensive_execution_component"].to_numpy(),
            "advancement_execution_component": components[
                "advancement_execution_component"
            ].to_numpy(),
            "defense_provisional": (resolved & defense_provisional).to_numpy(),
            "advancement_provisional": (resolved & advancement_provisional).to_numpy(),
            "defense_unavailable": (resolved & defense_unavailable).to_numpy(),
            "advancement_unavailable": (resolved & advancement_unavailable).to_numpy(),
            "any_missing_input": (resolved & any_missing_input).to_numpy(),
            "any_component_available": (resolved & any_component_available).to_numpy(),
            "provisional_value": provisional_value.to_numpy(),
            "provisional_signed_value": provisional_signed_value.to_numpy(),
            "total_abs_value": total_abs_value.to_numpy(),
        }
    )

    grouped = working.groupby([batter_column, season_column], dropna=False, sort=False)

    eligible_rows = working[working["resolved"]]
    eligible_grouped = eligible_rows.groupby(
        [batter_column, season_column], dropna=False, sort=False
    )

    summary = grouped.agg(
        eligible_batted_balls=("resolved", "sum"),
        total_observed_minus_expected_runs=("observed_minus_expected", "sum"),
        total_contact_component_runs=("contact_component", "sum"),
        total_unexplained_residual_component_runs=("unexplained_residual_component", "sum"),
        total_defensive_execution_component_runs=("defensive_execution_component", "sum"),
        total_advancement_component_runs=("advancement_execution_component", "sum"),
        defense_provisional_play_count=("defense_provisional", "sum"),
        advancement_provisional_play_count=("advancement_provisional", "sum"),
        defense_unavailable_play_count=("defense_unavailable", "sum"),
        advancement_unavailable_play_count=("advancement_unavailable", "sum"),
        missing_input_play_count=("any_missing_input", "sum"),
        component_available_play_count=("any_component_available", "sum"),
        provisional_value_abs=("provisional_value", "sum"),
        provisional_value_signed=("provisional_signed_value", "sum"),
        total_abs_value=("total_abs_value", "sum"),
    ).reset_index()

    summary["total_observed_minus_expected_excluding_provisional_runs"] = (
        summary["total_observed_minus_expected_runs"] - summary["provisional_value_signed"]
    )

    summary["missing_input_frequency"] = np.where(
        summary["eligible_batted_balls"] > 0,
        summary["missing_input_play_count"] / summary["eligible_batted_balls"],
        np.nan,
    )
    summary["component_coverage_fraction"] = np.where(
        summary["eligible_batted_balls"] > 0,
        summary["component_available_play_count"] / summary["eligible_batted_balls"],
        np.nan,
    )

    games = eligible_grouped[game_column].nunique().rename("games").reset_index()
    summary = summary.merge(games, on=[batter_column, season_column], how="left")
    summary["games"] = summary["games"].fillna(0).astype(int)

    # Built via a plain dict accumulation (not groupby().apply()) to stay
    # robust across pandas versions rather than depend on `include_groups`
    # semantics that changed between pandas 2.x and 3.x, and to sidestep
    # MultiIndex.items() typing gaps in the pandas stubs.
    pathway_size = (
        eligible_rows.groupby([batter_column, season_column, "pathway"], dropna=False, sort=False)
        .size()
        .reset_index(name="count")
    )
    pathway_dict_by_key: dict[tuple[object, object], dict[str, int]] = {}
    for batter_val, season_val, pathway_val, count in zip(
        pathway_size[batter_column].tolist(),
        pathway_size[season_column].tolist(),
        pathway_size["pathway"].tolist(),
        pathway_size["count"].tolist(),
        strict=True,
    ):
        pathway_dict_by_key.setdefault((batter_val, season_val), {})[str(pathway_val)] = int(count)
    pathway_counts_list: list[dict[str, int]] = [
        pathway_dict_by_key.get((batter_val, season_val), {})
        for batter_val, season_val in zip(
            summary[batter_column], summary[season_column], strict=True
        )
    ]
    summary["pathway_counts"] = pd.Series(pathway_counts_list, index=summary.index, dtype=object)

    summary["share_of_value_from_provisional_components"] = np.where(
        summary["total_abs_value"] > 0,
        summary["provisional_value_abs"] / summary["total_abs_value"],
        np.nan,
    )

    per_100_scale = np.where(
        summary["eligible_batted_balls"] > 0,
        PER_100_DENOMINATOR / summary["eligible_batted_balls"],
        np.nan,
    )
    for base_col, rate_col in (
        ("total_observed_minus_expected_runs", "observed_minus_expected_per_100"),
        ("total_contact_component_runs", "contact_component_per_100"),
        ("total_unexplained_residual_component_runs", "unexplained_residual_component_per_100"),
        ("total_defensive_execution_component_runs", "defensive_execution_component_per_100"),
        ("total_advancement_component_runs", "advancement_execution_component_per_100"),
        (
            "total_observed_minus_expected_excluding_provisional_runs",
            "observed_minus_expected_excluding_provisional_per_100",
        ),
    ):
        summary[rate_col] = summary[base_col] * per_100_scale

    summary = summary.drop(
        columns=["provisional_value_abs", "provisional_value_signed", "total_abs_value"]
    )
    return summary


def verify_season_identity(summary: pd.DataFrame, *, atol: float = IDENTITY_ATOL) -> pd.Series:
    """Recheck the season-level accounting identity described in this module's
    docstring holds for every batter-season row, within `atol`.

    Returns:
        A boolean Series, index-aligned with `summary` -- `True` means the
        four additive totals reconstruct `total_observed_minus_expected_
        runs` exactly (within tolerance). Rows with zero eligible batted
        balls hold trivially (all totals are exactly 0.0).
    """
    lhs = summary["total_observed_minus_expected_runs"].to_numpy()
    rhs = (
        summary["total_contact_component_runs"]
        + summary["total_unexplained_residual_component_runs"]
        + summary["total_defensive_execution_component_runs"]
        + summary["total_advancement_component_runs"]
    ).to_numpy()
    return pd.Series(np.isclose(lhs, rhs, atol=atol), index=summary.index)
