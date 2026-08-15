"""Contact Luck v0.10: play-level attribution ledger and double-counting audit.

Reconciles Version 0.2's raw contact luck, Version 0.7B/0.8's defensive
execution, and Version 0.9's advancement execution into ONE run-value
accounting identity, per play. This module does NOT publish a combined
score -- it is an audit/reconciliation tool: every reported component is
either a genuinely new, non-overlapping quantity, or is explicitly derived
FROM another reported quantity (documented as such), never silently
re-adding the same information twice. Weather (Version 0.5/0.5.1) and
alignment (Version 0.6) are excluded entirely -- neither passed adoption
(see CLAUDE.md/AGENTS.md "Never silently adopt a candidate model variant as
the default") -- they do not appear anywhere in this module.

## The double-counting risk this module resolves

Three existing component reports each answer a question that PARTIALLY
overlaps the others, all ultimately explaining the same observed run value:

  - `mlb_luck_score.scoring.contact_luck` / `air_ball_components` /
    `infield_ball_components`: `residual_contact_luck_runs = actual_run_
    value - expected_run_value_contact_model`, where `actual_run_value`
    already reflects the FINAL RECORDED HIT OUTCOME (`outcome_class`) --
    i.e. it already bakes in whether the ball became an out or a hit, and
    if a hit, which type.
  - `mlb_luck_score.scoring.defensive_execution` / `infield_execution`:
    `actual_out - p_out_opportunity`, in PROBABILITY units -- ALSO answers
    "did this become an out or a hit," the exact same question `outcome_
    class` already answers, just from a different (opportunity-model)
    vantage point and in different units.
  - `mlb_luck_score.scoring.advancement_execution`: uses the batter's own
    hit type (`hit_type_group`) as a FEATURE and reports `batter_runner_
    advancement_execution` in BASE units (0-4), a third unit system.

Naively summing "raw contact luck" + "defensive execution" + "advancement
execution" would double-count the out-vs-hit question (both raw contact
luck and defensive execution independently explain it) and would mix THREE
incompatible unit systems (runs / probability / bases). This module fixes
both problems: (1) every component is converted to RUN-VALUE units via
`FINAL_BASE_RUN_VALUE_MAP` below, and (2) the components are built as a
TELESCOPING chain of successive conditional expectations, so each unit of
"surprise" is attributed to exactly ONE stage, never re-attributed to a
later one -- the SAME telescoping-sum technique that guarantees an exact
identity by construction, not by post-hoc rebalancing.

## The telescoping chain (all quantities in RUNS)

    E0  = baseline_expected_contact_run_value   (Version 0.2 contact model, unconditional)
    Rc  = observed_contact_result_run_value     (Version 0.1 outcome_class, via DEFAULT_RUN_VALUE_MAP)
    Eo  = opportunity-adjusted expected run value (contact model's own conditional
          single/double/triple/home_run distribution, reweighted by the opportunity
          model's P(out) -- STILL pre-outcome, no realized information used)
    Ea  = expected_advancement_value            (Version 0.9 model's predicted distribution
          over final bases, converted to runs via FINAL_BASE_RUN_VALUE_MAP)
    Rf  = observed final run value INCLUDING advancement (FINAL_BASE_RUN_VALUE_MAP applied
          to the batter-runner's actual final base -- equals Rc when advancement is not
          modeled for this row, e.g. infield hits, popups, or non-advancement-eligible
          outfield plays such as a plain over-the-fence home run or a batter retired on
          the batted ball itself)

    defensive_execution_contribution = Rc - Eo   -- defense's REALIZED performance vs.
                                                     the opportunity model's PRE-outcome estimate
    advancement_execution_contribution = Rf - Ea -- the batter-runner's REALIZED advancement
                                                     vs. the advancement model's PRE-outcome estimate
    unexplained_residual = (Rf - E0) - defensive_execution_contribution
                                       - advancement_execution_contribution
                          = (Eo - E0) + (Ea - Rc)   [derivable; see below]

By construction:

    Rf - E0 = unexplained_residual + defensive_execution_contribution
                                    + advancement_execution_contribution

This is THE accounting identity this module guarantees (`tests/
test_attribution_ledger.py` checks it numerically on every synthetic
scenario). `unexplained_residual` is not a dumping ground for error -- it is
the EXACT, well-defined sum of two real, interpretable, non-execution
quantities: `(Eo - E0)`, the shift in expectation from incorporating the
opportunity model's richer PRE-outcome difficulty assessment (hang time,
wall distance, alignment, ...) beyond what the contact model alone knows;
and `(Ea - Rc)`, the average batter-runner's PRE-outcome expected extra
value beyond the recorded hit type, given context. Neither corresponds to a
REALIZED execution event (nobody DID anything at either step -- these are
purely model-information updates), which is exactly why they are reported
together as "unexplained" rather than attributed to defense or the batter.

`contact_result_surprise = Rc - E0` (`residual_contact_luck_runs`, UNCHANGED
formula) is reported for continuity/interpretability but is NOT an
independent additive term in the identity above -- it already contains
`defensive_execution_contribution` as a sub-component (`Rc - E0 = (Eo - E0)
+ defensive_execution_contribution`), so adding it AND defensive_execution_
contribution together would be the exact double-count this module exists to
prevent.

## Scope and honest gaps

  - Reached-on-error rows (`events == "field_error"`): Version 0.1's
    `outcome_class` is deliberately left null for these (the batter's
    resulting base cannot be reliably determined from public data for
    run-value purposes -- see `mlb_luck_score.eligibility`). `Rc` is
    therefore unavailable, and this module reports the ENTIRE ledger row as
    `contact_result_unavailable_reached_on_error` rather than approximating
    `Rc` from Version 0.9's `hit_type_implied_floor_base` -- that would be
    a genuinely new, unvalidated assumption this module does not make.
  - Advancement is modeled ONLY for outfield air balls (Version 0.9 scope).
    For every other row (ground balls, popups, non-advancement-eligible
    outfield plays), `Rf` is defined equal to `Rc` -- i.e. `advancement_
    execution_contribution = 0` EXACTLY, not NaN, and `component_
    eligibility_status` says `advancement_not_modeled_for_this_play`
    explicitly, so this is never mistaken for "confirmed zero advancement
    occurred" as a factual claim.
  - `defensive_execution_contribution` requires an opportunity model
    (outfield Version 0.7A/0.7C or infield Version 0.8) to be eligible;
    otherwise it is `NaN` and `component_eligibility_status` records why.
  - `component_eligibility_status`/`component_confidence_status` surface
    the SAME provisional/calibrated statuses already computed elsewhere
    (`mlb_luck_score.scoring.gated_outfield_report`, `infield_ball_
    components`, `advancement_execution`) -- reused, never recomputed.

2025 stays untouched (this module makes no season-list changes and has no
CLI of its own to guard).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_luck_score.config import CLASS_ORDER
from mlb_luck_score.eligibility import ADVANCEMENT_LABELS
from mlb_luck_score.models.compare_advancement_models import ADVANCEMENT_BASE_VALUE
from mlb_luck_score.models.train_advancement_model import (
    TrainedAdvancementModel,
    predict_advancement_proba,
    validate_advancement_probabilities,
)
from mlb_luck_score.models.train_contact_model import (
    TrainedModel,
    predict_proba_ordered,
    validate_probabilities,
)
from mlb_luck_score.models.train_opportunity_model import (
    TrainedOpportunityModel,
    predict_opportunity_proba,
    validate_opportunity_probabilities,
)
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP
from mlb_luck_score.scoring.weather_attribution import compute_expected_run_value_vectorized

#: Reuses (never redefines) `DEFAULT_RUN_VALUE_MAP` -- reindexed by FINAL
#: BASE count (0=out/retired-while-advancing, 1=single-equivalent, ...,
#: 4=home-run-equivalent) instead of by hit-type label. A batter-runner
#: ending at 2nd base has approximately the same run value whether they got
#: there via an ordinary double or a single-plus-advance -- a standard,
#: documented sabermetric approximation, not a new run-value formula.
FINAL_BASE_RUN_VALUE_MAP: dict[int, float] = {
    0: DEFAULT_RUN_VALUE_MAP["out"],
    1: DEFAULT_RUN_VALUE_MAP["single"],
    2: DEFAULT_RUN_VALUE_MAP["double"],
    3: DEFAULT_RUN_VALUE_MAP["triple"],
    4: DEFAULT_RUN_VALUE_MAP["home_run"],
}

STATUS_CONTACT_RESULT_UNAVAILABLE = "contact_result_unavailable_reached_on_error"
STATUS_CONTACT_RESULT_UNAVAILABLE_OTHER = "contact_result_unavailable_other"
STATUS_DEFENSE_OUTFIELD = "defense_outfield_opportunity"
STATUS_DEFENSE_INFIELD = "defense_infield_opportunity"
STATUS_DEFENSE_UNAVAILABLE = "defense_opportunity_unavailable"
STATUS_ADVANCEMENT_MODELED = "advancement_modeled"
STATUS_ADVANCEMENT_NOT_MODELED = "advancement_not_modeled_for_this_play"

#: Numerical tolerance for the accounting identity check -- generous enough
#: to absorb float64 summation error over the handful of terms involved,
#: tight enough to catch a genuine logic bug (see `tests/
#: test_attribution_ledger.py::test_identity_tolerance_is_tight`).
IDENTITY_ATOL = 1e-9


class AttributionLedgerError(ValueError):
    """Raised when inputs to the attribution ledger are invalid."""


def assert_unique_play_identifiers(df: pd.DataFrame, *, id_column: str = "event_id") -> None:
    """Fail loudly if `id_column` is missing, null, or duplicated anywhere in `df`.

    `event_id` (`mlb_luck_score.data.clean_batted_balls.build_event_id`, a
    deterministic `game_pk`-`at_bat_number`-`pitch_number` composite) is the
    real-world play identifier -- unlike `df.index` (typically a plain
    `RangeIndex` after concatenation/filtering, which is essentially never
    duplicated by construction and so would not catch a genuine upstream
    ETL bug), a duplicate or null `event_id` is a real defect: a double
    -counted play, a bad join, or a row that lost its identity somewhere
    upstream. `build_attribution_ledger` calls this before doing any
    accounting so such a defect is caught immediately, not laundered into a
    silently wrong reconciliation total downstream.

    Raises:
        AttributionLedgerError: if `id_column` is missing from `df`, if any
            value is null, or if any value is duplicated.
    """
    if id_column not in df.columns:
        raise AttributionLedgerError(f"df is missing the play-identifier column {id_column!r}")

    null_mask = df[id_column].isna()
    if null_mask.any():
        raise AttributionLedgerError(
            f"{int(null_mask.sum())} row(s) have a null {id_column!r} -- every play must "
            "have a resolved identifier before it can enter the attribution ledger"
        )

    duplicated_mask = df[id_column].duplicated(keep=False)
    if duplicated_mask.any():
        dupes = df.loc[duplicated_mask, id_column].unique().tolist()
        raise AttributionLedgerError(
            f"{len(dupes)} duplicate {id_column!r} value(s) found -- every play must be "
            f"represented exactly once (first 10: {dupes[:10]})"
        )


def _not_out_conditional_expected_run_value(contact_proba: pd.DataFrame) -> pd.Series:
    """`E[run value | not out]` from the contact model's OWN conditional distribution:
    renormalize `single`/`double`/`triple`/`home_run` probabilities to sum to 1
    (excluding `out`), then take the run-value-weighted expectation.
    """
    hit_classes = [c for c in CLASS_ORDER if c != "out"]
    hit_mass = contact_proba[hit_classes].sum(axis=1)
    run_values = np.array([DEFAULT_RUN_VALUE_MAP[c] for c in hit_classes])
    weighted = contact_proba[hit_classes].to_numpy() @ run_values
    with np.errstate(invalid="ignore", divide="ignore"):
        conditional = weighted / hit_mass.to_numpy()
    return pd.Series(conditional, index=contact_proba.index)


def compute_opportunity_adjusted_expected_run_value(
    contact_proba: pd.DataFrame, p_out_opportunity: pd.Series
) -> pd.Series:
    """`Eo = p_out_opportunity * run_value(out) + (1 - p_out_opportunity) * E[rv | not out]`.

    Still entirely pre-outcome -- uses only the contact model's own
    conditional distribution and the opportunity model's P(out), never the
    realized result.
    """
    e_not_out = _not_out_conditional_expected_run_value(contact_proba)
    p_out = p_out_opportunity.reindex(contact_proba.index)
    not_out_weight = 1 - p_out
    # IEEE float arithmetic does NOT simplify 0 * NaN to 0 -- when p_out is
    # exactly 1 (or the contact model's own hit mass is exactly 0, making
    # e_not_out undefined via 0/0), the "not out" branch has ZERO weight and
    # must contribute exactly 0 regardless of e_not_out's value, not NaN.
    not_out_term = pd.Series(
        np.where(not_out_weight.to_numpy() == 0.0, 0.0, (not_out_weight * e_not_out).to_numpy()),
        index=p_out.index,
    )
    return p_out * DEFAULT_RUN_VALUE_MAP["out"] + not_out_term


def compute_expected_advancement_value_runs(advancement_proba: pd.DataFrame) -> pd.Series:
    """`Ea = sum(P(label) * FINAL_BASE_RUN_VALUE_MAP[ADVANCEMENT_BASE_VALUE[label]])`."""
    if list(advancement_proba.columns) != list(ADVANCEMENT_LABELS):
        raise AttributionLedgerError(
            f"Expected advancement_proba columns {ADVANCEMENT_LABELS}, "
            f"got {list(advancement_proba.columns)}"
        )
    run_values = np.array(
        [
            FINAL_BASE_RUN_VALUE_MAP[int(ADVANCEMENT_BASE_VALUE[label])]
            for label in ADVANCEMENT_LABELS
        ]
    )
    values = advancement_proba.to_numpy() @ run_values
    return pd.Series(values, index=advancement_proba.index)


def build_attribution_ledger(
    df: pd.DataFrame,
    contact_trained: TrainedModel,
    *,
    outfield_trained: TrainedOpportunityModel | None = None,
    infield_trained: TrainedOpportunityModel | None = None,
    advancement_trained: TrainedAdvancementModel | None = None,
    outfield_confidence_status: str | None = None,
    infield_confidence_status: str | None = None,
    advancement_confidence_status: str | None = None,
) -> pd.DataFrame:
    """Build the per-play attribution ledger for every row of `df`.

    Args:
        df: Rows already through `mlb_luck_score.eligibility.compute_
            eligibility` (for `outcome_class`/`eligible_for_training`), and,
            where applicable, `add_outfield_opportunity_eligibility` /
            `add_infield_opportunity_eligibility` / `add_advancement_
            eligibility` (+ their corresponding `add_*_features`, preferably
            via `mlb_luck_score.features.build_contact_features.
            add_opportunity_features_by_domain` for the opportunity pair) --
            must have every feature column each supplied trained model
            needs, plus a unique, non-null `event_id` per row (see
            `assert_unique_play_identifiers`, called first and fails loudly
            on a missing/null/duplicated identifier) and no row eligible
            for both the outfield and infield opportunity domains at once.
        contact_trained: A fitted 5-class contact model, typically
            `baseline_v02`.
        outfield_trained: A fitted Version 0.7A/0.7C outfield opportunity
            model, or `None` if unavailable (defensive components will be
            `NaN` for outfield rows).
        infield_trained: A fitted Version 0.8 infield opportunity model, or
            `None` if unavailable (defensive components will be `NaN` for
            infield rows).
        advancement_trained: A fitted Version 0.9 advancement model, or
            `None` if unavailable (advancement components will be `0`/
            `advancement_not_modeled_for_this_play` for every row).
        outfield_confidence_status / infield_confidence_status /
            advancement_confidence_status: the REUSED, already-computed
            gate verdicts from each phase (`mlb_luck_score.models.
            compare_near_wall_models.summarize_near_wall_validation`'s
            `near_wall_specialist_calibrated`-style status, `mlb_luck_
            score.models.compare_infield_opportunity.
            summarize_infield_calibration`'s `overall_status`, `mlb_luck_
            score.models.compare_advancement_models.
            summarize_advancement_calibration`'s `overall_status`) --
            recorded verbatim into `component_confidence_status`, never
            recomputed here. `None` (the default) means "not supplied,"
            reported as `"not_supplied"`, distinct from an actual
            `not_calibrated` gate verdict.

    Returns:
        A DataFrame (same index as `df`) with `baseline_expected_contact_
        run_value`, `p_out`/`p_single`/`p_double`/`p_triple`/`p_home_run`
        (the contact model's own probability vector, re-labeled verbatim --
        added in Version 1.4.0 for the play-ledger exporter's use, computed
        via the SAME single `predict_proba_ordered` call `e0` already uses,
        never a second inference pass), `observed_contact_result_run_value`,
        `contact_result_surprise`, `defensive_opportunity_probability`,
        `defensive_execution_contribution`, `expected_advancement_value`,
        `advancement_execution_contribution`, `unexplained_residual`,
        `observed_final_run_value`, `component_eligibility_status`,
        `component_confidence_status`. See module docstring for the exact
        accounting identity these reconcile to. The `p_*` columns are
        nulled together with every other result-linked column for a row
        with an unresolved `outcome_class` -- see the unresolved-nulling
        loop below.
    """
    required = ("event_id", "outcome_class", "bb_type", "events")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AttributionLedgerError(f"build_attribution_ledger requires column(s) {missing}")

    assert_unique_play_identifiers(df)

    if (
        "outfield_opportunity_eligible" in df.columns
        and "infield_opportunity_eligible" in df.columns
    ):
        both_eligible = df["outfield_opportunity_eligible"].astype(bool) & df[
            "infield_opportunity_eligible"
        ].astype(bool)
        if both_eligible.any():
            bad_ids = df.loc[both_eligible, "event_id"].tolist()
            raise AttributionLedgerError(
                f"{int(both_eligible.sum())} row(s) are eligible for BOTH outfield and "
                "infield opportunity domains simultaneously -- refusing to silently route "
                f"them to only one domain (event_id(s): {bad_ids[:10]})"
            )

    idx = df.index

    contact_feature_cols = contact_trained.numeric_features + contact_trained.categorical_features
    contact_proba = predict_proba_ordered(contact_trained, df[contact_feature_cols])
    validate_probabilities(contact_proba)
    e0 = compute_expected_run_value_vectorized(contact_proba)

    # Version 1.4.0 (Play Explorer foundation): retain the contact model's
    # OWN already-computed 5-class probability vector on the returned
    # ledger -- `contact_proba` above is the ONLY `predict_proba_ordered`
    # call in this function; these five columns are a pure re-labeling
    # (`p_<class>`) of its existing values, never a second inference pass.
    # They participate in the SAME unresolved-nulling loop below as every
    # other result-linked ledger column, so a downstream consumer never
    # observes a probability for a play whose outcome_class is unresolved.
    p_columns: dict[str, pd.Series] = {f"p_{cls}": contact_proba[cls].copy() for cls in CLASS_ORDER}

    has_outcome_class = df["outcome_class"].notna()
    is_reached_on_error = df["events"] == "field_error"
    rc = df["outcome_class"].astype(object).map(DEFAULT_RUN_VALUE_MAP).astype(float)

    p_out_opportunity = pd.Series(np.nan, index=idx)
    defense_status = pd.Series(STATUS_DEFENSE_UNAVAILABLE, index=idx, dtype=object)

    if outfield_trained is not None and "outfield_opportunity_eligible" in df.columns:
        mask = df["outfield_opportunity_eligible"].astype(bool)
        if mask.any():
            feature_cols = outfield_trained.numeric_features + outfield_trained.categorical_features
            p = predict_opportunity_proba(outfield_trained, df.loc[mask, feature_cols])
            validate_opportunity_probabilities(p)
            p_out_opportunity.loc[mask] = p
            defense_status.loc[mask] = STATUS_DEFENSE_OUTFIELD

    if infield_trained is not None and "infield_opportunity_eligible" in df.columns:
        # The `both_eligible` check above already guarantees outfield- and
        # infield-eligible rows are disjoint, so this mask is exactly the
        # infield-eligible rows, not a "still unclaimed" fallback -- kept as
        # an explicit AND (rather than trusting disjointness silently) so a
        # future change to that invariant fails the check above first,
        # never a silent double-route here.
        mask = df["infield_opportunity_eligible"].astype(bool) & (
            defense_status == STATUS_DEFENSE_UNAVAILABLE
        )
        if mask.any():
            feature_cols = infield_trained.numeric_features + infield_trained.categorical_features
            p = predict_opportunity_proba(infield_trained, df.loc[mask, feature_cols])
            validate_opportunity_probabilities(p)
            p_out_opportunity.loc[mask] = p
            defense_status.loc[mask] = STATUS_DEFENSE_INFIELD

    has_defense = p_out_opportunity.notna()
    eo = pd.Series(np.nan, index=idx)
    if has_defense.any():
        eo.loc[has_defense] = compute_opportunity_adjusted_expected_run_value(
            contact_proba.loc[has_defense], p_out_opportunity.loc[has_defense]
        )
    defensive_execution_contribution = rc - eo

    advancement_status = pd.Series(STATUS_ADVANCEMENT_NOT_MODELED, index=idx, dtype=object)
    ea = pd.Series(np.nan, index=idx)
    rf = rc.copy()

    if advancement_trained is not None and "advancement_eligible" in df.columns:
        mask = df["advancement_eligible"].astype(bool)
        if mask.any():
            feature_cols = (
                advancement_trained.numeric_features + advancement_trained.categorical_features
            )
            proba = predict_advancement_proba(advancement_trained, df.loc[mask, feature_cols])
            validate_advancement_probabilities(proba)
            ea.loc[mask] = compute_expected_advancement_value_runs(proba)
            actual_base = df.loc[mask, "batter_final_base"].map(ADVANCEMENT_BASE_VALUE)
            rf.loc[mask] = actual_base.map(FINAL_BASE_RUN_VALUE_MAP).astype(float)
            advancement_status.loc[mask] = STATUS_ADVANCEMENT_MODELED

    advancement_execution_contribution = rf - ea
    advancement_execution_contribution = advancement_execution_contribution.where(
        advancement_status == STATUS_ADVANCEMENT_MODELED, 0.0
    )

    contact_result_surprise = rc - e0
    unexplained_residual = (
        (rf - e0)
        - defensive_execution_contribution.fillna(0.0)
        - advancement_execution_contribution.fillna(0.0)
    )

    component_status = pd.Series(
        [
            f"{d};{a}"
            for d, a in zip(defense_status.to_numpy(), advancement_status.to_numpy(), strict=True)
        ],
        index=idx,
    )
    unresolved = ~has_outcome_class
    # Every row with no resolved outcome_class gets an explicit "contact
    # result unavailable" status -- field_error has a documented reason
    # (STATUS_CONTACT_RESULT_UNAVAILABLE); anything else unresolved (rare:
    # an outcome_class-ineligible event was passed in) gets a generic one,
    # rather than silently keeping a defense/advancement status string that
    # would misleadingly imply the row's ledger is fully resolved.
    component_status = component_status.where(
        ~is_reached_on_error, STATUS_CONTACT_RESULT_UNAVAILABLE
    )
    component_status = component_status.where(
        ~(unresolved & ~is_reached_on_error), STATUS_CONTACT_RESULT_UNAVAILABLE_OTHER
    )

    def _confidence_for_row(defense_kind: str, advancement_kind: str) -> str:
        defense_conf = "not_applicable"
        if defense_kind == STATUS_DEFENSE_OUTFIELD:
            defense_conf = outfield_confidence_status or "not_supplied"
        elif defense_kind == STATUS_DEFENSE_INFIELD:
            defense_conf = infield_confidence_status or "not_supplied"
        advancement_conf = "not_applicable"
        if advancement_kind == STATUS_ADVANCEMENT_MODELED:
            advancement_conf = advancement_confidence_status or "not_supplied"
        return f"defense={defense_conf};advancement={advancement_conf}"

    component_confidence_status = pd.Series(
        [
            _confidence_for_row(d, a)
            for d, a in zip(defense_status.to_numpy(), advancement_status.to_numpy(), strict=True)
        ],
        index=idx,
    )

    for series in (
        rc,
        e0,
        contact_result_surprise,
        defensive_execution_contribution,
        rf,
        advancement_execution_contribution,
        unexplained_residual,
        *p_columns.values(),
    ):
        series.loc[unresolved] = np.nan

    return pd.DataFrame(
        {
            "baseline_expected_contact_run_value": e0,
            **p_columns,
            "observed_contact_result_run_value": rc,
            "contact_result_surprise": contact_result_surprise,
            "defensive_opportunity_probability": p_out_opportunity,
            "defensive_execution_contribution": defensive_execution_contribution,
            "expected_advancement_value": ea,
            "advancement_execution_contribution": advancement_execution_contribution,
            "observed_final_run_value": rf,
            "unexplained_residual": unexplained_residual,
            "component_eligibility_status": component_status,
            "component_confidence_status": component_confidence_status,
        },
        index=idx,
    )


def verify_attribution_identity(ledger: pd.DataFrame, *, atol: float = IDENTITY_ATOL) -> pd.Series:
    """`observed_final_run_value - baseline_expected_contact_run_value` must equal
    `unexplained_residual + defensive_execution_contribution + advancement_execution_
    contribution`, within `atol`, for every row with a resolved `observed_contact_
    result_run_value`.

    Returns:
        A boolean Series (`True` = identity holds), `NaN`-row-aligned:
        rows with an unavailable `observed_contact_result_run_value`
        (reached-on-error, or unresolved `outcome_class`) are excluded
        (returned as `pd.NA`) rather than asserted True/False.
    """
    resolved = ledger["observed_contact_result_run_value"].notna()
    lhs = ledger["observed_final_run_value"] - ledger["baseline_expected_contact_run_value"]
    rhs = (
        ledger["unexplained_residual"]
        + ledger["defensive_execution_contribution"].fillna(0.0)
        + ledger["advancement_execution_contribution"].fillna(0.0)
    )
    holds = pd.Series(pd.NA, index=ledger.index, dtype="boolean")
    holds.loc[resolved] = np.isclose(
        lhs.loc[resolved].to_numpy(), rhs.loc[resolved].to_numpy(), atol=atol
    )
    return holds
