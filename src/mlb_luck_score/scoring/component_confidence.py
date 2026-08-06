"""Contact Luck v0.11 Phase 1-2: confidence taxonomy and play-level confidence records.

READ-ONLY interpretation layer over the FROZEN Version 0.2-0.10 pipeline. This
module trains, refits, recalibrates, or alters NOTHING -- it only reads
`mlb_luck_score.scoring.attribution_ledger`'s already-computed output columns
(`component_eligibility_status`, `component_confidence_status`) and a handful
of already-existing, already-documented eligibility/gating columns (`outfield_
opportunity_exclusion_reason`, `near_wall_20ft`, `sprint_speed`, ...) and
re-expresses them as four SEPARATE, independent taxonomies rather than one
collapsed confidence number -- per the task's explicit instruction not to
report "high confidence" for a row in an uncertain subgroup just because the
model's AGGREGATE metrics look good.

## The four taxonomies (never merged into one number)

  A. `MODEL_STATUS_*` -- model-status confidence: is the MODEL believed
     calibrated overall, for this play's specific subgroup?
  B. `DATA_QUALITY_*` -- row-level data quality: are this row's OWN input
     features directly measured, physics-estimated, imputed/fallback,
     genuinely missing, or is the row excluded from this component entirely?
  C. `SUPPORT_*` -- statistical support: how much evidence backs the
     calibration claim for the subgroup this specific row belongs to?
  D. `DOMAIN_*` -- domain status: which specific modeling domain (open-field
     outfield vs. near-wall vs. infield vs. advancement vs. unavailable) does
     this row's component belong to?

`confidence_tier` (`TIER_*`) is a FIFTH, DERIVED field -- a small, documented,
deterministic lookup over (A, C) into one of four ORDINAL labels (high/medium/
low/unavailable), never a numeric percentage (the task explicitly forbids
inferring a probability without a validated probabilistic interpretation --
these tiers carry no such claim, they are a coarse categorical summary only).

## Scope and deliberate simplifications

  - Model-status values for the outfield/infield/advancement components are
    NOT recomputed here -- they are read from each domain's real, already
    -saved `compare_*.py` CLI output (see `mlb_luck_score.scoring.
    run_attribution_ledger._read_existing_gate_status`) and passed in as
    plain strings by the caller (`normalize_model_status` maps the free-form
    strings that function produces onto this module's fixed `MODEL_STATUS_*`
    vocabulary).
  - Outfield near-wall vs. open-field domain status IS decomposed per row,
    reusing `mlb_luck_score.models.outfield_gating.assign_outfield_
    opportunity_gate` unchanged (it is a pure, already-frozen classification
    function, not a model) -- this directly matches this taxonomy's explicit
    `provisional_near_wall` domain value and the real subgroup-ECE finding in
    `outputs/tables/opportunity_model_comparison_detail.json` (near-wall
    subgroups fail; open-field does not).
  - Version 0.7A's OTHER flagged subgroup issue (`opportunity_time_q2`/`q3`,
    from the same file) is NOT decomposed per row -- recomputing those
    quartile boundaries here risks silently drifting from
    `compare_opportunity_models.py`'s own definition. This is a documented
    scoping decision, not an oversight: it is called out in this module's
    aggregate-level report (see `mlb_luck_score.scoring.
    run_season_aggregation`), not attributed to individual rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlb_luck_score.models.outfield_gating import (
    GATE_CONTACT_ONLY_FALLBACK,
    GATE_NEAR_WALL,
    assign_outfield_opportunity_gate,
)
from mlb_luck_score.scoring.attribution_ledger import (
    STATUS_ADVANCEMENT_MODELED,
    STATUS_CONTACT_RESULT_UNAVAILABLE,
    STATUS_CONTACT_RESULT_UNAVAILABLE_OTHER,
)

TAXONOMY_VERSION = "0.11.0"

# ---------------------------------------------------------------------------
# Taxonomy A: model-status confidence
# ---------------------------------------------------------------------------

MODEL_STATUS_CALIBRATED = "calibrated"
MODEL_STATUS_CALIBRATED_LIMITED_SUBGROUP_EVIDENCE = "calibrated_with_limited_subgroup_evidence"
MODEL_STATUS_PROVISIONAL = "provisional"
MODEL_STATUS_NOT_CALIBRATED = "not_calibrated"
MODEL_STATUS_UNAVAILABLE = "unavailable"
MODEL_STATUS_VALUES: tuple[str, ...] = (
    MODEL_STATUS_CALIBRATED,
    MODEL_STATUS_CALIBRATED_LIMITED_SUBGROUP_EVIDENCE,
    MODEL_STATUS_PROVISIONAL,
    MODEL_STATUS_NOT_CALIBRATED,
    MODEL_STATUS_UNAVAILABLE,
)

# ---------------------------------------------------------------------------
# Taxonomy B: row-level data quality
# ---------------------------------------------------------------------------

DATA_QUALITY_COMPLETE_MEASURED = "complete_measured_inputs"
DATA_QUALITY_ESTIMATED = "estimated_inputs"
DATA_QUALITY_FALLBACK = "fallback_inputs"
DATA_QUALITY_MISSING = "missing_inputs"
DATA_QUALITY_EXCLUDED = "excluded_play"
DATA_QUALITY_VALUES: tuple[str, ...] = (
    DATA_QUALITY_COMPLETE_MEASURED,
    DATA_QUALITY_ESTIMATED,
    DATA_QUALITY_FALLBACK,
    DATA_QUALITY_MISSING,
    DATA_QUALITY_EXCLUDED,
)

# ---------------------------------------------------------------------------
# Taxonomy C: statistical support
# ---------------------------------------------------------------------------

SUPPORT_STRONG = "strong"
SUPPORT_MODERATE = "moderate"
SUPPORT_LIMITED = "limited"
SUPPORT_INSUFFICIENT = "insufficient"
SUPPORT_VALUES: tuple[str, ...] = (
    SUPPORT_STRONG,
    SUPPORT_MODERATE,
    SUPPORT_LIMITED,
    SUPPORT_INSUFFICIENT,
)

# ---------------------------------------------------------------------------
# Taxonomy D: domain status
# ---------------------------------------------------------------------------

DOMAIN_CONTACT = "contact"
DOMAIN_OPEN_FIELD_OUTFIELD = "open_field_outfield"
DOMAIN_PROVISIONAL_NEAR_WALL = "provisional_near_wall"
DOMAIN_PROVISIONAL_INFIELD = "provisional_infield"
DOMAIN_PROVISIONAL_ADVANCEMENT = "provisional_advancement"
DOMAIN_UNAVAILABLE_OR_EXCLUDED = "unavailable_or_excluded"
DOMAIN_VALUES: tuple[str, ...] = (
    DOMAIN_CONTACT,
    DOMAIN_OPEN_FIELD_OUTFIELD,
    DOMAIN_PROVISIONAL_NEAR_WALL,
    DOMAIN_PROVISIONAL_INFIELD,
    DOMAIN_PROVISIONAL_ADVANCEMENT,
    DOMAIN_UNAVAILABLE_OR_EXCLUDED,
)

# ---------------------------------------------------------------------------
# Derived: confidence tier (ordinal label, NEVER a probability)
# ---------------------------------------------------------------------------

TIER_HIGH = "high"
TIER_MEDIUM = "medium"
TIER_LOW = "low"
TIER_UNAVAILABLE = "unavailable"
TIER_VALUES: tuple[str, ...] = (TIER_HIGH, TIER_MEDIUM, TIER_LOW, TIER_UNAVAILABLE)

# ---------------------------------------------------------------------------
# Reason codes -- explicit, machine-readable. Several are REUSED verbatim
# from mlb_luck_score.eligibility's own exclusion-reason vocabulary (never
# redefined under a different string) and mlb_luck_score.scoring.
# attribution_ledger's own status constants, so a caller can trace a reason
# code back to the exact upstream column/constant that produced it.
# ---------------------------------------------------------------------------

REASON_NEAR_WALL_PROVISIONAL = "near_wall_provisional"
REASON_SUBGROUP_INSUFFICIENT_EVIDENCE = "subgroup_insufficient_evidence"
REASON_MISSING_GEOMETRY = "missing_geometry"
REASON_CONTACT_ONLY_FALLBACK = "contact_only_fallback"
REASON_INFIELD_LIMITED_SUBGROUP_EVIDENCE = "infield_limited_subgroup_evidence"
REASON_ADVANCEMENT_RARE_CLASS_LIMITED = "advancement_rare_class_limited"
REASON_EXCLUDED_STRATEGIC_PLAY = (
    "excluded_strategic_play"  # matches eligibility.REASON_EXCLUDED_STRATEGIC_PLAY
)
REASON_UNAVAILABLE_MISSING_INPUTS = "unavailable_missing_inputs"
REASON_ESTIMATED_HANG_TIME_INPUT = "estimated_hang_time_input"
REASON_SPRINT_SPEED_MISSING = "sprint_speed_missing"
REASON_POSITION_SPRAY_SECTOR_PROXY = "position_spray_sector_proxy"
REASON_MODEL_STATUS_NOT_SUPPLIED = "model_status_not_supplied"
#: Reused verbatim from `mlb_luck_score.scoring.attribution_ledger` so a
#: reason code always traces back to the exact upstream status string.
REASON_CONTACT_RESULT_UNAVAILABLE_ERROR = STATUS_CONTACT_RESULT_UNAVAILABLE
REASON_CONTACT_RESULT_UNAVAILABLE_OTHER = STATUS_CONTACT_RESULT_UNAVAILABLE_OTHER
REASON_ADVANCEMENT_NOT_MODELED = STATUS_ADVANCEMENT_MODELED.replace(
    "modeled", "not_modeled_for_this_play"
)

# ---------------------------------------------------------------------------
# Component names
# ---------------------------------------------------------------------------

COMPONENT_CONTACT = "contact"
COMPONENT_OUTFIELD_DEFENSE = "outfield_defense"
COMPONENT_INFIELD_DEFENSE = "infield_defense"
COMPONENT_ADVANCEMENT = "advancement"
COMPONENTS: tuple[str, ...] = (
    COMPONENT_CONTACT,
    COMPONENT_OUTFIELD_DEFENSE,
    COMPONENT_INFIELD_DEFENSE,
    COMPONENT_ADVANCEMENT,
)

CONFIDENCE_RECORD_COLUMNS: tuple[str, ...] = (
    "event_id",
    "component_name",
    "eligible",
    "model_version",
    "model_status_confidence",
    "statistical_support",
    "row_data_quality",
    "domain_status",
    "fallback_status",
    "prediction_available",
    "confidence_tier",
    "reason_codes",
)


class ComponentConfidenceError(ValueError):
    """Raised when inputs to the component-confidence layer are invalid."""


def normalize_model_status(raw_status: str | None) -> str:
    """Map a free-form status string (e.g. `mlb_luck_score.scoring.
    run_attribution_ledger._read_existing_gate_status`'s `"False (see ...)"`)
    onto the fixed `MODEL_STATUS_*` vocabulary.

    Never invents a verdict -- an unrecognized or missing string maps to
    `MODEL_STATUS_UNAVAILABLE`, never silently to `MODEL_STATUS_CALIBRATED`.
    """
    if raw_status is None:
        return MODEL_STATUS_UNAVAILABLE
    text = str(raw_status).strip().lower()
    if text.startswith(MODEL_STATUS_CALIBRATED_LIMITED_SUBGROUP_EVIDENCE):
        return MODEL_STATUS_CALIBRATED_LIMITED_SUBGROUP_EVIDENCE
    if text.startswith(MODEL_STATUS_NOT_CALIBRATED):
        return MODEL_STATUS_NOT_CALIBRATED
    if text.startswith(MODEL_STATUS_PROVISIONAL):
        return MODEL_STATUS_PROVISIONAL
    if text.startswith("true") or text.startswith(MODEL_STATUS_CALIBRATED):
        return MODEL_STATUS_CALIBRATED
    if text.startswith("false"):
        return MODEL_STATUS_NOT_CALIBRATED
    return MODEL_STATUS_UNAVAILABLE


def derive_confidence_tier(model_status: str, statistical_support: str) -> str:
    """Deterministic, documented lookup table -- NOT a probability.

    Rule (checked in this order):
      1. `MODEL_STATUS_UNAVAILABLE` -> `TIER_UNAVAILABLE`.
      2. `MODEL_STATUS_NOT_CALIBRATED` -> `TIER_LOW`.
      3. `SUPPORT_INSUFFICIENT` -> `TIER_LOW` (insufficient evidence caps the
         tier regardless of the model's own status -- an unverifiable claim
         is never reported as high confidence).
      4. `MODEL_STATUS_CALIBRATED` and `SUPPORT_STRONG` -> `TIER_HIGH`.
      5. Otherwise (`calibrated_with_limited_subgroup_evidence`/
         `provisional`, or `SUPPORT_MODERATE`/`SUPPORT_LIMITED`) -> `TIER_MEDIUM`.
    """
    if model_status == MODEL_STATUS_UNAVAILABLE:
        return TIER_UNAVAILABLE
    if model_status == MODEL_STATUS_NOT_CALIBRATED:
        return TIER_LOW
    if statistical_support == SUPPORT_INSUFFICIENT:
        return TIER_LOW
    if model_status == MODEL_STATUS_CALIBRATED and statistical_support == SUPPORT_STRONG:
        return TIER_HIGH
    return TIER_MEDIUM


@dataclass(frozen=True)
class _ComponentInputs:
    eligible: pd.Series
    domain_status: pd.Series
    data_quality: pd.Series
    model_status: pd.Series
    support: pd.Series
    prediction_available: pd.Series
    fallback: pd.Series
    reason_codes: pd.Series  # object dtype, each cell a tuple[str, ...]


def _contact_component(df: pd.DataFrame, *, model_status: str | None) -> _ComponentInputs:
    resolved = df["outcome_class"].notna()
    is_error = df["events"] == "field_error"

    domain_status = pd.Series(DOMAIN_CONTACT, index=df.index, dtype=object)
    domain_status.loc[~resolved] = DOMAIN_UNAVAILABLE_OR_EXCLUDED

    data_quality = pd.Series(DATA_QUALITY_COMPLETE_MEASURED, index=df.index, dtype=object)
    data_quality.loc[~resolved] = DATA_QUALITY_EXCLUDED

    normalized_status = normalize_model_status(model_status)
    model_status_col = pd.Series(normalized_status, index=df.index, dtype=object)
    model_status_col.loc[~resolved] = MODEL_STATUS_UNAVAILABLE

    support = pd.Series(SUPPORT_STRONG, index=df.index, dtype=object)
    support.loc[~resolved] = SUPPORT_INSUFFICIENT

    reason_codes = [
        (REASON_CONTACT_RESULT_UNAVAILABLE_ERROR,)
        if (not res and err)
        else (REASON_CONTACT_RESULT_UNAVAILABLE_OTHER,)
        if not res
        else ()
        for res, err in zip(resolved.to_numpy(), is_error.to_numpy(), strict=True)
    ]

    return _ComponentInputs(
        eligible=resolved,
        domain_status=domain_status,
        data_quality=data_quality,
        model_status=model_status_col,
        support=support,
        prediction_available=resolved,
        fallback=pd.Series(False, index=df.index),
        reason_codes=pd.Series(reason_codes, index=df.index, dtype=object),
    )


def _outfield_component(df: pd.DataFrame, *, model_status: str | None) -> _ComponentInputs:
    """NOTE on `prediction_available`: this script's ledger (`mlb_luck_score.
    scoring.run_attribution_ledger`) applies ONE uniform `measured_contact_
    only_v07` model to every `outfield_opportunity_eligible` row -- it does
    NOT use `mlb_luck_score.scoring.gated_outfield_report`'s three-way gate
    (open-field / near-wall specialist / contact-only fallback). So a
    prediction genuinely exists for every eligible row here, REGARDLESS of
    geometry availability -- `prediction_available = eligible`, always.
    `assign_outfield_opportunity_gate` is still reused to determine near
    -wall domain status (the real, per-subgroup ECE failure this taxonomy
    exists to surface) and, informationally only, what the OTHER (gated)
    pipeline would have done for this row (`REASON_CONTACT_ONLY_FALLBACK`) --
    neither of those two facts changes whether THIS ledger's own prediction
    is available.
    """
    eligible = df["outfield_opportunity_eligible"].astype(bool)
    normalized_status = normalize_model_status(model_status)

    if "has_park_geometry" in df.columns:
        gate = assign_outfield_opportunity_gate(df)
        near_wall = eligible & (gate == GATE_NEAR_WALL)
        would_be_contact_only_fallback = eligible & (gate == GATE_CONTACT_ONLY_FALLBACK)
    else:
        near_wall = pd.Series(False, index=df.index)
        would_be_contact_only_fallback = pd.Series(False, index=df.index)
    open_field = eligible & ~near_wall

    domain_status = pd.Series(DOMAIN_UNAVAILABLE_OR_EXCLUDED, index=df.index, dtype=object)
    domain_status.loc[open_field] = DOMAIN_OPEN_FIELD_OUTFIELD
    domain_status.loc[near_wall] = DOMAIN_PROVISIONAL_NEAR_WALL

    hang_time_missing = (
        df["estimated_hang_time_s"].isna()
        if "estimated_hang_time_s" in df.columns
        else pd.Series(True, index=df.index)
    )
    geometry_missing = (
        df["wall_distance_in_spray_direction"].isna()
        if "wall_distance_in_spray_direction" in df.columns
        else pd.Series(True, index=df.index)
    )
    position_is_proxy = (
        (df["assigned_outfield_position_source"] == "spray_sector_proxy")
        if "assigned_outfield_position_source" in df.columns
        else pd.Series(False, index=df.index)
    )

    data_quality = pd.Series(DATA_QUALITY_EXCLUDED, index=df.index, dtype=object)
    data_quality.loc[eligible] = (
        DATA_QUALITY_ESTIMATED  # hang time/landing coords always physics-estimated
    )
    data_quality.loc[eligible & (geometry_missing | position_is_proxy)] = DATA_QUALITY_FALLBACK
    data_quality.loc[eligible & hang_time_missing] = DATA_QUALITY_MISSING

    model_status_col = pd.Series(MODEL_STATUS_UNAVAILABLE, index=df.index, dtype=object)
    model_status_col.loc[open_field] = normalized_status
    # Near-wall rows use the SAME open_field_v07 prediction in this script's
    # ledger (Version 0.7 is frozen; no near-wall specialist is adopted --
    # see this module's docstring), but the real subgroup-ECE evidence shows
    # that subgroup specifically failing, so it is reported as provisional
    # rather than inheriting the aggregate open-field verdict.
    model_status_col.loc[near_wall] = MODEL_STATUS_PROVISIONAL

    support = pd.Series(SUPPORT_INSUFFICIENT, index=df.index, dtype=object)
    support.loc[open_field] = SUPPORT_MODERATE
    support.loc[near_wall] = SUPPORT_INSUFFICIENT

    prediction_available = eligible

    reason_codes = []
    for (
        row_eligible,
        row_near_wall,
        row_would_be_fallback,
        row_geom_missing,
        row_pos_proxy,
        row_exclusion,
    ) in zip(
        eligible.to_numpy(),
        near_wall.to_numpy(),
        would_be_contact_only_fallback.to_numpy(),
        geometry_missing.to_numpy(),
        position_is_proxy.to_numpy(),
        df["outfield_opportunity_exclusion_reason"].to_numpy()
        if "outfield_opportunity_exclusion_reason" in df.columns
        else [None] * len(df),
        strict=True,
    ):
        codes: list[str] = []
        if not row_eligible:
            codes.append(
                REASON_UNAVAILABLE_MISSING_INPUTS if row_exclusion is None else str(row_exclusion)
            )
        else:
            codes.append(REASON_ESTIMATED_HANG_TIME_INPUT)
            if row_would_be_fallback:
                codes.append(REASON_CONTACT_ONLY_FALLBACK)
            if row_geom_missing:
                codes.append(REASON_MISSING_GEOMETRY)
            if row_pos_proxy:
                codes.append(REASON_POSITION_SPRAY_SECTOR_PROXY)
            if row_near_wall:
                codes.append(REASON_NEAR_WALL_PROVISIONAL)
        reason_codes.append(tuple(codes))

    return _ComponentInputs(
        eligible=eligible,
        domain_status=domain_status,
        data_quality=data_quality,
        model_status=model_status_col,
        support=support,
        prediction_available=prediction_available,
        fallback=eligible & (geometry_missing | position_is_proxy),
        reason_codes=pd.Series(reason_codes, index=df.index, dtype=object),
    )


def _infield_component(df: pd.DataFrame, *, model_status: str | None) -> _ComponentInputs:
    eligible = df["infield_opportunity_eligible"].astype(bool)
    normalized_status = normalize_model_status(model_status)

    domain_status = pd.Series(DOMAIN_UNAVAILABLE_OR_EXCLUDED, index=df.index, dtype=object)
    domain_status.loc[eligible] = DOMAIN_PROVISIONAL_INFIELD

    sprint_speed_missing = (
        df["sprint_speed"].isna()
        if "sprint_speed" in df.columns
        else pd.Series(True, index=df.index)
    )
    data_quality = pd.Series(DATA_QUALITY_EXCLUDED, index=df.index, dtype=object)
    data_quality.loc[eligible] = DATA_QUALITY_COMPLETE_MEASURED
    data_quality.loc[eligible & sprint_speed_missing] = DATA_QUALITY_FALLBACK

    model_status_col = pd.Series(MODEL_STATUS_UNAVAILABLE, index=df.index, dtype=object)
    model_status_col.loc[eligible] = normalized_status

    support = pd.Series(SUPPORT_INSUFFICIENT, index=df.index, dtype=object)
    support.loc[eligible] = SUPPORT_LIMITED

    reason_codes = []
    exclusion_col = (
        df["infield_opportunity_exclusion_reason"].to_numpy()
        if "infield_opportunity_exclusion_reason" in df.columns
        else [None] * len(df)
    )
    for row_eligible, row_sprint_missing, row_exclusion in zip(
        eligible.to_numpy(), sprint_speed_missing.to_numpy(), exclusion_col, strict=True
    ):
        codes = []
        if not row_eligible:
            codes.append(
                REASON_UNAVAILABLE_MISSING_INPUTS if row_exclusion is None else str(row_exclusion)
            )
        else:
            codes.append(REASON_INFIELD_LIMITED_SUBGROUP_EVIDENCE)
            if row_sprint_missing:
                codes.append(REASON_SPRINT_SPEED_MISSING)
        reason_codes.append(tuple(codes))

    return _ComponentInputs(
        eligible=eligible,
        domain_status=domain_status,
        data_quality=data_quality,
        model_status=model_status_col,
        support=support,
        prediction_available=eligible,
        fallback=eligible & sprint_speed_missing,
        reason_codes=pd.Series(reason_codes, index=df.index, dtype=object),
    )


def _advancement_component(
    df: pd.DataFrame, ledger: pd.DataFrame, *, model_status: str | None
) -> _ComponentInputs:
    eligible = df["advancement_eligible"].astype(bool)
    modeled = ledger["component_eligibility_status"].str.contains(
        STATUS_ADVANCEMENT_MODELED, na=False
    )
    normalized_status = normalize_model_status(model_status)

    domain_status = pd.Series(DOMAIN_UNAVAILABLE_OR_EXCLUDED, index=df.index, dtype=object)
    domain_status.loc[modeled] = DOMAIN_PROVISIONAL_ADVANCEMENT

    sprint_speed_missing = (
        df["sprint_speed"].isna()
        if "sprint_speed" in df.columns
        else pd.Series(True, index=df.index)
    )
    data_quality = pd.Series(DATA_QUALITY_EXCLUDED, index=df.index, dtype=object)
    data_quality.loc[modeled] = DATA_QUALITY_COMPLETE_MEASURED
    data_quality.loc[modeled & sprint_speed_missing] = DATA_QUALITY_FALLBACK

    model_status_col = pd.Series(MODEL_STATUS_UNAVAILABLE, index=df.index, dtype=object)
    model_status_col.loc[modeled] = normalized_status

    support = pd.Series(SUPPORT_INSUFFICIENT, index=df.index, dtype=object)
    support.loc[modeled] = SUPPORT_LIMITED

    reason_codes = []
    exclusion_col = (
        df["advancement_exclusion_reason"].to_numpy()
        if "advancement_exclusion_reason" in df.columns
        else [None] * len(df)
    )
    for row_eligible, row_modeled, row_exclusion in zip(
        eligible.to_numpy(), modeled.to_numpy(), exclusion_col, strict=True
    ):
        codes = []
        if not row_modeled:
            if row_eligible:
                # advancement_eligible=True but not modeled -- e.g. a
                # field_error row whose outcome_class/Rc is unavailable, so
                # the ledger correctly excludes it regardless of eligibility.
                codes.append(REASON_ADVANCEMENT_NOT_MODELED)
            else:
                codes.append(
                    REASON_UNAVAILABLE_MISSING_INPUTS
                    if row_exclusion is None
                    else str(row_exclusion)
                )
        else:
            codes.append(REASON_ADVANCEMENT_RARE_CLASS_LIMITED)
        reason_codes.append(tuple(codes))

    return _ComponentInputs(
        eligible=eligible,
        domain_status=domain_status,
        data_quality=data_quality,
        model_status=model_status_col,
        support=support,
        prediction_available=modeled,
        fallback=modeled & sprint_speed_missing,
        reason_codes=pd.Series(reason_codes, index=df.index, dtype=object),
    )


def build_play_level_confidence(
    df: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    contact_model_version: str = "baseline_v02",
    outfield_model_version: str = "measured_contact_only_v07",
    infield_model_version: str = "infield_hgb_v08",
    advancement_model_version: str = "advancement_speed_v09",
    contact_model_status: str | None = MODEL_STATUS_CALIBRATED,
    outfield_model_status: str | None = None,
    infield_model_status: str | None = None,
    advancement_model_status: str | None = None,
) -> pd.DataFrame:
    """Build the versioned, LONG-format (one row per `event_id` x `component_
    name`) play-level confidence record described in this module's docstring.

    Args:
        df: The same prepared DataFrame passed to `mlb_luck_score.scoring.
            attribution_ledger.build_attribution_ledger` (eligibility +
            opportunity/advancement features already attached), with
            `event_id` present and unique (same precondition as that
            function).
        ledger: That function's output, same index/order as `df`.
        *_model_version: Purely descriptive labels recorded verbatim
            (`mlb_luck_score.models.*`'s own `feature_set_label`/`variant`
            naming) -- never affects any computed value here.
        *_model_status: The real, already-computed status string for each
            domain (typically `mlb_luck_score.scoring.
            run_attribution_ledger._read_existing_gate_status`'s output, or
            `None` if not supplied -- mapped to `MODEL_STATUS_UNAVAILABLE`
            via `normalize_model_status`, never guessed).

    Returns:
        A DataFrame with columns `CONFIDENCE_RECORD_COLUMNS`.
    """
    if not df.index.equals(ledger.index):
        raise ComponentConfidenceError("df and ledger must share the identical index/row order")
    if "event_id" not in df.columns:
        raise ComponentConfidenceError("df must have an event_id column")

    builders: tuple[tuple[str, _ComponentInputs, str], ...] = (
        (
            COMPONENT_CONTACT,
            _contact_component(df, model_status=contact_model_status),
            contact_model_version,
        ),
        (
            COMPONENT_OUTFIELD_DEFENSE,
            _outfield_component(df, model_status=outfield_model_status),
            outfield_model_version,
        ),
        (
            COMPONENT_INFIELD_DEFENSE,
            _infield_component(df, model_status=infield_model_status),
            infield_model_version,
        ),
        (
            COMPONENT_ADVANCEMENT,
            _advancement_component(df, ledger, model_status=advancement_model_status),
            advancement_model_version,
        ),
    )

    frames = []
    for component_name, inputs, model_version in builders:
        tiers = [
            derive_confidence_tier(status, support)
            for status, support in zip(
                inputs.model_status.to_numpy(), inputs.support.to_numpy(), strict=True
            )
        ]
        frames.append(
            pd.DataFrame(
                {
                    "event_id": df["event_id"].to_numpy(),
                    "component_name": component_name,
                    "eligible": inputs.eligible.to_numpy(),
                    "model_version": model_version,
                    "model_status_confidence": inputs.model_status.to_numpy(),
                    "statistical_support": inputs.support.to_numpy(),
                    "row_data_quality": inputs.data_quality.to_numpy(),
                    "domain_status": inputs.domain_status.to_numpy(),
                    "fallback_status": inputs.fallback.to_numpy(),
                    "prediction_available": inputs.prediction_available.to_numpy(),
                    "confidence_tier": tiers,
                    "reason_codes": inputs.reason_codes.to_numpy(),
                },
                index=df.index,
            )
        )

    result = pd.concat(frames, ignore_index=True)
    return result[list(CONFIDENCE_RECORD_COLUMNS)]
