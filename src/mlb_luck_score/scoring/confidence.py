"""Version 0.1 confidence report: observable data completeness only.

This module reports how complete the *observable input data* was for a
given play -- it is NOT a statistical uncertainty estimate, a confidence
interval, or a measure of model quality. A play can have 100% data
completeness and still have a highly uncertain (poorly calibrated)
probability prediction; those are different questions.

Do not use this report to pull a raw-luck or public-score value toward
zero -- low data completeness means "we know less about how complete the
inputs were," not "the luck estimate should be dampened." Mixing the two
would silently redefine what the luck score means. If a confidence-weighted
score is ever wanted, it must be a new, explicitly documented, separate
quantity.

Future confidence components NOT implemented here (see README/CLAUDE.md):
    - Historical support (how many similar plays inform this prediction)
    - Model disagreement (e.g. across an ensemble)
    - Sensitivity analysis (how much the output moves under small input
      perturbations)
    - Calibration uncertainty (confidence in the calibration curve itself)
    - Bootstrap or posterior predictive intervals
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

#: Fields whose presence drives the headline "data completeness" percentage.
CORE_CONTACT_FIELDS: tuple[str, ...] = (
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "hc_x",
    "hc_y",
    "bb_type",
    "stand",
)

#: Additional fields that are nice-to-have but not part of the headline
#: percentage; individually reported as missing/present.
OPTIONAL_CONTEXT_FIELDS: tuple[str, ...] = (
    "venue",
    "if_fielding_alignment",
    "of_fielding_alignment",
    "sprint_speed",
)

HIGH_COMPLETENESS_THRESHOLD = 90.0
MEDIUM_COMPLETENESS_THRESHOLD = 60.0

HIGH_LABEL = "high_data_completeness"
MEDIUM_LABEL = "medium_data_completeness"
LOW_LABEL = "low_data_completeness"


def _is_present(value: object) -> bool:
    if value is None or value is pd.NA:
        return False
    return not (isinstance(value, float) and math.isnan(value))


@dataclass(frozen=True)
class ConfidenceReport:
    """Version 0.1 data-completeness report for a single play.

    Attributes:
        required_fields_present_pct: Percentage of `CORE_CONTACT_FIELDS`
            present (non-null) for this play.
        missing_optional_fields: Which `OPTIONAL_CONTEXT_FIELDS` are absent.
        spray_direction_available: Whether hc_x/hc_y were present.
        park_available: Whether venue was present.
        alignment_fields_available: Whether both infield and outfield
            alignment classifications were present.
        fully_eligible: Whether the play was fully eligible for training
            (per `mlb_luck_score.eligibility`), if that information was
            supplied.
        label: Provisional descriptive label -- see module-level thresholds.
            This is descriptive only, not a statistical confidence interval.
    """

    required_fields_present_pct: float
    missing_optional_fields: tuple[str, ...]
    spray_direction_available: bool
    park_available: bool
    alignment_fields_available: bool
    fully_eligible: bool | None
    label: str


def compute_confidence(row: Mapping[str, object] | pd.Series) -> ConfidenceReport:
    """Compute the Version 0.1 data-completeness report for one play.

    Args:
        row: A mapping (or pandas Series) with Statcast-derived fields.
            Missing fields are treated as absent rather than raising.

    Returns:
        A `ConfidenceReport`.
    """
    present_core = [f for f in CORE_CONTACT_FIELDS if _is_present(row.get(f))]
    pct_present = 100.0 * len(present_core) / len(CORE_CONTACT_FIELDS)

    missing_optional = tuple(f for f in OPTIONAL_CONTEXT_FIELDS if not _is_present(row.get(f)))

    spray_available = _is_present(row.get("hc_x")) and _is_present(row.get("hc_y"))
    park_available = _is_present(row.get("venue"))
    alignment_available = _is_present(row.get("if_fielding_alignment")) and _is_present(
        row.get("of_fielding_alignment")
    )

    fully_eligible_raw = row.get("eligible_for_training")
    fully_eligible = bool(fully_eligible_raw) if _is_present(fully_eligible_raw) else None

    if pct_present >= HIGH_COMPLETENESS_THRESHOLD:
        label = HIGH_LABEL
    elif pct_present >= MEDIUM_COMPLETENESS_THRESHOLD:
        label = MEDIUM_LABEL
    else:
        label = LOW_LABEL

    return ConfidenceReport(
        required_fields_present_pct=pct_present,
        missing_optional_fields=missing_optional,
        spray_direction_available=spray_available,
        park_available=park_available,
        alignment_fields_available=alignment_available,
        fully_eligible=fully_eligible,
        label=label,
    )


def compute_confidence_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized convenience wrapper: one `ConfidenceReport` per row."""
    reports = [compute_confidence(row) for _, row in df.iterrows()]
    return pd.DataFrame(r.__dict__ for r in reports)
