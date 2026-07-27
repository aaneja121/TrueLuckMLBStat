"""Version 0.2 empirical public score: signed percentile mapping.

Replaces the Version 0.1 tanh-based mapping
(`mlb_luck_score.scoring.public_score.raw_luck_to_public_score`, now
LEGACY -- see that module's docstring) as the default public-facing
-100..+100 score. Instead of an arbitrary functional form, this maps a
play's raw contact luck (in runs, see `mlb_luck_score.scoring.
contact_luck`) to its SIGNED PERCENTILE within a genuinely out-of-sample
reference distribution: the unweighted probability baseline's predictions
on 2024 validation data (trained only on 2021-2023; 2024 and 2025 are never
touched by that training).

    raw luck == 0             -> score == 0 (exactly)
    raw luck >  0 (favorable) -> score in (0, 100], this play's percentile
                                 among POSITIVE reference plays
    raw luck <  0 (unfavorable) -> score in [-100, 0), the percentile of
                                 this play's luck MAGNITUDE among the
                                 magnitudes of NEGATIVE reference plays,
                                 negated

+90 means more favorable raw luck than ~90% of positive-luck reference
plays. -90 means a larger unfavorable-luck magnitude than ~90% of
negative-luck reference plays. Values beyond the historical reference range
are clipped to +-100 -- this is a real limitation: the mapping is only as
good as the reference sample it was built from, and it is NOT refit on
2025 (see `mlb_luck_score.models.build_reference_score`).

This module only defines the reference-artifact data structure and the pure
scoring function -- building a new reference artifact from real data
requires training a model on real data and lives in
`mlb_luck_score.models.build_reference_score` (this module has no sklearn
or pandas dependency).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_SCORING_VERSION = "0.2.0"


class EmpiricalScoreValidationError(ValueError):
    """Raised when inputs to the empirical percentile score are invalid."""


@dataclass(frozen=True)
class ReferenceScoreArtifact:
    """A fitted Version 0.2 empirical public-score reference.

    Attributes:
        scoring_version: Version tag for this scoring methodology (e.g.
            "0.2.0").
        model_variant: Which trained-model variant produced the reference
            predictions. Must be the unweighted probability baseline (see
            `mlb_luck_score.models.train_contact_model.VARIANT_UNWEIGHTED`)
            -- never the class-balanced comparison model.
        training_seasons: Seasons the base model was trained on (2021-2023).
        reference_seasons: Seasons whose predictions built this reference
            distribution (2024 only -- never 2025).
        class_order: Class ordering used to build predictions.
        run_value_table: The fixed run-value map used (see
            `mlb_luck_score.scoring.run_values.DEFAULT_RUN_VALUE_MAP`).
        sample_counts: `{"total", "positive", "negative", "zero"}` counts of
            reference raw-luck values.
        quantile_percentiles: The percentile grid (0-100) the two quantile
            tables below are evaluated at.
        positive_quantile_values: Raw-luck value at each percentile among
            POSITIVE reference raw-luck values (ascending).
        negative_quantile_values: Raw-luck MAGNITUDE (absolute value) at
            each percentile among NEGATIVE reference raw-luck values
            (ascending).
        created_at_utc: ISO timestamp when this artifact was built.
        package_version: `mlb_luck_score.__version__` at build time.
        scikit_learn_version: scikit-learn version at build time.
        source_metadata: Free-form provenance notes (data source, caveats).
    """

    scoring_version: str
    model_variant: str
    training_seasons: tuple[int, ...]
    reference_seasons: tuple[int, ...]
    class_order: tuple[str, ...]
    run_value_table: dict[str, float]
    sample_counts: dict[str, int]
    quantile_percentiles: list[float]
    positive_quantile_values: list[float]
    negative_quantile_values: list[float]
    created_at_utc: str
    package_version: str
    scikit_learn_version: str
    source_metadata: dict[str, str] = field(default_factory=dict)

    def score(self, raw_luck_runs: float) -> float:
        """Map a raw-luck value (runs) to a signed -100..+100 empirical score."""
        return compute_empirical_public_score(raw_luck_runs, self)


def _interp_percentile(
    magnitude: float, percentiles: list[float], quantile_values: list[float]
) -> float:
    if not quantile_values:
        raise EmpiricalScoreValidationError(
            "Reference artifact has no quantile values for this sign -- cannot score. "
            "Rebuild the reference artifact with a larger reference sample."
        )
    return float(np.interp(magnitude, quantile_values, percentiles))


def compute_empirical_public_score(raw_luck_runs: float, artifact: ReferenceScoreArtifact) -> float:
    """Map `raw_luck_runs` to a signed percentile score using `artifact`.

    See `ReferenceScoreArtifact` and module docstring for exact semantics.
    Monotonic and sign-preserving by construction: `numpy.interp` against a
    non-decreasing quantile table is a monotonic non-decreasing function of
    its input, exactly 0 at 0 (handled explicitly, not via interpolation),
    and clipped to +-100 for values beyond the reference range (`numpy.
    interp` holds the boundary value flat outside its input range).

    Args:
        raw_luck_runs: A single play's Version 0.2 raw contact luck, in
            runs (see `mlb_luck_score.scoring.contact_luck.
            compute_raw_contact_luck_runs`).
        artifact: A `ReferenceScoreArtifact` built from out-of-sample data.

    Returns:
        A float in [-100, 100].

    Raises:
        EmpiricalScoreValidationError: If `raw_luck_runs` is not finite, or
            if the artifact lacks quantile values for the needed sign.
    """
    if not math.isfinite(raw_luck_runs):
        raise EmpiricalScoreValidationError(f"raw_luck_runs must be finite, got {raw_luck_runs!r}")
    if raw_luck_runs == 0.0:
        return 0.0
    if raw_luck_runs > 0:
        return _interp_percentile(
            raw_luck_runs, artifact.quantile_percentiles, artifact.positive_quantile_values
        )
    magnitude_percentile = _interp_percentile(
        -raw_luck_runs, artifact.quantile_percentiles, artifact.negative_quantile_values
    )
    return -magnitude_percentile


def compute_empirical_public_scores(
    raw_luck_values: Iterable[float], artifact: ReferenceScoreArtifact
) -> np.ndarray:
    """Vectorized convenience wrapper around `compute_empirical_public_score`.

    Useful for diagnostics (e.g. plotting the distribution of scores across
    many plays) -- NOT for constructing a season-level metric. Never sum or
    average the result; see `mlb_luck_score.scoring.aggregation`.
    """
    return np.array([compute_empirical_public_score(v, artifact) for v in raw_luck_values])


def save_reference_artifact(artifact: ReferenceScoreArtifact, path: Path) -> None:
    """Serialize a `ReferenceScoreArtifact` to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "scoring_version": artifact.scoring_version,
        "model_variant": artifact.model_variant,
        "training_seasons": list(artifact.training_seasons),
        "reference_seasons": list(artifact.reference_seasons),
        "class_order": list(artifact.class_order),
        "run_value_table": artifact.run_value_table,
        "sample_counts": artifact.sample_counts,
        "quantile_percentiles": artifact.quantile_percentiles,
        "positive_quantile_values": artifact.positive_quantile_values,
        "negative_quantile_values": artifact.negative_quantile_values,
        "created_at_utc": artifact.created_at_utc,
        "package_version": artifact.package_version,
        "scikit_learn_version": artifact.scikit_learn_version,
        "source_metadata": artifact.source_metadata,
    }
    path.write_text(json.dumps(data, indent=2))


def load_reference_artifact(path: Path) -> ReferenceScoreArtifact:
    """Load a `ReferenceScoreArtifact` previously saved with `save_reference_artifact`."""
    data = json.loads(path.read_text())
    data["training_seasons"] = tuple(data["training_seasons"])
    data["reference_seasons"] = tuple(data["reference_seasons"])
    data["class_order"] = tuple(data["class_order"])
    return ReferenceScoreArtifact(**data)
