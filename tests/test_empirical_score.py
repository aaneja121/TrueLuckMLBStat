from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlb_luck_score.scoring.empirical_score import (
    EmpiricalScoreValidationError,
    ReferenceScoreArtifact,
    compute_empirical_public_score,
    compute_empirical_public_scores,
    load_reference_artifact,
    save_reference_artifact,
)


@pytest.fixture
def artifact() -> ReferenceScoreArtifact:
    rng = np.random.default_rng(0)
    percentiles = list(np.linspace(0, 100, 101))
    positive_values = sorted(rng.exponential(scale=0.3, size=101))
    negative_values = sorted(rng.exponential(scale=0.2, size=101))
    return ReferenceScoreArtifact(
        scoring_version="0.2.0",
        model_variant="unweighted_probability_baseline",
        training_seasons=(2021, 2022, 2023),
        reference_seasons=(2024,),
        class_order=("out", "single", "double", "triple", "home_run"),
        run_value_table={
            "out": -0.254916,
            "single": 0.463266,
            "double": 0.763026,
            "triple": 1.033068,
            "home_run": 1.400288,
        },
        sample_counts={"total": 202, "positive": 101, "negative": 101, "zero": 0},
        quantile_percentiles=percentiles,
        positive_quantile_values=positive_values,
        negative_quantile_values=negative_values,
        created_at_utc="2026-01-01T00:00:00+00:00",
        package_version="0.1.0",
        scikit_learn_version="1.9.0",
        source_metadata={"source": "test fixture"},
    )


def test_zero_raw_luck_maps_exactly_to_zero(artifact: ReferenceScoreArtifact):
    assert compute_empirical_public_score(0.0, artifact) == 0.0


def test_sign_preservation(artifact: ReferenceScoreArtifact):
    assert compute_empirical_public_score(0.05, artifact) > 0
    assert compute_empirical_public_score(-0.05, artifact) < 0


def test_signed_percentile_is_monotonic_in_raw_luck(artifact: ReferenceScoreArtifact):
    xs = np.linspace(-2.0, 2.0, 41)
    scores = [compute_empirical_public_score(x, artifact) for x in xs]
    assert scores == sorted(scores)


def test_clips_beyond_historical_range_to_plus_100(artifact: ReferenceScoreArtifact):
    huge_positive = max(artifact.positive_quantile_values) * 100
    assert compute_empirical_public_score(huge_positive, artifact) == pytest.approx(100.0)


def test_clips_beyond_historical_range_to_minus_100(artifact: ReferenceScoreArtifact):
    huge_negative = -max(artifact.negative_quantile_values) * 100
    assert compute_empirical_public_score(huge_negative, artifact) == pytest.approx(-100.0)


def test_score_stays_within_valid_range(artifact: ReferenceScoreArtifact):
    xs = np.linspace(-10.0, 10.0, 101)
    for x in xs:
        score = compute_empirical_public_score(float(x), artifact)
        assert -100.0 <= score <= 100.0


def test_rejects_non_finite_input(artifact: ReferenceScoreArtifact):
    with pytest.raises(EmpiricalScoreValidationError, match="finite"):
        compute_empirical_public_score(float("nan"), artifact)
    with pytest.raises(EmpiricalScoreValidationError, match="finite"):
        compute_empirical_public_score(float("inf"), artifact)


def test_rejects_scoring_when_quantile_side_is_empty(artifact: ReferenceScoreArtifact):
    from dataclasses import replace

    no_positive = replace(artifact, positive_quantile_values=[])
    with pytest.raises(EmpiricalScoreValidationError, match="no quantile values"):
        compute_empirical_public_score(0.5, no_positive)


def test_compute_empirical_public_scores_vectorized_matches_scalar(
    artifact: ReferenceScoreArtifact,
):
    values = [-0.5, -0.1, 0.0, 0.1, 0.5]
    vectorized = compute_empirical_public_scores(values, artifact)
    scalar = [compute_empirical_public_score(v, artifact) for v in values]
    assert list(vectorized) == pytest.approx(scalar)


def test_reference_artifact_metadata_fields(artifact: ReferenceScoreArtifact):
    assert artifact.scoring_version == "0.2.0"
    assert artifact.model_variant == "unweighted_probability_baseline"
    assert artifact.training_seasons == (2021, 2022, 2023)
    assert artifact.reference_seasons == (2024,)
    assert artifact.class_order == ("out", "single", "double", "triple", "home_run")
    assert artifact.sample_counts["total"] == 202
    assert "source" in artifact.source_metadata


def test_save_and_load_reference_artifact_round_trip(
    artifact: ReferenceScoreArtifact, tmp_path: Path
):
    path = tmp_path / "reference_score_v0.2.0.json"
    save_reference_artifact(artifact, path)
    assert path.exists()
    loaded = load_reference_artifact(path)
    assert loaded == artifact
    assert isinstance(loaded.training_seasons, tuple)
    assert isinstance(loaded.reference_seasons, tuple)
    assert isinstance(loaded.class_order, tuple)
