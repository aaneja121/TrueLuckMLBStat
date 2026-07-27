from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb_luck_score.config import ProtectedSeasonError
from mlb_luck_score.models.build_reference_score import build_reference_artifact
from mlb_luck_score.models.train_contact_model import VARIANT_UNWEIGHTED


def _synthetic_season_df(
    season: int, rng: np.random.Generator, n_per_class: int = 40
) -> pd.DataFrame:
    profiles = {
        "out": (75.0, 20.0, 100.0),
        "single": (92.0, 8.0, 180.0),
        "double": (98.0, 18.0, 300.0),
        "triple": (100.0, 15.0, 340.0),
        "home_run": (105.0, 28.0, 410.0),
    }
    rows = []
    for outcome, (speed, angle, dist) in profiles.items():
        for i in range(n_per_class):
            rows.append(
                {
                    "launch_speed": speed + rng.normal(0, 1.5),
                    "launch_angle": angle + rng.normal(0, 1.5),
                    "spray_angle_approx": rng.normal(0, 10),
                    "hit_distance_sc": dist + rng.normal(0, 5),
                    "bb_type": "line_drive",
                    "stand": "R" if i % 2 == 0 else "L",
                    "venue": "Synthetic Park",
                    "outcome_class": outcome,
                    "season": season,
                    "eligible_for_training": True,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def multi_season_df() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    return pd.concat(
        [_synthetic_season_df(s, rng) for s in (2021, 2022, 2023, 2024)], ignore_index=True
    )


def test_build_reference_artifact_trains_only_on_2021_2023(
    multi_season_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
):
    import mlb_luck_score.models.build_reference_score as build_module

    seen_train_seasons: set[int] = set()
    original_train_model = build_module.train_model

    def _tracking_train_model(train_df, **kwargs):
        seen_train_seasons.update(train_df["season"].unique().tolist())
        return original_train_model(train_df, **kwargs)

    monkeypatch.setattr(build_module, "train_model", _tracking_train_model)

    build_reference_artifact(multi_season_df)

    assert seen_train_seasons == {2021, 2022, 2023}


def test_build_reference_artifact_uses_only_2024_reference_rows(multi_season_df: pd.DataFrame):
    artifact = build_reference_artifact(multi_season_df)
    assert artifact.reference_seasons == (2024,)
    assert artifact.training_seasons == (2021, 2022, 2023)
    assert artifact.sample_counts["total"] == 200  # 5 classes x 40 rows in season 2024


def test_build_reference_artifact_uses_unweighted_variant(multi_season_df: pd.DataFrame):
    artifact = build_reference_artifact(multi_season_df)
    assert artifact.model_variant == VARIANT_UNWEIGHTED or "unweighted" in artifact.model_variant
    assert "balanced" not in artifact.model_variant


def test_build_reference_artifact_rejects_2025_in_train_seasons(multi_season_df: pd.DataFrame):
    with pytest.raises(ProtectedSeasonError):
        build_reference_artifact(multi_season_df, train_seasons=(2021, 2022, 2025))


def test_build_reference_artifact_rejects_2025_in_reference_seasons(multi_season_df: pd.DataFrame):
    with pytest.raises(ProtectedSeasonError):
        build_reference_artifact(multi_season_df, reference_seasons=(2025,))


def test_build_reference_artifact_raises_on_missing_training_rows(multi_season_df: pd.DataFrame):
    only_2024 = multi_season_df[multi_season_df["season"] == 2024]
    with pytest.raises(ValueError, match="training"):
        build_reference_artifact(only_2024)


def test_build_reference_artifact_metadata_has_source_and_run_values(multi_season_df: pd.DataFrame):
    artifact = build_reference_artifact(multi_season_df)
    assert "source" in artifact.source_metadata
    assert set(artifact.run_value_table.keys()) == {
        "out",
        "single",
        "double",
        "triple",
        "home_run",
    }
    assert (
        artifact.sample_counts["positive"]
        + artifact.sample_counts["negative"]
        + artifact.sample_counts["zero"]
        == artifact.sample_counts["total"]
    )
