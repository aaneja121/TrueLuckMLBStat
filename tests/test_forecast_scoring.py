"""Contact Forecast R1: walk-forward scoring guards.

The load-bearing test here is `test_mutating_the_scored_season_cannot_move_
the_fitted_parameters`: it rewrites every contact and outcome column in the
season being scored and proves the fitted coefficients do not move by a
single bit. That is fit-time causality demonstrated rather than asserted.

Synthetic data only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from forecast import score_development_seasons as sds
from forecast.forecast_config import ForecastSeasonError

_VENUES = ("Alpha Park", "Beta Field", "Gamma Stadium")
_BB_TYPES = ("ground_ball", "line_drive", "fly_ball", "popup")


def make_development_frame(
    *,
    seasons: tuple[int, ...] = (2021, 2022, 2023, 2024),
    rows_per_season: int = 600,
    seed: int = 11,
) -> pd.DataFrame:
    """A synthetic multi-season development frame shaped like the real one.

    Outcomes are drawn with a launch-speed dependence so the fitted contact
    model is non-degenerate and its coefficients are sensitive to the
    training rows -- which is what makes the mutation test meaningful rather
    than vacuously true.
    """
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for season in seasons:
        n = rows_per_season
        launch_speed = rng.normal(88.0, 12.0, n)
        launch_angle = rng.normal(12.0, 20.0, n)
        # Outcomes are drawn from launch-speed/angle-dependent PROBABILITIES,
        # not from hard thresholds: a deterministic rule makes the classes
        # perfectly separable, and the frozen model's `max_iter=2000`
        # logistic regression then runs every iteration without converging,
        # which made this module's tests take ~100s instead of ~10s.
        quality = (launch_speed - 88.0) / 12.0 + (20.0 - np.abs(launch_angle - 15.0)) / 20.0
        p_hit = 1.0 / (1.0 + np.exp(-quality))
        draw = rng.random(n)
        # dtype=object, NOT a numpy fixed-width string array: assigning
        # "field_error" (11 chars) into a "<U9" array truncates it silently.
        events = np.empty(n, dtype=object)
        events[:] = "field_out"
        events[draw < p_hit * 0.60] = "single"
        events[draw < p_hit * 0.25] = "double"
        events[draw < p_hit * 0.08] = "home_run"
        events[draw < p_hit * 0.02] = "triple"
        # A realistic slice of unresolved batted balls.
        unresolved_idx = rng.choice(n, size=max(1, n // 50), replace=False)
        events[unresolved_idx] = "field_error"

        game_offset = np.arange(n) // 4
        frames.append(
            pd.DataFrame(
                {
                    "event_id": [f"{season}-{i}" for i in range(n)],
                    "game_pk": 100000 * (season - 2020) + game_offset,
                    "at_bat_number": (np.arange(n) % 4) + 1,
                    "pitch_number": 1,
                    "game_date": (
                        pd.Timestamp(f"{season}-04-01") + pd.to_timedelta(game_offset, unit="D")
                    ).strftime("%Y-%m-%d"),
                    "season": season,
                    "batter": rng.integers(1, 40, n),
                    "stand": rng.choice(["L", "R"], n),
                    "launch_speed": launch_speed,
                    "launch_angle": launch_angle,
                    "spray_angle_approx": rng.normal(0.0, 25.0, n),
                    "hit_distance_sc": np.clip(launch_speed * 3.5 + launch_angle * 2.0, 0, 480),
                    "bb_type": rng.choice(_BB_TYPES, n),
                    "venue": rng.choice(_VENUES, n),
                    "events": events,
                    "description": "hit_into_play",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def development_df() -> pd.DataFrame:
    from mlb_luck_score.eligibility import compute_eligibility

    return compute_eligibility(make_development_frame())


def score(development_df: pd.DataFrame, season: int) -> sds.SeasonScoringResult:
    return sds.score_season(
        season,
        development_df=development_df,
        input_path=sds.DEFAULT_DEVELOPMENT_INPUT,
        input_sha256="synthetic",
    )


def test_mutating_the_scored_season_cannot_move_the_fitted_parameters(
    development_df: pd.DataFrame,
) -> None:
    """Fit-time causality, demonstrated: rewrite 2024, fitted params do not move.

    Every contact input and every outcome in the scored season is replaced
    with different values. If a single scored-season row reached the
    estimator, the logistic-regression coefficients would shift.
    """
    baseline = score(development_df, 2024)
    baseline_model = sds.train_model(
        development_df[
            development_df["eligible_for_training"].fillna(False).astype(bool)
            & development_df["season"].isin(sds.walk_forward_train_seasons(2024))
        ],
        class_weight=sds.FROZEN_CONTACT_MODEL_CLASS_WEIGHT,
    )

    tampered = development_df.copy()
    scored_rows = tampered["season"] == 2024
    tampered.loc[scored_rows, "launch_speed"] = 55.0
    tampered.loc[scored_rows, "launch_angle"] = -30.0
    tampered.loc[scored_rows, "hit_distance_sc"] = 1.0
    tampered.loc[scored_rows, "spray_angle_approx"] = 44.0
    tampered.loc[scored_rows, "bb_type"] = "popup"
    tampered.loc[scored_rows, "venue"] = "Alpha Park"
    tampered.loc[scored_rows, "stand"] = "L"
    tampered.loc[scored_rows, "outcome_class"] = "home_run"
    tampered.loc[scored_rows, "events"] = "home_run"

    tampered_model = sds.train_model(
        tampered[
            tampered["eligible_for_training"].fillna(False).astype(bool)
            & tampered["season"].isin(sds.walk_forward_train_seasons(2024))
        ],
        class_weight=sds.FROZEN_CONTACT_MODEL_CLASS_WEIGHT,
    )

    np.testing.assert_array_equal(
        baseline_model.pipeline.named_steps["classify"].coef_,
        tampered_model.pipeline.named_steps["classify"].coef_,
    )
    np.testing.assert_array_equal(
        baseline_model.pipeline.named_steps["classify"].intercept_,
        tampered_model.pipeline.named_steps["classify"].intercept_,
    )
    assert baseline.manifest["feature_schema"] == sds.feature_schema(tampered_model)


def test_mutating_the_scored_season_does_change_its_scores(
    development_df: pd.DataFrame,
) -> None:
    """The companion check: the mutation test above is not vacuously true.

    If rewriting the scored season changed nothing at all, the parameter
    test would prove nothing. Scores must move even though parameters do not.
    """
    baseline = score(development_df, 2024)

    tampered = development_df.copy()
    scored_rows = tampered["season"] == 2024
    tampered.loc[scored_rows, "launch_speed"] = 110.0
    tampered.loc[scored_rows, "launch_angle"] = 28.0
    tampered_result = score(tampered, 2024)

    assert not np.allclose(
        baseline.ledger["baseline_expected_contact_run_value"].dropna().to_numpy(),
        tampered_result.ledger["baseline_expected_contact_run_value"].dropna().to_numpy(),
    )


def test_training_frame_never_contains_the_scored_season(
    development_df: pd.DataFrame,
) -> None:
    for season in (2022, 2023, 2024):
        result = score(development_df, season)
        confirmation = result.manifest["causality"]["fit_time"]
        assert confirmation["confirmed"] is True
        assert confirmation["scored_season_rows_in_training_frame"] == 0
        assert max(confirmation["seasons_present_in_training_frame"]) < season


def test_causality_guard_catches_a_contaminated_training_frame() -> None:
    train_df = pd.DataFrame({"season": [2021, 2022, 2023]})
    with pytest.raises(sds.ForecastLeakageError, match="scored season 2023"):
        sds.assert_no_scored_season_rows_in_training(train_df, 2023)


def test_causality_guard_catches_a_later_season_in_training() -> None:
    train_df = pd.DataFrame({"season": [2021, 2024]})
    with pytest.raises(sds.ForecastLeakageError, match="later season"):
        sds.assert_no_scored_season_rows_in_training(train_df, 2022)


def test_scoring_uses_the_frozen_unweighted_contact_model(
    development_df: pd.DataFrame,
) -> None:
    """`class_weight='balanced'` is documented to wreck probability calibration.

    Contact Luck is built from those probabilities, so this is not a style
    preference -- a reweighted model would corrupt every deserved value.
    """
    result = score(development_df, 2024)
    assert result.manifest["frozen_model"]["class_weight"] is None
    assert result.manifest["frozen_model"]["variant"] == sds.FROZEN_CONTACT_MODEL_VARIANT


def test_contact_luck_identity_holds_on_every_scored_row(
    development_df: pd.DataFrame,
) -> None:
    ledger = score(development_df, 2024).ledger
    scored = ledger[ledger["is_scored"].astype(bool)]
    residual = (
        scored["observed_contact_result_run_value"]
        - scored["baseline_expected_contact_run_value"]
        - scored["contact_result_surprise"]
    )
    assert float(np.max(np.abs(residual.to_numpy()))) < 1e-9


def test_unresolved_rows_are_nulled_exactly_as_production_does(
    development_df: pd.DataFrame,
) -> None:
    """Production nulls every result-linked column, probabilities included."""
    ledger = score(development_df, 2024).ledger
    unresolved = ledger[~ledger["is_scored"].astype(bool)]

    assert len(unresolved) > 0
    for column in (
        "observed_contact_result_run_value",
        "baseline_expected_contact_run_value",
        "contact_result_surprise",
    ):
        assert unresolved[column].isna().all()
    for column in ("p_out", "p_single", "p_double", "p_triple", "p_home_run"):
        assert unresolved[column].isna().all()
    assert (
        ledger[ledger["is_scored"].astype(bool)]["baseline_expected_contact_run_value"]
        .notna()
        .all()
    )


def test_ledger_matches_the_production_play_ledger_shape(
    development_df: pd.DataFrame,
) -> None:
    ledger = score(development_df, 2024).ledger
    assert list(ledger.columns) == list(sds.LEDGER_COLUMNS)


def test_manifest_carries_every_required_field(development_df: pd.DataFrame) -> None:
    manifest = score(development_df, 2023).manifest

    assert manifest["scored_season"] == 2023
    assert manifest["training_seasons"] == [2021, 2022]
    assert manifest["row_counts"]["training_rows"] > 0
    assert manifest["row_counts"]["eligible_batted_balls_in_scored_season"] > 0
    assert manifest["row_counts"]["unresolved_excluded_from_per_100_denominator"] > 0
    assert manifest["frozen_model"]["estimator"].endswith("train_model")
    assert manifest["frozen_model"]["run_value_map_sha256"]
    assert manifest["feature_schema"]["numeric_features"]
    assert manifest["feature_schema_sha256"]
    assert manifest["artifact_hashes"]["ledger_sha256"]
    assert manifest["artifact_hashes"]["input_dataset_sha256"] == "synthetic"
    assert manifest["expected_rv_coverage"]["coverage_fraction"] > 0
    assert manifest["causality"]["fit_time"]["confirmed"] is True
    assert manifest["causality"]["no_selection_performed_on_scored_season"] is True
    assert manifest["causality"]["selection_steps_performed_by_this_module"] == []


def test_manifest_records_the_design_time_caveat(development_df: pd.DataFrame) -> None:
    """The report must never call a 2022 fold a model that existed in 2022."""
    caveat = score(development_df, 2022).manifest["causality"]["design_time_caveat"]
    assert "WALK-FORWARD REFIT OF THE FROZEN CURRENT SPECIFICATION" in caveat
    assert "NOT a model that historically existed" in caveat


def test_manifest_reports_unseen_category_exposure(development_df: pd.DataFrame) -> None:
    """`handle_unknown='ignore'` silently zero-encodes an unseen venue."""
    coverage = score(development_df, 2024).manifest["expected_rv_coverage"]
    assert 0.0 <= coverage["unseen_venue_share_in_scored_season"] <= 1.0
    assert 0.0 <= coverage["unseen_bb_type_share_in_scored_season"] <= 1.0


def test_unseen_venue_share_detects_a_genuinely_new_venue(
    development_df: pd.DataFrame,
) -> None:
    tampered = development_df.copy()
    tampered.loc[tampered["season"] == 2024, "venue"] = "Brand New Park"

    coverage = score(tampered, 2024).manifest["expected_rv_coverage"]

    assert coverage["unseen_venue_share_in_scored_season"] == pytest.approx(1.0)


def test_feature_schema_drift_across_folds_is_refused() -> None:
    def result(season: int, numeric: list[str]) -> sds.SeasonScoringResult:
        return sds.SeasonScoringResult(
            season=season,
            train_seasons=(2021,),
            ledger=pd.DataFrame(),
            manifest={"feature_schema": {"numeric_features": numeric, "categorical_features": []}},
        )

    consistent = [result(2022, ["launch_speed"]), result(2023, ["launch_speed"])]
    sds.assert_consistent_feature_schema(consistent)

    drifted = [result(2022, ["launch_speed"]), result(2023, ["launch_speed", "hit_distance_sc"])]
    with pytest.raises(sds.ForecastFeatureSchemaDriftError, match="different feature schemas"):
        sds.assert_consistent_feature_schema(drifted)


def test_walk_forward_folds_are_identical_across_seasons(
    development_df: pd.DataFrame,
) -> None:
    """Only fitted parameters may vary across folds -- never the architecture."""
    results = [score(development_df, season) for season in (2022, 2023, 2024)]
    sds.assert_consistent_feature_schema(results)


def test_scoring_a_sealed_season_is_refused(development_df: pd.DataFrame) -> None:
    with pytest.raises(ForecastSeasonError):
        score(development_df, 2025)


def test_ledger_hash_is_row_order_independent(development_df: pd.DataFrame) -> None:
    ledger = score(development_df, 2024).ledger
    shuffled = ledger.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert sds.ledger_content_hash(ledger) == sds.ledger_content_hash(shuffled)


def test_ledger_hash_changes_when_a_value_changes(development_df: pd.DataFrame) -> None:
    ledger = score(development_df, 2024).ledger
    mutated = ledger.copy()
    mutated.loc[0, "baseline_expected_contact_run_value"] = 42.0
    assert sds.ledger_content_hash(ledger) != sds.ledger_content_hash(mutated)


def test_manifest_is_json_serializable(development_df: pd.DataFrame) -> None:
    manifest = score(development_df, 2024).manifest
    assert json.loads(json.dumps(manifest, sort_keys=True))["scored_season"] == 2024


def test_realized_only_reference_carries_no_deserved_column(
    development_df: pd.DataFrame,
) -> None:
    """2021 has no causal contact model; absence is the honest representation."""
    reference = sds.build_realized_only_reference(development_df, 2021)

    assert "baseline_expected_contact_run_value" not in reference.columns
    assert reference["observed_contact_result_run_value"].notna().all()
    assert (reference["season"] == 2021).all()


def test_realized_only_reference_excludes_unresolved_batted_balls(
    development_df: pd.DataFrame,
) -> None:
    reference = sds.build_realized_only_reference(development_df, 2021)
    eligible_2021 = development_df[
        development_df["is_eligible"].fillna(False).astype(bool)
        & (development_df["season"] == 2021)
    ]
    assert len(reference) == int(eligible_2021["outcome_class"].notna().sum())
    assert len(reference) < len(eligible_2021)


def test_realized_only_reference_refuses_a_sealed_season(
    development_df: pd.DataFrame,
) -> None:
    with pytest.raises(ForecastSeasonError):
        sds.build_realized_only_reference(development_df, 2025)
