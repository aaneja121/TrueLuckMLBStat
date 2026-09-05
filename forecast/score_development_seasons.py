"""Contact Forecast R1: walk-forward scoring of the development seasons.

Produces, for each analysis season, a per-batted-ball ledger carrying
`observed_contact_result_run_value` (Rc), `baseline_expected_contact_run_
value` (E0), and `contact_result_surprise` (Rc - E0) -- using the FROZEN
Contact Luck contact model, refit on strictly earlier seasons.

## These are CONTACT-STAGE quantities, not the published Contact Luck

The names above are production's own internal names for the contact stage
(`mlb_luck_score.scoring.attribution_ledger`). They are deliberately NOT the
production play-ledger names `observed_run_value`/`expected_run_value`/
`contact_luck_runs`, which hold Rf and Rf - E0 -- the FULL telescoping
quantity including batter-runner advancement, and the thing contactluck.com
actually publishes. (Production learned this distinction the hard way: the
Phase 3.1 fix in `play_ledger_export` exists because that exporter had been
sourcing Rc - E0 and failed per-batter reconciliation for 261/630 batters.)

Building Rf per fold would require training the infield and advancement
models, which carry two model-SELECTION procedures whose winners could
differ across folds -- architecture drift, not a pure refit. The maintainer
chose the contact stage on 2026-09-03 for exactly that reason. At the
batter-season level the two agree closely (2024, 216 qualified hitters:
Pearson 0.995, Spearman 0.994, mean absolute difference 0.16 runs/100
against a metric SD of 2.12), but they are NOT the same quantity and no
report may call these values "Contact Luck" without the qualifier.

## Fit-time causality versus design-time causality

These are two different claims and this module only establishes the first.

**Fit-time causality (established here, and enforced).** No row from the
scored season enters the estimator. 2022 is scored by parameters fit on 2021
alone; 2023 on 2021-2022; 2024 on 2021-2023. `assert_no_scored_season_rows_
in_training` proves it on the actual training frame, and the manifest records
the confirmation per season.

**Design-time causality (NOT established, and not claimable).** The Contact
Luck architecture itself -- its feature set, model family, class handling,
run-value table, and eligibility rules -- was selected during project
development that ran across 2021-2024 and was frozen only afterwards. A 2022
fold is therefore a **walk-forward refit of the frozen current Contact Luck
specification**, NOT a model that historically existed in 2022 and NOT a
reconstruction of what a 2022 analyst could have built. Any report language
implying the latter is wrong. The honest description is: architecture held
fixed at today's frozen specification; only fitted parameters vary across
folds.

Nothing here performs new feature selection, model-family selection,
threshold selection, calibration choice, or hyperparameter selection, and
nothing here consults the season being scored to make any such choice. The
one architectural step that IS data-dependent -- `select_available_features`,
which drops a candidate feature below 50% non-null in the TRAINING frame --
runs inside `train_model` on prior-season rows only, and this module compares
the resulting feature schema across folds and FAILS if it varies (see
`ForecastFeatureSchemaDriftError`). A fold-dependent feature set would be an
architecture change, not a refit, and must be surfaced rather than absorbed.

## What this module deliberately does not build

Production's full ledger telescopes contact, outfield/infield defensive
opportunity, and batter-runner advancement. This study needs only the contact
stage: "deserved" is E0, the contact model's expected run value, and Contact
Luck is `Rc - E0`. The three other component models are not trained here --
not to save time, but because training them would introduce three further
model-selection procedures (`run_infield_model_selection`,
`run_advancement_model_selection`) whose selection steps would then need
their own walk-forward treatment. Their outputs do not enter either target,
so the correct scope is to leave them out and say so.

## Unresolved batted balls

An eligible batted ball whose `events` is `field_error` or `fielders_choice`
has a null `outcome_class` (`mlb_luck_score.eligibility`), hence no observed
run value. Production nulls every result-linked column for those rows and
excludes them from the per-100 denominator; this module reproduces that
exactly, and the manifest reports the counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast.forecast_config import (
    FORECAST_ANALYSIS_SEASONS,
    FORECAST_DATA_DIR,
    RANDOM_SEED,
    assert_forecast_seasons_allowed,
    assert_path_outside_forbidden_namespaces,
    walk_forward_train_seasons,
)
from mlb_luck_score.config import CLASS_ORDER, PROCESSED_DATA_DIR, assert_seasons_allowed
from mlb_luck_score.eligibility import compute_eligibility
from mlb_luck_score.models.train_contact_model import (
    VARIANT_UNWEIGHTED,
    TrainedModel,
    predict_proba_ordered,
    train_model,
    validate_probabilities,
)
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP
from mlb_luck_score.scoring.weather_attribution import compute_expected_run_value_vectorized

logger = logging.getLogger(__name__)

#: Identifies the frozen Contact Luck configuration these ledgers were
#: produced under. Bump only for a genuine change to the scoring contract --
#: it is what tells a later reader whether two ledgers are comparable.
FORECAST_SCORING_CONTRACT_VERSION = "contact_forecast_r1_walk_forward_v1"

#: The frozen contact-model call this module makes. `class_weight=None` is
#: load-bearing, not a default to tidy away: `"balanced"` is documented to
#: severely miscalibrate the probabilities that Contact Luck is built from
#: (RESEARCH_RULES.md, "Never use class_weight='balanced'"). Contact Luck is
#: `sum(p * run_value) - observed`, so miscalibrated probabilities corrupt
#: every deserved value in this study.
#: Imported rather than restated as a literal so a rename upstream cannot
#: leave this module silently asserting against a stale variant name.
FROZEN_CONTACT_MODEL_VARIANT = VARIANT_UNWEIGHTED
FROZEN_CONTACT_MODEL_CLASS_WEIGHT: str | None = None

#: The name production logs for this configuration (`run_season_aggregation`
#: says "Contact model (baseline_v02)"). Recorded alongside the variant so a
#: manifest reader can tie a ledger back to the documented model version.
FROZEN_CONTACT_MODEL_NAME = "baseline_v02"

#: The default development dataset. The richest cleaned development file --
#: it carries the venue/geometry/sprint columns the baseline feature set
#: selects from. Never a prospective or final-evaluation path (guarded).
DEFAULT_DEVELOPMENT_INPUT = (
    PROCESSED_DATA_DIR / "cleaned_development_data_with_sprint_speed.parquet"
)

#: Columns of the emitted ledger. Deliberately the production play-ledger
#: shape (`prospective` snapshots' `play_ledger.parquet`) so the same window
#: and feature code reads either without a translation layer.
LEDGER_COLUMNS: tuple[str, ...] = (
    "event_id",
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
    "venue",
    "is_scored",
    "outcome_class",
    "observed_contact_result_run_value",
    *(f"p_{cls}" for cls in CLASS_ORDER),
    "baseline_expected_contact_run_value",
    "contact_result_surprise",
)


class ForecastScoringError(RuntimeError):
    """Raised when a walk-forward scoring precondition or invariant fails."""


class ForecastLeakageError(ForecastScoringError):
    """Raised when scored-season rows are found in a training frame."""


class ForecastFeatureSchemaDriftError(ForecastScoringError):
    """Raised when the selected feature schema differs across walk-forward folds.

    The frozen architecture is supposed to be constant across folds, with
    only fitted parameters changing. A differing feature set means an
    architectural difference crept in -- most likely a column whose
    availability changed between seasons crossing `select_available_
    features`' 50%-non-null bar in one fold but not another. That is a
    finding to document and decide on, never something to absorb silently.
    """


@dataclass(frozen=True)
class SeasonScoringResult:
    """One season's walk-forward ledger and its manifest."""

    season: int
    train_seasons: tuple[int, ...]
    ledger: pd.DataFrame
    manifest: dict[str, Any] = field(default_factory=dict)


def assert_no_scored_season_rows_in_training(
    train_df: pd.DataFrame, scored_season: int
) -> dict[str, Any]:
    """Prove, on the actual training frame, that the scored season is absent.

    The central fit-time-causality check. Deliberately operates on the frame
    that is about to be handed to the estimator rather than on the season
    list that was intended, so a filtering bug upstream cannot pass.

    Returns:
        A confirmation record for the manifest: the seasons actually present
        in the training frame, the scored season, and the row count.

    Raises:
        ForecastLeakageError: If any training row belongs to `scored_season`,
            or to a season later than it.
    """
    observed = sorted(int(s) for s in train_df["season"].unique())
    contaminated = int((train_df["season"].astype(int) == int(scored_season)).sum())
    if contaminated:
        raise ForecastLeakageError(
            f"{contaminated} row(s) from the scored season {scored_season} are present in "
            "its own training frame -- the fitted parameters would have seen the season "
            "they are used to score"
        )
    later = [s for s in observed if s > int(scored_season)]
    if later:
        raise ForecastLeakageError(
            f"Training frame for season {scored_season} contains later season(s) {later}; "
            "walk-forward scoring must use strictly earlier seasons only"
        )
    return {
        "scored_season": int(scored_season),
        "seasons_present_in_training_frame": observed,
        "scored_season_rows_in_training_frame": 0,
        "training_rows": int(len(train_df)),
        "confirmed": True,
    }


def feature_schema(trained: TrainedModel) -> dict[str, list[str]]:
    """The feature schema a fitted model actually selected."""
    return {
        "numeric_features": list(trained.numeric_features),
        "categorical_features": list(trained.categorical_features),
    }


def _prepare_development_frame(input_path: Path) -> pd.DataFrame:
    """Load the development dataset and apply frozen eligibility."""
    assert_path_outside_forbidden_namespaces(input_path)
    if not input_path.exists():
        raise ForecastScoringError(f"Development dataset not found: {input_path}")

    df = pd.read_parquet(input_path)
    observed: list[int | str] = sorted(int(s) for s in df["season"].dropna().unique())
    # Both guards: the repository-wide 2025 rule and this package's own
    # narrower authorization.
    assert_seasons_allowed(observed)
    assert_forecast_seasons_allowed(observed)

    # Recomputed rather than trusted from the file, exactly as production's
    # `run_season_aggregation.build_player_season_report` does -- eligibility
    # lives in `mlb_luck_score.eligibility` and is never re-implemented or
    # assumed here.
    return compute_eligibility(df)


def score_season(
    season: int,
    *,
    development_df: pd.DataFrame,
    input_path: Path,
    input_sha256: str,
) -> SeasonScoringResult:
    """Fit on strictly earlier seasons, then score `season`'s eligible batted balls.

    Args:
        season: The analysis season to score.
        development_df: The eligibility-computed development frame.
        input_path: Source dataset path, for the manifest.
        input_sha256: Source dataset hash, for the manifest.

    Returns:
        A `SeasonScoringResult` whose `.ledger` carries `LEDGER_COLUMNS`.

    Raises:
        ForecastLeakageError: If the scored season reaches the training frame.
        ForecastScoringError: On an empty training or scoring frame, or a
            probability-validation failure.
    """
    train_seasons = walk_forward_train_seasons(season)

    train_df = development_df[
        development_df["eligible_for_training"].fillna(False).astype(bool)
        & development_df["season"].isin(train_seasons)
    ]
    if train_df.empty:
        raise ForecastScoringError(
            f"No training-eligible rows for seasons {train_seasons} (scoring {season})"
        )
    causality = assert_no_scored_season_rows_in_training(train_df, season)

    trained = train_model(train_df, class_weight=FROZEN_CONTACT_MODEL_CLASS_WEIGHT)
    schema = feature_schema(trained)
    logger.info(
        "Season %d: contact model fit on %d rows from seasons %s; features %s",
        season,
        len(train_df),
        list(train_seasons),
        schema,
    )

    score_df = development_df[
        development_df["is_eligible"].fillna(False).astype(bool)
        & (development_df["season"].astype(int) == int(season))
    ].copy()
    if score_df.empty:
        raise ForecastScoringError(f"No eligible batted balls to score in season {season}")

    feature_cols = list(trained.numeric_features) + list(trained.categorical_features)
    proba = predict_proba_ordered(trained, score_df[feature_cols])
    validate_probabilities(proba)

    expected = compute_expected_run_value_vectorized(proba)
    observed = score_df["outcome_class"].astype(object).map(DEFAULT_RUN_VALUE_MAP).astype(float)
    resolved = score_df["outcome_class"].notna()

    ledger = pd.DataFrame(index=score_df.index)
    for column in (
        "event_id",
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
        "venue",
        "outcome_class",
    ):
        ledger[column] = score_df[column] if column in score_df.columns else np.nan
    for cls in CLASS_ORDER:
        ledger[f"p_{cls}"] = proba[cls]
    ledger["observed_contact_result_run_value"] = observed
    ledger["baseline_expected_contact_run_value"] = expected
    ledger["contact_result_surprise"] = observed - expected
    ledger["is_scored"] = resolved

    # Production nulls EVERY result-linked column on an unresolved row
    # (`attribution_ledger.build_attribution_ledger`'s `unresolved` loop),
    # including the probability vector -- so a consumer never sees a
    # probability for a play whose outcome is unresolved. Reproduced here.
    unresolved = ~resolved
    for column in (
        "observed_contact_result_run_value",
        "baseline_expected_contact_run_value",
        "contact_result_surprise",
        *(f"p_{cls}" for cls in CLASS_ORDER),
    ):
        ledger.loc[unresolved, column] = np.nan

    ledger = ledger[list(LEDGER_COLUMNS)].reset_index(drop=True)

    manifest = _build_manifest(
        season=season,
        train_seasons=train_seasons,
        train_df=train_df,
        score_df=score_df,
        ledger=ledger,
        trained=trained,
        schema=schema,
        causality=causality,
        input_path=input_path,
        input_sha256=input_sha256,
    )
    return SeasonScoringResult(
        season=int(season),
        train_seasons=train_seasons,
        ledger=ledger,
        manifest=manifest,
    )


def build_realized_only_reference(development_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Observed contact-result run value for a season, with NO model involved.

    Exists for 2021, which is a feature-source season: it has no strictly
    earlier development season to fit a causal contact model on, so it can
    never carry a `baseline_expected_contact_run_value`. But its REALIZED
    rate needs no model at all -- it is the frozen run-value table applied to
    the recorded outcome -- so it can still serve as a prior-season realized
    feature for 2022 windows.

    Returns a frame with `batter`, `season`, and
    `observed_contact_result_run_value`, restricted to resolved batted balls.
    Deliberately does NOT emit a deserved column: the honest representation
    of "no causal model exists for this season" is absence, not a value.

    Raises:
        ForecastScoringError: If the season has no resolved batted balls.
    """
    assert_forecast_seasons_allowed(int(season))
    rows = development_df[
        development_df["is_eligible"].fillna(False).astype(bool)
        & (development_df["season"].astype(int) == int(season))
        & development_df["outcome_class"].notna()
    ]
    if rows.empty:
        raise ForecastScoringError(f"No resolved batted balls in season {season}")

    return pd.DataFrame(
        {
            "batter": rows["batter"].to_numpy(),
            "season": int(season),
            "observed_contact_result_run_value": rows["outcome_class"]
            .astype(object)
            .map(DEFAULT_RUN_VALUE_MAP)
            .astype(float)
            .to_numpy(),
        }
    )


def _unseen_category_share(train_df: pd.DataFrame, score_df: pd.DataFrame, column: str) -> float:
    """Share of scored rows whose category never appeared in training.

    `OneHotEncoder(handle_unknown="ignore")` encodes an unseen category as
    all zeros rather than raising, so a venue that did not exist in the
    training seasons is scored as though it carried no venue information at
    all. That is silent by design, which is exactly why it is measured and
    reported rather than left to be discovered later.
    """
    if column not in train_df.columns or column not in score_df.columns:
        return float("nan")
    seen = set(train_df[column].dropna().unique())
    scored = score_df[column]
    unseen = ~scored.isin(seen) & scored.notna()
    return float(unseen.sum()) / float(len(scored)) if len(scored) else float("nan")


def _build_manifest(
    *,
    season: int,
    train_seasons: tuple[int, ...],
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    ledger: pd.DataFrame,
    trained: TrainedModel,
    schema: dict[str, list[str]],
    causality: dict[str, Any],
    input_path: Path,
    input_sha256: str,
) -> dict[str, Any]:
    """Assemble the per-season scoring manifest."""
    n_scored = int(ledger["is_scored"].sum())
    n_rows = int(len(ledger))
    unresolved_by_event: dict[str, int] = {}
    if "events" in score_df.columns:
        unresolved_events = score_df.loc[score_df["outcome_class"].isna(), "events"]
        unresolved_by_event = {str(k): int(v) for k, v in unresolved_events.value_counts().items()}

    return {
        "contract_version": FORECAST_SCORING_CONTRACT_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scored_season": int(season),
        "training_seasons": [int(s) for s in train_seasons],
        "causality": {
            "fit_time": causality,
            "design_time_caveat": (
                "The Contact Luck architecture (feature set, model family, class "
                "handling, run-value table, eligibility rules) was selected during "
                "project development spanning 2021-2024 and frozen afterwards. This "
                "ledger is a WALK-FORWARD REFIT OF THE FROZEN CURRENT SPECIFICATION on "
                "strictly earlier rows -- it is NOT a model that historically existed "
                "in the scored season, and must never be described as one."
            ),
            "no_selection_performed_on_scored_season": True,
            "selection_steps_performed_by_this_module": [],
        },
        "row_counts": {
            "training_rows": int(len(train_df)),
            "training_rows_by_season": {
                str(season_value): int(count)
                for season_value, count in train_df["season"].value_counts().sort_index().items()
            },
            "eligible_batted_balls_in_scored_season": n_rows,
            "scored_resolved_batted_balls": n_scored,
            "unresolved_excluded_from_per_100_denominator": n_rows - n_scored,
            "unresolved_by_event": unresolved_by_event,
        },
        "frozen_model": {
            "variant": trained.variant,
            "expected_variant": FROZEN_CONTACT_MODEL_VARIANT,
            "model_name": FROZEN_CONTACT_MODEL_NAME,
            "class_weight": trained.class_weight,
            "random_seed": RANDOM_SEED,
            "estimator": "mlb_luck_score.models.train_contact_model.train_model",
            "run_value_map": {k: float(v) for k, v in DEFAULT_RUN_VALUE_MAP.items()},
            "run_value_map_sha256": _hash_json(
                {k: float(v) for k, v in DEFAULT_RUN_VALUE_MAP.items()}
            ),
            "class_order": list(CLASS_ORDER),
        },
        "feature_schema": schema,
        "feature_schema_sha256": _hash_json(schema),
        "expected_rv_coverage": {
            "rows_with_expected_run_value": int(
                ledger["baseline_expected_contact_run_value"].notna().sum()
            ),
            "coverage_fraction": (float(n_scored) / n_rows if n_rows else float("nan")),
            "unseen_venue_share_in_scored_season": _unseen_category_share(
                train_df, score_df, "venue"
            ),
            "unseen_bb_type_share_in_scored_season": _unseen_category_share(
                train_df, score_df, "bb_type"
            ),
        },
        "artifact_hashes": {
            "input_dataset_path": str(input_path),
            "input_dataset_sha256": input_sha256,
            "ledger_sha256": ledger_content_hash(ledger),
        },
    }


def _hash_json(payload: Any) -> str:
    """Stable SHA-256 over a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ledger_content_hash(ledger: pd.DataFrame) -> str:
    """Deterministic content hash of a ledger, independent of row order.

    Sorted by `event_id` before hashing so two runs that emit identical
    content in a different order hash identically -- the hash identifies the
    DATA, not an incidental ordering.
    """
    ordered = ledger.sort_values("event_id").reset_index(drop=True)
    return hashlib.sha256(
        pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_consistent_feature_schema(results: list[SeasonScoringResult]) -> None:
    """Fail if the selected feature schema is not identical across folds.

    Only fitted PARAMETERS may vary across walk-forward seasons. A differing
    feature set is an architectural difference and must be surfaced.

    Raises:
        ForecastFeatureSchemaDriftError: If any two folds selected different
            features.
    """
    schemas = {r.season: r.manifest["feature_schema"] for r in results}
    distinct = {json.dumps(s, sort_keys=True) for s in schemas.values()}
    if len(distinct) > 1:
        raise ForecastFeatureSchemaDriftError(
            "Walk-forward folds selected different feature schemas, so the architecture "
            "is not constant across folds and the seasons are not comparable as pure "
            f"refits: {json.dumps(schemas, sort_keys=True, indent=2)}"
        )


def run(
    *,
    seasons: tuple[int, ...] = FORECAST_ANALYSIS_SEASONS,
    input_path: Path = DEFAULT_DEVELOPMENT_INPUT,
    output_dir: Path = FORECAST_DATA_DIR,
) -> list[SeasonScoringResult]:
    """Score every requested analysis season and write ledgers plus manifests."""
    assert_forecast_seasons_allowed(list(seasons))
    assert_path_outside_forbidden_namespaces(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    development_df = _prepare_development_frame(input_path)
    input_sha256 = file_sha256(input_path)

    results: list[SeasonScoringResult] = []
    for season in seasons:
        result = score_season(
            season,
            development_df=development_df,
            input_path=input_path,
            input_sha256=input_sha256,
        )
        results.append(result)

    assert_consistent_feature_schema(results)

    for result in results:
        ledger_path = output_dir / f"walk_forward_ledger_{result.season}.parquet"
        manifest_path = output_dir / f"walk_forward_manifest_{result.season}.json"
        result.ledger.to_parquet(ledger_path, index=False)
        manifest_path.write_text(json.dumps(result.manifest, indent=2, sort_keys=True))
        logger.info("Season %d: wrote %s and %s", result.season, ledger_path, manifest_path)

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=list(FORECAST_ANALYSIS_SEASONS),
        help="Analysis seasons to score (default: every Contact Forecast analysis season).",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DEVELOPMENT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=FORECAST_DATA_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    results = run(seasons=tuple(args.seasons), input_path=args.input, output_dir=args.output_dir)
    for result in results:
        counts = result.manifest["row_counts"]
        print(
            f"season {result.season}: trained on {list(result.train_seasons)} "
            f"({counts['training_rows']} rows), scored "
            f"{counts['scored_resolved_batted_balls']}/"
            f"{counts['eligible_batted_balls_in_scored_season']} eligible batted balls"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
