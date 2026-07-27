"""Reusable contact-feature preprocessing pipeline.

Defines the Version 0.1 baseline feature set and a scikit-learn
`ColumnTransformer` that imputes and scales/encodes it. Also provides an
explicit leakage check: nothing computed from or encoding the play's result
may enter the feature set (see `mlb_luck_score.config.LEAKAGE_COLUMNS`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlb_luck_score.config import LEAKAGE_COLUMNS

logger = logging.getLogger(__name__)

#: Identifier columns -- never features, never the target.
ID_COLUMNS: tuple[str, ...] = (
    "event_id",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
)

#: Descriptive metadata -- useful for reporting/joins, never model features.
METADATA_COLUMNS: tuple[str, ...] = (
    "player_name",
    "game_date",
    "season",
    "home_team",
    "away_team",
    "inning",
    "inning_topbot",
)

#: The target column predicted by the contact model.
TARGET_COLUMN = "outcome_class"

#: Version 0.1 baseline numeric contact features.
NUMERIC_FEATURES: tuple[str, ...] = (
    "launch_speed",
    "launch_angle",
    "spray_angle_approx",
    "hit_distance_sc",
)

#: Version 0.1 baseline categorical contact features.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "bb_type",
    "stand",
    "venue",
)

#: Documented extension points -- not used by the Version 0.1 baseline model,
#: but selectable via `include_optional=True` for experimentation.
OPTIONAL_NUMERIC_FEATURES: tuple[str, ...] = ("sprint_speed",)
OPTIONAL_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "if_fielding_alignment",
    "of_fielding_alignment",
)

#: Minimum fraction of non-null values a feature column must have to be
#: included automatically. Below this, `select_available_features` drops the
#: column and logs why, rather than silently modeling on a mostly-empty field.
MIN_NON_NULL_FRACTION = 0.5


class LeakageError(ValueError):
    """Raised when a proposed feature set includes a target-leakage column."""


def assert_no_leakage(feature_columns: Iterable[str]) -> None:
    """Raise if any proposed feature column is a known leakage column.

    This is the single enforcement point for "never use post-outcome
    fields as model features" -- see `mlb_luck_score.config.LEAKAGE_COLUMNS`
    and `tests/test_features.py` for the automated check.
    """
    leaked = set(feature_columns) & LEAKAGE_COLUMNS
    if leaked:
        raise LeakageError(
            f"Refusing to build a feature pipeline with leakage column(s): "
            f"{sorted(leaked)}. These are computed from or encode the play's "
            f"result and must never be model inputs."
        )


def select_available_features(
    df: pd.DataFrame,
    *,
    include_optional: bool = False,
    min_non_null_fraction: float = MIN_NON_NULL_FRACTION,
) -> tuple[list[str], list[str]]:
    """Choose numeric/categorical features actually usable in `df`.

    A candidate feature is dropped (with a log message) if it's absent from
    `df` or if it's present but below `min_non_null_fraction` non-null.
    Nothing is silently imputed away at the column-selection stage --
    per-row imputation happens inside the returned preprocessing pipeline.

    Returns:
        (numeric_features, categorical_features) actually usable.
    """
    numeric_candidates = list(NUMERIC_FEATURES)
    categorical_candidates = list(CATEGORICAL_FEATURES)
    if include_optional:
        numeric_candidates += list(OPTIONAL_NUMERIC_FEATURES)
        categorical_candidates += list(OPTIONAL_CATEGORICAL_FEATURES)

    def _keep(col: str) -> bool:
        if col not in df.columns:
            logger.info("Feature '%s' not present in data; excluding.", col)
            return False
        non_null_fraction = df[col].notna().mean() if len(df) else 0.0
        if non_null_fraction < min_non_null_fraction:
            logger.info(
                "Feature '%s' is only %.1f%% non-null (< %.0f%% threshold); excluding.",
                col,
                non_null_fraction * 100,
                min_non_null_fraction * 100,
            )
            return False
        return True

    numeric_features = [c for c in numeric_candidates if _keep(c)]
    categorical_features = [c for c in categorical_candidates if _keep(c)]

    assert_no_leakage(numeric_features + categorical_features)
    return numeric_features, categorical_features


def build_preprocessing_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes/scales numeric features and
    imputes/one-hot-encodes categorical features.

    Unknown categories seen at prediction time are ignored (encoded as all
    zeros) rather than raising, since new venues/bb_types can appear.
    """
    assert_no_leakage(numeric_features + categorical_features)

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
