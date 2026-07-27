"""Naive baseline models used as calibration/comparison benchmarks.

These are NOT candidates for the Contact Luck probability model -- they
exist purely as sanity-check floors. If a real, feature-driven model can't
beat (in log loss / calibration terms) a model that ignores every feature
and just predicts the training-set's marginal class distribution for every
row, something is badly wrong with the real model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlb_luck_score.config import CLASS_ORDER
from mlb_luck_score.features.build_contact_features import TARGET_COLUMN


@dataclass(frozen=True)
class PrevalenceBaseline:
    """Predicts the training-set marginal class distribution for every row.

    Attributes:
        frequencies: Training-set class frequency for each class in
            `class_order`. Sums to ~1 (assuming every class had at least
            one example; a class absent from training gets frequency 0.0).
        class_order: The class ordering `frequencies` and
            `predict_proba_prevalence` use.
    """

    frequencies: dict[str, float]
    class_order: tuple[str, ...] = CLASS_ORDER


def train_prevalence_baseline(
    train_df: pd.DataFrame, *, class_order: tuple[str, ...] = CLASS_ORDER
) -> PrevalenceBaseline:
    """Compute the training-set marginal class distribution.

    Args:
        train_df: Training-eligible rows with a `TARGET_COLUMN` column.
        class_order: Class ordering to use.

    Returns:
        A `PrevalenceBaseline` holding the observed training frequencies.
    """
    counts = train_df[TARGET_COLUMN].astype(str).value_counts(normalize=True)
    frequencies = {cls: float(counts.get(cls, 0.0)) for cls in class_order}
    return PrevalenceBaseline(frequencies=frequencies, class_order=tuple(class_order))


def predict_proba_prevalence(
    model: PrevalenceBaseline, n_rows: int, *, index: pd.Index | None = None
) -> pd.DataFrame:
    """Predict the same training-set marginal distribution for `n_rows` rows.

    Every row gets an identical probability row (the model ignores all
    features by design) -- columns are ordered per `model.class_order`.
    """
    data = {cls: [model.frequencies[cls]] * n_rows for cls in model.class_order}
    proba_df = pd.DataFrame(data)
    if index is not None:
        proba_df.index = index
    return proba_df[list(model.class_order)]
