"""Probability calibration evaluation for the contact model.

Why calibration matters more here than plain ranking accuracy: the luck
score is built from the *magnitude* of predicted probabilities (expected
ordinal value = sum(p(outcome) * value(outcome))), not just which outcome
is most likely. A model that ranks outcomes correctly but assigns
systematically over- or under-confident probabilities will produce raw-luck
values that are biased even when its classifications look accurate. A
"90% single" prediction should mean batters in that situation actually get
a single (or better) about 90% of the time -- if it happens 60% of the
time, every luck calculation built on that number is off, regardless of how
good the model's ranking/accuracy metrics look.

This module only diagnoses and optionally applies calibration; it does not
assert that the resulting probabilities are well-calibrated in general --
that must be judged from the calibration table/plots for the actual data at
hand, on real held-out (non-2025) data. Running this code successfully is
not evidence of good calibration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from mlb_luck_score.config import CLASS_ORDER

logger = logging.getLogger(__name__)

DEFAULT_N_BINS = 10
#: Below this many samples in a probability bin, treat the bin's calibration
#: error as unreliable -- reported but flagged, not hidden.
MIN_RELIABLE_BIN_SAMPLES = 20


@dataclass(frozen=True)
class CalibrationRow:
    outcome_class: str
    bin_low: float
    bin_high: float
    mean_predicted_probability: float
    observed_frequency: float
    sample_count: int
    abs_calibration_error: float
    reliable: bool


def compute_calibration_table(
    y_true: pd.Series | np.ndarray,
    proba_df: pd.DataFrame,
    *,
    n_bins: int = DEFAULT_N_BINS,
    min_reliable_bin_samples: int = MIN_RELIABLE_BIN_SAMPLES,
) -> pd.DataFrame:
    """Build a one-vs-rest calibration table for every outcome class.

    For each class and each equal-width probability bin in [0, 1], reports
    the mean predicted probability, the observed empirical frequency of that
    class among rows falling in the bin, the sample count, and the absolute
    calibration error. Empty bins are omitted. Bins with fewer than
    `min_reliable_bin_samples` rows are kept but marked `reliable=False`.
    """
    if list(proba_df.columns) != list(CLASS_ORDER):
        raise ValueError(f"Expected proba_df columns {CLASS_ORDER}, got {list(proba_df.columns)}")

    y_true_arr = np.asarray(y_true, dtype=str)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[CalibrationRow] = []

    for cls in CLASS_ORDER:
        predicted = proba_df[cls].to_numpy()
        observed = (y_true_arr == cls).astype(float)
        bin_idx = np.clip(np.digitize(predicted, edges[1:-1], right=True), 0, n_bins - 1)

        for b in range(n_bins):
            mask = bin_idx == b
            count = int(mask.sum())
            if count == 0:
                continue
            mean_pred = float(predicted[mask].mean())
            obs_freq = float(observed[mask].mean())
            rows.append(
                CalibrationRow(
                    outcome_class=cls,
                    bin_low=float(edges[b]),
                    bin_high=float(edges[b + 1]),
                    mean_predicted_probability=mean_pred,
                    observed_frequency=obs_freq,
                    sample_count=count,
                    abs_calibration_error=abs(mean_pred - obs_freq),
                    reliable=count >= min_reliable_bin_samples,
                )
            )

    table = pd.DataFrame(r.__dict__ for r in rows)
    if not table.empty:
        n_unreliable = int((~table["reliable"]).sum())
        if n_unreliable:
            logger.warning(
                "%d of %d calibration bin(s) have fewer than %d samples and are marked unreliable.",
                n_unreliable,
                len(table),
                min_reliable_bin_samples,
            )
    return table


def fit_calibrated_classifier(
    fitted_pipeline: Pipeline,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """Fit a post-hoc calibration layer on top of an already-trained pipeline.

    Uses held-out validation data (never training data, never 2025) with
    scikit-learn's `CalibratedClassifierCV` wrapping the frozen (already
    fit) pipeline via `FrozenEstimator`. `method="isotonic"` is a reasonable
    default for moderate validation-set sizes; `"sigmoid"` (Platt scaling)
    is more stable with fewer samples.
    """
    calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(fitted_pipeline), method=method)
    calibrated.fit(x_val, y_val)
    return calibrated


def plot_calibration_curves(calibration_table: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Save one reliability-diagram-style plot per outcome class.

    Requires matplotlib (a dev dependency). Returns the list of file paths
    written. Does nothing (with a log message) if the table is empty.
    """
    if calibration_table.empty:
        logger.warning("Calibration table is empty; no plots to generate.")
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for cls in CLASS_ORDER:
        subset = calibration_table[calibration_table["outcome_class"] == cls]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
        ax.scatter(
            subset["mean_predicted_probability"],
            subset["observed_frequency"],
            s=np.clip(subset["sample_count"], 10, 200),
            alpha=0.8,
        )
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"Calibration: {cls} (n_bins={len(subset)}, sample-size-scaled markers)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        path = output_dir / f"calibration_{cls}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written
