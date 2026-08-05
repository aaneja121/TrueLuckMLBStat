"""Contact Luck v0.7D: sample-size-aware calibration gate for the near-wall specialist.

Versions 0.7A/0.7C evaluated near-wall subgroup/venue calibration with a single
FIXED absolute ECE threshold (`MATERIAL_ECE_ABSOLUTE_THRESHOLD = 0.05`) and a
binary pass/fail per subgroup. AGENTS.md/CLAUDE.md ("Subgroup ECE against a fixed
absolute threshold does not transfer across population sizes") already documents
why this is unsound at near-wall subgroup scale: 28 of 30 reliably-sampled venues
(each only ~200-330 rows) nominally exceeded 0.05 on real 2024 data, almost
certainly sampling-variance noise rather than genuine miscalibration, and the
same fixed threshold was FLAGGING venues on point estimates alone with no sense
of how much those point estimates could plausibly vary.

This module does NOT change the near-wall model, its features, hyperparameters,
or predictions, and does NOT change `open_field_v07` or the contact-only
fallback -- see `mlb_luck_score.models.compare_near_wall_models` for all of
that, reused here UNCHANGED (`get_near_wall_rows`, `run_near_wall_model_selection`,
`run_near_wall_final_comparison`, `run_near_wall_perturbation_checks`,
`check_no_open_field_regression`). It replaces ONLY the subgroup/venue
CALIBRATION-GATING rule: instead of "point-estimate ECE > 0.05 => fail," every
subgroup and venue now gets a game_pk-clustered bootstrap confidence interval on
its own ECE, plus a PAIRED bootstrap CI comparing the specialist against
`open_field_v07` on the SAME rows, and is assigned one of three statuses:

  - `SUBGROUP_STATUS_CALIBRATED`: adequate outcome support, AND the ECE
    confidence interval sits entirely at or below the material threshold, AND
    no credible paired regression against the baseline.
  - `SUBGROUP_STATUS_NOT_CALIBRATED`: adequate outcome support, but CREDIBLE
    evidence of a problem -- either (a) the ECE confidence interval sits
    ENTIRELY above the material threshold (the whole plausible range is bad,
    not just the point estimate), or (b) the paired bootstrap CI for the
    specialist-minus-baseline log-loss delta sits entirely above zero (a
    credible regression vs. what `open_field_v07` would have predicted for
    the identical rows).
  - `SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE`: EITHER raw outcome support (plays,
    games, or minority-class count) falls below a documented minimum, OR
    support is nominally adequate but the ECE confidence interval straddles
    the material threshold -- the data cannot credibly distinguish "calibrated"
    from "not calibrated" at this subgroup's size. This is explicitly NOT a
    pass and NOT a failure; see "Measured miscalibration vs. insufficient
    evidence" in CLAUDE.md/AGENTS.md/README.md.

## Thresholds are centralized and NOT tuned to make 2024 pass

Every threshold below is either REUSED from an earlier, independently-established
Version 0.7A/C convention (`MATERIAL_ECE_ABSOLUTE_THRESHOLD`, `MIN_RELIABLE_
SAMPLES`-style minimums of 20/100) or is the standard 95%/500-rep game-clustered
bootstrap convention already used throughout this codebase (`mlb_luck_score.
models.compare_near_wall_models.compute_near_wall_paired_bootstrap`,
`compare_opportunity_models.compute_opportunity_bootstrap`). None of these were
selected by searching for values that flip `near_wall_specialist_calibrated` to
`True` on real 2024 data -- see module-level constants below, each with its own
provenance note.

## `near_wall_specialist_calibrated` recomputed (see `summarize_calibration_gate`)

Requires ALL of:
  1. Every REQUIRED physical perturbation check
     (`mlb_luck_score.models.compare_near_wall_models.run_near_wall_
     perturbation_checks`, UNCHANGED) passes.
  2. The architectural no-open-field-regression check
     (`check_no_open_field_regression`, UNCHANGED) passes.
  3. The aggregate (whole near-wall population) log-loss improvement and its
     game-clustered bootstrap CI (`run_near_wall_final_comparison`/
     `compute_near_wall_paired_bootstrap`, UNCHANGED) both support the
     specialist over `open_field_v07`.
  4. No adequately-supported subgroup or venue is credibly
     `SUBGROUP_STATUS_NOT_CALIBRATED`.

If all four hold and EVERY subgroup/venue additionally has adequate evidence
(none `SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE`), the overall status is
`OVERALL_STATUS_CALIBRATED` and `near_wall_specialist_calibrated: True`. If all
four hold but at least one subgroup/venue lacks adequate evidence, the overall
status is `OVERALL_STATUS_CALIBRATED_LIMITED_EVIDENCE` -- reported honestly as
its OWN distinct label (never silently folded into an unconditional "pass"),
and `near_wall_specialist_calibrated` STAYS `False` (`provisional_near_wall`
downstream in `mlb_luck_score.scoring.gated_outfield_report`) -- see "Reporting
must remain honest" in the task and CLAUDE.md/AGENTS.md. Otherwise the overall
status is `OVERALL_STATUS_NOT_CALIBRATED`.

## Version 0.7 is now FROZEN

See CLAUDE.md/AGENTS.md/README.md "Version 0.7 is frozen" -- this module is the
last near-wall change permitted to use 2024 as if it were untouched validation
data. 2024 must be treated as DEVELOPMENT validation, not a pristine test set,
for any FUTURE near-wall work.

Usage:

    python -m mlb_luck_score.models.compare_near_wall_calibration_gate \\
        --input data/processed/cleaned_development_data_with_geometry.parquet \\
        --output-dir outputs/tables \\
        --figures-dir outputs/figures/near_wall_calibration_gate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from mlb_luck_score.config import (
    CALIBRATION_BASE_TRAIN_SEASONS,
    CALIBRATION_EVAL_SEASONS,
    CALIBRATION_FIT_SEASONS,
    TABLES_DIR,
    TRAIN_SEASONS,
    assert_seasons_allowed,
)
from mlb_luck_score.features.build_contact_features import OUTFIELD_OPPORTUNITY_TARGET_COLUMN
from mlb_luck_score.models.compare_geometry_aware import _bool_mask
from mlb_luck_score.models.compare_near_wall_models import (
    MATERIAL_ECE_ABSOLUTE_THRESHOLD,
    VARIANT_NEAR_WALL_FINAL,
    VARIANT_OPEN_FIELD,
    NearWallPerturbationSuite,
    check_no_open_field_regression,
    compute_near_wall_paired_bootstrap,
    compute_wall_height_buckets,
    get_near_wall_rows,
    run_near_wall_final_comparison,
    run_near_wall_model_selection,
    run_near_wall_perturbation_checks,
)
from mlb_luck_score.models.compare_opportunity_models import (
    _prepare_opportunity_columns,
    compute_binary_ece,
    compute_opportunity_time_buckets,
)
from mlb_luck_score.models.compare_park_aware import _prepare_venue_column
from mlb_luck_score.models.train_opportunity_model import (
    TrainedOpportunityModel,
    predict_opportunity_proba,
    train_opportunity_model,
    validate_opportunity_probabilities,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Centralized, documented thresholds -- see module docstring "Thresholds are
# centralized and NOT tuned to make 2024 pass".
# ---------------------------------------------------------------------------

#: REUSED verbatim from `compare_near_wall_models`/`compare_opportunity_models`
#: (established in Version 0.7A/C, before this correction existed) -- the
#: absolute calibration-quality bar itself is unchanged; what changes is how
#: much evidence is required before treating a point estimate as credible.
_MATERIAL_ECE_ABSOLUTE_THRESHOLD = MATERIAL_ECE_ABSOLUTE_THRESHOLD

#: Minimum plays in a subgroup/venue before its calibration is evaluated for
#: pass/fail AT ALL -- matches `DEFAULT_MIN_SUBGROUP_SAMPLES` from Version
#: 0.7A/C (unchanged convention, not newly chosen for this correction).
MIN_SUBGROUP_PLAYS = 100
#: Minimum DISTINCT games in a subgroup/venue -- reuses `MIN_RELIABLE_BIN_
#: SAMPLES` (20, `compare_opportunity_models.compute_binary_calibration_table`)
#: as the general "minimum count for a statistic to be minimally reliable"
#: convention already established in this codebase, applied here to games
#: rather than bin samples: with fewer than this many distinct games, a
#: game-clustered bootstrap has too few resampling units to produce a
#: meaningfully varying distribution.
MIN_SUBGROUP_GAMES = 20
#: Minimum count of EACH outcome class (converted-to-out and not) -- same
#: reused 20-count convention; log loss/ECE/Brier for a class with fewer than
#: this many examples is not a meaningful estimate.
MIN_CLASS_SUPPORT = 20

DEFAULT_N_BOOTSTRAP_REPS = 500
DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_BOOTSTRAP_CI = 0.95

#: Adaptive-bin ECE -- number of bins shrinks with sample size (unlike the
#: fixed 10-bin table in `compare_opportunity_models.
#: compute_binary_calibration_table`) so a small subgroup isn't split into
#: bins too sparse to estimate reliably. `ADAPTIVE_ECE_MIN_BIN_SAMPLES` reuses
#: the SAME `MIN_RELIABLE_BIN_SAMPLES` convention (20) as the fixed-bin table.
ADAPTIVE_ECE_MIN_BIN_SAMPLES = 20
ADAPTIVE_ECE_MAX_BINS = 10

#: Minimum rows before attempting a calibration-intercept/slope fit -- a
#: logistic recalibration regression on too few rows (or with only one
#: outcome class present) is not estimable; see `compute_calibration_
#: intercept_slope`.
MIN_SAMPLES_FOR_INTERCEPT_SLOPE = 2 * MIN_CLASS_SUPPORT

SUBGROUP_STATUS_CALIBRATED = "calibrated"
SUBGROUP_STATUS_NOT_CALIBRATED = "not_calibrated"
SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE = "insufficient_evidence"

OVERALL_STATUS_CALIBRATED = "calibrated"
OVERALL_STATUS_CALIBRATED_LIMITED_EVIDENCE = "calibrated_with_limited_subgroup_evidence"
OVERALL_STATUS_NOT_CALIBRATED = "not_calibrated"

_MISSING_VENUE_LABEL = "__missing_venue__"


# ---------------------------------------------------------------------------
# Adaptive-bin ECE and calibration intercept/slope
# ---------------------------------------------------------------------------


def compute_adaptive_calibration_table(
    y_true: np.ndarray,
    p_out: pd.Series,
    *,
    min_bin_samples: int = ADAPTIVE_ECE_MIN_BIN_SAMPLES,
    max_bins: int = ADAPTIVE_ECE_MAX_BINS,
) -> pd.DataFrame:
    """Quantile-binned calibration table whose bin COUNT shrinks with sample size.

    `n_bins = max(1, min(max_bins, n // min_bin_samples))` -- a subgroup with
    e.g. 250 rows gets ~12 candidate bins capped at `max_bins`, while one with
    40 rows gets exactly 2, and one with 15 rows gets a single bin covering
    every row (an ECE of "how far off is the mean prediction, over everything
    we have," which is still informative even though it can't be broken down
    further). Same output shape (`sample_count`, `abs_calibration_error`) as
    `mlb_luck_score.models.compare_opportunity_models.
    compute_binary_calibration_table`, so `compute_binary_ece` (sample-count
    -weighted mean absolute calibration error) is reused directly.
    """
    predicted = np.asarray(p_out.to_numpy(), dtype=float)
    y_arr = np.asarray(y_true)
    n = len(predicted)
    columns = [
        "mean_predicted_probability",
        "observed_frequency",
        "sample_count",
        "abs_calibration_error",
        "reliable",
    ]
    if n == 0:
        return pd.DataFrame(columns=columns)

    n_bins = max(1, min(max_bins, n // min_bin_samples))

    def _single_bin_table() -> pd.DataFrame:
        mean_pred = float(predicted.mean())
        obs_freq = float(y_arr.mean())
        return pd.DataFrame(
            [
                {
                    "mean_predicted_probability": mean_pred,
                    "observed_frequency": obs_freq,
                    "sample_count": n,
                    "abs_calibration_error": abs(mean_pred - obs_freq),
                    "reliable": n >= min_bin_samples,
                }
            ]
        )

    if n_bins <= 1:
        return _single_bin_table()

    try:
        bins = pd.qcut(pd.Series(predicted), q=n_bins, duplicates="drop")
    except ValueError:
        # All values identical (or too few unique values for n_bins edges) --
        # can't form quantile bins at all; fall back to one bin.
        return _single_bin_table()

    frame = pd.DataFrame({"p": predicted, "y": y_arr, "bin": bins})
    rows: list[dict[str, Any]] = []
    for _, group in frame.groupby("bin", observed=True):
        count = len(group)
        if count == 0:
            continue
        mean_pred = float(group["p"].mean())
        obs_freq = float(group["y"].mean())
        rows.append(
            {
                "mean_predicted_probability": mean_pred,
                "observed_frequency": obs_freq,
                "sample_count": count,
                "abs_calibration_error": abs(mean_pred - obs_freq),
                "reliable": count >= min_bin_samples,
            }
        )
    return pd.DataFrame(rows) if rows else _single_bin_table()


def compute_calibration_intercept_slope(
    y_true: np.ndarray,
    p_out: np.ndarray,
    *,
    min_samples: int = MIN_SAMPLES_FOR_INTERCEPT_SLOPE,
) -> tuple[float | None, float | None]:
    """Calibration-recalibration intercept/slope, WHERE ESTIMABLE.

    Fits `y ~ logit(p_out)` via a single-feature `LogisticRegression` -- the
    standard calibration-recalibration diagnostic (a well-calibrated model has
    intercept ~= 0, slope ~= 1). Returns `(None, None)` if not estimable: fewer
    than `min_samples` rows, only one outcome class present, or `p_out` has no
    variance (a degenerate/constant predictor the logistic fit can't use).
    """
    y_arr = np.asarray(y_true)
    p_arr = np.asarray(p_out, dtype=float)
    if len(y_arr) < min_samples or len(np.unique(y_arr)) < 2:
        return None, None

    eps = 1e-6
    p_clip = np.clip(p_arr, eps, 1 - eps)
    logit = np.log(p_clip / (1 - p_clip))
    if np.allclose(logit, logit[0]):
        return None, None

    try:
        model = LogisticRegression()
        model.fit(logit.reshape(-1, 1), y_arr)
    except ValueError:
        return None, None
    return float(model.intercept_[0]), float(model.coef_[0][0])


# ---------------------------------------------------------------------------
# Game-clustered bootstrap: subgroup's own metrics + paired vs. baseline
# ---------------------------------------------------------------------------


def _safe_log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-15
    p_clip = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p_clip) + (1 - y_true) * np.log(1 - p_clip)))


def _safe_ece(y_true: np.ndarray, p: np.ndarray) -> float:
    table = compute_adaptive_calibration_table(y_true, pd.Series(p))
    return compute_binary_ece(table) if not table.empty else float("nan")


def compute_subgroup_bootstrap(
    y_true: np.ndarray,
    specialist_p: np.ndarray,
    baseline_p: np.ndarray,
    game_pks: np.ndarray,
    *,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci: float = DEFAULT_BOOTSTRAP_CI,
) -> dict[str, dict[str, Any]]:
    """Game_pk-clustered bootstrap, restricted to the GAMES PRESENT IN THIS
    SUBGROUP ONLY -- both each model's own absolute log-loss/ECE and the
    PAIRED specialist-minus-baseline delta, in one resampling loop (matching
    `mlb_luck_score.models.compare_near_wall_models.
    compute_near_wall_paired_bootstrap`'s existing resampling pattern, applied
    per-subgroup instead of to the whole near-wall population).

    Requires at least 2 distinct games -- with fewer, resampling with
    replacement cannot produce a non-degenerate distribution; callers should
    check `n_unique_games >= MIN_SUBGROUP_GAMES` (via `has_adequate_support`)
    before relying on these CIs.
    """
    unique_games = np.unique(game_pks)
    game_to_rows = {g: np.where(game_pks == g)[0] for g in unique_games}
    n_games = len(unique_games)

    rng = np.random.default_rng(seed)
    specialist_ll = np.empty(n_reps)
    specialist_ece = np.empty(n_reps)
    baseline_ll = np.empty(n_reps)
    baseline_ece = np.empty(n_reps)

    for rep in range(n_reps):
        sampled_games = rng.choice(unique_games, size=n_games, replace=True)
        rows = np.concatenate([game_to_rows[g] for g in sampled_games])
        y_rep = y_true[rows]
        specialist_ll[rep] = _safe_log_loss(y_rep, specialist_p[rows])
        specialist_ece[rep] = _safe_ece(y_rep, specialist_p[rows])
        baseline_ll[rep] = _safe_log_loss(y_rep, baseline_p[rows])
        baseline_ece[rep] = _safe_ece(y_rep, baseline_p[rows])

    alpha = (1.0 - ci) / 2.0

    def _summary(point: float, arr: np.ndarray) -> dict[str, Any]:
        return {
            "point_estimate": point,
            "ci_low": float(np.nanquantile(arr, alpha)),
            "ci_high": float(np.nanquantile(arr, 1.0 - alpha)),
            "n_reps": n_reps,
            "seed": seed,
            "resampling_unit": "game_pk",
            "n_unique_games": int(n_games),
        }

    point_specialist_ll = _safe_log_loss(y_true, specialist_p)
    point_specialist_ece = _safe_ece(y_true, specialist_p)
    point_baseline_ll = _safe_log_loss(y_true, baseline_p)
    point_baseline_ece = _safe_ece(y_true, baseline_p)

    return {
        "specialist_log_loss": _summary(point_specialist_ll, specialist_ll),
        "specialist_ece": _summary(point_specialist_ece, specialist_ece),
        "baseline_log_loss": _summary(point_baseline_ll, baseline_ll),
        "baseline_ece": _summary(point_baseline_ece, baseline_ece),
        "paired_log_loss_delta": _summary(
            point_specialist_ll - point_baseline_ll, specialist_ll - baseline_ll
        ),
        "paired_ece_delta": _summary(
            point_specialist_ece - point_baseline_ece, specialist_ece - baseline_ece
        ),
    }


# ---------------------------------------------------------------------------
# Three-way status classification
# ---------------------------------------------------------------------------


def has_adequate_support(n_plays: int, n_games: int, n_positive: int, n_negative: int) -> bool:
    """Minimum raw outcome support before a subgroup's calibration is evaluated
    for pass/fail at all -- see module-level threshold constants.
    """
    return (
        n_plays >= MIN_SUBGROUP_PLAYS
        and n_games >= MIN_SUBGROUP_GAMES
        and min(n_positive, n_negative) >= MIN_CLASS_SUPPORT
    )


def classify_subgroup_status(
    *,
    has_adequate_support: bool,
    specialist_ece_ci_low: float | None,
    specialist_ece_ci_high: float | None,
    paired_log_loss_delta_ci_low: float | None,
    ece_threshold: float = _MATERIAL_ECE_ABSOLUTE_THRESHOLD,
) -> tuple[str, str]:
    """The core Version 0.7D decision rule -- see module docstring for the full
    rationale. Returns `(status, reason)`.

    `not_calibrated` requires CREDIBLE evidence (an entire confidence interval
    on the wrong side of a line), never a bare point estimate. `insufficient_
    evidence` covers both inadequate raw support AND a CI that straddles the
    threshold (support was nominally enough to compute a number, but not
    enough to trust which side of the threshold it's really on).
    """
    if not has_adequate_support:
        return (
            SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE,
            "raw outcome support (plays/games/minority-class count) below the documented minimum",
        )
    if specialist_ece_ci_low is None or specialist_ece_ci_high is None:
        return (
            SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE,
            "bootstrap confidence interval could not be computed",
        )
    if paired_log_loss_delta_ci_low is not None and paired_log_loss_delta_ci_low > 0.0:
        return (
            SUBGROUP_STATUS_NOT_CALIBRATED,
            "paired bootstrap CI for specialist-minus-baseline log loss is entirely above zero "
            "-- credible regression vs. open_field_v07 on the identical rows",
        )
    if specialist_ece_ci_low > ece_threshold:
        return (
            SUBGROUP_STATUS_NOT_CALIBRATED,
            f"ECE confidence interval [{specialist_ece_ci_low:.4f}, {specialist_ece_ci_high:.4f}] "
            f"is entirely above the material threshold ({ece_threshold})",
        )
    if specialist_ece_ci_high <= ece_threshold:
        return (
            SUBGROUP_STATUS_CALIBRATED,
            f"ECE confidence interval [{specialist_ece_ci_low:.4f}, {specialist_ece_ci_high:.4f}] "
            f"is entirely at or below the material threshold ({ece_threshold}), and no credible "
            "paired regression vs. open_field_v07",
        )
    return (
        SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE,
        f"ECE confidence interval [{specialist_ece_ci_low:.4f}, {specialist_ece_ci_high:.4f}] "
        f"straddles the material threshold ({ece_threshold}) -- inconclusive at this sample size",
    )


# ---------------------------------------------------------------------------
# Per-subgroup evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubgroupCalibrationEvidence:
    """Everything the Version 0.7D gate reports for one subgroup or venue.

    Point-estimate metrics (`specialist_log_loss` etc.) are ALWAYS computed
    and reported, even when support is inadequate -- transparency first, per
    "reporting must remain honest". Only the CIs/status computation is
    withheld when support is inadequate (see `evaluate_subgroup_calibration`).
    """

    label: str
    n_plays: int
    n_games: int
    n_positive: int
    n_negative: int
    has_adequate_support: bool
    specialist_log_loss: float | None
    specialist_brier_score: float | None
    specialist_adaptive_ece: float | None
    specialist_calibration_intercept: float | None
    specialist_calibration_slope: float | None
    specialist_ece_ci_low: float | None
    specialist_ece_ci_high: float | None
    baseline_log_loss: float | None
    baseline_brier_score: float | None
    baseline_adaptive_ece: float | None
    paired_log_loss_delta: float | None
    paired_log_loss_delta_ci_low: float | None
    paired_log_loss_delta_ci_high: float | None
    paired_ece_delta: float | None
    paired_ece_delta_ci_low: float | None
    paired_ece_delta_ci_high: float | None
    status: str
    status_reason: str


def evaluate_subgroup_calibration(
    label: str,
    mask: np.ndarray,
    y_true: np.ndarray,
    specialist_p_out: pd.Series,
    baseline_p_out: pd.Series,
    game_pks: pd.Series,
    *,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci: float = DEFAULT_BOOTSTRAP_CI,
) -> SubgroupCalibrationEvidence:
    """Full Version 0.7D evidence + status for one subgroup/venue mask."""
    n_plays = int(mask.sum())
    y_sub = y_true[mask]
    n_positive = int(y_sub.sum()) if n_plays else 0
    n_negative = n_plays - n_positive
    game_arr = game_pks.to_numpy()[mask]
    n_games = int(len(np.unique(game_arr))) if n_plays else 0
    adequate = has_adequate_support(n_plays, n_games, n_positive, n_negative)

    if n_plays == 0:
        return SubgroupCalibrationEvidence(
            label=label,
            n_plays=0,
            n_games=0,
            n_positive=0,
            n_negative=0,
            has_adequate_support=False,
            specialist_log_loss=None,
            specialist_brier_score=None,
            specialist_adaptive_ece=None,
            specialist_calibration_intercept=None,
            specialist_calibration_slope=None,
            specialist_ece_ci_low=None,
            specialist_ece_ci_high=None,
            baseline_log_loss=None,
            baseline_brier_score=None,
            baseline_adaptive_ece=None,
            paired_log_loss_delta=None,
            paired_log_loss_delta_ci_low=None,
            paired_log_loss_delta_ci_high=None,
            paired_ece_delta=None,
            paired_ece_delta_ci_low=None,
            paired_ece_delta_ci_high=None,
            status=SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE,
            status_reason="no rows in this subgroup",
        )

    specialist_sub = specialist_p_out.to_numpy()[mask]
    baseline_sub = baseline_p_out.to_numpy()[mask]

    specialist_ll = float(log_loss(y_sub, specialist_sub, labels=[0, 1]))
    specialist_brier = float(brier_score_loss(y_sub, specialist_sub))
    specialist_table = compute_adaptive_calibration_table(y_sub, pd.Series(specialist_sub))
    specialist_ece = compute_binary_ece(specialist_table) if not specialist_table.empty else None
    specialist_intercept, specialist_slope = compute_calibration_intercept_slope(
        y_sub, specialist_sub
    )

    baseline_ll = float(log_loss(y_sub, baseline_sub, labels=[0, 1]))
    baseline_brier = float(brier_score_loss(y_sub, baseline_sub))
    baseline_table = compute_adaptive_calibration_table(y_sub, pd.Series(baseline_sub))
    baseline_ece = compute_binary_ece(baseline_table) if not baseline_table.empty else None

    ece_ci_low = ece_ci_high = None
    paired_ll_delta = paired_ll_ci_low = paired_ll_ci_high = None
    paired_ece_delta = paired_ece_ci_low = paired_ece_ci_high = None
    bootstrap_ran = adequate and n_games >= 2
    if bootstrap_ran:
        bootstrap = compute_subgroup_bootstrap(
            y_sub, specialist_sub, baseline_sub, game_arr, n_reps=n_reps, seed=seed, ci=ci
        )
        ece_ci_low = bootstrap["specialist_ece"]["ci_low"]
        ece_ci_high = bootstrap["specialist_ece"]["ci_high"]
        paired_ll_delta = bootstrap["paired_log_loss_delta"]["point_estimate"]
        paired_ll_ci_low = bootstrap["paired_log_loss_delta"]["ci_low"]
        paired_ll_ci_high = bootstrap["paired_log_loss_delta"]["ci_high"]
        paired_ece_delta = bootstrap["paired_ece_delta"]["point_estimate"]
        paired_ece_ci_low = bootstrap["paired_ece_delta"]["ci_low"]
        paired_ece_ci_high = bootstrap["paired_ece_delta"]["ci_high"]

    status, reason = classify_subgroup_status(
        has_adequate_support=bootstrap_ran,
        specialist_ece_ci_low=ece_ci_low,
        specialist_ece_ci_high=ece_ci_high,
        paired_log_loss_delta_ci_low=paired_ll_ci_low,
    )

    return SubgroupCalibrationEvidence(
        label=label,
        n_plays=n_plays,
        n_games=n_games,
        n_positive=n_positive,
        n_negative=n_negative,
        has_adequate_support=adequate,
        specialist_log_loss=specialist_ll,
        specialist_brier_score=specialist_brier,
        specialist_adaptive_ece=specialist_ece,
        specialist_calibration_intercept=specialist_intercept,
        specialist_calibration_slope=specialist_slope,
        specialist_ece_ci_low=ece_ci_low,
        specialist_ece_ci_high=ece_ci_high,
        baseline_log_loss=baseline_ll,
        baseline_brier_score=baseline_brier,
        baseline_adaptive_ece=baseline_ece,
        paired_log_loss_delta=paired_ll_delta,
        paired_log_loss_delta_ci_low=paired_ll_ci_low,
        paired_log_loss_delta_ci_high=paired_ll_ci_high,
        paired_ece_delta=paired_ece_delta,
        paired_ece_delta_ci_low=paired_ece_ci_low,
        paired_ece_delta_ci_high=paired_ece_ci_high,
        status=status,
        status_reason=reason,
    )


# ---------------------------------------------------------------------------
# Required subgroup masks (task item 8): wall-distance bands, wall-height
# bands, spray sectors, opportunity-time groups. Reuses the SAME mask
# definitions already established in `compare_near_wall_models`/
# `compare_opportunity_models` -- eligibility/subgroup definitions stay
# centralized there, not duplicated here.
# ---------------------------------------------------------------------------


def build_required_subgroup_masks(final_df: pd.DataFrame) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}

    for col in ("near_wall_5ft", "near_wall_10ft", "near_wall_20ft"):
        if col in final_df.columns:
            masks[col] = _bool_mask(final_df[col])

    if "wall_height_in_spray_direction" in final_df.columns:
        buckets = compute_wall_height_buckets(final_df["wall_height_in_spray_direction"])
        for bucket in ("short", "medium", "tall"):
            masks[f"wall_height_{bucket}"] = (buckets == bucket).to_numpy()

    if "spray_sector" in final_df.columns:
        for sector in ("left", "left_center", "center", "right_center", "right"):
            masks[f"spray_sector_{sector}"] = (final_df["spray_sector"] == sector).to_numpy()

    if "estimated_hang_time_s" in final_df.columns:
        buckets = compute_opportunity_time_buckets(final_df["estimated_hang_time_s"])
        for bucket in ("q1_shortest", "q2", "q3", "q4_longest"):
            masks[f"opportunity_time_{bucket}"] = (buckets == bucket).to_numpy()

    return masks


def build_venue_masks(venue_ids: pd.Series) -> dict[str, np.ndarray]:
    """One mask per distinct venue (missing venue grouped into one bucket) --
    "reliably sampled" is now determined by `has_adequate_support`/status,
    not a separate, redundant venue-specific sample-size convention.
    """
    grouped = venue_ids.fillna(_MISSING_VENUE_LABEL).to_numpy()
    return {f"venue_{v}": (grouped == v) for v in pd.unique(grouped)}


def evaluate_all_subgroups(
    final_df: pd.DataFrame,
    y_true: np.ndarray,
    specialist_p_out: pd.Series,
    baseline_p_out: pd.Series,
    *,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci: float = DEFAULT_BOOTSTRAP_CI,
) -> list[SubgroupCalibrationEvidence]:
    masks = build_required_subgroup_masks(final_df)
    return [
        evaluate_subgroup_calibration(
            label,
            mask,
            y_true,
            specialist_p_out,
            baseline_p_out,
            final_df["game_pk"],
            n_reps=n_reps,
            seed=seed,
            ci=ci,
        )
        for label, mask in masks.items()
    ]


def evaluate_all_venues(
    final_df: pd.DataFrame,
    y_true: np.ndarray,
    specialist_p_out: pd.Series,
    baseline_p_out: pd.Series,
    *,
    n_reps: int = DEFAULT_N_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci: float = DEFAULT_BOOTSTRAP_CI,
) -> list[SubgroupCalibrationEvidence]:
    if "venue_id" not in final_df.columns:
        return []
    masks = build_venue_masks(final_df["venue_id"])
    return [
        evaluate_subgroup_calibration(
            label,
            mask,
            y_true,
            specialist_p_out,
            baseline_p_out,
            final_df["game_pk"],
            n_reps=n_reps,
            seed=seed,
            ci=ci,
        )
        for label, mask in masks.items()
    ]


# ---------------------------------------------------------------------------
# Overall gate
# ---------------------------------------------------------------------------


def summarize_calibration_gate(
    comparison: dict[str, dict[str, Any]],
    aggregate_bootstrap: dict[str, dict[str, Any]],
    subgroup_evidence: list[SubgroupCalibrationEvidence],
    venue_evidence: list[SubgroupCalibrationEvidence],
    perturbation_suite: NearWallPerturbationSuite,
    open_field_regression_check: dict[str, Any],
) -> dict[str, Any]:
    """Recompute `near_wall_specialist_calibrated` (task item 9) with the new
    sample-size-aware subgroup/venue evaluation in place of the OLD fixed-ECE
    -threshold `find_material_subgroup_issues` rule -- see module docstring.
    """
    all_evidence = subgroup_evidence + venue_evidence
    not_calibrated = [e.label for e in all_evidence if e.status == SUBGROUP_STATUS_NOT_CALIBRATED]
    insufficient = [
        e.label for e in all_evidence if e.status == SUBGROUP_STATUS_INSUFFICIENT_EVIDENCE
    ]
    calibrated = [e.label for e in all_evidence if e.status == SUBGROUP_STATUS_CALIBRATED]

    perturbation_failures = [
        name for name, r in perturbation_suite.required.items() if not r.passed
    ]
    perturbation_checks_passed = bool(perturbation_suite.required) and not perturbation_failures
    architecture_ok = bool(open_field_regression_check.get("predictions_identical_via_gate", False))

    specialist = comparison[VARIANT_NEAR_WALL_FINAL]
    baseline = comparison[VARIANT_OPEN_FIELD]
    improves_log_loss = specialist["binary_log_loss"] < baseline["binary_log_loss"]
    bootstrap_supports = (
        aggregate_bootstrap.get("log_loss_delta", {}).get("ci_high", float("inf")) <= 0.0
    )

    no_credible_subgroup_failures = not not_calibrated
    baseline_requirements_met = (
        perturbation_checks_passed
        and architecture_ok
        and improves_log_loss
        and bootstrap_supports
        and no_credible_subgroup_failures
    )

    if not baseline_requirements_met:
        overall_status = OVERALL_STATUS_NOT_CALIBRATED
    elif insufficient:
        overall_status = OVERALL_STATUS_CALIBRATED_LIMITED_EVIDENCE
    else:
        overall_status = OVERALL_STATUS_CALIBRATED

    near_wall_specialist_calibrated = overall_status == OVERALL_STATUS_CALIBRATED

    return {
        "improves_log_loss_vs_open_field": improves_log_loss,
        "log_loss_delta": specialist["binary_log_loss"] - baseline["binary_log_loss"],
        "bootstrap_supports_improvement": bootstrap_supports,
        "perturbation_checks_passed": perturbation_checks_passed,
        "perturbation_failures": perturbation_failures,
        "open_field_regression_check": open_field_regression_check,
        "subgroup_evidence": [asdict(e) for e in subgroup_evidence],
        "venue_evidence": [asdict(e) for e in venue_evidence],
        "not_calibrated_groups": not_calibrated,
        "insufficient_evidence_groups": insufficient,
        "calibrated_groups": calibrated,
        "overall_status": overall_status,
        "near_wall_specialist_calibrated": near_wall_specialist_calibrated,
        "reporting_rule": (
            "Outfield execution score available for calibrated open-field opportunities; "
            "provisional or unavailable for wall-adjacent opportunities."
        ),
        "measured_miscalibration_vs_insufficient_evidence_note": (
            "'not_calibrated' means CREDIBLE evidence of a problem (a confidence interval "
            "entirely on the wrong side of a line); 'insufficient_evidence' means the data "
            "cannot yet tell -- it is reported honestly and separately, never treated as a "
            "pass or a fail. See CLAUDE.md/AGENTS.md 'Measured miscalibration vs. insufficient "
            "evidence'."
        ),
    }


# ---------------------------------------------------------------------------
# Reliability plots (task item 8)
# ---------------------------------------------------------------------------


def plot_subgroup_reliability_diagrams(
    y_true: np.ndarray,
    specialist_p_out: pd.Series,
    subgroup_masks: dict[str, np.ndarray],
    output_dir: Path,
) -> list[Path]:
    """One reliability-diagram PNG per subgroup/venue mask, using the SAME
    adaptive-bin calibration table the gate itself is evaluated on. Requires
    matplotlib (a dev dependency, already used by `mlb_luck_score.models.
    calibrate_model.plot_calibration_curves`).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    specialist_arr = specialist_p_out.to_numpy()
    written: list[Path] = []
    for label, mask in subgroup_masks.items():
        n = int(mask.sum())
        if n == 0:
            continue
        table = compute_adaptive_calibration_table(y_true[mask], pd.Series(specialist_arr[mask]))
        if table.empty:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
        ax.scatter(
            table["mean_predicted_probability"],
            table["observed_frequency"],
            s=np.clip(table["sample_count"], 10, 200),
            alpha=0.8,
        )
        ax.set_xlabel("Mean predicted P(out)")
        ax.set_ylabel("Observed out rate")
        ax.set_title(f"{label} (n={n}, adaptive bins={len(table)})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        path = output_dir / f"reliability_{label}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _train_open_field_baseline(df: pd.DataFrame) -> TrainedOpportunityModel:
    """EXACT same training path as `compare_near_wall_models.main` -- `open_field_v07`
    fit on all outfield-opportunity-eligible `TRAIN_SEASONS` rows (2021-2023),
    INCLUDING near-wall ones, unchanged and not retrained differently here.
    """
    prepared = _prepare_opportunity_columns(df)
    prepared = _prepare_venue_column(prepared) if "venue_id" in prepared.columns else prepared
    eligible = prepared[prepared["outfield_opportunity_eligible"].astype(bool)]
    train_rows = eligible[eligible["season"].isin(TRAIN_SEASONS)]
    return train_opportunity_model(train_rows, class_weight=None)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Cleaned development data joined with venue metadata AND park geometry",
    )
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--n-bootstrap-reps", type=int, default=DEFAULT_N_BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    assert_seasons_allowed(
        CALIBRATION_BASE_TRAIN_SEASONS + CALIBRATION_FIT_SEASONS + CALIBRATION_EVAL_SEASONS
    )

    df = pd.read_parquet(args.input) if args.input.suffix != ".csv" else pd.read_csv(args.input)

    try:
        near_wall_df = get_near_wall_rows(df)
    except ValueError as exc:
        logger.error(str(exc))
        return 2
    logger.info("Near-wall-gated rows: %d", len(near_wall_df))

    winner_variant, selection_metrics, trained_by_candidate = run_near_wall_model_selection(
        near_wall_df
    )
    winner_trained = trained_by_candidate[winner_variant]
    open_field_trained = _train_open_field_baseline(df)

    comparison = run_near_wall_final_comparison(
        near_wall_df, open_field_trained, winner_variant, winner_trained
    )

    final_df = near_wall_df[near_wall_df["season"].isin(CALIBRATION_EVAL_SEASONS)]
    y_true = final_df[OUTFIELD_OPPORTUNITY_TARGET_COLUMN].astype(int).to_numpy()
    open_field_feature_cols = (
        open_field_trained.numeric_features + open_field_trained.categorical_features
    )
    winner_feature_cols = winner_trained.numeric_features + winner_trained.categorical_features
    baseline_p_out = predict_opportunity_proba(
        open_field_trained, final_df[open_field_feature_cols]
    )
    specialist_p_out = predict_opportunity_proba(winner_trained, final_df[winner_feature_cols])
    validate_opportunity_probabilities(baseline_p_out)
    validate_opportunity_probabilities(specialist_p_out)

    aggregate_bootstrap = compute_near_wall_paired_bootstrap(
        y_true,
        baseline_p_out,
        specialist_p_out,
        final_df["game_pk"],
        n_reps=args.n_bootstrap_reps,
        seed=args.bootstrap_seed,
    )
    perturbation_suite = run_near_wall_perturbation_checks(winner_trained, final_df)
    open_field_regression_check = check_no_open_field_regression(open_field_trained, df)

    subgroup_evidence = evaluate_all_subgroups(
        final_df,
        y_true,
        specialist_p_out,
        baseline_p_out,
        n_reps=args.n_bootstrap_reps,
        seed=args.bootstrap_seed,
    )
    venue_evidence = evaluate_all_venues(
        final_df,
        y_true,
        specialist_p_out,
        baseline_p_out,
        n_reps=args.n_bootstrap_reps,
        seed=args.bootstrap_seed,
    )

    gate_summary = summarize_calibration_gate(
        comparison,
        aggregate_bootstrap,
        subgroup_evidence,
        venue_evidence,
        perturbation_suite,
        open_field_regression_check,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "near_wall_calibration_gate_detail.json"
    detail_path.write_text(
        json.dumps(
            {
                "winner_variant": winner_variant,
                "selection_metrics": selection_metrics,
                "comparison": comparison,
                "aggregate_bootstrap": aggregate_bootstrap,
                "gate_summary": gate_summary,
            },
            indent=2,
            default=str,
        )
    )

    if args.figures_dir is not None:
        masks = {
            **build_required_subgroup_masks(final_df),
            **(build_venue_masks(final_df["venue_id"]) if "venue_id" in final_df.columns else {}),
        }
        written = plot_subgroup_reliability_diagrams(
            y_true, specialist_p_out, masks, args.figures_dir
        )
        logger.info("Wrote %d reliability diagrams to %s", len(written), args.figures_dir)

    logger.info("overall_status=%s", gate_summary["overall_status"])
    logger.info(
        "near_wall_specialist_calibrated=%s", gate_summary["near_wall_specialist_calibrated"]
    )
    logger.info(
        "calibrated=%d not_calibrated=%d insufficient_evidence=%d",
        len(gate_summary["calibrated_groups"]),
        len(gate_summary["not_calibrated_groups"]),
        len(gate_summary["insufficient_evidence_groups"]),
    )
    logger.info("Saved detail to %s", detail_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
