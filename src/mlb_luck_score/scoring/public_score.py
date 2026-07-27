"""Provisional -100..+100 public score mapping -- Version 0.1 placeholder.

IMPORTANT: This is explicitly a placeholder interface, not a validated
scientific mapping. The final public score should be learned empirically
from the distribution of raw-luck values observed across a large held-out
sample of historical plays (e.g. mapping raw luck to its percentile/tail
position in that distribution), which this repository does not yet do.

What this module guarantees for Version 0.1:
    - The mapping is monotonic non-decreasing in raw luck.
    - The sign of the input is preserved (positive raw luck -> non-negative
      score, negative raw luck -> non-positive score, zero -> zero).
    - Output is always clipped to [-100, 100] and finite.
    - Inputs are validated (must be finite).

What it does NOT guarantee:
    - That equal raw-luck values across different situations are "equally
      lucky" in a population sense -- that requires the empirical
      calibration described below.
    - Any particular real-world interpretation of a given score value
      (e.g. "a score of 50 means X% percentile") until that calibration
      exists.

The public score is deliberately NOT additive (do not sum public scores
across plays) -- see raw_luck.py for the additive quantity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Provisional scaling constant for the default tanh-based mapping. Chosen
#: only so that the Version 0.1 value map's largest plausible single-play
#: raw-luck swing (roughly a home run appearing near-certain-out, raw_luck
#: close to +4) maps to a score well away from the +-100 clip boundary, NOT
#: from any empirical distribution. Replace via `EmpiricalScoreCalibrator`
#: once real historical data supports it.
DEFAULT_SCALE = 2.0
SCORE_MIN = -100.0
SCORE_MAX = 100.0


class PublicScoreValidationError(ValueError):
    """Raised when raw_luck_to_public_score receives invalid input."""


def raw_luck_to_public_score(raw_luck: float, *, scale: float = DEFAULT_SCALE) -> float:
    """Map a raw-luck value to a provisional -100..+100 public display score.

    Uses `100 * tanh(raw_luck / scale)`: monotonic, sign-preserving, and
    naturally bounded in (-100, 100) (then explicitly clipped for safety).
    This specific functional form is a Version 0.1 convenience choice, not a
    scientifically validated transformation -- see module docstring.

    Args:
        raw_luck: Output of `mlb_luck_score.scoring.raw_luck.compute_raw_luck`.
        scale: Controls how quickly the score approaches +-100. Larger
            values compress the score toward the middle of the range.

    Returns:
        A float in [-100, 100].

    Raises:
        PublicScoreValidationError: If `raw_luck` is not finite or `scale`
            is not positive.
    """
    if not math.isfinite(raw_luck):
        raise PublicScoreValidationError(f"raw_luck must be finite, got {raw_luck!r}")
    if scale <= 0:
        raise PublicScoreValidationError(f"scale must be positive, got {scale!r}")

    score = SCORE_MAX * math.tanh(raw_luck / scale)
    return max(SCORE_MIN, min(SCORE_MAX, score))


@dataclass(frozen=True)
class EmpiricalScoreCalibrator:
    """Future extension point: an empirically-fit raw-luck -> score mapping.

    Version 0.1 does not populate this from data. The intended design is a
    monotonic mapping fit on the empirical distribution of raw-luck values
    from a large held-out historical sample (e.g. mapping to a signed
    percentile/tail position within that distribution), so a given score
    has a stable population-level interpretation. Once fit, an instance of
    this class would be used in place of `raw_luck_to_public_score`.

    Attributes:
        reference_quantiles: Sorted raw-luck quantile breakpoints from the
            historical reference distribution used to fit this calibrator.
        fitted_on_seasons: Which seasons' data the calibrator was fit on
            (must exclude the protected 2025 final-test season).
    """

    reference_quantiles: tuple[float, ...]
    fitted_on_seasons: tuple[int, ...]

    def score(self, raw_luck: float) -> float:
        raise NotImplementedError(
            "EmpiricalScoreCalibrator is a documented extension point for a "
            "future, data-fit public score mapping. It is not implemented in "
            "Version 0.1 -- use raw_luck_to_public_score() instead."
        )
