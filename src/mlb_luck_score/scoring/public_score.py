"""LEGACY (Version 0.1): tanh-based -100..+100 public score mapping.

> **Superseded by Version 0.2.** The default public score is now
> `mlb_luck_score.scoring.empirical_score.compute_empirical_public_score`,
> which maps raw contact luck to its SIGNED PERCENTILE within a genuine
> out-of-sample 2024 reference distribution instead of an arbitrary
> functional form. This module is kept ONLY for backward compatibility and
> explicit Version-0.1-vs-0.2 comparisons (see notebook
> `04_luck_score_demo.ipynb`'s "Legacy Version 0.1 comparison" section) --
> do not use `raw_luck_to_public_score` as the default for new work.

What this module guarantees for Version 0.1:
    - The mapping is monotonic non-decreasing in raw luck.
    - The sign of the input is preserved (positive raw luck -> non-negative
      score, negative raw luck -> non-positive score, zero -> zero).
    - Output is always clipped to [-100, 100] and finite.
    - Inputs are validated (must be finite).

What it does NOT guarantee:
    - That equal raw-luck values across different situations are "equally
      lucky" in a population sense -- that requires the empirical mapping
      in `mlb_luck_score.scoring.empirical_score`.
    - Any particular real-world interpretation of a given score value
      (e.g. "a score of 50 means X% percentile").

The public score is deliberately NOT additive (do not sum public scores
across plays) -- see `mlb_luck_score.scoring.aggregation` for the additive
Version 0.2 season-aggregation primitives (which operate on raw luck, never
on this score).
"""

from __future__ import annotations

import math

#: Provisional scaling constant for the legacy tanh-based mapping. Chosen
#: only so that the Version 0.1 ordinal value map's largest plausible
#: single-play raw-luck swing (roughly a home run appearing near-certain-out,
#: raw_luck close to +4) maps to a score well away from the +-100 clip
#: boundary, NOT from any empirical distribution. See
#: `mlb_luck_score.scoring.empirical_score` for the Version 0.2 replacement.
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
