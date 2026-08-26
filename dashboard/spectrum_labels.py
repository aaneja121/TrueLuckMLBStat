"""Deterministic Luck Spectrum label selection (Version 1.4.2).

A Luck Spectrum can plot several players/plays that sit at nearly the same
value -- e.g. three showcase plays at +1.59 / +1.57 / +1.55. Labeling every
one of them produces overlapping or crowded text no runtime collision-avoidance
pass can fully fix. `select_spectrum_labels` instead decides, at the data
level, which points are worth a persistent visible label versus which stay
plotted as a small unlabeled dot (still inspectable via hover/focus in the
UI layer) -- see `dashboard/templates/_macros.html::luck_spectrum` (server-
rendered: homepage, player page) and `dashboard/static/explore.js`'s
`selectSpectrumLabels` (a deliberate line-for-line port for the client-
rendered Explore spectrum, matching this module's own docstring precedent
in `build.py`'s `_spectrum`/`visuals.spectrum_position_pct` for why the two
copies must stay in sync).

This module is presentation-only: it never reads, ranks, or filters real
Contact Luck scores itself -- callers pass in whatever values they already
have. It contains no scoring logic and imports nothing from `mlb_luck_score`
(enforced structurally for every `dashboard/*.py` module by
`tests/test_dashboard_isolation.py`).
"""

from __future__ import annotations

from collections.abc import Sequence

#: Default minimum separation between two labeled points on the SAME side of
#: zero, expressed as a fraction of that side's own half-range (the most
#: extreme value's own magnitude -- the distance from zero to the edge of
#: this side's data, NOT the padded chart domain). Chosen from the middle of
#: the task's requested 10-15% band.
DEFAULT_MIN_SEPARATION_FRACTION = 0.12


def select_spectrum_labels(
    values: Sequence[float],
    *,
    max_labels: int,
    min_separation_fraction: float = DEFAULT_MIN_SEPARATION_FRACTION,
) -> list[int | None]:
    """Choose which of `values` get a visible Luck Spectrum label.

    `values` must all be on the SAME side of zero (a side's favorable and
    unfavorable candidates are each passed in separately by the caller) --
    extremity is measured as `abs(value)`.

    Algorithm, applied independently per call (i.e. per side):
      1. Order candidates by descending extremity (most extreme first).
      2. Always label the single most extreme candidate (rank 0 -- the
         "primary" label).
      3. Walk the remaining candidates in the same order; each one gets the
         next label rank ONLY if its value differs from EVERY
         already-labeled value by at least `min_separation_fraction` of this
         side's half-range (`abs(values[order[0]])`, i.e. the most extreme
         value's own magnitude). Stops once `max_labels` labels are handed
         out.
      4. Everything else stays unlabeled.

    Returns a list parallel to `values` (same order, same length): each
    entry is the point's label rank (`0` = primary/most prominent, `1` =
    secondary, ...) if it should be labeled, or `None` if it should render
    as an unlabeled dot only. Purely a function of the data -- never depends
    on rendered pixel geometry, so it gives the identical result whether
    called at build time (Python) or render time (the JS port).

    Edge cases (see `tests/test_dashboard_spectrum_labels.py`):
      - Empty `values` or `max_labels <= 0`: nothing labeled.
      - A single candidate: always labeled (rank 0) -- there's nothing to
        separate it from.
      - Every value exactly 0 (half-range is 0, so no separation threshold
        is meaningful): only the first candidate is labeled, regardless of
        `max_labels` -- labeling several "+0.00" points would be noise, not
        information.
      - Tightly clustered values (e.g. +1.59, +1.57, +1.55) with a 12%
        threshold on a ~1.59 half-range (~0.19): only the most extreme value
        gets a label; the rest fall under the threshold.
    """
    n = len(values)
    result: list[int | None] = [None] * n
    if n == 0 or max_labels <= 0:
        return result

    order = sorted(range(n), key=lambda i: -abs(values[i]))
    half_range = abs(values[order[0]])
    if half_range <= 0:
        result[order[0]] = 0
        return result

    threshold = half_range * min_separation_fraction
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= max_labels:
            break
        if all(abs(values[i] - values[j]) >= threshold for j in chosen):
            chosen.append(i)

    for rank, i in enumerate(chosen):
        result[i] = rank
    return result


def label_tier(rank: int | None) -> str:
    """Maps a `select_spectrum_labels` rank to the three CSS-visible tiers
    `dashboard/templates/_macros.html::luck_spectrum` renders: `"primary"`
    (rank 0, the most prominent label), `"secondary"` (any other labeled
    rank -- this app never asks for more than 2 labels per side, so in
    practice this is only ever rank 1), or `"unlabeled"` (rank is `None`).
    """
    if rank is None:
        return "unlabeled"
    return "primary" if rank == 0 else "secondary"
