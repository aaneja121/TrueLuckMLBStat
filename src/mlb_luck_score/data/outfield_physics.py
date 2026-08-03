"""Contact Luck v0.7A: documented physics/geometry ESTIMATES for outfield opportunity difficulty.

This module is the ONE place that defines two ESTIMATED (not measured)
quantities used by the outfield-opportunity model:

  1. `estimate_hang_time_seconds`: a vacuum (no-drag) projectile-motion
     estimate of how long a batted ball stays airborne, from `launch_speed`
     and `launch_angle` alone.
  2. `estimate_landing_coordinates_ft`: a Cartesian (x, y) landing-point
     estimate from `hit_distance_sc` and `spray_angle_approx`, in the SAME
     polar convention already used everywhere else in this codebase (see
     `mlb_luck_score.data.park_geometry`'s `STANDARD_ANGLES` and
     `mlb_luck_score.data.weather_physics`'s spray-angle/wind-bearing
     convention note): 0 degrees is straightaway center field, negative is
     the third-base/left-field side, positive is the first-base/right-field
     side.

IMPORTANT -- these are documented approximations, not measured Statcast
fields, and (unlike `weather_physics.moist_air_density_kg_m3`, which can be
checked against known physical constants like sea-level ISA density) there
is NO public ground-truth "hang time" or "landing coordinate" field to
validate either estimate against -- Baseball Savant's own Catch Probability
methodology, which presumably uses real tracked hang time and landing
location, is not publicly exposed at the per-play level (see
`mlb_luck_score.models.compare_opportunity_models` module docstring for the
full audit). Treat both as directional, physically-motivated signals for
opportunity-difficulty MODELING, never as measured quantities, and never
describe them as "hang time" or "landing location" without the word
"estimated" attached in any downstream reporting.

The hang-time formula ignores air resistance entirely (real MLB fly balls
experience meaningful drag, which shortens both hang time and distance
relative to the vacuum estimate for hard-hit balls) and assumes the ball
lands at the same height it was struck (ignoring the ~2-3 ft contact height
and any ground/wall interaction) -- both are known, documented
simplifications, not oversights.

Nothing in this module reads or writes real data -- it is pure functions,
directly unit-testable without any network access or real Statcast data.
"""

from __future__ import annotations

import math

from mlb_luck_score.data.weather_physics import MPH_TO_MPS, STANDARD_GRAVITY_M_S2

#: Minimum returned hang time -- the vacuum projectile formula is undefined
#: (or negative) for a launch_angle <= 0, which some real line drives have.
#: A ball with a non-positive launch angle still spends SOME real time in
#: the air before being fielded; rather than returning a negative or zero
#: value that would be nonsensical as a feature, clip to this small floor
#: and treat it as "the formula breaks down for very low trajectories,"
#: documented here rather than silently producing a physically impossible
#: negative duration.
MIN_HANG_TIME_SECONDS = 0.1


def estimate_hang_time_seconds(launch_speed_mph: float, launch_angle_deg: float) -> float:
    """Vacuum (no-drag) projectile-motion estimate of a batted ball's time airborne.

    `t = 2 * v0 * sin(theta) / g`, where `v0` is exit velocity in m/s and
    `theta` is launch angle in radians -- the standard "time of flight"
    formula for a projectile launched and landing at the same height,
    ignoring air resistance. See module docstring for why this is a
    documented approximation with no public ground truth to validate
    against, not a measured quantity.

    Args:
        launch_speed_mph: Exit velocity in mph (Statcast `launch_speed`).
        launch_angle_deg: Launch angle in degrees (Statcast `launch_angle`).

    Returns:
        Estimated hang time in seconds, floored at `MIN_HANG_TIME_SECONDS`
        for non-positive launch angles (see that constant's docstring).
        `NaN` if either input is `NaN`.
    """
    if math.isnan(launch_speed_mph) or math.isnan(launch_angle_deg):
        return math.nan

    v0_mps = launch_speed_mph * MPH_TO_MPS
    theta_rad = math.radians(launch_angle_deg)
    hang_time = 2.0 * v0_mps * math.sin(theta_rad) / STANDARD_GRAVITY_M_S2
    return max(hang_time, MIN_HANG_TIME_SECONDS)


#: Exact unit conversion (not an estimate, unlike `MIN_HANG_TIME_SECONDS`) --
#: used only to convert a feet-denominated distance into the meters the
#: vacuum projectile formulas below are stated in.
FEET_TO_METERS = 0.3048


def solve_launch_speed_for_matched_range_mph(
    target_distance_ft: float, launch_angle_deg: float
) -> float:
    """Exit velocity (mph) that reaches `target_distance_ft` at `launch_angle_deg`,
    under the SAME vacuum (no-drag) projectile model as `estimate_hang_time_seconds`.

    This is the companion "range" formula to `estimate_hang_time_seconds`'s
    "time of flight" formula -- both come from the same vacuum projectile
    model launched and landing at the same height: `R = v0^2 * sin(2*theta) /
    g`, solved here for `v0` given a target `R` and `theta`.

    Used to build a TRAJECTORY-MATCHED counterfactual row: same landing
    distance (and therefore the same spray-direction wall geometry) as a
    real row, but a different launch angle, with exit velocity solved so the
    ball still reaches that same distance -- see `mlb_luck_score.models.
    compare_near_wall_models`'s trajectory-matched opportunity-time check for
    why this is needed (a naive launch-angle override that leaves exit
    velocity untouched silently changes the landing point too, entangling
    "launch angle" with "how far this ball happened to travel").

    Args:
        target_distance_ft: Desired projected/observed distance in feet.
        launch_angle_deg: Launch angle in degrees to solve exit velocity for.

    Returns:
        Exit velocity in mph. `NaN` if either input is `NaN`, or if
        `launch_angle_deg` is at or outside `(0, 90)` where the vacuum range
        formula is undefined or non-physical (`sin(2*theta) <= 0`).
    """
    if math.isnan(target_distance_ft) or math.isnan(launch_angle_deg):
        return math.nan

    theta_rad = math.radians(launch_angle_deg)
    sin_2theta = math.sin(2.0 * theta_rad)
    # Floating-point `sin` at exactly 0/90 degrees lands on a tiny nonzero
    # value (e.g. ~1.2e-16 at 90 deg) rather than exactly 0, so a strict
    # `<= 0` guard would let a degenerate near-0/90-degree angle through as
    # a spuriously huge finite speed instead of `NaN`.
    if sin_2theta <= 1e-9:
        return math.nan

    distance_m = target_distance_ft * FEET_TO_METERS
    v0_mps = math.sqrt(distance_m * STANDARD_GRAVITY_M_S2 / sin_2theta)
    return v0_mps / MPH_TO_MPS


def estimate_landing_coordinates_ft(
    hit_distance_ft: float, spray_angle_deg: float
) -> tuple[float, float]:
    """Estimated (x, y) landing point in feet from home plate, in a fixed field frame.

    `x = distance * sin(spray_angle)`: positive = first-base/right-field
    side, negative = third-base/left-field side.
    `y = distance * cos(spray_angle)`: positive = toward center field/the
    outfield wall, away from home plate.

    Matches the SAME polar convention as `mlb_luck_score.data.park_geometry`
    (`STANDARD_ANGLES`) and `mlb_luck_score.data.weather_physics` (wind
    bearing) -- 0 degrees is straightaway center field.

    Args:
        hit_distance_ft: Projected/observed hit distance in feet (Statcast
            `hit_distance_sc`).
        spray_angle_deg: Approximate spray angle in degrees
            (`spray_angle_approx`).

    Returns:
        `(x_ft, y_ft)`. `(NaN, NaN)` if either input is `NaN`.
    """
    if math.isnan(hit_distance_ft) or math.isnan(spray_angle_deg):
        return math.nan, math.nan

    angle_rad = math.radians(spray_angle_deg)
    x_ft = hit_distance_ft * math.sin(angle_rad)
    y_ft = hit_distance_ft * math.cos(angle_rad)
    return x_ft, y_ft
