"""Contact Luck v0.5: documented physics formulas for weather-aware features.

This module is the ONE place that defines:

  1. Unit conversions (Fahrenheit -> Celsius, mph -> m/s, inches-of-mercury
     altimeter setting -> hectopascals).
  2. A scientifically documented MOIST-AIR density calculation (ideal gas
     law applied separately to dry-air and water-vapor partial pressures --
     see `moist_air_density_kg_m3`), not a crude temperature-only proxy.
  3. The hypsometric equation for estimating station/sea-level-referenced
     pressure at a different elevation (see `pressure_at_elevation_hpa`).
  4. Parsing the MLB Stats API's park-relative wind-direction text (e.g.
     `"8 mph, Out To CF"`) into a `WindObservation`, and rotating that wind
     into following/headwind/crosswind components relative to a SPECIFIC
     batted ball's spray direction (see `wind_relative_components`).

Nothing in this module reads or writes real data -- it is pure functions
plus documented constants, so it's directly unit-testable without any
network access or real weather data (see `tests/test_weather_physics.py`).

## Spray-angle / wind-bearing convention

Matches `mlb_luck_score.data.park_geometry`'s `STANDARD_ANGLES` convention
exactly: 0 degrees is straightaway center field, negative is the third-base
/left-field side, positive is the first-base/right-field side. Wind
"movement bearing" (the direction the wind is BLOWING TOWARD, not the
meteorological "coming from" convention) uses the SAME scale: a movement
bearing of 0 points from home plate toward center field ("blowing out to
center"); 180 (or -180) points from center field back toward home plate
("blowing in from center"); +90 points toward the first-base/right-field
side (a pure "left to right" crosswind, as MLB's own text describes it);
-90 points toward the third-base/left-field side ("right to left").
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

MPH_TO_MPS = 0.44704
INHG_TO_HPA = 33.8639
KELVIN_OFFSET_C = 273.15

#: ICAO International Standard Atmosphere sea-level air density (dry air,
#: 15 deg C, 1013.25 hPa) -- used as a fixed, documented, data-independent
#: reference point for `air_density_deviation_from_reference` (see
#: `mlb_luck_score.data.join_weather_features`). Deliberately NOT derived
#: from this repository's own training data (which would make the
#: "reference" season-sample-dependent and could silently drift if the
#: training window ever changes) -- a standard physical constant instead.
DEFAULT_REFERENCE_AIR_DENSITY_KG_M3 = 1.225

#: Standard gravitational acceleration, m/s^2 (used by the hypsometric
#: equation below).
STANDARD_GRAVITY_M_S2 = 9.80665
#: Specific gas constant for dry air, J/(kg*K) = universal gas constant
#: (8.31446 J/(mol*K)) / molar mass of dry air (0.0289652 kg/mol).
SPECIFIC_GAS_CONSTANT_DRY_AIR = 287.058
#: Specific gas constant for water vapor, J/(kg*K) = universal gas constant
#: / molar mass of water (0.018016 kg/mol).
SPECIFIC_GAS_CONSTANT_WATER_VAPOR = 461.495


def fahrenheit_to_celsius(temp_f: float | None) -> float | None:
    if temp_f is None or (isinstance(temp_f, float) and math.isnan(temp_f)):
        return None
    return (temp_f - 32.0) * 5.0 / 9.0


def mph_to_mps(speed_mph: float | None) -> float | None:
    if speed_mph is None or (isinstance(speed_mph, float) and math.isnan(speed_mph)):
        return None
    return speed_mph * MPH_TO_MPS


def inhg_to_hpa(pressure_inhg: float | None) -> float | None:
    if pressure_inhg is None or (isinstance(pressure_inhg, float) and math.isnan(pressure_inhg)):
        return None
    return pressure_inhg * INHG_TO_HPA


# ---------------------------------------------------------------------------
# Moist-air density (documented formula, not a temperature-only proxy)
# ---------------------------------------------------------------------------


def saturation_vapor_pressure_hpa(temp_c: float) -> float:
    """Saturation vapor pressure of water at `temp_c`, in hectopascals.

    Arden Buck equation (Buck, 1996) over liquid water -- a standard,
    widely cited meteorological approximation, accurate to within ~0.2%
    over typical outdoor temperature ranges:

        es(T) = 6.1121 * exp((18.678 - T/234.5) * (T / (257.14 + T)))

    where T is in degrees Celsius and the result is in hPa (millibars).
    """
    return 6.1121 * math.exp((18.678 - temp_c / 234.5) * (temp_c / (257.14 + temp_c)))


def moist_air_density_kg_m3(
    temp_c: float | None, pressure_hpa: float | None, relative_humidity_pct: float | None
) -> float | None:
    """Moist-air density via the ideal gas law applied to dry-air + water-vapor partial pressures.

    Documented formula (standard meteorological/thermodynamic approximation,
    e.g. as presented in ASHRAE Fundamentals and CIPM-2007 moist-air
    reference material):

        e  = saturation_vapor_pressure_hpa(T) * (RH / 100)      -- actual vapor pressure (hPa)
        p_d = P - e                                              -- dry-air partial pressure (hPa)
        rho = (p_d * 100) / (R_d * T_K) + (e * 100) / (R_v * T_K)

    where P is total (station-level) pressure in hPa, T_K is temperature in
    Kelvin, R_d/R_v are the specific gas constants for dry air/water vapor
    (see module constants), and the `* 100` converts hPa to Pa for SI-unit
    consistency. Result is in kg/m^3.

    Args:
        temp_c: Air temperature, Celsius.
        pressure_hpa: Total (station/venue-level, NOT sea-level) air
            pressure, hectopascals -- see `pressure_at_elevation_hpa` to
            convert from a sea-level-referenced observation first.
        relative_humidity_pct: Relative humidity, 0-100.

    Returns:
        Air density in kg/m^3, or `None` if any input is missing/`NaN` --
        never silently substitutes a default (missing humidity/pressure
        means "cannot compute a real moist-air density", not "assume dry").

    Raises:
        ValueError: If `relative_humidity_pct` is outside [0, 100].
    """
    if (
        temp_c is None
        or pressure_hpa is None
        or relative_humidity_pct is None
        or (isinstance(temp_c, float) and math.isnan(temp_c))
        or (isinstance(pressure_hpa, float) and math.isnan(pressure_hpa))
        or (isinstance(relative_humidity_pct, float) and math.isnan(relative_humidity_pct))
    ):
        return None
    if not (0.0 <= relative_humidity_pct <= 100.0):
        raise ValueError(
            f"relative_humidity_pct must be in [0, 100], got {relative_humidity_pct!r}"
        )

    temp_k = temp_c + KELVIN_OFFSET_C
    vapor_pressure_hpa = saturation_vapor_pressure_hpa(temp_c) * (relative_humidity_pct / 100.0)
    dry_pressure_hpa = pressure_hpa - vapor_pressure_hpa

    dry_density = (dry_pressure_hpa * 100.0) / (SPECIFIC_GAS_CONSTANT_DRY_AIR * temp_k)
    vapor_density = (vapor_pressure_hpa * 100.0) / (SPECIFIC_GAS_CONSTANT_WATER_VAPOR * temp_k)
    return dry_density + vapor_density


def pressure_at_elevation_hpa(
    sea_level_pressure_hpa: float | None, elevation_m: float | None, temperature_c: float | None
) -> float | None:
    """Estimate station-level pressure at `elevation_m` from a sea-level-referenced observation.

    Hypsometric equation (a standard, documented meteorological formula for
    converting between sea-level and station-level pressure using the
    actual observed temperature, more accurate than the fixed-lapse-rate
    ISA barometric formula when a real temperature observation is
    available):

        P(z) = P0 * exp(-(g * z) / (R_d * T_K))

    where P0 is sea-level pressure (hPa), z is elevation (m), g is standard
    gravity, R_d is the specific gas constant for dry air, and T_K is the
    observed temperature in Kelvin (used as a simplifying stand-in for the
    mean temperature of the air column between sea level and the venue --
    a standard, documented approximation, not an exact integration).

    Returns:
        Estimated pressure at `elevation_m`, in hPa, or `None` if any input
        is missing/`NaN`.
    """
    if (
        sea_level_pressure_hpa is None
        or elevation_m is None
        or temperature_c is None
        or (isinstance(sea_level_pressure_hpa, float) and math.isnan(sea_level_pressure_hpa))
        or (isinstance(elevation_m, float) and math.isnan(elevation_m))
        or (isinstance(temperature_c, float) and math.isnan(temperature_c))
    ):
        return None
    temp_k = temperature_c + KELVIN_OFFSET_C
    exponent = -(STANDARD_GRAVITY_M_S2 * elevation_m) / (SPECIFIC_GAS_CONSTANT_DRY_AIR * temp_k)
    return sea_level_pressure_hpa * math.exp(exponent)


# ---------------------------------------------------------------------------
# Wind-text parsing and relative-wind decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindObservation:
    """A parsed MLB wind-condition string.

    Attributes:
        speed_mps: Wind speed, meters/second. 0.0 for "Calm"/"None".
        movement_bearing_degrees: Direction the wind is blowing TOWARD, in
            the spray-angle convention (see module docstring) -- `None` if
            the wind is calm/zero (direction is undefined) or the text
            could not be parsed.
        raw_text: The original, unparsed string, preserved for debugging.
        parse_status: `"ok"`, `"calm"`, or `"unparseable"`.
    """

    speed_mps: float
    movement_bearing_degrees: float | None
    raw_text: str
    parse_status: str


#: Movement bearing (degrees, spray-angle convention) for each MLB wind-text
#: category. "Out To X" bearings are the representative direction of travel
#: toward that field segment; "In From X" is the exact opposite bearing
#: (wind blowing from that segment back toward home plate). Pure crosswinds
#: ("L To R" / "R To L") are +-90 degrees exactly, matching MLB's own
#: broadcast-view left/right convention (see module docstring).
_OUT_BEARINGS: dict[str, float] = {"lf": -45.0, "cf": 0.0, "rf": 45.0}
_WIND_TEXT_RE = re.compile(r"^\s*([\d.]+)\s*mph\s*,\s*(.+?)\s*$", re.IGNORECASE)


def _opposite_bearing(bearing_degrees: float) -> float:
    """Bearing 180 degrees from `bearing_degrees`, normalized to (-180, 180]."""
    return ((bearing_degrees + 180.0 + 180.0) % 360.0) - 180.0


def parse_wind_text(wind_raw: str | None) -> WindObservation | None:
    """Parse an MLB Stats API wind string (e.g. `"8 mph, Out To CF"`).

    Returns `None` only if `wind_raw` itself is missing (`None`/empty) --
    an unparseable non-empty string still returns a `WindObservation` with
    `parse_status="unparseable"` and `movement_bearing_degrees=None` so the
    speed (if it was extractable) is not silently discarded.
    """
    if not wind_raw or not isinstance(wind_raw, str):
        return None

    match = _WIND_TEXT_RE.match(wind_raw)
    if not match:
        return WindObservation(0.0, None, wind_raw, "unparseable")

    speed_mph = float(match.group(1))
    speed_mps = speed_mph * MPH_TO_MPS
    direction_text = match.group(2).strip().lower()

    if direction_text in ("calm", "none", ""):
        return WindObservation(speed_mps, None, wind_raw, "calm" if speed_mps == 0.0 else "ok")

    out_match = re.match(r"^out to (lf|cf|rf)$", direction_text)
    if out_match:
        return WindObservation(speed_mps, _OUT_BEARINGS[out_match.group(1)], wind_raw, "ok")

    in_match = re.match(r"^in from (lf|cf|rf)$", direction_text)
    if in_match:
        bearing = _opposite_bearing(_OUT_BEARINGS[in_match.group(1)])
        return WindObservation(speed_mps, bearing, wind_raw, "ok")

    if direction_text == "l to r":
        return WindObservation(speed_mps, 90.0, wind_raw, "ok")
    if direction_text == "r to l":
        return WindObservation(speed_mps, -90.0, wind_raw, "ok")

    return WindObservation(speed_mps, None, wind_raw, "unparseable")


@dataclass(frozen=True)
class WindComponents:
    """Wind decomposed relative to one batted ball's spray direction.

    Attributes:
        following_wind_mps: Positive = blowing toward where the ball is
            headed (tailwind, helps carry); negative = blowing back toward
            home plate (headwind, hurts carry).
        headwind_mps: `-following_wind_mps` -- positive = true headwind
            (against the ball), negative = tailwind. Provided as its own
            field (in addition to `following_wind_mps`) because both are
            explicitly required, documented outputs -- they carry the same
            information with an inverted sign, not independently measured.
        crosswind_mps: Signed component perpendicular to the ball's path,
            using the same sign convention as spray angle (positive =
            blowing toward the first-base/right-field side, negative =
            toward the third-base/left-field side).
    """

    following_wind_mps: float
    headwind_mps: float
    crosswind_mps: float


def wind_relative_components(
    wind: WindObservation | None, spray_angle_degrees: float | None
) -> WindComponents | None:
    """Decompose `wind` into following/headwind/crosswind relative to `spray_angle_degrees`.

    Returns `None` if `wind` is `None`, if `spray_angle_degrees` is `None`/
    `NaN`, or if `wind.movement_bearing_degrees` is `None` (calm wind or an
    unparseable direction) -- calm wind is handled separately by callers
    (speed 0 in all three components is a valid, non-`None` result only
    when the wind truly has zero speed; see `zero_wind_components`).
    """
    if wind is None or spray_angle_degrees is None:
        return None
    if isinstance(spray_angle_degrees, float) and math.isnan(spray_angle_degrees):
        return None
    if wind.movement_bearing_degrees is None:
        return None

    relative_angle_rad = math.radians(wind.movement_bearing_degrees - spray_angle_degrees)
    following = wind.speed_mps * math.cos(relative_angle_rad)
    crosswind = wind.speed_mps * math.sin(relative_angle_rad)
    return WindComponents(
        following_wind_mps=following, headwind_mps=-following, crosswind_mps=crosswind
    )


def zero_wind_components() -> WindComponents:
    """The (following=0, headwind=0, crosswind=0) result for calm/roof-suppressed wind."""
    return WindComponents(following_wind_mps=0.0, headwind_mps=0.0, crosswind_mps=0.0)
