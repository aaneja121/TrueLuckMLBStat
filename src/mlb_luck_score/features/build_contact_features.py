"""Reusable contact-feature preprocessing pipeline.

Defines the Version 0.1 baseline feature set and a scikit-learn
`ColumnTransformer` that imputes and scales/encodes it. Also provides an
explicit leakage check: nothing computed from or encoding the play's result
may enter the feature set (see `mlb_luck_score.config.LEAKAGE_COLUMNS`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlb_luck_score.config import LEAKAGE_COLUMNS
from mlb_luck_score.data.outfield_physics import (
    estimate_hang_time_seconds,
    estimate_landing_coordinates_ft,
)
from mlb_luck_score.data.weather_physics import DEFAULT_REFERENCE_AIR_DENSITY_KG_M3

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

#: Version 0.4 park-geometry numeric feature set -- NOT used by the default
#: `baseline_v02` feature set, only by the `geometry_only_v04_candidate` /
#: `venue_plus_geometry_v04_candidate` comparison variants (see
#: `mlb_luck_score.models.compare_geometry_aware`). Produced by
#: `mlb_luck_score.data.join_park_geometry.join_park_geometry` plus the
#: interaction terms computed by `add_geometry_interaction_features` below.
#: None of these are computed from the play's outcome -- wall geometry is
#: known before the pitch is thrown, and the interaction terms only combine
#: pre-outcome contact/geometry values -- so none are target-leakage columns
#: (see `mlb_luck_score.config.LEAKAGE_COLUMNS`).
GEOMETRY_NUMERIC_FEATURES: tuple[str, ...] = (
    "wall_distance_in_spray_direction",
    "wall_height_in_spray_direction",
    "projected_distance_to_wall_margin",
    "absolute_distance_to_wall",
    "wall_interpolation_distance_degrees",
    "launch_angle_x_wall_height",
    "wall_margin_x_launch_angle",
    "spray_angle_x_wall_distance",
)

#: Version 0.4 park-geometry categorical/boolean feature set. Boolean
#: columns (nullable `"boolean"` dtype, see `join_park_geometry`) are cast
#: to nullable strings by `add_geometry_interaction_features` so they flow
#: through the same impute-constant + one-hot-encode categorical pipeline as
#: every other categorical feature (missing -> its own "missing" category,
#: never silently dropped or coerced to False).
GEOMETRY_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "wall_segment_label",
    "high_wall_indicator",
    "near_wall_5ft",
    "near_wall_10ft",
    "near_wall_20ft",
    "projected_beyond_wall",
    "temporary_or_special_venue",
    "geometry_uncertain",
)


def add_geometry_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Version 0.4 geometry interaction features and cast boolean flags.

    A no-op (returns `df` unchanged) if `df` has no geometry columns --
    ordinary (non-geometry) callers see no behavior change. Must be called
    AFTER `mlb_luck_score.data.join_park_geometry.join_park_geometry`.

    Adds:
        - `launch_angle_x_wall_height`: `launch_angle * wall_height_in_spray_direction`.
        - `wall_margin_x_launch_angle`: `projected_distance_to_wall_margin * launch_angle`.
        - `spray_angle_x_wall_distance`: `spray_angle_approx * wall_distance_in_spray_direction`.

    All three propagate NaN when an input is missing (e.g. no reviewed wall
    height at this point) rather than guessing a value -- the numeric
    preprocessing pipeline's median imputer (see `build_preprocessing_pipeline`)
    handles the resulting missingness the same way it does for every other
    numeric feature.

    Also casts every column in `GEOMETRY_CATEGORICAL_FEATURES` present in
    `df` (nullable `"boolean"`/`"string"` dtypes coming out of
    `join_park_geometry`) to plain `object` dtype with `None` for missing,
    stringifying non-null values (see that constant's docstring). This is
    NOT the same as pandas' own nullable `"string"` dtype: scikit-learn's
    `SimpleImputer` raises `TypeError: boolean value of NA is ambiguous` on
    pandas' `pd.NA` sentinel (verified against the installed pandas/
    scikit-learn versions), so a `None`-sentinel plain-object column is used
    instead -- `SimpleImputer` handles `None` correctly.
    """
    if "wall_distance_in_spray_direction" not in df.columns:
        return df

    out = df.copy()
    out["launch_angle_x_wall_height"] = out["launch_angle"] * out["wall_height_in_spray_direction"]
    out["wall_margin_x_launch_angle"] = (
        out["projected_distance_to_wall_margin"] * out["launch_angle"]
    )
    out["spray_angle_x_wall_distance"] = (
        out["spray_angle_approx"] * out["wall_distance_in_spray_direction"]
    )

    for col in GEOMETRY_CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].apply(lambda v: None if pd.isna(v) else str(v)).astype(object)

    return out


#: Version 0.5 weather numeric feature set used by `weather_basic_v05_candidate`
#: (see `mlb_luck_score.models.compare_weather_aware`). Produced by
#: `mlb_luck_score.data.join_weather_features.join_weather_features`. None of
#: these are computed from the play's outcome -- weather is known before the
#: pitch is thrown -- so none are target-leakage columns.
WEATHER_BASIC_NUMERIC_FEATURES: tuple[str, ...] = (
    "temperature_c",
    "humidity_pct",
    "air_density_kg_m3",
)
WEATHER_BASIC_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "roof_status",
    "indoor_indicator",
)

#: Version 0.5 weather numeric feature set used by `weather_vector_v05_candidate`
#: -- adds the per-play wind decomposition (following/head/crosswind, relative
#: to THIS play's own spray direction) and the air-density deviation from the
#: fixed reference atmosphere on top of `WEATHER_BASIC_NUMERIC_FEATURES`.
WEATHER_VECTOR_NUMERIC_FEATURES: tuple[str, ...] = (
    *WEATHER_BASIC_NUMERIC_FEATURES,
    "air_density_deviation_from_reference",
    "following_wind_mps",
    "headwind_mps",
    "crosswind_mps",
)
WEATHER_VECTOR_CATEGORICAL_FEATURES: tuple[str, ...] = (
    *WEATHER_BASIC_CATEGORICAL_FEATURES,
    "weather_match_quality",
    "weather_uncertain",
)

#: Version 0.5.1 CORRECTION candidates -- see `mlb_luck_score.models.
#: compare_weather_variants` module docstring for why `weather_vector_v05_
#: candidate` above (which includes BOTH the raw components AND the derived
#: `air_density_kg_m3`/`air_density_deviation_from_reference`) is a flawed
#: feature set: `air_density_kg_m3` is a near-deterministic function of
#: `temperature_c`/`pressure_hpa`/`humidity_pct`, and including a derived
#: quantity alongside every variable used to derive it creates severe
#: multicollinearity -- individual coefficients (and therefore per-play
#: attribution) become unstable and can flip sign even when the model's
#: aggregate calibration looks fine. Each candidate below is deliberately
#: INDEPENDENT (not cumulative) so the two hypotheses -- "density alone
#: captures what matters" vs. "the individual components matter" -- can be
#: tested separately, each with only one representation of the
#: temperature/humidity/pressure/density family, never both.

#: `density_only_v051_candidate`: air density + wind components + roof
#: status. NO temperature/humidity/pressure -- if air density really is a
#: sufficient physical summary, this alone should carry the signal.
WEATHER_DENSITY_ONLY_NUMERIC_FEATURES: tuple[str, ...] = (
    "air_density_kg_m3",
    "following_wind_mps",
    "headwind_mps",
    "crosswind_mps",
)
WEATHER_DENSITY_ONLY_CATEGORICAL_FEATURES: tuple[str, ...] = WEATHER_BASIC_CATEGORICAL_FEATURES

#: `components_only_v051_candidate`: the three raw atmospheric measurements
#: + wind components + roof status. NO derived air density -- lets the
#: model find its own (possibly nonlinear-via-interaction) combination of
#: the raw variables instead of being handed a fixed, collinear derived one.
WEATHER_COMPONENTS_ONLY_NUMERIC_FEATURES: tuple[str, ...] = (
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "following_wind_mps",
    "headwind_mps",
    "crosswind_mps",
)
WEATHER_COMPONENTS_ONLY_CATEGORICAL_FEATURES: tuple[str, ...] = WEATHER_BASIC_CATEGORICAL_FEATURES

#: `density_anomaly_v051_candidate`: each venue's air-density ANOMALY (see
#: `mlb_luck_score.data.join_weather_features.add_venue_air_density_anomaly`
#: -- actual density minus that venue's own training-season-only normal
#: density) + wind components + roof status. NO raw `air_density_kg_m3` --
#: this is meant to separate "was today unusual weather for THIS park" from
#: "this park is persistently high/low altitude", which the raw density
#: value conflates (Coors Field's raw density is almost always low, so it
#: mostly just encodes venue identity, not day-specific weather).
WEATHER_DENSITY_ANOMALY_NUMERIC_FEATURES: tuple[str, ...] = (
    "air_density_venue_anomaly_kg_m3",
    "following_wind_mps",
    "headwind_mps",
    "crosswind_mps",
)
WEATHER_DENSITY_ANOMALY_CATEGORICAL_FEATURES: tuple[str, ...] = WEATHER_BASIC_CATEGORICAL_FEATURES


def add_weather_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cast Version 0.5 weather boolean/categorical flags for the model pipeline.

    A no-op (returns `df` unchanged) if `df` has no weather columns --
    ordinary (non-weather) callers see no behavior change. Must be called
    AFTER `mlb_luck_score.data.join_weather_features.join_weather_features`.

    Casts every column in `WEATHER_VECTOR_CATEGORICAL_FEATURES` present in
    `df` to plain `object` dtype with `None` for missing, stringifying
    non-null values -- the SAME fix as `add_geometry_interaction_features`
    (see that function's docstring): scikit-learn's `SimpleImputer` raises
    on pandas' nullable `"boolean"`/`"string"` dtype `pd.NA` sentinel.

    IMPORTANT for callers computing subgroup masks (e.g. by `roof_status` or
    `weather_uncertain`) on a DataFrame that has been through this function:
    use a boolean coercion that handles BOTH native booleans and the
    stringified `"True"`/`"False"` this function produces (see
    `mlb_luck_score.models.compare_geometry_aware._bool_mask`, reused by
    `mlb_luck_score.models.compare_weather_aware`) -- a naive `.astype(bool)`
    on the stringified column treats `"False"` as truthy.
    """
    if "temperature_c" not in df.columns and "air_density_kg_m3" not in df.columns:
        return df

    out = df.copy()
    for col in WEATHER_VECTOR_CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].apply(lambda v: None if pd.isna(v) else str(v)).astype(object)
    return out


#: Version 0.5 standardized-environment reference (ICAO International
#: Standard Atmosphere sea-level values -- the SAME reference used by
#: `mlb_luck_score.data.weather_physics.DEFAULT_REFERENCE_AIR_DENSITY_KG_M3`,
#: so the standardized row's `air_density_kg_m3` and its own temperature/
#: pressure/humidity are mutually consistent, not an arbitrary mix). Zero
#: effective wind and an `"outdoor_open_air"` roof-neutral status isolate
#: the weather-specific counterfactual: "what would this identical batted
#: ball have looked like under a fixed, neutral atmosphere with no wind?"
#: Humidity is 0% by convention, matching the ISA's own dry-air definition
#: (the reference density figure is a DRY-air value) -- not a claim that 0%
#: humidity is "typical" game weather.
STANDARD_ENVIRONMENT_TEMPERATURE_C = 15.0
STANDARD_ENVIRONMENT_HUMIDITY_PCT = 0.0
STANDARD_ENVIRONMENT_PRESSURE_HPA = 1013.25
STANDARD_ENVIRONMENT_ROOF_STATUS = "outdoor_open_air"
#: IMPORTANT: must be a category the model actually saw during training
#: (`"good"`, the most common `weather_match_quality` value -- a standardized
#: condition is by construction fully certain/reliable, so this is also the
#: semantically correct choice, not just a technical workaround). A novel,
#: never-seen-in-training string (e.g. a literal `"standardized"` label) gets
#: one-hot-encoded as ALL ZEROS by `OneHotEncoder(handle_unknown="ignore")`
#: -- a pattern the fitted model never learned to interpret, which produced
#: verified, physically backwards counterfactual predictions (e.g. Coors
#: Field's thin actual air scoring WORSE than the denser standardized
#: reference) before this was caught and fixed. See
#: `generate_standardized_environment_rows`'s docstring.
STANDARD_ENVIRONMENT_MATCH_QUALITY = "good"


def generate_standardized_environment_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Replace every weather feature in `df` with the Version 0.5 standardized environment.

    Every NON-weather column (contact physics, park geometry, venue
    identity, batter-controlled features) is left EXACTLY as-is -- only
    `WEATHER_BASIC_NUMERIC_FEATURES` / `WEATHER_VECTOR_NUMERIC_FEATURES` /
    `WEATHER_BASIC_CATEGORICAL_FEATURES` / `WEATHER_VECTOR_CATEGORICAL_FEATURES`
    columns present in `df` are overwritten with the fixed standardized
    values, and only for rows where actual weather was itself available
    (`has_effective_weather`, if present) -- a row with no real weather to
    begin with has no meaningful "actual vs. standardized" comparison.

    Args:
        df: A feature-ready DataFrame that has been through
            `add_weather_interaction_features` (so weather categorical
            columns are already in the plain-object-with-None form the
            model pipeline expects -- this function writes the SAME
            representation for its overridden values).

    IMPORTANT: every overridden categorical value MUST be a category the
    model actually saw during training -- a novel category is one-hot
    -encoded as all zeros by `OneHotEncoder(handle_unknown="ignore")`, a
    pattern the fitted model never learned to interpret (verified: an
    earlier version of this function used a synthetic, never-seen
    `weather_match_quality="standardized"` label and produced physically
    backwards counterfactual predictions as a result -- see
    `STANDARD_ENVIRONMENT_MATCH_QUALITY`'s comment). `weather_match_quality`
    is set to the already-seen `"good"` value (a standardized condition is,
    by construction, fully certain/reliable) rather than any new label.

    Returns:
        A copy of `df` with weather columns replaced by the standardized
        environment. Raises nothing -- rows without weather columns present
        at all are returned unchanged (this is a no-op for non-weather data).
    """
    out = df.copy()
    if "temperature_c" not in out.columns and "air_density_kg_m3" not in out.columns:
        return out

    mask = (
        out["has_effective_weather"]
        if "has_effective_weather" in out.columns
        else pd.Series(True, index=out.index)
    )

    numeric_overrides = {
        "temperature_c": STANDARD_ENVIRONMENT_TEMPERATURE_C,
        "humidity_pct": STANDARD_ENVIRONMENT_HUMIDITY_PCT,
        "pressure_hpa": STANDARD_ENVIRONMENT_PRESSURE_HPA,
        "air_density_kg_m3": DEFAULT_REFERENCE_AIR_DENSITY_KG_M3,
        "air_density_deviation_from_reference": 0.0,
        # "Standardized" for the venue-anomaly candidate means "normal for
        # that venue" (zero anomaly), NOT the fixed ISA reference -- this
        # candidate's whole point is to model deviations from each venue's
        # own baseline, so its counterfactual must hold that baseline fixed
        # rather than substituting an unrelated global constant.
        "air_density_venue_anomaly_kg_m3": 0.0,
        "wind_speed_mps": 0.0,
        "following_wind_mps": 0.0,
        "headwind_mps": 0.0,
        "crosswind_mps": 0.0,
    }
    for col, value in numeric_overrides.items():
        if col in out.columns:
            out.loc[mask, col] = value

    categorical_overrides = {
        "roof_status": STANDARD_ENVIRONMENT_ROOF_STATUS,
        "indoor_indicator": "False",
        "weather_match_quality": STANDARD_ENVIRONMENT_MATCH_QUALITY,
        "weather_uncertain": "False",
    }
    for col, str_value in categorical_overrides.items():
        if col in out.columns:
            out.loc[mask, col] = str_value

    return out


#: Version 0.6 alignment-label feature set -- the SAME two raw Statcast
#: columns as `OPTIONAL_CATEGORICAL_FEATURES`, given their own name/docstring
#: for this version's comparison variants (see `mlb_luck_score.models.
#: compare_alignment_aware`). This is the ONLY defensive-positioning signal
#: public Statcast data exposes -- coarse, pre-pitch labels (`if_fielding_
#: alignment`: "Standard"/"Strategic"/"Infield shift"/"Infield shade";
#: `of_fielding_alignment`: "Standard"/"Strategic"/"4th outfielder"), never
#: exact fielder coordinates, movement, or reaction. The alignment is set
#: BEFORE the pitch is thrown, so neither column is a target-leakage column
#: (see `mlb_luck_score.config.LEAKAGE_COLUMNS`). Real coverage (verified
#: against 2021-2024 development data): ~99.6% non-null in every season.
ALIGNMENT_LABELS_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "if_fielding_alignment",
    "of_fielding_alignment",
)

#: `alignment_interactions_v06`'s feature set = `ALIGNMENT_LABELS_CATEGORICAL_
#: FEATURES` (base labels) + these physically-motivated interaction terms.
#: Chosen to mirror the task's stated priorities: infield alignment x batter
#: handedness / spray direction (categorical x categorical -- combined into
#: one joint category so a linear model can learn a distinct coefficient per
#: combination, which additive one-hot encoding of the two columns
#: separately cannot represent), outfield alignment x launch angle /
#: projected distance (categorical x numeric, via a 0/1 shift indicator),
#: and shift status x pull-side ground ball (boolean x boolean, as a 0/1
#: product). None of these are computed from the play's outcome -- alignment,
#: handedness, spray direction, launch angle, and batted-ball type are all
#: pre-outcome or swing-mechanics values -- so none are target-leakage
#: columns.
ALIGNMENT_INTERACTION_NUMERIC_FEATURES: tuple[str, ...] = (
    "if_alignment_shift_indicator",
    "of_alignment_shift_indicator",
    "of_shift_x_launch_angle",
    "of_shift_x_hit_distance",
    "if_shift_x_pull_groundball",
)
ALIGNMENT_INTERACTION_CATEGORICAL_FEATURES: tuple[str, ...] = (
    *ALIGNMENT_LABELS_CATEGORICAL_FEATURES,
    "if_alignment_x_stand",
    "if_alignment_x_spray_sector",
)

#: The single most common value of BOTH `if_fielding_alignment` and
#: `of_fielding_alignment` in real 2021-2024 data -- used as the "typical"
#: reference alignment for the Version 0.6 positioning counterfactual (see
#: `generate_typical_alignment_rows`). MUST be a category the model actually
#: saw during training -- see that function's docstring for why (the same
#: `OneHotEncoder(handle_unknown="ignore")` lesson documented at
#: `STANDARD_ENVIRONMENT_MATCH_QUALITY` above).
STANDARD_ALIGNMENT_LABEL = "Standard"


def _alignment_shift_indicator(series: pd.Series) -> pd.Series:
    """0.0 for `STANDARD_ALIGNMENT_LABEL`, 1.0 for any other non-null alignment, NaN if missing."""
    return series.apply(
        lambda v: float("nan") if pd.isna(v) else (0.0 if v == STANDARD_ALIGNMENT_LABEL else 1.0)
    )


def _combine_categorical(a: pd.Series, b: pd.Series) -> pd.Series:
    """Join two categorical columns row-wise into one `"{a}_{b}"` category; `None` if either is null."""
    combined = [None if pd.isna(x) or pd.isna(y) else f"{x}_{y}" for x, y in zip(a, b, strict=True)]
    return pd.Series(combined, index=a.index, dtype=object)


def add_alignment_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Version 0.6 alignment interaction features from raw alignment labels.

    A no-op (returns `df` unchanged) if `df` has neither alignment column --
    ordinary (non-alignment) callers see no behavior change. Safe to call
    more than once (e.g. after `generate_typical_alignment_rows` overrides
    the raw labels) -- it always recomputes every derived column from
    whatever `if_fielding_alignment`/`of_fielding_alignment` currently hold.

    Adds:
        - `if_alignment_shift_indicator` / `of_alignment_shift_indicator`:
          0.0 if `STANDARD_ALIGNMENT_LABEL`, 1.0 if any other alignment, NaN
          if missing.
        - `of_shift_x_launch_angle`: `of_alignment_shift_indicator *
          launch_angle`.
        - `of_shift_x_hit_distance`: `of_alignment_shift_indicator *
          hit_distance_sc`.
        - `if_shift_x_pull_groundball`: `if_alignment_shift_indicator *
          (is_pull AND bb_type == "ground_ball")` -- missing `is_pull`
          (nullable boolean) is treated as "not pull", consistent with every
          other subgroup mask in this codebase (see `mlb_luck_score.models.
          compare_geometry_aware._bool_mask`'s docstring).
        - `if_alignment_x_stand`: `if_fielding_alignment` joined with
          `stand` (e.g. `"Infield shift_R"`).
        - `if_alignment_x_spray_sector`: `if_fielding_alignment` joined with
          `spray_sector`.

    All numeric outputs propagate NaN when an input is missing, rather than
    guessing a value -- the numeric preprocessing pipeline's median imputer
    (see `build_preprocessing_pipeline`) handles the resulting missingness
    like every other numeric feature.
    """
    if "if_fielding_alignment" not in df.columns and "of_fielding_alignment" not in df.columns:
        return df

    out = df.copy()
    nan_series = pd.Series(float("nan"), index=out.index)

    out["if_alignment_shift_indicator"] = (
        _alignment_shift_indicator(out["if_fielding_alignment"])
        if "if_fielding_alignment" in out.columns
        else nan_series
    )
    out["of_alignment_shift_indicator"] = (
        _alignment_shift_indicator(out["of_fielding_alignment"])
        if "of_fielding_alignment" in out.columns
        else nan_series
    )

    launch_angle = out["launch_angle"] if "launch_angle" in out.columns else nan_series
    hit_distance = out["hit_distance_sc"] if "hit_distance_sc" in out.columns else nan_series
    out["of_shift_x_launch_angle"] = out["of_alignment_shift_indicator"] * launch_angle
    out["of_shift_x_hit_distance"] = out["of_alignment_shift_indicator"] * hit_distance

    if "is_pull" in out.columns and "bb_type" in out.columns:
        pull_groundball = (
            out["is_pull"].fillna(False).astype(bool) & (out["bb_type"] == "ground_ball")
        ).astype(float)
    else:
        pull_groundball = nan_series
    out["if_shift_x_pull_groundball"] = out["if_alignment_shift_indicator"] * pull_groundball

    out["if_alignment_x_stand"] = (
        _combine_categorical(out["if_fielding_alignment"], out["stand"])
        if "if_fielding_alignment" in out.columns and "stand" in out.columns
        else None
    )
    out["if_alignment_x_spray_sector"] = (
        _combine_categorical(out["if_fielding_alignment"], out["spray_sector"])
        if "if_fielding_alignment" in out.columns and "spray_sector" in out.columns
        else None
    )

    return out


def generate_typical_alignment_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Replace every alignment feature in `df` with the Version 0.6 "typical" alignment.

    Used to build the positioning counterfactual (see `mlb_luck_score.
    scoring.positioning_attribution`): `positioning_effect = EV(actual
    alignment) - EV(typical alignment)`, holding every other (non-alignment)
    feature -- contact physics, venue, batter handedness, spray direction --
    EXACTLY fixed. Only rows where alignment was actually observed are
    overridden (a row with no real alignment data has no meaningful
    actual-vs-typical comparison).

    Both `if_fielding_alignment` and `of_fielding_alignment` are set to
    `STANDARD_ALIGNMENT_LABEL` ("Standard" -- each column's single most
    common real value, see that constant's docstring), and every alignment
    interaction column is then RECOMPUTED from the overridden labels via
    `add_alignment_interaction_features`, so `alignment_interactions_v06`'s
    derived columns (`if_alignment_x_stand`, the shift indicators, etc.)
    stay consistent with the overridden alignment rather than going stale.

    IMPORTANT: `STANDARD_ALIGNMENT_LABEL` MUST be a category the model
    actually saw during training -- the same `OneHotEncoder(handle_unknown=
    "ignore")` lesson as `generate_standardized_environment_rows` (an unseen
    category is silently one-hot-encoded as all zeros, a pattern the fitted
    model never learned to interpret). "Standard" is real data's dominant
    category for both columns, so this is both the technically-necessary
    choice and the semantically correct one ("typical" alignment).

    Returns:
        A copy of `df` with alignment columns replaced by the typical
        alignment and interaction columns recomputed. A no-op for rows/data
        with no alignment columns present at all.
    """
    if "if_fielding_alignment" not in df.columns and "of_fielding_alignment" not in df.columns:
        return df

    out = df.copy()
    if "if_fielding_alignment" in out.columns and "of_fielding_alignment" in out.columns:
        mask = out["if_fielding_alignment"].notna() & out["of_fielding_alignment"].notna()
    elif "if_fielding_alignment" in out.columns:
        mask = out["if_fielding_alignment"].notna()
    else:
        mask = out["of_fielding_alignment"].notna()

    for col in ("if_fielding_alignment", "of_fielding_alignment"):
        if col in out.columns:
            out.loc[mask, col] = STANDARD_ALIGNMENT_LABEL

    return add_alignment_interaction_features(out)


#: Minimum fraction of non-null values a feature column must have to be
#: included automatically. Below this, `select_available_features` /
#: `select_opportunity_features` drop the column and log why, rather than
#: silently modeling on a mostly-empty field.
MIN_NON_NULL_FRACTION = 0.5

#: The Version 0.7A opportunity model's binary target: did the batter-runner
#: get put out on this play? Reuses the EXISTING `outcome_class` column
#: (`mlb_luck_score.eligibility`) rather than reimplementing out/not-out
#: logic -- `outcome_class == "out"` already covers every batter-out result
#: (field outs, sac flies, double plays where the batter is out, etc.; see
#: `mlb_luck_score.eligibility._UNAMBIGUOUS_OUT_EVENTS`).
OPPORTUNITY_TARGET_COLUMN = "converted_to_out"

#: Version 0.7A `measured_contact_only_v07` feature set -- see
#: `mlb_luck_score.models.compare_opportunity_models` module docstring for
#: the full public-data audit and why a second, position-proxy-based
#: candidate (`typical_position_proxy_v07`) was NOT built. Every numeric
#: feature here is either directly measured (`launch_speed`, `launch_angle`,
#: `hit_distance_sc`, and the Version 0.4 wall-proximity columns from
#: `mlb_luck_score.data.join_park_geometry`) or physics-ESTIMATED
#: (`estimated_hang_time_s`, `landing_x_ft`, `landing_y_ft` -- see
#: `mlb_luck_score.data.outfield_physics`, which documents why these have no
#: public ground truth to validate against). NO assumed defender starting
#: position or distance-needed feature is included. None of these are
#: computed from the play's outcome, so none are target-leakage columns.
OPPORTUNITY_NUMERIC_FEATURES: tuple[str, ...] = (
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "estimated_hang_time_s",
    "landing_x_ft",
    "landing_y_ft",
    "wall_distance_in_spray_direction",
    "absolute_distance_to_wall",
)
#: `assigned_outfield_position` (7/8/9, see `mlb_luck_score.eligibility.
#: add_outfield_opportunity_eligibility`) is treated as CATEGORICAL, not
#: numeric -- the position codes are labels for LF/CF/RF, not an ordinal
#: quantity where 8 is meaningfully "between" 7 and 9 for a linear model.
OPPORTUNITY_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "bb_type",
    "of_fielding_alignment",
    "assigned_outfield_position",
)

#: Version 0.7C near-wall specialist feature set -- `OPPORTUNITY_NUMERIC_
#: FEATURES`/`OPPORTUNITY_CATEGORICAL_FEATURES` plus the remaining Version
#: 0.4 wall-geometry columns (`wall_height_in_spray_direction`,
#: `projected_distance_to_wall_margin`, `wall_segment_label`) that
#: `measured_contact_only_v07` does not use. See `mlb_luck_score.models.
#: compare_near_wall_models` module docstring: the SAME feature set is used
#: for both the logistic and HistGradientBoostingClassifier candidates --
#: the comparison is about MODEL CLASS (can a nonlinear model find
#: interactions among hang time/launch angle/wall distance/wall height/spray
#: direction that a linear model can't), not about giving one candidate
#: hand-engineered interaction terms the other lacks. None of these are
#: computed from the play's outcome, so none are target-leakage columns.
NEAR_WALL_NUMERIC_FEATURES: tuple[str, ...] = (
    *OPPORTUNITY_NUMERIC_FEATURES,
    "wall_height_in_spray_direction",
    "projected_distance_to_wall_margin",
)
NEAR_WALL_CATEGORICAL_FEATURES: tuple[str, ...] = (
    *OPPORTUNITY_CATEGORICAL_FEATURES,
    "wall_segment_label",
)

#: Version 0.8 `infield_contact_only_v08` target column -- distinct from
#: `OPPORTUNITY_TARGET_COLUMN` (outfield): `y_out` is built directly by
#: `mlb_luck_score.eligibility.add_infield_opportunity_eligibility` (1 =
#: `field_out`, 0 = safely reached INCLUDING on an error -- see that
#: function's docstring for why a reached-on-error play is `y_out = 0`, never
#: a model input on its own). `add_infield_opportunity_features` below
#: ALIASES `y_out` into `OPPORTUNITY_TARGET_COLUMN` so the EXISTING, UNCHANGED
#: `mlb_luck_score.models.train_opportunity_model` trainer (hardcoded to read
#: `OPPORTUNITY_TARGET_COLUMN`) can be reused as-is for the infield model,
#: rather than duplicating ~370 lines of training/evaluation machinery for a
#: differently-named target column.
INFIELD_TARGET_COLUMN = "y_out"

#: Version 0.8 `infield_contact_only_v08` feature set -- the ONLY candidate
#: built (see `mlb_luck_score.models.compare_infield_opportunity` module
#: docstring for why `infield_time_margin_proxy_v08_candidate` was
#: INVESTIGATED but NOT built: no citable, calibrated public physics exists
#: for ground-ball roll deceleration the way `mlb_luck_score.data.
#: outfield_physics`'s vacuum projectile formula exists for airborne
#: trajectories, and building one would require inventing an unvalidated
#: friction constant -- exactly the kind of assumption the task instructs
#: against). Every numeric feature here is either directly measured
#: (`launch_speed`, `launch_angle`, `hit_distance_sc`, `outs_when_up`) or a
#: real, documented SEASON-LEVEL public leaderboard value (`sprint_speed` --
#: see `mlb_luck_score.data.download_sprint_speed`/`join_sprint_speed`; NOT a
#: per-play measurement, and NOT available for every batter -- see
#: `select_infield_opportunity_features`'s missingness handling).
#: `on_1b_occupied` is a pre-contact game-state indicator (double-play-depth
#: positioning depends on it) computed by `add_infield_opportunity_features`,
#: not a raw column. None of these are computed from the play's OWN outcome,
#: so none are target-leakage columns.
INFIELD_NUMERIC_FEATURES: tuple[str, ...] = (
    "launch_speed",
    "launch_angle",
    "spray_angle_approx",
    "hit_distance_sc",
    "sprint_speed",
    "outs_when_up",
    "on_1b_occupied",
)
#: `assigned_infield_position` (1-6, see `mlb_luck_score.eligibility.
#: add_infield_opportunity_eligibility`) is CATEGORICAL, not numeric -- the
#: same "position codes are labels, not an ordinal quantity" reasoning as
#: `assigned_outfield_position` above. `surface_type` (Grass/Artificial Turf)
#: is the ONE venue-context feature used here: park GEOMETRY is irrelevant to
#: an infield ground ball (no wall involved), but playing surface genuinely
#: affects ground-ball speed/bounce consistency, unlike geometry.
INFIELD_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "stand",
    "if_fielding_alignment",
    "assigned_infield_position",
    "surface_type",
)


def add_opportunity_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the Version 0.7A binary target column from the existing `outcome_class`.

    `converted_to_out = 1` if `outcome_class == "out"`, else `0`. Rows with a
    null `outcome_class` (should not occur for `outfield_opportunity_
    eligible` rows, which are a subset of `eligible_for_training`) get a
    null target rather than a guessed value.
    """
    out = df.copy()
    out[OPPORTUNITY_TARGET_COLUMN] = (out["outcome_class"] == "out").astype("Int64")
    out.loc[out["outcome_class"].isna(), OPPORTUNITY_TARGET_COLUMN] = pd.NA
    return out


def add_outfield_opportunity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Version 0.7A opportunity-difficulty features and the binary target.

    A no-op (returns `df` unchanged) if `df` has no `launch_speed`/
    `launch_angle`/`hit_distance_sc`/`spray_angle_approx` columns. Must be
    called AFTER `mlb_luck_score.eligibility.add_outfield_opportunity_
    eligibility` (uses `assigned_outfield_position`) and, for wall
    -proximity features to be available, after `mlb_luck_score.data.
    join_park_geometry.join_park_geometry` (a no-op for the wall columns,
    not an error, if geometry hasn't been joined -- they simply won't be in
    `df` and `select_opportunity_features` will drop them for insufficient
    non-null coverage like any other feature).

    Adds:
        - `estimated_hang_time_s`, `landing_x_ft`, `landing_y_ft`: see
          `mlb_luck_score.data.outfield_physics`.
        - `converted_to_out`: see `add_opportunity_target`.
        - `assigned_outfield_position` is cast to plain `object` dtype with
          `None` for missing (same `SimpleImputer`-compatibility fix as
          `add_geometry_interaction_features`/`add_weather_interaction_
          features` -- pandas' nullable `Int64`/`pd.NA` raises inside
          `SimpleImputer`).
    """
    required = ("launch_speed", "launch_angle", "hit_distance_sc", "spray_angle_approx")
    if not all(col in df.columns for col in required):
        return df

    out = df.copy()
    hang_times = []
    landing_xs = []
    landing_ys = []
    # .astype(float) turns pandas nullable-dtype pd.NA into plain np.nan --
    # estimate_hang_time_seconds/estimate_landing_coordinates_ft use
    # math.isnan and stay pandas-independent, so the conversion happens here.
    launch_speed = out["launch_speed"].astype(float)
    launch_angle = out["launch_angle"].astype(float)
    hit_distance = out["hit_distance_sc"].astype(float)
    spray_angle = out["spray_angle_approx"].astype(float)
    for speed, angle, distance, spray in zip(
        launch_speed, launch_angle, hit_distance, spray_angle, strict=True
    ):
        hang_times.append(estimate_hang_time_seconds(speed, angle))
        x, y = estimate_landing_coordinates_ft(distance, spray)
        landing_xs.append(x)
        landing_ys.append(y)
    out["estimated_hang_time_s"] = hang_times
    out["landing_x_ft"] = landing_xs
    out["landing_y_ft"] = landing_ys

    if "outcome_class" in out.columns:
        out = add_opportunity_target(out)

    if "assigned_outfield_position" in out.columns:
        out["assigned_outfield_position"] = (
            out["assigned_outfield_position"].apply(lambda v: None if pd.isna(v) else str(int(v)))
        ).astype(object)

    return out


def add_infield_opportunity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Version 0.8 infield-opportunity features and alias the binary target.

    A no-op (returns `df` unchanged) if `df` has no `y_out` column. Must be
    called AFTER `mlb_luck_score.eligibility.add_infield_opportunity_
    eligibility` (uses `y_out`, `assigned_infield_position`) and, for
    `sprint_speed` to be available, after `mlb_luck_score.data.
    join_sprint_speed.join_sprint_speed` (a no-op for that column, not an
    error, if it hasn't been joined -- `select_infield_opportunity_features`
    drops it for insufficient non-null coverage like any other feature, same
    as `select_opportunity_features`).

    Adds:
        - `on_1b_occupied`: `1` if `on_1b` is non-null (a runner was on
          first), else `0` -- a pre-contact game-state indicator (double
          -play-depth positioning depends on it), NOT computed from this
          play's own outcome.
        - `converted_to_out`: `y_out`, ALIASED under
          `mlb_luck_score.features.build_contact_features.
          OPPORTUNITY_TARGET_COLUMN`'s name so the existing, UNCHANGED
          `mlb_luck_score.models.train_opportunity_model` trainer (hardcoded
          to read that column name) can be reused as-is -- see
          `INFIELD_TARGET_COLUMN`'s docstring.
        - `assigned_infield_position` is cast to plain `object` dtype with
          `None` for missing (same `SimpleImputer`-compatibility fix as
          `assigned_outfield_position` above).
    """
    if "y_out" not in df.columns:
        return df

    out = df.copy()
    out["on_1b_occupied"] = out["on_1b"].notna().astype(int) if "on_1b" in out.columns else 0

    out[OPPORTUNITY_TARGET_COLUMN] = out["y_out"]

    if "assigned_infield_position" in out.columns:
        out["assigned_infield_position"] = (
            out["assigned_infield_position"].apply(lambda v: None if pd.isna(v) else str(int(v)))
        ).astype(object)

    return out


def select_opportunity_features(
    df: pd.DataFrame,
    *,
    min_non_null_fraction: float = MIN_NON_NULL_FRACTION,
) -> tuple[list[str], list[str]]:
    """Choose Version 0.7A `measured_contact_only_v07` features actually usable in `df`.

    A dedicated selector (NOT `select_available_features`, which always
    forces in the Version 0.1 baseline set) -- the opportunity model has its
    own, independent feature list (`OPPORTUNITY_NUMERIC_FEATURES`/
    `OPPORTUNITY_CATEGORICAL_FEATURES`), not the 5-class contact model's.
    Same drop-if-missing-or-too-sparse logic as `select_available_features`.

    Deliberately excludes `responsible_outfielder_id`: the opportunity
    model's whole definition is "probability an AVERAGE MLB outfielder
    converts this," so training on the specific fielder's identity would
    make the model encode THAT fielder's actual skill rather than physical
    difficulty -- exactly the opportunity/execution conflation `mlb_luck_
    score.models.compare_opportunity_models` is designed to avoid. Fielder
    identity is used only for evaluation subgroups and Version 0.7B's
    per-defender execution aggregation, never as a training feature.

    Returns:
        (numeric_features, categorical_features) actually usable.
    """

    def _keep(col: str) -> bool:
        if col not in df.columns:
            logger.info("Opportunity feature '%s' not present in data; excluding.", col)
            return False
        non_null_fraction = df[col].notna().mean() if len(df) else 0.0
        if non_null_fraction < min_non_null_fraction:
            logger.info(
                "Opportunity feature '%s' is only %.1f%% non-null (< %.0f%% threshold); excluding.",
                col,
                non_null_fraction * 100,
                min_non_null_fraction * 100,
            )
            return False
        return True

    numeric_features = [c for c in OPPORTUNITY_NUMERIC_FEATURES if _keep(c)]
    categorical_features = [c for c in OPPORTUNITY_CATEGORICAL_FEATURES if _keep(c)]

    assert_no_leakage(numeric_features + categorical_features)
    return numeric_features, categorical_features


def select_infield_opportunity_features(
    df: pd.DataFrame,
    *,
    min_non_null_fraction: float = MIN_NON_NULL_FRACTION,
) -> tuple[list[str], list[str]]:
    """Choose Version 0.8 `infield_contact_only_v08` features actually usable in `df`.

    Same drop-if-missing-or-too-sparse logic as `select_opportunity_features`
    -- in particular, `sprint_speed` is DROPPED (not imputed or guessed) if
    its non-null coverage in `df` falls below `min_non_null_fraction`
    (real 2021-2024 coverage is ~98.9% of eligible rows, well above the
    default threshold -- see README.md "Infield opportunity (Version 0.8)").

    Deliberately excludes `responsible_infielder_id`: the opportunity
    model's whole definition is "probability an AVERAGE MLB infielder
    converts this," so training on the specific fielder's identity would
    make the model encode THAT fielder's actual skill rather than physical
    difficulty -- the SAME opportunity/execution conflation `mlb_luck_
    score.models.compare_opportunity_models` avoids for outfielders. Fielder
    identity is used only for evaluation subgroups and post-hoc per-defender
    execution aggregation, never as a training feature.

    Returns:
        (numeric_features, categorical_features) actually usable.
    """

    def _keep(col: str) -> bool:
        if col not in df.columns:
            logger.info("Infield opportunity feature '%s' not present in data; excluding.", col)
            return False
        non_null_fraction = df[col].notna().mean() if len(df) else 0.0
        if non_null_fraction < min_non_null_fraction:
            logger.info(
                "Infield opportunity feature '%s' is only %.1f%% non-null (< %.0f%% "
                "threshold); excluding.",
                col,
                non_null_fraction * 100,
                min_non_null_fraction * 100,
            )
            return False
        return True

    numeric_features = [c for c in INFIELD_NUMERIC_FEATURES if _keep(c)]
    categorical_features = [c for c in INFIELD_CATEGORICAL_FEATURES if _keep(c)]

    assert_no_leakage(numeric_features + categorical_features)
    return numeric_features, categorical_features


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
    extra_numeric_features: Sequence[str] = (),
    extra_categorical_features: Sequence[str] = (),
    min_non_null_fraction: float = MIN_NON_NULL_FRACTION,
) -> tuple[list[str], list[str]]:
    """Choose numeric/categorical features actually usable in `df`.

    A candidate feature is dropped (with a log message) if it's absent from
    `df` or if it's present but below `min_non_null_fraction` non-null.
    Nothing is silently imputed away at the column-selection stage --
    per-row imputation happens inside the returned preprocessing pipeline.

    Args:
        df: Training-eligible rows.
        include_optional: Whether to include the standard optional features
            (sprint speed, alignment classifications).
        extra_numeric_features: Additional numeric feature names to consider
            beyond the standard set (e.g. `GEOMETRY_NUMERIC_FEATURES` for a
            geometry-aware comparison variant -- see `mlb_luck_score.models.
            compare_geometry_aware`), subject to the SAME missingness
            threshold and leakage check as every other feature. Empty by
            default -- passing nothing here is identical to prior behavior.
        extra_categorical_features: Additional categorical feature names to
            consider beyond the standard set, subject to the SAME
            missingness threshold and leakage check as every other feature.
            Used for one-off, explicitly-labeled comparison variants (e.g.
            adding `venue_id` for a park-aware candidate model) without
            changing the default feature set for ordinary callers. Empty by
            default -- passing nothing here is identical to prior behavior.
        min_non_null_fraction: Minimum non-null fraction required to keep a
            candidate feature.

    Returns:
        (numeric_features, categorical_features) actually usable.
    """
    numeric_candidates = list(NUMERIC_FEATURES)
    categorical_candidates = list(CATEGORICAL_FEATURES)
    if include_optional:
        numeric_candidates += list(OPTIONAL_NUMERIC_FEATURES)
        categorical_candidates += list(OPTIONAL_CATEGORICAL_FEATURES)
    numeric_candidates += [c for c in extra_numeric_features if c not in numeric_candidates]
    categorical_candidates += [
        c for c in extra_categorical_features if c not in categorical_candidates
    ]

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
