"""Descriptive check of hit_distance_sc on cached 2021-2024 development data.

Questions (v1.2 step 1 re-scope, 2026-09-30):
  Q1 coverage by bb_type / outcome / season
  Q2 home runs: is distance a projected landing point beyond the wall?
  Q3 caught fly balls near the wall: distance vs wall
  Q4 hits off the wall: contact point or projected landing?
  Q5 does distance already carry weather? Within launch-speed/angle bins,
     does the distance residual move with air density and following wind
     (outdoor games only), in the physical direction?
  Q5b the same, within venue (so park altitude cannot drive it)
  Q5c per-venue following-wind point estimates (descriptive, no CI)
No 2025 or 2026 data is read. Nothing is fitted for model use. Results and
their interpretation: docs/plans/v1_2_park_weather_plan.md, "Step 1 re-scope".

Run from the repository root (reads cached files under data/processed/):
    .venv/bin/python scripts/v1_2_hit_distance_check.py
"""

import numpy as np
import pandas as pd

from mlb_luck_score.config import DEVELOPMENT_SEASONS, assert_seasons_allowed

assert_seasons_allowed(DEVELOPMENT_SEASONS)

GEO = "data/processed/cleaned_development_data_with_geometry.parquet"
WX = "data/processed/cleaned_development_data_with_weather.parquet"

geo_cols = [
    "event_id",
    "season",
    "game_pk",
    "bb_type",
    "events",
    "outcome_class",
    "eligible_for_training",
    "hit_distance_sc",
    "launch_speed",
    "launch_angle",
    "wall_distance_in_spray_direction",
    "wall_height_in_spray_direction",
    "projected_distance_to_wall_margin",
    "has_park_geometry",
    "venue_name",
    "des",
]
df = pd.read_parquet(GEO, columns=geo_cols)
assert set(df["season"].unique()) <= set(DEVELOPMENT_SEASONS), df["season"].unique()
df = df[df["eligible_for_training"].fillna(False).astype(bool)].copy()
print(f"eligible rows: {len(df):,}  seasons: {sorted(df['season'].unique())}")

# Q1 coverage
df["has_dist"] = df["hit_distance_sc"].notna()
print("\nQ1 hit_distance_sc coverage by bb_type")
print(df.groupby("bb_type", dropna=False)["has_dist"].agg(["mean", "size"]).round(4))
print("\nQ1 coverage by outcome_class")
print(df.groupby("outcome_class", dropna=False)["has_dist"].agg(["mean", "size"]).round(4))
print("\nQ1 coverage by season")
print(df.groupby("season")["has_dist"].agg(["mean", "size"]).round(4))

g = df[df["has_dist"] & df["has_park_geometry"].fillna(False).astype(bool)].copy()
g["margin"] = g["hit_distance_sc"] - g["wall_distance_in_spray_direction"]
q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]

# Q2 home runs
hr = g[g["events"] == "home_run"]
print(f"\nQ2 home runs with geometry: n={len(hr):,}")
print("margin (distance - wall) quantiles, ft:", hr["margin"].quantile(q).round(1).to_dict())
print(
    "share margin < 0:",
    round((hr["margin"] < 0).mean(), 4),
    " share < -10:",
    round((hr["margin"] < -10).mean(), 4),
)

# Q3 caught air balls (outs) deep
air = g["bb_type"].isin(["fly_ball", "line_drive"])
outs = g[air & (g["outcome_class"] == "out")]
deep_outs = outs[outs["margin"] > -30]
print(
    f"\nQ3 air-ball outs with geometry: n={len(outs):,}; within 30 ft of wall or beyond: n={len(deep_outs):,}"
)
print("margin quantiles for deep outs:", deep_outs["margin"].quantile(q).round(1).to_dict())
print(
    "share of deep outs with margin > 0:",
    round((deep_outs["margin"] > 0).mean(), 4),
    " > +10:",
    round((deep_outs["margin"] > 10).mean(), 4),
)

# Q4 hits off the wall: doubles/triples with des mentioning wall
xbh = g[air & g["events"].isin(["double", "triple"])]
wall_des = xbh["des"].fillna("").str.contains(r"\bwall\b|off the wall|warning track", case=False)
print(
    f"\nQ4 air doubles/triples: n={len(xbh):,}; description mentions wall: n={int(wall_des.sum()):,}"
)
print(
    "margin quantiles, wall-mentioned:", xbh.loc[wall_des, "margin"].quantile(q).round(1).to_dict()
)
print("margin quantiles, all air XBH:", xbh["margin"].quantile(q).round(1).to_dict())

# Q5 weather embedded in distance?
wx = pd.read_parquet(
    WX,
    columns=[
        "event_id",
        "roof_status",
        "has_effective_weather",
        "air_density_kg_m3",
        "following_wind_mps",
        "effective_temperature_c",
        "weather_uncertain",
    ],
)
m = df[df["has_dist"] & (df["bb_type"] == "fly_ball")].merge(wx, on="event_id", how="left")
print(f"\nQ5 fly balls with distance: n={len(m):,}; roof_status counts:")
print(m["roof_status"].value_counts(dropna=False))
out = m[
    m["has_effective_weather"].fillna(False).astype(bool)
    & m["air_density_kg_m3"].notna()
    & m["following_wind_mps"].notna()
].copy()
print(f"with effective density + wind: n={len(out):,}")

# Bin on launch speed (2 mph) x launch angle (2 deg), restricted to a carry-relevant band.
out = out[out["launch_speed"].between(90, 115) & out["launch_angle"].between(20, 40)].copy()
out["ls_bin"] = (out["launch_speed"] // 2).astype(int)
out["la_bin"] = (out["launch_angle"] // 2).astype(int)
out["resid"] = out["hit_distance_sc"] - out.groupby(["ls_bin", "la_bin"])[
    "hit_distance_sc"
].transform("mean")
out["rho_c"] = out["air_density_kg_m3"] - out.groupby(["ls_bin", "la_bin"])[
    "air_density_kg_m3"
].transform("mean")
out["w_c"] = out["following_wind_mps"] - out.groupby(["ls_bin", "la_bin"])[
    "following_wind_mps"
].transform("mean")
print(
    f"carry band (EV 90-115, LA 20-40), outdoor-with-weather fly balls: n={len(out):,}, games={out['game_pk'].nunique():,}"
)

X = np.column_stack([out["rho_c"], out["w_c"]])
y = out["resid"].to_numpy()
beta, *_ = np.linalg.lstsq(X, y, rcond=None)

# Game-clustered bootstrap CI.
rng = np.random.default_rng(20260930)
games = out["game_pk"].to_numpy()
ug, inv = np.unique(games, return_inverse=True)
idx_by_game = pd.Series(np.arange(len(out))).groupby(inv).apply(np.array).to_list()
boots = []
for _ in range(500):
    pick = rng.integers(0, len(ug), len(ug))
    ii = np.concatenate([idx_by_game[k] for k in pick])
    b, *_ = np.linalg.lstsq(X[ii], y[ii], rcond=None)
    boots.append(b)
lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
print("\nQ5 distance residual ~ density + following wind (within EV/LA bins)")
print(
    f"  ft per 0.01 kg/m^3 density: {beta[0] * 0.01:+.2f}  95% CI [{lo[0] * 0.01:+.2f}, {hi[0] * 0.01:+.2f}]  (physics: negative)"
)
print(
    f"  ft per 1 mph following wind: {beta[1] * 0.44704:+.2f}  95% CI [{lo[1] * 0.44704:+.2f}, {hi[1] * 0.44704:+.2f}]  (physics: positive)"
)
print(
    f"  density SD in sample: {out['air_density_kg_m3'].std():.4f} kg/m^3; following-wind SD: {out['following_wind_mps'].std() / 0.44704:.2f} mph"
)

# Q5b: same, within venue (venue x EV/LA bin demeaning) -- day-to-day variation only,
# so altitude differences between parks cannot drive the density coefficient.
keys = ["venue_name", "ls_bin", "la_bin"]
cnt = out.groupby(keys)["hit_distance_sc"].transform("size")
v = out[cnt >= 5].copy()
for c, src in [
    ("resid_v", "hit_distance_sc"),
    ("rho_v", "air_density_kg_m3"),
    ("w_v", "following_wind_mps"),
]:
    v[c] = v[src] - v.groupby(keys)[src].transform("mean")
Xv = np.column_stack([v["rho_v"], v["w_v"]])
yv = v["resid_v"].to_numpy()
bv, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
ugv, invv = np.unique(v["game_pk"].to_numpy(), return_inverse=True)
ibg = pd.Series(np.arange(len(v))).groupby(invv).apply(np.array).to_list()
bs = []
for _ in range(500):
    pick = rng.integers(0, len(ugv), len(ugv))
    ii = np.concatenate([ibg[k] for k in pick])
    b, *_ = np.linalg.lstsq(Xv[ii], yv[ii], rcond=None)
    bs.append(b)
lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
print(f"\nQ5b within-venue: n={len(v):,}, games={len(ugv):,}, venues={v['venue_name'].nunique()}")
print(
    f"  ft per 0.01 kg/m^3 density: {bv[0] * 0.01:+.2f}  95% CI [{lo[0] * 0.01:+.2f}, {hi[0] * 0.01:+.2f}]"
)
print(
    f"  ft per 1 mph following wind: {bv[1] * 0.44704:+.2f}  95% CI [{lo[1] * 0.44704:+.2f}, {hi[1] * 0.44704:+.2f}]"
)
print(
    f"  within-venue density SD: {v['rho_v'].std():.4f}; within-venue wind SD: {v['w_v'].std() / 0.44704:.2f} mph"
)

# Per-venue wind slope (simple, within venue x bin), sign only -- input to the later per-venue wind check.
rows = []
for name, gv in v.groupby("venue_name"):
    if len(gv) < 500:
        continue
    Xg = np.column_stack([gv["rho_v"], gv["w_v"]])
    b, *_ = np.linalg.lstsq(Xg, gv["resid_v"].to_numpy(), rcond=None)
    rows.append((name, len(gv), round(b[1] * 0.44704, 2)))
pv = pd.DataFrame(rows, columns=["venue", "n", "ft_per_mph_following"]).sort_values(
    "ft_per_mph_following"
)
print("\nQ5c per-venue point estimates (no CI; descriptive only):")
print(pv.to_string(index=False))
