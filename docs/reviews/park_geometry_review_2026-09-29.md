# Park geometry — human spot-check worksheet

Prepared 2026-09-29 by an AI agent (Claude Code) to make the maintainer's review faster.
**This worksheet is not a human review.** Every record in
`mlb_luck_score.data.park_geometry.PARK_GEOMETRY_POINTS` stays
`agent_sourced_pending_human_review` until the maintainer confirms it; only then does its
`review_status` change, per `RESEARCH_RULES.md`.

Why it matters now: under the v1.2 decision (park is context, not luck), an error in wall
geometry becomes an error in the score (`docs/plans/v1_2_park_weather_plan.md`).

## How this was checked

- **Stored values:** 33 configurations, 30 venues, 165 points (5 per configuration at spray
  angles −45 / −22.5 / 0 / +22.5 / +45°). Mostly sourced from Wikipedia.
- **Second source:** [Clem's Baseball — Stadium statistics](http://www.andrewclem.com/Baseball/Stadium_statistics.html),
  retrieved 2026-09-29. It gives one row per stadium (current or "typical lifetime"
  dimensions), fence heights for LF/CF/RF only, and marks its own estimates in
  parentheses where posted markers are "inaccurate, off-center, or not marked."
- Flagged: any posted-value difference of 3 ft or more, any height difference of 2 ft or
  more, and any Clem estimate differing by 10 ft or more.
- **Clem is itself stale in places.** It still shows Oracle Park's pre-2020 center field and
  Houston's pre-2017 436 ft center field (Tal's Hill). A disagreement is a reason to look,
  not proof the stored value is wrong.
- **Lesson from Oracle:** summary pages (Clem, Seamheads) lag renovations. For each flagged
  park, prefer the team's own announcement or official diagram.
- A rejected source: [ballparksavvy.com/comparisons](https://www.ballparksavvy.com/comparisons/)
  lists, for example, Kauffman's left field as 387 and Petco's as 357 (left-center values in
  the left-field column), so it was not used.

## Where corrections go

`src/mlb_luck_score/data/park_geometry.py` is a **frozen Version 1.0 input**
(`evaluation/v1_final_evaluation_manifest.FROZEN_ARTIFACT_RELATIVE_PATHS`). It is never
edited: doing so would break the sealed 2025 evaluation's hash chain and change what the
production v1.1 near-wall model reads. Every confirmed correction below is recorded here
and applied only in the separately versioned v1.2 geometry table.

**Confirmed corrections for v1.2:**

| Config | Point | Frozen v1.0 value | Corrected value | Confirmed | Sources |
|---|---|---|---|---|---|
| `oracle_park_v1` | RCF (+22.5°) height | 20 ft | **25 ft** | Maintainer, 2026-09-29 | Ballpark Pal; Seamheads |
| `oracle_park_v1` | RF (+45°) height | 24 ft | **25 ft** | Maintainer, 2026-09-29 | Ballpark Pal; Seamheads |

The frozen v1.0 values stay as they are for v1.0/v1.1 scoring. This is a known,
documented data error in the provisional near-wall component, not a silent one.

## Systematic issue — decide this first

**Posted alley distances vs. the 22.5° points.** The table places each park's posted
left-center and right-center marker at exactly ±22.5°. Posted alley markers sit at
different angles in different parks, and Clem's estimates of the true alley distance run
10–20 ft shorter than the posted marker at several parks:

| Park | Stored LCF / RCF | Clem estimate LCF / RCF |
|---|---|---|
| T-Mobile Park | 378 / 381 | (367) / (367) |
| Citizens Bank Park | 374 / 369 | (360) / (355) |
| Petco Park | 390 / 391 | (380) / (372) |
| Yankee Stadium | 399 / 385 | (382) / (360) |
| Oracle Park | 399 / 415 | (365) / (385) |
| Fenway Park | 379 / — | (335) / — |

If the posted markers overstate the wall at 22.5°, the model sees walls farther away than
they are in the alleys — exactly where near-wall and home-run decisions happen. Options:
keep posted values (documented, reproducible) or adopt a documented angle-accurate source.
**This is a maintainer decision, not a per-park correction.**

## Rows that need a human look

☐ = not yet reviewed. Tick and note what you confirmed and from where.

| ☐ | Config | Stored | Clem | Likely explanation / what to check |
|---|---|---|---|---|
| ☑ distances | `oracle_park_v1` | 339 / **399** / 391 / 415 / 309 | est (365) | **Maintainer, 2026-09-29: LCF 399 confirmed.** All five stored distances match the Giants' 2020 announcement ([Ballpark Digest, 2019-12-13](https://ballparkdigest.com/2019/12/13/giants-unveil-2020-oracle-park-dimensions/)): LF 339, left-center 399 (from 404), CF 391 (from 399), Triples Alley 415 (from 421), RF 309. The agent's "~364" was recalled, not sourced, and is withdrawn. Clem and [Seamheads](https://www.seamheads.com/ballparks/ballpark.php?parkID=SFO03) both omit the 2020 change. **Heights, right side:** stored RCF 20 / RF 24 ft; two sources say 25 / 25 — [Ballpark Pal](https://www.ballparkpal.com/Park-Description.php?VenueId=2395) ("a 25-foot brick wall which takes a nearly vertical line from the shallow '309' foul pole marker to the deep '415' sign in right-center", retrieved 2026-09-29) and Seamheads (RCF wall 25, RF wall 25). **Correction CONFIRMED by the maintainer, 2026-09-29: RCF 25 ft, RF 25 ft** (to be applied in the v1.2 geometry table, not in the frozen v1.0 file — see "Where corrections go" below). **Open:** LF and CF heights (no new source; stored CF 8.5 is a midpoint), and whether the team's deepest left-center point belongs at 22.5° (Seamheads also lists 364 and 378 markers nearer the line — see the systematic issue above). `review_status` unchanged until heights are confirmed. |
| ☐ | `oakland_coliseum_v1` | LCF 388, RCF 388 | 362, 362 | Real disagreement on posted alleys. Only matters for 2021–24 (A's left after 2024). |
| ☐ | `coors_field_v1` | LF height **13** | 8 | Stored note says LF raised to 13 ft in 2016. Confirm which wall was raised. |
| ☐ | `wrigley_field_v1` | LF/RF height 11.5 | 16 | Stored note: 11.5 ft bleacher wall. Clem may include the basket. Which height matters for a ball in play? |
| ☐ | `petco_park_v1` | RF 331, RF height 8 | 322, 10 | Check the current posted RF line and wall height. |
| ☐ | `american_family_field_v1` | RF 337 | 345 | Check the posted RF line. |
| ☐ | `guaranteed_rate_field_v1` | LF 330, RF 335 | 335, 330 | **Possible LF/RF swap.** Confirm which line is 330. |
| ☐ | `pnc_park_v1` | LF 320 | 325 | Check the posted LF line (the famous one is RF 320). |
| ☐ | `dodger_stadium_v1` | CF 400 | 395 | Check the posted CF. |
| ☐ | `progressive_field_v1` | CF 400 | 405 | Check the posted CF. |
| ☐ | `camden_yards_pre2022` | LCF 376 | 364 | Check pre-2022 posted LCF (the 2022 change moved left field back). |
| ☐ | `loandepot_park_v1` | CF 400, RCF 387 | 407, 392 | Stored values match the 2020 fence move-in; Clem likely pre-2020. **This is the park where v0.4 calibration regressed** — worth a careful look. |

## Rows where the disagreement is probably Clem being out of date

Worth a glance, but the stored value is probably right.

| Config | Stored | Clem | Why Clem is probably stale |
|---|---|---|---|
| `angel_stadium_v1` | 347 / 390 / 396 / 370 / 350 | 330 / 387 / 404 / 370 / 330 | 2018 reconfiguration (stored note). |
| `camden_yards_2022_2024` | LCF 384, LF height 13 | 364, 7 | 2022 left-field move and raise. |
| `comerica_park_2023_2024` | LF 342, CF 412, heights 7 | 345, 420, 9 | 2023 fence changes. |
| `rogers_centre_2023_2024` | LCF 368, RCF 359, varied heights | 375, 375, 8 | 2023 wall changes. |
| `rogers_centre_pre2023` | heights 10 | 8 | Check which pre-2023 figure is right. |
| `oracle_park_v1` | CF 391 | 399 | 2020 fence move. |
| `minute_maid_park_v1` | CF 409 | 436 | Tal's Hill removed before 2017. |
| `globe_life_field_v1` | 329 / 407 / 374 | 332 / 400 / 381 | Clem's matching row is the *old* Globe Life Park, not Globe Life Field. |

## Rows that agree

Truist, Fenway (posted), GABP (posted), Comerica pre-2023 (posted), Kauffman, Target,
Citi, Yankee (posted), Citizens Bank (posted), T-Mobile (posted), Busch III, Tropicana,
Nationals, Chase.

## Coverage gaps for scoring 2026 and 2027

The table covers 2021–2024 venues only. Before v1.2 can score 2026 or 2027:

- **Kauffman Stadium's configuration ends 2025-12-31**, implying a known wall change for
  2026 that has no record yet.
- **Venues used after 2024 are absent**, including the Athletics' and Rays' temporary homes.
  Which venues each team used in 2025–2027 must be confirmed from an official source, not
  assumed.
- Any other wall changes made after the 2024 season.
