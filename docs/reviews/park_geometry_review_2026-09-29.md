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
| `pnc_park_v1` | LF (−45°) distance | 320 ft | **325 ft** | Maintainer, 2026-09-29 | MLB.com PNC Park guide (2026-03-25); Clem |
| `camden_yards_pre2022` | LCF (−22.5°) distance | 376 ft | **364 ft** | Maintainer, 2026-09-29 | FanGraphs (2024-11-18); Clem |
| `progressive_field_v1` | CF (0°) distance | 400 ft | **405 ft** | Maintainer, 2026-09-29 | Ballparks of Baseball; Clem; Guardians history page |
| `petco_park_v1` | RF (+45°) distance | 331 ft | **322 ft** | Maintainer, 2026-09-29 | Yahoo (2013); Clem |
| `wrigley_field_v1` | LF/RF (±45°) corner barrier height | 11.5 ft | **15 ft** | Maintainer, 2026-09-29 | Wikipedia (post-2015, incl. signage); Ballpark Pal says 16 |

**Rule confirmed by the maintainer, 2026-09-29: measured over posted.** Where a posted sign
differs from the true distance or barrier (e.g. Dodger Stadium's 395 signs vs. 400 true
center; American Family Field's 345 RF sign vs. 337 measured), the v1.2 table uses the
**measured, physical value** — the ball meets the real wall, not the sign. For heights this
means the barrier a ball must clear to leave the park, including any structure above the
wall (see Wrigley below).

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

**Decision, maintainer, 2026-09-29: one marker per side at ±22.5°, and it is the
power-alley marker.** Where a park has several markers on a side (a power alley plus a
deeper left-/right-center or corner point), the v1.2 table puts the **power-alley** distance
at ±22.5°. Deeper points are not discarded: they are recorded in the worksheet for a later
version that adds points at sourced angles (option 2). The measured-over-posted rule still
applies to the chosen marker.

Known consequences to resolve per park:

- **Oracle Park:** stored ±22.5° values 399 / 415 are the deep left-center corner and
  Triples Alley, not the power alleys. Seamheads lists the alleys at 364 (left) and 365
  (right). The maintainer's confirmation that 399 is a correct *distance* stands; under this
  rule it moves to the deferred deep-point list, and the ±22.5° values become the alley
  markers once confirmed from a current source.
- **American Family Field:** stored 371 / 374 are already the alley markers (Seamheads
  371 / 371; deeper 382 / 383 go to the deferred list).
- **Every other park:** not yet checked for which marker the stored ±22.5° value is.

## Rows that need a human look

☐ = not yet reviewed. Tick and note what you confirmed and from where.

| ☐ | Config | Stored | Clem | Likely explanation / what to check |
|---|---|---|---|---|
| ☑ distances | `oracle_park_v1` | 339 / **399** / 391 / 415 / 309 | est (365) | **Maintainer, 2026-09-29: LCF 399 confirmed.** All five stored distances match the Giants' 2020 announcement ([Ballpark Digest, 2019-12-13](https://ballparkdigest.com/2019/12/13/giants-unveil-2020-oracle-park-dimensions/)): LF 339, left-center 399 (from 404), CF 391 (from 399), Triples Alley 415 (from 421), RF 309. The agent's "~364" was recalled, not sourced, and is withdrawn. Clem and [Seamheads](https://www.seamheads.com/ballparks/ballpark.php?parkID=SFO03) both omit the 2020 change. **Heights, right side:** stored RCF 20 / RF 24 ft; two sources say 25 / 25 — [Ballpark Pal](https://www.ballparkpal.com/Park-Description.php?VenueId=2395) ("a 25-foot brick wall which takes a nearly vertical line from the shallow '309' foul pole marker to the deep '415' sign in right-center", retrieved 2026-09-29) and Seamheads (RCF wall 25, RF wall 25). **Correction CONFIRMED by the maintainer, 2026-09-29: RCF 25 ft, RF 25 ft** (to be applied in the v1.2 geometry table, not in the frozen v1.0 file — see "Where corrections go" below). **Open:** LF and CF heights (no new source; stored CF 8.5 is a midpoint), and whether the team's deepest left-center point belongs at 22.5° (Seamheads also lists 364 and 378 markers nearer the line — see the systematic issue above). `review_status` unchanged until heights are confirmed. |
| ☑ approved | `pnc_park_v1` | LF **320** | 325 | Agent research 2026-09-29: official guide says "left field, 325 feet; left-center, 383 feet; center field, 399 feet; right-center, 375 feet; right field, 320 feet" ([MLB.com PNC Park guide, 2026-03-25](https://www.mlb.com/news/featured/pnc-park-guide-capacity-seating-chart-parking-and-more)); Clem agrees. **Proposed correction: LF 325.** Other four distances match. RF height 21 (Clemente Wall) matches. |
| ☑ approved | `camden_yards_pre2022` | LCF **376** | 364 | Pre-2022 left-center marker was 364; 2022 moved it to 384 ([FanGraphs, 2024-11-18](https://blogs.fangraphs.com/wall-over-but-the-shoutin-camden-yards-gets-new-dimensions/): "The left-center field marker went out an additional 20 feet, from 364 feet to 384 feet"); Clem agrees. 376 is a later bullpen-edge figure. **Proposed correction: LCF 364.** |
| ☑ approved | `progressive_field_v1` | CF **400** | 405 | Three sources say 405: [Ballparks of Baseball](https://www.ballparksofbaseball.com/ballparks/progressive-field/) ("325-L, 370-LC, 405-C, 375-RC, 325-R"), Clem, and the Guardians' history page per search summary (direct fetch blocked, HTTP 406; it also lists 410 to deep center). **Proposed correction: CF 405.** LF wall 19 ft matches. |
| ☑ approved | `petco_park_v1` | RF **331** | 322 | Multiple sources give 322 down the right-field line, unchanged by the 2013 fence moves ([Yahoo, 2013](https://sports.yahoo.com/petco-park-dimensions-shrink-2013-082013530--mlb.html); Clem). Some listings show 331, origin unclear. **Proposed correction: RF 322** — confirm 331 isn't a measured-vs-posted figure. RF height 8 is right (lowered from 10 to just under 8 in 2013; Clem is pre-2013). |
| ☐ correct | `guaranteed_rate_field_v1` | LF 330, RF 335 | 335, 330 | **No swap.** [Ballparks of Baseball](https://www.ballparksofbaseball.com/ballparks/rate-field/): "330-L, 377-LC, 400-C, 372-RC, 335-R"; Clem transposed. Alleys differ by 2–3 ft (stored 375/375), below the flag threshold. |
| ☐ correct | `loandepot_park_v1` | CF 400, RCF 387 | 407, 392 | Stored values are the 2020 move-in: 407 → 400 in center, right-center moved in 7 ft ([MLB.com: "Marlins moving in fences"](https://www.mlb.com/news/marlins-unveil-major-changes-for-ballpark)). Clem is pre-2020. |
| ☐ correct | `coors_field_v1` | LF height 13 | 8 | 2016 raised the left-field wall from 8 to 13 ft and the right-center segment to 16 ft 6 in ([MLB.com: "Coors Field outfield walls being raised"](https://www.mlb.com/news/coors-field-outfield-walls-being-raised-c165840326)). Stored 13 / 16.5 match; Clem is pre-2016. |
| ☐ correct | `oakland_coliseum_v1` | LCF 388, RCF 388 | 362, 362 | Posted: 330 lines, 388 power alleys, 400 center; 367 is straightaway left/right, between line and alley ([BR Bullpen](https://www.baseball-reference.com/bullpen/Oakland_Coliseum) per search summary). Stored matches. |
| ☑ approved (15) | `wrigley_field_v1` | LF/RF (±45°) height 11.5 | 16 | Corner ivy walls were cut from 15 to 11 ft before 2015, but signs added above keep the front-row-to-field height "still 15 feet" ([Wikipedia](https://en.wikipedia.org/wiki/Wrigley_Field)); [Ballpark Pal](https://www.ballparkpal.com/Park-Description.php?VenueId=17): "the corners of the field are guarded by 16-foot-tall walls on each side." 11.5 ft is the main bleacher wall, not the corners. Under measured-over-posted (the barrier a ball must clear), **proposed correction: LF/RF corner height 15 ft** (the only explicitly post-2015 figure; Ballpark Pal says 16). **Maintainer chose 15 ft, 2026-09-29.** |
| ☑ rule confirmed | `dodger_stadium_v1` | CF 400 | 395 | Posted 395 since 1980, but the signs sit left and right of dead center; true center is 400 (BR Bullpen / Clem per search summary). Stored uses **true** 400 at 0°. Correct under the measured-distance principle below. |
| ☑ checked | `american_family_field_v1` | 344 / 371 / 400 / 374 / 337 | RF 345 | **RF:** posted 345, measured 337; stored uses measured (rule confirmed). **Alleys: no correction.** Stored 371/374 are supported by Wikipedia (371/374, "not posted", i.e. actual distances) and Clem (370/374); [Seamheads](https://www.seamheads.com/ballparks/ballpark.php?parkID=MIL06) lists alley markers at 371/371 and deeper left/right-center markers at 382/383. Ballparks of Baseball's 390/381 matches no other source and is rejected. Which marker belongs at ±22.5° is the systematic alley-angle question. **Minor:** LF 342 (Wikipedia) vs 344 (Clem, Seamheads, stored), 2 ft. **Heights missing in stored record;** Seamheads alone gives 8/8/8/8/6 (LF→RF), single source. |

**Measured vs. posted — a principle to confirm.** Dodger Stadium and American Family Field
both post a distance that differs from the true one, and the stored table already uses the
**true (measured)** figure in both. For a physical model that is the right choice: the ball
meets the real wall, not the sign. Confirming this as the rule also answers part of the
systematic alley question above.

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
- **Camden Yards changed again for 2025** (left-field wall moved in to about 373–374 ft and
  lowered to 8 ft, per [FanGraphs, 2024-11-18](https://blogs.fangraphs.com/wall-over-but-the-shoutin-camden-yards-gets-new-dimensions/)),
  but `camden_yards_2022_2024` has **no end date**, so it would silently cover 2025 onward.
  v1.2 needs an end date on it and a new 2025+ configuration.
- **Venues used after 2024 are absent**, including the Athletics' and Rays' temporary homes.
  Which venues each team used in 2025–2027 must be confirmed from an official source, not
  assumed.
- Any other wall changes made after the 2024 season.
