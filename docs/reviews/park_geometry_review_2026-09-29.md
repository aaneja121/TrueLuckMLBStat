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
| `pnc_park_v1` | LF (−45°) distance | 320 ft | **325 ft** | Maintainer, 2026-09-29 | MLB.com PNC Park guide (2026-03-25); Clem |
| `camden_yards_pre2022` | LCF (−22.5°) distance | 376 ft | **364 ft** | Maintainer, 2026-09-29 | FanGraphs (2024-11-18); Clem |
| `progressive_field_v1` | CF (0°) distance | 400 ft | **405 ft** | Maintainer, 2026-09-29 | Ballparks of Baseball; Clem; Guardians history page |
| `petco_park_v1` | RF (+45°) distance | 331 ft | **322 ft** | Maintainer, 2026-09-29 | Yahoo (2013); Clem |
| `wrigley_field_v1` | LF/RF (±45°) corner barrier height | 11.5 ft | **15 ft** | Maintainer, 2026-09-29 | Wikipedia (post-2015, incl. signage); Ballpark Pal says 16 |
| `wrigley_field_v1` | (confirmation) | — | 15 ft corners | Cubs team page | Cubs: "In corners - 15.0 feet" |
| `citi_field_v1` | LC (−22.5°) distance | 358 ft | **370 ft** | Maintainer, 2026-09-29 | Mets team page ("Left Center Field 370"); Ballparks of Baseball ("370-LC") |
| `citi_field_v1` | RC (+22.5°) distance | 375 ft | **380 ft** | Maintainer, 2026-09-29 | Mets team page; Ballparks of Baseball; 2014 move-in 390 → 380 |
| `citi_field_v1` | all five heights | none | **8 ft** | Maintainer, 2026-09-29 | Mets team page ("8 feet consistent from Foul Pole to Foul Pole") |
| `american_family_field_v1` | LF (−45°) distance | 344 ft | **342 ft** | Maintainer, 2026-09-29 | Brewers ground rules; Wikipedia |
| `american_family_field_v1` | LC (−22.5°) distance | 371 ft | **370 ft** | Maintainer, 2026-09-29 | Brewers ground rules ("Left-Field Power Alley: 370 feet"); Clem 370 |
| `american_family_field_v1` | RF (+45°) distance | 337 ft | **345 ft** | Maintainer, 2026-09-29 | Brewers ground rules; Clem; Seamheads; Ballparks of Baseball (337 was Wikipedia-only) |
| `american_family_field_v1` | all five heights | none | **8 ft** | Maintainer, 2026-09-29 | Brewers ground rules ("…/8 feet" at every point); Seamheads' RF 6 ft overruled by the team |
| `guaranteed_rate_field_v1` | LC / RC (±22.5°) distance | 375 / 375 ft | **377 / 372 ft** | Maintainer, 2026-09-29 | White Sox ballpark page; Ballparks of Baseball ("377-LC … 372-RC") |
| `rogers_centre_2023_2024` | LC (−22.5°) distance / height | 368 ft / 11.17 ft | **381 ft / 12.75 ft** | Maintainer, 2026-09-29 | Blue Jays page ("Left Center Power Alley: 381 feet; wall: 12 feet 9 inches"); MLB Trade Rumors; SI |
| `rogers_centre_2023_2024` | RC (+22.5°) distance / height | 359 ft / 14.33 ft | **372 ft / 10.75 ft** | Maintainer, 2026-09-29 | Blue Jays page ("Right Center Power Alley: 372 feet; wall: 10 feet 9 inches"); MLB Trade Rumors; SI |
| `camden_yards_2022_2024` | LC (−22.5°) distance | 384 ft | **398 ft** | Maintainer, 2026-09-29 | Orioles ground rules ("left-center corner: 398 ft."); FanGraphs ("Deep left-center corner: 398") |
| `camden_yards_2022_2024` | effective end date | none | **end of 2024 season** | Maintainer, 2026-09-29 | Orioles: wall changed again "Prior to the 2025 season" |
| `dodger_stadium_v1` | heights | none | **4.5 ft foul pole → bullpens; 8 ft between bullpens** | Maintainer, 2026-09-29 | Dodgers history page (official). Which ±22.5° points fall in each segment depends on the bullpen positions — resolve when building the v1.2 table |
| `dodger_stadium_v1` | distances | 330 / 375 / 400 / 375 / 330 | **unchanged** (true values) | Maintainer, 2026-09-29 | Measured-over-posted: posted 385 / 395 / 385; true 375 / 400 / 375 confirmed by a second source |

**Two rule refinements, maintainer, 2026-09-29 (confirmed against online sources):**

- **A. Several labelled left-/right-center points:** a point the team explicitly calls the
  **power alley** wins (Rogers Centre 381 / 372); otherwise the point the team labels
  **left-/right-center** (Camden 2022: "left-center corner" 398, not the "left field corner"
  384).
- **B. Team figure vs. an unsourced "measured" figure:** without an official or
  independent measured source, the **team figure** is the documented value (American
  Family Field RF 345, not Wikipedia's 337). Measured-over-posted still applies where the
  measured value is independently sourced (Dodger Stadium 375 / 400).

**Withdrawn, 2026-09-29:** Oracle RCF/RF heights 25 / 25 (approved earlier the same day from Ballpark Pal and Seamheads). The official MLB.com Oracle Park guide says "the 24-foot brick wall in right field"; the maintainer chose **24 ft for RF** — the frozen value, so no RF correction is needed. The RCF approval rested on the same rejected claim and is withdrawn; **RCF height is open again** (frozen 20 ft; no official figure found).

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

**Decision, maintainer, 2026-09-29: one marker per side at ±22.5°, the power-alley
marker — defined as the distance the team officially labels left-center / right-center.**
The team's own announcement or official diagram decides which sign that is, consistent
with team sources being the trusted source. Other markers on the same side (shallower
alley signs, deeper corners) are not discarded: they are recorded here for a later version
that adds points at sourced angles (option 2). The measured-over-posted rule still applies
to the chosen marker.

*(An earlier draft of this decision read "power alley" as the shallower alley sign and
proposed moving Oracle to ~364 / 365. The maintainer rejected that reading the same day;
it is withdrawn.)*

Consequences:

- **Oracle Park:** unchanged. The Giants label 399 "left-center" and 415 "right-center
  (Triples Alley)", so the stored ±22.5° values stand, as already confirmed. Seamheads'
  shallower 364 / 365 (and 378 / 384) markers go to the deferred list.
- **American Family Field:** stored 371 / 374; Wikipedia notes these are not posted.
  Needs the Brewers' own left-center / right-center figures to confirm under this rule;
  Seamheads' 382 / 383 go to the deferred list meanwhile.
- **Every other park:** check the stored ±22.5° value against the team's official
  left-center / right-center.

## Rows that need a human look

☐ = not yet reviewed. Tick and note what you confirmed and from where.

| ☐ | Config | Stored | Clem | Likely explanation / what to check |
|---|---|---|---|---|
| ☑ distances | `oracle_park_v1` | 339 / **399** / 391 / 415 / 309 | est (365) | **Maintainer, 2026-09-29: LCF 399 confirmed.** All five stored distances match the Giants' 2020 announcement ([Ballpark Digest, 2019-12-13](https://ballparkdigest.com/2019/12/13/giants-unveil-2020-oracle-park-dimensions/)): LF 339, left-center 399 (from 404), CF 391 (from 399), Triples Alley 415 (from 421), RF 309. The agent's "~364" was recalled, not sourced, and is withdrawn. Clem and [Seamheads](https://www.seamheads.com/ballparks/ballpark.php?parkID=SFO03) both omit the 2020 change. **Heights, right side:** stored RCF 20 / RF 24 ft; two sources say 25 / 25 — [Ballpark Pal](https://www.ballparkpal.com/Park-Description.php?VenueId=2395) ("a 25-foot brick wall which takes a nearly vertical line from the shallow '309' foul pole marker to the deep '415' sign in right-center", retrieved 2026-09-29) and Seamheads (RCF wall 25, RF wall 25). ~~Correction confirmed: RCF 25 ft, RF 25 ft~~ **Withdrawn 2026-09-29:** official MLB.com guide says 24 ft; maintainer chose RF 24 (frozen value stands); RCF open. **Open:** LF and CF heights (no new source; stored CF 8.5 is a midpoint), and whether the team's deepest left-center point belongs at 22.5° (Seamheads also lists 364 and 378 markers nearer the line — see the systematic issue above). `review_status` unchanged until heights are confirmed. |
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
  2026. *Now sourced — see the gap resolution pass below.*
- **Camden Yards changed again for 2025** (left-field wall moved in to about 373–374 ft and
  lowered to 8 ft, per [FanGraphs, 2024-11-18](https://blogs.fangraphs.com/wall-over-but-the-shoutin-camden-yards-gets-new-dimensions/)),
  but `camden_yards_2022_2024` has **no end date**, so it would silently cover 2025 onward.
  v1.2 needs an end date on it and a new 2025+ configuration. *End date approved; the 2025+
  configuration is drafted in the gap resolution pass below.*
- **Venues used after 2024 are absent**, including the Athletics' and Rays' temporary homes.
  Which venues each team used in 2025–2027 must be confirmed from an official source, not
  assumed. *Confirmed in the gap resolution pass below: Sutter Health Park 2025–27, Las Vegas
  Ballpark (six A's games each in 2026 and 2027), Steinbrenner Field 2025, Tropicana Field again
  from 2026.*
- Any other wall changes made after the 2024 season.

## Official alley pass (MLB.com ballpark guides) — in progress

Source: each park's MLB.com ballpark guide,
`https://www.mlb.com/news/featured/<park>-guide-capacity-seating-chart-parking-and-more`,
retrieved 2026-09-29 (pages carry no visible date). Official league/team content only, per
the maintainer. **Caveat: the guides describe the park as it is now (2025–26), not
necessarily as it was in the 2021–24 development seasons.** A difference can be a later
renovation rather than a stored error.

### Official left-center / right-center found — compare with stored ±22.5°

| Config | Stored LF / LC / CF / RC / RF | MLB.com guide (quoted labels) | Result |
|---|---|---|---|
| `busch_stadium_v1` | 336 / 375 / 400 / 375 / 335 | 336; left-center 375; 400; right-center 375; 335 | ✅ match |
| `citizens_bank_park_v1` | 329 / 374 / 401 / 369 / 330 | LF pole 329; LF power alley 374; CF 401; RF power alley 369; RF pole 330 | ✅ match |
| `comerica_park_2023_2024` | 342 / 370 / 412 / 365 / 330 | 342; left-center 370; 412; right-center 365; 330 | ✅ match |
| `coors_field_v1` | 347 / 390 / 415 / 375 / 350 | 347; left-center 390; 415; right-center 375; 350 | ✅ match |
| `globe_life_field_v1` | 329 / 372 / 407 / 374 / 326 | 329; LF power alley 372; CF 407; RF power alley 374; 326 | ✅ match |
| `pnc_park_v1` | 320→**325** / 383 / 399 / 375 / 320 | 325; left-center 383; 399; right-center 375; 320 | ✅ match (with approved LF fix) |
| `yankee_stadium_v1` | 318 / 399 / 408 / 385 / 314 | 318; left-center 399; 408; right-center 385; 314 | ✅ match |
| `truist_park_v1` | 335 / 385 / 400 / 375 / 325 | LF corner 335; "left field" 375; left-center 385; 400; right-center 375; 325 | ✅ match (385 is the labelled left-center) |
| `fenway_park_v1` | 310 / 379 / 390 / 380 / 302 | 310; deep left-center 379; CF 389; right-center triangle 420; right-center 380; 302 | ✅ alleys match; CF 389 vs 390 (1 ft) |
| `dodger_stadium_v1` | 330 / **375** / 400 / **375** / 330 | 330; left-center 385; CF 395; right-center 385; 330 | ⚠ **alleys 385 vs stored 375.** CF 395 is the posted sign; stored true 400 stays under measured-over-posted |
| `petco_park_v1` | 336 / **390** / 396 / 391 / 331→322 | 336; left-center 386; 396; right-center 391; 322 | ⚠ **LC 386 vs 390**; RF 322 confirms the approved fix |
| `guaranteed_rate_field_v1` | 330 / 375 / 400 / 375 / **335** | 330; left-center 375; 400; right-center 375; **right field 330** | ⚠ **RF 330 vs 335** — contradicts Wikipedia and Ballparks of Baseball |
| `kauffman_stadium_v1` (ends 2025) | 330 / 387 / 410 / 387 / 330 | **2026:** "Left and right field, 364"; LC/RC 379; CF 410; fence 8.5 | ℹ new 2026 configuration — the coverage gap, now sourced. *Correction (gap pass below):* 364 is straightaway LF/RF; the foul poles stay 330 |

### Guide gives lines and center only — alleys still need a team source

| Config | Stored LF / CF / RF | MLB.com guide | Result |
|---|---|---|---|
| `angel_stadium_v1` | 347 / 396 / 350 | 347 / 396 / 350 | ✅ |
| `camden_yards_2022_2024` | 333 / 400 / 318 | 333 / 400 / 318 | ✅ lines/CF |
| `citi_field_v1` | 335 / 408 / 330 | 335 / 408 / 330 | ✅ |
| `gabp_v1` | 328 / 404 / 325 | 328 / 404 / 325 | ✅ |
| `target_field_v1` | 339 / 404 / 328 | 339 / 404 / 328 | ✅ |
| `tropicana_field_v1` | 315 / 404 / 322 | 315 / 404 / 322 | ✅ |
| `rogers_centre_2023_2024` | 328 / 400 / 328 | 328 / 400 / 328 | ✅ |
| `wrigley_field_v1` | 355 / 400 / 353 | 355 / 400 / 353 | ✅ |
| `progressive_field_v1` | 325 / 400→**405** / 325 | 325 (19-ft wall) / 405 (9-ft) / 325 (9-ft) | ✅ confirms approved CF fix and stored heights |
| `american_family_field_v1` | **344** / 400 / 337 | **342** / 400 / 345 | ⚠ LF 342 vs 344; RF 345 is posted (stored measured 337 stays) |
| `chase_field_v1` | 330 / 407 / **334** | 330 / 407 / 335 | ⚠ RF 1 ft |
| `nationals_park_v1` | **337** / 402 / 335 | 336 / 402 / 335 | ⚠ LF 1 ft |
| `loandepot_park_v1` | **344** / 400 / 335 | 345 / 400 / 335 | ⚠ LF 1 ft |
| `minute_maid_park_v1` | 315 / **409** / 326 | 315 / **399** / 326 (as Daikin Park) | ⚠ **CF 399 vs 409** — later change or guide error; check dates |
| `t_mobile_park_v1` | 331 / **401** / 326 | 331 / **409** / 327 | ⚠ **CF 409 vs 401** — later change or guide error |
| `oracle_park_v1` | 339 / 391 / 309 | 339 / 391 / 309; "the **24-foot** brick wall in right field" | ⚠ **height conflict:** guide says 24 ft; the approved correction is 25 ft (Ballpark Pal, Seamheads) |

### New venues (2025+), not in the stored table

| Venue | MLB.com guide |
|---|---|
| Sutter Health Park (Athletics) | LF 330, CF 403, RF 325 |
| George M. Steinbrenner Field (Rays) | guide not found at the standard address |

### Conflicts, date-checked (2026-09-29)

| Config | Stored | Official figure | Dated history | Resolution |
|---|---|---|---|---|
| `minute_maid_park_v1` | CF 409 | MLB.com guide 399 | CF 436 → 409 with Tal's Hill removal (announced 2015, done by 2017); no later change found | **Keep 409.** Guide's 399 matches what other listings give as deep left-center. |
| `t_mobile_park_v1` | CF 401 | MLB.com guide 409 | CF 405 → 401 before 2013; no later change found | **Keep 401.** Guide is the outlier. |
| `guaranteed_rate_field_v1` | RF 335; LC/RC 375 / 375 | White Sox ballpark page: "Left Field Line: 330 feet Left Centerfield: 377 feet Centerfield: 400 feet Right Centerfield: 372 feet Right Field Line: 335 feet" ([team page](https://www.mlb.com/whitesox/ballpark/information/guide)); MLB.com guide says 330 / 375 / 400 / 375 / 330 | No change found | **RF: keep 335** (team page beats the guide). **Proposed under team-official rule: LC 377, RC 372** — awaiting maintainer. |
| `dodger_stadium_v1` | 330 / 375 / 400 / 375 / 330 | Dodgers history page: "Left field: 330; Left-center: 385; Center field: 395; Right-center: 385; Right field: 330"; "From foul pole to the bullpens, the outfield fence is 55 inches high (about 4.5 feet). From bullpen to bullpen, the fence is 8 feet high." ([team page](https://www.mlb.com/dodgers/history/ballparks)) | 385 / 395 are posted signs; true CF is 400 (signs sit left and right of dead center); stored 375 alleys are listed elsewhere as "true" | **Rule conflict — maintainer decision:** team-official LC/RC (385) vs measured-over-posted (375). **Heights:** official 4.5 ft (lines to bullpens) and 8 ft (between bullpens) proposed; stored has none. |
| `petco_park_v1` | LC 390 | MLB.com guide 386 | 2013 fence move took left-center 402 → 390 (announcements); no later change found | **Unresolved.** No dated source for 386. Keep 390 until one is found. |

### Team-site pass (official team pages on MLB.com, retrieved 2026-09-29)

| Config | Stored LF / LC / CF / RC / RF (heights) | Official team figure (quoted) | Result |
|---|---|---|---|
| `wrigley_field_v1` | 355 / 368 / 400 / 368 / 353 (11.5 → approved 15 at corners) | Cubs: "Height of wall: Bleachers - 11 1/2 feet In corners - 15.0 feet Distances from plate: Left field - 355 feet Left-center - 368 feet Center field - 400 feet Right-center - 368 feet Right field - 353 feet" ([team page](https://www.mlb.com/cubs/ballpark/information/history)) | ✅ all match; **confirms the approved 15 ft corners** |
| `progressive_field_v1` | 325 / 370 / 400→405 / 375 / 325 | Guardians: "Left field: 325 ft.; left-center: 370 ft.; center field: 405 ft.; right-center: 375 ft.; right field: 325 ft." ([team page](https://www.mlb.com/guardians/history/ballparks)) | ✅ all match (with approved CF fix) |
| `rogers_centre_pre2023` | 328 / 375 / 400 / 375 / 328 (10) | Blue Jays: "previously measuring 328 feet down the foul lines, 375 feet to the power alleys, and 400 feet to dead center field, with 10-foot-high walls" ([team page](https://www.mlb.com/bluejays/ballpark/information/history)) | ✅ all match |
| `camden_yards_pre2022` | 333 / 376→**364** / 400 / 373 / 318 | Orioles: "333 feet to left field, 364 feet to left center, 400 feet to center, 373 feet to right center and 318 feet to right" ([team page](https://www.mlb.com/orioles/ballpark/ground-rules)) | ✅ confirms the approved 364 |
| `american_family_field_v1` | 344 / 371 / 400 / 374 / 337 (none) | Brewers: "Left-Field Foul Line: 342 feet/8 feet Left-Field Power Alley: 370 feet/8 feet Center Field: 400 feet/8 feet Right-Field Power Alley: 374 feet/8 feet Right-Field Foul Line: 345 feet/8 feet" ([ground rules](https://www.mlb.com/brewers/ballpark/ground-rules)) | ⚠ **Proposed: LF 342, LC 370, heights 8 ft all.** RF: official 345 vs stored 337 ("measured", sourced only to Wikipedia) — see question B |
| `citi_field_v1` | 335 / **358** / 408 / **375** / 330 (none) | Mets: "Left Field Foul Pole 335 feet Left Field 358 feet Left Center Field 370 feet Center Field 408 feet Right Center Field 380 feet Right Field 370 feet Right Field Foul Pole 330 feet Height of Wall: 8 feet consistent from Foul Pole to Foul Pole" ([team page](https://www.mlb.com/mets/ballpark/information/guide)) | ⚠ **Stored 358 is the Mets' "Left Field" marker, not "Left Center" (370).** Proposed: **LC 370, RC 380, heights 8 ft all** |
| `rogers_centre_2023_2024` | 328 / 368 / 400 / 359 / 328 (14.33 / 11.17 / 8 / 14.33 / 12.58) | Blue Jays: "Left Center: 368 feet; wall: 11 feet 2 inches … Right Center: 359 feet; wall: 14 feet 4 inches Left Center Power Alley: 381 feet; wall: 12 feet 9 inches Right Center Power Alley: 372 feet; wall: 10 feet 9 inches" (same page) | ⚠ Team labels **both** "Left Center" (368 / 359, as stored) and "Power Alley" (381 / 372) — see question A |
| `camden_yards_2022_2024` | 333 / **384** / 400 / 373 / 318 (13) | Orioles, 2022: "Wall height raised from 7 ft. to 13 ft. Distance from home plate – left field foul line: 333 ft., left field corner: 384 ft., left-center corner: 398 ft." | ⚠ Team labels 384 "left field corner" and **398 "left-center corner"** — see question A |
| *new* Camden Yards 2025+ | — | Orioles, 2025: "Lowering the previous 13-foot wall to 8 feet near the left field foul pole and to 6 feet, 11 inches closer to the left-center bullpens … left field corner: 373 ft., left-center: 363 ft., left-center bullpen 376 ft." | ℹ new configuration; `camden_yards_2022_2024` needs an end date |

**Still no official alley figure** (team pages checked: guide, facts, history, ground rules):
Angel Stadium, Great American Ball Park, Target Field, Nationals Park, Tropicana Field,
T-Mobile Park, Chase Field, Daikin Park, loanDepot park. Their lines and center match the
MLB.com guides (above); their stored alleys stay as they are, unconfirmed.
*Superseded by the gap resolution pass below:* official figures were later found for six of the
nine (T-Mobile, Chase, Nationals, Target, loanDepot and Daikin) and for Angel Stadium's LC.

**Questions for the maintainer:**

- **A. When a team labels more than one left-/right-center point**, which one is the
  power alley? Rogers Centre labels "Left Center" 368 and "Left Center Power Alley" 381;
  Camden 2022 labels "left field corner" 384 and "left-center corner" 398.
  *Suggestion:* a point the team explicitly calls the **power alley** wins (Rogers 381 /
  372); otherwise the point labelled **left-/right-center** (Camden 398).
- **B. American Family Field RF:** official 345 (Brewers ground rules) vs stored 337, whose
  only source for being "measured" is Wikipedia. *Suggestion:* 345 — without an official or
  independent measured source, the team figure is the documented value.

## Gap resolution pass (2026-09-29)

Covers every item left open above. All rows are **proposals awaiting the maintainer**;
nothing here changes a `review_status`. Sources retrieved 2026-09-29.

**New corroborating source: MLB Stats API `fieldInfo`**
(`https://statsapi.mlb.com/api/v1/venues/<id>?hydrate=fieldInfo&season=<year>`). This is the
league's own data, but it is **not reliable on its own**. It still has loanDepot's pre-2020
CF 407 / RC 392, T-Mobile's pre-2013 alleys 390 / 387 and Angel Stadium's pre-2014 signs
389 / 365. For some parks it gives deep-corner values (Coors 420 / 424, Chase 412 / 414,
Tropicana 410 / 404). Use it only to corroborate a team figure, never as the deciding source.

### Petco left-center: 386 or 390 — resolved in favour of 386

| Evidence | Quote / value |
|---|---|
| 2013 move-in ([Ballpark Digest, 2012-10-23](https://ballparkdigest.com/201210235751/major-league-baseball/news/padres-moving-in-petco-park-fences-for-2013)) | "The deepest portion of the left center field gap will be decreased from 402 feet to 390 feet" |
| **2015 move-in** (Padres press release, [MLB.com, 2014-11-06](https://www.mlb.com/padres/news/san-diego-padres-announce-left-field-renovations-for-2015/c-100712862)) | "the padded outfield wall in left field and left center field will be brought in approximately 34"-38" from where it begins in left field to the bullpen entrance in left center field. That same portion of the wall will be lowered to seven-feet tall, from its existing height of eight-feet." |
| MLB.com guide; Stats API (2021–2026) | left-center 386 |

The dated 2015 change explains the gap: 390 minus about 3 ft is about 386–387. **Proposed:**
`petco_park_v1` LC **386** for 2021–24 (the frozen config starts 2013, so strictly it has
been stale since 2015). **Heights:** proposed **7 ft at LC**. Stored LF height is none: the
wall "begins in left field", so it is unclear whether the 7 ft section reaches the −45° point
at the Western Metal building corner. Leave LF open.

### Oracle Park wall heights

| Point | Frozen | Evidence | Proposal |
|---|---|---|---|
| CF (0°) | 8.5 (midpoint) | Giants' 2020 announcement: "The new center-field wall will be seven feet high instead of eight" ([MLB.com, 2019-12-16](https://www.mlb.com/news/giants-to-move-in-outfield-fences-in-2020)). Then, before the 2020 home opener: "The center field wall, which used to be seven feet tall, has been lengthened to 10 feet tall. The change happened after the Giants' exhibition against the Oakland A's last week." ([NBC Sports Bay Area, 2020-07-28](https://www.nbcsportsbayarea.com/mlb/why-giants-raised-oracle-park-center-field-wall-to-10-feet-last-week/1305229/); quote re-verified verbatim 2026-09-29) | **10 ft** for 2021–24. Dated, specific, and it explains the 7 / 8 / 10 spread behind the midpoint |
| LF (−45°) | 8 | Wikipedia 8; Clem 8; Seamheads 8. No team figure found | **Keep 8.** Three sources agree, but none is official |
| RCF (+22.5°, 415) | 20 | Triples Alley is "where the center field and the brick right field wall intersect" ([SFGate, 2019-12](https://www.sfgate.com/giants/article/SF-Giants-new-bullpen-location-at-Oracle-Park-14904163.php)). Wikipedia 20; Seamheads 25 (stale, pre-2020); Ballpark Pal 25 (already rejected for RF) | **Maintainer decision.** The +22.5° point sits on the junction between the ~7 ft CF fence and the 24 ft brick wall, so no single height is correct there. Options: 24 (the brick wall, consistent with RF), the lower CF-fence height, or keep 20 (no source). *Suggestion:* 24. Ballpark Pal describes the brick wall running "to the deep '415' sign", so the 415 point is the brick wall's end |

### Parks that had no official alley figure

| Config | Stored LC / RC | Official figure found (quoted) | Proposal |
|---|---|---|---|
| `t_mobile_park_v1` | 378 / 381 | Mariners, 2012-10-02 ([MLB.com](https://www.mlb.com/news/mariners-revise-outfield-wall-dimensions-in-2013/c-39366456)): "The distance at the left field power alley will decrease from 390-feet to 378-feet … At straightaway center field, the distance will decrease from 405-feet to 401-feet … The distance at the power alley will decrease from 385-feet to 381-feet … the height of the outfield wall will be eight-feet from foul pole to foul pole." | ✅ **LC/RC confirmed.** The dated source also settles the CF conflict: **401 stands.** **Heights 8 ft all** (stored none) |
| `chase_field_v1` | 374 / 374 (RF 334) | D-backs facts page ([team page](https://www.mlb.com/dbacks/ballpark/information/facts-figures)): "Field dimensions : LF: 330' LCF: 376' CF: 407' RCF: 376' RF: 335' Outfield Wall Height : LF: 7'6" CF: 25' RF: 7'6"" | ⚠ **Proposed LC/RC 376, RF 335** (matches the MLB.com guide). **Heights LF 7.5, CF 25, RF 7.5**; alley heights not given |
| `nationals_park_v1` | 377 / 370 (LF 337) | Nationals facts page ([team page](https://www.mlb.com/nationals/ballpark/information/facts-and-figures)): "Left Field: 336 Feet Left-Center Field: 377 Feet Center Field: 402 Feet Right-Center Field: 370 Feet Right Field: 335 Feet" | ✅ **LC/RC confirmed.** ⚠ **Proposed LF 336** (team page, MLB.com guide and Stats API agree) |
| `target_field_v1` | 377 / 367 | Minnesota Ballpark Authority, the public owner ([facts](https://ballparkauthority.com/about/target-field-facts/)): "339' to left; 377' to left field power alley; 404' to center; 367' to right field power alley; and 328' to right. The outfield walls are 8' from the left field foul pole to right center field and 23' from right center field to the right field foul pole." | ✅ **LC/RC confirmed** (the owner, not the team; Stats API agrees). **Heights: LF/LC/CF 8, RF 23.** RC falls on the 8 → 23 boundary: maintainer decision, like Oracle RCF |
| `loandepot_park_v1` | 386 / 387 | 2012 original "386 feet in left-center"; 2016 change left it alone ("the out-of-town scoreboard built into the wall in left-center will not change", [MLB.com, 2015-12](https://www.mlb.com/marlins/news/marlins-park-fences-to-be-moved-in-soon/c-159184524)); 2020: "Center field went from 407 feet to 400 feet, and right-center moved in from 399 feet to 387 feet" ([MLB.com, 2020-07](https://www.mlb.com/news/marlins-think-ballpark-new-dimensions-will-help)) | ✅ **LC 386 and RC 387 confirmed** by dated official articles. LF: the 2015 article says 344 ("Down the line, the distances will remain 344 and 335"), the guide says 345. Keep 344; 1 ft |
| `minute_maid_park_v1` (Daikin) | 366 / 370 | Astros facts page ([team page](https://www.mlb.com/astros/ballpark/information/facts-and-figures)): "Left field - 315 feet Left-center - 366-399 feet Center field - 409 feet Right-center - 370 feet Right field - 326 feet Height of wall: Left field - 19 feet Left-center - 25 feet Center field - 10 feet Deepest point - 10 feet Right-center - 10 feet Right field - 7 feet" | ✅ **RC 370 confirmed; CF 409 confirmed** (settles that conflict). **LC: question C below.** **Heights: LF 19 (as stored), LC 25, CF 10, RC 10, RF 7** |
| `angel_stadium_v1` | 390 / 370 | Angels, 2014-03-31 ([MLB.com](https://www.mlb.com/angels/news/angel-stadiums-wall-markers-get-true-distance-updates/c-70561164)): "Left-center has gone from 387 to 390 … the numbers now read 'true distance'" | ✅ **LC 390 confirmed.** RC 370: no team figure; stays unconfirmed |
| `gabp_v1` | 379 / 370 | Reds FAQ ([team page](https://www.mlb.com/reds/news/great-american-ball-park-dimensions-faq)): "The left-field distance is 328 feet with a 12-foot wall, center field is 404 feet and right field is 325 feet, with both having a wall height of 8 feet." No alleys | Alleys stay unconfirmed; Stats API agrees with 379 / 370. **Heights: LF 12, CF 8, RF 8** |
| `tropicana_field_v1` | 370 / 370 | Nothing found (Rays history page not retrievable; the Stats API values are unusable) | Stays unconfirmed |

**Question C (Daikin Park): the team gives left-center as a range, "366-399 feet".** Rule A
does not cover a range. *Suggestion:* **366**, the stored value. It is the Crawford Boxes
face at the alley. 399 is the deep notch next to center field, which the MLB.com guide
mislabelled as CF. Put 399 on the deferred list for a later version with sourced angles.

### Coverage gaps — configurations to add in v1.2

| New config | Dates | LF / LC / CF / RC / RF | Heights | Source |
|---|---|---|---|---|
| `kauffman_stadium_2026` | from 2026 (v1 ends 2025-12-31) | **330 / 379 / 410 / 379 / 330** | **8.5 all** | Royals history page ([team page](https://www.mlb.com/royals/history/ballparks)): "Foul Poles 330 feet Left Field Corner 347 feet Straightaway Left Field 364 feet Left-Center Gap 379 feet Center Field 410 feet Right Field Corner 344 feet Straightaway Right Field 364 feet Right-Center Gap 379 feet Fence Height 8.5 feet". The MLB.com guide's "Left and right field, 364" is **straightaway** left/right, not the lines (corrected in the table above). Corners 347 / 344 and straightaway 364 / 364 go to the deferred list |
| `camden_yards_2025` | from 2025 (2022–24 config ends with the 2024 season, already approved) | **333 / 363 / 400 / 373 / 318** | LF **8**; LC between 8 and 6.92 (see note) | Orioles ground rules ([team page](https://www.mlb.com/orioles/ballpark/ground-rules)): "Lowering the previous 13-foot wall to 8 feet near the left field foul pole and to 6 feet, 11 inches closer to the left-center bullpens … Distance from home plate – left field corner: 373 ft., left-center: 363 ft., left-center bullpen 376 ft." Lines, CF and RC are not listed as changed. LC 363 is the point the team labels left-center (rule A). *Note:* the LC height is not stated at 363; the wall slopes from 8 ft to 6 ft 11 in. *Suggestion:* 6.92 (6 ft 11 in), the nearer figure, pending an official diagram. Corner 373 and bullpen 376 go to the deferred list |
| `sutter_health_park` (venue 2529) | 2025–2027 (Athletics; [MLB.com press release, 2024-04-04](https://www.mlb.com/press-release/press-release-sutter-health-park-in-west-sacramento-to-host-a-s-for-2025-2027-seasons)) | **330 / 380 / 403 / 380 / 325** | none found | Lines and CF from the MLB.com guide. **Alleys from the Stats API only** (lower confidence: no team, River Cats or guide figure exists; Clem marks them unknown) |
| `las_vegas_ballpark` (venue 5355) | **6 A's home games in 2026 and 6 in 2027** ([MLB.com 2026 schedule](https://www.mlb.com/news/a-s-announce-2026-schedule-including-homestand-in-las-vegas): "June 8-14 at Las Vegas Ballpark") | **340 / 380 / 415 / 380 / 340** | none found | Stats API and Wikipedia agree; the Aviators' facts page gives no dimensions. **A venue the earlier gap list missed** |
| `steinbrenner_field` (venue 2523) | 2025 only | 318 / ? / 408 / ? / 314 | none found | MLB.com guide: "Left field, 318 feet; center field, 408 feet; right field, 314 feet". Alleys conflict: MLB.com calls the field "identical to Yankee Stadium" (which would mean 399 / 385), Seamheads gives 385 / 360, the Stats API has none. **Needed only to score 2025, which is sealed**, so it does not block 2026–27; left open |
| Tropicana Field return | 2026– | reuse `tropicana_field_v1` (no end date) | — | Rays returned for the 2026 home opener ([SI](https://www.si.com/mlb/rays/onsi/news/rays-return-to-tropicana-field-in-long-awaited-2026-home-opener)); the 2026 upgrades article ([MLB.com, 2025-11-12](https://www.mlb.com/news/rays-announce-tropicana-field-upgrades-ticket-information-2026)) names no field-dimension change. Unconfirmed: whether the rebuilt walls kept their exact positions |

**Still open after this pass:** Oracle RCF height, Target RC height and Daikin LC (the three
maintainer decisions above); Camden 2025 LC height; Petco LF height; Angel RC,
GABP LC/RC and Tropicana LC/RC (no team figure exists online); Steinbrenner alleys
(2025 only). Not yet checked: one-off neutral-site games in 2025–27 (e.g. special-event
venues), which need a schedule check against the Stats API venue IDs.

### Maintainer decisions on the gap pass (2026-09-29)

| Question | Decision | Reason |
|---|---|---|
| Oracle RCF (+22.5°, 415) height | **24 ft** | The 415 point is the end of the brick right-field wall; consistent with the RF 24 ft decision |
| Target Field RC (+22.5°) height | **8 ft** | The owner: walls are "8' from the left field foul pole to right center field", read as including right-center; 23 ft starts after it |
| Question C, Daikin LC | **366 ft** (stored value) | Crawford Boxes face at the alley; 399 (deep notch beside center) goes to the deferred list |

The other rows in the gap pass remain **proposals** until the maintainer approves them.
