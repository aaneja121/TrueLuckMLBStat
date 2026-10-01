# Park wall heights — review worksheet (v1.2)

Prepared 2026-10-01 by an AI agent (Claude Code) for the maintainer. **This worksheet is not a
human review.** No height enters `park_geometry_v12.py` until the maintainer approves it here,
under the same process as `park_geometry_review_2026-09-29.md`.

**Why:** the v1.2 candidate uses wall height. The v1.2 table has a height for only 86 of 185
points, which covers 29.0% of gated balls (46,218 fly balls and line drives within 20 ft of the
wall or beyond it, 2021–2024). The frozen trainer drops any feature below 50% coverage
(`docs/plans/v1_2_park_weather_plan.md`, "Amendment 2026-10-01").

**Effect if approved** (measured on geometry only, no outcomes read): approving every row
marked O, S2 or S1 below raises gated-ball height coverage to **77.4%** (76.7–78.3% by
season). Also taking every suggestion in "Decisions" gives **88.2%**.

## Evidence tiers

- **O — official:** the team or MLB.com, quoted. Proposed as `maintainer_approved_official_source`.
- **S2 — two independent secondary sources agree.** Proposed as secondary.
- **S1 — one secondary source.** Proposed as secondary. A weaker basis, flagged for that reason.
- **X — conflicting or missing evidence.** Maintainer decision. A suggestion is given where one
  is defensible; otherwise the point stays empty.

## Sources (all retrieved 2026-10-01)

- **Seamheads Ballparks Database** (`seamheads.com/ballparks/ballpark.php?parkID=…`): heights
  at LF / LCF / CF / RCF / RF by year. **It is stale wherever a wall was renovated.** It shows
  Camden at 7 ft after the 2022 raise to 13 ft, Comerica unchanged after 2023, Oracle before
  2020, and loanDepot's 2016 centre field after the 2020 move. Its LCF/RCF are its own
  "left-/right-center" distances, which can differ from our ±22.5° point; this is noted where
  it matters. It is never used against an official figure or across a renovation.
- **Clem's Baseball, Stadium statistics** (`andrewclem.com/Baseball/Stadium_statistics.html`)
  and per-park pages: heights for LF / CF / RF only. Already known to lag renovations.
- Wikipedia, thisgreatgame.com, CBS Sports and Ballpark Pal are quoted where used.

## Proposals (points currently without a height)

### Official source for every point

| Config | Point (stored distance) | Proposed | Tier | Evidence |
|---|---|---|---|---|
| `truist_park_v1` | −45 (335) / −22.5 (385) / 0 (400) / +22.5 (375) / +45 (325) | 6 / 8.67 / 8.67 / 16 / 16 | O | Braves ballpark guide: "Right field 325' 16' 0" Right center field 375' 16' 0" Center field 400' 8' 8" Left center field 385' 8' 8" Left field 375' 8' 8" Left field corner 335' 6' 0"" ([mlb.com/braves/ballpark/information/guide](https://www.mlb.com/braves/ballpark/information/guide)). Seamheads gives RCF 8, which disagrees; the official figure wins. |
| `citizens_bank_park_v1` | −45 (329) / −22.5 (374) / 0 (401) / +22.5 (369) / +45 (330) | 10.5 / 10.5 / 6 / 13.25 / 13.25 | O | Phillies guide: "Right field foul pole: Distance: 330' Height: 13'3" Right field power alley: Distance: 369' Height: 13'3" Center field, straightaway: Distance: 401' Height: 6' Monty's Angle (left of CF to LCF): Distance: 409'-381'-387' Height: 19'-12'8" Left field foul pole: Distance: 329' Height: 10'6" Left field power alley: Distance: 374' Height: 10'6"" ([guide](https://www.mlb.com/phillies/ballpark/information/guide)) |
| `globe_life_field_v1` | all five | 8 | O | Rangers release, 2019-12-04: "The outfield walls will be eight feet in height from the left field foul pole to the right field foul pole." ([mlb.com press release](https://www.mlb.com/press-release/playing-field-information-for-globe-life-field)). Seamheads gives 8 at all five. |
| `wrigley_field_v1` | −22.5 / 0 / +22.5 | 11.5 | O | Cubs: "Height of wall: Bleachers - 11 1/2 feet In corners - 15.0 feet" ([history page](https://www.mlb.com/cubs/ballpark/information/history)). The corners are already stored at 15. |

### Partly official

| Config | Point | Proposed | Tier | Evidence |
|---|---|---|---|---|
| `camden_yards_pre2022` | −22.5 (364) | 7 | O | Orioles: "Prior to the 2022 season, the Orioles renovated the outfield wall from the left field corner to the bullpens in left-center field … Wall height raised from 7 ft." ([ground rules](https://www.mlb.com/orioles/ballpark/ground-rules)). 364 lies in that stretch. |
| `camden_yards_2022_2024` | −22.5 (398, "left-center corner" by the bullpens) | 13 | O | Same source: the 2022 renovation raised the wall to 13 ft from the corner to the bullpens. Seamheads (7) is stale here. |
| `camden_yards_pre2022`, `_2022_2024`, `_2025` | 0 (400) / +45 (318) | 7 / 21 | S2 | Seamheads CF 7, RF 21 (every year). Clem's current table: CF 7, RF 21. The renovations touched only the left-field wall. |
| same three | +22.5 (373) | 7 | S1 | Seamheads RCF 7 only. |
| `fenway_park_v1` | 0 (390) | 17 | O | Red Sox guide: "The center field wall is 17 feet (5.2 meters) high" ([guide](https://www.mlb.com/redsox/ballpark/information/guide)). Clem's text says 18 and Seamheads says 9; the official figure wins. |
| `fenway_park_v1` | +22.5 (380) | 5 | O | Same: "the bullpen fences measure five feet". The bullpens sit in right-center; Seamheads RCF 5. |
| `coors_field_v1` | +45 (350) | 16.5 | O | Rockies 2016 (MLB.com, 2016-03-01): "The fence from the center-field end of the visiting bullpen to the right-field out-of-town scoreboard will be raised 8 feet, 9 inches so that it is consistent with the height of the out-of-town scoreboard at 16 feet, 6 inches" ([article](https://www.mlb.com/news/coors-field-outfield-walls-being-raised-c165840326)). Seamheads RF 16.5. |
| `comerica_park_pre2023` | 0 (420) / +45 (330) | 8.5 / 8.5 | O | Tigers via MLB.com, 2023-01-11: "The center-field wall … will be moved in from 422 to 412 feet, and lowered from 8 1/2 to 7 feet … The right-field wall will be lowered to the same height, from 8 1/2 to 7 feet." ([article](https://www.mlb.com/news/tigers-changing-outfield-dimensions-at-comerica-park)) |
| `loandepot_park_v1` | −45 (344) / +45 (335) | 7 / 7 | O | MLB.com, 2016-01-25: "The fences, however, will be lowered in left and right field from 11 1/2 feet to 7 feet." The 2020 changes moved only centre and right-center ([2016 article](https://www.mlb.com/news/fences-being-moved-in-lowered-at-marlins-park-c162809448); [2020](https://www.mlb.com/marlins/news/marlins-unveil-major-changes-for-ballpark)). |
| `loandepot_park_v1` | −22.5 (386) | 11.5 | O | Same 2016 article: "the out-of-town scoreboard, which extends from the 386-foot marker in left-center to the base of the home run sculpture. That stretch of wall is 100 feet, and it will remain 11 1/2 feet tall." The 386 point is the start of that stretch (a junction). |
| `petco_park_v1` | 0 (396) / +22.5 (391) | 7 / 7 | O + S2 | Padres 2013 (MLB.com): "From the right field porch to the right center field gap, the fence will be moved in 11 feet and lowered to match the sub-eight-foot height in left and center field." Seamheads and Clem both give CF 7. Right-center "matches" it, so 7. Seamheads RCF 10 predates 2013. |

### Secondary sources only

| Config | Point | Proposed | Tier | Evidence |
|---|---|---|---|---|
| `oakland_coliseum_v1` | −45 / −22.5 / 0 / +22.5 / +45 | 8 / 15 / 8 / 15 / 8 | S2 | Seamheads 8/15/8/15/8 (2016–2024). Clem: LF/CF/RF 8; and "a high (15-foot) section in the power alleys where the out-of-town scoreboards are" ([Clem](http://www.andrewclem.com/Baseball/OaklandColiseum.html)). The stored alleys (388) are those power alleys. |
| `busch_stadium_v1` | lines and CF / alleys | 8 / 8 | S2 / S1 | Seamheads 8 at all five (2006–2025). Clem LF/CF/RF 8. |
| `tropicana_field_v1` | lines and CF / alleys | 11 / 9 (CF) / 11 | S2 / S1 | Seamheads 11/11/9/11/11. Clem LF/CF/RF 11/9/11. ⚠ The Rays returned to Tropicana in 2026 after the 2024 roof repairs; any wall change then is unchecked. The approval covers 2021–2024; 2026 scoring needs a separate check. |
| `angel_stadium_v1` | −45 (347) / 0 (396) | 5 / 8 | S2 | Seamheads LF 5, CF 8 (2018+). Clem LF 5, CF 8. |
| `angel_stadium_v1` | +22.5 (370) | 8 | S2 | CBS Sports, 2018-02-20: "the Angels announced that the right-center wall will be lowered from 18 feet to eight feet before the season begins" ([CBS](https://www.cbssports.com/mlb/news/the-shohei-ohtani-effect-angels-to-lower-right-field-wall-angel-stadium-by-10-feet/)). Seamheads RCF 8 (2018+). |
| `angel_stadium_v1` | −22.5 (390) | 8 | S1 | Seamheads LCF 8 only. |
| `yankee_stadium_v1` | −22.5 (399) / 0 (408) | 8 / 8 | S2 | Wikipedia: "The outfield fences measure 8 ft high from the left-field foul pole until the Yankees' bullpen, when the fences begin to gradually descend in height until the right field foul pole". Seamheads 8. |
| `yankee_stadium_v1` | +22.5 (385) | 8 | S1 | Seamheads RCF 8. Wikipedia says the fence descends from the bullpen (right-center) toward the pole, so 8 may be slightly high. |
| `progressive_field_v1` | +22.5 (375) | 9 | S2 | thisgreatgame: "nine-foot walls elsewhere" (other than the 19-ft left-field wall). Seamheads RCF 9. |
| `progressive_field_v1` | −22.5 (370) | 19 | S1 | Seamheads LCF 19. No source says where the 19-ft wall ends. |
| `gabp_v1` | −22.5 (379) / +22.5 (370) | 12 / 8 | S1 | Seamheads LCF 12 (at 379, our distance), RCF 8 (at 370, our distance). |
| `chase_field_v1` | −22.5 / +22.5 (376) | 7.5 | S1 | thisgreatgame: the centre-field section "connects the gaps in center field and is much taller at 25 feet as opposed to the 7.5 feet elsewhere". The "gaps" are the deep 413 ft corners; 376 is the alley on the 7.5 ft wall. Seamheads says 8. The lines are already stored at 7.5. |

## Decisions for the maintainer (X)

1. **Fenway −22.5 (379).** This is the junction where the 37-ft wall meets the 17-ft centre-field
   wall (Seamheads LCF 18). It's the same situation as the Oracle 415 junction. *Suggestion:* 17
   (the centre-field wall), or 37 if a ball at that angle must clear the taller wall.
2. **Fenway +45 (302).** Official: "the right field fence is 3 to 5 feet". Seamheads 3.
   *Suggestion:* 3 (the lower end is at the pole).
3. **Coors −22.5 (390) and 0 (415).** Official 2016: "Five feet is being added to the wall from
   the left-field foul pole to the beginning of the pavilion seating in center." Where the
   pavilion begins isn't stated. Seamheads (updated after 2016: LF 13, RCF 16.5) gives LCF 8,
   CF 8, which would put 390 beyond the raised section. *Suggestion:* 8 and 8 (Seamheads);
   otherwise 13 at 390.
4. **PNC −22.5 (383).** Official: "from a mere six feet in left, to 10 feet by the left-center
   bullpens". Is 383 by the bullpens? Seamheads LCF 10 (at 395). *Suggestion:* 10.
5. **PNC +22.5 (375).** Clem: "From the right-center corner at the 375 mark to the bullpens,
   the wall is 10 feet high"; the 21-ft Clemente Wall ends at that corner. It's a junction.
   *Suggestion:* 21 (a ball at the corner faces the tall wall), or 10.
6. **Angel +45 (350).** Seamheads and Clem's table give 5. But the Angels' current ground rules
   mention "the 18-foot wall located in front of the seating area in right field". It's unclear
   whether that wall reaches the line. *Suggestion:* 5. The 18-ft section is the right-field
   seating wall, not the corner, and the right-center part was lowered in 2018.
7. **Comerica, pre-2023 and 2023–24, −45 and −22.5.** Official: "Left field will be unchanged",
   but no left-field height is given. Seamheads LF 6 vs Clem LF 7. Seamheads' 8.5 is at its
   398 ft left-center point, not our 370. *Suggestion:* leave empty unless a team figure turns up.
8. **Comerica pre-2023 +22.5 (365).** Official: "The massive wall in right-center field above
   the out-of-town scoreboard will by lowered from 13 to 7 feet in height." The 2023 table
   already stores 7 here. *Suggestion:* 13 before 2023.
9. **loanDepot 0 (400) and +22.5 (387).** Both walls moved in 2020; no source gives their new
   height. Clem: "about 13 feet … recedes"; Seamheads is stale. *Suggestion:* leave empty.
10. **Nationals Park, all five.** The sources disagree. Seamheads 8/8/12/12/12; Clem's table
    gives two values per field ("10, 9 / 10, 9 / 16, 9", meaning unexplained); Ballpark Pal
    says the wall "measures 16 feet high across most of right field". *Suggestion:* leave
    empty until a team figure is found.
11. **Oracle −22.5 (399).** The 2020 centre-field rebuild changed this stretch; the only figure
    (Seamheads LCF 11) predates it. *Suggestion:* leave empty.

## Out of scope here

- **Dodger alleys:** still the open decision from the 2026-09-29 handoff (which side of the
  bullpen the ±22.5° points fall). Seamheads gives 8 at its 385 ft points, not our 375.
- **Sutter Health Park and Las Vegas Ballpark (2025–27 configurations):** not used in the
  2021–2024 evaluation. Needed before 2026/2027 scoring; deferred.
- **Adjacent, not changed:** stored heights that sources dispute. Progressive's RF line is
  stored at 9, but Seamheads and Clem's text say ~14 ("the wall in the right field corner is
  taller … about 14 feet"). Coors' LF line is stored at 13, which agrees with the 2016 addition.
  Flagged only.
