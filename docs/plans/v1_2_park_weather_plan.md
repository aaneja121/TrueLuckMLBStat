# Plan: Version 1.2 — park and weather in the contact model

Status: **PLAN — nothing here is authorized or built.** Drafted 2026-09-29.
Any adoption is a **redefinition of the score**, not a bug fix (`RESEARCH_RULES.md`,
"Never silently redefine the Luck Score").

## Where things stand today (verified 2026-09-29)

- **The contact model has no park or weather information.** Its features are
  `launch_speed`, `launch_angle`, `spray_angle_approx`, `hit_distance_sc`, `bb_type`,
  `stand`, `venue` (`features/build_contact_features.py`).
- **`venue` is listed but always empty.** Raw Statcast has no `venue` column;
  `clean_batted_balls.OPTIONAL_COLUMNS` adds it as all-null. Measured: 0% filled across all
  494,173 development batted balls, and absent from the 2026 raw files. **`PRODUCT.md` says
  the model uses venue; it does not in practice.** That doc line needs correcting
  independently of this plan.
- **The near-wall defensive model does use park geometry.** The outfield opportunity
  model is gated: balls within about 20 ft of the wall go to `near_wall_hgb_v07c`, which adds
  `wall_height_in_spray_direction`, `projected_distance_to_wall_margin` and
  `wall_segment_label`. It feeds the defensive-execution component and is labeled
  provisional. Selected on 2023 (log loss 0.369 vs. logistic 0.406).
- **Weather is not used anywhere in production.**

## What was tried before, and why it failed

| Version | Idea | Result | Why not adopted |
|---|---|---|---|
| v0.3 | `venue_id` as a category | Real park signal (Coors shifts HR → XBH, consistent with its deep outfield) | Material calibration regression elsewhere |
| v0.4 | Wall geometry per spray direction | Large, bootstrap-confirmed log-loss and near-wall calibration gain | Material regression at loanDepot park |
| v0.5 | Weather (basic, vector) | Passed all 8 automated criteria, tiny real log-loss gain | Physical-plausibility check found weak, partly wrong-signed attribution |
| v0.5.1 | Density-only, density anomaly, components | Density variants not significant on 2024; components significant | Components failed perturbation checks (backwards density and wind) |

The pattern: when the model learns park and weather effects freely from observational
data, it finds real aggregate signal but attaches it to the wrong mechanism. Weather is
confounded with park, season timing and roof state; park identity lets the model memorize
venue quirks unevenly.

## Decision 0 — before any code: what should luck mean?

A wall-scraper that is a home run in a small park and an out in a big one.

- **Option A, park-neutral (today's definition).** "Did the ball deserve better, *anywhere*?"
  The small-park homer is favorable luck. Park is part of the luck.
- **Option B, park-specific.** "Did the ball deserve better, *in that park, that day*?"
  The homer is expected. Park and weather are context, not luck.

The same question applies to weather: is a wind-aided homer luck?

Neither is wrong. B makes the number closer to "the outcome the contact earned in its real
conditions"; A keeps it "how the result compared to the contact itself." **This is the
maintainer's decision and it must be recorded, with reasons, before development starts.**
If A is kept, this plan reduces to "no contact-model change" and the work below applies
only to the defensive components.

### Decision record

**Park: Option B (park-specific).** Decided by the maintainer on 2026-09-29, before any
v1.2 development. Reason, in the maintainer's words: park effects should be "considered as
part of what that contact earns in the park considering every park is different," because
it "tells the actual story" batted ball by batted ball.

Consequences accepted with this decision:

- Park effects leave the luck number. A hitter's home park no longer registers as luck.
- v1.2 scores are not directly comparable with v1.1 (park-neutral) scores. Adoption needs
  its own `scoring_version`, separate artifacts, and public methodology copy saying the
  definition changed.
- Errors in park geometry become errors in the score, so the human geometry spot-check
  (Data preconditions) is a hard precondition, not a nicety.

**Weather: Option B (conditions-specific), same logic as park.** Decided by the
maintainer on 2026-09-29, before any v1.2 development. Reason: weather "is an everyday part
of MLB games," so outside a dome "the logic shouldn't be any different from that of the
parks." The maintainer flagged roofed stadiums as the case needing separate logic.

Roof handling starts from the existing rules (`classify_roof_status`, `RESEARCH_RULES.md`
"Never fabricate weather, roof status, or indoor climate conditions"):

- **Outdoor:** measured weather drives the carry adjustment.
- **Fixed dome / retractable roof explicitly closed:** wind = 0 (certain). Temperature,
  humidity, pressure and density stay null; no invented indoor climate.
- **Roof status unknown:** never guessed.

Open development questions (do not block the decision):

1. **Altitude still applies indoors.** A roof removes wind and controls temperature but not
   elevation (e.g. Chase Field, retractable, ~1,100 ft). Options: an elevation-based
   pressure term plus cited per-venue indoor climate data (new reviewed data required), or
   no density adjustment indoors.
2. **Unknown roof status.** Measure how many batted balls it affects, then choose: leave
   those games unadjusted, or find a documented roof-state source.

## Approach, if Option B is chosen

Constrain the model with known physics instead of letting it learn park and weather freely.

1. ~~**Carry adjustment from physics, not from fitting.**~~ **Withdrawn 2026-09-30 by the
   maintainer — see "Step 1 re-scope" below.** The measured `hit_distance_sc` already
   reflects the real conditions, so adjusting it for weather would double-count under
   Option B. Physics (Kagan & Nathan trajectory model) is kept only to set the expected
   direction of the perturbation checks, never as a model input.
2. **Geometry only where it can matter.** Apply wall geometry to balls whose adjusted
   projected distance is near or beyond the wall in that spray direction, extending the
   gating idea the near-wall defensive model already uses. Open-field balls are
   bit-identical to v1.1.
3. **No bare venue identity.** v0.3 showed it memorizes venue quirks unevenly.
4. **Monotonic constraints** where a direction is physically certain (farther wall → more
   catchable; more carry → more likely to leave), where the model class supports them.
   **Checked 2026-09-30: not available.** On the installed scikit-learn 1.9.0,
   `HistGradientBoostingClassifier(monotonic_cst=...)` raises "monotonic constraints are not
   supported for multiclass classification" when fitted on a three-class target, and
   multinomial logistic regression has no such option. The perturbation checks carry the
   physical-direction burden instead.
5. **Airport wind is not in-park wind.** Weather comes from nearby ASOS stations (airports),
   not from inside parks. Some parks were designed to block or redirect wind. Oracle Park
   is the documented example: its forecast wind blows out to center 98% of the time, yet
   "the tall flags overlooking McCovey Cove can often be seen blowing back toward home
   plate" and "many players have discussed in-blowing winds at Oracle despite the
   out-blowing forecast" ([Ballpark Pal](https://www.ballparkpal.com/Park-Description.php?VenueId=2395),
   retrieved 2026-09-29). Applying station wind to that park feeds the model the wrong
   direction, and correct physics on wrong wind could look "backwards" — **a plausible
   contributor to v0.5/v0.5.1's wrong-signed wind results, to be tested, not assumed.**
   Consequences for the design:
   - Treat the **wind term per venue as unproven** until station wind is shown to relate
     to in-park carry there (on development seasons only). Where it cannot be shown, the
     wind term is off for that venue, documented.
   - Temperature and air density travel from station to park far better than wind;
     keep them separate from wind in every check so one cannot mask the other.
   - Never invent in-park wind. If no in-park source exists, the documented fallback is
     "no wind adjustment", not a guess.
6. **Roof and indoor games** follow existing rules: an explicit "Roof Closed" string only,
   wind 0 when closed, climate otherwise null. No invented indoor climate.

Every derived feature must be recomputed whenever its source is overridden in
perturbation checks (`RESEARCH_RULES.md`, "Any override of a raw feature must recompute
every feature DERIVED from it").

## Step 1 re-scope (decided 2026-09-30)

**Decision (maintainer, 2026-09-30):** do not adjust `hit_distance_sc` for weather. Under
Option B, weather is context, and the contact model's measured distance already contains
it. The v1.2 core is park geometry applied to the measured distance. A published
trajectory model is used only for the perturbation checks' expected directions:
D. Kagan and A. M. Nathan, "Statcast and the Baseball Trajectory Calculator," *The
Physics Teacher* (<https://baseball.physics.illinois.edu/Statcast-TPT.pdf>), with Nathan's
calculator (<https://baseball.physics.illinois.edu/trajectory-calculator-new3D.html>).
**Update 2026-09-30:** the Kagan & Nathan PDF is a scanned image, so its text could not
be read or quoted. Physics is only needed for *directions*, and the direction source used
instead is verified text: drag and lift both scale with air density (Nathan, "Analysis of
Baseball Trajectories," 2017, <https://baseball.physics.illinois.edu/TrajectoryAnalysis.pdf>).
Nathan's Physics of Baseball page (<https://baseball.physics.illinois.edu/aero.html>,
retrieved 2026-09-30) also says: "an increase of temperature by 1-deg F increases fly ball
distances by about 0.33 ft." That is about 3.3 ft per 10 °F, which matches the step 1
within-venue estimate (about 3.5 ft).

**Evidence: descriptive check on 2021–2024 development data only** (486,443 eligible
batted balls; no 2025/2026 read; nothing fitted for model use;
`scripts/v1_2_hit_distance_check.py`, bootstrap seed 20260930):

| Question | Result |
|---|---|
| Coverage of `hit_distance_sc` | 99.9–100% for every batted-ball type, outcome and season |
| Home runs: projected landing point? | Yes. 97.4% land beyond the recorded wall (median 29 ft beyond); 0.7% are more than 10 ft short (geometry or spray-angle error, not investigated) |
| Deep air-ball outs (within 30 ft of the wall) | n=24,451. 11.2% sit beyond the recorded wall, 2.2% by more than 10 ft. **Unresolved:** we can't tell whether that is the catch point plus geometry/spray error, or a projected landing point for robbed homers |
| Hits off the wall | **Inconclusive.** Play descriptions almost never say "wall" (7 of 23,768), so this can't be isolated this way |
| Does distance carry the density effect? | Yes, in the physical direction. Within launch-speed × launch-angle bins, outdoor fly balls (EV 90–115, LA 20–40; n=45,554, 7,360 games): −1.29 ft per +0.01 kg/m³ (game-clustered 95% CI −1.33 to −1.26). **Within venue** (day-to-day only, so altitude can't drive it; n=43,576): −1.45 ft (−1.52 to −1.37). A 10 °F change is about 0.024 kg/m³, or about 3.5 ft, close to the widely quoted ~3 ft per 10 °F |
| Does distance carry the station-wind effect? | Positive but small: +0.25 ft per mph following wind within venue (CI +0.21 to +0.28). Per-venue point estimates range from −0.52 (Oakland) to +0.95 (Wrigley). This fits airport wind being a noisy proxy for in-park wind (approach item 5). It is descriptive only: no per-venue CIs yet, and it doesn't prove anything about in-park wind |

**Consequences for the plan:**

- The per-venue wind check (next step 2) no longer feeds a carry term. It becomes a
  perturbation-check question: where station wind can't be shown to matter, that venue's
  wind-direction check is reported as not applicable, not as a pass.
- Correlation, not causation: the density coefficient sits in the physical direction and
  near the published magnitude, but it's an observational association with other things
  held only partly fixed (bins, venue).
- **Resolved 2026-09-30: for a caught ball it is the catch point.** MLB's Statcast
  glossary, "Hit Distance (DST)" (<https://www.mlb.com/glossary/statcast/hit-distance>,
  retrieved 2026-09-30): "Hit Distance represents the distance away from home plate that a
  batted ball lands -- whether by hitting the ground, the seats, the wall or a fielder's
  glove." It is a different metric from Projected Home Run Distance (flight "unhindered by
  obstructions"). So a home run's distance is its seat landing point, and a robbed homer
  reads at the glove, near the wall. The deep outs "beyond the wall" in the table above are
  therefore glove points, and may include reach-over catches or geometry/spray error. The
  glossary also says Statcast "can record Hit Distances at the moment a ball touches the
  ground or where a ball ultimately ends up". That ambiguity is recorded, not resolved.
- Adjacent, not chased: because a caught ball's distance is the glove point, the feature
  partly reflects where the fielder made the play. It has been a baseline feature since
  v0.1; flagged only.

## Step 2: per-venue station-wind check (rule frozen 2026-09-30, before any per-venue CI)

**Question:** at each venue, does station (ASOS) wind relate to in-park carry? The answer
decides only whether that venue's wind-direction perturbation check applies. It is never a
model input, and it chooses no model.

**Disclosure:** step 1 already printed per-venue point estimates (no CIs) before this rule
was written. The rule below reuses existing repository conventions and was not chosen to
produce any particular venue's result. 2024 was used by v0.5/v0.5.1 for weather and is
not pristine.

**Frozen rule** (`mlb_luck_score.models.venue_wind_support`):

- **Rows:** development seasons 2021–2024 only; `eligible_for_training`; `bb_type ==
  fly_ball`; `hit_distance_sc`, `air_density_kg_m3` and `following_wind_mps` all present;
  roof status `outdoor_open_air` or `retractable_roof_open` (closed or indoor games have
  no station wind to test); launch speed 90–115 mph and launch angle 20–40° (the step 1
  carry band).
- **Estimator, per venue (`venue_id`):** OLS of `hit_distance_sc` on following wind and
  air density, with fixed effects for 2 mph × 2° launch-speed/launch-angle cells
  (within-cell demeaning). Cells with fewer than 5 rows are dropped. The wind coefficient
  is reported as ft per mph.
- **Interval:** game-clustered bootstrap, 500 replicates, seed 42, 95% percentile CI,
  following the `compare_near_wall_calibration_gate` convention.
- **Support minimum:** at least 100 plays and 20 games (reusing `MIN_SUBGROUP_PLAYS` and
  `MIN_SUBGROUP_GAMES`).
- **Three-way status** (v0.7D pattern; insufficient evidence is never a pass):
  - `wind_supported`: adequate support and the whole CI is above 0. The venue's
    wind-direction perturbation check **applies**.
  - `wind_contradicted`: adequate support and the whole CI is below 0. Station wind points
    the wrong way there. The check is **not applicable**, and this is reported as a
    finding.
  - `insufficient_evidence`: below the support minimum, or the CI straddles 0. The check is
    **not applicable**.
- **Multiplicity (reported, not gating):** with about 28 venues, 95% intervals would give
  about 0.7 venues falsely `wind_supported` by chance alone. A Bonferroni sensitivity
  column (a normal approximation from the bootstrap SE) is reported alongside.
- Only the direction is tested. A supported venue's slope says nothing about in-park wind
  magnitude.

### Step 2 result (computed 2026-09-30, code `d298d26`)

Command: `.venv/bin/python -m mlb_luck_score.models.venue_wind_support`. Output:
`outputs/tables/v1_2_venue_wind_check.csv` (gitignored). 29 venues in total. Slopes are ft per
mph of station wind blowing out, with the 95% game-clustered CI.

| Status | Venues |
|---|---|
| `wind_supported` (15) | Wrigley +0.95 [0.80, 1.10]; PNC +0.50; Citizens Bank +0.47; **Oracle +0.46 [0.21, 0.91]**; Kauffman +0.43; Camden Yards +0.40; Angel +0.38; Comerica +0.38; Fenway +0.36; Nationals +0.31; Guaranteed Rate +0.25; Great American +0.22; Target +0.21; Yankee +0.17; Busch +0.15 |
| `wind_contradicted` (5) | Oakland −0.52 [−0.73, −0.28]; Petco −0.39 [−0.60, −0.16]; loanDepot −0.51 (n=152); American Family −0.27; Rogers Centre −0.22 |
| `insufficient_evidence` (9) | Citi, Coors, Dodger, Progressive, T-Mobile, Truist, Chase (n=295); Globe Life (n=22) and Minute Maid (0 rows after the cell minimum): too few roof-open games |

**Bonferroni sensitivity (reported, not gating):** 9 venues stay supported (Wrigley, PNC,
Citizens Bank, Kauffman, Camden Yards, Angel, Comerica, Fenway, Guaranteed Rate). 2 stay
contradicted (Oakland, Petco). Oracle, Busch, Yankee, Target, Great American and Nationals
drop to insufficient evidence under the correction.

**Reading it:**

- **Oracle Park is supported, not contradicted.** This cuts against the hypothesis (approach
  item 5) that station wind points the wrong way there. Its interval is the widest among the
  supported venues, and it doesn't survive Bonferroni.
- The contradicted venues are Oakland and Petco, which survive the correction, plus three
  roofed parks with roof open (loanDepot, American Family, Rogers Centre), which don't. The
  roofed-park result fits open-roof wind not reaching the field the way airport wind
  suggests, but that isn't tested.
- Every slope is 1 ft/mph or less. Any physical carry effect is presumably larger, so
  station wind is a heavily attenuated proxy even where the direction holds. The magnitude
  is unusable; only the direction is used.
- These are observational associations. Game-level wind can travel with other conditions
  that aren't controlled (only launch cell and density are held fixed).
- **Consequence:** the v1.2 wind-direction perturbation check applies at the 15 supported
  venues under the frozen rule. Every other venue reports "not applicable", never a pass.

## Data preconditions

- **v1.2 gets its own versioned geometry table — built 2026-09-29:**
  `mlb_luck_score.data.park_geometry_v12` (tests: `tests/test_park_geometry_v12.py`).
  `park_geometry.py` is a frozen v1.0 input and is never edited; the v1.2 table imports it,
  applies the maintainer-approved corrections from
  `docs/reviews/park_geometry_review_2026-09-29.md` as sourced overrides, and adds the
  2025–27 configurations. Nothing in scoring reads it yet.
- Geometry records are `agent_sourced_pending_human_review`. **A human spot-check of the
  30 venues is required before adoption**, recorded honestly in `review_status`.
- Weather comes only from the MLB Stats API per-game field and the IEM ASOS archive.
- 2026 geometry and weather must be joinable for prospective scoring (confirm coverage,
  including any venue or wall changes after 2024).
- Any per-venue baseline is fit on `TRAIN_SEASONS` only.

## Evaluation design (frozen before any metric is computed)

- **Development seasons: 2021–2024 only.** 2025 sealed; 2026 never used for a decision.
- **2024 is not pristine** for park and weather: v0.3–v0.5.1 were judged on it. Report 2024
  results, but the plan should not rest on them. Prefer season-level cross-validation
  within 2021–2023 (fit two seasons, evaluate the third) for model decisions, with 2024 as
  a confirmation that is disclosed as previously used.
- **Adoption requires ALL of, stated in advance:**
  - log-loss improvement with a game-clustered bootstrap CI excluding zero;
  - no venue or subgroup credibly `not_calibrated` under the v0.7D three-way gate
    (`calibrated` / `not_calibrated` / `insufficient_evidence`; insufficient evidence is
    never a pass);
  - controlled-perturbation checks passing in the physical direction: Coors-like thin air
    increases carry; wind out increases HR probability and wind in decreases it; a farther
    wall reduces HR probability for the same ball;
  - no open-field regression (bit-identical by construction).
- **Candidates stay candidates.** A `recommend_*` rule is input to a maintainer decision,
  never the decision.

The outline above (2026-09-29) is kept for the record. The step 3 specification below
replaces it (approved and frozen 2026-09-30).

### Step 3: v1.2 evaluation specification — FROZEN 2026-09-30

**Status: frozen 2026-09-30.** The maintainer approved decisions **[D1]–[D4]** exactly as
proposed on 2026-09-30, before any v1.2 model was fitted or any v1.2 metric computed. From
here, a change to this specification is a new decision: record it here, with its date and
reason, before the result it would affect is looked at.

**One candidate, declared in advance: `gated_geometry_v12`.** A single candidate keeps
the number of looks small (v0.4 and v0.5 each compared several).

- **Gate (which rows use geometry)** [D3]: v1.2 geometry resolved (`geometry_status ==
  ok` from `park_geometry_v12`), `bb_type` in {`fly_ball`, `line_drive`}, and
  `hit_distance_sc − wall_distance_in_spray_direction ≥ −20 ft` (every ball at or beyond
  the wall is included). The 20 ft band reuses the near-wall defensive model's existing
  convention and was not chosen from v1.2 data. Gate membership uses only pre-outcome
  inputs the baseline already reads.
- **Specialist for gated rows:** multinomial logistic regression, the same model class and
  preprocessing as `baseline_v02` (`train_model`, `class_weight=None`). It adds exactly
  these features: `wall_distance_in_spray_direction`, `wall_height_in_spray_direction`
  (missing heights go through the existing impute + missing-category path),
  `projected_distance_to_wall_margin`, `wall_margin_x_launch_angle` and
  `launch_angle_x_wall_height`. It is trained on the gated rows of the training seasons only.
- **Everything else:** the `baseline_v02` prediction, bit-identical.
- **Never included:** venue identity (v0.3), and any weather column (step 1: measured
  distance already carries the weather).
- **Comparator:** `baseline_v02`, retrained on exactly the same training rows.

**Evaluation scheme** [D1]:

- **Primary:** leave-one-season-out within 2021–2023 (three folds: fit on two seasons,
  predict the third). Out-of-fold predictions are pooled over 2021–2023. These seasons
  have been training data for earlier versions but never the evaluation set for a park or
  weather decision.
- **Confirmation:** fit on 2021–2023, evaluate 2024. Disclosed as previously used. It can
  block adoption (a credible regression on 2024 blocks) but cannot establish it.
- **Intervals:** game-clustered bootstrap, 500 replicates, seed 42, 95%. These are the
  repository's conventions, not new choices.

**Adoption criteria. ALL must hold on the primary pool, stated before any computation:**

1. **Log loss:** the paired candidate-minus-baseline log-loss CI over all eligible rows sits
   entirely below 0. The gated-rows-only delta is also reported.
2. **Calibration:** overall ECE and home-run ECE do not materially worsen under the
   existing v0.3 material-regression rule (`compare_park_aware`, absolute margin 0.01 AND
   relative margin 50%).
3. **No credible venue or subgroup regression** [D2]: for every venue, and for the
   subgroups wall height (short/medium/tall/unknown), spray sector, `bb_type`, the
   0–5/5–10/10–20 ft bands short of the wall and beyond the wall, the paired log-loss delta
   CI must not sit entirely above 0 (support minimums: 100 plays, 20 games). This is the
   v0.7D `not_calibrated` rule, part (b).
   **Deviation from the outline:** v0.7D's absolute-ECE part (a) is reported for both models
   side by side, but it is not gating. At near-wall sample sizes it flagged 19 groups as
   `not_calibrated` even for a model that credibly improved on its baseline (v0.7D, real
   2024 data). Under v0.7D's own rules, a gate that the baseline itself would fail can't
   tell a good candidate from a bad one. `insufficient_evidence` is still never reported as
   a pass.
4. **Physical-direction perturbation checks** [D4], on the gated rows, home-run
   probability. Each check recomputes every derived column (margin and both interaction
   terms). Gate membership is held at its unperturbed value.
   - (a) Farther wall: wall distance +10 ft vs −10 ft → P(HR) lower.
   - (b) Taller wall: height 16 ft vs 8 ft, on rows with a known height → P(HR) lower.
   - (c) Longer ball: `hit_distance_sc` +10 ft vs −10 ft → P(HR) higher. This is where
     weather physics now enters: thinner air or a tailwind shows up as a longer measured
     distance (step 1; Nathan).
   The aggregate direction is required for each check. Per-venue directions are reported,
   and at the 15 `wind_supported` venues check (c) is labelled as also standing in for wind.
   **Deviation from the outline:** its "Coors thin air" and "wind in/out" checks cannot apply
   to a model with no weather inputs. They are replaced by (c) plus the diagnostic below.
5. **Open field unchanged:** a test asserts that every non-gated row's predicted
   probabilities are bit-identical to `baseline_v02`'s.

**Reported, never gating:**

- **Gate-boundary continuity:** mean |specialist − baseline| P(HR) for rows within 2 ft of
  the −20 ft boundary. A jump there means a ball's expectation changes because it crossed
  an arbitrary line.
- **Weather absorbed?** Within-venue association of the per-play home-run residual
  (indicator minus P(HR)) with air density (all outdoor/roof-open venues) and following
  wind (the 15 `wind_supported` venues only), for both models, with CIs. Under Option B
  this should sit near zero. It is not gating because it is observational and confounded
  (step 1).
- Geometry coverage of gated rows by venue and season, and the count of rows whose wall
  height is unknown.

**Output:** a `recommend_v12_adoption`-style summary. It is input to the maintainer's
decision, never the decision.

**Amendment 2026-10-01: wall-height precondition (before any v1.2 model was fitted or
outcome read).** Building the candidate showed that the v1.2 table gives a wall height for
only 29.0% of gated balls (46,218 gated, 2021–2024). 19 of 30 venues have none, and only 86
of 185 points carry a height. Two consequences:

- The frozen trainer (`select_available_features`, `MIN_NON_NULL_FRACTION = 0.5`) would
  drop both height features without warning.
- The spec wrongly said missing heights take a "missing category". Numeric features are
  median-imputed with no indicator.

**Maintainer decision (2026-10-01):** source the missing heights first, through the same
proposal-and-approval review as the distances. The evaluation waits for that review. The
five-feature candidate is unchanged. Coverage of gated rows is re-measured after the review
and recorded here before anything is fitted.

**Height review done (2026-10-01).** The maintainer approved 75 heights from
`docs/reviews/park_wall_height_review_2026-10-01.md`: all official, two-source and
one-source rows, plus the suggested values for eight contested points. Eleven contested
points stay empty (Comerica LF/LC, loanDepot CF/RC, Nationals Park, Oracle LC), as do the
Dodger alleys and the 2025–27 neutral-site parks. They are applied in `park_geometry_v12.py`
with the evidence tier in each note. **Re-measured coverage of gated rows: 88.2%** (87.8–88.4%
by season; 88.3% in each leave-one-season-out training set). Five of 30 venues are below 50%.
Missing heights are still median-imputed with no indicator, as the frozen trainer does.

**Implementation details the spec left open, fixed 2026-10-01 before the run**
(`mlb_luck_score.models.evaluate_gated_geometry_v12`). None was chosen from v1.2 results.

- **Wall-height subgroups:** short ≤ 8 ft; tall ≥ 15 ft (the existing
  `DEFAULT_HIGH_WALL_THRESHOLD_FT`); medium in between; unknown when missing. These are fixed
  bins, not the v0.7C terciles, which fail on the many tied 8 ft values.
- **Wall bands:** beyond (margin ≥ 0) and short of the wall by 0–5, 5–10 and 10–20 ft.
- **Perturbations:** wall distance ±10 ft; hit distance ±10 ft; wall height 8 vs 16 ft,
  only on rows whose height is known. The specialist always scores the perturbed rows.
- **Pooling:** each fold's held-out gated rows are perturbed with that fold's specialist,
  and mean P(HR) is pooled across folds.
- **Gate-boundary diagnostic:** air balls with geometry whose margin is within 2 ft of −20.
- **Absolute home-run ECE per group (reported only):** binary P(HR) through the v0.7D
  bootstrap and classifier, for both models.
- **Weather-residual diagnostic (reported only):** home-run indicator minus P(HR), demeaned
  within venue, regressed on within-venue-demeaned air density (outdoor/roof-open gated rows)
  and on following wind (the 15 `wind_supported` venues only), with a game-clustered
  bootstrap.
- Output: `outputs/tables/v1_2_gated_geometry_evaluation.json` (gitignored). **One run.**
  If it fails partway, the fix is a code defect fix, recorded here, never a change in
  response to results.

**Decisions approved by the maintainer on 2026-09-30 (all as proposed):**

- **[D1]** Leave-one-season-out within 2021–2023 as the decision basis, with 2024 able only
  to block (plan open question 3).
- **[D2]** Venue/subgroup gate on credible *paired regression*, with absolute ECE reported
  but not gating.
- **[D3]** One candidate: a logistic specialist, a 20 ft gate, and the five geometry
  features listed.
- **[D4]** The weather perturbation checks replaced by the longer-ball check plus a
  weather-residual diagnostic.

### Step 3 result: the single run (2026-10-01, code `4c40763`, clean tree)

Command: `.venv/bin/python -m mlb_luck_score.models.evaluate_gated_geometry_v12`. Output:
`outputs/tables/v1_2_gated_geometry_evaluation.json` (gitignored). Pooled out-of-fold rows
for 2021–2023: 364,311, of which 34,861 are gated. All five specialist features were kept
in every fold. Gated-row height coverage was 88.3%.

| Criterion | Result | Pass |
|---|---|---|
| Log loss, all rows (candidate − baseline) | −0.0360 [−0.0368, −0.0351] | ✅ |
| Log loss, gated rows only | −0.3759 [−0.3849, −0.3672] | (reported) |
| Overall ECE | 0.01468 → 0.01539; paired Δ +0.00071 [+0.00045, +0.00097] | ✅ not material |
| Home-run ECE | 0.00272 → 0.00331; paired Δ +0.00059 [+0.00011, +0.00115] | ✅ not material |
| Venue/subgroup paired regression | none credible | ✅ |
| Perturbations (pooled ΔP(HR)) | farther wall −0.261; taller wall −0.039 (n=30,792); longer ball +0.261. No venue backwards; same on 2024 | ✅ |
| Open field identical | asserted in every fold | ✅ |
| 2024 confirmation | −0.0350 [−0.0366, −0.0334] | ✅ |

**Automated rule:** `recommend_adopt: True`. Under the plan this is input to the
maintainer's decision, not the decision. Calibration worsened slightly but credibly (both
ECE intervals are above 0), though below the materiality rule.

**Reported only:**

- **Gate-boundary jump:** mean |ΔP(HR)| = 0.114 for 3,116 balls within 2 ft of the −20 ft
  line. That's a real discontinuity: a ball's expectation changes by about 11 points when it
  crosses an arbitrary line.
- **Weather residual (within venue):** air density. Baseline slope −0.27 [−0.45, −0.10] per
  kg/m³ (thinner air → more home runs than predicted). Candidate −0.02 [−0.15, +0.10].
  Following wind at the 15 supported venues: both CIs include 0.

**⚠ Plausibility finding — not automated, needs the maintainer.** The gated-row gain
(0.376 nats) is very large. `hit_distance_sc` is the glove point for a caught ball, the wall
contact point for a ball off the wall, and the seat landing point for a home run (Statcast
definition, step 1). So `distance − wall` partly records the outcome itself. On gated rows,
2021–2023:

| Margin vs wall | n | Home-run share |
|---|---|---|
| −20 to −10 ft | 7,481 | 1.0% |
| −5 to 0 | 3,165 | 7.0% |
| 0 to +5 | 2,761 | 19.0% |
| +10 to +30 | 7,747 | 81.5% |
| ≥ +30 | 7,835 | 99.0% |

The sign of the margin alone classifies home run vs not correctly 84.0% of the time. Some of
this is physics: a ball 30 ft past the wall was never catchable. Some is truncation by the
result. A robbed home run reads as "at the wall", so the model learns that such balls are
outs, and the robbery stops looking like bad luck. **The gain therefore overstates what
geometry adds to "what the contact earned."** The baseline has had the same property since
v0.1 (it also uses `hit_distance_sc`), but geometry makes it much sharper: knowing where the
wall is turns the truncated distance into a near-label. This is the
`RESEARCH_RULES.md` target-leakage concern in spirit, though `hit_distance_sc` is not on
`LEAKAGE_COLUMNS`.

### Redesign: outcome-free air-ball model — FROZEN 2026-10-01

**Decision (maintainer, 2026-10-01): option 1, redesign.** `gated_geometry_v12` is **not
adopted**. Decisions D5–D8 were approved as proposed on 2026-10-01, before any redesign
code existed.

**Disclosure:** the redesign responds to the plausibility finding above, and 2021–2023 have
now been evaluated once (the `gated_geometry_v12` run). The same folds are reused. They are
no longer untouched for v1.2, and every result below says so. Nothing from that run's
metrics is used to choose features or settings below; the features follow D6, and the model
settings are repository defaults.

**Candidate: `airball_geometry_v12b`.**

- **Scope [D5]:** eligible rows with `bb_type` in {fly_ball, line_drive} and v1.2 geometry
  resolved. Everything else (ground balls, popups, air balls without geometry) keeps the
  `baseline_v02` prediction, bit-identical.
- **Inputs [D6]:** `launch_speed`, `launch_angle`, `spray_angle_approx`, `stand`,
  `bb_type`, `wall_distance_in_spray_direction`, `wall_height_in_spray_direction`. **No
  `hit_distance_sc` and nothing derived from it.** Caveat: `spray_angle_approx` comes from
  the fielded hit coordinates. It is kept as the only direction data and flagged.
- **Model [D7]:** `HistGradientBoostingClassifier` with scikit-learn defaults,
  `random_state = RANDOM_SEED`, no class weighting (the repository's existing HGB
  convention). Numeric inputs pass through unimputed, so an unknown wall height reaches
  the model as missing (native handling) rather than as the median. Categoricals are
  one-hot encoded. It is fitted on in-scope rows of the training seasons only.
- **Reference model [D8]:** identical, minus the two wall features. The comparison
  isolates what geometry adds without outcome-revealing inputs.

**Folds:** as before: leave-one-season-out within 2021–2023, pooled, plus 2021–2023 → 2024
as a confirmation that can only block. Game-clustered bootstrap, 500 replicates, seed 42,
95%.

**Adoption criteria. ALL must hold on in-scope rows of the primary pool:**

1. **Geometry adds:** the paired log-loss CI (candidate − reference) sits entirely below 0.
2. **Calibration:** the candidate's overall ECE and home-run ECE are not materially worse
   than the reference's (v0.3 rule: +0.01 absolute or +50% relative).
3. **No credible venue/subgroup regression:** no adequately supported group (≥100 plays,
   ≥20 games) whose paired log-loss CI vs the reference sits entirely above 0. Groups:
   venue, wall-height bin (short ≤ 8 / medium / tall ≥ 15 / unknown), spray sector and
   `bb_type`. Absolute home-run ECE status (v0.7D) is reported for both models but is
   not gating.
4. **Perturbations** (in-scope rows, pooled, P(HR)): wall distance ±10 ft → P(HR) lower
   when farther; wall height 8 vs 16 ft (rows with a known height) → lower when taller;
   launch speed ±3 mph → higher when harder. No derived columns exist to recompute.
5. **Other rows unchanged:** every out-of-scope row is bit-identical to `baseline_v02`
   (asserted).
6. **2024 confirmation:** no credible regression vs the reference (CI entirely above 0
   blocks).

**Reported, never gating:**

- **The leak gap:** log loss and ECE of `baseline_v02` vs the candidate on in-scope rows.
  This measures how much `hit_distance_sc` was worth, not a quality gap.
- **Robbery-zone behaviour:** mean P(HR) under `baseline_v02` vs the candidate for caught
  air balls whose measured distance is within 5 ft of the wall (measured margin used for
  reporting only).
- Weather residual (as before), per-venue perturbation directions, and wall-height coverage.

**Redefinition notice:** if adopted, air-ball luck no longer uses where the ball was
fielded. That changes what the score means for every fly ball and line drive, not just
near the wall. Ground balls and popups still use `hit_distance_sc` (pre-existing since
v0.1; flagged, out of scope).

Output: `outputs/tables/v1_2b_airball_geometry_evaluation.json`. **One run.**

### v1.2b result: the single run (2026-10-01, code `3ccc227`, clean tree)

364,311 pooled rows from 2021–2023; 172,066 in scope (air balls with geometry). Wall-height
known for 87.3% of them.

| Criterion | Result | Pass |
|---|---|---|
| Geometry adds (candidate − reference log loss) | **+0.0051 [+0.0014, +0.0081]**: geometry makes it worse | ❌ |
| Calibration vs reference | ECE 0.00507 → 0.00583; HR ECE 0.00281 → 0.00231; not material | ✅ |
| Venue/subgroup regression | **11 credible regressions**: venues 4, 10, 14, 680, 3289 (Guaranteed Rate, Oakland, Rogers Centre, T-Mobile, Citi); wall height short and medium; spray sectors center, right-center and right; line drives | ❌ |
| Perturbations | farther wall −0.028, taller wall −0.009, harder hit +0.082, all in the right direction (taller wall backwards at venue 22 only) | ✅ |
| Out of scope identical | asserted | ✅ |
| 2024 confirmation | **+0.0031 [+0.0011, +0.0049]**: a credible regression | ❌ |

**Under the frozen rule: `recommend_adopt: False`.** `airball_geometry_v12b` fails three
criteria.

**Reported only — read before deciding anything:**

- **Leak gap, the largest number here.** The candidate's log loss is **0.251 lower than
  `baseline_v02`'s** on in-scope rows [−0.255, −0.246]. ECE is 0.0058 vs 0.0292. A
  gradient-boosting model on launch conditions alone (the reference does about as well)
  predicts air-ball outcomes far better than the production logistic model, even though
  production reads the outcome-revealing distance. This is a separate finding about the
  production contact model, not a v1.2 result. 2021–2023 have now been used twice, which
  any follow-up must disclose.
- **Weather moves into "luck" without distance.** Within-venue residual vs air density:
  baseline −0.08 [−0.12, −0.04]; candidate **−0.37 [−0.41, −0.33]**. Following wind at the
  15 supported venues: baseline ~0; candidate +0.0023 [+0.0018, +0.0028]. Measured distance
  had been carrying weather (step 1). Dropping it puts weather back into the residual, which
  goes against the Option B weather decision unless weather is modelled explicitly.
- **Robbery zone** (caught air balls within 5 ft of the wall, n=2,727): mean P(HR) is 0.336
  under the baseline and 0.234 under the candidate. This doesn't show the hoped-for
  "robberies now count as bad luck" effect. Read as descriptive only.
- **Why geometry hurts (hypothesis, untested):** wall distance plus height at a spray angle
  nearly fingerprints the venue. Like v0.3's `venue_id`, a flexible model can memorize venue
  quirks that don't carry across seasons. The five regressing venues and the 2024
  regression fit that, but don't prove it.

## Freeze and prospective test

1. Freeze the v1.2 code, features and adoption decision with a recorded date **before the
   first 2027 regular-season game** (date to be cited from the official schedule, not
   guessed).
2. In 2027, score with both v1.1 (unchanged) and v1.2 into separate namespaces.
3. Pre-register the 2027 comparison: calibration of each on 2027 batted balls, overall and
   by venue, with the same three-way gate.
4. Which version the public site shows in 2027 is a separate, explicit decision.
5. If adopted: a new `scoring_version` and reference artifact, never overwriting v0.2.0 or
   v1.x artifacts; methodology copy updated through `public_labels`.

## Interaction with the 2027 projection plan

The projection model (`docs/plans/2027_projection_model_plan.md`) uses deserved values as
inputs. If v1.2 is adopted, projections should use v1.2 values, so **v1.2's adoption
decision comes first.** If v1.2 is not adopted, projections use v1.0/v1.1 values and
nothing waits.

## Open questions for the maintainer

1. Option A or Option B (and separately for weather)?
2. Contact model only, or also revisit the defensive components' park handling?
3. Is season-level cross-validation within 2021–2023 acceptable as the decision basis,
   given 2024 is not pristine?
4. Who does the human geometry spot-check, and when?
5. Separately from this plan: fix the `PRODUCT.md` "venue" claim now?
