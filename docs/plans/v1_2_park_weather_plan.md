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
   The contact model is a five-class logistic regression; verify against current
   scikit-learn docs whether any candidate class supports multiclass monotonic constraints
   before relying on this.
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
The papers themselves were not read in full when this was recorded; read them before
writing the check code.

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
- Open: whether deep caught balls' distance is the catch point or a projection
  (Statcast's definition should be cited, not inferred) matters for geometry gating near
  the wall.

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
