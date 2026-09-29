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

## Approach, if Option B is chosen

Constrain the model with known physics instead of letting it learn park and weather freely.

1. **Carry adjustment from physics, not from fitting.** Compute an expected change in
   carry distance from air density and wind direction/speed using a published,
   cited physical relationship, then derive an adjusted projected distance. The model
   receives the adjusted distance, not raw weather columns. Its sign is fixed by physics
   (thinner air → farther; wind out → farther) and cannot be learned backwards.
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
5. **Roof and indoor games** follow existing rules: an explicit "Roof Closed" string only,
   wind 0 when closed, climate otherwise null. No invented indoor climate.

Every derived feature must be recomputed whenever its source is overridden in
perturbation checks (`RESEARCH_RULES.md`, "Any override of a raw feature must recompute
every feature DERIVED from it").

## Data preconditions

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
