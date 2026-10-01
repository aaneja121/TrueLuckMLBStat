# Plan: Version 1.3 — an outcome-free contact model

Status: **spec FROZEN 2026-10-01** (maintainer decisions D9–D12, all approved as
proposed). It was written before any v1.3 code existed. Adoption would be a
**redefinition of the score** (`RESEARCH_RULES.md`, "Never silently redefine the Luck
Score"), with its own `scoring_version`.

## Why

From the v1.2 work (`docs/plans/v1_2_park_weather_plan.md`):

- `hit_distance_sc` is where a ball was fielded: the glove, the wall or the seats (MLB
  Statcast glossary). Any model that reads it partly reads the outcome. `baseline_v02`
  has used it since v0.1.
- In the v1.2b run, a gradient-boosting model on launch conditions alone beat
  `baseline_v02` on air balls by **0.25 log loss** (CI −0.255 to −0.246). Its ECE was 0.006
  vs 0.029. That was a reported number, not a gate. It needs its own frozen evaluation,
  which is this plan.
- Without distance, weather moves back into the residual (air-density slope −0.37), so
  weather has to be modelled explicitly to keep the Option B weather decision.

## Decisions (maintainer, 2026-10-01)

- **D9 — Scope: all batted balls.** Ground balls and popups have the same leak (fielded
  location), so one model covers every training-eligible batted ball.
- **D10 — Weather: air density only.** `air_density_kg_m3` is an input. It is null when the
  roof is closed, indoors, or when status is unknown, and reaches the model as missing.
  Station wind is reported but not used (step 2: a weak, noisy proxy).
- **D11 — Park: park-neutral for v1.3.** Wall geometry worsened the model twice (v1.2,
  v1.2b), so park effects stay in luck (Option A) for v1.3. The 2026-09-29 Option B park
  decision is **deferred, not reversed**, and is to be revisited with a different approach.
- **D12 — Judging: development gates plus a pre-registered 2027 test.** No development
  season is untouched: 2021–2023 were evaluated twice in v1.2/v1.2b, and 2024 earlier. The
  development gates below are necessary, not sufficient. **2027 is the real test.**

## Candidate `outcome_free_v13`

- **Rows:** every training-eligible batted ball, development seasons only.
- **Inputs:** `launch_speed`, `launch_angle`, `spray_angle_approx`, `stand`, `bb_type`,
  `air_density_kg_m3`. **No `hit_distance_sc`, no wall geometry, no venue identity.**
  - `spray_angle_approx` comes from the fielded hit coordinates. That is a weaker leak for
    air balls and a stronger one for ground balls (a ball through the hole is fielded in the
    outfield). It is kept as the only direction data and flagged; its effect is measured in
    the reported spray ablation.
  - `bb_type` is a stringer classification already used by the baseline. Kept.
- **Model:** `HistGradientBoostingClassifier` with scikit-learn defaults,
  `random_state = 42`, no class weighting. Numeric inputs pass through unimputed (native
  missing-value handling); categoricals are one-hot encoded. This is the v1.2b
  `build_airball_model` construction, reused.
- **Comparator:** `baseline_v02` (production), retrained on the same folds.

## Evaluation (frozen)

- **Folds:** leave-one-season-out within 2021–2023, out-of-fold predictions pooled (primary,
  disclosed as reused). 2021–2023 → 2024 as a confirmation that can only block.
  Game-clustered bootstrap, 500 replicates, seed 42, 95%.
- **Adoption gates. ALL must hold on the primary pool:**
  1. **Log loss:** the paired candidate − baseline CI sits entirely below 0.
  2. **Calibration:** overall ECE and home-run ECE not materially worse than the baseline
     (v0.3 rule: +0.01 absolute or +50% relative).
  3. **No credible venue/subgroup regression:** no adequately supported group (≥100 plays,
     ≥20 games) whose paired log-loss CI vs the baseline sits entirely above 0. Groups:
     venue, `bb_type`, spray sector, and roof state (outdoor; roof open; closed or indoor;
     unknown).
  4. **Physical directions** (P(HR), pooled across folds):
     - launch speed ±3 mph → P(HR) higher when harder, all rows;
     - air density −0.05 vs +0.05 kg/m³ → P(HR) higher in thinner air, rows where density
       is known.
  5. **2024 confirmation:** no credible regression vs the baseline (CI entirely above 0
     blocks).
- **Reported, never gating:**
  - **Density ablation:** the same model without density, paired log loss.
  - **Spray ablation:** the same model without `spray_angle_approx`, paired log loss, overall
    and for ground balls. This sizes the remaining fielding-location leak.
  - **Weather residual:** within-venue home-run residual vs air density, and vs following
    wind at the 15 `wind_supported` venues, for both models (the v1.2 diagnostic).
  - **Park residual:** mean (actual − expected) home runs per venue for both models. This
    documents what D11 puts into luck.
  - **Robbery zone:** mean P(HR) for caught air balls measured within 5 ft of a v1.2 wall
    (the measured distance is used for this report only).
  - Per-class calibration for both models.
- Output: `outputs/tables/v1_3_outcome_free_evaluation.json`. **One run.** A failure partway
  through is fixed as a code defect, recorded here; never as a response to results.

## Result: the single development run (2026-10-01, code `deebae1`, clean tree)

`outputs/tables/v1_3_outcome_free_evaluation.json` (gitignored). 364,311 pooled rows from
2021–2023, reused as disclosed.

| Gate | Result | Pass |
|---|---|---|
| Log loss (candidate − baseline) | **−0.2014 [−0.2041, −0.1989]** | ✅ |
| Calibration | ECE 0.0147 → 0.0045; home-run ECE 0.0027 → 0.0014 (both better; paired CIs below 0) | ✅ |
| Venue/subgroup regression | none credible in 44 adequately supported groups. Popups are the only group with a worse point estimate (+0.010, CI to +0.027) | ✅ |
| Physical directions | harder hit +0.040 P(HR); thinner air +0.008, backwards only at Chase Field (venue 15), also in 2024 | ✅ |
| 2024 confirmation | −0.2088 [−0.2137, −0.2041] | ✅ |

**Development gates: all pass (`recommend_adopt: True`).** Under D12 that is necessary, not
sufficient: the real test is 2027.

**Reported only — caveats the maintainer should weigh:**

- **Two changes are bundled.** The gain mixes removing the outcome-revealing distance with
  moving from multinomial logistic regression to gradient boosting. They aren't separated,
  so the gain can't be credited to "outcome-free" alone.
- **Air density doesn't do its job.** Ablation: *with* density, log loss is **worse** by
  +0.0043 [+0.0022, +0.0069]. The within-venue weather residual on air density is −0.073
  [−0.092, −0.055] for the candidate vs −0.042 [−0.062, −0.025] for the baseline. Following
  wind at the 15 supported venues: candidate +0.0009 [+0.0007, +0.0012]; baseline ~0. So
  weather remains partly in luck, somewhat more than in production, despite D10. Density
  does appear to absorb altitude: Coors Field's home-run residual is −1.64 per 100 batted
  balls for the baseline and +0.06 for the candidate.
- **Spray angle carries a lot:** removing it worsens log loss by 0.103 overall and 0.084 on
  ground balls. It comes from fielded coordinates, so part of this may be fielding-location
  information. Direction is also genuinely predictive (pull, gaps, holes), and this run
  can't separate the two.
- **Park:** the baseline was already park-neutral (its `venue` input is 100% null), so D11
  keeps the status quo rather than changing it. The largest candidate park residuals are at
  temporary venues: Estadio Alfredo Harp Helu +5.4 per 100 (n=101); TD Ballpark +1.4;
  Sahlen Field +1.2. Kauffman −1.2 and Great American +1.1 are next.
- **Robbery zone** (caught air balls within 5 ft of the wall, n=2,727): mean P(HR) is 0.336
  for the baseline and 0.244 for the candidate. The outcome-free model gives these balls
  less home-run credit than production, not more. Read as descriptive.
- **Per-class ECE (baseline → candidate):** out 0.031 → 0.011; single 0.037 → 0.007;
  double 0.003 → 0.002; triple 0.0001 → 0.002 (worse, tiny); home run 0.003 → 0.001.

## If the gates pass: freeze and the 2027 test

1. Freeze code, inputs and settings with a recorded date **before the first 2027
   regular-season game** (the date cited from the official schedule, not guessed).
2. Pre-register the 2027 comparison: log loss and calibration of v1.1 and v1.3 on 2027
   batted balls, overall and by venue, with the v0.7D three-way gate. Score both into
   separate namespaces.
3. 2026 may be scored by v1.3 as prospective scoring only. It is already observed, so it is
   never evidence for adopting v1.3 (`RESEARCH_RULES.md`, Version 1.1 rule 5).
4. Which version the public site shows is a separate, explicit decision. Downstream
   components (run values, defensive models, attribution ledger, public copy through
   `public_labels`) are out of scope until then.

## Interaction with other plans

- v1.2: closed. Neither candidate adopted; v1.1 is unchanged.
- 2027 projections (`docs/plans/2027_projection_model_plan.md`) use v1.0/v1.1 deserved
  values unless v1.3 is adopted before the projection freeze.
