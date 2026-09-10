# Question F erratum — the batter-side reproduction control (post-replication audit)

**Status: F IMPLEMENTATION CLEARED, recorded as a documented erratum.**
The sealed 2025 result is unchanged and must stay unchanged. Nothing below reopens 2025,
alters the sealed artifacts, or changes a scientific definition.

The sealed 2025 replication result records, inside question F:

```
"implementation_trustworthy": false
"batter_side_reproduction": { "reproduces": false, ... }
```

This erratum records what that flag actually means. It means the **control was measured
against the wrong season**, not that the split-half procedure is wrong.

## 1. What the control is

`replication/pitcher_split_half.py::reproduce_batter_side` runs the question-F code path
on the **batter** grouping key and compares the result against
`BATTER_SIDE_2024_REFERENCE` — a hard-coded copy of the committed Version 0.11 Phase 6
numbers:

| split | n | Pearson | Spearman |
|---|---|---|---|
| calendar | 399 | 0.050297542421047836 | 0.07131881210564099 |
| odd_even | 487 | 0.09654304470474931 | 0.11889983530505965 |

It exists to prove that porting the estimator from the batter key to the pitcher key did
not change the procedure. It is a test *of the module*, not of the metric.

## 2. Where the reference came from

`outputs/tables/aggregation_stability_v011_report.json`, key `split_half_reliability`
(`calendar` and `odd_even_game_pk`). Verified byte-for-byte against the constant during
this audit — the four reference values match exactly.

That report was produced by
`mlb_luck_score.models.evaluate_aggregation_stability` over
`run_season_aggregation.build_player_season_report`, which trains on `TRAIN_SEASONS`
(2021–2023) and scores `VALIDATION_SEASONS` (**2024**). The reference is therefore a
**2024 batter-season** population, under the frozen eligibility rules, with
`min_eligible_each_half = 20`.

## 3. What the 2025 run actually measured

`build_question_f_report(artifacts)` calls `reproduce_batter_side(artifacts)` on **the
same artifacts it is handed** — it takes no season argument and applies no season filter.

In the replication runner, those artifacts come from
`evaluation.run_v1_final_evaluation.train_and_score_2025`, which builds
`SeasonAggregationArtifacts(scoring_df=eval_df, ...)` and **enforces** that `eval_df`
contains only season 2025:

```
if eval_seasons_observed != [EVALUATION_SEASON]:
    raise FinalEvaluationError("Evaluation dataset must contain ONLY season 2025 ...")
```

So during the sealed run the control computed **2025 batter-season** split-half
correlations and compared them against a **2024 batter-season** reference.

That comparison cannot succeed. It is a category error, not a numerical near-miss.

## 4. Why n = 403 vs 399 and n = 502 vs 487

They are two different seasons' batter populations, not two measurements of one
population:

| split | measured (2025 batters) | reference (2024 batters) |
|---|---|---|
| calendar | n = 403, r = 0.0380 | n = 399, r = 0.0503 |
| odd_even | n = 502, r = 0.1007 | n = 487, r = 0.0965 |

The small gaps are what one expects from season-to-season variation in how many batters
clear 20 resolved eligible batted balls in both halves. They are not evidence of a
changed inclusion rule.

## 5. The audit questions, answered

Re-run today at HEAD `30b4d8210127724f664d54ac8771ccf5561fc474` via
`make question-f-2024` (2024 development data, offline, no 2025 or 2026 access):

```
implementation_trustworthy: True
reproduces: True   tolerance: 1e-09
  calendar: matches=True  measured n=399  ref n=399  diffs {n: 0, pearson: 0.0, spearman: 0.0}
  odd_even: matches=True  measured n=487  ref n=487  diffs {n: 0, pearson: 0.0, spearman: 0.0}
```

The pitcher-side 2024 figures also reproduced the frozen spec exactly
(calendar n=439 r=0.1023; odd_even n=550 r=0.0768).

| # | Question | Answer |
|---|---|---|
| 1 | What stored reference is reproduced? | The Version 0.11 Phase 6 batter split-half figures, copied into `BATTER_SIDE_2024_REFERENCE`. |
| 2 | Where was it created? | `outputs/tables/aggregation_stability_v011_report.json`, by `evaluate_aggregation_stability`. |
| 3 | Which dataset / range / eligibility? | 2024 (`VALIDATION_SEASONS`), models trained on 2021–2023; frozen eligibility; `min_eligible_each_half = 20`. |
| 4 | Which implementation produced the measurement? | The same `pitcher_split_half.compute_split_half_reliability`, group `batter` — unchanged code. |
| 5 | Why do the n's differ? | Different seasons: 2025 batters measured against a 2024 reference. |
| 6 | Code drift? | **No.** `pitcher_split_half.py` has exactly one commit (`6276461`, the freeze); the constant was never edited. Freeze validates 15/15, zero drift. |
| 7 | Data drift? | **No.** The development parquet (2026-08-03) predates the reference report (2026-08-05) and still reproduces it to 0.0. |
| 8 | Eligibility drift? | **No.** The threshold and both mask helpers are imported from the accepted module, never restated; the 2024 rerun matches exactly. |
| 9 | Source artifact / version drift? | **No.** Reference constant and committed report agree exactly. |
| 10 | Is the accepted reference stale? | **No.** It reproduces exactly on the season it describes. |
| 11 | Is the current reproducer wrong? | **No.** It is correct; it was invoked against the wrong season's artifacts. |
| 12 | Does this affect the pitcher split-half values? | **No.** `report["pitcher"]` is computed by a separate call on the same 2025 artifacts and never reads the reproduction result. |
| 13 | Does it affect any primary question A–E? | **No.** A–E take the pitcher-season `frame`; none consults question F or the control. |
| 14 | Does it affect the REPLICATED classification? | **No.** `classify()` reads only each question's `agrees`. F is secondary and `agrees` was `true` on its own merits. |
| 15 | Does it affect the public product? | **No.** Nothing in the pitcher product framing derives from this control. |

## 6. Why this is not being "fixed" in code

`replication/pitcher_split_half.py` is frozen source **15 of 15**
(`pitcher_replication_freeze.FROZEN_SOURCE_RELATIVE_PATHS`). Editing it would change
`freeze_content_hash`, breaking the provenance chain that the sealed 2025 result, its
seal, and the recovery receipt are all bound to.

The scientific record must keep saying exactly what it said at execution time. So the
correct remedy is this erratum plus regression tests, not a code change.

**If the control is ever re-wired** (for a future replication, under a new freeze), the
fix is to run `reproduce_batter_side` against the **2024 development artifacts** — the
population its reference describes — and keep it out of the held-out evaluation path
entirely. `make question-f-2024` already does exactly that, and is the check that
actually validates the implementation.

## 7. Regression tests

`tests/test_pitcher_split_half.py::TestTheReproductionControlIsScopedToItsOwnSeason`
pins the diagnosis offline, on synthetic artifacts:

- the control takes no season parameter and compares against a fixed constant;
- foreign artifacts fail it while the reference side stays untouched;
- a failed control never alters the pitcher figures it is reported alongside;
- `classify()` ignores `implementation_trustworthy`, so the verdict cannot move.

## 8. Interpretation of question F itself

Unchanged and unaffected. Pitcher split-half correlations on 2025 remain near zero
(calendar Pearson 0.0767 / Spearman 0.1092; odd/even 0.1464 / 0.1207 at
`min_eligible_each_half = 20`, staying low at higher floors). For a retrospective luck
quantity, near-zero persistence **confirms** the framing rather than indicting the
metric. F `agrees = true`, and F is secondary and cannot override the package either way.
