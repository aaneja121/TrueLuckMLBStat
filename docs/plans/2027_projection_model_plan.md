# Plan: a 2027 season projection model

Status: **PLAN — nothing here is authorized or built.** Drafted 2026-09-29.
Research-only. No public surface is implied (see "Product scope" below).

## The question

Given what a hitter did in one season, what will his run value per 100 batted balls be the
next season? And does knowing his *luck-adjusted* (deserved) performance make that
projection better than knowing only his realized results?

This is the season-to-season version of Contact Forecast, which asked the same question
*within* a season (first 100 batted balls → next 100 or 200).

## What we already know, and from where

| Evidence | Seasons | What it says | Can it inform design? |
|---|---|---|---|
| R1 (frozen 2026-09-03) | 2023–24 dev | Shrunk deserved beat shrunk realized by 3.9% MAE, 95% CI [−0.31, −0.03], pre-specified. Small; each season's interval crosses zero. | **Yes** — development evidence. |
| Ridge stage (frozen 2026-09-04) | 2024 dev | Full-profile ridge vs. shrunk deserved: −3.4% MAE, CI [−0.34, +0.03], crosses zero. Adding realized to deserved did not help. | **Yes** — development evidence. |
| Resolution pass (2026-09-29) | 2026 | Model D vs. shrunk deserved: no evidence of improvement at H100 (n=31) or H200 (n=48). Deserved beat realized descriptively in all cohorts, no interval. | **No.** 2026 may be scored, never used to choose a model. |

The maintainer has now seen the 2026 result. Any design choice that matches it (for
example "prefer the simple shrunk-deserved projection") must be justified by the
development rows above, and the write-up must say the 2026 result was known at design
time.

## Season roles for this model

| Role | Seasons | Notes |
|---|---|---|
| Development input → target pairs | **2022→2023, 2023→2024** | Deserved values are walk-forward: 2022 scored by a contact model fit on 2021 only; 2023 on 2021–22. |
| Not a development pair | 2021→2022 | 2021 has no strictly-earlier season, so no out-of-sample deserved value exists for it (R1 limitation `primary_comparison_covers_2023_2024_only`). Cross-fitting within 2021 would be a new procedure needing its own spec and freeze. |
| Sealed | 2025 | Not an input, target, or evaluation season unless the authorization below is signed. |
| Projection input | 2026 | Scored by the frozen v1.1 system (fit 2021–23, out of sample for 2026). Scoring, not tuning. |
| Prospective test | **2027** | Projections frozen and hashed before the first 2027 regular-season game. |

**The honest weakness:** two development pairs is very little. Fit on 2022→2023 and
evaluate on 2023→2024 is one evaluation pair. Every development result will be imprecise,
and the plan must accept that rather than add pairs by weakening the walk-forward rule.

## The 2025 question — needs a separate, explicit sign-off

A 2027 projection that skips the most recent completed season but one is weaker. If
multi-season inputs are wanted (2026 **and** 2025 for 2027), 2025 has to be read as an
input. That requires a new, narrowly scoped authorization on the pattern of the pitcher
replication exception in `RESEARCH_RULES.md`, written there **before** any code reads 2025.

### Draft authorization text (for the maintainer to accept, edit, or reject)

> **A third sealed-2025 use is authorized — as a 2027 projection INPUT only.**
>
> - **Scope.** 2025 Contact Luck season aggregates, as already produced by the sealed
>   Version 1.0 final evaluation, may be read as input features for the frozen 2027
>   projection model. Nothing is recomputed on 2025 raw data.
>   *(Unverified: whether that evaluation wrote per-hitter season aggregates with the
>   features the model needs. If it did not, producing them is a new computation on sealed
>   data, and the scope must say so explicitly.)*
> - **Not authorized.** Using 2025 as a target, an evaluation season, or a
>   development/tuning season for this or any model; choosing features, the model class,
>   shrinkage, or any hyperparameter using 2025; feeding anything back into Contact Luck;
>   publishing 2025 values; reading 2025 raw Statcast.
> - **Bound by hash.** Names the exact sealed Version 1.0 output files and their SHA-256s,
>   and the frozen projection specification's hash. Amending either voids it.
> - **One code path.** Only the dedicated 2027 projection entry point may read the 2025
>   files, through the existing sealed-namespace guards. Tests prove no development runner
>   can.
> - **Timing.** Read only after the projection specification is frozen and committed.

Open point for the maintainer: if the model is developed on single-season inputs (the
only thing 2022→2023/2023→2024 can support without 2021 deserved values), a two-season
input at projection time is a model that was never developed. The simpler, more
defensible choice may be **single-season inputs (2026 → 2027)**, which needs no 2025
authorization at all.

## Model ladder (fixed before any development metric is computed)

1. League mean.
2. Shrunk realized persistence.
3. **Shrunk deserved persistence — the primary benchmark.** Any model must beat this.
4. Challenger: the frozen Contact Forecast Model D feature set, refit at season level.

Adopt the challenger only if it beats (3) on development evidence with the rule stated
in advance. Otherwise ship (3). Given the ridge-stage result, (3) is the expected outcome,
and that is fine.

## Target and denominator

- Target: next-season **realized run value per 100 resolved batted balls**, the same
  quantity Contact Forecast used (`target_realized_rv_per_100`). Resolved-BBE denominator
  (field errors and fielder's choices excluded, matching production).
- Decide in the spec whether the target is the contact stage or the published metric
  (which includes batter-runner advancement). R1 measured the contact stage.
- Minimum next-season sample to be evaluated (for example ≥100 resolved BBE), fixed in
  the spec.

## Pre-registered 2027 test

Written and hashed before the first 2027 game:

- **Primary:** MAE(chosen projection) − MAE(shrunk realized persistence) on 2027, with a
  batter-clustered bootstrap interval. This is the "deserved vs. actual" question the 2026
  pass could only answer descriptively.
- **Secondary:** chosen projection vs. league mean; RMSE, Pearson, Spearman, reported
  together with no privileged metric.
- The four-way classification rule from Contact Forecast, applied to the primary only.
- Survivorship reported: only hitters who reach the 2027 minimum are evaluated, and they
  are not representative of everyone projected.

## Product scope

`PRODUCT.md` lists "Not a projection or regression tool" as a non-goal. This plan is
research-only. Publishing projections would be a deliberate product revision decided
separately, after the 2027 result, not a side effect of building the model.

## Sequencing

1. ~~Contact Forecast resolution pass~~ — done 2026-09-29.
2. ~~Inputs decided: single-season (2026 only); no 2025 authorization needed.~~
3. Decide ordering against the v1.2 park/weather plan (`docs/plans/v1_2_park_weather_plan.md`).
   If v1.2 is adopted, projections should use v1.2 deserved values, so v1.2's adoption
   decision comes first.
4. Write and freeze the projection specification (targets, ladder, adoption rule, 2027
   test). Commit.
5. Develop on 2022→2023 / 2023→2024 only. Freeze.
6. Score 2026 once as input. Hash the 2027 projections before Opening Day 2027.
7. Resolve after the 2027 regular season.

## Decision record

Decided by the maintainer on 2026-09-29, before any projection development:

1. **Inputs: 2026 only (single-season).** Matches what the two development pairs can
   support. **The 2025 authorization is not needed and is not requested**; the draft text
   above stays unsigned, and 2025 remains sealed for this model.
2. **Target: the contact stage** (Rc − E0 scale, `target_realized_rv_per_100`), the same
   quantity R1 and Contact Forecast measured, so the 2027 result is comparable with the
   existing evidence. Not the published metric with batter-runner advancement.

## Open questions for the maintainer

1. v1.0 or v1.2 deserved values as inputs — i.e., does v1.2 come first?
2. Minimum 2027 sample for a hitter to be evaluated.
