# Pre-registration: v1.3 vs v1.1 on the 2027 season

**Status: DRAFT, 2026-10-01. Not binding until the maintainer approves it and the approval
is committed.** Once approved, it is final for the 2027 test. A later change is a new
decision, recorded here with its date and reason, and is never permitted after any 2027
batted ball has been scored by either model.

This is the real test that `docs/plans/v1_3_outcome_free_contact_plan.md` (D12) defers to.
The development gates there passed on 2021–2023, which had been reused three times; that
result is necessary, not sufficient.

## 1. Deadline

- The 2027 regular season "is slated to begin on Wednesday, March 24, with an Opening Night
  game" (MLB.com, "MLB releases 2027 regular-season schedule", published 2026-07-16,
  <https://www.mlb.com/news/mlb-2027-schedule-released>, retrieved 2026-10-01).
- **Everything in §2 must be committed, with its commit hash recorded here, before the first
  pitch of the first 2027 regular-season game.** If that game moves later (for example a
  labor stoppage — the current agreement expires after 2026), an earlier freeze still
  stands. If it moves earlier, the deadline moves with it. The date must be re-checked
  against the official schedule when the freeze is committed.

## 2. What is frozen

| Item | Frozen as |
|---|---|
| v1.3 model | `outcome_free_v13` exactly as in `mlb_luck_score.models.evaluate_outcome_free_v13` at the freeze commit: inputs, HGB settings, seed 42 |
| v1.3 density-free variant | the same, without `air_density_kg_m3` (§5) |
| v1.1 model | the production Version 1.1 path, unchanged |
| Training data, all three models | **2021–2023** (`TRAIN_SEASONS`), the same seasons v1.1 trains on. 2024 is not added (keeps the comparison like-for-like); 2025 is sealed; 2026 is never training data |
| Eligibility | `mlb_luck_score.eligibility` as at the freeze commit |
| Weather source | the existing MLB Stats API + IEM ASOS pipeline and roof rules, with the 2027 venue/station table reviewed before the freeze |
| Analysis code | the comparison script (§7), tested on synthetic data only |
| This document | its own commit hash |

**Engineering preconditions, all before the deadline, none reading a 2027 outcome:**

1. A v1.3 prospective scoring path in its own namespace (`outputs/prospective/v1_3/`,
   `artifacts/prospective/v1_3/`, `data/prospective/2027/`), with the same guards as v1.1:
   clean tree, completed dates only, immutable snapshots.
2. A 2027 weather join (station table, roof status) for prospective rows.
3. The §7 comparison script with synthetic tests.
4. 2027 park geometry is **not** needed (v1.3 is park-neutral).

## 3. Population

All 2027 regular-season batted balls that are training-eligible under the frozen
`eligibility.py`, from games final as of the recorded season-end date. Each ball is scored
once by each model. No venue, team or player is excluded. Rows whose air density is missing
(closed roof, indoor, unknown) are kept; the model handles missing density natively.

## 4. Primary endpoint and decision rule

- **Primary:** paired log loss, v1.3 − v1.1, over the whole population. Game-clustered
  bootstrap, 500 replicates, seed 42, 95% percentile CI.
- **v1.3 is confirmed** only if ALL hold:
  1. the primary CI sits entirely below 0;
  2. overall ECE and home-run ECE are not materially worse than v1.1 (v0.3 rule: +0.01
     absolute or +50% relative);
  3. no venue with adequate support (≥100 plays, ≥20 games) has a paired log-loss CI
     entirely above 0;
  4. both physical directions hold on 2027 rows: launch speed ±3 mph raises P(HR), and air
     density −0.05 vs +0.05 kg/m³ raises P(HR) (rows with known density).
- **Not confirmed:** the primary CI is entirely above 0, or any of 2–4 fails.
- **Inconclusive:** the primary CI straddles 0, or the population is below **60,000 batted
  balls** (about half a normal season). Inconclusive is never reported as a pass.
- Absolute home-run ECE per venue (v0.7D three-way) is reported for both models; it does not
  gate.

## 5. Pre-registered weather check (density)

The development run found that density *worsened* log loss slightly (+0.0043 [+0.0022,
+0.0069]), and weather stayed partly in the residual. The maintainer kept density
(2026-10-01). 2027 settles it:

- **Density test:** paired log loss, v1.3 − density-free variant, on 2027 rows with known
  density. If the CI sits entirely below 0, density is **supported**. Otherwise it is **not
  supported**, and the pre-committed consequence is that the next version drops it. No
  other weather input is added in its place without a new pre-registration.
- **Weather residual (reported):** within-venue home-run residual vs air density and vs
  following wind (the 15 `wind_supported` venues), for v1.3, the variant and v1.1.

## 6. Timing — one look

- **No comparison metric is computed before the regular season has ended** (the recorded
  season-end date, verified as for 2026). Routine snapshot monitoring of each model on its
  own is allowed. Paired metrics, venue comparisons and the density test are not.
- The §7 script runs **once** after the season ends. A crash is fixed as a code defect and
  recorded here; nothing is changed in response to results.

## 7. Analysis script

A dedicated module reads only the two 2027 namespaces, refuses any other season, and writes
`outputs/prospective/v1_3/comparison_2027.json` once (it refuses to overwrite). It reuses
`compute_paired_bootstrap`, the v0.3 materiality rule and the v0.7D classifier.

## 8. What this does not decide

- Which model the public site shows. That is a separate maintainer decision, made after the
  result, with a new `scoring_version` and methodology copy through `public_labels`.
- Run values, defensive components, the attribution ledger and Contact Forecast. They are
  untouched by this test.
- Park handling (Option B, deferred).

## 9. Disclosures

- 2021–2023 were evaluated in v1.2, v1.2b and v1.3 development (three looks); 2024 earlier.
  None of them counts as untouched evidence.
- 2026 may be scored by v1.3 as prospective scoring only, never as evidence (RESEARCH_RULES,
  Version 1.1 rule 5).
- Model class and distance removal are bundled in v1.3; the test does not separate them.

## Decisions for the maintainer before this becomes binding

- **[P1]** Train all three models on 2021–2023 (matching v1.1), not 2021–2024.
- **[P2]** Exactly one analysis, after the regular season; no interim paired metrics.
- **[P3]** "Inconclusive" below 60,000 batted balls (a shortened or cancelled season).
- **[P4]** If density is not supported in 2027, the next version drops it (§5).
