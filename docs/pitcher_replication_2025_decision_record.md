# 2025 Pitcher Contact Luck Replication Decision

**Classification: REPLICATED.** The one-time held-out 2025 replication of Pitcher Contact
Luck under the frozen Version 0.14 specification is **complete, sealed, and permanently
closed**.

This document is the tracked index to that result. The result files themselves live in
intentionally gitignored namespaces and are **not** committed — they are identified here
by path and hash instead. Do not `git add -f` them.

## 1. Identity

| Field | Value |
|---|---|
| Replication season | 2025 (held-out, `FINAL_TEST_SEASONS`) |
| Specification version | 0.14.0 |
| Classification | **REPLICATED** |
| Execution ID | `780aa2e2b8ec43f5` |
| Repository commit | `30b4d8210127724f664d54ac8771ccf5561fc474` |
| Freeze content hash | `58d8235b8d8324d55fe296875c1c8cdbd9857c5375be887f273822d1b1d40714` |
| Spec content hash | `23320d4654b930112575fa7915b77cac618ff567ed3dabe5c314bf00002d0069` |
| Recovery manifest revision 2 hash | `b05b8f1b273eebfc26d6a62fba3608fabfb2cf41929c63bef82270dcb1925eb6` |
| Final results SHA-256 | `d483342a2549653845e88153c73e51329d0a9a1da8d41fa1be9d6d0c16d5b041` |
| Sealed at | `2026-09-08T19:07:12.740995+00:00` |
| Regular-season range scored | 2025-03-18 → 2025-09-28 |
| Pitcher-season rows | 1,098 |

The seal's `execution_manifest_hash` is the **recovery** manifest revision 2 hash, because
the successful run executed under the recovery path (see §5).

## 2. Where the sealed artifacts live (gitignored)

| Artifact | Path |
|---|---|
| Final results | `outputs/pitcher_replication/v0_14/pitcher_replication_2025_results.json` |
| Final provenance | `outputs/pitcher_replication/v0_14/pitcher_replication_2025_provenance.json` |
| Final seal | `artifacts/pitcher_replication/v0_14/pitcher_replication_2025_seal.json` |
| Recovery receipt | `artifacts/pitcher_replication/v0_14/recovery_receipt.json` |
| Research freeze | `artifacts/pitcher_replication/v0_14/pitcher_replication_freeze.json` |
| Original execution manifest | `artifacts/pitcher_replication/v0_14/pitcher_replication_execution_manifest.json` |
| First-exposure receipt | `artifacts/pitcher_replication/v0_14/execution_start_receipt.json` |
| Recovery manifests | `.../recovery_manifest.json`, `.../recovery_manifest_rev2.json` |
| Incident history (append-only) | `artifacts/pitcher_replication/v0_14/failure_records.jsonl` |

The research freeze validates **15/15 frozen source files with zero drift**.

## 3. The classification rule (frozen before 2025 was opened)

- `REPLICATED` when every primary question A–E agrees.
- `NO_GO` when the metric's own coherence failed (A or E).
- `REVISE` otherwise.
- Question F is **secondary** and can never alter the package classification.
- Question G is **qualitative** and is never auto-scored.

Outcome: `primary_disagreements = []`, `secondary_disagreements = []` →
**REPLICATED**.

## 4. Primary findings — A–E all agreed

**A — population and centering.** 1,098 pitcher-season rows; play-level mean Pitcher
Contact Luck ≈ +0.003435 runs/play; cumulative total distribution centered essentially at
zero.

**B — opportunity heterogeneity.** |cumulative total| vs BBE: Pearson ≈ 0.6267,
Spearman ≈ 0.7029. Signed cumulative total vs BBE: Pearson ≈ 0.2745, Spearman ≈ 0.1523.
Workload strongly governs *magnitude*, far less so *sign/order*.

**C — totals vs rate.** All 20 of the 20 most extreme per-100 pitcher seasons fell below
the frozen board display minimum. Tiny samples produce extreme per-100 rates while
cumulative impact stays near zero. Supports **cumulative runs** as the primary board
quantity.

**D — reliever single-play dominance.** Among 2025 reliever-like seasons, one play
exceeded 25% of the net total in ≈ 84.69% of cases, 50% in ≈ 61.48%, and 100% in
≈ 29.49%.

**E — rate precision.** Resolving power stayed below 1 at every practical workload floor
(≥60 BBE: 0.6247; ≥150: 0.5518; ≥300: 0.5437; ≥450: 0.3246). Resolved BBE required for
precision: ±5 runs/100 → 202; ±4 → 316; ±3 → 561; ±2 → 1,262. Maximum observed 2025
workload was **609** resolved BBE. Per-100 is too noisy to be the primary pitcher ranking
statistic.

## 5. Incident and recovery history

2025 was originally opened under a one-time authorization, execution `8edc32d6ca8830ce`,
at `2026-09-08T17:00:33Z`.

1. **Original execution failed before any model fitting.** The runner passed the full
   2021–2024 development frame into `train_and_score_2025()`, whose first argument
   contract requires data already filtered to `TRAIN_SEASONS` (2021–2023). The guard
   raised `FinalEvaluationError: Training data contains season(s) outside TRAIN_SEASONS`.
   Recorded exposure: 2025 was ingested and the evaluation frame constructed, but **no**
   model parameters were fit, no predictions produced, no pitcher aggregates, no
   questions A–G, no classification. Recorded append-only in `failure_records.jsonl`.
2. **The first recovery launch also failed, before any second 2025 read**, because it
   incorrectly entered the pristine first-look namespace-empty guard. No recovery receipt
   was written, so that attempt was **not** consumed.
3. **The recovery architecture was corrected** so the two gates are separate:
   - first look: `run_replication` → `assert_ready_for_2025` → requires an empty
     pre-exposure namespace;
   - recovery: `run_recovery` → `assert_ready_for_recovery` → requires the exact cached
     2025 inputs by hash and requires completed outputs to be absent.

   Both paths share the same scientific execution stages.
4. **Recovery manifest revision 2 was sealed and separately authorized**, and the one
   authorized recovery ran successfully once, producing `recovery_receipt.json`.

**The successful recovery is not a "first look."** The receipt says so itself
(`is_a_first_look: false`) and names the execution it supersedes. 2025 was already open;
the recovery was a resumption after exposure, and no document, commit, or report may
describe it otherwise. The original execution-start receipt is preserved and must never
be deleted or replaced.

**The recovery authorization is CONSUMED.** Do not run `--run-recovery`,
`make run-pitcher-replication-2025`, or any equivalent 2025 execution.

## 6. Question F — secondary, and its implementation audit

F reports `agrees = true`. Pitcher split-half correlations remain near zero
(`min_eligible_each_half = 20`: calendar Pearson ≈ 0.0767 / Spearman ≈ 0.1092; odd/even
≈ 0.1464 / 0.1207), staying low at higher workload floors. For a retrospective luck
quantity, near-zero persistence **confirms** the framing.

The sealed result also records `implementation_trustworthy = false`, because the
batter-side reproduction control did not reproduce its stored reference.

**Audit outcome: F IMPLEMENTATION CLEARED — recorded as a documented erratum.**

Root cause: the control compares against a fixed **2024** batter reference but measures
whatever artifacts it is handed, and during the replication it was handed the **2025**
evaluation artifacts. It compared a 2025 measurement to a 2024 reference — a season
mismatch that cannot match, regardless of implementation correctness. That explains
n = 403 vs 399 and n = 502 vs 487: two different seasons' batter populations.

Re-run today on the 2024 development data it was designed for, the same code reproduces
the committed Version 0.11 Phase 6 numbers to an absolute difference of **0.0** on every
value. There is **no** code drift, data drift, eligibility drift, or source-artifact
drift; the reference is not stale and the reproducer is not wrong.

The control's failure does not touch the pitcher split-half values, any primary question
A–E, or the classification (`classify()` reads only each question's `agrees`). It is not
corrected in code because `replication/pitcher_split_half.py` is frozen source 15/15 —
editing it would invalidate the freeze the sealed result is bound to.

Full detail: **`docs/pitcher_replication_2025_question_f_erratum.md`**.

## 7. Question G — human review

**G HUMAN REVIEW: PASSED.** Reviewed from the frozen examples in the sealed result; no
new examples were selected after the fact, and G was not auto-scored.

| Bucket | Pitcher | Cumulative | BBE | /100 |
|---|---|---|---|---|
| extreme favorable total | Nick Pivetta | +29.5777 | 483 | +6.1237 |
| extreme unfavorable total | Charlie Morton | −17.0295 | 420 | −4.0546 |
| high workload | Zack Littell | +11.0931 | 609 | +1.8215 |
| reliever-like | Dennis Santana | +17.2061 | 192 | +8.9615 |
| near-zero total | Matt Brash | +0.0001 | 119 | +0.0001 |

Checks performed:

- **Sign convention is internally consistent.** In all five buckets the largest favorable
  play is hard contact that became an out (101.5–105.9 mph fly balls, +1.03 to +1.44
  runs) and the largest unfavorable play is a damaging outcome (home runs, and one
  75.3 mph line-drive triple, −0.83 to −1.53 runs). Positive = the realized outcome was
  more favorable to the pitcher than the contact deserved. That holds throughout.
- **The weak-contact case is the sharpest evidence.** Dennis Santana's 75.3 mph line
  drive that went for a triple scores −0.8322: weak contact producing a damaging result
  reads as unfavorable to the pitcher, exactly as the definition requires.
- **Totals and rates are arithmetically consistent.** Every bucket satisfies
  `cumulative / BBE × 100 = /100` to the recorded precision.
- **Role assignment is consistent with the frozen thresholds.** Starter-like buckets run
  12.4–17.9 BBE per appearance (≥10); both reliever-like buckets run 2.3–2.8 (≤8).
- **No attribution reversal or run-value sign error is indicated.** Nothing in the frozen
  examples suggests the pitcher negation was applied inconsistently.

The one thing worth stating plainly: this is a qualitative read of five frozen examples,
not a systematic audit of all 1,098 pitcher-seasons. It is evidence that the sign
convention and attribution are sane, not proof that every play is.

## 8. Season policy after this result

- **2025 is permanently closed.** The result is final and must never be rerun,
  regenerated, retuned against, or reopened. The sealed result, its seal, the recovery
  receipt, the research freeze, the original execution manifest, the recovery manifests,
  and the incident history are all immutable.
- **2026 remains prospective and unopened for the pitcher line.**
  `PROSPECTIVE_SEASONS = (2026,)`. 2026 pitcher data is being preserved for a separate
  prospective evaluation after the 2026 regular season. It must not be inspected,
  downloaded, scored, counted, used for debugging, used for UI fixtures, or used for
  validation. No 2026 data was read in producing this record.

## 9. Product implications

These were decided before the 2025 outcome was observed and are **not** revised on the
strength of having now seen it:

- **Cumulative Pitcher Contact Luck runs is the primary quantity** for boards and
  ranking (question C).
- **Contact Luck /100 is secondary**, and must always be shown with BBE and uncertainty
  (question E: per-100 is uncertainty-limited at every realistic 2025 workload).
- **Separate Starter-like and Reliever-like boards.** No single merged pitcher
  leaderboard. Starter-like ≥10 BBE/appearance; reliever-like ≤8; 8–10 is ambiguous
  (question D: reliever totals are frequently dominated by one play).
- **Sub-60 BBE is player-page-only.** No board rank on player cards.
- **Surface the largest favorable and unfavorable plays** (question G).
- **Positive Pitcher Contact Luck means the realized outcome was more favorable to the
  pitcher than the contact deserved.** Public wording stays centralized in
  `mlb_luck_score.scoring.public_labels`.

## 10. Recommendation

**READY FOR PITCHER PRODUCTIZATION**, under the product decisions in §9 and the public
copy rules in `PRODUCT.md` / `CONTEXT.md`.
