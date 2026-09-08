# 2025 Pitcher Replication — recovery plan (PROPOSED, not authorized, not executed)

**Status: awaiting explicit maintainer authorization. Nothing in this plan has been run.**

2025 is **open**. Execution `8edc32d6ca8830ce` opened the held-out season at
`2026-09-08T17:00:33Z` and failed inside the frozen scorer. Any run that follows is a
**resumption after exposure**, never a first look, and no document, commit message, or
report may describe it otherwise.

## 1. What happened

`evaluation.run_v1_final_evaluation.train_and_score_2025(development_df, evaluation_df)`
names its first parameter `development_df`, but its contract is *training data already
filtered to `TRAIN_SEASONS`*. It **validates** that precondition; it does not enforce it by
filtering. Version 1.0's own caller filters first:

```python
full_development_df = pd.read_parquet(DEVELOPMENT_INPUT_PATH)
training_df = full_development_df[full_development_df["season"].isin(TRAIN_SEASONS)].copy()
artifacts, trained = train_and_score_2025(training_df, evaluation_df, ...)
```

`replication/run_pitcher_replication_2025.py::run_replication` omitted that filter and
passed the unfiltered 2021–2024 parquet. 2024 reached the training argument and the guard
refused:

```
FinalEvaluationError: Training data contains season(s) outside TRAIN_SEASONS
(2021, 2022, 2023): [2024]
```

The guard did its job. It fired **before** the eligibility pipeline and **before** every
`train_*` call, so no component model was fit — on 2024 or anything else.

## 2. Exposure, precisely

| | |
|---|---|
| Execution receipt written | **yes** — preserved, never to be deleted |
| Held-out 2025 downloaded | **yes** — 3 files, ~107 MB |
| 2025 evaluation dataframe constructed | **yes**, in memory |
| Component models fit | **no** |
| Any model fit on 2024 | **no** — the guard fired first |
| 2025 predictions produced | **no** |
| Pitcher aggregates produced | **no** |
| Questions A–G computed | **no** |
| Classification computed | **no** |
| Result artifact written | **no** — outputs namespace holds only `.gitkeep` |

**One 2025 quantity has been observed**, and it must be recorded rather than glossed: the
terminal logged `Combined development dataset: 129929 rows across 1 requested season(s)
(126386 training-eligible)` — a `clean_development_data` row-count line emitted while
building the 2025 dataset. That is exposure/volume metadata, not a Contact Luck result. It
bears on no A–G structural hypothesis (all of which concern correlations, resolving power,
dominance shares, and reliability), and it is not the pitcher-season count question A
reports. It is nonetheless a 2025 number that has been seen, and it is logged in the
incident record.

## 3. Why this is a technical recovery, not a scientific one

The frozen Version 0.14 specification already fixes training to 2021–2023 and 2025 to
evaluation only. The correction **restores** that intent; it does not choose it.

- No metric, denominator, estimator, interval procedure, threshold, role boundary or
  presentation rule changes.
- No model architecture changes.
- Nothing is informed by an observed 2025 result — none exists.
- The research freeze validates unchanged: 15/15 sources, zero drift,
  `spec_content_hash 23320d46…`.

Had the guard *not* existed, this would have silently trained on 2024 and produced a
contaminated replication. It is worth stating plainly that the frozen system caught its own
misuse.

## 4. The minimal correction

Two lines in `replication/run_pitcher_replication_2025.py::run_replication`:

```python
-    development_df = pd.read_parquet(DEVELOPMENT_INPUT_PATH)
-    artifacts, _trained = train_and_score_2025(development_df, evaluation_df)
+    from mlb_luck_score.config import TRAIN_SEASONS
+
+    development_df = pd.read_parquet(DEVELOPMENT_INPUT_PATH)
+    training_df = development_df[development_df["season"].isin(TRAIN_SEASONS)].copy()
+    artifacts, _trained = train_and_score_2025(training_df, evaluation_df)
```

The expression is available as a named, tested unit:
`replication.pitcher_replication_incident.select_training_frame`, which additionally
refuses an empty result and a frame with no season column. Proven synthetically in
`tests/test_pitcher_replication_incident.py` (21 tests): filtering yields exactly
`{2021, 2022, 2023}`; 2024, 2025 and 2026 can each be excluded from training; the
evaluation frame stays `{2025}`; and no frozen scientific setting moves.

## 5. Files that would change

| File | Change |
|---|---|
| `replication/run_pitcher_replication_2025.py` | the two-line filter above |
| `tests/test_pitcher_replication_runner.py` | flip the "defect still present" assertion to assert the filter IS applied |
| `artifacts/pitcher_replication/v0_14/recovery_manifest.json` | **new** — see §6 |

Nothing else. The frozen research spec, the estimators, split-half, questions and
authorization modules are untouched.

## 6. Recovery manifest (new artifact, never an overwrite)

The sealed pre-execution manifest hashes `run_pitcher_replication_2025.py`, so applying the
fix **will** invalidate it — by design. It must not be resealed in place. Instead:

- Write `recovery_manifest.json` **beside** the originals. The pre-execution manifest, the
  execution-start receipt and `failure_records.jsonl` all remain byte-identical.
- It must record: the superseded `execution_manifest_hash cb34daf7…`; the incident's
  `execution_id 8edc32d6ca8830ce`; the SHA-256 of the corrected runner and of the exact
  diff; the unchanged freeze and spec hashes; the recovery authorization's own hash; a
  statement that 2025 was already open; and the byte hashes of the three reusable 2025
  artifacts (§7).
- Write-once, no amendment path, same contract as the pre-execution manifest.
- Exactly **one** recovery attempt: a `recovery_receipt.json` written before the resumption
  reads the cached 2025 files, after which further attempts refuse.

## 7. Reuse the downloaded 2025 data — no second network fetch

The three artifacts are complete and integrity-verifiable, and the recovery should reuse
them rather than re-downloading:

```
statcast_2025_regular_season.parquet  107,596,739 B  ba53159340b6a2a78b0a4e5eccb700d24a4766eceaeb275543ecf8fafef2fe3b
game_metadata_2025.parquet                 30,836 B  257b9938616203d628f18cf8db48dc7d09f196dd948786f87526a91593af4389
sprint_speed_2025.parquet                   7,165 B  96541a63eff71da1e7235bad52e8c1932ab16c7657c372e537baba546f6b62d9
```

The ingestion functions already skip download when the target path exists, so a resumption
performs **no network operation**. The recovery manifest must pin these hashes so a silent
substitution is impossible. Re-downloading 2025 would require its own separate explicit
authorization and is not proposed.

## 8. Preconditions for authorizing the recovery

All must hold, and the runner must enforce them:

1. Explicit maintainer authorization naming this incident and permitting one resumption.
2. The original receipt, pre-execution manifest and failure record intact and unmodified.
3. The research freeze still validating at 15/15.
4. The three 2025 artifacts matching the hashes in §7.
5. A clean, committed working tree.
6. A sealed recovery manifest built from that commit.
7. No prior recovery receipt.

## 9. What the recovery may never do

- Describe itself as a first look, or reset/delete the receipt.
- Change anything on the basis of observed 2025 performance.
- Alter the specification, role boundaries, metrics, estimators or presentation rules.
- Re-download 2025 without separate authorization.
- Read 2026.
- Retry automatically after any failure.

## 10. Recommendation

**RECOVERY SAFE TO AUTHORIZE.** The failure is a pure plumbing defect with an unambiguous
frozen intent, no model was fit, and no replication result exists or was observed. The
recovery requires no methodological decision.

It nevertheless requires the maintainer's explicit sign-off, asked for in the moment,
because it will re-open already-exposed held-out data under a spent one-time authorization.
