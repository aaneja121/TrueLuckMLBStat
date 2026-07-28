# Contact Luck Prototype v0.1 (TrueLuckMLBStat)

> **This repository is a research prototype, not a validated public baseball statistic.**
> Every number it produces -- probabilities, raw luck, the public score, the confidence
> label -- is a provisional Version 0.1 research artifact. See "Scientific limitations"
> below before drawing any conclusion from it.

## Project purpose

MLB outcomes on a fair batted ball are a mix of skill and fortune. This project measures
one specific slice of that fortune: **contact luck** -- how much better or worse a
batter-runner's *continuous, uninterrupted* result was than what an ordinary, modelable
set of conditions (exit velocity, launch angle, spray direction, batted-ball type, venue)
would predict.

It deliberately does **not** yet try to separate out weather, exact defensive
positioning/routes, throwing execution, or baserunning decisions -- those are documented
extension points, not implemented behavior (see "Future work").

## Version 0.1 scope

Version 0.1 predicts a 5-class outcome distribution for each eligible fair-batted-ball
event:

```
out | single | double | triple | home_run
```

and turns the gap between that prediction and the actual outcome into:

1. a **preliminary additive raw-luck value** (research placeholder), and
2. a **provisional -100..+100 public display score** (not scientifically calibrated yet).

### What IS implemented

- Statcast data acquisition via `pybaseball`, in small date chunks, with retries
  (`mlb_luck_score.data.download_statcast`).
- A single centralized fair-batted-ball eligibility definition
  (`mlb_luck_score.eligibility`).
- A cleaned, one-row-per-eligible-event table with engineered spray features
  (`mlb_luck_score.data.clean_batted_balls`).
- A reusable scikit-learn contact-feature preprocessing pipeline with an explicit
  target-leakage check (`mlb_luck_score.features.build_contact_features`).
- A baseline multinomial logistic regression contact model
  (`mlb_luck_score.models.train_contact_model`).
- Calibration diagnostics: reliability tables and per-class plots, plus an optional
  post-hoc isotonic/sigmoid calibration layer (`mlb_luck_score.models.calibrate_model`).
- Preliminary additive raw luck (`mlb_luck_score.scoring.raw_luck`).
- A provisional, monotonic, sign-preserving public score interface
  (`mlb_luck_score.scoring.public_score`).
- A Version 0.1 data-completeness confidence report (`mlb_luck_score.scoring.confidence`).
- Offline synthetic tests, starter notebooks, and a time-based development design that
  protects the 2025 season.

### What is NOT implemented (see "Future work")

Precise weather/air-density modeling, defensive route/reaction/throwing modeling,
baserunning-decision modeling, Shapley-style attribution, and confidence intervals.
Documented extension points exist for all of these; none are implemented behavior in
Version 0.1.

## Eligible events (Version 0.1)

Centralized in `mlb_luck_score.eligibility.ELIGIBLE_EVENTS_V0_1`:

```
field_out, force_out, grounded_into_double_play, double_play,
fielders_choice, fielders_choice_out, single, double, triple, home_run,
field_error, sac_bunt, sac_fly, sac_fly_double_play
```

This is a **documented Version 0.1 research choice, not a permanent taxonomy**. Two of
these event types (`fielders_choice`, `field_error`) have an ambiguous batter-runner
resulting base in public Statcast data; rows with those events are kept in the cleaned
table but excluded from model training with a specific reason -- no label is ever
invented for them.

## Excluded events (Version 0.1)

Strikeouts, walks, hit-by-pitches, catcher interference, foul balls/tips, dropped third
strikes, wild pitches, passed balls, stolen bases, pickoffs, balks, fan/outside
interference, discretionary base awards, identified player-rule violations, and rows
lacking required contact variables (`launch_speed`, `launch_angle`).

## Contact luck vs. decision quality

Contact luck asks "given this batted ball's physical characteristics, was the *outcome*
better or worse than expected?" It does **not** ask whether the batter's swing decision,
pitch selection, or approach was good -- a well-executed check swing single and a
mis-hit bloop single can carry the same contact-luck value even though the decisions
behind them were very different. Decision-quality modeling is out of scope for v0.1.

## Contact luck vs. defensive execution

The model's expected-outcome distribution is fit only on the contact event's own physical
characteristics (exit velocity, launch angle, approximate spray direction, batted-ball
type). It does not yet model exact fielder positioning, reaction time, route efficiency,
or throwing execution. A ball that "should" be a double but becomes a single because of
an excellent defensive play will look identical, in Version 0.1, to a double turned
single by bad luck. Separating skill-driven defense from luck is future work.

## Public-data limitations

All data comes from public Statcast fields exposed via `pybaseball`. Some fields
(alignment classifications, bat speed, sprint speed) are only available for a subset of
seasons/players and are treated as optional, not required.

## Approximate spray-direction limitations

`hc_x`/`hc_y` are Statcast's raw visualization coordinates, not verified physical spray
angles. `spray_angle_approx` (and everything derived from it: `spray_sector`, `is_pull`,
`is_center`, `is_opposite_field`) uses a widely-used community approximation formula that
has **not** been independently verified against Baseball Savant's internal geometry.
Treat these as directional signals, not precise measurements.

## Installation

Requires Python 3.11+ (developed against pybaseball 2.2.4+ / scikit-learn 1.6+ -- the
calibration module uses `sklearn.frozen.FrozenEstimator`, added in scikit-learn 1.6).

```bash
python3 -m venv .venv          # if .venv does not already exist
source .venv/bin/activate
make setup                      # pip install -e ".[dev]"
```

Or manually:

```bash
pip install -e ".[dev]"
```

## Environment activation

```bash
source .venv/bin/activate
```

All `make` targets invoke `.venv/bin/python` directly, so they work whether or not the
venv is currently activated in your shell.

## Sample download

Requires internet access -- it scrapes public Baseball Savant CSV endpoints via
`pybaseball`. Defaults to one week of the 2024 season so a repository checkout never
triggers a full-season download:

```bash
make download-sample
# equivalent to:
python -m mlb_luck_score.data.download_statcast \
    --start-date 2024-04-01 --end-date 2024-04-07 \
    --output data/raw/statcast_2024_sample.parquet
```

Use `--dry-run` to see the download plan (date chunks, output path) with **no** network
access. The command refuses to overwrite an existing output file unless `--overwrite` is
passed, and refuses to touch the protected 2025 season unless `--allow-final-evaluation`
is explicitly passed.

## Cleaning

```bash
make clean-data
# equivalent to:
python -m mlb_luck_score.data.clean_batted_balls \
    --input data/raw/statcast_2024_sample.parquet \
    --output data/processed/cleaned_batted_balls.parquet
```

Produces one row per eligible fair-batted-ball event, with a deterministic `event_id`,
engineered spray features, a `season` column, and eligibility/training-exclusion flags.
Never requires network access; never crashes on missing optional columns, empty input,
or malformed dates (each is logged and handled explicitly, not silently coerced).

## Full development dataset (2021-2024)

The one-week sample above is too small to train on (0 rows fall in the 2021-2023 training
window). To actually train the baseline model, download full regular-season Statcast data
for all four development seasons:

```bash
make download-development-data
# equivalent to:
python -m mlb_luck_score.data.download_development_data \
    --seasons 2021 2022 2023 2024 --output-dir data/raw

make clean-development-data
# equivalent to:
python -m mlb_luck_score.data.clean_development_data \
    --seasons 2021 2022 2023 2024 --raw-dir data/raw \
    --output data/processed/cleaned_development_data.parquet
```

**Before running this for real:**

- **Requires internet access** and downloads roughly 26x more data than the one-week
  sample, per season, across all four seasons.
- **Storage (rough estimate, scaled from the one-week sample's actual size — 4.1MB raw /
  26,072 pitch-level rows / 4,451 cleaned rows for 7 days of 2024):**
  - ~100-110MB raw Parquet per season (~700,000 pitch-level rows/season) -> **~400-450MB
    total** across `data/raw/statcast_2021_regular_season.parquet` through `_2024_`.
  - ~120-150MB total for the combined `data/processed/cleaned_development_data.parquet`
    (~450,000-500,000 eligible batted-ball rows across all four seasons).
  - All of this stays under git-ignored `data/` -- nothing here gets committed.
- **Runtime:** downloads happen in small chunks (5 days by default) with retries between
  requests; based on Baseball Savant's typical response times this is commonly in the
  range of tens of minutes to a few hours for all four seasons combined, depending on
  network conditions and server load. This is an estimate, not a measurement -- run
  `--dry-run` first (see below) to confirm the chunk plan, and consider timing a single
  season before committing to all four.
- Per-season files are **resumable**: if the command is interrupted or a season fails
  (e.g. a transient network error), re-running it skips seasons that already downloaded
  successfully and only retries what's missing or failed -- it does not restart from
  scratch.
- **Never overwrites the one-week sample**: this command always writes
  `statcast_<season>_regular_season.parquet` / `cleaned_development_data.parquet`,
  distinct filenames from `statcast_2024_sample.parquet` /
  `cleaned_batted_balls.parquet`. Both workflows can coexist.
- Use `--dry-run` on the downloader to see the exact per-season chunk plan with **no**
  network access:
  ```bash
  python -m mlb_luck_score.data.download_development_data --dry-run
  ```
- `mlb_luck_score.data.clean_development_data` fails clearly (before doing any work) if
  any requested season's raw file is missing, logs row totals and training-exclusion
  reasons **separately by season**, and deterministically deduplicates across season
  files (defense in depth -- `game_pk` is globally unique, so cross-season duplicates
  should not occur in practice).
- Both commands refuse to touch the protected 2025 season unless
  `--allow-final-evaluation` is explicitly passed -- and the downloader has no
  configured date range for 2025 at all, as a second layer of protection.

## Training

```bash
make train
# equivalent to:
python -m mlb_luck_score.models.train_contact_model \
    --input data/processed/cleaned_batted_balls.parquet \
    --output-dir artifacts
```

Trains on `TRAIN_SEASONS` (2021-2023 by default) and evaluates on `VALIDATION_SEASONS`
(2024 by default) -- see "Time-based validation design" below. Model artifacts and
metadata are written under `artifacts/` (git-ignored). Note the default `--input` here is
the one-week sample, which has 0 rows in the 2021-2023 training window; point `--input` at
`data/processed/cleaned_development_data.parquet` (see "Full development dataset" above)
to actually train the model.

By default `train_model`/`make train` use `class_weight=None` (the unweighted probability
baseline) -- see "Model comparison and probability calibration" immediately below for why.

## Model comparison and probability calibration

> **Do not calculate or publish Luck Scores until the selected model has acceptably
> calibrated probabilities.**

An earlier baseline trained `LogisticRegression(class_weight="balanced")`. On real
2021-2024 Statcast data this produced **severely miscalibrated probabilities** -- e.g. on
2024 validation data, rows the model called ~74% likely to be a single were observed to
be a single ~22% of the time (n=638, a reliable bin); similar gaps appeared for double
(pred 0.45 / observed 0.17, n=5,507), triple (pred 0.54 / observed 0.036, n=2,496), and
home_run (pred 0.55 / observed 0.32, n=1,294).

A controlled comparison (same 2021-2023 training rows, same untouched 2024 validation
rows, changing **only** `class_weight`) confirmed the cause: `class_weight="balanced"`
reweights the loss inversely to training-class frequency, which shifts the model's fitted
probabilities away from the true class prior. Run it yourself:

```bash
make compare-models
# equivalent to:
python -m mlb_luck_score.models.compare_models \
    --input data/processed/cleaned_development_data.parquet \
    --output-dir outputs/tables \
    --figures-dir outputs/figures/model_comparison \
    --include-post-hoc-calibration
```

This compares four variants on the same untouched 2024 validation season (results from
the real dataset, reproduced 2026-07-27):

| variant | log loss | expected calibration error (ECE) | accuracy | recall(triple) |
|---|---|---|---|---|
| `unweighted` (`class_weight=None`) | **0.670** | **0.014** | 0.751 | 0.001 |
| `class_balanced_comparison_only` (`class_weight="balanced"`) | 1.078 | 0.134 | 0.549 | 0.429 |
| `naive_prevalence` (ignores all features) | 0.937 | 0.002 | 0.675 | 0.000 |
| `unweighted_post_hoc_calibrated` (isotonic, time-ordered) | 0.723 | 0.048 | 0.699 | 0.000 |

**Recall is not calibration.** `class_balanced_comparison_only` shows much higher recall
on rare classes (e.g. triple: 0.429 vs. 0.001) -- it's tuned to predict them more often --
but a far worse log loss and nearly 10x worse ECE. That "better" recall is not evidence of
trustworthy probabilities; for a luck metric built on probability *magnitude*
(`expected_value = sum(p(outcome) * value(outcome))`), calibration is what matters, and
`class_balanced_comparison_only` fails badly on it. It is preserved in the codebase only
as an explicitly-labeled comparison model (`train_model(..., class_weight="balanced")`,
CLI `--class-weight balanced`) -- never use it for the Contact Luck score.

**`naive_prevalence`** (predicts the training-set marginal distribution for every row,
ignoring all features) is a sanity floor, not a candidate: it has a trivially low ECE
(it's "calibrated on average" by construction) but is beaten by `unweighted` on log loss
and every per-class Brier score, meaning `unweighted` adds real discriminative value on
top of being well-calibrated.

**Post-hoc calibration did not help here.** `unweighted_post_hoc_calibrated` uses a
time-ordered design (base model trained on 2021-2022 only, isotonic calibration fit on
2023 only, evaluated on 2024 -- three non-overlapping season groups, see
`mlb_luck_score.config.CALIBRATION_BASE_TRAIN_SEASONS`/`CALIBRATION_FIT_SEASONS`/
`CALIBRATION_EVAL_SEASONS`) specifically so the calibrator is never fit and evaluated on
the same rows. It made both log loss and ECE *worse* than the plain `unweighted` model on
this data. Current recommendation: use the plain unweighted model; do not add a post-hoc
calibration layer unless a future check shows it actually helps.

**Current recommendation:** `unweighted` (`class_weight=None`, the default) is the
candidate Contact Luck probability baseline. Its remaining known imperfections (e.g. the
`single` class shows real, reliable miscalibration in two bins -- predicted 0.45 vs.
observed 0.63 at n=8,899, and predicted 0.62 vs. observed 0.27 at n=840) are far smaller
than `class_balanced_comparison_only`'s, but are not zero -- treat this as a Version 0.1
finding to keep monitoring, not a final validated result. Full per-bin calibration tables
and plots are in `outputs/tables/model_comparison_detail.json` and
`outputs/figures/model_comparison/` after running `make compare-models`, and in notebook
`03_calibration.ipynb`.

## Testing

```bash
make test          # or: .venv/bin/python -m pytest
```

The entire suite runs offline against synthetic fixtures -- no test requires network
access or real Statcast data.

```bash
make check          # format-check + lint + typecheck + test
```

## Notebook usage

```bash
make notebook        # launches Jupyter in notebooks/
```

- `01_data_exploration.ipynb` -- outcome counts, missingness, eligible vs. excluded
  counts, contact-variable distributions, and a per-season summary (eligible/
  training-eligible counts and outcome-class breakdown by season). Prefers
  `cleaned_development_data.parquet` when present, falling back to the one-week sample.
- `02_contact_model.ipynb` -- time-based split demo (training rows by season for
  2021-2023, validation rows for 2024, outcome counts by split, and a check for whether
  all five outcome classes are represented in each split), feature pipeline, trains both
  the `unweighted` and `class_balanced_comparison_only` variants side by side with core
  evaluation metrics. Same dataset-preference fallback as notebook 01.
- `03_calibration.ipynb` -- runs the full model comparison
  (`mlb_luck_score.models.compare_models`): unweighted vs. class-balanced vs.
  naive-prevalence vs. time-ordered post-hoc-calibrated, with calibration tables, expected
  calibration error (ECE) overall and by class, plots per variant, and an explicit
  recall-vs-calibration interpretation section. See "Model comparison and probability
  calibration" above for the real-data findings this notebook reproduces.
- `04_luck_score_demo.ipynb` -- **Version 0.2**: one out-of-sample 2024 example play
  end-to-end using the fixed run-value table and empirical percentile public score
  (predicted distribution, run values, raw contact luck in runs, empirical score,
  confidence report), a clearly-labeled Version 0.1 legacy comparison for the same play,
  and a 2024 reference-distribution summary (stats, quantiles, histograms).
- `05_park_aware_analysis.ipynb` -- **Version 0.3**: venue coverage and counts by park,
  `baseline_v02` vs. `park_aware_v03_candidate` metrics, calibration by park, and the 2024
  example plays whose predicted distribution changed the most after adding `venue_id`.

All five notebooks detect missing data/artifacts and print clear instructions instead of
crashing; `04` additionally falls back to a small synthetic example so it is always
runnable immediately after bootstrap.

## Time-based validation design

| Seasons   | Role                                  |
|-----------|---------------------------------------|
| 2021-2023 | Training                              |
| 2024      | Validation / model tuning             |
| 2025      | **Untouched final test** (protected)  |
| 2026      | Prospective evaluation                |

Rolling validation is also supported for iteration, e.g. train on 2021-2022 / validate
on 2023, then train on 2021-2023 / validate on 2024, by passing `--train-seasons` /
`--validation-seasons` explicitly.

For post-hoc probability calibration specifically, a separate, non-overlapping
three-way split is used (`mlb_luck_score.config.CALIBRATION_BASE_TRAIN_SEASONS` /
`CALIBRATION_FIT_SEASONS` / `CALIBRATION_EVAL_SEASONS`): base model on 2021-2022, the
calibration layer fit on 2023, evaluation on 2024. This is distinct from
`TRAIN_SEASONS`/`VALIDATION_SEASONS` above so a calibrator is never fit and evaluated on
the same rows.

## Untouched 2025 test rule

> **Never tune, iterate, or select features using 2025 results.**

This is enforced in code, not just documentation:
`mlb_luck_score.config.assert_seasons_allowed()` raises `ProtectedSeasonError` whenever
2025 appears in a season list, unless the caller explicitly passes
`allow_final_evaluation=True` (CLI: `--allow-final-evaluation`). Every season-aware
command calls this guard: the single-range downloader, the trainer, and both
`download_development_data` / `clean_development_data`. The development downloader has a
second layer of protection -- `mlb_luck_score.config.MLB_REGULAR_SEASON_DATE_RANGES` has
no entry for 2025 at all, so it can't be downloaded via that command even if the guard
were bypassed. `clean_development_data` also re-checks the *actual* `season` values found
in the combined output (not just the requested season list) before returning, as defense
in depth. Do not add a code path that uses 2025 without the explicit, one-time,
intentional flag.

## Version 0.2 scoring: fixed run values + empirical public score

Version 0.2 replaces both Version 0.1 placeholders below with empirically grounded ones.
**This is the current default** -- see `mlb_luck_score.scoring.contact_luck` and
`mlb_luck_score.scoring.empirical_score`.

**Fixed run-value table** (`mlb_luck_score.scoring.run_values.DEFAULT_RUN_VALUE_MAP`):
recovered from published FanGraphs Guts constants (2021-2024) via the standard
linear-weights identity `run_value(event) = (wOBA_event_weight - league_wOBA) /
wOBA_scale` (an out has `wOBA_event_weight = 0`), then averaged with equal weight across
the four seasons. Computed programmatically from the source constants, not hardcoded:

| outcome  | run value (runs) |
|----------|------------------:|
| out      | -0.254916 |
| single   |  0.463266 |
| double   |  0.763026 |
| triple   |  1.033068 |
| home_run |  1.400288 |

This table is **context-neutral**: it does not use a play's actual base/out state, runs
scored, win probability, or RE24 -- a bases-loaded triple and a bases-empty triple get the
identical value. That's a deliberate Version 0.2 scope choice (see `mlb_luck_score.scoring.
contact_luck` module docstring), not an oversight.

```
expected_run_value   = sum(predicted_probability[outcome] * run_value[outcome])
actual_run_value      = run_value[observed_outcome]
raw_contact_luck_runs = actual_run_value - expected_run_value
```

`raw_contact_luck_runs` is additive (in runs) -- see `mlb_luck_score.scoring.aggregation`
for season-level `total_raw_contact_luck_runs`, `positive_raw_luck_runs`,
`negative_raw_luck_runs`, and a per-100-eligible-events rate. **No leaderboard
qualification threshold is defined yet.**

**Empirical public score** (`mlb_luck_score.scoring.empirical_score.
compute_empirical_public_score`): maps `raw_contact_luck_runs` to a **signed percentile**
within a genuinely out-of-sample reference distribution -- the `unweighted` baseline's
predictions on 2024 validation data (trained only on 2021-2023; 2024 and 2025 never used
for training). Zero raw luck maps to exactly 0; positive raw luck maps to its percentile
among positive reference plays (0 to +100); negative raw luck maps to the percentile of
its magnitude among negative reference plays' magnitudes, negated (0 to -100); values
beyond the historical range are clipped to +-100. Build the reference artifact with:

```bash
make build-reference-score
# equivalent to:
python -m mlb_luck_score.models.build_reference_score \
    --input data/processed/cleaned_development_data.parquet --output-dir artifacts
```

This writes `artifacts/reference_score_v0.2.0.json` (git-ignored) with the model variant,
training/reference seasons, run-value table, sample counts, quantile tables, and source
metadata. **Never sum or average empirical public scores to build a season metric** -- the
mapping is non-linear and per-play; use raw contact luck (runs) instead (`mlb_luck_score.
scoring.aggregation`). The reference distribution is never refit on 2025.

## Park-aware model comparison (Version 0.3)

Statcast itself has no usable venue field, so venue metadata is recovered separately from
the public MLB Stats API (no key required) and joined by `game_pk`:

```bash
make download-game-metadata   # per-season game_pk/venue_id/venue_name/roof_type/surface_type
make join-venue-metadata      # left-join onto the cleaned development dataset, with a report
make compare-park-aware       # baseline_v02 vs park_aware_v03_candidate, see below
```

`is_neutral_site` is a derived heuristic (a game at a different venue than that team's most
common home park that season), not a raw API field -- verified against real 2021-2024 data,
it correctly identifies every known neutral-site game (the London Series, Mexico City
Series, Little League Classic, and the Rickwood Field tribute game). A small number of
real games (the two MLB "Field of Dreams" games, 2021 and 2022) have no venue at all in the
raw API response; the raw per-season cache always preserves that gap exactly as returned
(re-fetchable, never guessed at), but a small, hand-reviewed override table
(`mlb_luck_score.data.game_metadata_overrides`, git-tracked, not git-ignored -- this is
curated reference data, not downloaded data) is applied on top at load time
(`join_venue_metadata.apply_metadata_overrides`), giving both games a shared sentinel
`venue_id` (`-1`, guaranteed never to collide with a real MLB venue id) and the name "MLB
Field of Dreams (Dyersville, Iowa)". Real park geometry and GPS coordinates for that venue
are explicitly NOT populated yet -- documented fields on the override reserved for future
work. With the override applied, real 2021-2024 venue match rate is **100%** (up from
99.98% before the override), 0 conflicting mappings.

**`park_aware_v03_candidate`** is the exact same unweighted model and features as
`baseline_v02`, plus one additional categorical feature, `venue_id`
(`mlb_luck_score.models.compare_park_aware`). Real 2021-2024 results:

| variant | log loss | ECE | home-run ECE | accuracy (secondary) |
|---|---|---|---|---|
| `baseline_v02` | 0.670321 | 0.014413 | 0.002706 | 0.7515 |
| `park_aware_v03_candidate` | 0.668930 | 0.014167 | 0.002977 | 0.7509 |

Both overall log loss and overall ECE improve slightly. However, the automated
"material regression" rule (`find_material_venue_regressions`) uses a **scale-sensitive**
threshold, not a single large absolute cutoff: a reliably-sampled venue (at least
`min_venue_samples`, default 100, baseline rows) is flagged if its ECE worsens by more
than 0.01 in absolute terms, OR by more than 50% relative to its own baseline ECE --
whichever is more sensitive at that venue's scale. A single large absolute threshold
(e.g. 0.15) is inappropriate here: real per-venue ECE values run roughly 0.01-0.07, so
such a threshold would almost never fire and would miss real regressions that are small
in absolute terms but large relative to that venue's own baseline.

Applying that rule to the real data: **`Fenway Park` is flagged** -- its per-venue ECE
increases from 0.016032 to 0.025384 (+0.0094 absolute, **+58% relative**, on a reliable
4,103-row sample), crossing the relative margin even though it falls just under the
absolute one. Home-run ECE also gets slightly worse overall (0.002706 -> 0.002977). **The
automated recommendation is therefore `recommend_adopt_park_aware_v03: False`.** The
improvement is real but small, and it does not come for free -- it costs a material
calibration regression at one specific, well-sampled park.

Qualitatively, `venue_id` is still learning something real, not just noise: the five 2024
validation rows whose predicted distribution changed the most after adding venue are all
Coors Field plays, where the park-aware model shifts probability mass *away* from
home run and *toward* triple/double/out for hard-hit fly balls -- consistent with Coors
Field's famously oversized outfield (built deep specifically to counteract altitude-driven
carry), which is well known to convert would-be home runs elsewhere into extra-base hits.
The net effect is a genuine but mixed signal: real park-specific information, at the cost
of a real park-specific calibration regression elsewhere.

**Adoption rule: `park_aware_v03_candidate` has NOT been adopted as the default**, both per
the task's adoption rule and because the tightened, scale-sensitive threshold now correctly
flags a material regression. Per the task's adoption rule, a park-aware model is only ever
a candidate; switching the default requires an explicit, deliberate decision informed by
(not dictated by) the numbers above. See notebook `05_park_aware_analysis.ipynb` for the
full breakdown, including calibration by park and fly-ball/high-projected-distance subgroup
calibration. Published park factors (altitude, wall height/distance, prevailing wind) are
explicitly NOT used yet -- only the venue's bare categorical identity. If
`park_aware_v03_candidate` is ever adopted, build its reference artifact under a distinct
version (never overwriting `reference_score_v0.2.0.json`):

```bash
python -m mlb_luck_score.models.build_reference_score \
    --input data/processed/cleaned_development_data_with_venue.parquet \
    --output-dir artifacts --scoring-version 0.3.0 --extra-categorical-features venue_id
```

## Preliminary raw-luck definition (Version 0.1, LEGACY)

> Superseded by Version 0.2 above. Kept only for backward compatibility and explicit
> Version-0.1-vs-0.2 comparisons (see notebook `04_luck_score_demo.ipynb`'s "Legacy
> Version 0.1 comparison" section).

```
expected_value = sum(predicted_probability[outcome] * value[outcome])
actual_value    = value[observed_outcome]
raw_luck        = actual_value - expected_value
```

using the Version 0.1 ordinal value map `{out: 0, single: 1, double: 2, triple: 3,
home_run: 4}` (`mlb_luck_score.config.DEFAULT_VALUE_MAP`, configurable). This is a
research placeholder: it ignores base/out state, park, and win-expectancy context, and
the ordinal spacing (0/1/2/3/4) is not a validated run-value model -- Version 0.2 fixes
this with real run values. Raw luck IS additive by construction (see
`mlb_luck_score.scoring.raw_luck`, now legacy).

## Why the legacy public score is not additive (Version 0.1, LEGACY)

> Superseded by the empirical public score above.

The legacy public -100..+100 score (`mlb_luck_score.scoring.public_score.
raw_luck_to_public_score`) applies a bounded, monotonic, sign-preserving transform
(`100 * tanh(raw_luck / scale)`) to raw luck so it fits a stable, intuitive display range.
That transform is non-linear, so summing public scores across plays does **not** equal
the public score of the summed raw luck -- if you need an additive quantity (e.g. season
aggregation), use raw luck, not the public score. This arbitrary functional form is why
Version 0.2 replaced it with the empirical percentile mapping above.

## Confidence limitations

The Version 0.1 confidence report (`mlb_luck_score.scoring.confidence`) describes
**observable data completeness only** -- e.g. what fraction of core contact fields were
present, whether spray direction/park/alignment data was available. It is **not** a
statistical uncertainty estimate, and a low-completeness label must never be used to pull
a raw-luck or public-score value toward zero (that would silently redefine what the score
means). Historical support, model disagreement, sensitivity analysis, calibration
uncertainty, and bootstrap/posterior intervals are reserved for future work.

## Data licensing and redistribution caution

Statcast data is provided publicly by MLB Advanced Media / Baseball Savant via
`pybaseball`. Venue/roof/surface game metadata (Version 0.3) is provided publicly by the
MLB Stats API (`statsapi.mlb.com`, no key required). This repository does not redistribute
any downloaded data (see `.gitignore`); downloaded files stay local under `data/raw/`,
`data/interim/`, and `data/processed/`, all git-ignored. Review MLB Advanced Media/
Baseball Savant's and the MLB Stats API's terms before redistributing any derived data
outside this repository.

## Reproducibility

- Deterministic random seeds are set where applicable (`RANDOM_SEED = 42` in
  `train_contact_model.py`).
- Every saved model artifact is accompanied by a metadata JSON recording model version,
  training seasons, feature list, class order, training timestamp, and package versions.
- `event_id` is built deterministically from stable identifiers
  (`game_pk-at_bat_number-pitch_number`), not a random hash.

## Current status

Data acquisition (one-week sample and full 2021-2024 development dataset, both
downloaded and cleaned against real Statcast data), cleaning, feature engineering, a
baseline model, calibration diagnostics, the Version 0.2 fixed run-value table, raw
contact luck, the empirical percentile public score, season-aggregation primitives, and a
data-completeness report are all implemented and covered by offline synthetic tests. The
real 2021-2024 dataset has been trained, evaluated, and used to build the real Version
0.2 reference artifact; see "Model comparison and probability calibration" and "Version
0.2 scoring" above for the real, honestly-reported results -- including a confirmed
severe miscalibration issue in an earlier `class_weight="balanced"` variant, now fixed by
defaulting to the unweighted variant. Notebook `04_luck_score_demo.ipynb` has been run
against real data and shows real Version 0.2 raw-luck and empirical-score values for a
genuinely out-of-sample 2024 example play. Venue metadata (Version 0.3) has been
downloaded and joined for real 2021-2024 data (99.98% match rate); the park-aware model
comparison has been run for real -- see "Park-aware model comparison (Version 0.3)" above
for the exact results. `park_aware_v03_candidate` is a documented candidate, not adopted
as the default.

## Future work

- Published park factors (altitude, wall height/distance, prevailing wind) -- Version 0.3
  uses only the venue's bare categorical identity, not park-specific physical factors
- Weather normalization and air-density modeling
- Park geometry effects
- Exact defender positioning, reaction, and route modeling
- Fielding and throwing execution modeling
- Batter-runner decision-quality modeling
- Batter-runner execution modeling
- Park and bounce effects
- Context-aware run values (using actual base/out state, win probability, or RE24,
  instead of Version 0.2's fixed context-neutral table)
- Shapley-style attribution across luck components
- Confidence intervals (historical support, model disagreement, sensitivity analysis,
  calibration uncertainty, bootstrap/posterior intervals)
- Leaderboard qualification thresholds (season-aggregation primitives exist -- see
  `mlb_luck_score.scoring.aggregation` -- but no minimum-eligible-events threshold is
  defined yet)
- Rare-play rule expansion (the eligible-event list is a v0.1 starting point, not final)

---

See `CLAUDE.md` / `AGENTS.md` for persistent rules that govern how this repository should
be extended by AI coding agents.
