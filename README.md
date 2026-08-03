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

The **production** model's expected-outcome distribution (`baseline_v02`) is fit only on
the contact event's own physical characteristics (exit velocity, launch angle,
approximate spray direction, batted-ball type). It does not model exact fielder
positioning, reaction time, route efficiency, or throwing execution. A ball that "should"
be a double but becomes a single because of an excellent defensive play will look
identical to a double turned single by bad luck.

Version 0.6 (`mlb_luck_score.models.compare_alignment_aware`) evaluated whether adding the
defense's coarse, pre-pitch STARTING alignment (never execution -- reaction, route,
pickup, transfer, throw remain entirely unmodeled, public data cannot observe them at
all) improves the model; see "Alignment-aware positioning (Version 0.6)" below. Neither
candidate was adopted, so the production model's blind spot described above is currently
unchanged.

Version 0.7A/0.7B (`mlb_luck_score.models.compare_opportunity_models`/`mlb_luck_score.
scoring.defensive_execution`) is the first phase to actually separate opportunity
difficulty from execution, but ONLY for outfield air balls and ONLY at the
"was the opportunity converted" level -- reaction time, route efficiency, and throwing
accuracy remain entirely unmodeled (unavailable in public data; see "Outfield opportunity
and execution (Version 0.7A / 0.7B)" below for the full audit). Infield pickup/throwing
execution and batter-runner advancement are explicitly deferred to future phases.

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
- `06_park_geometry_analysis.ipynb` -- **Version 0.4**: reviewed geometry schema/coverage,
  wall-profile visuals per park, spray-angle orientation verification, interpolation examples,
  the four-way `baseline_v02`/`venue_only_v03_candidate`/`geometry_only_v04_candidate`/
  `venue_plus_geometry_v04_candidate` comparison, calibration by outcome/venue, near-wall
  subgroup calibration, paired-bootstrap intervals, example plays, and the adoption
  recommendation. See "Park geometry (Version 0.4)" above for the real-data results.
- `07_weather_air_density_analysis.ipynb` -- **Version 0.5**: weather-source/station-mapping
  overview, coverage by season/venue, observation-time-offset distribution, roof-status
  coverage, air-density formula/distribution, the four-way `selected_production_baseline`/
  `weather_basic_v05_candidate`/`weather_vector_v05_candidate`/
  `geometry_plus_weather_v05_candidate` comparison, calibration by outcome/venue/roof
  /environmental subgroup, paired-bootstrap intervals, actual-vs-standardized environment
  examples, and the adoption recommendation. See "Weather and air density (Version 0.5)" above
  for the real-data results.
- `08_alignment_positioning_analysis.ipynb` -- **Version 0.6**: alignment-label coverage,
  interaction-feature examples, the three-way `baseline_v02`/`alignment_labels_v06`/
  `alignment_interactions_v06` comparison, required-subgroup and per-venue calibration,
  plays whose probabilities changed most, controlled-perturbation directional checks (with
  interpretation of the confounding-by-indication finding), positioning-counterfactual
  stability, the `positioning_effect` distribution, the adoption recommendation, and
  explicit limitations. See "Alignment-aware positioning (Version 0.6)" above for the
  real-data results.
- `09_outfield_opportunity_execution_analysis.ipynb` -- **Version 0.7A/0.7B/0.7C**:
  outfield-opportunity eligibility coverage, estimated-hang-time/landing-location feature
  examples, `measured_contact_only_v07` training/evaluation, required-subgroup/per-venue
  /per-defender calibration, a game-level bootstrap CI, the validation summary (including
  why `typical_position_proxy_v07` was not built), defensive-execution examples, the
  four-component per-play report (contact expectation, opportunity difficulty, execution,
  residual contact luck -- explicitly not combined); then the Version 0.7C near-wall
  specialist: gate distribution, model selection (2023) and final comparison (2024),
  required wall-band/wall-height/spray-sector/opportunity-time/venue subgroup calibration,
  the paired bootstrap, controlled-perturbation checks (with the launch-angle finding
  interpreted), the no-open-field-regression check, the validation summary, and the gated
  report's `execution_availability_status` breakdown; and explicit limitations for all
  three sub-versions. See "Outfield opportunity and execution (Version 0.7A / 0.7B)" and
  "Near-wall opportunity correction (Version 0.7C)" above for the real-data results.

All nine notebooks detect missing data/artifacts and print clear instructions instead of
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

## Park geometry (Version 0.4)

Version 0.3 asked "which venue was this play in?" (a bare categorical identity). Version 0.4
asks the more specific question: "given this ball's spray direction and projected travel,
what wall distance and wall height were relevant?"

**Geometry reference data** (`mlb_luck_score.data.park_geometry.PARK_GEOMETRY_POINTS`) is
small, hand-curated reference data checked into git -- like `run_values.py` and
`game_metadata_overrides.py`, not a downloaded dataset. It describes each of the 30 primary
2021-2024 MLB venues' outfield wall as five discrete points (left-field line, left-center
alley, straightaway center, right-center alley, right-field line) at spray angles `-45 /
-22.5 / 0 / +22.5 / +45` degrees -- the same sign convention and fair-territory boundary
`clean_batted_balls.spray_angle_approx` already uses (0 = center, negative = third-base/left
side, positive = first-base/right side; verified in `tests/test_park_geometry.py`). Wall
distance is populated for essentially every point; wall height only where a source explicitly
gave one for that location. Angles strictly between reviewed points are piecewise-linearly
interpolated (`interpolate_wall_geometry`); angles outside +-45 degrees are never extrapolated
-- treated as outside the modeled range.

**Sourcing is important to be honest about**: every record was gathered by an AI coding agent
(Claude Code) from publicly cited sources (mostly individual-ballpark Wikipedia articles,
cross-checked against independent web searches where a figure looked internally inconsistent
-- one such inconsistency, in Kauffman Stadium's dimensions, was caught and corrected this
way). Every record's `review_status` is `agent_sourced_pending_human_review` -- **this is not
the same as human review**, and should be treated as a documented Version 0.4 research
placeholder, not validated ground truth, until a maintainer spot-checks it (see the module
docstring for the full caveat list, including known simplifications like Citizens Bank Park's
zigzag "Monty's Angle" and Citi Field's former right-field "nook" being folded into the
nearest standard point).

**Three venues have more than one geometry configuration** because their real outfield walls
changed during 2021-2024, each a cited, dated renovation: Oriole Park at Camden Yards (left
field moved back and raised before Opening Day 2022), Rogers Centre (walls moved in and
heights staggered before 2023), and Comerica Park (fences adjusted, and a previously-posted
distance corrected via laser measurement, before 2023). Configuration resolution
(`resolve_geometry_config`) is date-based and never guesses the nearest configuration for an
uncovered date -- it returns "unavailable" instead (see `ParkGeometryValidationError` for the
overlap/conflict checks enforced at import time).

**Temporary and neutral-site venues have NO geometry** by design
(`TEMPORARY_OR_SPECIAL_VENUE_IDS`): the MLB Field of Dreams games, the 2021 Blue Jays
alternate home venues (Sahlen Field, TD Ballpark), the London Series, the Mexico City Series,
the Little League Classic field, and the 2024 Rickwood Field tribute game. These rows are
retained in baseline analyses with `has_park_geometry=False` and an explicit `geometry_status`
reason, never dropped or given fabricated geometry.

```bash
make build-park-geometry      # validates the reviewed table and prints a coverage summary
make join-park-geometry       # joins geometry onto the venue-joined dataset by
                               # venue_id + game_date + spray_angle_approx, with a report
make compare-geometry-aware   # 4-way comparison, see below
```

**Per-play join** (`mlb_luck_score.data.join_park_geometry`) adds `wall_distance_in_spray_
direction`, `wall_height_in_spray_direction`, `projected_distance_to_wall_margin`
(`hit_distance_sc - wall_distance_in_spray_direction`; positive means the measured distance
exceeded the modeled wall distance along that direction -- this is NOT alone proof of a home
run, since wall height, trajectory, spin, and measurement error remain unmodeled),
`near_wall_5ft`/`10ft`/`20ft`, `projected_beyond_wall`, `high_wall_indicator`, and an explicit
`geometry_status` reason for every row (`missing_venue_id`, `missing_venue_metadata`,
`temporary_or_special_venue`, `missing_game_date`, `no_geometry_reviewed_for_venue_or_date`,
`missing_spray_angle`, `outside_modeled_angular_range`, or `ok`). On the real 2021-2024
dataset: **89.80% of rows** (443,785 / 494,173) resolve usable geometry; the remainder is
47,012 rows outside the +-45 degree modeled angular range, 3,147 rows at temporary/special
venues, and 229 rows with a missing spray angle -- zero rows lost to missing venue metadata
(match rate is 100%, per Version 0.3 above).

**Four-way model comparison** (`mlb_luck_score.models.compare_geometry_aware`) trains
`baseline_v02`, `venue_only_v03_candidate`, `geometry_only_v04_candidate` (baseline +
geometry, NO `venue_id`), and `venue_plus_geometry_v04_candidate` (baseline + `venue_id` +
geometry) on IDENTICAL 2021-2023 training rows, evaluated on the SAME untouched 2024
validation rows (n=122,132; real results, reproduced 2026-07-27):

| variant | log loss | ECE | home-run ECE | accuracy (secondary) |
|---|---|---|---|---|
| `baseline_v02` | 0.670321 | 0.014413 | 0.002706 | 0.7515 |
| `park_aware_v03_candidate` | 0.668930 | 0.014167 | 0.002977 | 0.7509 |
| `geometry_only_v04_candidate` | **0.563283** | 0.016284 | **0.001019** | 0.7779 |
| `venue_plus_geometry_v04_candidate` | **0.559193** | 0.016177 | **0.001074** | 0.7792 |

Both geometry candidates cut log loss by roughly 16% relative to `baseline_v02` and cut
home-run ECE by roughly 62% -- a genuinely large improvement, and it holds up under a paired,
`game_pk`-level bootstrap (500 replicates, seed 42, resampling GAMES with replacement so
within-game correlation is preserved; both the baseline and each candidate are scored on the
exact same resampled rows each replicate): `geometry_only_v04_candidate`'s log-loss delta is
-0.107 (95% CI **[-0.109, -0.104]**, entirely below zero) and `venue_plus_geometry_v04_
candidate`'s is -0.111 (95% CI **[-0.114, -0.108]**) -- the improvement is not noise.

The clearest and most physically meaningful signal is in the **near-wall subgroup**: for plays
within 5 ft of the modeled wall distance (n=1,952, the hardest calibration regime -- exactly
where "how far the ball went" alone is most ambiguous about the outcome), ECE drops from
**0.1181 (`baseline_v02`) to 0.0323 (`geometry_only_v04_candidate`)**, a ~73% relative
improvement, with the same pattern (smaller but still substantial) at 10 ft and 20 ft. This is
exactly the kind of effect physical wall geometry should produce, and it is not achievable
from bare venue identity alone -- `park_aware_v03_candidate` has no way to know how close a
specific batted ball came to a specific wall.

This improvement does **not** come for free. Applying the Version 0.3 material-venue-
regression rule (reliably-sampled venue, ECE worsens by >0.01 absolute OR >50% relative)
flags **`loanDepot park`** (venue_id 4169; a reliable 4,304-row sample; ECE 0.013983 ->
0.021972, +57% relative) for `geometry_only_v04_candidate`, and **`Estadio Alfredo Harp Helu`**
(venue_id 5340, the Mexico City Series venue; n=109; ECE 0.038569 -> 0.067957/0.076984) for
BOTH candidates -- note this venue has no reviewed geometry at all (it's in
`TEMPORARY_OR_SPECIAL_VENUE_IDS`), so this regression is a real, honestly-reported side effect
of shared model weights changing overall once geometry features enter the (single, joint)
logistic regression, not evidence that its own geometry was wrong. Every other reliably-
sampled venue is unaffected or improved; no near-wall or high-distance subgroup regresses
materially.

**Adoption rule: neither `geometry_only_v04_candidate` nor `venue_plus_geometry_v04_candidate`
has been adopted as the default.** `recommend_geometry_adoption` checks 7 of the task's 8
adoption criteria automatically (log-loss improvement, bootstrap-supported improvement,
overall/home-run/near-wall-subgroup ECE not materially worse, no material venue regression,
sufficient geometry coverage) and both candidates fail on the single material-venue-regression
criterion above -- `recommend_adopt_any_v04_candidate: False`. The 8th criterion ("the model
is learning physically plausible effects") is deliberately **not automated**; see notebook
`06_park_geometry_analysis.ipynb` for the qualitative check (top probability-change examples
compared against `projected_distance_to_wall_margin`). `baseline_v02` remains the production
default; both v0.4 candidates are retained as evaluated, documented infrastructure.

```bash
python -m mlb_luck_score.models.build_reference_score \
    --input data/processed/cleaned_development_data_with_geometry.parquet \
    --output-dir artifacts --scoring-version 0.4.0 \
    --extra-categorical-features venue_id wall_segment_label near_wall_5ft near_wall_10ft \
        near_wall_20ft projected_beyond_wall temporary_or_special_venue geometry_uncertain
```

(shown for completeness -- since no v0.4 candidate is adopted, this has NOT been run to
produce a real `reference_score_v0.4.0.json`; if a v0.4 candidate is ever adopted, build its
reference artifact under this distinct version, never overwriting `reference_score_v0.2.0.json`
or `v0.3.0.json`.)

**Limitations**: geometry data is agent-sourced pending human review (see above); each park-era
is a 5-point piecewise-linear simplification, not a true wall curve; wall height is sparse and
never guessed; no published park factors, weather, air density, or roof-state modeling; a
positive `projected_distance_to_wall_margin` is not proof of a home run. Contact Luck v0.4
remains a contact-only research prototype -- it does not yet model weather, air density, roof
status, exact ball trajectory, defensive positioning, defensive execution, or batter-runner
advancement. See notebook `06_park_geometry_analysis.ipynb` for the full breakdown (schema
overview, wall-profile visuals, interpolation examples, coverage by season/venue/batted-ball
type, four-way comparison, calibration by outcome/venue, near-wall subgroup calibration, paired
-bootstrap intervals, example plays, and this same adoption recommendation).

## Weather and air density (Version 0.5)

Version 0.5 asks whether actual game-time atmospheric conditions -- temperature, humidity,
pressure, wind, and roof status, combined into a physically derived air density -- improve
probability quality beyond the currently-selected production baseline (`baseline_v02`; as of
this writing, neither the Version 0.3 venue-only nor either Version 0.4 geometry candidate was
adopted, so `selected_production_baseline` == `baseline_v02`).

**Two independent, complementary sources**, both real, historical, and publicly documented (no
API key required for either):

1. **MLB Stats API schedule weather** (`/schedule?hydrate=weather`) -- the SAME endpoint and
   chunked-date-range pattern already used for Version 0.3 venue metadata, just with one extra
   query parameter. Gives official per-game temperature, wind, and condition text. Critically,
   **MLB's wind field is already expressed relative to the park's own orientation** (e.g. `"8
   mph, Out To CF"`, `"5 mph, L To R"`, `"0 mph, Calm"`) -- this repository never needed to
   independently source true-north field-orientation angles per park (a documented, deliberately
   deferred gap the task allowed), because MLB has effectively already done that rotation.
2. **Iowa Environmental Mesonet ASOS archive** (`mesonet.agron.iastate.edu`) -- the ONLY source
   of humidity and pressure in this repository (MLB's own field has neither). Routine hourly
   observations (`report_type=3`) from each of the 30 primary venues' nearest reliable station,
   verified working via a live query before being adopted (`mlb_luck_score.data.
   venue_environment`). Real 2021-2024 download: **4 schedule-weather files + 116 station-season
   files (29 unique stations x 4 seasons, two New York ballparks share LaGuardia), zero
   failures.**

**Venue-environment reference data** (`mlb_luck_score.data.venue_environment.
VENUE_ENVIRONMENTS`) is small, hand-curated reference data checked into git, following the exact
same pattern as `park_geometry.py`: approximate venue coordinates, elevation, IANA timezone, and
assigned weather station, with every station's own coordinates/elevation independently verified
against a live query (not guessed) and `review_status="agent_sourced_pending_human_review"` on
every record. **Coors Field is the one venue with a stadium-specific (not station-proxy)
elevation** (1609.344 m -- the famous "mile-high" figure), since its own assigned station
(Buckley SFB) is at a materially different elevation and this is exactly the park where altitude
matters most. All 30 assigned stations are within **25 km** of their venue; roof/retractable
-roof classification is NOT duplicated here -- it reuses Version 0.3's existing per-game
`roof_type` field combined with the schedule-weather condition text (see below).

**Roof handling** (`mlb_luck_score.data.build_game_weather.classify_roof_status`) distinguishes
`outdoor_open_air`, `retractable_roof_open`, `retractable_roof_closed`, `fixed_indoor`, and
`roof_status_unknown` -- a closed roof is NEVER inferred from precipitation or any other
condition text, only an explicit `"Roof Closed"` string. For closed/indoor games, raw external
weather is preserved but `effective_temperature_c`/`effective_relative_humidity_pct`/
`effective_pressure_hpa`/`air_density_kg_m3` are left null (no reviewed indoor-climate assumption
exists in this repository); only `effective_wind_speed_mps=0.0` is set, since a closed roof
genuinely blocking outdoor wind is a physical certainty, not a guess.

**Observation-time matching**: each game is matched to the nearest ASOS observation at or before
its scheduled UTC start time (falling back to shortly after only if none qualifies before), with
the exact offset preserved and a game NEVER matched to an observation farther than 3 hours away
-- real data: **coverage 98.6% "good" (<=60 min), 1.1% "fair" (<=180 min\), the remainder
unmatched**.

**Air density** (`mlb_luck_score.data.weather_physics.moist_air_density_kg_m3`) uses the ideal
gas law applied separately to dry-air and water-vapor partial pressures (Arden Buck vapor
-pressure equation for the humidity term) -- a documented moist-air formula, not a
temperature-only proxy. Station sea-level pressure is converted to venue-elevation pressure via
the hypsometric equation. Verified against textbook values: standard sea-level conditions
(15C/1013.25 hPa/0% RH) give exactly the ICAO ISA reference density (1.225 kg/m^3, used as the
fixed comparison reference throughout); Coors Field's estimated conditions come out roughly
15-20% less dense, matching the well-known real-world figure.

**Per-play join** (`mlb_luck_score.data.join_weather_features`) computes following/head/crosswind
relative to EACH PLAY's own spray direction (not just the park's center-field axis) by rotating
the game-level wind vector -- real coverage: **99.36% of rows have some weather data, 81.26%
have full EFFECTIVE (actionable) conditions** (443,785 -> the remainder is mostly indoor/closed
-roof games, where effective conditions are correctly left unavailable, not fabricated).

**Four-way model comparison** (`mlb_luck_score.models.compare_weather_aware`) trains
`selected_production_baseline`, `weather_basic_v05_candidate` (temperature/humidity/air
density/roof status), `weather_vector_v05_candidate` (adds air-density deviation from reference
and following/head/crosswind), and `geometry_plus_weather_v05_candidate` (weather-vector +
Version 0.4 geometry) on IDENTICAL 2021-2023 rows, evaluated on the SAME untouched 2024
validation rows (n=122,132; real results):

| variant | log loss | ECE | home-run ECE | accuracy (secondary) |
|---|---|---|---|---|
| `selected_production_baseline` | 0.670321 | 0.014413 | 0.002706 | 0.7515 |
| `weather_basic_v05_candidate` | 0.670015 | 0.014500 | 0.003401 | 0.7518 |
| `weather_vector_v05_candidate` | 0.670038 | 0.014391 | 0.003409 | 0.7516 |
| `geometry_plus_weather_v05_candidate` | 0.562149 | 0.016042 | 0.001047 | 0.7784 |

`geometry_plus_weather_v05_candidate`'s large log-loss improvement is almost entirely the
already-documented Version 0.4 geometry effect (see "Park geometry (Version 0.4)" above), not a
new weather finding -- and it inherits geometry's material venue regressions (loanDepot park,
**newly also Truist Park**, and the Mexico City neutral-site venue), so it fails the adoption
rule for the same reason `geometry_only_v04_candidate` did.

The two pure-weather candidates' log-loss improvement is REAL but **tiny**: paired game_pk-level
bootstrap (500 replicates, seed 42) gives `weather_basic_v05_candidate` a delta of -0.000307
(95% CI **[-0.000525, -0.000109]**, entirely below zero) and `weather_vector_v05_candidate`
-0.000283 (95% CI **[-0.000504, -0.000076]**) -- statistically distinguishable from zero at
n=122,132 rows, but a ~0.045% relative change, not a practically large effect. Home-run ECE gets
slightly WORSE for both (+0.00069 / +0.00070, roughly +26% relative) -- below the material
-regression threshold (0.01 absolute / 50% relative) but a real, honestly-reported tradeoff, not
nothing. No reliably-sampled venue and no roof/weather-quality/wind/air-density subgroup
triggers a material regression for either pure-weather candidate; weather coverage (80.4% of
2024 validation rows) is sufficient and transparently reported; complete-case vs. all-row ECE
shows no evident selection bias.

**`recommend_adopt_any_v05_candidate: True`, best candidate `weather_basic_v05_candidate`** --
both pure-weather candidates pass every one of the 8 AUTOMATABLE adoption criteria.

**However, this repository's adoption rule explicitly reserves criterion 9 -- "the model is
learning physically plausible effects" -- for human judgment, never automation, and that check
does NOT support adoption here.** Two real implementation bugs were caught specifically by
attempting this qualitative check (see "Real bugs found and fixed" in the notebook and this
file's history) and, after fixing them, the corrected per-play weather attribution
(`mlb_luck_score.scoring.weather_attribution`) shows only a WEAK and partly WRONG-SIGNED
relationship between actual weather and its predicted effect: the correlation between
`weather_run_value_effect` and `air_density_kg_m3` across 2024 validation rows is **+0.026**
(expected clearly negative -- thinner air should favor the batter) and with `following_wind_mps`
is **-0.086** (expected clearly positive -- a tailwind should favor the batter). Coors Field's
mean attributed effect is slightly NEGATIVE, the opposite of the well-documented real-world
thin-air effect. The most likely explanation is multicollinearity: `temperature_c`, `pressure_
hpa`, `humidity_pct`, and `air_density_kg_m3` are numerically near-redundant (density is a
deterministic function of the other three) as inputs to a linear model, which is well known to
produce unstable, sometimes sign-flipped individual coefficients even when the model's aggregate
calibration is fine -- consistent with the tiny-but-real aggregate log-loss improvement above
coexisting with an unreliable per-play decomposition.

**Adoption rule: neither weather candidate has been adopted as the default.** `baseline_v02`
remains the production model. This is a deliberate exercise of the "never a substitute for
judgment" principle already established for Version 0.3/0.4 -- the automated criteria are a
starting point, and here they would have recommended adopting a model whose per-play weather
attribution does not hold up under direct physical inspection. The weather pipeline
(download/build/join/compare/attribution) is retained as evaluated, tested infrastructure.

```bash
make download-weather-data     # 2021-2024 only, refuses 2025, resumable
make build-game-weather        # game_pk-keyed table, time/station matching, roof handling
make join-weather-features     # per-play join, following/head/crosswind by spray direction
make compare-weather-aware     # 4-way comparison, see above
```

**Limitations**: venue-environment coordinates are approximate and agent-sourced pending human
review; humidity/pressure come from a station up to 25 km away, not an on-site sensor; indoor
-climate conditions are never fabricated (left unavailable) for closed-roof/domed games (~18% of
plays); true stadium field orientation was never independently sourced -- this repository relies
entirely on MLB's own pre-rotated wind text; the standardized-environment counterfactual is a
fixed ISA reference, not a claim about "typical" game weather; and -- per the finding above --
the weather-feature set as currently specified (four numerically collinear atmospheric variables)
produces a per-play attribution too unstable to trust, a concrete, actionable target for future
work (e.g. dropping `air_density_kg_m3` in favor of its three components, or vice versa, or
regularizing the model). Contact Luck v0.5 remains a contact-only research prototype -- it does
not yet include exact defensive positioning, defensive execution, batter-runner advancement, or
full multi-factor Shapley attribution. See notebook `07_weather_air_density_analysis.ipynb` for
the full breakdown.

## Weather correction (Version 0.5.1)

Version 0.5's `weather_vector_v05_candidate` combined `air_density_kg_m3` with the raw
`temperature_c`/`humidity_pct`/`pressure_hpa` used to derive it -- since density is a
near-deterministic function of those three variables, this created severe multicollinearity.
The Version 0.5 physical-plausibility check caught exactly this: a real, tiny, bootstrap
-confirmed log-loss improvement paired with a per-play weather attribution that correlated the
WRONG direction with both air density and following wind.

Version 0.5.1 tests three INDEPENDENT (not cumulative) corrected feature sets
(`mlb_luck_score.models.compare_weather_variants`), each using only ONE representation of the
temperature/humidity/pressure/density family:

- **`density_only_v051_candidate`**: `air_density_kg_m3` + wind components + roof status. NO
  temperature/humidity/pressure.
- **`components_only_v051_candidate`**: temperature + humidity + pressure + wind components +
  roof status. NO derived air density.
- **`density_anomaly_v051_candidate`**: each venue's air-density ANOMALY -- actual density minus
  that venue's own **training-season-only** normal density (`mlb_luck_score.data.
  join_weather_features.compute_venue_air_density_baseline` / `add_venue_air_density_anomaly`,
  a "fit on train, apply to all" statistic, exactly like a scaler, so validation-season weather
  never leaks into a venue's own baseline) -- + wind components + roof status. NO raw density.
  Meant to separate "was today unusual weather for this park" from "this park is persistently
  high/low altitude" (Coors Field's raw density is almost always low, so it mostly just encodes
  venue identity, not day-specific weather).

**Adoption rule -- a candidate is never recommended on a log-loss improvement alone, however
small or statistically significant** (`mlb_luck_score.models.compare_weather_variants.
recommend_variant_adoption`). Beyond the usual log-loss/bootstrap/ECE/venue-regression/coverage
checks, every candidate must ALSO pass **controlled-perturbation directional checks**
(`mlb_luck_score.models.weather_perturbation`): override real validation rows to controlled
low/high density (or anomaly) and strong-following/strong-headwind scenarios (holding
everything else, including roof status, fixed), predict with the candidate's own trained model,
and check whether the AGGREGATE mean predicted home-run probability moves in the physically
expected direction -- plus, for the two density-based candidates, a Coors Field-specific check
(real Coors rows' actual thin-air prediction vs. a hypothetical denser-air counterfactual for
the SAME rows). These checks are combined with the rest via AND, not OR.

**Real 2024 validation results** (n=122,132; 500-replicate paired game_pk-level bootstrap, seed
42):

| variant | log loss | log-loss delta | 95% CI | bootstrap-supported |
|---|---|---|---|---|
| `selected_production_baseline` | 0.670321 | -- | -- | -- |
| `density_only_v051_candidate` | 0.670192 | -0.000130 | [-0.000313, +0.000048] | No (crosses zero) |
| `components_only_v051_candidate` | 0.670032 | -0.000290 | [-0.000512, -0.000077] | Yes |
| `density_anomaly_v051_candidate` | 0.670306 | -0.000015 | [-0.000106, +0.000086] | No (crosses zero) |

**Controlled-perturbation results** -- mean predicted P(home run) under each scenario:

| candidate | density direction (low density -> should be HIGHER) | wind direction (following -> should be HIGHER than headwind) | Coors: actual (thin) vs. hypothetical dense |
|---|---|---|---|
| `density_only` | **BACKWARDS**: 0.038 (thin) vs 0.054 (dense) | backwards (tiny): 0.0456 vs 0.0444 | **BACKWARDS**: 0.050 (actual) vs 0.073 (hypothetical dense) |
| `components_only` | backwards (tiny): 0.0461 (hot/thin-implied) vs 0.0470 (cold/dense-implied) | backwards (tiny): 0.0460 vs 0.0444 | n/a (no single density column) |
| `density_anomaly` | **CORRECT**: 0.047 (low anomaly) vs 0.042 (high anomaly) | ~zero, wrong sign: 0.0449 vs 0.0449 | **CORRECT**: 0.062 (actual) vs 0.059 (hypothetical denser) |

The venue-anomaly formulation **does fix the backwards density signal** that both raw-density
-based candidates show (validating the hypothesis that raw `air_density_kg_m3` mostly just
encodes venue identity, e.g. "is this Coors", rather than day-specific weather) -- a genuine,
interpretable, non-buggy finding, not noise (`n=122,132`, consistent sign, consistent with the
known real-world Coors Field effect). Wind, however, shows no reliably-signed effect in ANY of
the three candidates (deltas are tiny and sign-inconsistent) -- most likely reflecting genuinely
weak marginal wind signal once launch-condition features are already in the model, though a
regularized default `LogisticRegression` (this repository's baseline throughout) shrinking a
weak feature toward zero is a plausible contributing factor worth a future check.

**`recommend_adopt_any_v051_candidate: False`.** `components_only_v051_candidate` is the only
candidate whose log-loss improvement clears the bootstrap-significance bar, but it fails the
perturbation checks (backwards density and wind direction). `density_only` and `density_anomaly`
don't even clear bootstrap significance. **No candidate is adopted; `baseline_v02` remains the
production model.** Per the task's explicit fallback, weather is retained as evaluated,
tested infrastructure and marked **unresolved** rather than forced into adoption.

```bash
make compare-weather-variants   # 3-way corrected comparison, see above
```

**Next steps**: `density_anomaly_v051_candidate`'s correctly-signed density/Coors effects are
the most promising lead in this line of work -- a follow-up could try combining it with a wind
feature engineered/verified independently (or dropping wind entirely and re-testing density
-anomaly alone, since wind's failure is currently blocking an otherwise-plausible candidate),
and/or trying an unregularized or explicitly-tuned model to rule out coefficient shrinkage as
the cause of wind's null signal.

## Alignment-aware positioning (Version 0.6)

Estimates whether the defense's **starting alignment** (not execution) changed the
expected outcome of a batted ball. Deliberately kept separate from defensive
**execution** (reaction, route, pickup, transfer, throw) -- public data cannot observe
execution at all. With only Statcast's `if_fielding_alignment`/`of_fielding_alignment`
labels available, this is scoped as **alignment-aware positioning**, not exact defender
positioning: coarse, pre-pitch category labels only (infield: `Standard`/`Strategic`/
`Infield shift`/`Infield shade`; outfield: `Standard`/`Strategic`/`4th outfielder`) --
never exact coordinates, pre-contact movement, reaction time, route efficiency, or a
judgment of whether the chosen alignment was strategically appropriate. Real coverage is
~99.6% non-null in every 2021-2024 season for both columns.

Three controlled variants (`mlb_luck_score.models.compare_alignment_aware`), IDENTICAL
2021-2023 training rows / 2024 validation rows, `class_weight=None` throughout:

- **`baseline_v02`**: the currently-selected production model, unchanged.
- **`alignment_labels_v06`**: `baseline_v02` + the two raw alignment labels.
- **`alignment_interactions_v06`**: labels + physically-motivated interaction terms --
  infield alignment x batter handedness, infield alignment x spray direction, outfield
  alignment x launch angle, outfield alignment x projected distance, and infield-shift x
  pull-side-ground-ball.

**`position_depth_v06` (baseline + public per-fielder positioning coordinates/depths) was
investigated and is NOT implemented**: no reliable, publicly downloadable per-play or
per-fielder-position coordinate/depth dataset was found to exist -- Baseball Savant's
public data exposes only the coarse alignment labels used above, not exact or average
fielder (x, y) positions joinable to individual plays. This is exactly the public-data
limitation this section's framing anticipated -- a scope gap, not a decision to exclude
numeric depth data on purpose.

**Adoption rule (7 criteria, all combined with AND)**: (1) 2024 log-loss improves; (2) a
paired game_pk-level bootstrap (500 reps, seed 42) supports the improvement; (3) overall
ECE does not materially worsen; (4) no major batted-ball-type/pull-oppo/shift-status/
handedness/base-state subgroup materially regresses; (5) no reliably-sampled venue
triggers the existing venue-regression rule; (6) controlled-perturbation directional
checks (`mlb_luck_score.models.weather_perturbation`, reused as-is -- it is generic
despite the module name) behave in the physically expected direction; (7) the positioning
counterfactual is stable (well-formed predictions on genuinely re-derived "typical
alignment" rows, AND `positioning_effect` is ~0 by construction for rows whose actual
alignment already IS the typical one). Per the task's explicit constraint, **the model is
never adopted merely because it predicts shifted ground balls more accurately** --
criterion (4) checks every required subgroup, not just the shift-related ones.

**Real 2024 validation results** (n=122,132):

| variant | log loss | ECE | home-run ECE |
|---|---|---|---|
| `baseline_v02` | 0.670321 | 0.014413 | 0.002706 |
| `alignment_labels_v06` | 0.670287 | 0.014974 | 0.002918 |
| `alignment_interactions_v06` | 0.629065 | 0.013314 | 0.000912 |

`alignment_labels_v06`'s log-loss "improvement" (-0.0000344) does not clear bootstrap
significance (95% CI `[-0.000129, +0.000074]`, crosses zero). `alignment_interactions_v06`
shows a real, bootstrap-confirmed improvement (-0.0413, 95% CI `[-0.0431, -0.0395]`) and
better aggregate/home-run ECE -- but fails criterion (4): `bb_type_ground_ball` calibration
gets MATERIALLY WORSE (ECE 0.00776 -> 0.01367, a +76% relative regression) even as the
aggregate metrics improve, exactly the subgroup the shift-interaction feature was
designed to help. Every other required subgroup improves or holds steady.

**Controlled-perturbation results** -- mean predicted probability under each scenario
(pull-side rows only):

| candidate | infield: Standard vs Infield shift, P(single) (expect LOWER under shift) | outfield: Standard vs Strategic, P(double) (expect LOWER under Strategic) |
|---|---|---|
| `alignment_labels_v06` | correct: 0.229 -> 0.215 | **backwards**: 0.057 -> 0.061 |
| `alignment_interactions_v06` | correct, large effect: 0.231 -> 0.151 | **backwards**: 0.060 -> 0.070 |

The infield-shift check passes for both candidates with a real, sensible-sized effect.
The outfield-strategic check fails for both -- "Strategic" outfield alignment predicts
MORE doubles, not fewer. This is very plausibly **confounding by indication, not a code
bug**: alignment is not randomly assigned -- outfielders are more likely to shade/shift
against batters already known to be extra-base threats, so a model trained on
observational alignment labels can learn "this alignment co-occurs with more doubles"
(because of who gets shifted against) rather than "this alignment causes fewer doubles."
The `bb_type_ground_ball` regression above points at the same root cause on the infield
side of `alignment_interactions_v06`'s aggregate gain.

**`recommend_adopt_any_v06_candidate: False`.** Neither candidate clears every adoption
criterion. `alignment_labels_v06` fails on bootstrap significance and the outfield
perturbation check; `alignment_interactions_v06` fails on the ground-ball subgroup
regression and the same outfield perturbation check, despite a real aggregate log-loss
and ECE improvement. **No candidate is adopted; `baseline_v02` remains the production
model.** Per the task's explicit fallback, alignment-aware positioning is retained as
evaluated, tested infrastructure and marked **unresolved** rather than forced into
adoption.

The positioning counterfactual (`mlb_luck_score.scoring.positioning_attribution`,
`positioning_effect = EV(actual alignment) - EV(typical alignment)`, holding every other
feature fixed; positive = actual alignment favored the batter more than a typical
alignment would have) is implemented and exercised in notebook
`08_alignment_positioning_analysis.ipynb`, but -- since no candidate passed adoption --
its output is exploratory only, not a validated Contact Luck component, and it is
explicitly an external-circumstance contribution, never raw residual luck and never a
measure of defensive execution.

```bash
make join-venue-metadata        # if not already run for Version 0.3
make compare-alignment-aware    # 3-way comparison, see above
make notebook-alignment
```

**Next steps**: investigate whether the outfield-strategic confounding can be reduced
(e.g. a batter-power control feature, or restricting the perturbation check to
plays/batters where alignment changed within-season) before re-testing; re-examine
whether `of_shift_x_hit_distance` is the specific interaction term driving the ground
-ball subgroup regression despite targeting a different batted-ball type.

## Outfield opportunity and execution (Version 0.7A / 0.7B)

Estimates how difficult an outfield fielding opportunity was (Version 0.7A) and compares
the actual result against that difficulty (Version 0.7B) -- for outfield AIR BALLS only
(infield plays are explicitly out of scope; see "After outfield opportunity and execution
are validated, proceed to infield pickup/throwing models" in the task). Public tracking is
far stronger for airborne balls than infield plays, which is why this phase is split this
way.

### Public-data audit (required checkpoint, done BEFORE any model code)

Before writing any model code, the full 119-column raw Statcast schema already downloaded
for this project (2021-2024) and every fielding-related function in the installed
`pybaseball` package (`statcast_outfield_catch_prob`, `statcast_outfield_directional_oaa`,
`statcast_outfielder_jump`, `statcast_outs_above_average`, `statcast_fielding`) were
audited for six specific fields:

| Field | Availability |
|---|---|
| Defender starting location | **Not available** -- no column, no pybaseball function |
| Defender endpoint | **Not available**, same reason |
| Opportunity time (hang time) | **Not available as a measured field** -- estimated via physics instead (see below) |
| Distance needed | **Not available** (requires a real starting location) |
| Catch-probability inputs | **Not available per-play** -- every pybaseball catch-probability/OAA/jump function returns a SEASON-LEVEL AGGREGATE leaderboard (players binned into 1-5 "star" difficulty categories), never a per-play value -- and per the task's modeling rule these are outcome-derived anyway, so they could only ever be validation/comparison metrics, never model inputs, regardless of granularity |
| Responsible fielder | **Available** -- `hit_location` (91.75% coverage among real 2024 air balls; ~100% for outs/singles/triples/sac-flies, ~0.1% for home runs since nobody fields one) plus `fielder_7`/`fielder_8`/`fielder_9` (100% coverage, the specific player ID at each outfield position for that exact play) together give a real individual-defender identity |

**Conclusion**: exact defender coordinates, movement, and distance-needed cannot be
measured from public data. Per the task's explicit fallback, Version 0.7A implements only
the conservative, measured/estimated-only candidate, `measured_contact_only_v07`:
landing-location estimate, estimated hang time (physics), wall proximity (reused from
Version 0.4 park geometry), exit velocity/launch angle, batted-ball type, and coarse
outfield-alignment label. **No assumed defender starting coordinates.**

A second candidate, `typical_position_proxy_v07` (an assumed average starting depth/angle
per outfield position), was investigated but is **NOT implemented**: the only citable
public figures found (MLB.com/Statcast-sourced 2015-2016 league averages -- 316 ft average
CF depth, 294 ft average RF depth) are 5-9 years stale relative to this project's
2021-2024 training window, and multiple independent sources describe a sustained,
directional trend toward deeper outfield positioning since then (one cited example: the
Astros moved their center fielders back 16 feet specifically during 2021-2022), meaning
the stale figures would introduce a known, systematic bias, not just noise. Per CLAUDE.md's
park-geometry provenance rule ("use the best-supported figure with a caveat or leave it
without... rather than guessing"), no defensible source exists for this project's era.

**"Forward/lateral/backward defender movement" (a required evaluation subgroup) is also
NOT computed**, for the identical reason -- it requires a real starting position that does
not exist in public data. This is a documented gap, reported explicitly wherever subgroup
results are shown, not a silently-dropped requirement.

### Eligibility

`mlb_luck_score.eligibility.add_outfield_opportunity_eligibility`: `bb_type` in
(`fly_ball`, `line_drive`) AND `hit_location` is either an outfield code (7/8/9) or
missing (the home-run/deep-double case). `popup` is excluded from the primary eligible
population -- verified against real 2024 data, popups are assigned an INFIELD
`hit_location` 99.5% of the time (8,756 of 8,803), overwhelmingly an infield play type in
this dataset, not an outfield one. Real 2021-2024 result: 227,334 of 494,173
`eligible_for_training` rows are outfield-opportunity-eligible.

### Version 0.7A: opportunity-difficulty model

`mlb_luck_score.models.compare_opportunity_models` / `mlb_luck_score.models.
train_opportunity_model`: a SEPARATE binary logistic regression (`class_weight=None`,
same probability-calibration reasoning as the 5-class model) predicting `P(an average MLB
outfielder converts this into an out)`, trained on 2021-2023, evaluated on 2024 (2025
untouched). Estimated hang time uses the vacuum (no-drag) projectile formula `t = 2 * v0 *
sin(theta) / g`; estimated landing coordinates use `hit_distance_sc` + `spray_angle_approx`
in the same polar convention as park geometry/weather. **Both are documented
approximations with no public ground truth to validate against** -- Baseball Savant's own
Catch Probability methodology presumably uses real tracked hang time/landing location, but
that is not publicly exposed per-play.

**Real 2024 validation results** (n=57,598; 169,736 training rows from 2021-2023):

| metric | value |
|---|---|
| Binary log loss | 0.385044 (95% bootstrap CI [0.380225, 0.389777], 500 game-level reps) |
| Expected calibration error | 0.017771 (95% CI [0.015367, 0.020986]) |
| Brier score | 0.122296 |
| Actual out rate / mean predicted P(out) | 0.5462 / 0.5386 |

Overall calibration clears the 0.05 absolute-ECE quality bar (there is no prior production
model to compare against, so this is an absolute bar, not a relative one -- see
`compare_opportunity_models` module docstring). **Subgroup calibration reveals a real,
honest weakness**: near-wall plays have MUCH worse calibration than the aggregate
(near_wall_5ft ECE 0.254, near_wall_10ft 0.234, near_wall_20ft 0.180 -- roughly 10-15x the
overall figure), and the two middle opportunity-time quartiles are also worse (0.074 and
0.065 vs. 0.047/0.034 for the shortest/longest quartiles). The model is least reliable
exactly where judging defense is hardest: ambiguous warning-track plays and borderline
-difficulty opportunities. `bb_type_fly_ball`/`bb_type_line_drive`, LF/CF/RF, and all 30
reliably-sampled venues clear the quality bar. 164 individual defenders have >=100 sampled
plays each, enabling real per-defender calibration.

`recommend`-style automated summary: `passes_basic_validation: False` for
`measured_contact_only_v07` (driven entirely by the near-wall/mid-opportunity-time
subgroup issues above) -- reported honestly as a real, documented model limitation, not
hidden. `typical_position_proxy_v07_built: False` (see audit above).

```bash
make join-park-geometry          # if not already run for Version 0.4
make compare-opportunity-models  # trains + evaluates measured_contact_only_v07
make notebook-outfield-opportunity
```

### Version 0.7B: defensive execution

`mlb_luck_score.scoring.defensive_execution.compute_defensive_execution`:

```
defensive_execution = actual_out_indicator - p_out_opportunity
```

Positive = an out was made on a LOW-probability opportunity (strong positive defensive
execution -- unfavorable circumstance for the batter); negative = a HIGH-probability
opportunity was NOT converted (poor defensive execution -- favorable circumstance for the
batter); near zero = a routine play went as expected. `batter_favorable_defensive_
circumstance = -defensive_execution` is provided explicitly for the batter's-perspective
sign flip the task calls for. On real 2024 data (n=57,598): `defensive_execution` mean
0.0076, std 0.350, ranging the full [-1, 1] span (a play with `p_out_opportunity` near 0
that was converted scores near +1; a near-certain opportunity that was missed scores near
-1).

This is a BINARY, opportunity-relative execution measure -- it says nothing about reaction
time, route efficiency, or throwing accuracy specifically (none of which are in public
data). A spectacular diving catch and a workmanlike catch of the same difficulty register
identically.

### Four separate components, not combined

`mlb_luck_score.scoring.air_ball_components.build_air_ball_component_report` assembles,
per play: `expected_run_value_contact_model`/`residual_contact_luck_runs` (the EXISTING
Version 0.2 contact model and formula), `p_out_opportunity` (Version 0.7A),
`defensive_execution`/`batter_favorable_defensive_circumstance` (Version 0.7B). **Per the
task's explicit instruction, these are reported side by side and NOT summed into one
score** -- `residual_contact_luck_runs` comes from a contact model that does not currently
include opportunity/execution features, so its relationship to `defensive_execution` is
not yet a clean decomposition (same double-counting caution as `mlb_luck_score.scoring.
weather_attribution`).

### Limitations

- Alignment-aware POSITIONING only for opportunity difficulty -- no exact defender
  coordinates, pre-contact movement, reaction time, or route efficiency anywhere in this
  phase.
- `typical_position_proxy_v07` and the forward/lateral/backward movement subgroup are both
  documented gaps, not silent omissions -- see the audit above.
- Estimated hang time and landing coordinates have no public ground truth to validate
  against.
- Calibration is materially worse for near-wall and mid-opportunity-time plays -- treat
  Version 0.7B execution values for those specific plays with extra caution. **See "Near
  -wall opportunity correction (Version 0.7C)" below -- this limitation now has a
  specialist model addressing it, though not yet fully validated.**
- Infield pickup/throwing models and batter-runner advancement are explicitly deferred to
  future phases per the task.

## Near-wall opportunity correction (Version 0.7C)

Addresses Version 0.7A's real, honestly-reported near-wall calibration weakness (ECE
0.180-0.254 in the 5/10/20-ft wall bands vs. an overall 0.018) with a GATED architecture
(`mlb_luck_score.models.outfield_gating`) that leaves open-field scoring completely
untouched:

- **`open_field_v07`**: the EXISTING `measured_contact_only_v07` model, unchanged, for
  plays NOT within 20 ft of the wall.
- **`near_wall_v07c_candidate`**: a specialist trained ONLY on wall-adjacent plays
  (`mlb_luck_score.models.compare_near_wall_models`), comparing a logistic baseline against
  an unweighted `HistGradientBoostingClassifier` -- can find nonlinear interactions among
  hang time/launch angle/wall distance/wall height/spray direction automatically, with the
  IDENTICAL feature set as the logistic candidate (no hand-engineered interaction terms
  favoring either model class).
- **Contact-only fallback**: rows where park geometry itself is unreliable (missing,
  flagged `geometry_uncertain`, or missing wall distance) get NO opportunity/execution
  score at all -- only the existing Version 0.2 contact expectation/residual luck.

**Season split** (per the task's explicit instruction): fit on 2021-2022, select the
winning candidate on 2023, final comparison on 2024 -- reusing `mlb_luck_score.config.
CALIBRATION_BASE_TRAIN_SEASONS`/`CALIBRATION_FIT_SEASONS`/`CALIBRATION_EVAL_SEASONS`
(already built for post-hoc calibration) rather than inventing new constants. 2025 remains
untouched. `open_field_v07` is evaluated exactly as already trained in Version 0.7A (on the
standard `TRAIN_SEASONS`, 2021-2023) -- not retrained on the smaller near-wall-specific
split.

**Model selection (2023, n=8,194)**: `near_wall_logistic_v07c` log loss 0.406239, ECE
0.016695; `near_wall_hgb_v07c` log loss 0.369277, ECE 0.017665. **HGB selected** (lower log
loss).

**Final comparison (2024 near-wall rows, n=7,853)**:

| variant | log loss | ECE |
|---|---|---|
| `open_field_v07` (applied to these same wall-adjacent plays) | 0.670934 | 0.180439 |
| `near_wall_v07c_candidate` (HGB) | 0.394988 | 0.027564 |

A large, real, **bootstrap-confirmed** improvement (500 game-level reps): log-loss delta
-0.275946, 95% CI [-0.292007, -0.259978]; ECE delta -0.152875, 95% CI [-0.166817,
-0.135739].

**Required checks**:

- **Wall bands**: all three pass cleanly -- near_wall_5ft ECE 0.039060 (n=1,947),
  near_wall_10ft 0.030853 (n=3,983), near_wall_20ft 0.027564 (n=7,853).
- **Opportunity-time buckets**: all four pass -- 0.043402 / 0.031433 / 0.030886 / 0.024130
  (shortest to longest quartile).
- **Wall height** (short/medium/tall terciles): 0.052954 (n=366) / 0.074802 (n=330) /
  0.113173 (n=348) -- all nominally exceed the 0.05 bar, but coverage is sparse (only
  ~12.3% of geometry-available rows have a reviewed wall height -- a pre-existing Version
  0.4 data limitation, not new to this phase) and sample sizes are small; the degradation
  with taller/rarer wall configs is plausible but not conclusively distinguishable from
  sampling noise at this size.
- **Spray sector**: left 0.051353 (n=1,388) and left_center 0.057079 (n=1,612) nominally
  exceed the bar; center 0.024389, right_center 0.027043, right 0.028325 all pass cleanly
  -- a real, if modest, pull-side/oppo-side asymmetry worth further investigation.
- **Venue**: 28 of 30 reliably-sampled venues nominally exceed the SAME fixed 0.05 ECE
  threshold used for the much larger open-field population -- most per-venue near-wall
  samples here are only ~200-330 rows, and ECE estimated from a sample this size has
  substantial sampling variance; this is very plausibly a scale-mismatched-threshold
  artifact rather than genuine per-venue miscalibration, but is reported honestly as-is
  rather than adjusting the threshold specifically to make this pass.
- **Architectural no-open-field-regression check**: `predictions_identical_via_gate:
  True` -- confirmed bit-identical whether an open-field row is scored via the gate or
  directly; gating only routes rows, it never alters the model or its inputs. (A
  descriptive-only side note: `open_field_v07`'s ECE restricted to just its own
  gated subset is 0.063025, numerically worse than the blended whole-population figure of
  0.017771 from Version 0.7A -- independently verified NOT to be a bug: aggregate ECE
  across a mixed population is not a simple decomposition of subgroup ECEs, and mixing
  subpopulations can shift the aggregate figure in either direction for purely
  compositional reasons. This is reported for context, not treated as a pass/fail
  criterion.)
- **Controlled perturbations**: the wall-distance check passes cleanly and with a large,
  sensible effect (close wall mean P(out) 0.255731 vs. far wall 0.586909, delta +0.331178
  -- a farther wall makes the SAME batted ball comparatively more catchable, as physically
  expected).

  A real bug WAS caught and fixed during this work: an earlier version of both perturbation
  checks overrode a raw feature (`wall_distance_in_spray_direction`, `estimated_hang_time_
  s`) without recomputing its DEPENDENT derived features (`projected_distance_to_wall_
  margin`/`absolute_distance_to_wall`, which are computed FROM wall distance; hang time
  itself is computed FROM launch angle/speed) -- feeding the model an internally
  -inconsistent row and producing a spurious backwards wall-distance result. Fixed by
  recomputing dependents on every override (`_override_wall_distance_and_recompute`/
  `_override_launch_angle_and_recompute_hang_time`).

  **The original launch-angle requirement was invalid, not just backwards -- and the
  corrected check confirms the open-field intuition was right all along.** The original
  band-restricted check (`_override_launch_angle_and_recompute_hang_time`, band-restricted
  to `BAND_SHORT_OF_WALL`) varies `launch_angle` while leaving `launch_speed` -- and
  therefore the counterfactual ball's landing distance -- at whatever the real row
  happened to have. That is not a clean test of "does more opportunity time help the
  fielder"; it is a test of "launch angle, entangled with however hard this particular ball
  needed to be hit to travel this particular distance." On real 2024 data this proxy check
  gives low angle/unmatched distance P(out) 0.475014 vs. high angle/unmatched distance
  0.944219 (delta +0.469205, n=2,508) -- BACKWARDS from the open-field "more opportunity
  time -> more catchable" intuition, exactly as before. But this is LAUNCH ANGLE, not
  opportunity time: reaching the same distance via a low angle requires much higher exit
  velocity (a flatter, harder-hit ball) than via a high angle (a softer, higher-arcing one),
  so this proxy conflates "more opportunity time" with "how hard the ball was hit." This
  check is kept and reported (renamed `launch_angle_proxy_short_of_wall`, plus the same
  proxy construction for `launch_angle_proxy_effect_at_wall`/`_beyond_wall`), but is now
  DESCRIPTIVE ONLY -- it no longer gates `near_wall_specialist_calibrated`.

  The corrected replacement, `trajectory_matched_opportunity_time_short_of_wall`
  (`_override_launch_angle_matched_trajectory`, `mlb_luck_score.data.outfield_physics.
  solve_launch_speed_for_matched_range_mph`), holds landing distance -- and therefore every
  wall-geometry feature derived from it -- fixed at the row's REAL value, and instead SOLVES
  the exit velocity (same vacuum projectile model as `estimate_hang_time_seconds`) that
  reaches that same distance at the new angle. Because both trajectories are matched to the
  SAME distance, the higher-angle one is mathematically guaranteed (and verified at runtime,
  `trajectory_match_invariants`) to have strictly greater estimated opportunity time: mean
  hang time 2.69s (18 deg) vs. 4.33s (40 deg). With the exit-velocity confound removed, the
  result flips to the physically expected direction and PASSES: mean P(out) 0.560452 (low
  angle, matched) vs. 0.949601 (high angle, matched), delta +0.389149, n=2,508. This
  confirms the confounding hypothesis directly rather than leaving it as a suspicion --
  "more opportunity time" genuinely does increase catchability near the wall once launch
  angle's entanglement with exit velocity is removed; the original required check's
  direction was not a real physical finding, and demoting it (rather than simply flipping
  its sign) is the right fix, per CLAUDE.md's guidance to treat a backwards-signed
  perturbation as a likely confound first.

  A REAL-DATA (never synthetic) grouped response curve corroborates this independently:
  `compute_grouped_opportunity_time_response` bins real rows' own `estimated_hang_time_s`
  into quartiles within each wall band and `bb_type`, with no override at all. Every one of
  the 6 (band x bb_type) strata shows mean predicted P(out) increasing monotonically from
  the shortest to the longest hang-time quartile (e.g. `short_of_wall`/`fly_ball`: 0.700 ->
  0.847 -> 0.931 -> 0.950; `short_of_wall`/`line_drive`: 0.293 -> 0.392 -> 0.470 -> 0.609),
  while each stratum's mean `hit_distance_sc` stays reasonably flat across bins (e.g.
  361.7/364.5/360.5/354.4 ft for that same `short_of_wall`/`fly_ball` stratum -- if anything
  trending slightly DOWN at the longest-hang-time bin, the opposite of what a
  distance-confound would produce) -- landing distance is not silently driving the curve.

- **Launch angle is not the same variable as opportunity time.** A pure "vary opportunity
  time alone" perturbation is not physically constructible: `estimated_hang_time_s` is
  fully determined by `launch_speed`/`launch_angle`, so directly overriding it while
  leaving those two at their real values would recreate the same internal-inconsistency bug
  class documented above, one feature removed. The trajectory-matched check and the grouped
  real-data response curve are the two valid substitutes -- one varies hang time only
  through a jointly-solved, physically valid change to angle AND exit velocity; the other
  uses real rows' own natural variation with no override at all.

**`near_wall_specialist_calibrated: False`.** Both required perturbation checks now pass
cleanly (`perturbation_failures: []`) alongside the large, bootstrap-confirmed aggregate
improvement and clean wall-band results -- but the specialist still doesn't clear every
required check, because the wall-height/spray-sector/venue subgroup issues above are
unrelated to the perturbation-check fix and remain open. Per the task's reporting rule:

> Outfield execution score available for calibrated open-field opportunities; provisional
> or unavailable for wall-adjacent opportunities.

`mlb_luck_score.scoring.gated_outfield_report.build_gated_outfield_report` implements this
exactly: open-field rows get `available_open_field`; near-wall rows are STILL SCORED (never
silently blanked) but labeled `provisional_near_wall`, not `available_near_wall_calibrated`,
until the open findings above are resolved; contact-only-fallback rows (geometry
unreliable, 3,116 of 57,598 real 2024 validation rows) get `unavailable_geometry_fallback`
with `p_out_opportunity`/`defensive_execution` left `NaN`.

```bash
make compare-near-wall-models   # model selection + final comparison, see above
```

**Next steps**: re-examine whether the wall-height/spray-sector/venue subgroup ECE issues
above are genuine or a fixed-threshold-at-small-scale artifact -- see "Version 0.7D:
calibration-gate correction" immediately below, which replaces the fixed-threshold
subgroup rule with a sample-size-aware one and answers this question directly (with a
sharper, less comfortable answer than "probably just noise").

## Version 0.7D: calibration-gate correction

Version 0.7C's subgroup/venue gate flagged issues using a single fixed `MATERIAL_ECE_
ABSOLUTE_THRESHOLD` (0.05) point-estimate comparison -- see "Subgroup ECE against a fixed
absolute threshold does not transfer across population sizes" above. That rule could not
distinguish "this point estimate looks bad because the model really is worse here" from
"this point estimate looks bad because a ~250-row sample is noisy" -- it just reported
every crossing honestly, with a caveat, and left the question open. Version 0.7D
(`mlb_luck_score.models.compare_near_wall_calibration_gate`) answers it: every subgroup and
venue now gets a game_pk-clustered bootstrap confidence interval on its own adaptive-bin
ECE, plus a PAIRED bootstrap CI comparing the specialist against `open_field_v07` on the
identical rows, and is assigned one of three statuses instead of a pass/fail boolean:

- **`calibrated`**: adequate outcome support (documented minimums: >=100 plays, >=20
  distinct games, >=20 of EACH outcome class) AND the ECE confidence interval sits entirely
  at or below 0.05, with no credible paired regression vs. `open_field_v07`.
- **`not_calibrated`**: CREDIBLE evidence of a problem -- an ENTIRE confidence interval on
  the wrong side of a line (the ECE CI entirely above 0.05, or the paired
  specialist-minus-baseline log-loss delta CI entirely above zero), never a bare point
  estimate.
- **`insufficient_evidence`**: either raw support is below the documented minimum, or
  support is nominally adequate but the confidence interval straddles the threshold -- the
  data genuinely cannot tell yet. This is explicitly neither a pass nor a fail.

This module does NOT change the near-wall model, its features, hyperparameters, or
predictions, `open_field_v07`, or the contact-only fallback -- it reuses all of Version
0.7A/0.7C's trained models and required physical perturbation checks UNCHANGED, and
replaces ONLY the subgroup/venue calibration-gating rule.

**Real 2024 result** (`make compare-near-wall-calibration-gate`): of the 15 required
subgroups (wall-distance bands, wall-height bands, spray sectors, opportunity-time groups)
plus 30 venues (45 total):

| status | count | which |
|---|---|---|
| `calibrated` | 6 | `near_wall_10ft`, `near_wall_20ft`, `spray_sector_center`, `spray_sector_right_center`, `opportunity_time_q3`, `opportunity_time_q4_longest` |
| `not_calibrated` | 19 | `wall_height_medium`, `wall_height_tall`, and 17 individual venues (~250-330 rows each) |
| `insufficient_evidence` | 20 | `near_wall_5ft`, `wall_height_short`, `spray_sector_left`/`left_center`/`right`, `opportunity_time_q1_shortest`/`q2`, and 13 smaller venues |

This is a SHARPER, less comfortable finding than the Version 0.7C writeup's working
hypothesis: e.g. venue 3's specialist ECE is 0.129 (CI [0.096, 0.198], entirely above
0.05) -- credibly `not_calibrated` by the absolute bar -- even though the SAME rows'
paired log-loss delta CI is [-0.447, -0.247] (entirely negative), meaning the specialist is
a credible, LARGE improvement over `open_field_v07` (whose own ECE on those rows is
0.324) for that same venue. Both things are true at once: comparatively much better, and
still not meeting the absolute 0.05 bar. This disproves the earlier "probably just
small-sample noise" hypothesis for a meaningful fraction of these venues (17 of 30 hold up
under bootstrap resampling as genuinely `not_calibrated`, not merely nominal crossings) --
some of Version 0.7C's flagged venues were real, not artifacts. The aggregate-level checks
carried over unchanged from 0.7C all still pass: log-loss improves (-0.275946), the
aggregate bootstrap supports it (CI entirely negative), the architectural
no-open-field-regression check passes, and BOTH required physical perturbation checks
(`wall_distance_direction`, `trajectory_matched_opportunity_time_short_of_wall`) pass.

**`near_wall_specialist_calibrated: False`, `overall_status: not_calibrated`.** Because at
least one adequately-supported group (17 venues plus 2 wall-height bands) is CREDIBLY
`not_calibrated`, the overall status is `not_calibrated`, not the softer
`calibrated_with_limited_subgroup_evidence` tier -- per the task's reporting rule, near-wall
rows stay `provisional_near_wall`. (Had every group instead been either `calibrated` or
`insufficient_evidence` -- i.e. no CREDIBLE failures, just some groups too small to judge --
the overall status would have been `calibrated_with_limited_subgroup_evidence`, which ALSO
does not flip `near_wall_specialist_calibrated` to `True`: insufficient evidence is never
reported as a pass. See "Measured miscalibration vs. insufficient evidence" in
CLAUDE.md/AGENTS.md.)

Reliability tables (every subgroup/venue's full evidence -- n plays, n games, positive/
negative outcome counts, log loss, Brier score, adaptive-bin ECE, calibration
intercept/slope where estimable, and both CIs) are written to `outputs/tables/near_wall_
calibration_gate_detail.json`; reliability-diagram PNGs for every subgroup/venue are
written to `outputs/figures/near_wall_calibration_gate/` when `--figures-dir` is passed.

```bash
make compare-near-wall-calibration-gate
```

**Version 0.7 is now FROZEN.** This near-wall design line (0.7A-0.7D) has had its checks,
required-check set, and calibration-gating rule redesigned multiple times specifically in
response to what 2024 results looked like each time. 2024 must now be treated as
DEVELOPMENT validation for near-wall work, not an untouched test set, even though it
remains `VALIDATION_SEASONS`/`CALIBRATION_EVAL_SEASONS` formally -- see CLAUDE.md/AGENTS.md
"Version 0.7 is now FROZEN" for the full rule. Any further near-wall model, feature,
perturbation-check, or calibration-gate change needs either a season range not yet used
for near-wall design decisions, or explicit maintainer sign-off treating 2024 as
non-pristine for that specific change.

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
as the default. Park geometry (Version 0.4) has been researched, built, joined, and
evaluated for real 2021-2024 data -- see "Park geometry (Version 0.4)" above for the exact
results, including a large, bootstrap-confirmed near-wall calibration improvement and the
material venue regressions that keep neither `geometry_only_v04_candidate` nor
`venue_plus_geometry_v04_candidate` adopted as the default. Weather and air density (Version
0.5) have been researched, downloaded (real 2021-2024 MLB schedule weather + Iowa Environmental
Mesonet station data), joined, and evaluated for real data -- see "Weather and air density
(Version 0.5)" above for the exact results. Both pure-weather candidates pass all 8 automatable
adoption criteria with a real but tiny log-loss improvement, but a direct physical-plausibility
check (required by the adoption rule, never automated) found the per-play weather attribution to
be weak and partly wrong-signed after fixing two real bugs caught by that very check -- neither
candidate has been adopted as the default. Version 0.5.1 corrected this by testing three
INDEPENDENT feature sets that each avoid mixing air density with the raw variables used to
derive it, plus automated controlled-perturbation directional checks -- see "Weather correction
(Version 0.5.1)" above for the exact real results. The venue-anomaly formulation fixes the
backwards density signal (a genuine, non-buggy finding), but no candidate clears BOTH the
bootstrap-significance bar AND the perturbation checks together; `recommend_adopt_any_v051_
candidate: False`. `baseline_v02` remains the production model; weather is retained as
evaluated, tested infrastructure and marked unresolved rather than forced into adoption.
Alignment-aware positioning (Version 0.6) has been evaluated for real 2021-2024 data --
see "Alignment-aware positioning (Version 0.6)" above for the exact results.
`alignment_interactions_v06` shows a real, bootstrap-confirmed aggregate log-loss and ECE
improvement, but fails adoption on a material ground-ball subgroup regression and a
backwards-signed outfield-alignment perturbation check (very plausibly confounding by
indication -- alignment is not randomly assigned); `alignment_labels_v06` does not even
clear bootstrap significance. `position_depth_v06` was investigated and found to have no
reliable public data source. `recommend_adopt_any_v06_candidate: False`; `baseline_v02`
remains the production model; alignment-aware positioning is retained as evaluated,
tested infrastructure and marked unresolved rather than forced into adoption.
Outfield opportunity and execution (Version 0.7A/0.7B) have been built and evaluated for
real 2021-2024 data -- see "Outfield opportunity and execution (Version 0.7A / 0.7B)"
above for the exact results and the public-data audit that shaped scope (defender
starting/ending location, distance needed, and per-play catch-probability inputs are all
NOT available publicly; only `measured_contact_only_v07` was built, with no assumed
defender coordinates). `measured_contact_only_v07` clears the overall calibration quality
bar (log loss 0.385, ECE 0.018) but fails it on near-wall plays (ECE 0.18-0.25) and
mid-range opportunity-time buckets -- reported honestly as a real, documented limitation.
`typical_position_proxy_v07` was investigated and not built (only stale 2015-2016 sources
found); `defensive_execution` (Version 0.7B) and the four-component per-play report are
implemented and exercised against real data, explicitly NOT combined into one score.
Version 0.7C's near-wall specialist has been built and evaluated for real 2021-2024 data
-- see "Near-wall opportunity correction (Version 0.7C)" above for the exact results. A
gated architecture (`open_field_v07`/`near_wall_v07c_candidate`/contact-only fallback)
routes wall-adjacent plays to an `HistGradientBoostingClassifier` specialist selected over
a logistic baseline (fit 2021-2022, selected 2023, final-compared 2024); it shows a large,
real, bootstrap-confirmed improvement over applying `open_field_v07` to those same plays
(log loss 0.671 -> 0.395, ECE 0.180 -> 0.028) and passes all wall-band checks, but
`near_wall_specialist_calibrated: False` -- an unresolved, honestly-reported backwards
-signed launch-angle perturbation-check finding (plausibly a genuine, different physical
relationship at near-fixed near-wall distance, not yet confirmed) plus several
subgroup/venue ECE issues (plausibly small-sample noise) keep it short of full validation.
Per the task's reporting rule, `mlb_luck_score.scoring.gated_outfield_report` labels
near-wall execution scores `provisional_near_wall` (still computed, never blanked) rather
than `available_near_wall_calibrated`.

## Future work

- Re-test `density_anomaly_v051_candidate` without wind (or with an independently verified wind
  feature) now that its density/Coors direction is confirmed correct -- see "Weather correction
  (Version 0.5.1)"; also try an unregularized/tuned model to rule out coefficient shrinkage as
  the cause of wind's null signal in all three Version 0.5.1 candidates
- Human review of the Version 0.4 park-geometry AND Version 0.5 venue-environment reference
  tables (currently `agent_sourced_pending_human_review` for every record)
- Published park factors (altitude, prevailing wind) beyond static wall geometry and per-game
  weather already implemented in Versions 0.4-0.5
- Finer-grained wall geometry (more than 5 points per park-era; true wall curvature;
  localized irregularities like Citizens Bank Park's "Monty's Angle")
- Investigate and reduce the confounding-by-indication problem in Version 0.6's
  outfield-alignment perturbation check (e.g. a batter-power control feature) before
  re-testing `alignment_interactions_v06`; identify which specific interaction term drives
  its `bb_type_ground_ball` subgroup regression -- see "Alignment-aware positioning
  (Version 0.6)"
- A reliable public per-fielder positioning/depth data source for `position_depth_v06`
  (none was found as of Version 0.6) or `typical_position_proxy_v07` (none was found as of
  Version 0.7A) -- exact defender positioning, reaction, and route modeling remain out of
  scope for any publicly-sourced dataset
- Investigate Version 0.7C's unresolved launch-angle/hang-time perturbation-check finding
  directly (does exit velocity really trade off against launch angle at near-fixed
  near-wall distance in this data); re-examine the wall-height/spray-sector/venue subgroup
  ECE issues with a larger sample or a size-aware threshold before revisiting `near_wall_
  specialist_calibrated` -- see "Near-wall opportunity correction (Version 0.7C)"
- Per the task's stated roadmap: infield pickup/throwing execution modeling next, then
  batter-runner advancement, once outfield opportunity/execution (Versions 0.7A/0.7B/0.7C)
  is considered validated
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
