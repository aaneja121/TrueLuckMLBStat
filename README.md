# Contact Luck

Contact Luck is an MLB batted-ball metric that measures the difference, in run value,
between what a batter's contact was expected to produce and what actually occurred.

The system was developed on 2021-2024 data, frozen before a one-time held-out evaluation
on the 2025 MLB season, and is now scored prospectively on 2026 data. The public metric is
**Contact Luck Runs per 100 Eligible Batted Balls**, accompanied by game-clustered 95%
uncertainty intervals.

The project decomposes realized outcomes across contact, outfield defense, infield
defense, and batter-runner advancement, with component-specific validation/status labels.
Contact Luck is retrospective and is not intended as a stable measure of batting talent or
a predictive statistic.

**Live dashboard:** https://contactluck.com

The production system includes immutable prospective snapshots, frozen-input
verification, durable Cloudflare R2 archival, automated GitHub Actions scoring, and
static Cloudflare Pages deployment.

### Validation design

- **2021-2023**: model fitting
- **2024**: development validation / model selection
- **2025**: one-time sealed final evaluation
- **2026**: prospective scoring

The 2025 evaluation classified the frozen system as validated with documented
limitations. Individual component estimates retain their own calibration/evidence labels
and should not be interpreted as independent causal effects.

## Development history

The remainder of this README documents the full research progression from Version 0.1
onward, including rejected candidate models and the reasoning behind adoption decisions.

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

## Infield opportunity and execution (Version 0.8)

Extends the opportunity/execution pattern from Version 0.7A/0.7B to fair GROUND balls
fielded by an infielder, restricted to plays where the batter-runner is the unambiguous,
sole out opportunity. Built on a NEW eligibility question (`mlb_luck_score.eligibility.
add_infield_opportunity_eligibility`) deliberately independent of Version 0.1's
`eligible_for_training`/`map_outcome_class` -- see that function's module comment for why
`field_error` is a clean "not retired" case here (no base classification needed) while
`force_out`/`fielders_choice`-type events are excluded entirely (the batter himself is
typically not the one retired). Version 0.7 remains FROZEN; nothing here touches it.

### Public-data audit (required checkpoint, done BEFORE any model code)

The real, already-downloaded 2021-2024 dataset and the installed `pybaseball` package were
audited for every field the task named:

| Field | Availability |
|---|---|
| Responsible fielder identity | **Available** -- `fielder_2`..`fielder_6` plus `pitcher` for position 1 (no separate `fielder_1` column). Used only for post-hoc evaluation, never a model input. |
| Responsible fielding position | **Available** via `hit_location` (1=P..6=SS), 99.98% coverage on ground balls |
| Batter sprint speed | **NOT a per-pitch field** -- available only as Baseball Savant's separate SEASON-LEVEL "Sprint Speed" leaderboard (`pybaseball.statcast_sprint_speed`). Downloaded per season (`mlb_luck_score.data.download_sprint_speed`) and joined by `(batter, season)` (`mlb_luck_score.data.join_sprint_speed`). |
| Infield alignment | **Available** via `if_fielding_alignment` (Standard/Infield shift/Infield shade/Strategic), the same column Version 0.6 uses |
| Batted-ball coordinates | **Available** (`hc_x`/`hc_y`, `spray_angle_approx`/`spray_sector`) |
| Fielded distance | **Available and REAL** -- `hit_distance_sc` is Statcast's own measured distance to where the ball was fielded (99.4% coverage), unlike the outfield model's physics-*estimated* landing point |
| Errors / force outs / fielder's choices / double plays / bunts | Controlled `events` taxonomy, except bunts (no dedicated flag; detected via a `des` keyword) |
| Exact defender starting position, route distance, exchange/throw time | **Not available anywhere in public data -- never assumed** |

Real 2021-2024 ground balls (`bb_type == "ground_ball"`, 214,032 rows) break down as:

| Category | Rows |
|---|---|
| `eligible` | 143,377 |
| `outfield_credited_hit_location` (ball got through the infield before any fielder touched it) | 34,647 |
| `excluded_strategic_play` (force out / fielder's choice / double play / sac bunt or fly) | 30,322 |
| `bunt_excluded` (`des` keyword match) | 4,393 |
| `missing_required_contact_data` | 1,151 |
| `interference_or_obstruction` | 113 |
| `missing_responsible_infield_position` | 29 |

`reviewed_or_overturned` (informational only, never excluded) matches 1,199 of 143,377
eligible plays (0.84%) -- Statcast's recorded `events`/`des` already reflect the final,
corrected ruling, so these are kept.

### Candidate B (timing-margin proxy) was investigated and NOT built

Unlike the outfield model's hang-time estimate (a real, citable vacuum-projectile-motion
formula), there is no comparable citable physics for ground-ball roll deceleration on
grass/turf -- building one would require an unvalidated friction/restitution constant,
exactly the kind of invented assumption the task instructs against. `hit_distance_sc` and
`sprint_speed`/`hp_to_1b` are both real, but there is no defensible way to combine them into
a genuine "time margin" feature. Per the task's explicit permission, this candidate is
skipped -- the only feature set built is `infield_contact_only_v08` (`launch_speed`,
`launch_angle`, `spray_angle_approx`, `hit_distance_sc`, `sprint_speed`, `outs_when_up`,
`on_1b_occupied` numeric; `stand`, `if_fielding_alignment`, `assigned_infield_position`,
`surface_type` categorical).

### Model selection and real 2024 result

Fit on 2021-2022, model CLASS selected on 2023 by log loss (`LogisticRegression` vs.
unweighted `HistGradientBoostingClassifier`, both `class_weight=None`), final comparison on
2024 (`make compare-infield-opportunity`):

| Candidate | Selection (2023) log loss | Selection ECE |
|---|---|---|
| `infield_logistic_v08` | 0.368668 | 0.016779 |
| `infield_hgb_v08` (winner) | 0.331572 | 0.012220 |

**Final 2024 comparison** (`infield_contact_only_v08`, n=35,526): log loss 0.335659, Brier
0.098767, ECE 0.009108, calibration intercept -0.179 / slope 1.103, outcome prevalence
86.40% (an average infield ground ball is retired ~86% of the time). Feature missingness is
low across the board; the largest is `sprint_speed` at 0.64% (real coverage matches the
Phase 1 audit -- unmatched batters are disproportionately below Savant's own `min_opp=10`
qualification threshold, not a pipeline defect).

Like Version 0.7A, this is a genuinely NEW opportunity-difficulty category with no prior
public per-play infield baseline -- per CLAUDE.md/AGENTS.md's "a model with no prior
baseline needs an ABSOLUTE quality bar, not a relative one," it is judged against the same
`MATERIAL_ECE_ABSOLUTE_THRESHOLD` (0.05) reused from Version 0.7A/0.7C/0.7D, using the
v0.7D sample-size-aware three-status calibration gate FROM THE START, never a bare
point-estimate.

### Perturbation checks

- **`sprint_speed_direction` (required)**: among otherwise-identical batted balls, a slow
  batter (23.0 ft/s) shows mean predicted `P(out)` 0.9279 vs. a fast batter (29.0 ft/s) at
  0.8379 (delta -0.0899, n=35,298) -- correctly signed (faster batters are less likely to be
  retired) and **passes**. `sprint_speed` has no derived-dependent feature in this set, so no
  recomputation-on-override concern applies (see "Any override of a raw feature must
  recompute every feature DERIVED from it" in CLAUDE.md/AGENTS.md).
- **Timing-margin direction check**: `not_applicable` -- Candidate B was never built (see
  above), reported explicitly rather than silently omitted.
- **Exit velocity partial dependence**: descriptive only, no required sign (a harder-hit
  ground ball can both reduce a fielder's reaction time and reach a well-positioned defender
  before a bad hop) -- 1 sign change across the grid, not flagged as erratic.
- A real-data (never synthetic) grouped response curve for `sprint_speed` by infield
  position corroborates the required check using rows' own natural variation.

### Calibration gate: `calibrated_with_limited_subgroup_evidence`

Of 28 required subgroups (fielding position, pull/center/opposite-field, batter handedness,
sprint-speed quartile, standard-vs-shifted alignment, ground-ball hardness tercile, spray
sector) plus 34 venues (62 total):

| status | count | notes |
|---|---|---|
| `calibrated` | 47 | 26 of 28 subgroups, 21 of 34 venues |
| `not_calibrated` | 0 | none |
| `insufficient_evidence` | 15 | `position_2` (catcher fielding a grounder, n=289), `alignment_infield_shift` (n=0 -- no eligible rows tagged that exact category in the 2024 eval slice), and 13 smaller venues |

Overall ECE (0.009108) clears the absolute bar, the required perturbation check passes, and
no adequately-supported group is credibly `not_calibrated` -- but because some groups lack
adequate evidence rather than being confirmed either way, `overall_status` is
`calibrated_with_limited_subgroup_evidence`, and per CLAUDE.md/AGENTS.md's "Measured
miscalibration vs. insufficient evidence" rule, **`calibrated_infield_opportunity` is
`False`** -- insufficient evidence is never reported as a pass.

**Classification: adopted for real scoring on eligible infield plays, not described as
fully validated.** Selected model `infield_hgb_v08`; `overall_status =
calibrated_with_limited_subgroup_evidence`; `calibrated_infield_opportunity = False`. This
is stronger than leaving the model unused (log loss/ECE are excellent, the sprint-speed
counterfactual is correctly signed, no supported subgroup is credibly miscalibrated, and
2025 remains untouched) but more honest than calling it universally calibrated (15 groups
are genuinely uncertain, not failed). `mlb_luck_score.scoring.infield_ball_components`
implements this as a boolean gate -- the SAME pattern `mlb_luck_score.scoring.
gated_outfield_report` uses for the Version 0.7C near-wall specialist: every eligible row's
`p_out_opportunity`/`defensive_execution_probability` IS computed and reported (never
silently blanked), tagged `infield_opportunity_status = "provisional_infield_opportunity"`
(not the finer-grained `insufficient_evidence` -- that label describes the 15 flagged
subgroups/venues at the model-evaluation level, not individual rows; collapsing it to a
per-row label would understate that most subgroups, including most venues, genuinely ARE
calibrated). Only once `overall_status` reaches full `calibrated` does a row get
`calibrated_infield_opportunity` instead.

Reached-on-error plays (n=976, mean predicted `P(out)` 0.8608) are DESCRIPTIVE ONLY (task
Phase 3 forbids using the error flag as a predictor) but plausible: they score noticeably
higher than genuine non-error safe outcomes (n=3,854, mean predicted `P(out)` 0.7064) --
consistent with errors disproportionately happening on plays that "should" have been
routine outs, from pre-contact features alone (no post-outcome leakage).

Full reliability tables are written to `outputs/tables/infield_opportunity_detail.json`;
reliability-diagram PNGs for every subgroup/venue are written to
`outputs/figures/infield_opportunity/` when `--figures-dir` is passed.

```bash
make download-sprint-speed     # requires internet access
make join-sprint-speed
make compare-infield-opportunity
```

### Combined infield execution and component report

Same formula/sign-convention pattern as Version 0.7B, parallel but NOT shared with it (see
`mlb_luck_score.scoring.infield_execution` module docstring for why):

```
defensive_execution_probability = actual_out - p_out_opportunity
batter_perspective_infield_execution = -defensive_execution_probability
```

`mlb_luck_score.scoring.infield_ball_components.build_infield_ball_component_report`
assembles, per ground-ball play: the existing Version 0.2 contact-model expectation/residual
(null for reached-on-error rows, since Version 0.1 leaves `outcome_class` undetermined for
`field_error`), `p_out_opportunity`, `defensive_execution_probability`/`batter_perspective_
infield_execution`, `reached_on_error` (reporting only), and `infield_opportunity_status`.
**Reported side by side, never summed into one score** -- same reasoning as
`air_ball_components`.

### Limitations

- This is a BINARY, opportunity-relative execution measure -- it says nothing about pickup,
  transfer, footwork, or throwing specifically; a clean barehand play and a workmanlike play
  of the same difficulty register identically.
- Does not isolate positioning, scorer judgment, or unobserved route quality from execution.
- `calibrated_infield_opportunity` is `False` pending more data for `position_2`,
  `alignment_infield_shift`, and 13 individual venues -- treat those specific
  subgroups/venues with extra caution; the other 47 groups are genuinely `calibrated`.
- Individual-defender identity (`responsible_infielder_id`) is available for post-hoc
  evaluation only, never a training feature -- training on it would encode that specific
  fielder's skill rather than physical opportunity difficulty.

## Batter-runner advancement (Version 0.9)

Models what happens AFTER the batter safely reaches at least first on a fair outfield air
ball: `P(batter_final_base = held_at_first / advanced_to_second / advanced_to_third /
inside_the_park_home_run / retired_while_advancing)`. Batter-runner advancement only --
preexisting-runner advancement is explicitly out of scope. Infield hits are deferred (per
the task, "overthrows and hurried throws complicate attribution"). Version 0.7/0.8 remain
frozen/unaffected.

### The target is not a Statcast column -- it is parsed from `des`

`events` only records the batter's HIT TYPE (single/double/triple/home_run), not where
they ended up: real 2021-2024 examples include a "single" where the batter is later
thrown out stretching to second, and a "triple" where the batter scores anyway on a
subsequent throwing error while `events` stays `triple`. The target is reconstructed by
`mlb_luck_score.eligibility.parse_batter_advancement_des`: extract the batter's own name
from the leading clause of `des` (validated at 99.98% reliability across 159,743 real hit
rows), strip any MLB Gameday review/challenge preamble, then search the remainder for four
specific clause patterns (retired-while-advancing, error-driven advance, simple advance,
scores). Every batter-name mention must be accounted for by a known pattern or the row is
excluded, never guessed.

**Two real bugs were caught and fixed during development, not left for later discovery:**
a verb-lookup dict keyed by regex pattern strings instead of matched text (silently failed
on every "reaches on error" row until fixed to loop through patterns directly), and an
unstripped review/challenge preamble that was corrupting name extraction on ~0.8% of real
rows (`retired_while_advancing` coverage moved from 734 to 887 real rows once fixed).

Real 2021-2024 coverage, restricted to fair outfield air balls (`fly_ball`/`line_drive`)
with a safe event (single/double/triple/home_run/field_error) -- 107,721 rows:

| Parse status | Count |
|---|---|
| `parsed_unambiguous` | 107,718 (99.997%) |
| `unsupported_play_sequence` (excluded, never guessed) | 3 |
| `ambiguous` / `no_batter_match` / `parse_failure` | 0 |

| Category | Rows |
|---|---|
| `eligible` | 85,282 |
| `not_advancement_safe_event` (batter retired on the batted ball itself) | 137,952 |
| `trivial_no_advancement_opportunity` (plain over-the-fence home run -- advancement to home is certain and defense-independent, no genuine opportunity to model) | 22,436 |
| `unresolved_des_parse` | 3 |

Eligible population by label: `held_at_first` 53,417; `advanced_to_second` 27,982;
`advanced_to_third` 2,910; `retired_while_advancing` 887; `inside_the_park_home_run` 86
(this bucket deliberately conflates a genuine inside-the-park home run with the rarer
case of scoring via error from a lesser hit -- both are the SAME terminal state for the
batter-runner, kept distinguishable via a separate `is_true_inside_the_park_home_run`
informational flag).

**Manual review** (task safeguard #5): 125 freshly stratified rows (25 per label, with
the two rarest labels sampled at their full real proportion) plus all 3 `unsupported_
play_sequence` failures -- all correct on inspection. `des`, the matched clause, and every
parser audit-trail column are registered in `mlb_luck_score.config.LEAKAGE_COLUMNS` and
enforced by the same `assert_no_leakage` check as every other leakage rule in this
codebase (task safeguard #7).

### A real feature-omission bug, caught by comparing against the empirical baseline

The task specifies a simple empirical baseline (`advancement_empirical_baseline`: outcome
frequency by hit-type-implied floor base x sprint-speed quartile, fit ONLY on
`CALIBRATION_BASE_TRAIN_SEASONS`) specifically so the learned candidates have something
concrete to be checked against, not just their own internal calibration. That check caught
a real bug: an earlier feature set never included the batter's own hit type (single/
double/triple) as an explicit feature -- only continuous physics features and the contact
model's own probability ESTIMATE of hit type (`contact_p_triple`, which barely
distinguishes double-floor rows from triple-floor rows: mean 0.018 vs. 0.021 on real
2021-2022 data -- the contact model's pre-fielding guess is a much weaker signal than the
umpire-scorer's actual, already-final hit-type ruling). Without it, ALL THREE learned
candidates scored WORSE (multiclass log loss 0.44-0.53) than the two-column empirical
baseline (0.18) on real 2024 data, driven largely by the tree-based candidate assigning
near-zero probability (median 8.7e-6) to real `inside_the_park_home_run` rows. Adding
`hit_type_group` as an explicit categorical feature (legitimate and NOT leakage -- it only
establishes the FLOOR of possible outcomes, determined by the batted ball itself, not by
whether the batter later advances further or is retired) fixed this: selection-round log
loss dropped from 0.53/0.53/0.44 to 0.155/0.154/0.272. `beats_empirical_baseline` is now a
permanent, real-data-motivated criterion in `summarize_advancement_calibration`, not just a
one-time diagnostic -- a model that cannot beat a two-column frequency table has no
business being called calibrated regardless of its own internal ECE.

### Model selection and real 2024 result

Three candidates, IDENTICAL feature set except `sprint_speed` and model class, fit on
2021-2022, selected on 2023 by multiclass log loss (tie-broken by mean one-vs-rest ECE):

| Candidate | Selection (2023) log loss | Selection mean one-vs-rest ECE |
|---|---|---|
| `advancement_context_v09` (logistic, no speed) | 0.154857 | 0.002679 |
| `advancement_speed_v09` (logistic, +speed) -- winner | 0.154231 | 0.002320 |
| `advancement_nonlinear_v09_candidate` (HGB, +speed) | 0.272223 | 0.004680 |

(The nonlinear candidate is markedly WORSE here, not better -- HistGradientBoosting's
tree splits are more prone to the near-zero rare-class collapse described above than
logistic regression's smoother probability estimates, even with `hit_type_group` included.)

**Final 2024 comparison** (`advancement_speed_v09`, n=21,200): multiclass log loss
0.155301, **beating the empirical baseline's 0.181123**. Every one-vs-rest class's ECE
clears the 0.05 absolute bar (max 0.0045, `retired_while_advancing`). Sprint speed adds a
real but modest improvement over context alone (speed adding sprint_speed as a feature
narrowly beats context on both selection and reflects the real, if modest, marginal value
of runner speed once hit type is already known).

### Perturbation check

`sprint_speed_direction` (required): among otherwise-identical rows, a slow batter (23.0
ft/s) shows mean "expected advancement index" 1.3735 vs. a fast batter (29.0 ft/s) at
1.3886 (delta +0.0151, n=21,112) -- correctly signed (faster batters advance further) and
**passes**. `sprint_speed` has no derived-dependent feature anywhere in this set
(`contact_p_*` comes from an earlier, independent model stage; hang time/landing
coordinates are derived from launch speed/angle/distance, not speed), so overriding it
needs no recomputation.

### Calibration gate: `calibrated_with_limited_subgroup_evidence`

Required subgroups (sprint-speed quartile, contact type, spray sector, outfielder
position, wall proximity) plus venues, evaluated per one-vs-rest CLASS (the "final-base
outcome" dimension) -- 255 (subgroup/venue, class) pairs total:

| status | count | notes |
|---|---|---|
| `calibrated` | 137 | broad coverage across the three common labels (`held_at_first`, `advanced_to_second`, `advanced_to_third`) and most subgroups |
| `not_calibrated` | 0 | none -- no credible miscalibration anywhere |
| `insufficient_evidence` | 118 | overwhelmingly `inside_the_park_home_run` (n=86 total) across nearly every subgroup/venue, `retired_while_advancing` (n=887) across many venues, and a few `contact_type_triple` pairs (triples are inherently rare, ~2,900 total) |

Zero credibly `not_calibrated` pairs, but per CLAUDE.md/AGENTS.md's "Measured
miscalibration vs. insufficient evidence" rule, `overall_status` is `calibrated_with_
limited_subgroup_evidence` and **`calibrated_advancement_model` is `False`** --
insufficient evidence for the two rarest outcome classes is never reported as a pass, even
though every criterion that CAN be checked passes cleanly (beats the empirical baseline,
every class ECE within threshold, the required perturbation check passes, zero credible
regressions).

**Classification: adopted for real scoring on eligible outfield-air-ball plays, not
described as fully validated.** Same pattern as Version 0.8's `provisional_infield_
opportunity` -- `mlb_luck_score.scoring.advancement_execution` implements this as the SAME
boolean gate (`overall_status == calibrated` -> `calibrated_advancement_model`; anything
else -> `provisional_advancement_model`), scores are always computed and reported for
eligible rows, never silently blanked.

### Six components, reported side by side, never summed

`mlb_luck_score.scoring.advancement_execution.build_advancement_component_report`
assembles, per eligible play: `expected_contact_base` (the existing contact model's own
pre-fielding prediction of contact value), `actual_final_batter_base` (the real observed
outcome), `advancement_opportunity` (this model's predicted expected base value),
`batter_runner_advancement_execution` (`actual - opportunity`), `defensive_advancement_
effect` (`= -batter_runner_advancement_execution`, the same batter/defense sign-flip
convention `defensive_execution`/`infield_execution` already use -- NOT an independently
-estimated causal split of batter speed vs. defensive positioning, which a single joint
model cannot support), and `residual_uncertainty` (the variance of the predicted
base-value distribution -- how much genuine outcome variance remains even after
conditioning on everything known).

### Limitations

- `calibrated_advancement_model` is `False` pending more data specifically for `inside_
  the_park_home_run` (n=86) and, to a lesser extent, `retired_while_advancing` (n=887) and
  triple-floor rows -- the other 137 (subgroup, class) pairs are genuinely `calibrated`,
  and zero pairs are credibly `not_calibrated`.
- Infield hits, preexisting-runner advancement, and discretionary scorer decisions are
  explicitly out of scope for this phase.
- `defensive_advancement_effect` is a sign-flipped restatement of the SAME execution
  quantity, not an independent estimate of the defense's own contribution.
- Plain over-the-fence home runs are excluded from the eligible population by design (no
  genuine advancement opportunity exists once a ball clears the fence).

## Play-level attribution ledger (Version 0.10)

`mlb_luck_score.scoring.attribution_ledger` reconciles Version 0.2's raw contact luck,
Version 0.7B/0.8's defensive execution, and Version 0.9's advancement execution into ONE
run-value accounting identity per play, via a telescoping chain of successive conditional
expectations (`E0` -> `Eo` -> `Ea` -> `Rf`) so every unit of "surprise" is attributed to
exactly one stage, never re-attributed to a later one. This is an audit/reconciliation
tool -- it does NOT publish a combined score. Weather (Version 0.5/0.5.1) and alignment
(Version 0.6) are excluded entirely (neither passed adoption).

`mlb_luck_score.scoring.run_attribution_ledger` runs this end to end on real,
already-downloaded 2021-2024 data: `baseline_v02` trained on 2021-2023, `measured_
contact_only_v07` (uniform, no near-wall gating -- Version 0.7 is frozen) trained on
2021-2023, the Version 0.8 infield model and Version 0.9 advancement model reusing their
own real winner-selection procedures, scored on the held-out 2024 season. Real result:
the accounting identity holds EXACTLY for all 122,493 resolved rows of 123,980 scored
2024 plays -- maximum absolute reconciliation error `2.22e-16` (float64 machine epsilon).
Domain routing is confirmed mutually exclusive (zero rows eligible for both outfield and
infield opportunity domains). No materially large residuals (0 rows at or above the
2.0-run provisional threshold). See `outputs/tables/attribution_ledger_v010_real_data_
report.json` for the full breakdown (pathway counts, field-error handling, residual
distribution).

`mlb_luck_score.features.build_contact_features.add_opportunity_features_by_domain` is
the hardened, PREFERRED way to compute outfield+infield opportunity features on one
combined DataFrame: it routes each row to at most one domain-specific builder (they now
write to DIFFERENT target columns, `outfield_converted_to_out`/`infield_converted_to_out`,
eliminating a real collision bug the two builders used to share), fails fast on a row
eligible for both domains simultaneously, and reassembles in the original row order. Run
`make run-attribution-ledger` to regenerate the real-data report.

## Confidence, uncertainty, and season aggregation (Version 0.11)

Turns the Version 0.10 play-level ledger into a statistically honest season-level
reporting layer. Freezes Versions 0.2-0.10 completely -- nothing in this phase refits,
recalibrates, or tunes any component model; every module here is a read-only
interpretation/aggregation layer over the frozen ledger's already-computed predictions.
Deliberately does NOT produce the final public-facing composite score -- that is an
explicit next phase.

### Confidence taxonomy (Phase 1-2)

`mlb_luck_score.scoring.component_confidence` defines FOUR separate, independent
taxonomies rather than collapsing everything into one confidence number -- reporting
"high confidence" for a row in an uncertain subgroup merely because a model's aggregate
metrics look good is exactly what this design prevents:

- **A. Model-status confidence**: `calibrated` / `calibrated_with_limited_subgroup_
  evidence` / `provisional` / `not_calibrated` / `unavailable`.
- **B. Row-level data quality**: `complete_measured_inputs` / `estimated_inputs` /
  `fallback_inputs` / `missing_inputs` / `excluded_play`.
- **C. Statistical support**: `strong` / `moderate` / `limited` / `insufficient`.
- **D. Domain status**: `contact` / `open_field_outfield` / `provisional_near_wall` /
  `provisional_infield` / `provisional_advancement` / `unavailable_or_excluded`.

`confidence_tier` (`high`/`medium`/`low`/`unavailable`) is a fifth, DERIVED field -- a
small, documented, deterministic lookup over (A, C), never a numeric probability.
`build_play_level_confidence` builds one row per (play, component) -- `contact`,
`outfield_defense`, `infield_defense`, `advancement` -- with explicit, machine-readable
reason codes (`near_wall_provisional`, `missing_geometry`, `infield_limited_subgroup_
evidence`, `advancement_rare_class_limited`, plus every `mlb_luck_score.eligibility`
exclusion reason reused verbatim). Near-wall vs. open-field domain status is decomposed
PER ROW by reusing `mlb_luck_score.models.outfield_gating.assign_outfield_opportunity_
gate` unchanged; the OTHER real subgroup issue found in Version 0.7A (`opportunity_time_
q2`/`q3` ECE) is deliberately NOT decomposed per row (would risk drifting from `compare_
opportunity_models.py`'s own quartile definition) -- reported only in aggregate.

### Additive season aggregation (Phase 3)

`mlb_luck_score.scoring.aggregate_attribution` aggregates ONLY quantities that are
additive in run-value units. Version 0.10's own identity has exactly three additive
terms (`unexplained_residual` + `defensive_execution_contribution` +
`advancement_execution_contribution`); there is no independently-additive "contact" term.
This module satisfies the requested four-way split WITHOUT inventing a new double-count,
by PARTITIONING the existing `unexplained_residual` column (never altering any row's
value) into `contact_component` (rows where neither a defense nor advancement model
applies at all) and `unexplained_residual_component` (every other resolved row) -- so the
reported identity

```
season_observed_minus_expected
  = season_contact_component + season_unexplained_residual_component
    + season_defensive_execution_component + season_advancement_component
```

reduces algebraically to Version 0.10's own, unchanged. `assert_additive_only` hard
-rejects known non-additive quantities (`contact_result_surprise`, probabilities, status
strings) from ever being summed as if numeric. Per-batter-season output also includes
eligible-play/game counts, per-100-eligible-play rates, pathway/provisional/unavailable
component counts, `missing_input_frequency`, `component_coverage_fraction`, and `share_of_
value_from_provisional_components` (bounded in [0, 1] by construction).

### Uncertainty intervals (Phase 4)

`mlb_luck_score.scoring.aggregation_uncertainty` produces game_pk-CLUSTERED percentile
bootstrap intervals -- the resampling unit is the GAME (every play from a resampled game
moves together), and each batter-season resamples from THAT BATTER's own games only.
**Interval method: percentile**, chosen and documented before any real result was
examined, for consistency with every other bootstrap already in this codebase (`compare_
opportunity_models`, `compare_near_wall_calibration_gate`, `compare_infield_opportunity`,
`compare_advancement_models` all use the same method) and because BCa's jackknife
acceleration cost scales with average games-per-batter with no house precedent to justify
it. These are SAMPLING-VARIABILITY intervals conditional on the frozen models -- they do
NOT include model-specification, measurement, or labeling uncertainty. Default 1000
replications, deterministic seed 42, documented in `BootstrapDesign`.

### Qualification rules (Phase 5)

`mlb_luck_score.scoring.qualification` defines qualification status using PREDETERMINED
thresholds (three documented sets: `primary`/`strict`/`lenient`, loosely modeled on MLB's
own games-played qualification convention) fixed BEFORE this pipeline was ever run
against real data -- never searched for values that produce an appealing 2024
leaderboard. Five statuses in fixed precedence order: `not_reportable` (zero eligible
plays) > `small_sample` (below the volume bar) > `insufficient_component_coverage`
(enough plays, but too much of the profile falls outside both opportunity models) >
`provisionally_qualified` (enough plays and coverage, but too much value/too many inputs
are provisional/missing) > `qualified`. Players below `qualified` still get their computed
values reported -- only the main rankings exclude them.

### Stability and sensitivity (Phase 6)

`mlb_luck_score.models.evaluate_aggregation_stability` is a descriptive DEVELOPMENT
analysis over 2021-2024 only (not an untouched validation) -- calendar first/second-half
and odd/even-`game_pk` split-half reliability, bootstrap interval width vs. sample size,
qualification-threshold ranking sensitivity, provisional-pathway-exclusion sensitivity
(with the specific biggest-moving players reported), and player-season-level component
covariance.

### Real 2021-2024 result

Run `make run-season-aggregation` then `make evaluate-aggregation-stability` to
(re)generate `outputs/tables/player_season_attribution_v011.json`, `season_aggregation_
v011_report.json`, and `aggregation_stability_v011_report.json`. See notebook
`12_confidence_and_season_aggregation.ipynb` for an inspection walkthrough. As of this
writing, on the held-out 2024 season (123,980 scored plays, 647 batter-seasons):

- **Season identity holds exactly** for every batter-season row (re-verified after
  aggregation, as required).
- **Qualification** (`primary` thresholds): 216 `qualified`, 431 `small_sample`, 0
  `provisionally_qualified`, 0 `insufficient_component_coverage`, 0 `not_reportable` --
  every player who clears the volume bar also clears coverage/provisional-share/
  missing-input thresholds this season. Threshold sensitivity is real: `strict`
  qualifies only 2 players, `lenient` qualifies 378; the top-25 (by `observed_minus_
  expected_per_100`) overlaps `primary` by 0% under `strict` (too few qualifiers to
  overlap) and 32% under `lenient`.
- **Interval width shrinks monotonically with sample size**, as it should: median 95%
  interval width for `observed_minus_expected_per_100` goes from ~26.1 runs (median 8
  games) down to ~7.2 runs (median 146 games) across five games-played bins.
- **Split-half reliability is LOW**: Pearson r = 0.050 (calendar first-half vs.
  second-half) and 0.097 (odd vs. even `game_pk`), among players with >= 20 eligible
  plays in both halves. This is a genuinely informative, expected-shape result for a
  metric explicitly designed to isolate LUCK (variance from expectation) rather than
  skill -- low half-to-half persistence is consistent with the metric capturing what it
  claims to, not a defect. It also means single-half or small-sample per-100 values
  should be read with real caution, exactly what the qualification/interval-width
  machinery above is for.
- **Provisional-pathway sensitivity**: excluding provisional/limited-evidence component
  value gives a Spearman rank correlation of 0.598 against the full metric across all
  647 batter-seasons -- a real, moderate effect, concentrated overwhelmingly among
  players with the FEWEST eligible plays (the biggest movers when provisional value is
  excluded all have single-digit eligible-batted-ball counts).
- **Component covariance**: `total_unexplained_residual_component_runs` correlates
  negatively with both `total_contact_component_runs` (r = -0.37) and `total_defensive_
  execution_component_runs` (r = -0.48) at the player-season level -- descriptive only,
  no causal claim.

## Public Contact Luck score, leaderboard, and presentation contract (Version 0.12)

Freezes Versions 0.2-0.11 completely -- no model, ledger component, confidence rule,
bootstrap procedure, or qualification threshold was refit, recalibrated, tuned, or
altered to produce this phase. This is the first phase that defines a PUBLIC-FACING
contract (schema, ranking, language) on top of the frozen Version 0.11 season
aggregation; it deliberately does not create a Version 1.0 final score.

### Naming note

The task asked for `scoring/public_score.py`, but that name is already taken by the
genuinely live, tested, FROZEN Version 0.1 legacy tanh-based score mapping
(`raw_luck_to_public_score`, imported by `mlb_luck_score.models.predict_outcomes` and
covered by `tests/test_scoring.py`). Overwriting it would have destroyed real, in-use
code. The Version 0.12 assembly module is named `mlb_luck_score.scoring.
public_score_table` instead; the legacy module was not touched.

### Official score definition (task decisions 1-3, 9, 10)

- **Official metric**: `Contact Luck Runs per 100 Eligible Batted Balls`
  (`contact_luck_runs_per_100`, Version 0.11's `observed_minus_expected_per_100`
  renamed for the public contract).
- **Also reported**: `Total Contact Luck Runs` (`total_contact_luck_runs`), the additive
  season total -- reportable whenever `eligible_batted_balls > 0` regardless of any
  component's provisional status, because it is an EXACT observed-minus-expected
  accounting total. Provisional status affects how the component DECOMPOSITION should
  be read, not whether the total exists.
- **Direction**: positive = more favorable realized outcomes than expected; negative =
  less favorable.
- Any percentile/index is secondary, non-additive, and never substituted for the
  runs-per-100 value.

### `batter_name` is not populated

`mlb_luck_score.scoring.public_score_schema` includes `batter_name` ("if available from
a reviewed ID join") because the task asked for it conditionally. It is NOT available:
the raw Statcast `player_name` column is the PITCHER's name, not the batter's (verified
directly against the real data), and there is no reviewed batter-id-to-name join
anywhere in this repository. `batter_name` is always `None` -- left null rather than
populated with a silently wrong name.

### Leaderboard eligibility and ranking policy (Phase 2)

Only rows whose frozen Version 0.11 `qualification_status` is `qualified` may carry an
official rank (`official_rank_eligible`). **Ranking method: competition ranking**
(`.rank(method="min")`) -- ties share the same rank, and the next distinct value skips
ahead by the number of tied rows, matching the convention most public sports
leaderboards already use. **Tie-break for display order**: `batter_id` ascending,
deterministic. Both chosen and documented in `mlb_luck_score.scoring.leaderboard`'s
module docstring BEFORE any real leaderboard output was inspected. Ranking is on
`contact_luck_runs_per_100` alone -- interval endpoints are never used to reorder
players. Two leaderboards are produced: "Most favorable realized luck" and "Least
favorable outcomes relative to expectation" (never described as "worst players").
Non-qualified rows retain their computed values (where Version 0.11 permits reporting
them) but never receive a rank.

### Interval presentation (Phase 3)

`lower_95_interval`/`upper_95_interval` are on the SAME per-100 scale as the point
estimate and are always populated together with it -- never behind a flag or optional
expansion. `interval_interpretation` (`entirely_above_zero` / `overlaps_zero` /
`entirely_below_zero`) is descriptive only and is computed for every row with a
non-null interval, including non-qualified ones -- a row is never suppressed for
crossing zero.

### Component presentation (Phase 5)

The additive decomposition (contact / unexplained residual / defensive execution /
advancement) may be displayed with its run total, per-100 value, and status/reason
codes (`mlb_luck_score.scoring.public_score_table.describe_components`) -- but there is
NO official leaderboard for any individual component. `mlb_luck_score.scoring.
leaderboard` deliberately implements no ranking function for provisional near-wall,
provisional infield, provisional advancement, or any not-calibrated component.

### Optional display index (Phase 4)

`mlb_luck_score.scoring.public_score_table.compute_development_percentile` implements
the OPTIONAL candidate (a frozen-development-distribution percentile among qualified
batter-seasons, rank-based, non-additive, preserving reference seasons/population/
transformation version/tie behavior/out-of-range handling) but it is **NOT adopted**
and **NOT included in the default public table** -- runs per 100 remains the only
official score, per the task's preferred default.

### Public language (Phase 6)

Centralized in `mlb_luck_score.scoring.public_labels` -- the exact required
definitions, the "least favorable" (never "worst players") framing, the retrospective
-not-a-projection limitation sentence, and a banned-phrase checker (`deserved hits`,
`true talent luck`, `guaranteed regression`, `should have produced`, `defense
-independent` unless justified, `statistically significant player`).

### Real 2021-2024 result

Run `make run-public-score` to (re)generate `outputs/tables/public_score_v012.{csv,
parquet,json}`, the favorable/unfavorable leaderboard JSON files, `public_score_v012_
scorecard.json`, and `public_score_v012_review_tables.json`. See notebook
`13_public_score_review.ipynb` for an inspection walkthrough. As of this writing (2024
season, 647 player-season rows, 216 `qualified`, 0 escaped `--allow-final-evaluation`
guards, 2025 never read):

- **Schema validation passes** for the full 647-row table (no duplicate batter-seasons,
  no missing identifiers, no non-finite official values, `official_rank_eligible`
  consistent with `qualification_status` for every row).
- **Zero single-digit-sample players are ever officially ranked** -- the minimum
  `eligible_batted_balls` among ranked rows is 217 (qualification's own 200-play floor,
  as expected).
- **203 of 216 qualified rows (94.0%) have a 95% interval that crosses zero** -- for the
  large majority of qualified players this season, realized Contact Luck Runs per 100
  is not distinguishable from zero at typical sample sizes. This is reported honestly
  and is exactly what the interval is for; it is not a reason to suppress those rows
  (Phase 3's explicit instruction), and per Phase 6's low split-half reliability finding
  (Version 0.11), it is the expected shape for a metric describing realized luck rather
  than persistent skill.
- **Provisional-component sensitivity is real for specific individual players**: the
  development-only ranking comparison (with vs. without provisional-component value)
  shows rank deltas as large as 183 positions for individual qualified players, even
  though the AGGREGATE rank correlation across all 647 rows was 0.598 (Version 0.11).
  Component values remain displayable with their status/reason codes; no official
  component leaderboard was created for any provisional or not-calibrated component.
- **No threshold or model was adjusted** in response to how the 2024 leaderboard looked
  -- per the task's explicit instruction, the review tables above are development
  diagnostics only.

## Pitching Contact Luck (Version 0.13, research spike)

Freezes Versions 0.2-0.12 completely. **This version trains nothing, predicts nothing,
and redefines nothing.** Contact Luck is `observed run value - expected run value` on a
batted ball, and that quantity is a property of the contact, not of the batter -- so the
pitcher-side metric needs no new model. It re-groups the frozen play-level attribution
ledger by `pitcher` instead of `batter` and flips the sign.

Status: **provisional research spike.** It produces a 2024 development table and the
findings below. It is not wired into Version 1.1 prospective scoring, the public-score
schema, or the dashboard, and no public-facing language for it exists in
`public_labels`.

### Why no new model was needed

`SeasonAggregationArtifacts` (`mlb_luck_score.scoring.run_season_aggregation`) already
exposes `scoring_df`/`ledger`/`confidence`, explicitly so a second re-aggregation can be
built "from the SAME scored plays without retraining a second copy of the four component
models" -- `mlb_luck_score.models.evaluate_aggregation_stability` was the first consumer
of that contract, and `run_pitching_contact_luck` is the second. `pitcher` was already
carried through `clean_batted_balls` and `build_contact_features` as an ID column (never
a model feature), and both aggregation entry points were already parameterized by their
grouping key (`aggregate_to_batter_season(..., batter_column=...)`,
`bootstrap_batter_season_intervals(..., batter_column=...)`).

Verified on real 2021-2024 data: `pitcher` is non-null on all 494,173 rows, covering
1,567 distinct pitchers across 3,494 pitcher-seasons.

A useful accident: raw Statcast's `player_name` column is the PITCHER's name (see
"`batter_name` is not populated"), so pitcher names resolve locally with no API join --
61 of 61 qualified 2024 pitchers named, against `batter_name` being permanently null.

### Sign convention

`DEFAULT_RUN_VALUE_MAP` is stated from the batter's perspective: a home run is positive,
an out is negative. The same batted ball means the opposite for the pitcher who allowed
it, so

    pitching_contact_luck = -1 x batting_contact_luck

Positive pitching Contact Luck means outcomes more favorable to the PITCHER than the
contact predicted -- the same sentence the batter metric makes about the batter. The flip
is applied once, to a named list (`SIGNED_VALUE_COLUMNS`), and never inside a component
computation, so the season accounting identity survives it: if `a+b+c+d = T` then
`-a-b-c-d = -T`. Verified on the real 2024 table: max absolute reconstruction error
1.0e-10 across all 61 qualified rows.

Two failure modes this design avoids, both caught by tests written before the code:

- **Negating the total without its components** leaves the four-way decomposition summing
  to the exact negative of its own headline number. No existing identity check covers a
  new code path, so this would have been silent.
- **Negating an interval in place.** `[lo, hi]` negated is `[-hi, -lo]`, not `[-lo, -hi]`.
  Negating both endpoints where they sit leaves every interval reading `low > high`.

### Why the batter qualification thresholds cannot be reused

`QUALIFICATION_THRESHOLD_SETS["primary"]` requires >=200 eligible BBE **and >=100 games**,
mirroring a batting-title convention. Measured on real 2021-2024 data: **0 of 3,494
pitcher-seasons reach 100 games** (max 80, median 19). Reused unchanged it disqualifies
every pitcher who has ever thrown a pitch, including a 674-BBE workhorse starter.

### How `pitcher_primary` was derived (exposure only, fixed before any luck value)

The batter set mirrors MLB's batting-title rule; the pitcher analogue is the ERA-title
rule (1 IP per team game, 162 IP), which qualifies roughly 40-60 pitchers a season.
Innings pitched **cannot** be reconstructed from this project's batted-ball-only table
(strikeouts and walks are not rows), so the bar is set on the project's own unit at the
value whose qualifying COUNT matches what MLB's rule admits:

| BBE bar | 2021 | 2022 | 2023 | 2024 | mean/season |
|---|---|---|---|---|---|
| 400 | 70 | 82 | 76 | 85 | 78.2 -- looser than the ERA title |
| **450** | 41 | 51 | 57 | 62 | **52.8 -- matches it** |
| 500 | 25 | 34 | 31 | 32 | 30.5 -- stricter than it |

`pitcher_primary` = 450 eligible BBE, 15 games. `pitcher_inclusive` = 300 BBE, 10 games,
as a sensitivity cut. Component-quality thresholds are inherited unchanged from the
batter set: they describe how well the four component models covered a set of plays,
which is a property of the plays, not of who threw them. `min_games` is a low guard well
under the 27-game 10th percentile of the >=450 BBE population -- it never binds for the
intended population and exists only to reject a pathological row.

**Only exposure was examined to set these** -- eligible batted balls and games, never a
luck value, rank, or leaderboard shape, per CLAUDE.md's rule against tuning a threshold
to what a result looks like. The counts above were computed before the first pitcher
Contact Luck value was produced.

### This is a starting-pitcher metric

No reliever reaches a starter-scale bar. Across 2021-2024 the highest-volume
pitcher-season with >=50 appearances faced **311** eligible batted balls, and 309 of the
313 pitcher-seasons at >=400 BBE came in <=35 games. The 2024 qualified population is 61
pitchers at 456-610 BBE and 28-35 games.

Relievers are excluded **by exposure, not by choice**, and that must be stated wherever
this metric is shown rather than left as a silent filter. A reliever-scale threshold set
is deliberately deferred rather than guessed: a per-100 rate over ~160 batted balls is a
materially different precision claim, and picking a bar that admits relievers would mean
choosing one without the external anchor the ERA-title rule provides here.

### Every run self-checks against the frozen batter pipeline

`verify_batter_side_reproduction` re-runs the pitcher code path on the `batter` key,
undoes the sign flip, and compares every component total against
`aggregate_to_batter_season`'s own output on identical inputs. On the real 2024 data it
reports `max_abs_difference: 0.0` across all 647 batter-seasons -- exact, not merely
within tolerance -- which is the evidence that re-grouping introduces no arithmetic of
its own.

`pitcher_values_trustworthy` in the report is gated on this check, and the CLI exits
non-zero when it fails. A failed self-check invalidates the run rather than appearing as
a footnote under an otherwise healthy-looking summary.

### 2024 results (development validation season)

854 pitcher-seasons; 61 qualified under `pitcher_primary`. Runs per 100 eligible BBE
among qualified: mean +0.54, sd 1.82, range -2.60 to +4.16. Zero intervals read
backwards; zero nulls; max absolute identity error 1.0e-10. Five of 61 intervals sit entirely above
zero and none entirely below.

### Finding: zero is not the neutral point of a qualified board

Qualified pitchers average **+0.54** runs/100, not ~0. This is a **selection effect, not
a modelling defect**: clearing a starter-scale exposure bar requires having kept a
rotation spot all season, and favorable realized outcomes are part of why a pitcher keeps
one.

The control is that the same conditioning moves the BATTER mean the same direction on the
same plays, so this is a property of qualification rather than of the pitcher side:

| population | all rows | qualified only |
|---|---|---|
| pitchers | -0.520 (n=854) | **+0.544** (n=61) |
| batters | -1.033 (n=647) | **-0.138** (n=216) |

`corr(exposure, luck per 100)` is positive on both sides (+0.080 pitchers, +0.164
batters). The 5-above-zero / 0-below-zero interval split is consistent with the whole
distribution being shifted, not with pitchers having a skill at contact luck.

`qualified_population_reference` in the report records this explicitly --
`mean_runs_per_100`, `median_runs_per_100`, `sd_runs_per_100`, computed on qualified rows
only, so a reader is never left inferring that zero is the neutral point of the board in
front of them. **The score is never recentred** (`score_is_recentered` is always False):
subtracting the reference would redefine Contact Luck, break comparability with every
batter number, and turn a descriptive fact into a different metric. It is context
recorded alongside the score, exactly as the confidence report is descriptive and never
dampens it.

Any future user-facing pitcher surface must carry this. A leaderboard that implies zero
is average would misrepresent every row on it.

### Finding: the contact/residual split is not a pitcher artifact

Component means among qualified pitchers looked alarming in isolation -- contact -1.591,
unexplained residual +1.983, nearly cancelling. Running the batter side on the same plays
gives the near mirror image (contact +1.681, residual -2.038). This is a pre-existing
property of the Version 0.10 decomposition on a qualified population, not something the
pitcher path introduced, and it is recorded here so it is not rediscovered as a
pitcher-side bug.

### What Version 0.13 does NOT do

- No prospective (2026) pitcher scoring, no snapshot integration, no dashboard surface,
  and no `public_labels` entry. `BANNED_PHRASES` already forbids "defense-independent",
  which is the phrase this metric sits nearest to and must not adopt.
- **The `defensive_execution_component` is KEPT in the pitcher total.** It is the defense
  playing behind that pitcher, which he does not control. Excluding it would produce a
  different quantity this project has not defined or validated, and would make the
  pitcher metric non-comparable with the batter metric.
- No reliever threshold set, no role (SP/RP) split, and no per-pitcher name overlay
  beyond the raw `player_name` column used for local review.
- No claim that any pitcher's value reflects talent, or that it will persist. Contact
  Luck is retrospective on both sides.

### Files

| Module | Role |
|---|---|
| `mlb_luck_score.scoring.pitching_contact_luck` | Sign convention, interval swap, pitcher threshold sets, season table, self-check, report |
| `mlb_luck_score.scoring.run_pitching_contact_luck` | CLI. Calls `build_player_season_report` unchanged, then re-aggregates its exposed artifacts |
| `tests/test_pitching_contact_luck.py` | 9 tests |

```bash
.venv/bin/python -m mlb_luck_score.scoring.run_pitching_contact_luck
```

Writes `outputs/tables/pitcher_season_pitching_contact_luck_v013.json` and
`pitching_contact_luck_v013_report.json`. 2025 protection is inherited from
`build_player_season_report` (which calls `assert_seasons_allowed` and independently
re-checks every season present in the input); this script exposes no flag that could
reach a final-test season.

## Pitching Contact Luck presentation research (Version 0.13.1)

Freezes Versions 0.2-0.13 completely. **This version trains nothing, predicts nothing, and
redefines nothing.** Version 0.13 established the pitcher-side quantity and showed it could
be produced exactly. This version answers a different question: **given that quantity, what
can honestly be put in front of a reader?** Its output is a set of presentation findings, a
committed 2024 development fixture, and a local UI prototype -- no model, no threshold, no
score.

Status: **provisional research spike, 2024 development season only.** Nothing here is
scored for 2025 or 2026, nothing is in the public-score schema, and the prototype surface is
gated behind an explicit build flag that no production invocation passes.

### The quantity, restated (unchanged from Version 0.13)

For each outcome-resolved eligible batted ball,

    pitcher Contact Luck contribution = expected run value - observed run value

which is exactly the negative of the batter-side contribution on the same play, and is the
`-1 x batting_contact_luck` convention Version 0.13 already implements in
`SIGNED_VALUE_COLUMNS`. **Positive** means the realized outcome was more favorable to the
pitcher than the contact itself predicted; **negative** means it was less favorable. The
frozen scoring architecture is reused unchanged: no second model, no retraining, no second
scoring path. The pitcher aggregation reproduces the frozen batter-side arithmetic exactly
when run against the `batter` key -- `max_abs_difference: 0.0` across all 647 2024
batter-seasons, re-verified on every run of this version's generator, which refuses to write
its fixture if the check fails.

### The finding that decides the presentation

**Do not rank pitchers on Contact Luck per 100 BBE.** Normalized pitcher rates are too noisy
to support an authoritative season ranking at any workload that actually occurs.

Measured on the 2024 development season
(`outputs/tables/pitcher_prototype_research_report.json`), decomposing the observed spread of
season rates into signal and the measurement error the bootstrap already reports
(`var_observed = var_signal + mean(var_measurement)`, with each row's measurement SD read off
its own 95% interval as `half_width / 1.96`):

| workload floor | n | SD observed | mean SD measurement | resolving power |
|---|---|---|---|---|
| >=60 BBE | 537 | 3.342 | 2.980 | **0.51** |
| >=150 BBE | 305 | 2.547 | 2.323 | **0.45** |
| >=300 BBE | 129 | 1.803 | 1.804 | **0.00** (signal variance -0.002, i.e. negative) |
| >=450 BBE | 61 | 1.819 | 1.626 | **0.50** |

**Resolving power is below 1 at every achievable pitcher workload.** The typical difference
between two pitchers' true rates is smaller than the error bar on either of them, so an
ordering on that quantity is substantially an ordering of noise. At the >=300 BBE floor the
signal-variance estimate comes out *negative*, which is reported raw rather than clipped to
zero: it means the observed spread is no wider than measurement error alone would produce.

The >=1 BBE row is deliberately excluded from that table. It computes to 1.94, and that
number is an artifact, not a finding: the bootstrap resamples GAMES within a pitcher-season,
so the 73 pitcher-seasons with exactly one appearance get zero-width intervals, their
measurement variance is recorded as zero, and the decomposition then attributes all of their
(enormous) spread to signal. Any future use of this estimator must exclude single-appearance
seasons.

Verified directly on the shipped fixture: **exactly 73 rows have a zero-width per-100
interval, and they are exactly the 73 rows with one appearance** -- no zero-width row has
more than one appearance, and no single-appearance row has a non-zero width. Those 73
seasons carry a median of 5 BBE and a per-100 SD of 25.8 (range -143.9 to +55.7), all of it
booked as signal because their measured variance is zero. Dropping them takes resolving
power from 1.939 to **0.877** -- below 1, in line with every other floor in the table. The
1.94 is the artifact; 0.877 is what that population actually supports.

Approximate 95% rate precision requirements, from the same season's own measured intervals
fitted as `half_width = k / sqrt(bbe)` (k = 70.6, median over the 658 rows with >=30 BBE):

| target half-width | resolved BBE required |
|---|---|
| +/-5 runs / 100 | ~199 |
| +/-4 runs / 100 | ~311 |
| +/-3 runs / 100 | ~554 |
| +/-2 runs / 100 | 1,246 |

Every row above is `(k / target)^2` at the single measured `k = 70.588`: 199.3, 311.4,
553.7, 1245.7. An earlier pass reported ~1,275 for the last row; that figure is **not
reproducible from this repository's estimator** and is superseded. It implies `k = 71.41`
(a per-play SD of 0.3644 against the measured 0.3601) -- about 1.2% more noise, the size of
gap produced by fitting `k` on a different subset of rows, not by a different formula. It
was a rough pre-generator estimate with no recorded derivation; only 1,246 has one.

**The 2024 maximum pitcher workload is 610 BBE.** No realistic one-season pitcher sample can
make small differences in per-100 Contact Luck separable. Split-half reliability was also
approximately zero at all pitcher workloads (carried forward from the analysis that produced
this version; not recomputed by this version's generator, which produces the resolving-power
and precision figures above).

**This is not a failure of the metric.** Contact Luck is retrospective luck, not persistent
pitcher skill, and a quantity that does not persist is not supposed to have high split-half
reliability. The finding constrains how it may be PRESENTED, not whether it is correct.

### Primary presentation: cumulative Contact Luck runs

The pitcher surface leads with the **cumulative total**, which answers:

> Across the batted balls this pitcher actually allowed, how many runs did realized outcomes
> differ from what the contact itself deserved?

**The displayed cumulative total is exact for the observed plays, conditional on the frozen
Contact Luck scoring model.** That sentence is the exact claim and its exact limit. The
following are NOT claimed and must never be written:

- that there is no model uncertainty;
- that the number is objectively true independent of the model;
- that the pitcher owns a persistent luck skill.

**Cumulative totals are intentionally workload-sensitive.** A pitcher with more BBE has more
opportunities to accumulate favorable or unfavorable outcomes. They therefore answer an
ACCUMULATION question, and are not workload-independent pitcher quality, skill, talent, a
forecast, or expected future performance. Because of this, **workload must always be visible
beside the total** -- that is a presentation requirement of the quantity, not a design
preference.

### Secondary presentation: per 100 BBE

Contact Luck per 100 BBE remains useful as a normalized secondary statistic and **must not
be the primary ranking key**. Wherever it is shown it must appear with:

- the resolved eligible BBE it is normalized over, and
- a 95% uncertainty interval.

Reading: *per 100 normalizes the observed Contact Luck for opportunity.* Because it is
normalized across unequal samples, it must never be presented with more visual authority
than the cumulative total.

### Why totals matter: heterogeneous opportunity

The pitcher population's opportunity is extremely heterogeneous. 2024 descriptive usage
split, read off the observed bimodal BBE-per-appearance distribution:

| bucket | rule | n | median BBE | max BBE |
|---|---|---|---|---|
| starter-like | >=10 BBE per appearance | 273 | 261 | 610 |
| ambiguous | 8-10 BBE per appearance | 29 | -- | -- |
| reliever-like | <=8 BBE per appearance | 552 | 64 | 282 |

**No reliever-like 2024 season reached 300 BBE.** The gap between the two medians is a factor
of four.

This split is **descriptive only**. The repository has NO authoritative role metadata. Do NOT
call anyone a closer, a starter, or a reliever as an official role. Use **Starter-like** and
**Reliever-like**. Factual usage may be described ("72 appearances, 2.6 BBE per appearance");
"closer" may not be inferred from it.

### Totals vs. rates: the structural failure mode

For a population with this much opportunity spread, cumulative totals avoid a failure mode
that per-100 rankings have by construction. 2024 examples, all position players who pitched:

| pitcher | total runs | BBE | per 100 |
|---|---|---|---|
| Miguel Sano | +0.16 | 1 | +16.3 |
| Oswaldo Cabrera | +0.22 | 1 | +21.9 |
| Carl Edwards Jr. | -0.47 | 1 | -47.1 |

A per-100 ranking places each of these at or near an extreme of the full pitcher population
while their actual cumulative impact is approximately zero. This is why cumulative totals lead
the pitcher presentation.

### The workload relationship (2024, n=854 pitcher-seasons)

| relationship | Pearson | Spearman |
|---|---|---|
| \|cumulative total\| vs. BBE | 0.536 | 0.579 |
| signed cumulative total vs. BBE | 0.194 | 0.102 |

Opportunity moderately affects **how large** the total can become. Opportunity does **not**
strongly determine **whether** the season total is favorable or unfavorable. This is what
supports cumulative totals as an accumulation measure -- provided workload is always visible.

### Reliever-like seasons are frequently one or two plays

Share of 2024 reliever-like seasons (n=552) whose single largest batted ball accounts for
more than a given fraction of the net season total:

| threshold | share of reliever-like seasons |
|---|---|
| >25% of the net total | 83.2% |
| >50% | 52.5% |
| >100% (larger than the whole net) | 27.7% |

**Over half** of reliever-like seasons have one batted ball worth more than half their net,
and better than a quarter have one worth more than the whole of it. The player surface
therefore **surfaces the largest favorable and largest unfavorable Contact Luck plays**. This
is not a flaw to hide: it is how the cumulative total was produced, and concealing it would
make a season total look steadier than it is.

Two display consequences found while building the prototype:

- Where the net total is near zero the ratio `largest |play| / |net|` explodes and stops being
  informative (Josh Winckowski 2024: net -0.10 runs over 241 BBE, ratio 1365%). Above roughly
  2x, say that the season's batted balls very nearly cancelled instead of quoting a percentage.
- A season with ONE resolved batted ball has the same play at both ends of its own
  distribution. Seven 2024 pitcher-seasons are in this position; each must show one play, not
  the same play twice labelled "most favorable" and "least favorable".

### The sub-60-BBE rule is a PRESENTATION rule

Pitcher-seasons below 60 resolved BBE appear on player pages only, not on the ranked
prototype board. If this rule is retained it must be treated **strictly as a presentation
rule** motivated by the extreme width of the normalized-rate interval at that exposure.

It is NOT qualification, NOT eligibility, NOT an official minimum, and NOT an MLB threshold.
Do not invent a baseball qualification rule for it. A season below the line keeps its page,
its total, its rate and its interval, and is never described as having failed anything.

2024 board population at that line: 231 starter-like, 290 reliever-like, 317 pitcher-seasons
withheld from the boards (of which 29 are the ambiguous-usage bucket that belongs to neither
board regardless of workload).

### The hitter product is unchanged, and here is why

Do **not** change the existing hitter leaderboard. At the shipped qualified bar, ranking 2024
qualified hitters by cumulative runs and by runs/100 produce almost the same order:

| measure | value |
|---|---|
| qualified hitters (2024, `primary` threshold set) | 216 |
| resolved eligible BBE range | 217 - 609 (median 369; max/min = 2.8x) |
| Pearson(cumulative total, per-100) | 0.977 |
| Spearman(cumulative total, per-100) | **0.992** |
| median rank change switching to totals | 5 places of 216 |
| p90 rank change | 13.5 places |
| maximum rank change | 24 places (11% of the board) |
| top-10 overlap | 6 of 10 |
| top-25 overlap | 19 of 25 |
| bottom-10 overlap | 6 of 10 |
| bottom-25 overlap | 21 of 25 |
| \|total\| vs. BBE | Pearson 0.041, Spearman -0.028 |
| signed total vs. BBE | Pearson 0.093, Spearman 0.077 |

Totals-first would not materially change the hitter leaderboard, because qualified hitters
already have relatively homogeneous opportunity -- a 2.8x BBE range, against 10.2x across the
pitcher prototype's own boards (60-610) and 610x across all 854 pitcher-seasons. The pitcher
product is allowed a different primary presentation. **Do not alter hitter official-rank
semantics for visual symmetry.**

One honest qualification on that conclusion: it holds for the ORDERING (median 5 places,
max 24 of 216), not uniformly for the HEAD of the board. Four of the ten most-favorable rows
change identity, and all four arrivals are high-exposure seasons (425-541 BBE) that the rate
key ranks 11th or lower. This does not overturn the conclusion -- it is the same
workload-sensitivity the pitcher section documents, showing up in miniature -- but a claim
that the visible top ten is unaffected would be false, and is not made here.

**Provenance of 216 vs. the earlier 283.** An earlier pass reported n = 283, Spearman ~0.988
and ~8 places. Both are arithmetically correct; they are different populations, not a
corrected error:

| n | definition | Spearman | median rank change |
|---|---|---|---|
| 283 | `eligible_batted_balls >= 200` alone | 0.988 | 8 places |
| **216** | `qualification_status == "qualified"` -- the shipped contract | 0.992 | 5 places |

The gap is the `min_games` half of the volume bar. `QUALIFICATION_THRESHOLD_SETS["primary"]`
requires >=200 resolved eligible BBE **and >=100 games**; the 283-row cut applied only the
first. The 67 extra rows are all `small_sample` on games alone (70-99 scored games, every one
of them >=200 BBE), and the other three gates never bind on this season (0
`provisionally_qualified`, 0 `insufficient_component_coverage`). Both paths use the same
`eligible_batted_balls` column -- outcome-resolved rows only
(`observed_contact_result_run_value.notna()`, so `field_error`/`fielders_choice` rows without
an `outcome_class` are excluded from both) -- so the denominator is not the difference.

Only the 216-row figures answer the question this section asks, which is about the SHIPPED
board. The 283-row figures answer "what if the board dropped its games requirement", which is
not a shipped configuration. The conclusion is unchanged and slightly stronger on the shipped
population.

### Limits on every number in this section

- **2024 development season only.** No 2025, no 2026, no prospective scoring.
- **No predictive or out-of-sample claim** is made or supported by anything here.
- Development-only: this is not a validation of the pitcher metric against held-out data, and
  2024 has been iterated against elsewhere in this repository (see "Version 0.7 is now FROZEN"
  in `RESEARCH_RULES.md`).
- Conditional on the frozen Contact Luck scoring model throughout.

### Files

| Module | Role |
|---|---|
| `demo/build_pitcher_prototype_fixture.py` | Offline generator. Calls the frozen batter runner unchanged, re-aggregates by pitcher, projects names/usage/largest plays, and writes the fixture plus the research report backing this section |
| `dashboard/pitcher_prototype_fixture.json` | Committed 2024 development fixture (854 pitcher-seasons). Reviewed reference data, same convention as `demo_fixture.json` |
| `dashboard/pitcher_prototype_content.py` | Fail-closed loader + view-models. Recomputes nothing |
| `dashboard/templates/pitchers.html`, `pitcher.html` | The two prototype surfaces |
| `tests/test_dashboard_pitcher_prototype.py` | 20 tests: fail-closed routing, ranking key, population separation, vocabulary, small-sample honesty |

```bash
# Regenerate the fixture (local dev only; trains the frozen models on 2021-2023,
# scores 2024; no network access):
.venv/bin/python demo/build_pitcher_prototype_fixture.py

# Build the site WITH the local prototype (no production invocation passes this flag):
.venv/bin/python dashboard/build.py \
    --explore-artifacts-dir dashboard/explore_fixture \
    --pitcher-prototype-fixture dashboard/pitcher_prototype_fixture.json
```

Routes emitted only when that flag is passed: `/pitchers/` and `/pitchers/<pitcher_id>/`.
A bare `build.py` emits neither, and no navigation entry points at them -- the same
fail-closed convention the Play Explorer uses, verified by
`tests/test_dashboard_pitcher_prototype.py::TestFailClosedRouting`.

### A fourth `ZeroScale`, declared deliberately

`dashboard/visuals.ZeroScale`'s docstring states the product has exactly three canonical
scales and that adding a fourth is a design review rather than a code change. The prototype
adds `pitcher_cumulative_runs`, and this is that decision written down: its ranked quantity is
a cumulative RUN TOTAL, which no existing scale measures. Drawing it on `league_per_100` would
put a +20-run season off the end of a rate axis; drawing it on `run_value` would put a season
total on a single-play axis. Both would be false alignments.

Invariant Z still holds and is what makes the fourth scale safe: it is built with
`from_values(..., zero_fraction=league_scale.zero_fraction)`, so zero sits at the same
`--cl-zero` as every other figure on the site. Invariant D deliberately does not apply -- the
same exception the `run_value` scale takes -- and the price of that exception is paid in the
template, which prints the scale's own ticks, its own unit and its own domain on every surface
that uses it.

## Version 1.1: prospective 2026 scoring

Version 1.0 (`evaluation/run_v1_final_evaluation.py`) is the sealed, one-time final
evaluation against 2025 -- see CLAUDE.md "The one narrow exception: the sealed Version
1.0 final evaluation" for that protocol. Version 1.1 (`prospective/
run_v1_1_2026_scoring.py`) is a DIFFERENT kind of tool: a repeatable, scoring-only
pipeline that applies the exact frozen Version 1.0 system to 2026 season-to-date data,
producing a new immutable dated snapshot each time it is run. See CLAUDE.md "Version
1.1: prospective 2026 scoring" for the full policy (2025/2026 isolation, the
no-tuning-on-2026 rule, and what a future model change must look like).

**What each snapshot run does**, unchanged from Version 1.0's own component choices:
trains the contact, outfield-opportunity, and infield-opportunity models fresh on
`TRAIN_SEASONS` (2021-2023) via the same frozen `train_model`/`train_opportunity_model`/
`run_infield_model_selection`/`run_advancement_model_selection`/
`run_near_wall_model_selection` functions Version 0.10-1.0 already use, then scores the
requested 2026 window through the frozen eligibility rules, attribution ledger,
confidence framework, additive season aggregation, bootstrap intervals, qualification
thresholds, and public-score/leaderboard contract (Versions 0.2-0.12, unmodified).

**Namespaces**: `data/prospective/2026/`, `outputs/prospective/v1_1/<snapshot>/`,
`artifacts/prospective/v1_1/<snapshot>/` -- fully isolated from both the development
caches (`data/raw`, `data/processed`, `outputs/tables`) and the sealed Version 1.0
namespaces (`data/final_evaluation/2025`, `outputs/final_evaluation/v1`, `artifacts/
final_evaluation/v1`, which the prospective runner refuses to read as anything other
than an optional, read-only seal-integrity check).

**CLI**:

```bash
.venv/bin/python prospective/run_v1_1_2026_scoring.py \
    --data-through 2026-04-15 \
    [--snapshot-label mid-april] \
    [--force-redownload]
```

Deliberately exposes no model-selection, calibration, feature-selection, or
threshold-tuning flag. A completed snapshot directory is never overwritten; rerunning
the same `--data-through` date with identical inputs and code is accepted as a
deterministic no-op, and a rerun that differs raises a conflict error rather than
silently replacing the prior result -- this holds regardless of `--force-redownload`,
which only refreshes the shared, mutable raw cache before a NEW snapshot is built and
can never touch an already-completed one.

Two guards run before any 2026 data is touched. First, the working tree must be clean
-- a dirty tracked file OR an untracked-but-not-ignored file (e.g. an uncommitted new
script) blocks the run, so a snapshot's manifest always names a commit its code can
actually be reproduced from; files under the prospective namespaces above are
gitignored specifically so a prior snapshot's own outputs never falsely trip this.
Second, the requested `--data-through` date must be fully complete -- every scheduled
game on that date must be Final (checked against the MLB Stats API); a postponed or
cancelled game is excluded from that requirement (recorded separately), but a
suspended game is treated exactly like an in-progress one and blocks the date. Both
guards fail loudly rather than silently adjusting anything -- see CLAUDE.md "Version
1.1" for the full detail and rationale.

**Outputs per snapshot**: `public_score.csv`/`.parquet`/`.json`, `favorable_
leaderboard.json`, `unfavorable_leaderboard.json`, `scorecard.json`,
`coverage_and_schema_report.json`, `component_status_summary.json`, `name_
resolution_report.json` (under `outputs/prospective/v1_1/<snapshot>/`), plus
`manifest.json` and `integrity_hashes.json` (under `artifacts/prospective/v1_1/
<snapshot>/`). Notebook `15_prospective_snapshot_review.ipynb` is a read-only reviewer
for an already-completed snapshot -- it performs no fitting, scoring, downloading, or
file mutation.

**Player names**: the public score schema's `batter_name` column (present since Version
0.12 but always null before Version 1.1) is filled via a presentation-only, post-hoc
overlay keyed on MLBAM `batter_id`, sourced from the public MLB Stats API `/people`
endpoint (`mlb_luck_score.data.download_player_names`). Names are never used as model
features and never affect a score, interval, rank, or qualification status -- see
CLAUDE.md for the exact guarantee.

**v1.1.2 operational correctness fix**: `data/prospective/2026/statcast_2026_regular_
season.parquet` was downloaded once (2026-08-06) and then silently reused across three
later `--data-through` requests whose actual coverage had already moved past it -- the
old guard only checked whether the file existed, never whether its coverage reached the
request. Every one of those snapshots is left exactly as it was (immutable, never
modified) but is now known to have scored the same underlying 2026-03-25..2026-08-05 data
regardless of its own `--data-through` label. Fixed by making cache reuse
coverage-provenance-based (a persisted sidecar recording every observed game date, not
just min/max) plus a second, fully independent pre-scoring assertion that never trusts
the earlier reuse decision -- see CLAUDE.md "Version 1.1.2" for the complete incident
writeup and `tests/test_prospective_statcast_cache_coverage.py` for the regression tests.
`--force-redownload` is no longer required for ordinary forward-moving snapshots; the
cache now refreshes itself automatically when it doesn't cover what was requested.

**Status as of this writing**: `outputs/prospective/v1_1/2026-08-05/` is the first real,
genuinely-covered snapshot. `2026-08-06/` and the first `2026-08-08/` both exist as
immutable historical records but reflect the pre-v1.1.2 staleness bug (their own
`scorecard.json.data_through_date` honestly shows `2026-08-05`, since the public-score
schema's own date field is always derived from the actual scored data, never the
requested cutoff). `2026-08-08__refreshed/` is the first snapshot generated with a
genuinely refreshed cache (621 rows, 108 qualified, vs. the earlier 617/103). The real
2026 season-opening date is VERIFIED -- `prospective.prospective_config.PROSPECTIVE_2026_
SEASON_START_DATE = date(2026, 3, 25)`, `PROSPECTIVE_2026_SEASON_START_VERIFIED = True` --
via a maintainer-provided citation of MLB's official 2026 schedule (Opening Night: New
York Yankees at San Francisco Giants, 2026-03-25; the official schedule confirms that
game was played), recorded in `PROSPECTIVE_2026_SEASON_START_SOURCE`/`PROSPECTIVE_2026_
SEASON_START_VERIFIED_AT`.

## Version 0.14: the 2025 pitcher-replication PRE-REGISTRATION freeze

Freezes Versions 0.2-0.13.1 completely. **This version computes nothing, scores nothing,
and opens no season.** It writes down, and hashes, the pitcher specification and the
questions a future held-out 2025 replication would have to answer -- before any 2025
pitcher output exists.

Status: **specification frozen; the 2025 run is NOT authorized.** See "Authorization"
below, which is the most important paragraph in this section.

### Why a freeze precedes the data

The 2024 pitcher work (Versions 0.13, 0.13.1) is development. A replication is only
evidence if the questions, the estimators and the rule that classifies the answers are
fixed before the answers are visible. Otherwise "does it replicate?" collapses into
"which cut of 2025 supports what we already built?"

Three separable things are frozen, and conflating them is the failure mode this version
exists to prevent:

| frozen thing | where |
|---|---|
| the measured quantity | `PLAY_LEVEL_DEFINITION`, `PRIMARY_QUANTITY`, `SECONDARY_QUANTITY`, `DENOMINATOR` |
| the presentation architecture | `ROLE_LIKE_GROUPING`, `PRESENTATION_RULES` |
| the questions and the classification rule | `REPLICATION_QUESTIONS`, `FROZEN_ESTIMATORS`, `CLASSIFICATION_RULE` |

### What is frozen

- **Play level.** `pitcher_contact_luck = expected_run_value - observed_run_value`,
  which is `-1 x batting_contact_luck` on the same play. The frozen scoring architecture
  is reused: no pitcher-specific contact model, no retraining, no second scoring path.
- **Primary season quantity.** Cumulative Contact Luck runs. Exact for the observed
  plays, conditional on the frozen scoring model. Retrospective and deliberately
  workload-sensitive; never skill, talent, persistence, quality, or a forecast.
- **Secondary quantity.** Contact Luck per 100 resolved eligible BBE, never the ranking
  key, always shown with its resolved BBE and a 95% interval from the frozen bootstrap.
- **Denominator.** The repository's existing resolved eligible BBE, unchanged.
  `field_error` and `fielders_choice` are marked ambiguous by
  `mlb_luck_score.eligibility` and never reach the resolved set, so they contribute to
  neither numerator nor denominator nor the games count; `fielders_choice_out` is a
  distinct, unambiguous event and is included. `games` counts distinct games containing
  at least one resolved eligible batted ball.
- **Role-like grouping.** Starter-like >=10 BBE/appearance, reliever-like <=8,
  ambiguous between. Descriptive only. The repository has no authoritative role
  metadata, so "closer", "starter" and "reliever" remain banned as official labels.
- **Presentation rules.** Separate boards, totals-first ordering, BBE always visible,
  per-100 never primary, no board rank on the player card, largest favorable and
  unfavorable plays surfaced, and the sub-60-BBE board exclusion as a display rule that
  is explicitly not qualification. These are not reopened after 2025 is seen unless the
  replication is FIRST reported and frozen as REVISE or NO-GO.

### The seven pre-registered questions

Each names what it reports, its structural hypothesis, and what disagreement would look
like. **None requires 2025 to reproduce a 2024 number** -- each asks whether a structural
relationship holds.

| | question | structural hypothesis |
|---|---|---|
| A | population / centering | shape comparable to 2024; play-level mean near zero |
| B | opportunity heterogeneity | opportunity drives magnitude, only weakly drives sign |
| C | totals vs. rate | all-pitcher rate ordering stays vulnerable to tiny-sample extremes |
| D | reliever single-play dominance | the phenomenon stays materially present |
| E | rate precision | resolving power stays below 1 at every practical floor |
| F | persistence | approximately zero, which confirms the retrospective framing |
| G | real-play sanity check | large contributions stay baseball-sensible |

Question E's estimators live in `replication/pitcher_replication_estimators.py` as of
Version 0.14. They were previously inside `demo/build_pitcher_prototype_fixture.py`, which
meant the freeze depended on presentation code and a purely visual edit could invalidate
it. The extraction is a relocation, not a rewrite: it reproduces **every** Version 0.13.1
value exactly (k = 70.588; 199 / 311 / 554 / 1,246; resolving power 0.508 / 0.4486 / 0.0 /
0.5007; the artifact pair 1.9386 -> 0.8766 over 73 zero-width rows). The fixture generator
now IMPORTS that module, so the committed dashboard fixture and the replication are
computed by the same functions and cannot drift apart. **The frozen source set contains
research code only** -- nothing under `demo/` or `dashboard/`.

Question E carries a **mandatory guard**, now enforced in code rather than remembered: the
>=1 BBE floor is excluded from `PRACTICAL_WORKLOAD_FLOORS`, because a game-clustered
bootstrap gives one-appearance seasons zero-width intervals, records their measurement
variance as zero, and books their enormous spread as signal. On 2024 that produced a
spurious 1.94 against 0.877 with those 73 rows removed. Every report returns
`n_zero_width_intervals`, and negative signal variance is returned raw, never clipped.

Question F is **implemented and frozen** (`replication/pitcher_split_half.py`), resolving
the gap the first freeze recorded. It is a PORT of the accepted Version 0.11 Phase 6
procedure, not a new estimator: the split rules, the inclusion threshold
(`MIN_ELIGIBLE_EACH_HALF = 20`) and the metric are **imported** from
`evaluate_aggregation_stability`, never restated, so "the same design" holds by
construction. The only change is the grouping key.

Both splits are game-clustered -- `calendar` (`game_date <= median`) and `odd_even`
(`game_pk % 2 == 1`), the latter splitting on whole GAMES so no appearance is divided.
Denominators and grouping are inherited unchanged. The pitcher sign flip is irrelevant to
a correlation, which is invariant under a common sign change.

**2024 result** (this repository's first recorded pitcher measurement):

| inclusion bar | split | n | Pearson | Spearman |
|---|---|---|---|---|
| >=20 each half (primary) | calendar | 439 | +0.102 | +0.072 |
| >=20 each half (primary) | odd/even | 550 | +0.077 | +0.055 |
| >=50 each half | calendar | 335 | +0.046 | +0.027 |
| >=50 each half | odd/even | 392 | +0.001 | +0.022 |
| >=100 each half | calendar | 134 | −0.160 | −0.099 |
| >=100 each half | odd/even | 150 | +0.075 | +0.079 |

Approximately zero at every workload examined, with the highest bar turning negative --
what a non-persistent quantity looks like, and a confirmation of the retrospective
framing rather than an indictment of the metric.

**The implementation is validated by exact reproduction.** Run on the BATTER key it
reproduces the committed Version 0.11 Phase 6 numbers to **0.0 absolute difference** on
every value (calendar n=399, Pearson 0.050297542421047836, Spearman 0.07131881210564099;
odd/even n=487, Pearson 0.09654304470474931, Spearman 0.11889983530505965).

Two things are documented rather than tuned away. First, **no precise 2024 pitcher
split-half figure was ever recorded** -- the prior claim was the qualitative
"approximately zero at all pitcher workloads" -- so there is no exact prior pitcher value
to reproduce, and the batter-side exact match is what validates the port. Second, that
wording implies a by-workload breakdown while the accepted procedure reports one number
per split; rather than invent a by-workload reliability estimator, the sweep re-runs the
SAME estimator at three frozen values of its own existing `min_eligible_each_half`
parameter. The >=20 row is primary.

**Question F is secondary.** It is not a success criterion on its own and cannot override
the package-level classification.

### The classification rule

`REPLICATED` / `REVISE` / `NO_GO`, decided on the prespecified findings **as a package**.
Statistical significance on one arbitrary statistic is explicitly not the rule. Every
question that disagrees with its hypothesis is reported, including under a REPLICATED
verdict, and the classification is reported and frozen before any presentation rule
changes in response to it.

### Provenance and write-once

`make freeze-pitcher-replication` writes
`artifacts/pitcher_replication/v0_14/pitcher_replication_freeze.json` (gitignored, same
convention as the Version 1.0 seal), containing the specification verbatim, its content
hash, the frozen constants resolved from their real modules, a SHA-256 for each of **15**
frozen source files, the repository commit, the working-tree state, and a pre-outcome
attestation evidenced by the replication namespace being empty.

The frozen set is research code only: twelve `mlb_luck_score` modules (including
`models/evaluate_aggregation_stability.py`, which question F imports its split rules from,
and which is therefore a frozen input) plus the three `replication/` modules. It
deliberately excludes `demo/build_pitcher_prototype_fixture.py`.

It is write-once. An identical rebuild is idempotent; a different one is refused and
points at `amend_freeze`, which writes a new numbered revision beside the original and
appends to `amendments.jsonl` -- never mutating or deleting a revision, and refusing
unless the caller attests the amendment precedes any 2025 access (re-checked against the
filesystem, not taken on trust). The one narrow exception is `--rebuild-provisional`,
which supersedes a freeze built on a dirty tree while the specification is still being
written; it archives the outgoing copy under `superseded/` and stops working once a
freeze has been written from a clean tree.

### Authorization: granted 2026-09-08, for this replication only

**A 2025 pitcher replication is a SECOND sealed evaluation through a SECOND code path.**
`RESEARCH_RULES.md` permits 2025 to enter this repository exactly once, through
`evaluation/run_v1_final_evaluation.py`, and reserves any second use for "a new, separate
decision requiring the user's explicit sign-off."

That sign-off was given on **2026-09-08** and is recorded in
`replication/pitcher_replication_authorization.py`. Scope: *"One-time held-out 2025
full-season replication of Pitcher Contact Luck under the frozen Version 0.14
specification."*

Three properties make it safe to have written down:

- **It binds by hash.** The record names the authorized `freeze_content_hash` AND
  `spec_content_hash`. `resolve_authorization` refuses to apply it to anything else, so
  amending the specification silently voids the authorization rather than inheriting it.
- **It is one-time.** `assert_no_replication_outputs_exist` fails readiness the moment the
  replication namespace holds output, so the run cannot be repeated.
- **It sits outside the freeze.** The frozen spec's
  `AUTHORIZATION_STATUS["second_sealed_2025_evaluation_authorized"]` still reads `False`,
  because it records the state *at freeze time*. That flag is the evidence the questions
  were fixed before the sign-off; "fixing" it would change `spec_content_hash` and
  invalidate the freeze. The authorization module is likewise absent from
  `FROZEN_SOURCE_RELATIVE_PATHS` — a frozen set cannot contain its own later
  authorization.

Authorization is necessary, not sufficient. `assert_ready_for_2025` still requires a clean
tree, a non-provisional freeze, intact 2025 protection in `mlb_luck_score.config`, an
empty replication namespace, and full freeze validation. It withholds everything listed in
`AUTHORIZATION_EXCLUSIONS` — changing the metric, denominator, estimators, role/display
rules or thresholds; adding metrics after seeing 2025; selecting subsets; redesigning the
UI before the result is reported; reading 2026; touching Contact Forecast; deploying.

**No other research line may read 2025 on the strength of this**, and it does not extend
to 2026.

### Files

| Module | Role |
|---|---|
| `replication/pitcher_replication_spec.py` | The pre-registration itself: constants only, no I/O |
| `replication/pitcher_replication_estimators.py` | Question E's estimators, extracted from the fixture generator |
| `replication/pitcher_split_half.py` | Question F, ported from the accepted Version 0.11 procedure |
| `replication/pitcher_replication_authorization.py` | The maintainer's 2026-09-08 sign-off, bound to the freeze by hash |
| `replication/pitcher_replication_freeze.py` | Hashing, guards, write-once artifact, amendment path, the 2025 gate |
| `tests/test_pitcher_replication_freeze.py` | 94 tests, including proof that building the freeze reads no data |
| `tests/test_pitcher_replication_estimators.py` | 26 tests, including exact reproduction of every Version 0.13.1 value |
| `tests/test_pitcher_split_half.py` | 18 tests, including that the procedure is imported, not copied |

```bash
make freeze-pitcher-replication
# still writing the spec, superseding a dirty-tree freeze:
make freeze-pitcher-replication REBUILD_PROVISIONAL=1

# replication/ is a plain script directory like evaluation/ and prospective/,
# so `make check` does not lint or typecheck it -- run these explicitly:
.venv/bin/python -m ruff format replication && .venv/bin/python -m ruff check replication
.venv/bin/python -m mypy replication
```

## Version 1.2: dashboard deployment and operations

Version 1.2 (`dashboard/`) is a read-only, static-site presentation layer over Version
1.1's immutable prospective snapshots -- see CLAUDE.md and `dashboard/snapshot_data.py`'s
module docstring for the full snapshot-selection/precedence rules and integrity
guarantees. This section documents the existing production workflow: generate a
snapshot, rebuild the dashboard, deploy the static output. It changes no model,
prospective-scoring, snapshot, or dashboard logic.

**1. Generate the newest snapshot only after the requested MLB slate is fully
complete.** Version 1.1's own `assert_data_through_date_is_complete` guard already
refuses an incomplete date (see "Version 1.1: prospective 2026 scoring" above), so this
is enforced, not just a convention:

```bash
.venv/bin/python prospective/run_v1_1_2026_scoring.py \
    --data-through YYYY-MM-DD
```

**2. `--force-redownload` is not part of the normal workflow.** The prospective
runner's raw Statcast cache is coverage-aware -- it refreshes itself automatically
whenever its verified coverage doesn't reach the requested `--data-through` date (see
"v1.1.2 operational correctness fix" above). Only pass `--force-redownload` if there is
a specific reason to force a cache refresh regardless of coverage.

**3. Verify the snapshot completed successfully** by checking its manifest
(`artifacts/prospective/v1_1/<snapshot>/manifest.json`) reports:
- the requested `data_through_date`
- `observed_date_coverage` actually reaching that date
- `schema_checks.raw_statcast_cache_coverage_validation.decision` (`reused` or
  `refreshed`) and its `reason`
- `schema_checks.final_pre_scoring_coverage_check.missing_completed_game_dates` is empty
- `output_hashes`/`frozen_artifact_hashes` are present -- the same information
  `dashboard/snapshot_data.py` independently re-verifies (via `integrity_hashes.json`)
  before the dashboard will ever display the snapshot; see "Integrity validation" in
  that module's docstring

**4. Build the dashboard:**

```bash
.venv/bin/python dashboard/build.py
```

**5. The dashboard automatically resolves the newest valid preferred snapshot** -- no
flag is needed to point it at a specific date. Historical snapshots remain immutable; a
corrected/refreshed snapshot for an earlier date never affects which snapshot is newest
overall (see `dashboard/snapshot_data.py`'s "Snapshot precedence" docstring section).

**6. Local preview:**

```bash
cd dashboard/dist
python3 -m http.server 8000
```

**7. Production artifact:** `dashboard/dist/` -- a plain static site (HTML/CSS/JS plus a
couple of small JSON payloads). This entire directory can be deployed to a static host
as-is.

**8. Production architecture:**

```
completed MLB slate
    -> immutable prospective snapshot (prospective/run_v1_1_2026_scoring.py)
    -> dashboard/build.py
    -> dashboard/dist/
    -> static hosting
```

**9. Operational guarantees:**
- Production hosting never trains or scores models -- `dashboard/dist/` is static
  output; nothing in it executes Python.
- The dashboard never downloads Statcast -- see `tests/test_dashboard_isolation.py` for
  the structural check that no module under `dashboard/` imports scoring, training, or
  download code.
- Sealed 2025 evaluation data is never used as live product data -- the dashboard only
  reads `outputs/prospective/v1_1/`/`artifacts/prospective/v1_1/`.
- Missing historical snapshot dates are allowed and are never interpolated -- a gap
  (e.g. a day with no snapshot run) simply has no entry in the trend or snapshot
  history; see `dashboard/visuals.py`'s trend-chart docstring.
- Corrected snapshots follow the dashboard's deterministic precedence rules -- a later,
  differently-labeled snapshot for the same `--data-through` date supersedes an earlier
  one for display (both remain on disk, unmodified); a retrospective-backfill snapshot
  (if one is ever produced by a future, separate mechanism) never supersedes a genuine
  or corrected one for the same date.

**Deploying.** The dashboard is hosted on Cloudflare Pages (project `contact-luck`),
deployed via `wrangler pages deploy dashboard/dist --project-name=contact-luck`.
`scripts/publish_snapshot.sh` (below) wraps the whole snapshot-to-deploy chain behind
one command and remains the only orchestration entry point -- nothing reproduces its
commands elsewhere.

### Operational entry point: `scripts/publish_snapshot.sh`

```bash
scripts/publish_snapshot.sh --data-through YYYY-MM-DD [options]
```

Runs, in order, and stops at the first failure: `prospective/run_v1_1_2026_scoring.py
--data-through <date>` -> `dashboard/build.py` -> (unless `--skip-deploy`) a confirmation
prompt -> `wrangler pages deploy dashboard/dist --project-name=contact-luck`. It
duplicates none of Version 1.1's guards (clean working tree, date completeness, coverage
validation, conflict detection) -- it only calls the existing entry points and reports
their exit codes. Flags: `--snapshot-label`, `--project-name` (default `contact-luck`),
`--skip-deploy` (build only, no deploy -- the dry-run path), `--yes` (skip the
interactive confirmation, required for any non-interactive/CI invocation), `--help`.

**Manual publication.** Run the command above directly from a clean working tree once a
date's MLB slate is fully complete. Omit `--skip-deploy` to deploy for real (you'll be
asked to confirm unless `--yes` is also passed).

### Scheduled publication: `.github/workflows/publish-prospective.yml`

A GitHub Actions workflow triggers `scripts/publish_snapshot.sh` on a schedule, so
publishing doesn't require a human to run the command by hand every day. It is a thin
trigger only -- it builds the same `.venv` the script expects
(`python -m venv .venv && .venv/bin/pip install -e ".[dashboard]"`), calls the script as
a single atomic step, and stops there. It never reimplements or bypasses any Version 1.1
guard.

- **Manual trigger**: GitHub -> Actions -> "Publish prospective snapshot" -> "Run
  workflow". Inputs: `data_through` (optional -- blank auto-resolves yesterday in
  America/New_York) and `deploy` (checkbox, default OFF -- manual runs default to a dry
  run, matching the scheduled default below).
- **Schedule**: once daily at `13:00 UTC` (`0 13 * * *`). GitHub Actions cron is fixed
  UTC and does not shift for daylight saving: `13:00 UTC` is `09:00 America/New_York`
  during EDT (roughly mid-March to early November -- most of the season) and `08:00`
  during EST. Either is a conservative morning buffer after even a late West Coast
  extra-inning game; the cron time only needs to land "safely after games usually end,"
  not be exact, because the actual `--data-through` date is resolved separately (next
  point) and a slate that somehow isn't complete yet is rejected by Version 1.1's own
  date-completeness guard rather than silently scored partial.
- **Date resolution**: `scripts/resolve_data_through_date.sh` resolves "yesterday in
  America/New_York" via the runner's tzdata (`TZ="America/New_York" date --date="...
  yesterday"`), not a fixed UTC offset -- a run firing at, say, 02:00 UTC can still be
  evening of the previous day in New York, and a naive UTC-calendar-date approach would
  be off by one. Extracted into its own script specifically so this logic has direct
  test coverage (`tests/test_resolve_data_through_date.py`, Linux-only -- see that
  script's header for why) independent of running the workflow or real prospective
  scoring. It also fails loudly (rather than silently returning a wrong date) if the
  `tzdata` package is missing -- discovered during development that a bare
  `TZ=America/New_York` conversion against a missing zoneinfo file does not error, it
  just silently fails to convert.
- **Current mode: `PROSPECTIVE_AUTO_DEPLOY` is set to `true` -- scheduled runs perform
  real archive writes and real deploys.** The rollout gate this variable provides was
  exercised as designed before being flipped: scheduled/dry-run cycles were inspected
  (via the uploaded `dashboard/dist/` and snapshot-manifest artifacts) to confirm the
  season-to-date trend charts showed the full historical series before any unattended
  real deploy was allowed to happen -- see "Durable archival"/"Frozen input bundle
  portability" below for that verification work, and the "known gap" note there for what
  is still not automated. With the variable unset or anything other than `true`,
  scheduled runs fall back to `scripts/publish_snapshot.sh --data-through "$DATE"
  --skip-archive --skip-deploy --yes` -- snapshot generation, the read-only history sync,
  and the dashboard rebuild happen for real, but nothing is written to R2 and nothing is
  deployed. `publish_snapshot.sh` refuses `--skip-archive` without `--skip-deploy` (see
  the flag table above), so this workflow only ever produces one of two states: both
  flags, or neither.
- **Toggling the mode**: set the repository variable `PROSPECTIVE_AUTO_DEPLOY` (GitHub ->
  Settings -> Secrets and variables -> Actions -> Variables) to `true` for real scheduled
  deploys, or to anything else (or unset it) to fall back to scheduled dry runs. That is
  the *only* change needed either direction -- the workflow YAML does not change. A manual
  run can independently opt into a real deploy any time via the `deploy` checkbox,
  regardless of this variable.
- **Required GitHub secrets.** `CLOUDFLARE_API_TOKEN` and the actual Cloudflare Pages
  deploy are only consumed when an actual deploy happens (scope the token to Pages edit
  access for this project only, not full account access). The four `R2_*` credentials
  (`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, plus `CLOUDFLARE_ACCOUNT_ID` reused as
  `R2_ACCOUNT_ID`), in contrast, are consumed on **every** run now, dry or not -- history
  sync always reads from R2, even in dry-run mode. Set under GitHub -> Settings -> Secrets
  and variables -> Actions -> Secrets. Never committed anywhere in this repository.
- **Concurrency**: all runs share a single `publish-prospective` concurrency group with
  `cancel-in-progress: false` -- a second trigger while one is in flight queues rather
  than racing it or cancelling a possibly-mid-deploy run.
- **When a guard rejects a run**: the workflow step fails with `scripts/
  publish_snapshot.sh`'s own exit code and error message (e.g. a dirty working tree, an
  incomplete `--data-through` date, or a snapshot that already exists with different
  inputs) -- exactly as it would locally. The workflow does not retry (a retry could race
  or attempt to bypass Version 1.1's own immutability/conflict handling) and never falls
  back to an earlier date on its own.

### Frozen input bundle portability: `scripts/ensure_frozen_inputs.py`

**The problem, in full.** Version 1.1's snapshot manifest freezes its inputs by hashing
every path in `evaluation.v1_final_evaluation_manifest.FROZEN_ARTIFACT_RELATIVE_PATHS` --
the SAME 32-path list the sealed Version 1.0 evaluation uses (`prospective.
prospective_manifest.build_snapshot_manifest` calls `build_artifact_hashes(repo_root,
relative_paths=FROZEN_ARTIFACT_RELATIVE_PATHS)` directly, never a second,
prospective-specific list). 27 of those 32 are `src/mlb_luck_score/*.py` source files --
already git-tracked, present on any fresh checkout automatically. The remaining FOUR are
gitignored, local-only data/detail files (CLAUDE.md "Never commit datasets") that have
only ever existed on a maintainer's own long-lived machine:

| Local path | Produced by |
|---|---|
| `data/processed/cleaned_development_data_with_sprint_speed.parquet` | the Version 0.x data-cleaning pipeline |
| `outputs/tables/opportunity_model_comparison_detail.json` | `mlb_luck_score.models.compare_opportunity_models` |
| `outputs/tables/infield_opportunity_detail.json` | `mlb_luck_score.models.compare_infield_opportunity` |
| `outputs/tables/advancement_detail.json` | `mlb_luck_score.models.compare_advancement_models` |

The first scheduled GitHub Actions dry-run failed on the parquet alone (fixed first, as an
initially parquet-only version of this module). The SECOND scheduled dry-run then got all
the way through downloading/cleaning 2026 data and fitting/selecting the frozen component
models, only to fail building the snapshot manifest on the remaining three -- exposing
that the portability gap was never just the parquet. `scripts/ensure_frozen_inputs.py`
now covers the complete, audited set as one versioned bundle, not three more one-off
patches.

**The fix, and why it never regenerates anything.** `scripts/ensure_frozen_inputs.py`
runs as the very first stage of `scripts/publish_snapshot.sh`, before scoring, and
ensures EVERY bundle entry in order: reuse the local file if it's already there AND its
sha256 matches a hash pinned in source (never trusted blindly), otherwise fetch it from a
dedicated, immutable Cloudflare R2 object and verify the same pinned hash before writing
anything to disk. Processing is fail-fast -- the first entry that can't be satisfied stops
the whole stage immediately, and no later entry is attempted. There is no third option for
any entry: the module contains no import of, or call into, any cleaning/feature
-engineering/model-comparison/Statcast-download code (see `tests/
test_ensure_frozen_inputs.py::TestNoRegenerationFallback`'s structural AST check), so
"regenerate it" is not a code path that exists here, not just a policy this script happens
to follow.

- **Storage layout: one bundle, relative-path-preserving keys.** The three detail JSONs
  use keys that mirror their local relative path exactly, e.g.
  `frozen-inputs/v1/outputs/tables/opportunity_model_comparison_detail.json`. The parquet
  keeps its already-live key unchanged (`frozen-inputs/v1/cleaned_development_data_with_
  sprint_speed.parquet`, flat) rather than being migrated to match -- that object was
  already uploaded and independently verified against real R2 before this bundle existed,
  and re-keying it would silently orphan a working, already-verified production object for
  no functional benefit. Every key lives under the SAME R2 bucket `scripts/
  archive_snapshot.py` uses (so CI needs no new secrets), under a deliberately separate
  top-level prefix (`frozen-inputs/` vs. `prospective/`) so none of this bundle can ever be
  discovered by, or confused with, `list_archived_snapshots()`/`--sync-history`'s per-date
  snapshot enumeration. The `v1` segment versions the ARCHIVED INPUT BUNDLE itself; a
  legitimately different frozen input (a new Version 0.x data-cleaning or model-selection
  run) would become a `frozen-inputs/v2/...` key with its own pinned hash, never an
  in-place overwrite.
- **The pinned hashes**, each independently cross-checked against `outputs/
  final_evaluation/v1/v1_final_report.json`'s own `manifest.artifact_hashes` entry for that
  same path (recorded when the real, sealed Version 1.0 evaluation actually ran against
  it) -- every entry agrees exactly:

  | Local path | SHA256 | Size |
  |---|---|---|
  | `data/processed/cleaned_development_data_with_sprint_speed.parquet` | `f791415d218334aa578fd8104e7c3aec5f52716931fc69287aa2ee05b12443f8` | 111,445,297 bytes |
  | `outputs/tables/opportunity_model_comparison_detail.json` | `a694f6f65f0f94b7ed30fd785144a363d5f075a4a465f6a5b9a0eda2237a1d38` | 49,619 bytes |
  | `outputs/tables/infield_opportunity_detail.json` | `2b879a72af11618b8d9f8939d900120c325c20990698261d7f4dcbb95b8141f0` | 85,434 bytes |
  | `outputs/tables/advancement_detail.json` | `8826bae37c6b51489098018e95154d47ec0e1f05cd7c52bd29a633b20d01e222` | 18,174 bytes |

- **Local-first, zero network dependency in the common case.** The R2 client is
  constructed AT MOST ONCE per `ensure_frozen_inputs.py` invocation, shared across every
  bundle entry that actually needs one, and never constructed at all if every local file
  is already present and valid -- true of a maintainer's own machine, false of a fresh CI
  runner (before the one-time seed) or any runner missing part of the bundle.
- **Fails loudly, never silently, on every anomaly, for every entry**: a local file that
  exists but doesn't match its pinned hash (never redownloaded or overwritten -- investigate
  by hand), a missing R2 object, or a downloaded object whose hash doesn't match the pin
  (discarded, never written to disk looking like a verified artifact). See
  `FrozenInputLocalHashMismatchError` / `FrozenInputMissingRemoteObjectError` /
  `FrozenInputRemoteHashMismatchError`.
- **The bundle can never silently drift out of sync with the real frozen-artifact list.**
  `tests/test_ensure_frozen_inputs.py::TestBundleCoversEveryGitignoredFrozenArtifact`
  cross-references every path in `FROZEN_ARTIFACT_RELATIVE_PATHS` against `git
  check-ignore` and asserts every gitignored one is present in `FROZEN_INPUT_BUNDLE` --
  exactly the audit that would have caught today's gap automatically before it ever
  reached a real dry-run, and the guard that keeps a FUTURE frozen artifact from repeating
  it.
- **The one-time seeding upload (`--upload`) is a separate action**, never invoked by
  `ensure_frozen_input_bundle()` or `scripts/publish_snapshot.sh` -- run by hand, once per
  entry that hasn't been seeded yet, with explicit human intent, mirroring `scripts/
  archive_snapshot.py`'s own write-once philosophy (refuses to upload a local file that
  doesn't match the pin, refuses to overwrite an existing, DIFFERENT R2 object, re-verifies
  each upload by fetching it back). Safe to re-run over an already-fully-seeded bundle --
  every entry simply no-ops.
- **This stage always runs**, even under `--skip-archive --skip-deploy`, because scoring
  and manifest generation need the complete bundle regardless of what happens to the
  snapshot's output afterward -- the same reasoning that makes history sync always run.
  See `tests/test_publish_snapshot_orchestration.py::TestFailuresBlockLaterStages::
  test_ensure_frozen_inputs_failure_prevents_everything_after_it` for the orchestration
  proof that a failure here blocks scoring (and everything after it), exactly like a
  scoring failure would.

### Durable archival: `scripts/archive_snapshot.py`

**The problem.** `outputs/prospective/v1_1/`/`artifacts/prospective/v1_1/` are gitignored
by design (see "Version 1.1" above) -- correct for a snapshot generated on a
maintainer's own machine, but a snapshot generated by a GitHub Actions run exists only
on that run's disposable runner and vanishes when the job ends. `scripts/
archive_snapshot.py` copies a completed local snapshot's files, byte for byte, into a
durable Cloudflare R2 bucket, so an official snapshot survives runner destruction.

**Pipeline ordering: score -> archive -> sync history -> build -> deploy.** Archival runs
immediately after scoring, history sync runs immediately after that, and the dashboard is
never built (let alone deployed) if EITHER fails. This is deliberate: the raw scoring
output is the precious, comparatively irreplaceable artifact (re-scoring an old date
depends on the same historical Statcast data still being fetchable, which is not
guaranteed indefinitely), while the dashboard build is cheap and already iterated on
constantly. Archiving first means an official snapshot is durable even if a LATER stage
breaks, and it means "the public site must never deploy if durable archival failed"
falls directly out of `scripts/publish_snapshot.sh`'s existing sequential
`set -euo pipefail` structure -- no special-cased check was needed. This ordering
guarantee has a dedicated regression test (`tests/test_publish_snapshot_orchestration.py`)
that runs the real script against fake `.venv/bin/python`/`npx` executables and asserts,
by inspecting what was actually invoked, that a failed archive OR a failed history sync
genuinely prevents the build and deploy steps from running -- not just an argument that
`set -e` ought to guarantee it.

**`--skip-archive` and `--skip-deploy` are independently controllable**, specifically so
real R2 archival can be validated on its own before production deploys are enabled.
History sync is NOT gated by either flag -- it always runs, because it only ever READS
from R2 (never writes), so even a full dry run exercises the real CI dashboard-build
behavior (see "History sync" below for why that matters):

| Flags | Behavior |
|---|---|
| (neither) | `score -> archive -> sync history -> build -> deploy` |
| `--skip-deploy` | `score -> archive -> sync history -> build -> stop` (archives for real, doesn't deploy) |
| `--skip-archive --skip-deploy` | `score -> sync history (read-only) -> build -> stop` (writes to neither R2 nor Cloudflare Pages) |
| `--skip-archive` alone | **refused** -- the script will not deploy a dashboard built from a snapshot that wasn't just durably archived; there is no override flag for this |

Scoring, history sync, and the dashboard build happen for real in every row above; only
the archive WRITE and the deploy are ever skipped.

**History sync (why it exists, and why the dashboard itself stays filesystem-only).** A
GitHub Actions runner starts from a fresh git checkout -- `outputs/prospective/v1_1/`/
`artifacts/prospective/v1_1/` are gitignored, so a CI job that just scored today's date
has ONLY today's snapshot locally. Without more, `dashboard/build.py`'s season-to-date
trend charts would show a single point on every CI-built dashboard, no matter how much
history is sitting in R2 -- `dashboard/snapshot_data.py` itself was deliberately never
changed to fetch from R2 directly (see its own module docstring's "read-only boundary");
it still only ever reads local disk. Instead, `scripts/archive_snapshot.py --sync-history
--season <year>` runs as its own pipeline stage, between archiving and the dashboard
build, and repopulates local disk from R2 before the dashboard ever sees it:

- `list_archived_snapshots(season=...)` enumerates every candidate `<snapshot_dir_name>`
  under `prospective/<season>/` in the archive (never a hardcoded date list). A key only
  becomes a candidate at all if its name has the SHAPE of a real snapshot directory
  (`YYYY-MM-DD` or `YYYY-MM-DD__<label>`, a genuine calendar date) -- a key that reaches
  the right depth but isn't shaped like a snapshot (some unrelated object under the same
  prefix) is silently ignored, never reported. Every candidate that DOES pass the shape
  check then gets a completeness determination: complete only if `artifacts/
  integrity_hashes.json` exists, parses, AND every file it references is also actually
  present in the archive.
- `sync_missing_snapshots(season=...)` restores every COMPLETE archived snapshot missing
  locally, using the exact same `restore_snapshot()` hash-verification machinery. A
  `<snapshot_dir_name>` that already exists locally is either an identical, safe no-op
  (compares `integrity_hashes.json` AND re-hashes every referenced local file) or a
  `HistorySyncConflictError` -- partial/corrupt/differing local snapshots are NEVER
  auto-repaired or overwritten. A candidate that passed the shape check but failed
  completeness raises `HistorySyncIncompleteArchiveError` and stops the whole sync
  immediately -- a broken OFFICIAL snapshot entry is never silently skipped, only a
  genuinely unrelated key (one that never became a candidate) is ignored without blocking
  anything else.
- The representative end-to-end regression test (`tests/test_archive_history_sync.py`)
  builds a scenario with several historical snapshots that exist ONLY in the archive plus
  a freshly-scored local "today" snapshot -- exactly what a real CI runner sees -- and
  confirms `dashboard/snapshot_data.discover_snapshots()` sees only today's point BEFORE
  sync, and the full, correctly-precedented date history AFTER it.

**Archive layout** mirrors the existing two-namespace local contract exactly, for every
file actually present (enumerated dynamically, never a hardcoded filename list, so a
future Version 1.1 output file is archived automatically):

```
prospective/<season>/<snapshot_dir_name>/outputs/<every file from
    outputs/prospective/v1_1/<snapshot_dir_name>/>
prospective/<season>/<snapshot_dir_name>/artifacts/<every file from
    artifacts/prospective/v1_1/<snapshot_dir_name>/>
```

`<season>` comes from the snapshot's own `manifest.json` (`prospective_season`), and
`<snapshot_dir_name>` is `YYYY-MM-DD` or `YYYY-MM-DD__<label>` -- identical to the local
directory-naming convention `run_v1_1_2026_scoring.py`/`dashboard/snapshot_data.py`
already use, so a corrected/refreshed snapshot (e.g. `2026-08-08__refreshed`) archives as
its own separate, additional entry, exactly mirroring how it exists locally.

**Write-once.** `artifacts/integrity_hashes.json` (already written by Version 1.1) is
reused directly as the identity anchor -- no second hashing scheme was invented. If the
archive has no entry yet for a `<snapshot_dir_name>`, every local file is uploaded, then
immediately re-fetched and compared to confirm the upload actually took. If an entry
already exists, its `integrity_hashes.json` is compared (as parsed JSON, not raw bytes)
against the local one: identical -> no-op (safe to rerun); different -> `ArchiveConflictError`,
and the archive is NEVER silently overwritten.

**Recovery -- two related but distinct tools.** `scripts/archive_snapshot.py --restore
--data-through YYYY-MM-DD --season 2026 [--snapshot-label LABEL]` is the explicit,
on-demand reverse operation for ONE known snapshot -- downloads it back into the normal
local paths, verifies every file against the archived hashes, and refuses (never silently
overwriting) if different local files already exist at that path. `--sync-history` (above)
is the bulk analog: every missing snapshot for a season, discovered automatically rather
than named one at a time, which is what the automated pipeline actually runs. **Either
way, this is the only way R2 data reaches local disk** -- the dashboard (`dashboard/
snapshot_data.py`) and Version 1.1's own guards are UNCHANGED by this: they still only
ever read local disk. Nothing auto-restores from R2 outside these two explicit calls; if
a future version wants the dashboard itself to depend on R2 at runtime, that is a real
architectural change (a new runtime dependency on remote state) and deserves its own
explicit decision -- not something either tool does quietly.

**Cloudflare R2 configuration.**
- A bucket dedicated to this archive (`contact-luck-prospective-archive`) -- created and in
  active use (see below); nothing in this codebase creates one automatically, so a future
  fork/redeploy still needs this step done by hand.
- An **R2 API token** scoped to Object Read & Write on that one bucket only (Cloudflare
  dashboard -> R2 -> Manage R2 API Tokens) -- deliberately narrower than the general
  `CLOUDFLARE_API_TOKEN` used for Pages deploys, and never full account/admin access.
- Required GitHub secrets: `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` (from that R2 API
  token). `R2_ACCOUNT_ID` is NOT a separate secret -- R2 shares the same Cloudflare
  account as Pages, so the workflow reuses the existing `CLOUDFLARE_ACCOUNT_ID` secret.
- Required GitHub repository variable: `PROSPECTIVE_ARCHIVE_BUCKET` (the bucket name --
  not sensitive, so a variable rather than a secret, matching `PROSPECTIVE_AUTO_DEPLOY`).
- Implementation: `boto3` (the `archive` extra, `pip install -e ".[archive]"`) against
  R2's S3-compatible endpoint (`https://<account_id>.r2.cloudflarestorage.com`) -- a
  library rather than a CLI specifically so the write-once/identity-comparison logic
  could be unit-tested against a plain in-memory fake
  (`tests/test_archive_snapshot.py`) without contacting real Cloudflare services or
  needing a mocking library.
- The bucket (`contact-luck-prospective-archive`) has been created and is in active use.
  It was first validated with a real R2 integration test (archive, remote hash
  verification, idempotent-rerun check, and a restore into an isolated temp directory that
  passed the real dashboard integrity validator) using the existing, already-official
  `2026-08-09` snapshot -- no new date was scored to test this, and nothing was deployed.
  Every pre-existing local snapshot (`2026-08-05`, `2026-08-06`, `2026-08-08`,
  `2026-08-08__refreshed`, `2026-08-09`) was subsequently archived for real, and history
  sync was independently verified against the real bucket with real multi-date history --
  restoring all five into a fully empty simulated fresh-runner directory and confirming
  the dashboard's own precedence logic picks `2026-08-08__refreshed` for that date's trend
  point (both Aug 8 variants stay archived for auditability; Aug 7 correctly has no
  entry).

**Known gap this does not solve**: every CI run still starts with a cold local
Statcast/game-metadata cache (`data/prospective/2026/` is gitignored and ephemeral on the
runner, same as before archival existed) -- durable archival of the SCORED OUTPUT does not
change that every scheduled run currently re-downloads the season-to-date raw data from
scratch. Worth solving eventually, but kept out of scope here deliberately so this change
stays focused on output durability/auditability.

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
- A reliever-scale pitcher qualification threshold set, and a role (SP/RP) split, for
  Pitching Contact Luck (Version 0.13) -- deferred rather than guessed, because a
  per-100 rate over ~160 batted balls is a different precision claim and no external
  anchor comparable to the ERA-title rule was identified for relievers
- Whether the qualified-population selection effect documented in Version 0.13 ("zero is
  not the neutral point of a qualified board") warrants a reported reference point on the
  BATTER side too -- it is present there as well (-1.033 all rows -> -0.138 qualified),
  and is currently recorded only in the pitcher report
- Prospective (2026) pitcher scoring, a pitcher public-score contract, and pitcher
  `public_labels` copy -- none exist; Version 0.13 is a development-season spike only

---

See `CLAUDE.md` / `AGENTS.md` for persistent rules that govern how this repository should
be extended by AI coding agents.
