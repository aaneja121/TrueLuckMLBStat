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
metadata are written under `artifacts/` (git-ignored).

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
  counts, contact-variable distributions.
- `02_contact_model.ipynb` -- time-based split demo, feature pipeline, baseline
  training, predictions, core evaluation metrics.
- `03_calibration.ipynb` -- calibration table generation, one plot per outcome class,
  interpretation warnings, sample-size reporting.
- `04_luck_score_demo.ipynb` -- one example play end-to-end: actual outcome, predicted
  distribution, raw luck, public score, confidence report, explicit disclaimer.

All four notebooks detect missing data/artifacts and print clear instructions instead of
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

## Untouched 2025 test rule

> **Never tune, iterate, or select features using 2025 results.**

This is enforced in code, not just documentation:
`mlb_luck_score.config.assert_seasons_allowed()` raises `ProtectedSeasonError` whenever
2025 appears in a season list, unless the caller explicitly passes
`allow_final_evaluation=True` (CLI: `--allow-final-evaluation`). The downloader and the
trainer both call this guard. Do not add a code path that uses 2025 without this
explicit, one-time, intentional flag.

## Preliminary raw-luck definition

```
expected_value = sum(predicted_probability[outcome] * value[outcome])
actual_value    = value[observed_outcome]
raw_luck        = actual_value - expected_value
```

using the Version 0.1 ordinal value map `{out: 0, single: 1, double: 2, triple: 3,
home_run: 4}` (`mlb_luck_score.config.DEFAULT_VALUE_MAP`, configurable). This is a
research placeholder: it ignores base/out state, park, and win-expectancy context, and
the ordinal spacing (0/1/2/3/4) is not a validated run-value model. Raw luck IS additive
by construction (see `mlb_luck_score.scoring.raw_luck`).

## Why the public score is not additive

The public -100..+100 score (`mlb_luck_score.scoring.public_score`) applies a bounded,
monotonic, sign-preserving transform (`100 * tanh(raw_luck / scale)`) to raw luck so it
fits a stable, intuitive display range. That transform is non-linear, so summing public
scores across plays does **not** equal the public score of the summed raw luck -- if you
need an additive quantity (e.g. season aggregation), use raw luck, not the public score.
The current mapping is an explicitly provisional placeholder; the final version should be
fit empirically against a historical held-out distribution of raw-luck values (see
`EmpiricalScoreCalibrator`, an unimplemented extension point).

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
`pybaseball`. This repository does not redistribute any downloaded data (see
`.gitignore`); downloaded files stay local under `data/raw/`, `data/interim/`, and
`data/processed/`, all git-ignored. Review MLB/Baseball Savant's terms before
redistributing any derived data outside this repository.

## Reproducibility

- Deterministic random seeds are set where applicable (`RANDOM_SEED = 42` in
  `train_contact_model.py`).
- Every saved model artifact is accompanied by a metadata JSON recording model version,
  training seasons, feature list, class order, training timestamp, and package versions.
- `event_id` is built deterministically from stable identifiers
  (`game_pk-at_bat_number-pitch_number`), not a random hash.

## Current status

Version 0.1 bootstrap: data acquisition, cleaning, feature engineering, a baseline
model, calibration diagnostics, preliminary raw luck, a placeholder public score, and a
data-completeness report are all implemented and covered by offline synthetic tests. No
real predictive-performance claims have been validated against a real held-out sample as
part of this bootstrap -- run the pipeline against real data and inspect the actual
evaluation metrics before drawing any conclusion.

## Future work

- Weather normalization and air-density modeling
- Park geometry effects
- Exact defender positioning, reaction, and route modeling
- Fielding and throwing execution modeling
- Batter-runner decision-quality modeling
- Batter-runner execution modeling
- Park and bounce effects
- Shapley-style attribution across luck components
- Confidence intervals (historical support, model disagreement, sensitivity analysis,
  calibration uncertainty, bootstrap/posterior intervals)
- Season-level aggregation
- Leaderboard qualification thresholds
- Rare-play rule expansion (the eligible-event list is a v0.1 starting point, not final)

---

See `CLAUDE.md` / `AGENTS.md` for persistent rules that govern how this repository should
be extended by AI coding agents.
