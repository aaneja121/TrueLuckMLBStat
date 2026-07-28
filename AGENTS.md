# AGENTS.md -- persistent rules for AI coding agents in this repository

This file governs how any AI coding agent (Claude Code, other CLI agents, IDE
assistants) should work in **Contact Luck Prototype v0.1**. Read `README.md` first
for project scope and terminology; this file is about *how to work here safely*, not
*what the project is*. (`CLAUDE.md` contains the same rules for Claude Code
specifically -- keep the two in sync if you update either.)

## The single most important rule

> **Never tune, iterate, or select features using 2025 results.**

2025 is the untouched final-test season (see `mlb_luck_score.config.FINAL_TEST_SEASONS`
and `assert_seasons_allowed`). It is enforced in code -- any command that touches a
season list must call `assert_seasons_allowed(...)` and must not pass
`allow_final_evaluation=True` except for a genuine, intentional, one-time final
evaluation the user has explicitly asked for. Do not add a new code path (notebook
cell, script, ad hoc analysis) that reads 2025 data during normal development. If
unsure whether something counts as "tuning" on 2025, treat it as tuning and ask first.

This applies equally to `mlb_luck_score.data.download_development_data` and
`mlb_luck_score.data.clean_development_data` (the full 2021-2024 dataset workflow): both
call `assert_seasons_allowed`, and the downloader has no configured date range for 2025
in `mlb_luck_score.config.MLB_REGULAR_SEASON_DATE_RANGES` as a second layer of
protection. If you ever add a new season to that dict or to `DEVELOPMENT_SEASONS`,
2025 must never be one of them.

## Avoid target leakage

Never use `events`, `outcome_class`, `description`, `estimated_ba_using_speedangle`,
`estimated_woba_using_speedangle`, `woba_value`, `delta_home_win_exp`, or any other
column computed from or encoding the play's result as a model input feature. The
enforced list lives in `mlb_luck_score.config.LEAKAGE_COLUMNS`; the check is
`mlb_luck_score.features.build_contact_features.assert_no_leakage`. If you add a new
feature, check whether it's post-outcome before wiring it in, and prefer running the
leakage check over trusting your own judgment.

## Keep eligibility logic centralized

All fair-batted-ball eligibility rules, outcome-class mapping, and training-exclusion
reasons live in `mlb_luck_score/eligibility.py` and nowhere else. Do not duplicate the
eligible-event list, the outcome mapping, or the ambiguous-event handling in another
module, script, or notebook -- import from `eligibility.py`. If the rules need to
change, change them there and update its docstrings and `README.md`/this file together.

## Never silently redefine the Luck Score

The current (Version 0.2) default is `compute_raw_contact_luck_runs`
(`mlb_luck_score.scoring.contact_luck`, using the fixed run-value table in
`mlb_luck_score.scoring.run_values.DEFAULT_RUN_VALUE_MAP`) and
`compute_empirical_public_score` (`mlb_luck_score.scoring.empirical_score`, using a
`ReferenceScoreArtifact` built by `mlb_luck_score.models.build_reference_score`). The
Version 0.1 ordinal functions (`mlb_luck_score.scoring.raw_luck.compute_raw_luck`,
`mlb_luck_score.scoring.public_score.raw_luck_to_public_score`) are LEGACY -- kept only
for backward compatibility and explicit Version-0.1-vs-0.2 comparison, never as the
default for new work. All of these have exact, documented formulas. Do not change a
formula, the default run-value/value map, or the class ordering without: (1) updating
every docstring that states the formula, (2) updating the corresponding tests
(`tests/test_contact_luck.py`, `tests/test_run_values.py`, `tests/test_empirical_score.py`,
`tests/test_scoring.py`), and (3) calling out the change clearly to the user as a
redefinition, not a bug fix. Silent redefinition breaks comparability across any results
already produced. If you ever add a new run-value source (e.g. a different season range
or a context-aware model), version it explicitly (`scoring_version` in
`ReferenceScoreArtifact`) rather than overwriting `DEFAULT_RUN_VALUE_MAP` in place.

## Never use class_weight="balanced" (or similar) for the probability baseline

`train_model`'s default `class_weight=None` is load-bearing, not arbitrary. It was
changed FROM `class_weight="balanced"` after that setting was confirmed (via a controlled
comparison on real 2021-2024 data, isolating `class_weight` as the only variable changed)
to cause severe probability miscalibration -- e.g. rows called ~54% likely to be a triple
were observed to be one ~3.6% of the time. See "Model comparison and probability
calibration" in README.md and `mlb_luck_score.models.compare_models` for the exact numbers
and methodology. `class_weight="balanced"` is preserved ONLY as the explicitly-labeled
`VARIANT_CLASS_BALANCED` comparison model (`mlb_luck_score.models.train_contact_model`) --
never use its output, or any other class-reweighting/oversampling scheme, to produce
probabilities for the Contact Luck score, no matter how good its accuracy or per-class
recall looks. Recall/accuracy and calibration are different questions -- a model can
excel at one while failing the other. If you ever change the default `class_weight` or
add a new reweighting scheme, verify calibration with
`mlb_luck_score.models.compare_models.run_comparison` first and report the actual ECE/log
loss numbers, not just accuracy.

## Do not compute or publish Luck Scores against real data with a miscalibrated model

`mlb_luck_score.scoring.raw_luck` requires well-calibrated probabilities to mean anything
(`expected_value = sum(p(outcome) * value(outcome))`) -- a model that ranks outcomes
correctly but assigns systematically wrong probabilities produces biased raw-luck values
even though its classification metrics look fine. Before computing real (non-synthetic)
raw-luck or public-score values, check the current model's calibration via `make
compare-models` / notebook 03 and report the actual numbers. If a change makes
calibration meaningfully worse, do not fold it into the default without calling that out
explicitly.

## Never silently adopt a candidate model variant as the default

`park_aware_v03_candidate` (`mlb_luck_score.models.compare_park_aware`, adds `venue_id` to
the Version 0.2 `baseline_v02` features) is a documented CANDIDATE, not the default --
`train_model`'s default feature set does not include `venue_id`. The same is true of the
Version 0.4 park-geometry candidates (`geometry_only_v04_candidate` and
`venue_plus_geometry_v04_candidate`, `mlb_luck_score.models.compare_geometry_aware`):
despite a large, bootstrap-confirmed log-loss and near-wall-calibration improvement on real
2021-2024 data, `recommend_geometry_adoption` reports `recommend_adopt_any_v04_candidate:
False` because a reliably-sampled venue (loanDepot park) shows a material calibration
regression -- see "Park geometry (Version 0.4)" in README.md for the full numbers. The
Version 0.5 weather candidates (`weather_basic_v05_candidate` and `weather_vector_v05_
candidate`, `mlb_luck_score.models.compare_weather_aware`) are an even sharper illustration of
why: they pass ALL 8 automatable adoption criteria (`recommend_adopt_any_v05_candidate: True`)
with a real, bootstrap-confirmed (if tiny) log-loss improvement, but a direct physical
-plausibility check -- the one criterion the rule deliberately never automates -- found the
per-play weather attribution weak and partly wrong-signed (see "Weather and air density
(Version 0.5)" in README.md), so neither has been adopted either. `baseline_v02` remains the
default. This pattern generalizes: any future comparison variant (more park factors, defensive
positioning, etc.) stays a candidate, reported with
its exact metrics via `recommend_*`-style rule-based logic, until a maintainer explicitly
decides to adopt it. A rule-based recommendation is a starting point for judgment (read the
full by-venue/by-subgroup table yourself -- a real, noteworthy regression can exist below a
conservative automated threshold, and criteria that are inherently a judgment call, like
"is the model learning physically plausible effects", are deliberately NOT automated at
all), never a substitute for it. If a candidate is adopted, give it a distinct
`scoring_version` in its `ReferenceScoreArtifact` (see `mlb_luck_score.models.
build_reference_score`) rather than overwriting an earlier version's artifact file.

## Never commit datasets, secrets, virtual environments, or model artifacts

`.gitignore` already excludes `.venv/`, `.env`, `data/raw/*`, `data/interim/*`,
`data/processed/*`, `artifacts/*`, and `outputs/{figures,tables}/*` (keeping only
`.gitkeep` placeholders). Before committing, run `git status` and double-check nothing
under those paths, and nothing that looks like an API key or credential, is staged.
`mlb_luck_score/data/game_metadata_overrides.py` (like `mlb_luck_score/scoring/
run_values.py` and `mlb_luck_score/data/park_geometry.py`) is a deliberate exception: it is
small, hand-reviewed reference data written as source code, not a downloaded dataset, so it
IS tracked in git. Every entry in it must cite a documented, verifiable `source_note` --
never add or edit an entry without one.

## Park-geometry reference data must document provenance and review status

Every record in `mlb_luck_score.data.park_geometry.PARK_GEOMETRY_POINTS` must have a
non-empty `source_name`/`source_reference`/`source_accessed_date` (enforced at import time
by `validate_geometry_points`) and an honest `review_status`. Every record added so far by
an AI coding agent uses `REVIEW_STATUS_AGENT_SOURCED` ("agent_sourced_pending_human_
review") -- do not upgrade a record's `review_status` to imply human review has happened
unless a human maintainer actually did it. Never fabricate a wall distance, height, or
configuration date -- if a source is ambiguous or conflicting (e.g. a park's exact pre-
renovation dimensions can't be pinned down), document the ambiguity in `notes` and either
use the best-supported figure with a caveat or leave the venue/era without geometry
(`geometry_status` will correctly report it as unavailable downstream -- see
`mlb_luck_score.data.join_park_geometry`) rather than guessing. Never add geometry for a
venue in `TEMPORARY_OR_SPECIAL_VENUE_IDS` (temporary/neutral-site venues, including the
Field of Dreams sentinel `venue_id=-1`) without the same level of verified, cited evidence
required for any other venue -- the current default is to leave them without geometry, and
that is a deliberate choice, not a gap to casually fill in.

## Never fabricate weather, roof status, or indoor climate conditions

`mlb_luck_score.data.venue_environment.VENUE_ENVIRONMENTS` (station/elevation/timezone
reference data) follows the same provenance rules as park geometry above -- every record
needs a `source_note`, and weather-station coordinates/elevation must be independently
verified against the actual data source (see that module's docstring), not guessed.
Historical weather itself (`mlb_luck_score.data.download_historical_weather`) comes from
exactly two sources -- the MLB Stats API's own per-game `weather` field and the public Iowa
Environmental Mesonet ASOS archive -- and nothing else; never substitute current/live
weather for a historical game, and never invent an hourly observation to fill a gap.
`mlb_luck_score.data.build_game_weather.classify_roof_status` never infers a closed roof
from precipitation or any other condition text -- only an explicit `"Roof Closed"` string
does that; everything else defaults to `roof_status_unknown` rather than a guess. For
`retractable_roof_closed`/`fixed_indoor`/`roof_status_unknown` games, `effective_
temperature_c`/`effective_relative_humidity_pct`/`effective_pressure_hpa`/
`air_density_kg_m3` MUST stay null (no reviewed indoor-climate assumption exists in this
repository) -- only `effective_wind_speed_mps=0.0` is set, because a closed roof physically
blocking outdoor wind is a certainty, not an assumption. Do not add an indoor-climate
default (e.g. "72F, 50% humidity") without first adding genuinely reviewed, cited
per-venue indoor climate-control data -- until then, leave it unavailable.

## Weather attribution must stay separate from raw contact luck

`mlb_luck_score.scoring.weather_attribution.compute_weather_attribution`
(`weather_run_value_effect = expected_run_value_actual_environment -
expected_run_value_standard_environment`) is a DIFFERENT quantity from `mlb_luck_score.
scoring.contact_luck.compute_raw_contact_luck_runs` (`raw_contact_luck_runs = actual_run_
value - expected_run_value`, using the selected baseline model's own prediction). Never add
them together -- see `weather_attribution`'s module docstring for exactly why that would
double-count weather. Building and testing the weather-attribution module does not itself
constitute using it for real scores; per the task's "only after a weather-aware candidate
passes validation" rule, treat its output as informational infrastructure, not a validated
headline number, until a maintainer has actually adopted a weather-aware candidate.

## When generating counterfactual/standardized feature rows, only use categories the model has seen

`mlb_luck_score.features.build_contact_features.generate_standardized_environment_rows` was
verified to produce physically backwards predictions (Coors Field's thin actual air scoring
WORSE than a denser standardized reference) when an earlier version set
`weather_match_quality` to a synthetic `"standardized"` label the model never saw in
training -- `OneHotEncoder(handle_unknown="ignore")` silently encodes any unseen category as
all zeros, a pattern the fitted model never learned to interpret, rather than raising an
error. Any function that builds a counterfactual/synthetic feature row for prediction must
only assign categorical values that genuinely occur in real training data (see that
function's fix and `tests/test_compare_weather_aware.py::
test_standardized_environment_categorical_overrides_are_realistic_categories` for the
regression test). This class of bug is easy to miss because it fails silently -- no
exception, no NaN, just a quietly wrong prediction -- so treat any new synthetic/
counterfactual row generator with the same suspicion and verify its categorical overrides
against real data before trusting its output.

## Confidence must never dampen the score

The Version 0.1 confidence report (`mlb_luck_score.scoring.confidence`) is descriptive
data-completeness reporting only. Do not use it to pull `raw_luck` or the public score
toward zero, and do not present it as a statistical uncertainty estimate -- it isn't
one.

## Use conservative scientific language

Do not describe the model as "accurate," "validated," or "calibrated" without pointing
to the specific evaluation numbers that support the claim, computed on real held-out
(non-2025) data. Running calibration or evaluation code successfully is not evidence
that the results are good -- report the actual numbers and let them speak. Prefer
"provisional," "preliminary," and "Version 0.1 research placeholder" over stronger
language, matching the existing docstrings.

## Document assumptions and provisional choices

Every provisional constant (the ordinal value map, the public-score scale, the
spray-angle formula, the eligible-event list) already has a docstring explaining it is
a Version 0.1 choice, not a validated result. When you add a new one, do the same --
future readers (human or agent) should be able to tell a documented placeholder from a
validated result at a glance.

## Prefer small, reviewable changes

This is a research prototype under active iteration. Prefer focused diffs over broad
refactors; don't restructure multiple modules in one change unless asked.

## Inspect existing work before editing

Before modifying a module, read it (and its tests) in full. Before running any git
operation, run `git status` first -- do not assume the working tree is clean.

## Avoid destructive Git operations

Never run `git push --force`, `git reset --hard`, `git checkout -- <path>` /
`git restore` over uncommitted work, `git clean -f`, or `git branch -D` unless the user
explicitly asks for that specific action in that specific moment. Prefer creating a new
commit over amending an existing one.

## Ask before pushing, publishing, or changing external resources

Never `git push`, never create or modify a git remote, never create cloud resources,
and never touch CI/CD configuration without the user explicitly asking first in that
conversation. Downloading Statcast data requires internet access -- that's expected and
fine, but always say so before running a download. The full development dataset
download (`make download-development-data`, all four 2021-2024 seasons) is large and
can take a long time (see README.md "Full development dataset" for storage/runtime
estimates) -- implement and test that workflow with synthetic data first, then show the
user the exact command and its storage/runtime considerations, and get explicit
approval before actually running it against the network.

## Before finishing any change

Run, in this order, and fix failures before reporting done (or explain clearly why you
couldn't):

```bash
.venv/bin/python -m ruff format src tests   # or: make format
.venv/bin/python -m ruff check src tests    # or: make lint
.venv/bin/python -m mypy src                # or: make typecheck
.venv/bin/python -m pytest                  # or: make test
```

All four are bundled in `make check`. The test suite must remain fully offline --
never add a test that requires network access or real Statcast data.
