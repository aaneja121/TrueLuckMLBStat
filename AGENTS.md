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
`train_model`'s default feature set does not include `venue_id`. This pattern generalizes:
any future comparison variant (park factors, weather, defensive positioning, etc.) stays a
candidate, reported with its exact metrics via `recommend_*`-style rule-based logic, until
a maintainer explicitly decides to adopt it. A rule-based recommendation is a starting
point for judgment (read the full by-venue/by-subgroup table yourself -- a real,
noteworthy regression can exist below a conservative automated threshold), never a
substitute for it. If a candidate is adopted, give it a distinct `scoring_version` in its
`ReferenceScoreArtifact` (see `mlb_luck_score.models.build_reference_score`) rather than
overwriting an earlier version's artifact file.

## Never commit datasets, secrets, virtual environments, or model artifacts

`.gitignore` already excludes `.venv/`, `.env`, `data/raw/*`, `data/interim/*`,
`data/processed/*`, `artifacts/*`, and `outputs/{figures,tables}/*` (keeping only
`.gitkeep` placeholders). Before committing, run `git status` and double-check nothing
under those paths, and nothing that looks like an API key or credential, is staged.

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
