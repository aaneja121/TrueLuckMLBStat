# CONTEXT.md -- domain and project terminology

Terms a future agent needs to read this repository, its data, and its UI copy correctly.
Definitions only — no model methodology (see `RESEARCH_RULES.md` and `README.md` for
that), no product intent (see `PRODUCT.md`).

## The metric

**Contact Luck** — The run-value gap between what a fair batted ball was *expected* to
produce and what *actually* happened, summed across a player's eligible batted balls.
Retrospective: it describes results, not talent or future performance.

**Contact Luck Runs** (`total_contact_luck_runs`) — The additive season total, in runs.
Exactly `observed run value − expected run value`.

**Contact Luck Runs per 100** (`contact_luck_runs_per_100`) — **The official public
metric.** Contact Luck Runs normalized to 100 eligible batted balls. Displayed as
"Runs / 100".

**Positive / negative** — Positive = more favorable realized outcomes than expected.
Negative = less favorable. Never phrased as good/bad hitting, deserved, or owed.

**Expected run value (Expected RV)** — What the frozen model predicted the batted ball
was worth, from the contact's own physical characteristics.

**Observed / actual run value** — The run value of what actually happened on the play.

**Run value** — Runs above/below average credited to an outcome, from a fixed run-value
table (`DEFAULT_RUN_VALUE_MAP`). Not recomputed anywhere in the dashboard.

**Pitching Contact Luck** — The same quantity re-grouped by the pitcher who allowed the
batted ball, with the sign flipped: `pitching_contact_luck = −1 × batting_contact_luck`.
Positive = outcomes more favorable to the **pitcher** than the contact predicted. Trains
no new model — it re-aggregates the same frozen ledger. **Not scored for 2025 or 2026 and
not in the public-score schema.** The **2024** season is published at `/pitchers/2024/`
from the committed fixture; `scripts/publish_snapshot.sh` passes
`--pitcher-season-fixture` explicitly, and `dashboard_config.PITCHER_PUBLIC_SEASONS`
gates which seasons may ever be published (2024 only — see RESEARCH_RULES.md "Public
launch of the 2024 pitcher surface"). See README "Pitching Contact
Luck (Version 0.13, research spike)" and "Pitching Contact Luck presentation research
(Version 0.13.1)".

**Starter-only scope** — Pitching Contact Luck's `pitcher_primary`
threshold set (≥450 eligible BBE) describes **starting pitchers**. No reliever-season in
2021–2024 reached it (highest with ≥50 appearances: 311 BBE). Relievers are excluded by
exposure, not by choice, and any surface showing this metric must say so. Discharged on
the published board by `public_labels.PITCHER_EXPOSURE_SCOPE`, which also states that the
boards rank on the 60-BBE display minimum rather than on that threshold set.

**Cumulative Contact Luck allowed** — The pitcher surface's PRIMARY quantity (Version
0.13.1): the sum of the per-play contributions over the batted balls a pitcher actually
allowed, in runs. Exact for the observed plays, conditional on the frozen scoring model.
Deliberately **workload-sensitive** — it answers an accumulation question, so the
batted-ball count is always shown beside it. Never skill, talent, quality, or a forecast.
Per 100 BBE is the SECONDARY quantity and is never the ranking key, because pitcher rates
have resolving power below 1 at every achievable workload. See README "Pitching Contact
Luck presentation research (Version 0.13.1)".

**Starter-like / Reliever-like** — DESCRIPTIVE usage buckets from batted balls per
appearance (≥10 / ≤8; 8–10 is ambiguous and belongs to neither board). Read off the
observed bimodal distribution. This repository has **no authoritative role metadata**: no
pitcher may be labelled a starter, a reliever, or a closer. Factual usage ("72 appearances,
2.6 BBE per appearance") is fine; inferring a role from it is not.

**Board display minimum** — The pitcher surface leaves pitcher-seasons under 60 resolved
BBE off its ranked boards. A **presentation rule only**, motivated by normalized-rate
interval width — never qualification, eligibility, an official minimum, or an MLB rule.
Those seasons keep a page, a total, a rate and an interval, and never read as having failed
anything. Distinct from `qualification_status`, which this surface does not use.

## Baseball / data terms

**Batted ball / ball in play** — A pitch put in fair play. Strikeouts, walks, HBP and
similar are never scored.

**Eligible batted ball (BBE)** — A fair batted ball that passes this project's
eligibility rules (`mlb_luck_score.eligibility`) and has the required contact inputs. The
denominator for "per 100". Strictly a subset of plate appearances.

**Outcome class** — The five-way target the contact model predicts:
`out | single | double | triple | home_run`.

**Recorded result** — The officially recorded result of the batted ball (what the box
score says), shown on the play page. Distinct from the run-value accounting below it.

**Exit velocity (EV)** — Speed of the ball off the bat, mph.

**Launch angle (LA)** — Vertical angle off the bat, degrees.

**Batted-ball type (`bb_type`)** — ground_ball / line_drive / fly_ball / popup.

**Spray angle (approximate)** — Horizontal direction, derived from Statcast's `hc_x`/
`hc_y` visualization coordinates via a community approximation. Directional signal, not a
verified measurement — never present it as exact.

**Statcast** — MLB's public tracking data, retrieved via `pybaseball`. The only source of
contact measurements here.

**`batter_id`** — MLBAM player id. The stable key for player pages, player JSON shards,
and headshot lookups.

**`pitcher`** — MLBAM player id of the pitcher who allowed the batted ball. Carried
through cleaning and feature-building as an identifier and **never a model feature**.
Unlike `batter_name`, pitcher names need no join: raw Statcast's `player_name` column is
the *pitcher's* name.

**`play_id`** — `<game_pk>-<at_bat_number>-<pitch_number>`. Stable per-play identity; the
play page derives `game_pk` from it rather than needing an index.

**`game_pk`** — MLB's game identifier; also the shard key for per-game play JSON.

**Scored Games** — Games with at least one outcome-resolved eligible batted ball. **Not**
official MLB games played, and the UI must keep saying so.

## Score presentation

**95% interval** (`lower_95_interval` / `upper_95_interval`) — Uncertainty band on the
same per-100 scale as the point estimate, from a **game-clustered** bootstrap (games
resampled, not individual plays). Always shown with the score.

**`interval_interpretation`** — `entirely_above_zero` / `overlaps_zero` /
`entirely_below_zero`. Purely descriptive. Never a significance claim, and never a reason
to suppress, fade, or de-emphasize a row.

**Qualification** (`qualification_status`) — `qualified` vs. `small_sample` etc., from the
`"primary"` threshold set (≥200 eligible BBE, ≥100 games, plus component-coverage,
provisional-share, and missing-input limits). Only `qualified` rows carry an official
rank; non-qualified players keep their own page. Pitching Contact Luck uses its own
sets (`pitcher_primary`, `pitcher_inclusive`) because the batter set's ≥100-games bar
disqualifies every pitcher who has ever thrown a pitch.

**Qualified-population reference point** (`qualified_population_reference`) — The mean
score among *qualified* players. **Zero is not the neutral point of a qualified board:**
zero means a batted ball matched its expectation, not that a player matched his peers.
Clearing an exposure bar selects for favorable realized outcomes, which shifts the
qualified mean off zero (2024 pitchers: +0.54 runs/100; the same conditioning shifts the
batter mean too). Recorded alongside the score for reading a row in context — **the score
is never recentred by it** (`score_is_recentered` is always `False`). Currently reported
on the pitcher side only.

**Official rank** — Competition ranking (`method="min"`: ties share a rank, the next
distinct value skips ahead) on `contact_luck_runs_per_100` alone. Interval endpoints
never reorder anyone. Tie-break for display: `batter_id` ascending.

**Leaderboards** — Exactly two, both frozen in `public_labels`:
"**Most favorable realized luck**" and "**Least favorable outcomes relative to
expectation**". Never "worst players". No leaderboard exists for any individual component.

**Sorted view** — A user-re-sorted table. Must be labeled as such and never presented as
an alternative official ranking; the Rank column restores the frozen order.

**Component decomposition** — The additive split of a player's score across *contact*,
*unexplained residual*, *defensive execution*, and *advancement*. Displayable with values
and status codes; never ranked.

**Component status** — `calibrated`, `calibrated_with_limited_subgroup_evidence`,
`provisional`, `not_calibrated`, `unavailable`. A displayed component value must always
carry its status.

**Provisional share** (`provisional_share`) — Fraction of a player's score coming from
components that are not yet fully calibrated. Surfaced explicitly, never hidden.

**Banned phrases** — Enforced in `public_labels.BANNED_PHRASES`: "deserved hits", "true
talent luck", "guaranteed regression", "should have produced", "defense-independent",
"statistically significant player". Applies to all user-facing copy.

## Pipeline / build terms

**Snapshot** — An immutable, dated scoring run for a `--data-through` date, living in
`outputs/prospective/v1_1/<date>/` + `artifacts/prospective/v1_1/<date>/`. Never
overwritten.

**`data_through_date`** — The last fully completed MLB slate included in a snapshot. The
"Data through …" badge in the site header.

**Snapshot type** — `genuine_prospective_snapshot`, `corrected_prospective_snapshot`,
`retrospective_backfill` (reserved; none exist), or `invalid_incomplete_snapshot`.
Precedence rules live in `dashboard/snapshot_data.py`.

**Frozen model** — The Version 1.0 source code plus fixed `TRAIN_SEASONS` (2021–2023),
re-run deterministically. No trained model object is persisted between runs.

**Fixture** — Committed, hand-reviewed sample data the dashboard build reads instead of
live scoring: `dashboard/demo_fixture.json`, `dashboard/demo_counterfactual_grid.json`,
`dashboard/explore_fixture/`. Fixtures are *development* Explorer data — production
builds pass real snapshot artifacts instead.

**Showcase Plays** — A curated set of the biggest favorable/unfavorable individual breaks,
selected offline; some carry a precomputed sensitivity grid for the play page's "What
if?" sliders.

## Season roles (never blur these)

| Seasons | Role |
|---|---|
| 2021–2023 | Model fitting (`TRAIN_SEASONS`) |
| 2024 | Development validation / model selection — **iterated against, not pristine** |
| 2025 | Sealed one-time final evaluation — **never development or product data** |
| 2026 | Prospective scoring — what the live site displays |
