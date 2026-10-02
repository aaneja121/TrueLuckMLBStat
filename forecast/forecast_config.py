"""Contact Forecast R1: season policy, window grid, paths, and hard season guards.

Contact Forecast is a FORWARD-LOOKING research layer. It is conceptually and
operationally separate from Contact Luck:

  - **Contact Luck** (frozen, `src/mlb_luck_score/`) is a DESCRIPTIVE
    attribution of what happened versus what the contact deserved.
  - **Contact Forecast** (this package) asks whether a hitter's
    luck-adjusted performance through a cutoff predicts his SUBSEQUENT
    performance better than his realized performance through that same
    cutoff.

Nothing in this package modifies, retrains, recalibrates, or redefines the
Contact Luck metric, its run-value table, its thresholds, or any published
output. It produces no public score and touches no dashboard surface.

## Why this lives outside `src/mlb_luck_score/`

Same rationale as `evaluation/` and `prospective/`: a plain script directory,
not part of the installed `mlb_luck_score` package, reachable from tests via
`pyproject.toml`'s `pythonpath`. Keeping it out of the package is what makes
"the forecast layer cannot be imported by the frozen scoring pipeline" true
by construction rather than by convention.

## Season policy (decided by the maintainer, 2026-09-03)

| Season | Role in Contact Forecast |
|---|---|
| 2021 | FEATURE SOURCE ONLY -- supplies prior-season features for 2022 windows and trains the 2022 contact model. Never an analysis season (it has no prior season to train a strictly-causal model on). |
| 2022-2024 | Analysis seasons (`FORECAST_ANALYSIS_SEASONS`). |
| 2025 | SEALED. Never read by this package, under any flag, ever. |
| 2026 | PHASE 2 ONLY. Unreachable from Phase 1 code -- see `PHASE_2_GATE` below. |

## Walk-forward scoring is strictly causal

`WALK_FORWARD_TRAIN_SEASONS` maps each analysis season to the development
seasons the contact model may be fit on to score it: strictly earlier
seasons, never the scored season itself and never a later one. 2024's entry
`(2021, 2022, 2023)` is IDENTICAL to production's frozen
`mlb_luck_score.config.TRAIN_SEASONS`, so the 2024 slice of this study is
scored by exactly the production configuration.

This is the frozen contact-model SOURCE CODE re-fit under a causal training
window -- not a redefinition of the metric, and not a new model. It does
produce expected-run-value numbers that differ from production's for
2022-2023, which is precisely why every artifact this package writes lands
in `FORECAST_OUTPUTS_DIR`/`FORECAST_DATA_DIR` and never in `outputs/tables`,
`data/processed`, or any prospective/final-evaluation namespace.

## Known, deliberately unfixed leakage vector

`mlb_luck_score.scoring.run_values.DEFAULT_RUN_VALUE_MAP` is averaged over
2021-2024, so it embeds information from seasons later than a 2022 window.
It is NOT corrected here: it is a fixed, context-neutral constant applied
identically to every hitter and to every predictor being compared, so it
cannot manufacture a spurious cross-hitter signal, and changing it would be
a redefinition of the score (RESEARCH_RULES.md "Never silently redefine the
Luck Score"). It is recorded in the leakage audit instead.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Season policy
# --------------------------------------------------------------------------

#: Seasons this package may build analysis windows for.
FORECAST_ANALYSIS_SEASONS: tuple[int, ...] = (2022, 2023, 2024)

#: Supplies prior-season features and the 2022 contact-model training set.
#: Never an analysis season -- no strictly-earlier season exists to fit a
#: causal contact model on, so its "deserved" values could only ever be
#: in-sample.
FORECAST_FEATURE_SOURCE_SEASONS: tuple[int, ...] = (2021,)

#: Every development season this package reads at all.
FORECAST_DEVELOPMENT_SEASONS: tuple[int, ...] = (
    FORECAST_FEATURE_SOURCE_SEASONS + FORECAST_ANALYSIS_SEASONS
)

#: Permanently sealed. `mlb_luck_score.config.FINAL_TEST_SEASONS` already
#: refuses this season repository-wide; this package refuses it a second
#: time, with no override parameter of any kind.
FORECAST_SEALED_SEASONS: tuple[int, ...] = (2025,)

#: Reserved for Phase 2 and unreachable from Phase 1. See `PHASE_2_GATE`.
FORECAST_PHASE_2_SEASONS: tuple[int, ...] = (2026,)

#: Analysis season -> the development seasons its contact model may be fit
#: on. Strictly earlier seasons only. The 2024 entry equals production's
#: frozen `TRAIN_SEASONS`.
WALK_FORWARD_TRAIN_SEASONS: dict[int, tuple[int, ...]] = {
    2022: (2021,),
    2023: (2021, 2022),
    2024: (2021, 2022, 2023),
}

#: Season-forward folds for every fitted quantity in this study -- ridge
#: alpha, the matched-shrinkage parameters, and the rescaling/calibration
#: transforms. `(fit_seasons, evaluate_season)`. Never a random row split:
#: the prediction problem is temporal, so the validation design is too.
SEASON_FORWARD_FOLDS: tuple[tuple[tuple[int, ...], int], ...] = (
    ((2022,), 2023),
    ((2022, 2023), 2024),
)

# --------------------------------------------------------------------------
# Window grid
# --------------------------------------------------------------------------

#: Observation cutoffs, in eligible batted balls from the hitter's first of
#: the season. Features may use events 1..K only.
FORECAST_CUTOFFS: tuple[int, ...] = (50, 100, 150, 200)

#: Forward target horizons, in eligible batted balls after the cutoff.
FORECAST_HORIZONS: tuple[int, ...] = (50, 100)

#: A hitter-season enters a (cutoff, horizon) cell only when it has at least
#: `cutoff + horizon` eligible batted balls. This conditions on future
#: playing time; both compared predictors share the condition, so it does
#: not bias the comparison, but it does bound what the results generalize
#: to. `forecast.windows.summarize_inclusion` reports the exclusion rate.
MIN_BBE_FOR_WINDOW = min(FORECAST_CUTOFFS) + min(FORECAST_HORIZONS)

#: Rate denominator for every reported run-value figure: runs per 100
#: eligible batted balls, matching `mlb_luck_score.scoring.aggregation.
#: raw_contact_luck_runs_per_100_eligible_events`.
RATE_SCALE = 100.0

#: Strict within-hitter-season chronological order for eligible batted
#: balls. `game_date` orders days; `game_pk` breaks ties on the 2,479
#: observed batter-days that contain a doubleheader (MLB assigns
#: doubleheader game_pks in schedule order -- a DOCUMENTED ASSUMPTION, not a
#: verified fact, carried in the leakage audit and probed by
#: `forecast.windows.count_doubleheader_boundary_windows`); `at_bat_number`
#: and `pitch_number` order within a game.
CHRONOLOGICAL_SORT_KEY: tuple[str, ...] = (
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
)

#: Resampling unit for every bootstrap in this study. `batter`, NOT
#: `(batter, season)`: the same hitter recurs across seasons and populates
#: multiple overlapping (cutoff, horizon) cells, so his windows are not
#: independent draws. Resampling a batter takes ALL of his hitter-seasons
#: and ALL of their windows together.
BOOTSTRAP_CLUSTER_COLUMN = "batter"

DEFAULT_BOOTSTRAP_REPS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260903
DEFAULT_BOOTSTRAP_ALPHA = 0.05
RANDOM_SEED = 20260903

# --------------------------------------------------------------------------
# Isolated namespaces
# --------------------------------------------------------------------------

FORECAST_DATA_DIR = PROJECT_ROOT / "data" / "forecast"
FORECAST_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "forecast_research"
FORECAST_REPORT_DIR = PROJECT_ROOT / "docs" / "forecast"

#: The frozen Phase 1 specification. Written once, hashed, and thereafter
#: read-only. See `PHASE_2_GATE`.
FORECAST_SPEC_PATH = FORECAST_OUTPUTS_DIR / "forecast_spec.json"

#: Namespaces this package must never read from. The first three are the
#: sealed Version 1.0 (2025) roots; the last three are the Version 1.1
#: prospective (2026) roots, which are Phase 2 territory.
FORBIDDEN_NAMESPACE_ROOTS: tuple[Path, ...] = (
    PROJECT_ROOT / "data" / "final_evaluation",
    PROJECT_ROOT / "outputs" / "final_evaluation",
    PROJECT_ROOT / "artifacts" / "final_evaluation",
    PROJECT_ROOT / "data" / "prospective",
    PROJECT_ROOT / "outputs" / "prospective",
    PROJECT_ROOT / "artifacts" / "prospective",
)

# --------------------------------------------------------------------------
# The Phase 2 gate
# --------------------------------------------------------------------------

#: What must be true, in this exact order, before any 2026 path may be read.
#: This gate is IMMUTABLE: once Phase 1 freezes the specification, none of
#: the frozen items may be altered in response to a 2026 result. A 2026
#: number can only ever confirm or fail to confirm the frozen spec -- it can
#: never revise it. Re-tuning after the gate opens is not a correction, it is
#: a new study that needs its own untouched season.
PHASE_2_GATE: tuple[str, ...] = (
    "forecast_spec.json exists, is complete, and its content hash is recorded",
    "frozen: feature definitions (names, pre-cutoff construction, imputation)",
    "frozen: target definitions (Target A realized, Target B deserved)",
    "frozen: model class and the hyperparameter TUNING PROCEDURE (not just its result)",
    "frozen: matched-shrinkage estimator and how its parameters are fit",
    "frozen: every rescaling/calibration transform and its walk-forward fitting rule",
    "frozen: the evaluation metric set and the pre-registered decision rule",
    "frozen: exclusion rules (eligibility, MIN_BBE_FOR_WINDOW, missing-data handling)",
    "frozen: the report template, including the sections that report negatives",
    "the maintainer has explicitly authorized opening 2026 in that moment",
)

#: Deliberately absent in Phase 1: any parameter, flag, environment
#: variable, or code path that resolves a 2026 file. `assert_forecast_
#: seasons_allowed` below has NO override argument by design -- an override
#: that always evaluates False is a weaker guard than a door that was never
#: built. Phase 2 must ADD its own separately-named entry point, reviewed on
#: its own terms, exactly as `evaluation/run_v1_final_evaluation.py` did for
#: 2025.
PHASE_2_ENTRY_POINT_EXISTS = False


class ForecastSeasonError(ValueError):
    """Raised when a season outside this package's authorization is requested."""


class ForecastNamespaceError(ValueError):
    """Raised when a path inside a sealed or Phase-2 namespace is requested."""


def assert_forecast_seasons_allowed(
    seasons: int | str | list[int | str] | tuple[int | str, ...],
) -> None:
    """Refuse any season this package is not authorized to read.

    Layered on top of -- never a replacement for -- `mlb_luck_score.config.
    assert_seasons_allowed`, which callers must also invoke when they touch
    the shared development dataset.

    Raises:
        ForecastSeasonError: If any requested season is 2025 (permanently
            sealed), 2026 (Phase 2, gated), or is not a development season
            this package reads.
    """
    if isinstance(seasons, int | str):
        requested: tuple[int | str, ...] = (seasons,)
    else:
        requested = tuple(seasons)

    years: set[int] = set()
    for season in requested:
        try:
            years.add(int(season))
        except (TypeError, ValueError) as exc:
            raise ForecastSeasonError(f"Season {season!r} is not an integer year") from exc

    sealed = years & set(FORECAST_SEALED_SEASONS)
    if sealed:
        raise ForecastSeasonError(
            f"Season(s) {sorted(sealed)} are permanently sealed final-evaluation data. "
            "Contact Forecast has no code path that reads them and no flag that enables "
            "one. See RESEARCH_RULES.md 'The single most important rule'."
        )

    phase_2 = years & set(FORECAST_PHASE_2_SEASONS)
    if phase_2:
        raise ForecastSeasonError(
            f"Season(s) {sorted(phase_2)} are reserved for Contact Forecast Phase 2 and "
            "are unreachable from Phase 1 code. The Phase 2 gate requires, in order: "
            f"{'; '.join(PHASE_2_GATE)}."
        )

    unknown = years - set(FORECAST_DEVELOPMENT_SEASONS)
    if unknown:
        raise ForecastSeasonError(
            f"Season(s) {sorted(unknown)} are not Contact Forecast development seasons "
            f"(authorized: {list(FORECAST_DEVELOPMENT_SEASONS)})"
        )


def assert_path_outside_forbidden_namespaces(path: Path) -> None:
    """Refuse a path resolving inside a sealed (2025) or Phase-2 (2026) namespace.

    A second, path-level layer behind `assert_forecast_seasons_allowed`:
    season lists guard what a caller ASKS for, this guards what a caller
    actually OPENS. Mirrors `prospective.prospective_config`'s
    `SealedNamespaceAccessError` check.

    Raises:
        ForecastNamespaceError: If `path` is inside a forbidden root.
    """
    resolved = Path(path).resolve()
    for root in FORBIDDEN_NAMESPACE_ROOTS:
        if resolved == root or root in resolved.parents:
            raise ForecastNamespaceError(
                f"{resolved} resolves inside the forbidden namespace {root}. Contact "
                "Forecast Phase 1 reads development data only."
            )


def walk_forward_train_seasons(analysis_season: int) -> tuple[int, ...]:
    """The development seasons a contact model may be fit on to score `analysis_season`.

    Raises:
        ForecastSeasonError: If `analysis_season` is not an analysis season,
            or if the configured training seasons are not strictly earlier
            than it (a self-check against a future edit that would silently
            introduce look-ahead).
    """
    if analysis_season not in WALK_FORWARD_TRAIN_SEASONS:
        raise ForecastSeasonError(
            f"{analysis_season} is not a Contact Forecast analysis season "
            f"(authorized: {list(FORECAST_ANALYSIS_SEASONS)})"
        )
    train = WALK_FORWARD_TRAIN_SEASONS[analysis_season]
    not_earlier = [s for s in train if s >= analysis_season]
    if not_earlier:
        raise ForecastSeasonError(
            f"Look-ahead in WALK_FORWARD_TRAIN_SEASONS[{analysis_season}]: training "
            f"season(s) {not_earlier} are not strictly earlier than the scored season"
        )
    assert_forecast_seasons_allowed(train)
    return train


def window_grid() -> tuple[tuple[int, int], ...]:
    """Every `(cutoff, horizon)` cell, in a deterministic order."""
    return tuple((k, h) for k in FORECAST_CUTOFFS for h in FORECAST_HORIZONS)


# --------------------------------------------------------------------------
# Contact-stage terminology (R1 quantities are NOT the published metric)
# --------------------------------------------------------------------------

#: The three research quantities, under production's own internal names for
#: the contact stage. `contact_result_surprise = observed_contact_result_run_
#: value - baseline_expected_contact_run_value`, i.e. Rc - E0.
OBSERVED_COLUMN = "observed_contact_result_run_value"
DESERVED_COLUMN = "baseline_expected_contact_run_value"
SURPRISE_COLUMN = "contact_result_surprise"

#: Every report, plot, artifact name, and conclusion must respect this.
#: `contact_result_surprise` (Rc - E0) is NOT the published Contact Luck
#: metric, which is Rf - E0 -- the full telescoping quantity including
#: batter-runner advancement. R1 uses the contact stage because reproducing
#: Rf walk-forward would introduce fold-specific model-selection and hence
#: architecture drift. The two are extremely highly correlated in the
#: causally comparable 2024 fold (Pearson 0.995, Spearman 0.994 across 216
#: qualified hitters; mean absolute difference 0.16 runs/100 against a
#: metric SD of 2.12), and the report MAY say so -- but every R1 conclusion
#: must be phrased as evidence about the contact-stage / luck-adjusted
#: contact signal, never as proof about the exact published metric.
CONTACT_STAGE_DISCLAIMER = (
    "R1 measures the CONTACT STAGE (Rc - E0), not the published Contact Luck "
    "metric (Rf - E0, which includes batter-runner advancement). The two are "
    "extremely highly correlated in the causally comparable 2024 fold, but "
    "these conclusions are evidence about the luck-adjusted contact signal, "
    "not proof about the exact published metric."
)

#: Phrases that must never be applied to an R1 quantity or conclusion.
#: `forecast.targets.assert_no_banned_metric_language` enforces this on
#: generated text, mirroring how `mlb_luck_score.scoring.public_labels`
#: centralizes the public metric's own banned-phrase list.
BANNED_R1_METRIC_PHRASES: tuple[str, ...] = (
    "the published contact luck metric",
    "the contact luck metric",
    "contactluck.com metric",
    "proves contact luck",
    "the official contact luck",
)

# --------------------------------------------------------------------------
# The conditional nature of the forecast (must appear in the spec and report)
# --------------------------------------------------------------------------

#: What a horizon actually means. A target is defined ONLY over hitters who
#: went on to accumulate the horizon's worth of additional resolved eligible
#: batted balls, so the forecast is conditional on that having happened. It
#: is not a projection of playing time, and the survivorship evidence is not
#: hypothetical: in 2024 at cutoff 200 / horizon 100, 41.7% of hitters who
#: reached the cutoff never reached the horizon, and those who did were
#: hitting materially better through the cutoff (5.21 vs 3.96 runs/100).
CONDITIONAL_FORECAST_LANGUAGE = (
    "A {horizon}-BBE target means: future performance CONDITIONAL ON the hitter "
    "subsequently accumulating {horizon} additional resolved eligible batted balls. "
    "It does not predict continued playing time, roster survival, injury, demotion, "
    "or whether the player will actually reach that horizon."
)


def conditional_forecast_language(horizon: int) -> str:
    """The conditional-interpretation sentence for a given horizon."""
    return CONDITIONAL_FORECAST_LANGUAGE.format(horizon=int(horizon))


# --------------------------------------------------------------------------
# PRIMARY SPECIFICATION -- frozen 2026-09-03, BEFORE any model result existed
# --------------------------------------------------------------------------

#: The principal user-facing forecasting target: future REALIZED contact-result
#: run value per 100 resolved BBE over the next 100 BBE.
PRIMARY_TARGET = "target_realized_rv_per_100"
PRIMARY_HORIZON = 100

#: All four cutoffs at the primary horizon are reported TOGETHER as the
#: primary family. No single cutoff may be promoted to "the" headline after
#: results are visible -- that is the post-hoc selection this freeze exists
#: to prevent.
PRIMARY_CUTOFFS: tuple[int, ...] = FORECAST_CUTOFFS

#: The cutoff the report presents first: 100 BBE observed -> next 100 BBE
#: realized RV/100.
#:
#: **This is a PRESENTATION VIEW, not a narrowing of the specification.**
#:
#: A product-focused amendment on 2026-09-03 proposed designating 100 -> 100
#: as the study's single headline cell. Treating it as a prespecification
#: would require repository history to PROVE the designation preceded any
#: computed metric. It does not: `forecast/` was untracked when the
#: amendment was made, so there is no commit -- of the constant, of
#: `forecast_config.py`, or of any R1 artifact -- that timestamps the
#: designation relative to the first metric. Working-file mtimes are not
#: repository history and do not close the gap.
#:
#: Under that failed test the conservative reading is the binding one:
#: `PRIMARY_CUTOFFS` stands unamended, all four cutoffs at the primary
#: horizon remain the primary family with none promoted above the others,
#: and 100 -> 100 is the focal view the report leads with for readability
#: only. No conclusion may rest on its selection, and every prespecified
#: (cutoff, horizon) cell stays visible in every table and artifact.
#:
#: The amendment and its rejection are recorded in
#: `forecast.freeze_r1.SPECIFICATION_AMENDMENTS`.
FOCAL_PRESENTATION_CUTOFF = 100

#: The principal attribution comparison: equivalently-shrunk realized versus
#: equivalently-shrunk deserved. Comparing RAW realized against RAW deserved
#: cannot separate a Contact-Luck-specific advantage from generic regression
#: to the mean, because deserved is the less variable quantity by
#: construction (measured: SD 5.54 vs 7.49 at cutoff 50 / horizon 50).
PRIMARY_COMPARISON: tuple[str, str] = (
    "shrunk_realized_persistence",
    "shrunk_deserved_persistence",
)

#: Complementary primary evaluation views. Reported together, none
#: privileged: MAE and RMSE in raw units, plus rank agreement. Pearson and
#: the rescaled MAE/RMSE are reported alongside as required secondary views.
PRIMARY_METRICS: tuple[str, ...] = ("mae", "rmse", "spearman")

#: Every cell that is not the primary family is a PRESPECIFIED SECONDARY
#: analysis, reported in full and never used to relocate the headline.
SECONDARY_HORIZONS: tuple[int, ...] = tuple(h for h in FORECAST_HORIZONS if h != PRIMARY_HORIZON)

#: The comparison ladder, in the order the report must present it. Rungs 4
#: and 5 are `PRIMARY_COMPARISON`; rungs 2 and 3 exist so a reader can see
#: exactly how much of any advantage is shrinkage rather than signal.
BASELINE_LADDER: tuple[str, ...] = (
    "league_mean",
    "raw_realized_persistence",
    "raw_deserved_persistence",
    "shrunk_realized_persistence",
    "shrunk_deserved_persistence",
)
