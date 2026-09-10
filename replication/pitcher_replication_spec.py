"""Version 0.14: the PRE-REGISTERED specification for the one-time 2025
held-out Pitcher Contact Luck replication.

This module is a DECLARATION, not a computation. It reads no data, opens no
season, trains nothing, and imports nothing that can reach a parquet file or
the network. Everything here is a constant, fixed on the basis of the 2024
development work (Versions 0.13 and 0.13.1) and frozen BEFORE any 2025
pitcher output exists.

## What "frozen" means here

Three separable things are frozen, and conflating them is the failure mode
this module exists to prevent:

1. **The measured quantity** (`PLAY_LEVEL_DEFINITION`, `PRIMARY_QUANTITY`,
   `SECONDARY_QUANTITY`, `DENOMINATOR`). Reused unchanged from the frozen
   scoring architecture. No pitcher-specific model, no retraining, no second
   scoring path.
2. **The presentation architecture** (`ROLE_LIKE_GROUPING`,
   `PRESENTATION_RULES`). Descriptive, 2024-derived, and not reopened after
   2025 is seen except through a reported REVISE/NO-GO outcome.
3. **The replication questions and the rule that classifies their answers**
   (`REPLICATION_QUESTIONS`, `CLASSIFICATION_RULE`). This is the actual
   pre-registration: the diagnostics are named, and what would count as
   agreement or disagreement is written down, before the data that answers
   them is opened.

## What this module deliberately does NOT do

- It does not authorize a 2025 run. 2025 is `FINAL_TEST_SEASONS`, and
  `RESEARCH_RULES.md` permits it to enter this repository exactly once,
  through the sealed Version 1.0 entry point, which has already been used.
  A second sealed evaluation is a separate decision requiring the
  maintainer's explicit sign-off; see `AUTHORIZATION_STATUS` and
  `pitcher_replication_freeze.assert_ready_for_2025`.
- It does not restate a threshold, run value, or interval procedure that
  already lives in `mlb_luck_score`. Where a frozen value is defined
  elsewhere, this module records WHERE, and the freeze artifact records the
  source file's hash. One canonical source per fact.
- It does not set a numeric pass/fail bar on any single statistic. See
  `CLASSIFICATION_RULE`.

Version 0.13/0.13.1 findings this replication tests are in `README.md`,
"Pitching Contact Luck presentation research (Version 0.13.1)".
"""

from __future__ import annotations

from typing import Any

SPEC_VERSION = "0.14.0"

#: The season this specification was derived from. Development only.
DEVELOPMENT_SEASON = 2024

#: The season this specification will be replicated against, exactly once.
#: Held-out. Not opened by this module, and not openable without the
#: separate authorization recorded in `AUTHORIZATION_STATUS`.
REPLICATION_SEASON = 2025

#: Seasons this specification and its replication may never touch. 2026 is
#: prospective scoring territory (Version 1.1) and is out of scope here.
PROHIBITED_SEASONS: tuple[int, ...] = (2026,)


# ---------------------------------------------------------------------------
# 1. Play-level definition
# ---------------------------------------------------------------------------

PLAY_LEVEL_DEFINITION: dict[str, Any] = {
    "formula": "pitcher_contact_luck = expected_run_value - observed_run_value",
    "equivalently": "-1 x batting_contact_luck on the same play",
    "positive_means": (
        "The realized outcome was MORE favorable to the pitcher than the frozen contact "
        "model expected."
    ),
    "negative_means": (
        "The realized outcome was LESS favorable to the pitcher than the frozen contact "
        "model expected."
    ),
    "implementation": {
        "sign_convention_module": "mlb_luck_score.scoring.pitching_contact_luck",
        "negated_columns_constant": "SIGNED_VALUE_COLUMNS",
        "per_play_source_column": "final_result_surprise",
        "per_play_source_module": "mlb_luck_score.scoring.attribution_ledger",
    },
    "reuses_frozen_architecture": True,
    "trains_a_pitcher_specific_model": False,
    "retrains_any_component_model_for_the_pitcher_stage": False,
    "note": (
        "The four component models are still fit fresh and deterministically on "
        "TRAIN_SEASONS (2021-2023) by the frozen runner, exactly as Version 1.0 and "
        "Version 1.1 do. 'No retraining' means no NEW or pitcher-specific model, not a "
        "cached model object -- this codebase persists none."
    ),
}


# ---------------------------------------------------------------------------
# 2. Primary season quantity
# ---------------------------------------------------------------------------

PRIMARY_QUANTITY: dict[str, Any] = {
    "name": "cumulative_contact_luck_runs",
    "definition": (
        "Sum of the per-play pitcher contribution over the outcome-resolved eligible "
        "batted balls the pitcher actually allowed, in runs."
    ),
    "exact_claim": (
        "The displayed cumulative total is exact for the observed plays, conditional on "
        "the frozen Contact Luck scoring model."
    ),
    "is_retrospective": True,
    "is_workload_sensitive_by_design": True,
    "answers": (
        "Across the batted balls this pitcher actually allowed, how many runs did "
        "realized outcomes differ from what the contact itself deserved?"
    ),
    "is_not": (
        "pitcher skill",
        "talent",
        "persistence",
        "pitcher quality",
        "a forecast",
        "expected future performance",
    ),
    "claims_that_must_never_be_written": (
        "that there is no model uncertainty",
        "that the number is objectively true independent of the model",
        "that the pitcher owns a persistent luck skill",
    ),
}


# ---------------------------------------------------------------------------
# 3. Secondary normalized quantity
# ---------------------------------------------------------------------------

SECONDARY_QUANTITY: dict[str, Any] = {
    "name": "contact_luck_per_100",
    "definition": "Cumulative Contact Luck runs per 100 resolved eligible batted balls.",
    "status": "secondary_only",
    "may_be_the_official_ranking_key": False,
    "must_always_appear_with": (
        "the resolved eligible BBE it is normalized over",
        "a 95% interval from the frozen interval procedure",
    ),
    "reason_it_is_secondary": (
        "Measured on 2024, resolving power (sd_signal / sd_measurement) is below 1 at "
        "every achievable pitcher workload, so an ordering on this quantity is "
        "substantially an ordering of noise."
    ),
    "visual_authority": (
        "Never more than the cumulative total. Only the cumulative total receives plot "
        "position, sign color and strong typography."
    ),
}

#: The 95% interval procedure, frozen. Defined in `mlb_luck_score.scoring.
#: aggregation_uncertainty.BootstrapDesign` and applied to pitchers by
#: `pitching_contact_luck.bootstrap_pitcher_season_intervals`, which reuses
#: `bootstrap_batter_season_intervals` unchanged -- resampling is over
#: `game_pk` within the grouping key, and which key that is does not change
#: the procedure. Recorded here by REFERENCE; the freeze artifact resolves
#: and hashes the real values.
INTERVAL_PROCEDURE: dict[str, Any] = {
    "design_object": "mlb_luck_score.scoring.aggregation_uncertainty.BootstrapDesign",
    "pitcher_entry_point": (
        "mlb_luck_score.scoring.pitching_contact_luck.bootstrap_pitcher_season_intervals"
    ),
    "resampling_unit_note": "game_pk clustered WITHIN the pitcher-season",
    "endpoints_are_swapped_on_negation": True,
    "endpoint_swap_reason": (
        "[lo, hi] negated is [-hi, -lo], not [-lo, -hi]. Negating both endpoints in "
        "place leaves every interval reading low > high."
    ),
    "may_be_changed_after_seeing_2025": False,
}


# ---------------------------------------------------------------------------
# 4. Denominator
# ---------------------------------------------------------------------------

DENOMINATOR: dict[str, Any] = {
    "name": "eligible_batted_balls",
    "meaning": "OUTCOME-RESOLVED eligible batted balls",
    "computed_as": '("resolved", "sum") in aggregate_to_batter_season',
    "resolved_test": 'ledger["observed_contact_result_run_value"].notna()',
    "defining_module": "mlb_luck_score.scoring.aggregate_attribution",
    "redefined_for_the_pitcher_stage": False,
    "unresolved_row_handling": {
        "rule": (
            "A row that is an eligible batted ball but has no resolved observed run "
            "value contributes NOTHING: not to the numerator, not to the denominator, "
            "and not to the games count."
        ),
        "field_error": (
            "`field_error` is marked INELIGIBLE by mlb_luck_score.eligibility "
            "(_AMBIGUOUS_OUTCOME_EVENTS / REASON_AMBIGUOUS_FIELD_ERROR) because the "
            "batter's base attainment is not determinable from the event alone. It "
            "therefore never reaches the resolved set and never enters this denominator."
        ),
        "fielders_choice": (
            "`fielders_choice` is likewise ambiguous "
            "(REASON_AMBIGUOUS_FIELDERS_CHOICE) and excluded on the same grounds. "
            "`fielders_choice_out` is a distinct, unambiguous event that DOES map to "
            "outcome_class 'out' and is included."
        ),
        "consequence": (
            "Resolved eligible BBE is strictly smaller than eligible BBE. Every rate, "
            "interval, precision requirement and workload figure in this replication is "
            "denominated in RESOLVED eligible BBE, on both the 2024 and 2025 sides."
        ),
        "games_definition": (
            "`games` counts DISTINCT games containing at least one RESOLVED eligible "
            "batted ball ('Scored Games'), not appearances and not MLB games played."
        ),
    },
}


# ---------------------------------------------------------------------------
# 5. Role-like descriptive grouping
# ---------------------------------------------------------------------------

ROLE_STARTER_LIKE_MIN_BBE_PER_APPEARANCE = 10.0
ROLE_RELIEVER_LIKE_MAX_BBE_PER_APPEARANCE = 8.0

ROLE_LIKE_GROUPING: dict[str, Any] = {
    "buckets": {
        "starter_like": ">= 10.0 resolved eligible BBE per appearance",
        "reliever_like": "<= 8.0 resolved eligible BBE per appearance",
        "ambiguous": "strictly between 8.0 and 10.0 BBE per appearance",
    },
    "starter_like_min_bbe_per_appearance": ROLE_STARTER_LIKE_MIN_BBE_PER_APPEARANCE,
    "reliever_like_max_bbe_per_appearance": ROLE_RELIEVER_LIKE_MAX_BBE_PER_APPEARANCE,
    "derived_from": "the observed bimodal 2024 BBE-per-appearance distribution",
    "is_official_role_metadata": False,
    "is_a_qualification_system": False,
    "banned_labels": ("closer", "starter", "reliever", "setup man", "swingman"),
    "banned_label_rule": (
        "This repository has NO authoritative pitcher role metadata. Factual usage may "
        "be described ('72 appearances, 2.6 BBE per appearance'); a role may not be "
        "inferred from it. 'closer' may be used only if authoritative role metadata is "
        "later added, which would be a separate, documented change."
    ),
    "ambiguous_bucket_belongs_to_neither_board": True,
    "boundaries_are_frozen_for_the_replication": True,
    "rebucketing_2025_to_taste_is_forbidden": True,
}


# ---------------------------------------------------------------------------
# 6. Presentation rules
# ---------------------------------------------------------------------------

#: PRESENTATION rule only. Never qualification, never eligibility, never an
#: official minimum, never an MLB rule. A season below it keeps its page,
#: its total, its rate and its interval, and is never described as having
#: failed anything.
BOARD_DISPLAY_MINIMUM_BBE = 60

PRESENTATION_RULES: dict[str, Any] = {
    "starter_like_and_reliever_like_analysed_separately": True,
    "single_combined_official_ranked_board": False,
    "primary_ordering": "cumulative_contact_luck_runs",
    "per_100_may_be_primary_ordering": False,
    "bbe_always_visible_beside_the_total": True,
    "board_rank_shown_on_player_card": False,
    "board_rank_rationale": (
        "A '#3' label would make a descriptive accumulation ordering read as an official "
        "pitcher-quality rank."
    ),
    "largest_favorable_and_unfavorable_plays_surfaced": True,
    "largest_play_rationale": (
        "Over half of 2024 reliever-like seasons had one batted ball worth more than "
        "half their net. Concealing it would make a season total look steadier than it "
        "is."
    ),
    "board_display_minimum_bbe": BOARD_DISPLAY_MINIMUM_BBE,
    "board_display_minimum_is_display_only": True,
    "board_display_minimum_is_qualification": False,
    "dedicated_zero_scale": {
        "name": "pitcher_cumulative_runs",
        "retained": True,
        "invariant_z_holds": True,
        "invariant_z_meaning": (
            "Built with zero_fraction inherited from the league scale, so zero sits at "
            "the same --cl-zero as every other figure on the site."
        ),
    },
    "near_zero_net_ratio_rule": (
        "Where |largest play| / |net total| exceeds roughly 2x the ratio stops being "
        "informative; say the season's batted balls very nearly cancelled instead of "
        "quoting a percentage."
    ),
    "single_resolved_play_rule": (
        "A season with one resolved batted ball must show that play once, not twice "
        "labelled both 'most favorable' and 'least favorable'."
    ),
    "reopening_rule": (
        "These rules are not reconsidered after 2025 is seen unless the replication is "
        "FIRST reported and frozen with a REVISE or NO-GO classification. A rule may "
        "never be quietly relaxed because a 2025 board looked better without it."
    ),
}


# ---------------------------------------------------------------------------
# The replication questions (A-G), pre-registered
# ---------------------------------------------------------------------------

REPLICATION_QUESTIONS: dict[str, dict[str, Any]] = {
    "A_population_and_centering": {
        "title": "Population / centering",
        "report": (
            "number of pitcher-seasons",
            "resolved eligible BBE distribution (min, p10, median, p90, max)",
            "play-level mean pitcher Contact Luck",
            "cumulative-total distribution (min, median, max, sd)",
            "per-100 distribution (min, median, max, sd)",
        ),
        "structural_hypothesis": (
            "The 2025 population is descriptively comparable in shape to 2024: a large "
            "pitcher-season count dominated by low-exposure rows, a play-level mean near "
            "zero, and a cumulative-total distribution centred near zero with heavy "
            "workload-driven tails."
        ),
        "requires_exact_numeric_replication": False,
        "disagreement_looks_like": (
            "A play-level mean materially displaced from zero, or a cumulative-total "
            "distribution whose centre is set by something other than workload."
        ),
    },
    "B_opportunity_heterogeneity": {
        "title": "Opportunity heterogeneity",
        "report": (
            "resolved eligible BBE min / median / max, for all pitchers and for each "
            "role-like group",
            "|cumulative total| vs BBE: Pearson and Spearman",
            "signed cumulative total vs BBE: Pearson and Spearman",
        ),
        "structural_hypothesis": (
            "Opportunity materially affects MAGNITUDE but only weakly determines SIGN or "
            "order. 2024 reference: |total| vs BBE Pearson 0.536 / Spearman 0.579; "
            "signed total vs BBE Pearson 0.194 / Spearman 0.102."
        ),
        "requires_exact_numeric_replication": False,
        "disagreement_looks_like": (
            "Signed-total-vs-BBE correlation approaching the magnitude correlation, "
            "which would mean workload determines whether a season reads favorable -- "
            "and would undercut cumulative totals as an accumulation measure."
        ),
    },
    "C_totals_vs_rate": {
        "title": "Totals vs. rate ordering",
        "report": (
            "Spearman(cumulative total, per-100) for all pitchers",
            "the same for starter-like only",
            "the same for reliever-like only",
            "the extreme rank movers under each, inspected individually",
        ),
        "structural_hypothesis": (
            "Heterogeneous opportunity makes an all-pitcher rate ranking vulnerable to "
            "tiny-sample extremes: seasons of a handful of batted balls occupy the ends "
            "of the per-100 distribution while their cumulative impact is approximately "
            "zero. 2024 reference: position players who pitched reached +21.9 and -47.1 "
            "runs/100 on a single batted ball."
        ),
        "requires_exact_numeric_replication": False,
        "disagreement_looks_like": (
            "An all-pitcher rate ordering that is NOT dominated at its extremes by "
            "tiny-sample rows, which would weaken the case for totals-first."
        ),
    },
    "D_reliever_single_play_dominance": {
        "title": "Reliever-like single-play dominance",
        "report": (
            "share of reliever-like pitcher-seasons whose largest absolute single-play "
            "contribution exceeds 25% of |net total|",
            "the same at 50%",
            "the same at 100%",
            "the count of rows with an undefined share (net total at or near zero)",
        ),
        "structural_hypothesis": (
            "The phenomenon remains materially present. 2024 reference: 83.2% / 52.5% / "
            "27.7%. These exact percentages are NOT required."
        ),
        "requires_exact_numeric_replication": False,
        "disagreement_looks_like": (
            "Shares low enough that a reliever-like season total no longer reads as "
            "one-or-two-play driven -- which would remove the justification for "
            "surfacing the largest plays."
        ),
    },
    "E_rate_precision": {
        "title": "Rate precision and resolving power",
        "report": (
            "interval half-width by workload",
            "resolving power at practical workload floors (>=60, >=150, >=300, >=450 resolved BBE)",
            "resolved BBE required for +/-5, +/-4, +/-3, +/-2 runs per 100",
            "the count of zero-width intervals and the count of one-appearance rows",
        ),
        "estimators": "see FROZEN_ESTIMATORS",
        "implementation_module": "replication.pitcher_replication_estimators",
        "structural_hypothesis": (
            "Resolving power stays below 1 at every practical workload floor, and the "
            "resolved BBE needed for +/-2 runs/100 exceeds the maximum workload any real "
            "pitcher-season reaches. 2024 reference: 0.51 / 0.45 / 0.00 / 0.50; 1,246 "
            "BBE required against a 610 BBE maximum."
        ),
        "requires_exact_numeric_replication": False,
        "mandatory_guard": (
            "The >=1 BBE floor is EXCLUDED from the headline table. A game-clustered "
            "bootstrap gives a one-appearance pitcher-season a zero-width interval "
            "(there is one game to resample, so every replicate reproduces the point "
            "estimate), its measurement variance records as zero, and the decomposition "
            "then books all of its enormous spread as signal. On 2024 this produced a "
            "spurious 1.94; excluding those 73 rows gives 0.877. Any >=1 BBE figure MUST "
            "be reported alongside its zero-width count and its "
            "excluding-zero-width recomputation, and must never be read as signal."
        ),
        "disagreement_looks_like": (
            "Resolving power at or above 1 at a practical floor AFTER the zero-width "
            "guard is applied, which would mean rates carry real separable signal and "
            "the secondary-only status should be revisited."
        ),
    },
    "F_persistence": {
        "title": "Persistence / split-half reliability",
        "report": (
            "split-half reliability for both frozen splits at min_eligible_each_half=20",
            "the same at each frozen sweep value (50, 100)",
            "the batter-side reproduction control, which validates the implementation",
        ),
        "structural_hypothesis": (
            "Approximately zero at all pitcher workloads. Contact Luck is retrospective "
            "luck, not persistent skill; a quantity that does not persist is not "
            "supposed to be reliable, so a near-zero result CONFIRMS the framing rather "
            "than indicting the metric."
        ),
        "requires_exact_numeric_replication": False,
        "procedure_status": "IMPLEMENTED_AND_FROZEN",
        "implementation_module": "replication.pitcher_split_half",
        "procedure": {
            "ported_from": (
                "mlb_luck_score.models.evaluate_aggregation_stability."
                "compute_split_half_reliability (Version 0.11 Phase 6)"
            ),
            "port_method": (
                "The split rules (_half_season_masks, _odd_even_masks), the inclusion "
                "threshold (MIN_ELIGIBLE_EACH_HALF = 20) and the metric (RANKING_METRIC) "
                "are IMPORTED from the accepted module, never restated, so 'the same "
                "design' holds by construction. The only change is the grouping key: "
                "aggregate_to_pitcher_season instead of aggregate_to_batter_season."
            ),
            "splits": ("calendar", "odd_even"),
            "splits_are_game_clustered": True,
            "split_definitions": {
                "calendar": "game_date <= median(game_date) vs. the rest",
                "odd_even": "game_pk % 2 == 1 vs. the rest",
            },
            "min_eligible_each_half": 20,
            "min_eligible_each_half_sweep": (20, 50, 100),
            "sweep_note": (
                "The sweep re-runs the SAME estimator at three frozen values of its own "
                "existing min_eligible_each_half parameter. It is not a second "
                "statistic, and it exists because the carried-forward 2024 wording said "
                "'at all pitcher workloads' while the accepted procedure reports one "
                "number per split. The 20 row is primary."
            ),
            "statistic": "Pearson and Spearman between the two halves' per-100 rates",
            "sign_convention_is_irrelevant": (
                "A correlation is invariant under a common sign flip, so the pitcher "
                "negation does not change this figure for a fixed grouping key."
            ),
            "denominator_unchanged": True,
        },
        "development_2024_result": {
            "measured_by": "replication.pitcher_split_half.build_question_f_report",
            "pitcher_min_each_half_20": {
                "calendar": {"n": 439, "pearson_r": 0.1023, "spearman_r": 0.0720},
                "odd_even": {"n": 550, "pearson_r": 0.0768, "spearman_r": 0.0553},
            },
            "pitcher_sweep": {
                "50": {
                    "calendar": {"n": 335, "pearson_r": 0.0458, "spearman_r": 0.0273},
                    "odd_even": {"n": 392, "pearson_r": 0.0008, "spearman_r": 0.0215},
                },
                "100": {
                    "calendar": {"n": 134, "pearson_r": -0.1604, "spearman_r": -0.0986},
                    "odd_even": {"n": 150, "pearson_r": 0.0749, "spearman_r": 0.0787},
                },
            },
            "reading": (
                "Approximately zero at every workload examined: every |r| is at or below "
                "0.16 and the highest floor turns negative, which is what a "
                "non-persistent quantity looks like. This CONFIRMS the retrospective "
                "framing."
            ),
            "batter_side_reproduction": (
                "This implementation run on the BATTER key reproduces the committed "
                "Version 0.11 Phase 6 numbers EXACTLY -- calendar n=399, Pearson "
                "0.050297542421047836, Spearman 0.07131881210564099; odd_even n=487, "
                "Pearson 0.09654304470474931, Spearman 0.11889983530505965; absolute "
                "difference 0.0 on every value. That is the evidence the port did not "
                "change the procedure."
            ),
            "known_discrepancy": (
                "No precise 2024 PITCHER split-half number was ever recorded -- the "
                "prior claim was the qualitative 'approximately zero at all pitcher "
                "workloads'. There is therefore no exact prior value to reproduce on the "
                "pitcher side, and the figures above are this repository's first "
                "recorded measurement of it. This is documented rather than papered "
                "over; the batter-side exact reproduction is what validates the "
                "implementation."
            ),
        },
        "is_a_success_criterion_on_its_own": False,
        "cannot_override_package_classification": True,
        "disagreement_looks_like": (
            "Materially non-zero reliability, which would mean the quantity partly "
            "persists and the 'retrospective only' framing is incomplete. Even then, F "
            "is one input to the package classification, never the verdict."
        ),
    },
    "G_real_play_sanity_check": {
        "title": "Real-play sanity check",
        "report": (
            "an audited example of an extreme favorable season total",
            "an audited example of an extreme unfavorable season total",
            "an audited high-workload season",
            "an audited reliever-like season",
            "an audited near-zero season",
            "for each: the largest contributing batted balls with launch speed, launch "
            "angle, batted-ball type and outcome",
        ),
        "structural_hypothesis": (
            "Large contributions remain baseball-sensible: hard-hit balls that were "
            "caught read favorable to the pitcher, and weakly-hit balls that fell in or "
            "left the park read unfavorable."
        ),
        "requires_exact_numeric_replication": False,
        "is_qualitative": True,
        "disagreement_looks_like": (
            "Large contributions that cannot be explained by the contact itself, which "
            "would indicate a scoring or joining defect rather than a presentation "
            "question."
        ),
    },
}

#: The exact estimator forms question E uses. Frozen as FORMULAS here AND
#: implemented in `replication.pitcher_replication_estimators`, which the
#: freeze artifact pins by source hash. As of Version 0.14 these no longer
#: live in `demo/build_pitcher_prototype_fixture.py`: the freeze depends on
#: research code only, and the fixture generator is now a CONSUMER of this
#: module, so the dashboard fixture and the replication cannot drift apart.
FROZEN_ESTIMATORS: dict[str, Any] = {
    "resolving_power": {
        "decomposition": "var_observed = var_signal + mean(var_measurement)",
        "measurement_sd_per_row": "(ci_high - ci_low) / (2 * 1.96)",
        "var_observed": "np.var(per_100, ddof=1)",
        "statistic": "sqrt(var_signal) / sqrt(var_measurement)",
        "negative_signal_variance_is_reported_raw": True,
        "negative_signal_variance_note": (
            "A negative estimate means the observed spread is no wider than measurement "
            "error alone would produce. That is the finding; it is never clipped to zero "
            "and presented as near-zero-but-real."
        ),
        "workload_floors": (60, 150, 300, 450),
        "excluded_floor": 1,
        "excluded_floor_reason": "zero-width one-appearance artifact; see question E",
    },
    "rate_precision_requirement": {
        "form": "half_width_runs_per_100 = k / sqrt(resolved_bbe)",
        "k": "median(half_width * sqrt(resolved_bbe)) over rows with >= 30 resolved BBE",
        "fitted_on_rows_with_min_bbe": 30,
        "bbe_required": "round((k / target)^2)",
        "targets": (5, 4, 3, 2),
        "z_multiplier_note": (
            "95% is already embedded in the bootstrap interval endpoints. No separate "
            "1.96 multiplier is applied when fitting k."
        ),
        "development_reference_k": 70.588,
        "development_reference_bbe_required": {
            "plus_minus_5": 199,
            "plus_minus_4": 311,
            "plus_minus_3": 554,
            "plus_minus_2": 1246,
        },
    },
    "implementation_module": "replication.pitcher_replication_estimators",
    "single_play_dominance": {
        "statistic": "abs(largest single-play contribution) / abs(net season total)",
        "thresholds": (0.25, 0.50, 1.00),
        "undefined_when": "net season total is at or near zero; counted separately",
    },
}


# ---------------------------------------------------------------------------
# The classification rule
# ---------------------------------------------------------------------------

CLASSIFICATION_VALUES: tuple[str, ...] = ("REPLICATED", "REVISE", "NO_GO")

CLASSIFICATION_RULE: dict[str, Any] = {
    "values": CLASSIFICATION_VALUES,
    "REPLICATED": {
        "meaning": "The main structural conclusions survive.",
        "requires_all_of": (
            "cumulative totals remain coherent retrospective accounting",
            "heterogeneous opportunity still makes totals preferable as the primary "
            "pitcher presentation",
            "normalized rates remain substantially uncertainty-limited",
            "reliever-like small samples remain visibly sensitive to individual plays",
            "no evidence requires changing the frozen presentation architecture",
        ),
    },
    "REVISE": {
        "meaning": (
            "The metric remains coherent, but one or more presentation assumptions fail "
            "materially enough that the design should change before shipment."
        ),
    },
    "NO_GO": {
        "meaning": (
            "2025 reveals a structural failure in the pitcher formulation or "
            "presentation that makes the 2024 design misleading."
        ),
    },
    "decision_procedure": (
        "Judge the prespecified structural findings AS A PACKAGE. Answer each of A-G "
        "against its own structural hypothesis, record agreement or disagreement for "
        "each, and classify on the pattern across all of them."
    ),
    "significance_testing_is_not_the_rule": True,
    "significance_testing_note": (
        "Statistical significance on one arbitrary statistic must NOT be the "
        "classification rule. A significant deviation on a single correlation is one "
        "input among seven questions, not a verdict."
    ),
    "every_disagreement_must_be_reported": True,
    "reporting_rule": (
        "Every question that disagrees with its structural hypothesis is reported "
        "explicitly, including under a REPLICATED classification. A finding may not be "
        "omitted because the overall verdict absorbed it."
    ),
    "classification_precedes_any_design_change": True,
    "classification_precedes_note": (
        "The classification is reported and frozen BEFORE any presentation rule is "
        "changed in response to it. Changing a rule first and classifying afterwards "
        "converts a replication into a tuning pass."
    ),
    "exact_numeric_replication_is_not_required": True,
    "exact_numeric_note": (
        "None of A-G requires 2025 to reproduce a 2024 number. Each asks whether a "
        "STRUCTURAL relationship holds."
    ),
}


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------

#: The 2025 replication is NOT authorized by this specification's existence.
#:
#: `RESEARCH_RULES.md` ("The one narrow exception: the sealed Version 1.0
#: final evaluation") permits 2025 to enter this repository EXACTLY ONCE,
#: through `evaluation/run_v1_final_evaluation.py`, and states plainly that
#: being asked to run a second final evaluation, to loosen any guard, or to
#: make 2025 reachable from a second code path is "a new, separate decision
#: requiring the user's explicit sign-off, not a natural extension of this
#: one."
#:
#: A pitcher replication on 2025 is exactly that: a second sealed evaluation,
#: through a second code path. Freezing the specification first is the
#: correct order of operations and is what this module does. It does not
#: grant the authorization, and no code in this namespace may.
AUTHORIZATION_STATUS: dict[str, Any] = {
    "second_sealed_2025_evaluation_authorized": False,
    "authorizing_rule": (
        "RESEARCH_RULES.md, 'The one narrow exception: the sealed Version 1.0 final "
        "evaluation', closing paragraph"
    ),
    "required_before_any_2025_access": (
        "explicit maintainer sign-off, in the moment, for a SECOND sealed 2025 "
        "evaluation through a second code path",
        "a dedicated entry point that verifies a clean tree, records a frozen commit, "
        "validates this freeze artifact, and re-verifies every frozen source hash",
        "an isolated namespace for 2025 pitcher raw/derived data, outputs and artifacts, "
        "never mixed into the development caches",
        "a locally-defined 2025 date range inside that entry point, never added to "
        "mlb_luck_score.config",
        "fail-fast guards proving development runners still cannot reach 2025",
    ),
    "what_this_spec_provides": (
        "the pre-registered questions, estimators and classification rule that such a "
        "run would have to follow, fixed and hashed before the data is opened"
    ),
}


def spec_payload() -> dict[str, Any]:
    """The complete frozen specification as a plain, JSON-serializable dict.

    This is the object the freeze artifact hashes. Key order is irrelevant
    (the hash canonicalizes with `sort_keys=True`), but CONTENT is not: any
    change to any value here changes the spec hash and therefore invalidates
    an existing freeze, which is the intended behaviour.
    """
    return {
        "spec_version": SPEC_VERSION,
        "development_season": DEVELOPMENT_SEASON,
        "replication_season": REPLICATION_SEASON,
        "prohibited_seasons": list(PROHIBITED_SEASONS),
        "play_level_definition": PLAY_LEVEL_DEFINITION,
        "primary_quantity": PRIMARY_QUANTITY,
        "secondary_quantity": SECONDARY_QUANTITY,
        "interval_procedure": INTERVAL_PROCEDURE,
        "denominator": DENOMINATOR,
        "role_like_grouping": ROLE_LIKE_GROUPING,
        "presentation_rules": PRESENTATION_RULES,
        "replication_questions": REPLICATION_QUESTIONS,
        "frozen_estimators": FROZEN_ESTIMATORS,
        "classification_rule": CLASSIFICATION_RULE,
        "authorization_status": AUTHORIZATION_STATUS,
    }
