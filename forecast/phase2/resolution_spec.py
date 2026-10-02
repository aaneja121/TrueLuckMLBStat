"""Contact Forecast: the end-of-season 2026 resolution pass, prespecified.

The Phase 2 evaluations were run against a mid-season snapshot pinned at
2026-09-01. At that cutoff most hitters who had received a forecast had not
yet completed their horizon: 139 of 380 at H=100 and 252 of 380 at H=200 were
carried as PENDING, with their predictions frozen and no target attached.
Those predictions are already sealed. This module prespecifies -- before any
end-of-season outcome exists -- exactly what may be done with them once the
2026 regular season is over.

## The problem this specification exists to solve

The resolution pass is a SECOND LOOK at 2026. The first look was reported. No
alpha was spent, no group-sequential boundary was set, and nothing about the
first look reserved error rate for a second. Left unspecified, the pass would
be free to keep looking until a confidence interval cleared zero, which is
exactly the failure mode the rest of this study is built to prevent.

So the honest structure is fixed here, in advance:

  - The PRIMARY analysis is the INCREMENTAL cohort -- only those hitters whose
    horizons resolve after the pinned snapshot. That is the only genuinely
    unopened data, and it is the only cohort the frozen four-way
    classification decides.
  - The FULL-SEASON cohort (first-look windows plus incremental) is reported
    for precision, and is explicitly NOT independent confirmation: it reuses
    every window already seen. It may never upgrade the conclusion.
  - Neither look was alpha-adjusted, so the two together are not a single 5%
    test. The reported family-wise error rate is above nominal and the report
    must say so.

## What is NOT re-done

Forecasts are never regenerated. Regenerating them at an end-of-season cutoff
would build features from a longer history and silently change the prediction
being tested. The pass reads the sealed pending ledgers, verifies them by hash
against the frozen result manifests, and attaches outcomes to predictions that
already exist. A hitter who still has not completed the horizon at season's
end remains pending forever and is never evaluated on a partial window.

## Season discipline, unchanged

2025 stays sealed. 2026 is never fitted on: the model, its alpha and both
shrinkage benchmarks remain fitted on 2022-2024 only, exactly as frozen.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

from prospective.prospective_config import (
    PROSPECTIVE_2026_SEASON_END_DATE,
    PROSPECTIVE_2026_SEASON_END_SOURCE,
    PROSPECTIVE_2026_SEASON_END_VERIFIED,
    PROSPECTIVE_2026_SEASON_END_VERIFIED_AT,
)

from forecast.forecast_config import (
    DEFAULT_BOOTSTRAP_ALPHA,
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
    FOCAL_PRESENTATION_CUTOFF,
    FORECAST_OUTPUTS_DIR,
    PRIMARY_HORIZON,
    PROJECT_ROOT,
)
from forecast.metrics import DELTA_SIGN_CONVENTION
from forecast.phase2.h200_spec import H200_CUTOFF, H200_HORIZON, H200_OUTPUTS_DIR
from forecast.phase2.phase2_config import (
    PHASE2_OUTPUTS_DIR,
    PHASE2_PERMITTED_TRAINING_SEASONS,
    SUCCESS_CLASSIFICATION,
)

logger = logging.getLogger(__name__)

RESOLUTION_SPEC_VERSION = "contact_forecast_resolution_spec_v1"
RESOLUTION_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "forecast_phase2_resolution"

PRIMARY_BENCHMARK = "shrunk_deserved_persistence"
TARGET = "target_realized_rv_per_100"
FORECAST_COLUMN = "contact_forecast"


class ResolutionSpecError(RuntimeError):
    """Raised when the resolution specification cannot be frozen as required."""


# --------------------------------------------------------------------------
# What the pass consumes: already-sealed predictions, verified by hash
# --------------------------------------------------------------------------

#: Each horizon's frozen pending ledger, and the manifest that records its
#: hash. The pass must re-hash the ledger against the manifest and refuse to
#: run on any mismatch -- a prediction that has moved is not the prediction
#: that was made.
FROZEN_PREDICTION_LEDGERS: dict[str, dict[str, Any]] = {
    "H100": {
        "cutoff": FOCAL_PRESENTATION_CUTOFF,
        "horizon": PRIMARY_HORIZON,
        "outputs_dir": str(PHASE2_OUTPUTS_DIR),
        "pending_ledger": "phase2_2026_pending_predictions.parquet",
        "completed_ledger": "phase2_2026_completed_evaluation.parquet",
        "result_manifest": "phase2_provenance_manifest.json",
        "n_pending_at_first_look": 139,
        "n_completed_at_first_look": 241,
    },
    "H200": {
        "cutoff": H200_CUTOFF,
        "horizon": H200_HORIZON,
        "outputs_dir": str(H200_OUTPUTS_DIR),
        "pending_ledger": "h200_2026_pending_predictions.parquet",
        "completed_ledger": "h200_2026_completed_evaluation.parquet",
        "result_manifest": "h200_2026_result_manifest.json",
        "n_pending_at_first_look": 252,
        "n_completed_at_first_look": 128,
    },
}

#: The verified close of the season this pass resolves. Read from
#: `prospective.prospective_config`, which records it by the same four-constant
#: mechanism as the verified season START date -- the date, its source
#: citation, when it was verified, and the verified flag. It is loaded here
#: rather than restated so the specification cannot drift from the record.
SEASON_END_VERIFICATION: dict[str, Any] = {
    "season": 2026,
    "regular_season_end_date": PROSPECTIVE_2026_SEASON_END_DATE.isoformat(),
    "source": PROSPECTIVE_2026_SEASON_END_SOURCE,
    "verified": PROSPECTIVE_2026_SEASON_END_VERIFIED,
    "verified_at": PROSPECTIVE_2026_SEASON_END_VERIFIED_AT,
    "recorded_by": (
        "prospective.prospective_config.PROSPECTIVE_2026_SEASON_END_DATE, the same "
        "mechanism that records PROSPECTIVE_2026_SEASON_START_DATE"
    ),
    "independently_fetched_by_tooling": False,
    "gates": (
        "the resolution pass may not open a single outcome before this date has "
        "passed and a snapshot whose data-through date is on or after it exists"
    ),
}


#: The one snapshot the pass may read, and the conditions it must satisfy.
PINNED_SNAPSHOT_RULE: dict[str, Any] = {
    "count": 1,
    "rule": (
        "ONE end-of-season snapshot, pinned before any outcome is attached, "
        "integrity-hash verified, read-only, and never refreshed during the "
        "analysis. If the pass is interrupted it restarts against the SAME "
        "pinned snapshot, never a newer one."
    ),
    "data_through_requirement": ("on or after the final game of the 2026 regular season"),
    "may_generate_a_production_snapshot": False,
    "may_overwrite_a_production_snapshot": False,
}


# --------------------------------------------------------------------------
# The cohorts, and which one decides
# --------------------------------------------------------------------------

COHORTS: dict[str, dict[str, Any]] = {
    "incremental": {
        "role": "PRIMARY -- confirmatory",
        "definition": (
            "hitters who had a frozen PENDING prediction at the 2026-09-01 snapshot "
            "and who completed the horizon by the end of the regular season"
        ),
        "why_this_is_primary": (
            "It is the only 2026 data whose outcomes have never been examined. Every "
            "other cohort available at the resolution pass contains windows that were "
            "already read at the first look."
        ),
        "four_way_classification_applied": True,
        "may_decide_the_conclusion": True,
        "independent_of_the_first_look": True,
    },
    "full_season": {
        "role": "SECONDARY -- precision, never confirmation",
        "definition": (
            "every completed window at the end of the season: the first-look completed "
            "windows plus the incremental cohort"
        ),
        "why_this_is_not_primary": (
            "It reuses every window already observed at the first look, so its interval "
            "is not an independent test of the same hypothesis. It is the most precise "
            "estimate of the effect and the least valid significance test of it."
        ),
        # Corrected by amendment 2. This originally read True, contradicting
        # `classification_applies_to`. A classification label on a cohort that
        # may not decide the conclusion is precisely what a reader would quote
        # as a result, so the cohort now receives a delta and an interval and
        # no class at all.
        "four_way_classification_applied": False,
        "may_decide_the_conclusion": False,
        "receives_instead": (
            "a delta_MAE and a paired interval under the NOT INDEPENDENT label, with no "
            "classification label"
        ),
        "independent_of_the_first_look": False,
        "mandatory_label": (
            "NOT INDEPENDENT -- contains windows already observed at the first look"
        ),
    },
    "never_completed": {
        "role": "REPORTED ONLY -- never evaluated",
        "definition": (
            "hitters who received a forecast and had still not accumulated the horizon "
            "when the regular season ended"
        ),
        "four_way_classification_applied": False,
        "may_decide_the_conclusion": False,
        "rule": (
            "Never evaluated on a partial window, never dropped from the prediction "
            "ledger, and never imputed. Their count and cutoff-side characteristics are "
            "reported so the conditional nature of the result stays visible."
        ),
    },
}

#: How the two cohorts may and may not be combined in prose.
COHORT_DISAGREEMENT_RULE: dict[str, str] = {
    "if_they_agree": (
        "Report both. The agreement is worth stating, but the full-season cohort adds "
        "precision, not independent confirmation."
    ),
    "if_they_disagree": (
        "The PRIMARY incremental cohort stands. The full-season cohort may never be "
        "used to upgrade a conclusion the incremental cohort did not support."
    ),
    "forbidden": (
        "selecting whichever cohort reads better after seeing both, reporting only "
        "the more favourable one, or describing the full-season classification as "
        "confirmation of the first look."
    ),
}


# --------------------------------------------------------------------------
# The two-look problem, stated before the second look
# --------------------------------------------------------------------------

MULTIPLICITY_DISCLOSURE: dict[str, Any] = {
    "headline": (
        "The resolution pass is a SECOND unadjusted look at 2026. The two looks "
        "together are not a single 5% test."
    ),
    "first_look": {
        "snapshot": "2026-09-01",
        "reported": True,
        "alpha_spent": "none reserved -- no group-sequential design was prespecified",
        "h100_result": "PROMISING BUT INCONCLUSIVE",
        "h200_result": "PROMISING BUT INCONCLUSIVE",
    },
    "consequence": (
        "Two looks at overlapping data, each at a nominal 95% interval, have a "
        "family-wise error rate above 5%. The incremental cohort is disjoint from the "
        "first look's windows, which limits but does not eliminate the problem: the "
        "decision to run a second look at all was made after seeing the first."
    ),
    "what_this_specification_does_about_it": (
        "It fixes the primary cohort, the deciding metric, the interval procedure and "
        "the classification before any end-of-season outcome exists, and forbids "
        "choosing between cohorts afterwards. It does NOT claim to restore a nominal "
        "5% error rate, and no report may claim that it does."
    ),
    "what_may_never_be_claimed": (
        "that the resolution pass confirms the first look; that two looks pointing the "
        "same way is stronger evidence than one properly powered test; or that the "
        "full-season interval is a valid 95% interval for a prespecified hypothesis"
    ),
    "no_further_looks": (
        "This is the FINAL look at 2026 for these frozen predictions. There is no "
        "third pass. Windows still unresolved at season's end stay unresolved."
    ),
}


# --------------------------------------------------------------------------
# Survivorship, which changes shape at the resolution pass
# --------------------------------------------------------------------------

REQUIRED_SURVIVORSHIP_ANALYSIS: dict[str, Any] = {
    "why": (
        "The incremental cohort is, by construction, the SLOWER accumulators: hitters "
        "who needed the rest of the season to reach the horizon the first-look cohort "
        "reached by 2026-09-01. They are not a random sample of the pending set and "
        "they are not exchangeable with the first-look cohort."
    ),
    "required_comparisons": (
        "incremental completers versus first-look completers, at the cutoff",
        "incremental completers versus hitters who never completed, at the cutoff",
        "end-of-season completion rate versus the 2022-2024 development rate",
    ),
    "cutoff_side_only": True,
    "rule": (
        "Descriptive only. No reweighting, matching, adjustment, imputation or "
        "exclusion follows from any of it, and no hitter is added to or removed from "
        "a cohort on the basis of it."
    ),
    "generalization_note": (
        "A resolution-pass result generalizes only to hitters who accumulated the "
        "horizon within the season. It does not describe hitters who did not, and it "
        "does not predict playing time, injury, roster survival or demotion."
    ),
}


# --------------------------------------------------------------------------
# Everything the pass may not touch
# --------------------------------------------------------------------------

FROZEN_AND_UNTOUCHABLE: tuple[str, ...] = (
    "the frozen predictions themselves -- outcomes are attached, never regenerated",
    "Model D architecture, for either horizon",
    "the feature list",
    "alpha, for either horizon",
    "the preprocessing procedure and its fitted standardization",
    "the target definition",
    "the benchmark ladder and its shrinkage fits",
    "the primary metric (MAE)",
    "the paired batter-clustered bootstrap procedure, its reps, seed and alpha",
    "the four-way success classification",
    "the first-look results, which stand exactly as computed",
)

NOT_AUTHORIZED: tuple[str, ...] = (
    "regenerating forecasts at an end-of-season cutoff",
    "retuning alpha on any 2026 outcome",
    "refitting any model on 2026",
    "removing, trimming or winsorizing individual hitters",
    "imputing an outcome for an unresolved window",
    "changing features, target, benchmarks or metric",
    "reinterpreting the success threshold",
    "choosing between cohorts after seeing their results",
    "a third look at 2026",
    "reading 2025",
    "testing another model family",
    "deploying anything, or writing to the dashboard",
)

#: Conditions that must ALL hold before the pass may open a single outcome.
RESOLUTION_GATE: tuple[str, ...] = (
    "the 2026 regular season has ended -- its final date is now recorded and "
    "verified the way the season START date was, with a citation rather than an "
    "assumption: see SEASON_END_VERIFICATION",
    "the pinned snapshot's data-through date is on or after the verified regular-season end date",
    "this specification is frozen and verifies with zero drift",
    "the R1, ridge, HGB and h200_spec freezes verify with zero drift",
    "both frozen result manifests verify, and both pending ledgers re-hash to what they recorded",
    "exactly one end-of-season snapshot is pinned, integrity-verified, and read-only",
    "the maintainer has authorized the second look, in the moment, knowing it is a "
    "second unadjusted look",
)


# --------------------------------------------------------------------------
# The specification
# --------------------------------------------------------------------------


def build_resolution_specification() -> dict[str, Any]:
    """The complete end-of-season resolution specification, frozen before 2026 ends.

    Returns a DEEP COPY. The specification is assembled from module-level
    constants, and handing out references to them would let any caller that
    edits the result -- a test probing a guard, a report renderer normalising a
    field -- silently mutate the frozen constants for the rest of the process.
    """
    return copy.deepcopy(_build_resolution_specification())


def _build_resolution_specification() -> dict[str, Any]:
    """Assemble the specification from the frozen constants."""
    return {
        "spec_version": RESOLUTION_SPEC_VERSION,
        "status": "FROZEN before any end-of-season 2026 outcome was opened",
        "purpose": (
            "Attach outcomes to the already-frozen PENDING predictions from the Phase 2 "
            "evaluations, once the 2026 regular season is over."
        ),
        "is_a_second_look": True,
        "multiplicity_disclosure": MULTIPLICITY_DISCLOSURE,
        "frozen_prediction_ledgers": FROZEN_PREDICTION_LEDGERS,
        "predictions_regenerated": False,
        "how_outcomes_are_attached": (
            "Read the sealed pending ledger, verify it by hash against the frozen "
            "result manifest, order the end-of-season resolved eligible events, and "
            "emit a target ONLY where the hitter actually accumulated cutoff + horizon "
            "resolved batted balls. The frozen prediction columns are carried through "
            "unchanged and are never recomputed."
        ),
        "season_end_verification": SEASON_END_VERIFICATION,
        "pinned_snapshot": PINNED_SNAPSHOT_RULE,
        "cohorts": COHORTS,
        "cohort_disagreement_rule": COHORT_DISAGREEMENT_RULE,
        "primary_cohort": "incremental",
        "primary_comparison": (
            f"delta_MAE = MAE({FORECAST_COLUMN}) - MAE({PRIMARY_BENCHMARK}), on the "
            "incremental cohort"
        ),
        "metrics": {
            "primary": "MAE",
            "sign_convention": DELTA_SIGN_CONVENTION,
            "displayed_sign_convention_rule": (
                "derived from the comparison's own definition, never from the shared "
                "R1 constant, which describes a different pair"
            ),
            "negative_delta_means": f"the {FORECAST_COLUMN} is better",
            "interval": "paired 95% batter-clustered bootstrap confidence interval",
            "secondary": ["RMSE", "R_squared", "Pearson", "Spearman"],
            "secondary_may_reclassify": False,
        },
        "bootstrap": {
            "reps": DEFAULT_BOOTSTRAP_REPS,
            "seed": DEFAULT_BOOTSTRAP_SEED,
            "alpha": DEFAULT_BOOTSTRAP_ALPHA,
            "cluster": "batter",
        },
        "success_classification": SUCCESS_CLASSIFICATION,
        "classification_applies_to": "the incremental cohort only",
        "required_analyses": [
            "survivorship audit across incremental, first-look and never-completed",
            "stability and outlier sensitivity, on each classified cohort",
            "distribution-shift diagnostics",
            "forecast bias and calibration diagnostics",
        ],
        "survivorship": REQUIRED_SURVIVORSHIP_ANALYSIS,
        "underpowered_rule": (
            "If the incremental cohort is small, the result is reported as imprecise "
            "and classified mechanically anyway. A wide interval is a wide interval; it "
            "is never grounds for falling back to the full-season cohort, pooling the "
            "looks, or declining to report."
        ),
        "minimum_sample": None,
        "stopping_rule": None,
        "exploratory_power_estimate_is_a_rule": False,
        "season_discipline": {
            "training_seasons": list(PHASE2_PERMITTED_TRAINING_SEASONS),
            "sealed": [2025],
            "evaluation": [2026],
            "fitted_on_2026": False,
            "end_of_season_outcomes_opened": False,
        },
        "frozen_and_untouchable": list(FROZEN_AND_UNTOUCHABLE),
        "not_authorized": list(NOT_AUTHORIZED),
        "gate": list(RESOLUTION_GATE),
        "first_look_results_modified": False,
        "deployed": False,
    }


def assert_no_resolution_outcome(specification: dict[str, Any]) -> None:
    """Prove the specification carries no end-of-season outcome.

    Raises:
        ResolutionSpecError: If any target-like value has leaked into the spec.
    """

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                names_an_outcome = key.startswith(
                    ("target_realized", "target_deserved", "delta_mae", "delta_rmse")
                )
                # A key that NAMES an outcome is fine while it holds prose -- the
                # sign convention describes `delta_mae` without carrying one. A
                # number under that key would be an actual measured outcome.
                if names_an_outcome and isinstance(value, bool | int | float):
                    raise ResolutionSpecError(
                        f"The resolution specification carries a numeric {path}.{key!r}, "
                        "which is an outcome. No end-of-season outcome may exist when "
                        "this is frozen."
                    )
                _walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                _walk(item, f"{path}[{index}]")

    _walk(specification, "specification")
    if specification["season_discipline"]["end_of_season_outcomes_opened"]:
        raise ResolutionSpecError(
            "The specification records end-of-season outcomes as already opened; it "
            "must be frozen before the second look."
        )


# --------------------------------------------------------------------------
# The frozen conclusion and its checks
# --------------------------------------------------------------------------

RESOLUTION_ARTIFACTS: tuple[str, ...] = (
    "resolution_specification.json",
    "resolution_multiplicity_disclosure.json",
    "resolution_amendment_record.json",
    "resolution_specification_report.md",
)

RESOLUTION_SOURCE_MODULES: tuple[str, ...] = (
    "forecast/phase2/resolution_spec.py",
    "forecast/phase2/phase2_config.py",
    "forecast/phase2/phase2_windows.py",
    "forecast/phase2/snapshot.py",
)

RESOLUTION_CONCLUSION: dict[str, Any] = {
    "verdict": (
        "RESOLUTION PASS SPECIFICATION FROZEN -- no end-of-season 2026 outcome has "
        "been opened, and no result is claimed."
    ),
    "statement": (
        "The end-of-season resolution pass attaches outcomes to predictions that are "
        "already sealed; it never regenerates a forecast. Its primary cohort is the "
        "INCREMENTAL set -- hitters pending at the 2026-09-01 snapshot who complete "
        "the horizon by season's end -- because that is the only 2026 data whose "
        "outcomes have never been examined. The full-season cohort is reported for "
        "precision and is explicitly not independent confirmation. The pass is a "
        "second unadjusted look, its family-wise error rate is above nominal, and no "
        "report may claim otherwise."
    ),
    "must_not_be_described_as": [
        "a result -- no end-of-season outcome has been opened",
        "confirmation of the first look",
        "a single prespecified 5% test across both looks",
        "grounds to revisit the model, alpha, features, target or benchmarks",
    ],
    "preserved_negative_results": [
        "Both first-look evaluations were PROMISING BUT INCONCLUSIVE, and this "
        "specification does not treat that as a result awaiting rescue.",
        "At H=200 the first-look advantage reversed sign when the 5 and 10 largest "
        "beneficiaries were removed, so an outlier-sensitivity analysis is required on "
        "every classified cohort here too.",
        "The incremental cohort is the slower accumulators by construction; it is not "
        "exchangeable with the first-look cohort and the survivorship audit is "
        "mandatory rather than optional.",
    ],
}

RESOLUTION_LIMITATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "second_unadjusted_look",
        "statement": (
            "No alpha was reserved at the first look and no group-sequential boundary "
            "was set. Prespecifying the second look fixes what is tested, not the "
            "error rate: across two looks it remains above nominal 5%."
        ),
    },
    {
        "id": "decision_to_look_again_followed_the_first_result",
        "statement": (
            "The incremental cohort's outcomes are unopened, but the decision to run a "
            "resolution pass at all was taken after seeing an inconclusive first look. "
            "That is a selection effect this specification discloses and cannot remove."
        ),
    },
    {
        "id": "incremental_cohort_is_not_exchangeable",
        "statement": (
            "Hitters resolving only by season's end are slower accumulators than those "
            "who resolved by 2026-09-01. The incremental estimate describes them, and "
            "differs from the first-look cohort for reasons unrelated to model quality."
        ),
    },
    {
        "id": "full_season_cohort_is_not_a_valid_test",
        "statement": (
            "The full-season interval reuses every already-observed window. It is the "
            "most precise estimate available and the least valid significance test, and "
            "it may never upgrade the conclusion."
        ),
    },
    {
        "id": "conditional_on_reaching_the_horizon",
        "statement": (
            "Every cohort is conditional on the hitter accumulating the horizon within "
            "the season. Hitters who never do are reported and never evaluated, and no "
            "result here predicts playing time, injury, roster survival or demotion."
        ),
    },
)


#: The one anticipated amendment to the frozen specification: inserting the
#: externally verified season end date the original gate REQUIRED rather than
#: assumed. Recorded as an artifact so the re-freeze carries its own account of
#: why it happened and what was true when it did.
RESOLUTION_AMENDMENTS: tuple[dict[str, Any], ...] = (
    {
        "amendment": 1,
        "change": (
            "Inserted the externally verified 2026 regular-season end date "
            "(2026-09-27) and the gate condition that the pinned snapshot's "
            "data-through date fall on or after it."
        ),
        "why_this_was_not_a_specification_change": (
            "The original freeze deliberately left the date absent and made recording "
            "a verified one a GATE CONDITION. Supplying it is the step that gate "
            "anticipated, not a revision of what is tested."
        ),
        "previous_freeze_manifest_sha256": (
            "4e16f0ee987545e40983565353a381d98c5fd898b31d9e0b23bad7cbbc8b8d66"
        ),
        "content_otherwise_unchanged": True,
        "verified_by": (
            "a leaf-by-leaf diff of the specification before and after: nothing was "
            "removed, and every value outside the new season_end_verification block "
            "and the gate list is identical"
        ),
        "cohort_outcomes_opened_at_amendment_time": {
            "incremental": False,
            "full_season": False,
            "h200_pending": False,
            "h100_pending": False,
        },
        "re_freeze_occurred_before_any_incremental_cohort_outcome_was_opened": True,
        "first_look_results_modified": False,
        "evaluation_run_early": False,
        "next_evaluation": (
            "the final 2026 regular-season resolution pass, after 2026-09-27; it is not run early"
        ),
    },
    {
        "amendment": 2,
        "change": (
            "Corrected cohorts.full_season.four_way_classification_applied from True to "
            "False, and recorded what that cohort receives instead."
        ),
        "why": (
            "The frozen specification contradicted itself: `classification_applies_to` "
            "read 'the incremental cohort only' while the full-season cohort was marked "
            "as receiving the four-way classification. Building the runner surfaced it. "
            "A classification label on a cohort that may not decide the conclusion is "
            "exactly what a reader would quote as a result, so the contradiction is "
            "resolved toward the stricter reading rather than left to prose at "
            "reporting time."
        ),
        "what_did_not_change": (
            "the primary cohort, the deciding metric, the interval procedure, the "
            "four-way rule itself, and the full-season cohort's "
            "may_decide_the_conclusion flag, which was already False"
        ),
        "resolved_toward": "the stricter reading -- the incremental cohort only",
        "invariant_added": (
            "assert_resolution_specification now requires classification_applies_to and "
            "the per-cohort four_way_classification_applied flags to agree, so the two "
            "cannot drift apart again"
        ),
        "previous_freeze_manifest_sha256": (
            "52da1f6f33556e8cb9f05acd430aa12a8e1662118b4311316a41c5629aa0e91e"
        ),
        "content_otherwise_unchanged": True,
        "verified_by": (
            "a leaf-by-leaf diff of the specification before and after: only the "
            "full-season cohort's classification flag changed, plus the added "
            "receives_instead note"
        ),
        "cohort_outcomes_opened_at_amendment_time": {
            "incremental": False,
            "full_season": False,
            "h200_pending": False,
            "h100_pending": False,
        },
        "re_freeze_occurred_before_any_incremental_cohort_outcome_was_opened": True,
        "first_look_results_modified": False,
        "evaluation_run_early": False,
        "next_evaluation": (
            "the final 2026 regular-season resolution pass, after 2026-09-27; it is not run early"
        ),
    },
)


def extract_resolution_key_results(outputs_dir: Path) -> dict[str, Any]:
    """The specification's own key facts, read back from its artifacts."""
    spec = json.loads((outputs_dir / "resolution_specification.json").read_text())
    return {
        "spec_version": spec["spec_version"],
        "is_a_second_look": spec["is_a_second_look"],
        "primary_cohort": spec["primary_cohort"],
        "classification_applies_to": spec["classification_applies_to"],
        "predictions_regenerated": spec["predictions_regenerated"],
        "end_of_season_outcomes_opened": spec["season_discipline"]["end_of_season_outcomes_opened"],
        "full_season_may_decide": spec["cohorts"]["full_season"]["may_decide_the_conclusion"],
        "minimum_sample": spec["minimum_sample"],
        "stopping_rule": spec["stopping_rule"],
        "secondary_may_reclassify": spec["metrics"]["secondary_may_reclassify"],
        "success_classification_keys": sorted(spec["success_classification"]),
        "n_frozen_ledgers": len(spec["frozen_prediction_ledgers"]),
        "gate_requirements": len(spec["gate"]),
        "season_end_date": spec["season_end_verification"]["regular_season_end_date"],
        "season_end_verified": spec["season_end_verification"]["verified"],
        "cohorts_receiving_the_classification": sorted(
            name
            for name, cohort in spec["cohorts"].items()
            if cohort["four_way_classification_applied"]
        ),
        "n_amendments": len(RESOLUTION_AMENDMENTS),
        "re_frozen_before_any_incremental_outcome": all(
            item["re_freeze_occurred_before_any_incremental_cohort_outcome_was_opened"]
            for item in RESOLUTION_AMENDMENTS
        ),
        "amendment_cohort_outcomes_opened": sorted(
            {
                cohort
                for item in RESOLUTION_AMENDMENTS
                for cohort, opened in item["cohort_outcomes_opened_at_amendment_time"].items()
                if opened
            }
        ),
    }


def assert_resolution_specification(key_results: dict[str, Any]) -> None:
    """Raise if the frozen prose no longer describes the frozen specification.

    Raises:
        ResolutionSpecError: If any load-bearing element has drifted.
    """
    if key_results["end_of_season_outcomes_opened"]:
        raise ResolutionSpecError(
            "The specification records end-of-season outcomes as opened; it must be "
            "frozen before the second look."
        )
    if key_results["predictions_regenerated"]:
        raise ResolutionSpecError(
            "The specification permits regenerating forecasts. Outcomes are attached to "
            "sealed predictions; regenerating them at a later cutoff would change the "
            "prediction being tested."
        )
    if key_results["primary_cohort"] != "incremental":
        raise ResolutionSpecError(
            f"The primary cohort is {key_results['primary_cohort']!r}, not the "
            "incremental cohort. Only unopened data may decide the conclusion."
        )
    if key_results["full_season_may_decide"]:
        raise ResolutionSpecError(
            "The full-season cohort is marked as able to decide the conclusion. It "
            "reuses already-observed windows and may never do so."
        )
    if key_results["secondary_may_reclassify"]:
        raise ResolutionSpecError(
            "The specification allows secondary metrics to reclassify a result."
        )
    if key_results["minimum_sample"] is not None or key_results["stopping_rule"] is not None:
        raise ResolutionSpecError(
            "The specification records a minimum sample or a stopping rule. Neither "
            "exists: a wide interval is reported as wide, never used to keep looking."
        )
    if key_results["success_classification_keys"] != sorted(SUCCESS_CLASSIFICATION):
        raise ResolutionSpecError(
            "The four-way classification no longer matches the prespecified rule."
        )
    classified = key_results["cohorts_receiving_the_classification"]
    if classified != [key_results["primary_cohort"]]:
        raise ResolutionSpecError(
            f"`classification_applies_to` is {key_results['classification_applies_to']!r} "
            f"but the cohorts flagged as receiving it are {classified}. A cohort that may "
            "not decide the conclusion must not carry a classification label, and these "
            "two fields may never disagree."
        )
    if not key_results["season_end_verified"]:
        raise ResolutionSpecError(
            "The season end date is not marked verified. The pass may not open an "
            "outcome against an assumed season boundary."
        )
    if key_results["season_end_date"] != PROSPECTIVE_2026_SEASON_END_DATE.isoformat():
        raise ResolutionSpecError(
            f"The frozen season end date {key_results['season_end_date']!r} no longer "
            f"matches the recorded {PROSPECTIVE_2026_SEASON_END_DATE.isoformat()!r}. "
            "The date, its citation and its verification date move together or not at "
            "all."
        )
    if not key_results["re_frozen_before_any_incremental_outcome"]:
        raise ResolutionSpecError(
            "An amendment does not record that it was made before any incremental "
            "cohort outcome was opened. A specification may not be amended after the "
            "data it governs has been seen."
        )
    if key_results["amendment_cohort_outcomes_opened"]:
        raise ResolutionSpecError(
            f"Outcomes were already open for "
            f"{key_results['amendment_cohort_outcomes_opened']} when the specification "
            "was amended."
        )
    if not key_results["is_a_second_look"]:
        raise ResolutionSpecError(
            "The specification no longer records the pass as a second look, which is "
            "the fact its multiplicity disclosure exists to state."
        )


def resolution_freeze_spec() -> Any:
    """The `StageFreezeSpec` for this specification."""
    from forecast.stage_freeze import StageFreezeSpec

    return StageFreezeSpec(
        stage="resolution_spec",
        version=RESOLUTION_SPEC_VERSION,
        conclusion=RESOLUTION_CONCLUSION,
        limitations=RESOLUTION_LIMITATIONS,
        artifacts=RESOLUTION_ARTIFACTS,
        source_modules=RESOLUTION_SOURCE_MODULES,
        upstream_manifests=(
            "r1_freeze_manifest.json",
            "ridge_freeze_manifest.json",
            "hgb_freeze_manifest.json",
            "h200_spec_freeze_manifest.json",
        ),
        key_results=extract_resolution_key_results,
        checks=assert_resolution_specification,
    )


# --------------------------------------------------------------------------
# The specification report
# --------------------------------------------------------------------------


def render_specification_report(spec: dict[str, Any]) -> str:
    """The frozen specification as prose, written before the second look."""
    verification = spec["season_end_verification"]
    lines: list[str] = [
        "# Contact Forecast -- end-of-season 2026 resolution pass (PRESPECIFIED)",
        "",
        f"Specification `{spec['spec_version']}`. **{spec['status']}.**",
        "",
        "No end-of-season outcome has been opened. Nothing below is a result.",
        "",
        "## What the pass does",
        "",
        spec["purpose"],
        "",
        spec["how_outcomes_are_attached"],
        "",
        "Forecasts are **never** regenerated. Regenerating them at an end-of-season",
        "cutoff would build features from a longer history and silently change the",
        "prediction being tested.",
        "",
        "## The frozen predictions it consumes",
        "",
        "| horizon | cutoff -> horizon | pending at first look | completed at first look |",
        "|---|---|---|---|",
    ]
    for name, ledger in spec["frozen_prediction_ledgers"].items():
        lines.append(
            f"| {name} | {ledger['cutoff']} -> {ledger['horizon']} | "
            f"{ledger['n_pending_at_first_look']} | {ledger['n_completed_at_first_look']} |"
        )
    disclosure = spec["multiplicity_disclosure"]
    lines += [
        "",
        "Each pending ledger is re-hashed against its frozen result manifest before use.",
        "A prediction that has moved is not the prediction that was made, and the pass",
        "refuses to run on any mismatch.",
        "",
        "## This is a second look, and it is not alpha-adjusted",
        "",
        f"**{disclosure['headline']}**",
        "",
        disclosure["consequence"],
        "",
        disclosure["what_this_specification_does_about_it"],
        "",
        f"May never be claimed: {disclosure['what_may_never_be_claimed']}.",
        "",
        f"{disclosure['no_further_looks']}",
        "",
        "## Cohorts, and which one decides",
        "",
    ]
    for name, cohort in spec["cohorts"].items():
        lines += [
            f"### `{name}` -- {cohort['role']}",
            "",
            cohort["definition"] + ".",
            "",
            f"- Four-way classification applied: **{cohort['four_way_classification_applied']}**",
            f"- May decide the conclusion: **{cohort['may_decide_the_conclusion']}**",
        ]
        for key in ("why_this_is_primary", "why_this_is_not_primary", "rule", "mandatory_label"):
            if key in cohort:
                lines.append(f"- {cohort[key]}")
        lines.append("")
    rule = spec["cohort_disagreement_rule"]
    lines += [
        "If the cohorts agree: " + rule["if_they_agree"],
        "",
        "If they disagree: " + rule["if_they_disagree"],
        "",
        f"Forbidden: {rule['forbidden']}.",
        "",
        "## The comparison, fixed in advance",
        "",
        f"- Primary cohort: **{spec['primary_cohort']}**",
        f"- Primary comparison: `{spec['primary_comparison']}`",
        f"- Primary metric: {spec['metrics']['primary']}; negative delta means "
        f"{spec['metrics']['negative_delta_means']}",
        f"- Interval: {spec['metrics']['interval']}",
        f"- Secondary metrics: {', '.join(spec['metrics']['secondary'])} -- reported, "
        "and may never reclassify",
        f"- Classification applies to: **{spec['classification_applies_to']}**",
        f"- Minimum sample: {spec['minimum_sample']}. Stopping rule: {spec['stopping_rule']}.",
        "",
        spec["underpowered_rule"],
        "",
        "## Survivorship",
        "",
        spec["survivorship"]["why"],
        "",
        "Required comparisons, cutoff-side only:",
        "",
    ]
    lines += [f"- {item}" for item in spec["survivorship"]["required_comparisons"]]
    lines += [
        "",
        spec["survivorship"]["rule"],
        "",
        spec["survivorship"]["generalization_note"],
        "",
        "## Frozen and untouchable",
        "",
    ]
    lines += [f"- {item}" for item in spec["frozen_and_untouchable"]]
    lines += ["", "## Not authorized", ""]
    lines += [f"- {item}" for item in spec["not_authorized"]]
    lines += [
        "",
        "## The gate -- every condition must hold before one outcome is opened",
        "",
    ]
    lines += [f"{i}. {item}" for i, item in enumerate(spec["gate"], start=1)]
    lines += [
        "",
        "",
        "## The verified season end date",
        "",
        f"- 2026 regular season ended: **{verification['regular_season_end_date']}**",
        f"- Verified: **{verification['verified']}** (recorded {verification['verified_at']})",
        f"- Source: {verification['source']}",
        f"- Recorded by: `{verification['recorded_by']}`",
        f"- Independently fetched by this repository's tooling: "
        f"**{verification['independently_fetched_by_tooling']}**",
        "",
        "The original freeze deliberately left this date absent and required it as a",
        "gate condition rather than assuming one. Inserting it is that anticipated",
        "step, and nothing else in the specification changed with it.",
        "",
        "## Season discipline",
        "",
        f"- Training seasons: {spec['season_discipline']['training_seasons']}",
        f"- Sealed: {spec['season_discipline']['sealed']} -- never read",
        f"- Fitted on 2026: {spec['season_discipline']['fitted_on_2026']}",
        f"- End-of-season outcomes opened: "
        f"{spec['season_discipline']['end_of_season_outcomes_opened']}",
        f"- First-look results modified: {spec['first_look_results_modified']}",
        f"- Deployed: {spec['deployed']}",
        "",
    ]
    return "\n".join(lines) + "\n"


def _json_default(value: Any) -> Any:
    raise TypeError(f"{type(value)!r} is not JSON-serializable")


def run(
    *,
    outputs_dir: Path = RESOLUTION_OUTPUTS_DIR,
    research_dir: Path = FORECAST_OUTPUTS_DIR,
    h200_dir: Path = H200_OUTPUTS_DIR,
) -> dict[str, Path]:
    """Build, write and freeze the resolution specification. Opens no outcome."""
    outputs_dir.mkdir(parents=True, exist_ok=True)

    specification = build_resolution_specification()
    assert_no_resolution_outcome(specification)

    (outputs_dir / "resolution_specification.json").write_text(
        json.dumps(specification, indent=2, sort_keys=True, default=_json_default)
    )
    (outputs_dir / "resolution_multiplicity_disclosure.json").write_text(
        json.dumps(MULTIPLICITY_DISCLOSURE, indent=2, sort_keys=True, default=_json_default)
    )
    (outputs_dir / "resolution_amendment_record.json").write_text(
        json.dumps(
            {"amendments": list(RESOLUTION_AMENDMENTS)},
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )
    (outputs_dir / "resolution_specification_report.md").write_text(
        render_specification_report(specification)
    )

    from forecast.stage_freeze import freeze_stage

    # The upstream manifests live in the stages' own namespaces; copy them in so
    # the freeze is self-contained, exactly as the H=200 specification does.
    for name in (
        "r1_freeze_manifest.json",
        "ridge_freeze_manifest.json",
        "hgb_freeze_manifest.json",
    ):
        (outputs_dir / name).write_text((research_dir / name).read_text())
    (outputs_dir / "h200_spec_freeze_manifest.json").write_text(
        (h200_dir / "h200_spec_freeze_manifest.json").read_text()
    )
    manifest = freeze_stage(resolution_freeze_spec(), outputs_dir=outputs_dir)
    logger.info("froze the resolution specification: %s", manifest["manifest_sha256"])
    return {
        name: outputs_dir / name
        for name in (*RESOLUTION_ARTIFACTS, "resolution_spec_freeze_manifest.json")
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=RESOLUTION_OUTPUTS_DIR)
    parser.add_argument("--research-dir", type=Path, default=FORECAST_OUTPUTS_DIR)
    parser.add_argument("--h200-dir", type=Path, default=H200_OUTPUTS_DIR)
    args = parser.parse_args(argv)
    written = run(
        outputs_dir=args.outputs_dir, research_dir=args.research_dir, h200_dir=args.h200_dir
    )
    for name, path in written.items():
        logger.info("%s -> %s", name, path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
