"""Contact Luck v0.10: real-data end-to-end run of the play-level attribution ledger.

Trains the four component models `mlb_luck_score.scoring.attribution_ledger.
build_attribution_ledger` reconciles (Version 0.2 contact, Version 0.7A
outfield opportunity, Version 0.8 infield opportunity, Version 0.9
advancement) on real, already-downloaded 2021-2024 Statcast data, builds the
ledger on the one season no component trained or selected against (2024),
and reports:

  - the exact accounting-identity reconciliation error (max/percentiles)
  - row counts by ledger pathway (which components applied to which rows)
  - component eligibility/fallback/provisional/unavailable counts
  - field-error handling
  - the residual distribution and any materially large residuals
  - confirmation that no 2025 data was read

## Model choices and why

  - Contact model: `baseline_v02` (`mlb_luck_score.models.train_contact_
    model`), trained on `TRAIN_SEASONS` (2021-2023) -- the ADOPTED default,
    unchanged. This is the model behind the ledger's own `baseline_expected_
    contact_run_value`/`observed_contact_result_run_value`.
  - Outfield opportunity: the single, already-validated `measured_contact_
    only_v07` candidate (`mlb_luck_score.models.train_opportunity_model`,
    default feature set), trained on `TRAIN_SEASONS`, applied UNIFORMLY to
    every outfield-opportunity-eligible row. The Version 0.7C/D near-wall
    specialist is deliberately NOT used here -- Version 0.7 is FROZEN (see
    CLAUDE.md "Version 0.7 is now FROZEN"), and gating between it and the
    open-field model is `mlb_luck_score.scoring.gated_outfield_report`'s
    job, not this script's. Real result as of this writing (`outputs/
    tables/opportunity_model_comparison_detail.json`): overall ECE clears
    the absolute threshold, but `passes_basic_validation` is `False` on
    near-wall/opportunity-time subgroup ECE -- reported honestly below via
    `outfield_confidence_status`, never silently upgraded.
  - Infield opportunity: Version 0.8, reusing `mlb_luck_score.models.
    compare_infield_opportunity.run_infield_model_selection`'s EXACT
    fit-on-2021-2022/select-on-2023 procedure (never retrained on more
    data than that procedure already validates) rather than inventing a
    different split. Real winner as of this writing: `infield_hgb_v08`,
    `overall_status=calibrated_with_limited_subgroup_evidence` (see
    `outputs/tables/infield_opportunity_detail.json`).
  - Advancement: Version 0.9, reusing `mlb_luck_score.models.
    compare_advancement_models.run_advancement_model_selection`'s identical
    fit/select procedure and `fit_advancement_contact_model`'s OWN
    train-only contact model for its `contact_p_*` input features (a
    DIFFERENT model instance from this script's own `baseline_v02` above --
    intentional, see that function's docstring on why). Real winner as of
    this writing: `advancement_speed_v09`, `overall_status=calibrated_with_
    limited_subgroup_evidence`, beats the empirical baseline (see `outputs/
    tables/advancement_detail.json`).

This script does not re-derive any of these three modules' full calibration
-gate machinery (subgroup/venue bootstrap confidence intervals, controlled
-perturbation checks) inline -- that machinery already exists, is expensive
(bootstrap alone is 500 game_pk-clustered resamples per candidate), and
re-implementing it here would risk silently drifting from the validated
version. Instead, `_read_existing_gate_status` reads each module's own
already-computed, already-saved JSON output (regenerate via `make compare-
opportunity-models` / `make compare-infield-opportunity` / `make compare-
advancement-models` if missing or stale) and cites it verbatim.

Weather (Version 0.5/0.5.1) and alignment (Version 0.6) never appear
anywhere in this script, consistent with `attribution_ledger`'s own scope --
neither passed adoption (see CLAUDE.md "Never silently adopt a candidate
model variant as the default").

## 2025 protection

`assert_seasons_allowed` is called on the exact season set this script
trains/selects/evaluates against, and `_validate_no_final_test_seasons`
independently re-checks every season value actually present in the loaded
input file against `SCRIPT_SEASONS` (2021-2024) before anything else runs.
There is deliberately NO `--allow-final-evaluation` flag here -- unlike the
per-model training CLIs, this reconciliation script has no legitimate reason
to ever touch the protected final-test season, so the escape hatch simply
does not exist for it.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlb_luck_score.config import (
    TABLES_DIR,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    assert_seasons_allowed,
)
from mlb_luck_score.eligibility import (
    add_advancement_eligibility,
    add_infield_opportunity_eligibility,
    add_outfield_opportunity_eligibility,
    compute_eligibility,
)
from mlb_luck_score.features.build_contact_features import (
    ADVANCEMENT_CONTACT_PROBABILITY_FEATURES,
    add_advancement_features,
    add_opportunity_features_by_domain,
)
from mlb_luck_score.models.compare_advancement_models import (
    fit_advancement_contact_model,
    get_advancement_rows,
    run_advancement_model_selection,
)
from mlb_luck_score.models.compare_infield_opportunity import run_infield_model_selection
from mlb_luck_score.models.train_contact_model import train_model
from mlb_luck_score.models.train_opportunity_model import train_opportunity_model
from mlb_luck_score.scoring.attribution_ledger import (
    STATUS_CONTACT_RESULT_UNAVAILABLE,
    STATUS_CONTACT_RESULT_UNAVAILABLE_OTHER,
    build_attribution_ledger,
    verify_attribution_identity,
)

logger = logging.getLogger(__name__)

#: This script only ever trains/scores these seasons -- there is no flag to
#: widen this set. Checked directly against the input file's own `season`
#: column, independent of (and in addition to) `assert_seasons_allowed`.
SCRIPT_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024)

#: Provisional threshold (runs) for "materially large residual" reporting --
#: a documented placeholder for this report, like every other provisional
#: constant in this codebase, NOT a validated statistical bound.
LARGE_RESIDUAL_RUNS_THRESHOLD = 2.0


class RunAttributionLedgerError(ValueError):
    """Raised when this script's own preconditions (season scope, required
    upstream files) are not met."""


def _validate_no_final_test_seasons(df: pd.DataFrame) -> list[int]:
    observed = sorted(int(s) for s in df["season"].dropna().unique().tolist())
    outside = [s for s in observed if s not in SCRIPT_SEASONS]
    if outside:
        raise RunAttributionLedgerError(
            f"Input contains season(s) outside {SCRIPT_SEASONS}: {outside} -- refusing to "
            "proceed (this script never touches the protected final-test season)."
        )
    return observed


def _read_existing_gate_status(path: Path, *, status_key: tuple[str, ...]) -> str:
    """Cite an already-computed real calibration verdict from a `compare_*.py`
    CLI's saved JSON output -- see module docstring for why this script does
    not re-derive that machinery itself.
    """
    if not path.exists():
        return f"not_available_missing_file:{path}"
    try:
        data: Any = json.loads(path.read_text())
    except json.JSONDecodeError:
        return f"not_available_unreadable_file:{path}"
    node = data
    for key in status_key:
        if not isinstance(node, dict) or key not in node:
            return f"not_available_missing_key_path:{'.'.join(status_key)}_in:{path}"
        node = node[key]
    return f"{node} (see {path.name})"


def build_ledger_from_real_data(
    input_path: Path,
    *,
    output_dir: Path,
    train_seasons: tuple[int, ...] = TRAIN_SEASONS,
    validation_seasons: tuple[int, ...] = VALIDATION_SEASONS,
) -> dict[str, Any]:
    """Train every component model and build the real-data attribution ledger.

    Returns the full report dict (see `summarize_ledger_run`); also the
    function a maintainer would call directly from a notebook/REPL.
    """
    assert_seasons_allowed(tuple(train_seasons) + tuple(validation_seasons))

    df = pd.read_parquet(input_path)
    observed_seasons = _validate_no_final_test_seasons(df)
    logger.info(
        "Loaded %d rows from %s; seasons observed: %s", len(df), input_path, observed_seasons
    )

    df = compute_eligibility(df)
    df = add_outfield_opportunity_eligibility(df)
    df = add_infield_opportunity_eligibility(df)
    df = add_advancement_eligibility(df)
    df = add_opportunity_features_by_domain(df)
    df = add_advancement_features(df)

    contact_train_df = df[
        df["eligible_for_training"].fillna(False) & df["season"].isin(train_seasons)
    ]
    contact_trained = train_model(contact_train_df, class_weight=None)
    logger.info(
        "Contact model (baseline_v02) trained on %d rows, seasons %s",
        len(contact_train_df),
        train_seasons,
    )

    outfield_train_df = df[
        df["outfield_opportunity_eligible"].astype(bool) & df["season"].isin(train_seasons)
    ]
    outfield_trained = train_opportunity_model(outfield_train_df, class_weight=None)
    logger.info("Outfield opportunity model trained on %d rows", len(outfield_train_df))

    infield_elig = df[df["infield_opportunity_eligible"].astype(bool)]
    infield_winner, infield_selection_metrics, infield_trained_by_candidate = (
        run_infield_model_selection(infield_elig)
    )
    infield_trained = infield_trained_by_candidate[infield_winner]
    logger.info("Infield opportunity model selection winner: %s", infield_winner)

    advancement_contact_trained = fit_advancement_contact_model(df)
    advancement_elig = get_advancement_rows(df, advancement_contact_trained)
    advancement_winner, advancement_selection_metrics, advancement_trained_by_candidate = (
        run_advancement_model_selection(advancement_elig)
    )
    advancement_trained = advancement_trained_by_candidate[advancement_winner]
    logger.info("Advancement model selection winner: %s", advancement_winner)

    prob_cols = list(ADVANCEMENT_CONTACT_PROBABILITY_FEATURES)
    for col in prob_cols:
        df[col] = np.nan
    df.loc[advancement_elig.index, prob_cols] = advancement_elig[prob_cols]

    outfield_confidence_status = _read_existing_gate_status(
        output_dir / "opportunity_model_comparison_detail.json",
        status_key=(
            "validation_summary",
            "per_candidate",
            "measured_contact_only_v07",
            "passes_basic_validation",
        ),
    )
    infield_confidence_status = _read_existing_gate_status(
        output_dir / "infield_opportunity_detail.json",
        status_key=("gate_summary", "overall_status"),
    )
    advancement_confidence_status = _read_existing_gate_status(
        output_dir / "advancement_detail.json",
        status_key=("gate_summary", "overall_status"),
    )

    scoring_df = df[df["season"].isin(validation_seasons)].copy()
    logger.info("Scoring %d rows from seasons %s", len(scoring_df), validation_seasons)

    ledger = build_attribution_ledger(
        scoring_df,
        contact_trained,
        outfield_trained=outfield_trained,
        infield_trained=infield_trained,
        advancement_trained=advancement_trained,
        outfield_confidence_status=outfield_confidence_status,
        infield_confidence_status=infield_confidence_status,
        advancement_confidence_status=advancement_confidence_status,
    )

    return summarize_ledger_run(
        scoring_df,
        ledger,
        train_seasons=train_seasons,
        validation_seasons=validation_seasons,
        observed_seasons=observed_seasons,
        infield_winner=infield_winner,
        advancement_winner=advancement_winner,
        outfield_confidence_status=outfield_confidence_status,
        infield_confidence_status=infield_confidence_status,
        advancement_confidence_status=advancement_confidence_status,
    )


def summarize_ledger_run(
    scoring_df: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    train_seasons: tuple[int, ...],
    validation_seasons: tuple[int, ...],
    observed_seasons: list[int],
    infield_winner: str,
    advancement_winner: str,
    outfield_confidence_status: str,
    infield_confidence_status: str,
    advancement_confidence_status: str,
) -> dict[str, Any]:
    """Build the full real-data report dict described in this module's docstring."""
    holds = verify_attribution_identity(ledger)
    resolved_mask = holds.notna()

    lhs = ledger["observed_final_run_value"] - ledger["baseline_expected_contact_run_value"]
    rhs = (
        ledger["unexplained_residual"]
        + ledger["defensive_execution_contribution"].fillna(0.0)
        + ledger["advancement_execution_contribution"].fillna(0.0)
    )
    abs_error = (lhs - rhs).abs()
    resolved_abs_error = abs_error[resolved_mask]

    percentiles = [50, 90, 95, 99, 100]
    reconciliation_error = {
        "resolved_row_count": int(resolved_mask.sum()),
        "identity_holds_for_every_resolved_row": bool(holds[resolved_mask].all()),
        "max_abs_error": float(resolved_abs_error.max()) if len(resolved_abs_error) else None,
        "percentile_abs_error": {
            str(p): float(np.percentile(resolved_abs_error, p)) if len(resolved_abs_error) else None
            for p in percentiles
        },
    }

    outfield_mask = scoring_df["outfield_opportunity_eligible"].astype(bool)
    infield_mask = scoring_df["infield_opportunity_eligible"].astype(bool)
    both_domains_mask = outfield_mask & infield_mask
    neither_domain_mask = ~outfield_mask & ~infield_mask
    advancement_mask = scoring_df["advancement_eligible"].astype(bool)
    error_mask = scoring_df["events"] == "field_error"

    domain_routing = {
        "outfield_eligible_rows": int(outfield_mask.sum()),
        "infield_eligible_rows": int(infield_mask.sum()),
        "eligible_for_both_domains_simultaneously": int(both_domains_mask.sum()),
        "excluded_from_both_domains_rows": int(neither_domain_mask.sum()),
        "advancement_eligible_rows": int(advancement_mask.sum()),
    }

    pathway_counts = ledger["component_eligibility_status"].value_counts().to_dict()
    confidence_status_counts = ledger["component_confidence_status"].value_counts().to_dict()

    field_error = {
        "row_count": int(error_mask.sum()),
        "all_marked_contact_result_unavailable": bool(
            (
                ledger.loc[error_mask, "component_eligibility_status"]
                == STATUS_CONTACT_RESULT_UNAVAILABLE
            ).all()
        )
        if error_mask.any()
        else None,
        "all_core_columns_null_for_error_rows": bool(
            ledger.loc[
                error_mask,
                [
                    "observed_contact_result_run_value",
                    "contact_result_surprise",
                    "observed_final_run_value",
                    "unexplained_residual",
                ],
            ]
            .isna()
            .all()
            .all()
        )
        if error_mask.any()
        else None,
        "excluded_from_identity_check": bool(holds[error_mask].isna().all())
        if error_mask.any()
        else None,
    }

    other_unresolved_mask = (~resolved_mask) & (~error_mask)
    other_unresolved = {
        "row_count": int(other_unresolved_mask.sum()),
        "all_marked_unavailable_other": bool(
            (
                ledger.loc[other_unresolved_mask, "component_eligibility_status"]
                == STATUS_CONTACT_RESULT_UNAVAILABLE_OTHER
            ).all()
        )
        if other_unresolved_mask.any()
        else None,
    }

    residual = ledger.loc[resolved_mask, "unexplained_residual"]
    residual_distribution = {
        "count": int(residual.count()),
        "mean": float(residual.mean()) if len(residual) else None,
        "std": float(residual.std()) if len(residual) else None,
        "min": float(residual.min()) if len(residual) else None,
        "max": float(residual.max()) if len(residual) else None,
        "percentile": {
            str(p): float(np.percentile(residual, p)) if len(residual) else None
            for p in percentiles
        },
    }

    large_residual_mask = resolved_mask & (
        ledger["unexplained_residual"].abs() >= LARGE_RESIDUAL_RUNS_THRESHOLD
    )
    large_residual_rows = ledger.loc[large_residual_mask].copy()
    large_residual_rows["abs_unexplained_residual"] = large_residual_rows[
        "unexplained_residual"
    ].abs()
    top_large_residuals = (
        large_residual_rows.sort_values("abs_unexplained_residual", ascending=False)
        .head(20)[["unexplained_residual", "component_eligibility_status"]]
        .assign(event_id=scoring_df.loc[large_residual_rows.index, "event_id"])
        .to_dict(orient="records")
    )
    large_residuals = {
        "threshold_runs": LARGE_RESIDUAL_RUNS_THRESHOLD,
        "row_count": int(large_residual_mask.sum()),
        "top_20_by_absolute_value": top_large_residuals,
    }

    weather_alignment_check = {
        "any_weather_or_alignment_columns_in_ledger": bool(
            any("weather" in c or "alignment" in c for c in ledger.columns)
        ),
    }

    return {
        "seasons": {
            "train_seasons": list(train_seasons),
            "validation_seasons": list(validation_seasons),
            "observed_in_input": observed_seasons,
            "final_test_season_touched": False,
        },
        "model_selection_winners": {
            "outfield": "measured_contact_only_v07 (single candidate, no selection)",
            "infield": infield_winner,
            "advancement": advancement_winner,
        },
        "component_confidence_status": {
            "outfield": outfield_confidence_status,
            "infield": infield_confidence_status,
            "advancement": advancement_confidence_status,
        },
        "reconciliation_error": reconciliation_error,
        "domain_routing": domain_routing,
        "pathway_counts": pathway_counts,
        "confidence_status_counts": confidence_status_counts,
        "field_error_handling": field_error,
        "other_unresolved_rows": other_unresolved,
        "residual_distribution": residual_distribution,
        "large_residuals": large_residuals,
        "weather_alignment_check": weather_alignment_check,
        "total_scored_rows": int(len(scoring_df)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/cleaned_development_data_with_sprint_speed.parquet"),
        help="Cleaned development data joined with venue+geometry+sprint speed",
    )
    parser.add_argument("--output-dir", type=Path, default=TABLES_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    report = build_ledger_from_real_data(args.input, output_dir=args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "attribution_ledger_v010_real_data_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    logger.info("max_abs_reconciliation_error=%s", report["reconciliation_error"]["max_abs_error"])
    logger.info(
        "identity_holds_for_every_resolved_row=%s",
        report["reconciliation_error"]["identity_holds_for_every_resolved_row"],
    )
    logger.info("Saved report to %s", report_path)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
