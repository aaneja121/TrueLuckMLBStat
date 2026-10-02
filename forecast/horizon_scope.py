"""Contact Forecast: horizon scoping study (100 -> 50 / 100 / 200), development only.

## What this is, and what it is NOT

**EXPLORATORY SCOPING. Not a frozen result, and not a prespecified cell.**

R1's frozen window grid used horizons of 50 and 100 only. A 200-BBE horizon
was never prespecified, so nothing here can be reported as a confirmatory
finding, and no conclusion from it may be attached to the frozen R1, ridge or
HGB packages. Its purpose is narrower and stated up front: decide whether a
100 -> 200 window is worth PRESPECIFYING for a future stage, by measuring
three things the earlier work could not.

  1. **Ceiling.** A longer target window averages away more outcome noise, so
     the maximum achievable R-squared rises. That ceiling is computed from the
     frozen shrinkage variance components, not fitted here.
  2. **Effect size.** If the ceiling rises, the separation between predictors
     should widen. R1 already showed the 50 -> 100 step doing exactly that.
  3. **Cost.** A longer horizon requires more batted balls, so fewer
     hitter-seasons qualify. The trade is only worth taking if the effect
     grows faster than the sample shrinks -- and because the sample size
     needed to resolve a difference scales with 1/delta^2, a doubling of the
     effect pays for a fourfold loss of windows.

## Guards

Development seasons only (2022-2024). 2025 is sealed and unreachable; 2026 is
the held-out season and is NOT touched -- this study exists precisely so that
a future 2026 evaluation can be prespecified without having looked at it.

Every frozen module is reused unmodified: `forecast.assemble.
build_prediction_table` already accepts the horizon, so no new window,
feature, shrinkage or ladder code exists here.

## The ridge at each horizon is REFIT, not the frozen H=100 model

Stated plainly because the distinction decides how these numbers may be used:
`_attach_ridge` calls `fit_ridge` against EACH horizon's own target, producing
fresh coefficients, a fresh intercept and fresh standardization statistics on
the subset of hitters who reached that horizon. Only the feature LIST and the
alpha VALUE come from the frozen ridge.

Consequently the H=200 row is NOT "the frozen Contact Forecast measured over a
longer window". It is a different fitted model, and it is therefore a NEW
FORECAST SPECIFICATION that must be frozen on development data before any 2026
H=200 outcome is opened. `forecast.phase2.h200_spec` does that.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast.assemble import build_prediction_table
from forecast.bootstrap import BootstrapDesign, run_paired_bootstrap
from forecast.forecast_config import (
    DEFAULT_BOOTSTRAP_ALPHA,
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
    FORECAST_ANALYSIS_SEASONS,
    FORECAST_DATA_DIR,
    FORECAST_OUTPUTS_DIR,
    PRIMARY_COMPARISON,
    PROJECT_ROOT,
    SEASON_FORWARD_FOLDS,
)
from forecast.metrics import assert_delta_convention, paired_delta
from forecast.ridge import fit_ridge
from forecast.ridge import predict as ridge_predict

logger = logging.getLogger(__name__)

SCOPE_CUTOFF = 100
SCOPE_HORIZONS: tuple[int, ...] = (50, 100, 200)
TARGET = "target_realized_rv_per_100"
REFERENCE, CHALLENGER = PRIMARY_COMPARISON
BENCHMARK = "shrunk_deserved_persistence"
RIDGE_MODEL = "D_full_contact_profile"
RIDGE_COLUMN = "ridge_forecast"

SCOPE_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "forecast_horizon_scope"

#: Two-sided 95% normal-approximation multiplier, for the sample-size sums.
_Z = 1.96


class HorizonScopeError(RuntimeError):
    """Raised when the scoping study cannot run as specified."""


def theoretical_ceiling(
    *, between_variance: float, within_variance: float, horizon: int
) -> dict[str, float]:
    """Max achievable R-squared for a target averaged over `horizon` batted balls.

    Signal is the between-hitter variance of true rates; noise is the
    per-event variance divided by the horizon length. Both come from the
    frozen shrinkage fit -- nothing is estimated here.
    """
    noise = within_variance / float(horizon)
    return {
        "signal_variance": between_variance,
        "noise_variance": noise,
        "max_achievable_r_squared": between_variance / (between_variance + noise),
    }


def required_sample_size(paired_differences: np.ndarray, *, delta: float) -> float | None:
    """Hitters needed to resolve `delta` at 95%, given the observed paired spread.

    `n = (z * SD / delta)^2`. The point of reporting it per horizon is that it
    scales with 1/delta^2, so an effect that doubles costs a quarter of the
    sample -- which is what decides whether a longer, sparser window pays.
    """
    if delta == 0.0:
        return None
    sd = float(np.std(paired_differences, ddof=1))
    return float((_Z * sd / abs(delta)) ** 2)


def _paired_absolute_differences(
    frame: pd.DataFrame, *, challenger: str, reference: str
) -> np.ndarray:
    y = frame[TARGET].to_numpy(dtype=float)
    return np.abs(frame[challenger].to_numpy(dtype=float) - y) - np.abs(
        frame[reference].to_numpy(dtype=float) - y
    )


def _r_squared(frame: pd.DataFrame, predictor: str) -> float:
    y = frame[TARGET].to_numpy(dtype=float)
    p = frame[predictor].to_numpy(dtype=float)
    ss_total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - p) ** 2)) / ss_total if ss_total > 0 else float("nan")


def _comparison(
    frame: pd.DataFrame,
    *,
    challenger: str,
    reference: str,
    design: BootstrapDesign,
    label: str,
) -> dict[str, Any]:
    """One paired comparison with its interval, spread and sample-size implication."""
    paired = frame.dropna(subset=[TARGET, challenger, reference]).copy()
    if paired.empty:
        return {"status": "no_paired_windows", "label": label}

    delta = paired_delta(paired, target=TARGET, reference=reference, challenger=challenger)
    assert_delta_convention(delta)
    bootstrap = run_paired_bootstrap(
        paired, target=TARGET, reference=reference, challenger=challenger, design=design
    )
    cells: dict[str, Any] = bootstrap["cells"]  # type: ignore[assignment]
    horizon = int(paired["horizon"].iloc[0])
    summary = cells[f"K{SCOPE_CUTOFF}_H{horizon}"]["pooled"]["delta_mae"]
    differences = _paired_absolute_differences(paired, challenger=challenger, reference=reference)
    return {
        "label": label,
        "challenger": challenger,
        "reference": reference,
        "seasons": sorted(int(s) for s in paired["season"].unique()),
        "n_windows": int(len(paired)),
        "mae_challenger": delta.mae_challenger,
        "mae_reference": delta.mae_reference,
        "delta_mae": delta.delta_mae,
        "pct_change_mae": delta.pct_change_mae,
        "delta_rmse": delta.delta_rmse,
        "pct_change_rmse": delta.pct_change_rmse,
        "direction_mae": delta.direction_mae,
        "ci_delta_mae": [summary["ci_lower"], summary["ci_upper"]],
        "ci_crosses_zero": summary["ci_crosses_zero"],
        "share_replicates_favouring_reference": summary["share_replicates_favouring_reference"],
        "paired_difference_sd": float(np.std(differences, ddof=1)),
        "required_n_to_resolve_this_delta": required_sample_size(
            differences, delta=delta.delta_mae
        ),
        "r_squared_challenger": _r_squared(paired, challenger),
        "r_squared_reference": _r_squared(paired, reference),
    }


def run_scope(
    *,
    seasons: tuple[int, ...] = FORECAST_ANALYSIS_SEASONS,
    horizons: tuple[int, ...] = SCOPE_HORIZONS,
    data_dir: Path = FORECAST_DATA_DIR,
    research_dir: Path = FORECAST_OUTPUTS_DIR,
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = DEFAULT_BOOTSTRAP_ALPHA,
) -> dict[str, Any]:
    """Build every horizon's windows and measure both comparisons at each."""
    if 2025 in seasons or 2026 in seasons:
        raise HorizonScopeError(
            "This scoping study is development-only: 2025 is sealed and 2026 is the "
            "held-out season, which must not be inspected before a prespecification."
        )

    ridge_manifest = json.loads((research_dir / "ridge_freeze_manifest.json").read_text())
    frozen_alpha = float(ridge_manifest["key_results"]["selected_alphas"][RIDGE_MODEL])

    table, audit = build_prediction_table(
        seasons=seasons, data_dir=data_dir, cutoffs=(SCOPE_CUTOFF,), horizons=horizons
    )
    shrinkage = audit["ladders"][str(max(seasons))]
    design = BootstrapDesign(reps=reps, seed=seed, alpha=alpha)

    results: dict[str, Any] = {}
    for horizon in horizons:
        cell = table[table["horizon"].astype(int) == int(horizon)].copy()
        if cell.empty:
            results[f"H{horizon}"] = {"status": "no_windows"}
            continue

        ceiling = theoretical_ceiling(
            between_variance=shrinkage["deserved_shrinkage"]["between_hitter_variance"],
            within_variance=shrinkage["realized_shrinkage"]["within_hitter_variance"],
            horizon=horizon,
        )
        cell = _attach_ridge(cell, frozen_alpha=frozen_alpha)
        evaluation = cell[cell["season"].astype(int) == SEASON_FORWARD_FOLDS[-1][1]]

        results[f"H{horizon}"] = {
            "cutoff": SCOPE_CUTOFF,
            "horizon": int(horizon),
            "n_windows_total": int(len(cell)),
            "windows_by_season": {
                str(s): int(n) for s, n in cell.groupby(cell["season"].astype(int)).size().items()
            },
            "target_sd": float(cell[TARGET].std(ddof=1)),
            "theoretical_ceiling": ceiling,
            "luck_adjustment_comparison": _comparison(
                cell,
                challenger=CHALLENGER,
                reference=REFERENCE,
                design=design,
                label="shrunk deserved vs shrunk realized (R1's question)",
            ),
            "forecast_layer_comparison": _comparison(
                evaluation,
                challenger=RIDGE_COLUMN,
                reference=BENCHMARK,
                design=design,
                label=(
                    "frozen Model D ridge vs shrunk deserved, evaluated on "
                    f"{SEASON_FORWARD_FOLDS[-1][1]} only"
                ),
            ),
            "empirical_r_squared_on_evaluation_season": {
                name: _r_squared(evaluation.dropna(subset=[TARGET, name]), name)
                for name in (RIDGE_COLUMN, BENCHMARK, "shrunk_realized_persistence")
                if evaluation[name].notna().any()
            },
        }
        logger.info(
            "H=%d: %d windows, ceiling R2 %.3f",
            horizon,
            len(cell),
            ceiling["max_achievable_r_squared"],
        )

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "study": "contact_forecast_horizon_scope_v1",
        "status": (
            "EXPLORATORY SCOPING -- not a frozen result, not a prespecified cell. "
            "R1's frozen grid used horizons 50 and 100 only; 200 is new here and no "
            "confirmatory claim may rest on it."
        ),
        "purpose": (
            "decide whether a 100 -> 200 window is worth PRESPECIFYING for a future "
            "stage, by measuring ceiling, effect size and sample cost together"
        ),
        "seasons": list(seasons),
        "seasons_not_touched": {"2025": "sealed", "2026": "held out"},
        "ridge_provenance": {
            "model_was_refit_per_horizon": True,
            "frozen_h100_coefficients_reused": False,
            "frozen_h100_preprocessing_reused": False,
            "inherited_from_the_frozen_ridge": ["feature list", "alpha value"],
            "estimated_fresh_per_horizon": [
                "coefficients",
                "intercept",
                "StandardScaler mean and scale",
            ],
            "statement": (
                "Each horizon's ridge is a DIFFERENT fitted model. The H=200 row is not "
                "the frozen H=100 Contact Forecast measured over a longer window."
            ),
        },
        "frozen_elements_reused": {
            "ridge_model": RIDGE_MODEL,
            "ridge_alpha": frozen_alpha,
            "alpha_source": "ridge freeze manifest; not retuned for any horizon",
            "windows_features_shrinkage_ladder": "forecast.assemble, unmodified",
        },
        "shrinkage_variance_components": {
            "deserved_between_hitter_variance": shrinkage["deserved_shrinkage"][
                "between_hitter_variance"
            ],
            "realized_within_hitter_variance": shrinkage["realized_shrinkage"][
                "within_hitter_variance"
            ],
            "source_season_fold": str(max(seasons)),
        },
        "by_horizon": results,
    }


def _attach_ridge(cell: pd.DataFrame, *, frozen_alpha: float) -> pd.DataFrame:
    """REFIT Model D on this horizon's target, season-forward: train 2022-2023.

    **This is a REFIT, not the frozen H=100 model applied to a new target.**
    `fit_ridge` estimates fresh coefficients, a fresh intercept and fresh
    StandardScaler statistics against THIS horizon's target column, on the
    subset of hitters who reached `cutoff + horizon`. Only the feature LIST and
    the alpha VALUE are inherited from the frozen ridge.

    Measured at H=200 versus H=100: coefficients differ, scaler means differ,
    intercept moves 5.218 -> 5.817, and the training set shrinks 564 -> 336
    rows because a longer horizon qualifies fewer hitters. So each horizon's
    number describes a DIFFERENT fitted model, which is why H=200 is a new
    specification requiring its own freeze rather than a new view of an
    existing one.
    """
    (_, _), (train_seasons, evaluate_season) = SEASON_FORWARD_FOLDS
    train = cell[cell["season"].astype(int).isin(train_seasons)]
    later = [s for s in train_seasons if int(s) >= int(evaluate_season)]
    if later:
        raise HorizonScopeError(f"Training seasons {later} are not strictly earlier")
    if train.empty:
        raise HorizonScopeError("No training windows for the ridge at this horizon")

    fit = fit_ridge(train, model_name=RIDGE_MODEL, alpha=frozen_alpha)
    out = cell.copy()
    out[RIDGE_COLUMN] = np.nan
    mask = out["season"].astype(int) == int(evaluate_season)
    out.loc[mask, RIDGE_COLUMN] = ridge_predict(fit, out[mask])
    return out


def render_report(results: dict[str, Any]) -> str:
    """A compact comparison of the three horizons."""
    lines = [
        "# Contact Forecast -- horizon scoping study (100 -> 50 / 100 / 200)",
        "",
        f"Generated {results['generated_at_utc']}.",
        "",
        f"**{results['status']}**",
        "",
        f"Purpose: {results['purpose']}.",
        "",
        f"Development seasons {results['seasons']} only. 2025 sealed, 2026 held out and "
        "not inspected.",
        "",
        "**The ridge at each horizon is REFIT on that horizon's target.** Fresh "
        "coefficients, intercept and standardization statistics are estimated per "
        "horizon; only the feature LIST and the alpha VALUE "
        f"({results['frozen_elements_reused']['ridge_alpha']}, from "
        f"{results['frozen_elements_reused']['alpha_source']}) are inherited. The "
        "H=200 row is therefore a DIFFERENT fitted model, not the frozen H=100 "
        "Contact Forecast measured over a longer window.",
        "",
        "The window, feature, shrinkage and ladder pipeline is the frozen one, unmodified.",
        "",
        "## The trade, in one table",
        "",
        "| horizon | windows (3 seasons) | ceiling R2 | luck-adjustment delta_MAE | forecast-layer delta_MAE |",
        "|---|---|---|---|---|",
    ]
    for key in sorted(results["by_horizon"], key=lambda k: int(k[1:])):
        entry = results["by_horizon"][key]
        if entry.get("status") == "no_windows":
            continue
        luck = entry["luck_adjustment_comparison"]
        fore = entry["forecast_layer_comparison"]
        lines.append(
            f"| 100 -> {entry['horizon']} | {entry['n_windows_total']} | "
            f"{entry['theoretical_ceiling']['max_achievable_r_squared']:.3f} | "
            f"{luck['delta_mae']:+.4f} ({luck['pct_change_mae']:+.2f}%) | "
            f"{fore['delta_mae']:+.4f} ({fore['pct_change_mae']:+.2f}%) |"
        )
    lines.append("")

    for key in sorted(results["by_horizon"], key=lambda k: int(k[1:])):
        entry = results["by_horizon"][key]
        if entry.get("status") == "no_windows":
            continue
        lines += [f"## Horizon 100 -> {entry['horizon']}", ""]
        lines += [
            f"- Windows: {entry['n_windows_total']} across "
            f"{entry['windows_by_season']}; target SD {entry['target_sd']:.3f}.",
            f"- Theoretical ceiling: max R2 "
            f"{entry['theoretical_ceiling']['max_achievable_r_squared']:.3f} "
            f"(signal {entry['theoretical_ceiling']['signal_variance']:.2f} vs noise "
            f"{entry['theoretical_ceiling']['noise_variance']:.2f}).",
            "",
            "| comparison | n | delta_MAE | 95% CI | pct | paired SD | n needed to resolve |",
            "|---|---|---|---|---|---|---|",
        ]
        for which in ("luck_adjustment_comparison", "forecast_layer_comparison"):
            c = entry[which]
            if c.get("status") == "no_paired_windows":
                lines.append(f"| {which} | 0 | n/a | n/a | n/a | n/a | n/a |")
                continue
            required = c["required_n_to_resolve_this_delta"]
            lines.append(
                f"| {c['label']} | {c['n_windows']} | {c['delta_mae']:+.4f} | "
                f"[{c['ci_delta_mae'][0]:+.4f}, {c['ci_delta_mae'][1]:+.4f}]"
                + ("" if c["ci_crosses_zero"] else " **excludes 0**")
                + f" | {c['pct_change_mae']:+.2f}% | {c['paired_difference_sd']:.3f} | "
                + (f"{required:,.0f}" if required else "n/a")
                + " |"
            )
        lines += [
            "",
            "Empirical R2 on the evaluation season: "
            + ", ".join(
                f"`{k}` {v:.3f}"
                for k, v in sorted(entry["empirical_r_squared_on_evaluation_season"].items())
            )
            + ".",
            "",
        ]
    lines += [
        "## How to read this",
        "",
        "The decision is not which horizon has the biggest delta. It is whether the "
        "effect grows FASTER than the sample shrinks. Sample size needed to resolve a "
        "difference scales with 1/delta^2, so an effect that doubles pays for a "
        "fourfold loss of windows; an effect that grows 20% while the sample halves "
        "does not.",
        "",
        "Nothing here is confirmatory. A 100 -> 200 window would still have to be "
        "prespecified, frozen and tested on a season that has not been looked at.",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=SCOPE_OUTPUTS_DIR)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    results = run_scope(reps=args.bootstrap_reps)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    (args.outputs_dir / "horizon_scope_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str)
    )
    (args.outputs_dir / "horizon_scope_report.md").write_text(render_report(results))
    print(f"wrote {args.outputs_dir}/horizon_scope_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
