"""Version 0.14: the frozen replication questions A-G and the package-level
classification, computed from an already-scored season.

This module opens no season. It takes a `SeasonAggregationArtifacts` the
caller produced and answers exactly the questions
`pitcher_replication_spec.REPLICATION_QUESTIONS` preregistered -- no more.

## Every estimator is imported, never restated

Question E uses `pitcher_replication_estimators`; question F uses
`pitcher_split_half`. Both are in the Version 0.14 frozen source set, so a
change to either invalidates the freeze rather than silently changing what
the replication measures. Nothing here reimplements a formula.

## Nothing here may be added to after seeing 2025

The report shape is fixed by the frozen spec. A metric that "looks
interesting in 2025" is exactly what a pre-registration exists to exclude.
`assert_report_covers_every_question` is the mechanical check that the
output answers all seven and only the seven.

## Classification is mechanical and package-level

`classify` reads each question's `agrees` verdict and applies the frozen
rule in `pitcher_replication_spec.CLASSIFICATION_RULE`. It never inspects a
p-value, never weights one correlation above the others, and reports EVERY
disagreement including under a REPLICATED verdict.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pitcher_replication_estimators import (
    PRACTICAL_WORKLOAD_FLOORS,
    rate_precision_report,
    resolving_power_report,
)
from pitcher_replication_spec import (
    BOARD_DISPLAY_MINIMUM_BBE,
    CLASSIFICATION_VALUES,
    REPLICATION_QUESTIONS,
    ROLE_RELIEVER_LIKE_MAX_BBE_PER_APPEARANCE,
    ROLE_STARTER_LIKE_MIN_BBE_PER_APPEARANCE,
)
from pitcher_split_half import build_question_f_report

QUESTION_KEYS: tuple[str, ...] = tuple(REPLICATION_QUESTIONS)

#: Sanity-audit buckets for question G. Fixed here, before 2025, so the
#: "representative" rows cannot be chosen to flatter a result.
SANITY_AUDIT_BUCKETS: tuple[str, ...] = (
    "extreme_favorable_total",
    "extreme_unfavorable_total",
    "high_workload",
    "reliever_like",
    "near_zero_total",
)

#: How many largest-|contribution| plays each audited season shows.
SANITY_AUDIT_PLAYS_PER_SEASON = 3


class QuestionError(ValueError):
    """Raised when a question cannot be answered from the supplied rows."""


def role_bucket(bbe: int, appearances: int) -> str:
    """The frozen DESCRIPTIVE usage bucket. Not role metadata; see
    `pitcher_replication_spec.ROLE_LIKE_GROUPING`.
    """
    if appearances <= 0:
        return "ambiguous"
    per_appearance = bbe / appearances
    if per_appearance >= ROLE_STARTER_LIKE_MIN_BBE_PER_APPEARANCE:
        return "starter_like"
    if per_appearance <= ROLE_RELIEVER_LIKE_MAX_BBE_PER_APPEARANCE:
        return "reliever_like"
    return "ambiguous"


def _describe(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    return {
        "n": int(len(values)),
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "median": float(values.median()),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
    }


def _corr_pair(left: pd.Series, right: pd.Series) -> dict[str, float | None]:
    if len(left) < 2:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": _opt(left.corr(right)),
        "spearman": _opt(left.corr(right, method="spearman")),
    }


def _opt(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), 4)


# ---------------------------------------------------------------------------
# A -- population and centering
# ---------------------------------------------------------------------------


def question_a(frame: pd.DataFrame, play_level_mean: float) -> dict[str, Any]:
    """Population size, exposure distribution, and where the distributions
    are centred. Descriptive; no exact 2024 number is required to replicate.
    """
    total = frame["cumulative_contact_luck_runs"]
    agrees = bool(abs(play_level_mean) < 0.05 and len(frame) > 0)
    return {
        "question": "A_population_and_centering",
        "pitcher_season_rows": int(len(frame)),
        "resolved_bbe_distribution": _describe(frame["eligible_batted_balls"]),
        "play_level_mean_pitcher_contact_luck": round(float(play_level_mean), 6),
        "cumulative_total_distribution": _describe(total),
        "per_100_distribution": _describe(frame["contact_luck_per_100"]),
        "agrees": agrees,
        "hypothesis": REPLICATION_QUESTIONS["A_population_and_centering"]["structural_hypothesis"],
        "agreement_basis": (
            "Population is non-empty and the play-level mean sits within 0.05 runs of "
            "zero, i.e. the accounting is centred where the construction implies."
        ),
    }


# ---------------------------------------------------------------------------
# B -- opportunity heterogeneity
# ---------------------------------------------------------------------------


def question_b(frame: pd.DataFrame) -> dict[str, Any]:
    """Opportunity should drive MAGNITUDE while only weakly driving SIGN."""
    total = frame["cumulative_contact_luck_runs"]
    bbe = frame["eligible_batted_balls"]
    magnitude = _corr_pair(total.abs(), bbe)
    signed = _corr_pair(total, bbe)

    by_group: dict[str, Any] = {}
    for bucket in ("starter_like", "reliever_like", "ambiguous"):
        rows = frame[frame["role_bucket"] == bucket]
        by_group[bucket] = {
            "n": int(len(rows)),
            "resolved_bbe": _describe(rows["eligible_batted_balls"]),
        }

    agrees = bool(
        magnitude["pearson"] is not None
        and signed["pearson"] is not None
        and abs(magnitude["pearson"]) > abs(signed["pearson"])
    )
    return {
        "question": "B_opportunity_heterogeneity",
        "all_pitchers_resolved_bbe": _describe(bbe),
        "by_role_like_group": by_group,
        "abs_total_vs_bbe": magnitude,
        "signed_total_vs_bbe": signed,
        "agrees": agrees,
        "hypothesis": REPLICATION_QUESTIONS["B_opportunity_heterogeneity"]["structural_hypothesis"],
        "agreement_basis": (
            "|total| vs BBE correlates more strongly than signed total vs BBE, i.e. "
            "opportunity governs how large a season total can get without governing "
            "whether it is favorable."
        ),
    }


# ---------------------------------------------------------------------------
# C -- totals vs rate
# ---------------------------------------------------------------------------


def _rank_movers(rows: pd.DataFrame, top_n: int = 10) -> list[dict[str, Any]]:
    if len(rows) < 2:
        return []
    rank_total = rows["cumulative_contact_luck_runs"].rank(ascending=False, method="min")
    rank_rate = rows["contact_luck_per_100"].rank(ascending=False, method="min")
    delta = (rank_total - rank_rate).abs()
    worst = delta.sort_values(ascending=False).head(top_n).index
    return [
        {
            "pitcher_id": int(rows.loc[i, "pitcher_id"]),
            "name": rows.loc[i, "name"],
            "eligible_batted_balls": int(rows.loc[i, "eligible_batted_balls"]),
            "cumulative_contact_luck_runs": round(
                float(rows.loc[i, "cumulative_contact_luck_runs"]), 4
            ),
            "contact_luck_per_100": round(float(rows.loc[i, "contact_luck_per_100"]), 4),
            "rank_change": int(delta.loc[i]),
        }
        for i in worst
    ]


def question_c(frame: pd.DataFrame) -> dict[str, Any]:
    """Spearman(total, per-100) for all pitchers and each role-like group,
    plus the extreme rank movers, which is where the tiny-sample failure
    mode shows itself.
    """
    populations = {
        "all_pitchers": frame,
        "starter_like": frame[frame["role_bucket"] == "starter_like"],
        "reliever_like": frame[frame["role_bucket"] == "reliever_like"],
    }
    result: dict[str, Any] = {}
    for label, rows in populations.items():
        result[label] = {
            "n": int(len(rows)),
            "spearman_total_vs_per_100": (
                _opt(
                    rows["cumulative_contact_luck_runs"].corr(
                        rows["contact_luck_per_100"], method="spearman"
                    )
                )
                if len(rows) > 1
                else None
            ),
        }

    # The structural claim: an all-pitcher per-100 ordering is dominated at
    # its extremes by rows with almost no exposure.
    extremes = frame.reindex(frame["contact_luck_per_100"].abs().sort_values().index).tail(20)
    tiny = extremes[extremes["eligible_batted_balls"] < BOARD_DISPLAY_MINIMUM_BBE]
    agrees = bool(len(extremes) > 0 and len(tiny) > len(extremes) / 2)

    return {
        "question": "C_totals_vs_rate",
        "by_population": result,
        "per_100_extremes_n": int(len(extremes)),
        "per_100_extremes_below_board_minimum": int(len(tiny)),
        "per_100_extreme_share_below_board_minimum": (
            round(float(len(tiny) / len(extremes)), 4) if len(extremes) else None
        ),
        "extreme_rank_movers": _rank_movers(frame),
        "agrees": agrees,
        "hypothesis": REPLICATION_QUESTIONS["C_totals_vs_rate"]["structural_hypothesis"],
        "agreement_basis": (
            "More than half of the 20 most extreme per-100 seasons fall below the "
            "board display minimum, i.e. an all-pitcher rate ordering is led by "
            "tiny-sample rows whose cumulative impact is near zero."
        ),
    }


# ---------------------------------------------------------------------------
# D -- reliever-like single-play dominance
# ---------------------------------------------------------------------------


def question_d(frame: pd.DataFrame) -> dict[str, Any]:
    """How often one batted ball accounts for a large share of a
    reliever-like season's net total.
    """
    relievers = frame[frame["role_bucket"] == "reliever_like"]
    share = pd.to_numeric(relievers["largest_play_share_of_net"], errors="coerce").dropna()
    shares = {
        "share_over_25_pct": _opt((share > 0.25).mean()) if len(share) else None,
        "share_over_50_pct": _opt((share > 0.50).mean()) if len(share) else None,
        "share_over_100_pct": _opt((share > 1.00).mean()) if len(share) else None,
    }
    agrees = bool(shares["share_over_50_pct"] is not None and shares["share_over_50_pct"] > 0.25)
    return {
        "question": "D_reliever_single_play_dominance",
        "reliever_like_n": int(len(relievers)),
        "n_with_defined_share": int(len(share)),
        "n_with_undefined_share": int(len(relievers) - len(share)),
        **shares,
        "agrees": agrees,
        "hypothesis": REPLICATION_QUESTIONS["D_reliever_single_play_dominance"][
            "structural_hypothesis"
        ],
        "agreement_basis": (
            "More than a quarter of reliever-like seasons have one batted ball worth "
            "over half the net total, i.e. the phenomenon remains materially present."
        ),
    }


# ---------------------------------------------------------------------------
# E -- rate precision
# ---------------------------------------------------------------------------


def question_e(frame: pd.DataFrame) -> dict[str, Any]:
    """Interval half-width by workload, resolving power at the frozen
    practical floors, and the resolved BBE each precision target needs.

    The >=1 BBE floor is reported ONLY as an artifact disclosure, alongside
    its zero-width count and its excluding-zero-width companion. It is never
    read as evidence of signal.
    """
    floors = [resolving_power_report(frame, min_bbe=floor) for floor in PRACTICAL_WORKLOAD_FLOORS]
    artifact_raw = resolving_power_report(frame, min_bbe=1)
    artifact_guarded = resolving_power_report(frame, min_bbe=1, drop_zero_width=True)

    half_width_by_workload = []
    for low, high in ((1, 60), (60, 150), (150, 300), (300, 450), (450, 10**9)):
        rows = frame[
            (frame["eligible_batted_balls"] >= low) & (frame["eligible_batted_balls"] < high)
        ]
        if not len(rows):
            continue
        widths = (rows["per_100_ci_high"] - rows["per_100_ci_low"]) / 2.0
        half_width_by_workload.append(
            {
                "bbe_range": [low, None if high >= 10**9 else high],
                "n": int(len(rows)),
                "median_half_width_per_100": round(float(widths.median()), 4),
            }
        )

    powers = [f["resolving_power"] for f in floors if f.get("resolving_power") is not None]
    agrees = bool(powers and all(p < 1.0 for p in powers))

    return {
        "question": "E_rate_precision",
        "interval_half_width_by_workload": half_width_by_workload,
        "resolving_power_by_practical_floor": floors,
        "practical_floors": list(PRACTICAL_WORKLOAD_FLOORS),
        "rate_precision_requirements": rate_precision_report(frame),
        "max_observed_resolved_bbe": int(frame["eligible_batted_balls"].max()),
        "one_bbe_floor_artifact": {
            "raw": artifact_raw,
            "excluding_zero_width": artifact_guarded,
            "single_appearance_rows": int((frame["appearances"] == 1).sum()),
            "is_an_artifact_never_evidence_of_signal": True,
            "explanation": REPLICATION_QUESTIONS["E_rate_precision"]["mandatory_guard"],
        },
        "agrees": agrees,
        "hypothesis": REPLICATION_QUESTIONS["E_rate_precision"]["structural_hypothesis"],
        "agreement_basis": (
            "Resolving power is below 1 at every practical workload floor, so an "
            "ordering on the normalized rate is substantially an ordering of noise."
        ),
    }


# ---------------------------------------------------------------------------
# F -- split-half persistence (secondary)
# ---------------------------------------------------------------------------


def question_f(artifacts: Any) -> dict[str, Any]:
    """The frozen split-half procedure. SECONDARY: it cannot decide the
    classification on its own.
    """
    report = build_question_f_report(artifacts)
    correlations = [
        split["pearson_r"]
        for split in report["pitcher"].values()
        if split.get("pearson_r") is not None
    ]
    agrees = bool(correlations and all(abs(r) < 0.35 for r in correlations))
    return {
        "question": "F_persistence",
        **report,
        "agrees": agrees,
        "hypothesis": REPLICATION_QUESTIONS["F_persistence"]["structural_hypothesis"],
        "agreement_basis": (
            "Every split-half correlation is near zero (|r| < 0.35), which CONFIRMS the "
            "retrospective framing rather than indicting the metric."
        ),
    }


# ---------------------------------------------------------------------------
# G -- real-play sanity audit
# ---------------------------------------------------------------------------


def question_g(frame: pd.DataFrame) -> dict[str, Any]:
    """One audited season per frozen bucket, each showing its largest
    contributing batted balls with the contact that produced them.

    Qualitative by design: `agrees` is left None because a human must read
    the plays. The classification treats it as reported-not-decided.
    """
    on_board = frame[frame["eligible_batted_balls"] >= BOARD_DISPLAY_MINIMUM_BBE]
    picks: dict[str, Any] = {}
    if len(on_board):
        by_total = on_board.sort_values("cumulative_contact_luck_runs")
        picks["extreme_unfavorable_total"] = by_total.index[0]
        picks["extreme_favorable_total"] = by_total.index[-1]
        picks["high_workload"] = on_board["eligible_batted_balls"].idxmax()
        picks["near_zero_total"] = on_board["cumulative_contact_luck_runs"].abs().idxmin()
        relievers = on_board[on_board["role_bucket"] == "reliever_like"]
        if len(relievers):
            picks["reliever_like"] = relievers["cumulative_contact_luck_runs"].abs().idxmax()

    audited: dict[str, dict[str, Any] | None] = {}
    for bucket in SANITY_AUDIT_BUCKETS:
        if bucket not in picks:
            audited[bucket] = None
            continue
        row = frame.loc[picks[bucket]]
        audited[bucket] = {
            "pitcher_id": int(row["pitcher_id"]),
            "name": row["name"],
            "role_bucket": row["role_bucket"],
            "eligible_batted_balls": int(row["eligible_batted_balls"]),
            "appearances": int(row["appearances"]),
            "cumulative_contact_luck_runs": round(float(row["cumulative_contact_luck_runs"]), 4),
            "contact_luck_per_100": round(float(row["contact_luck_per_100"]), 4),
            "largest_favorable_play": row.get("largest_favorable_play"),
            "largest_unfavorable_play": row.get("largest_unfavorable_play"),
        }

    return {
        "question": "G_real_play_sanity_check",
        "buckets": list(SANITY_AUDIT_BUCKETS),
        "audited": audited,
        "is_qualitative": True,
        "agrees": None,
        "hypothesis": REPLICATION_QUESTIONS["G_real_play_sanity_check"]["structural_hypothesis"],
        "agreement_basis": (
            "Qualitative: a reader must confirm the largest contributions are "
            "baseball-sensible. Recorded as reported, never auto-scored."
        ),
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify(answers: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen package-level rule.

    Mechanical and deliberately blunt: it counts which prespecified
    structural hypotheses held, never inspecting a p-value and never
    weighting one statistic above the others. Question F is secondary and
    cannot by itself move the verdict; question G is qualitative and is
    reported rather than scored.

    - `REPLICATED`: every PRIMARY question (A-E) agrees.
    - `NO_GO`: the metric's own coherence failed -- A (centering) or E
      (rates are uncertainty-limited) disagreed, which would make the 2024
      design misleading rather than merely mis-presented.
    - `REVISE`: otherwise. The metric holds but a presentation assumption
      did not.
    """
    primary = (
        "A_population_and_centering",
        "B_opportunity_heterogeneity",
        "C_totals_vs_rate",
        "D_reliever_single_play_dominance",
        "E_rate_precision",
    )
    secondary = ("F_persistence",)
    qualitative = ("G_real_play_sanity_check",)

    disagreements = [k for k in primary if answers[k].get("agrees") is False]
    secondary_disagreements = [k for k in secondary if answers[k].get("agrees") is False]

    structural_failure = [
        k
        for k in ("A_population_and_centering", "E_rate_precision")
        if answers[k].get("agrees") is False
    ]

    if not disagreements:
        verdict = "REPLICATED"
    elif structural_failure:
        verdict = "NO_GO"
    else:
        verdict = "REVISE"

    assert verdict in CLASSIFICATION_VALUES
    return {
        "classification": verdict,
        "primary_questions": list(primary),
        "secondary_questions": list(secondary),
        "qualitative_questions": list(qualitative),
        "primary_disagreements": disagreements,
        "secondary_disagreements": secondary_disagreements,
        "all_disagreements_reported": True,
        "decided_on_the_package_not_one_statistic": True,
        "significance_testing_was_not_used": True,
        "secondary_cannot_override": (
            "Question F is secondary; its verdict is reported but never changes the "
            "classification on its own."
        ),
        "qualitative_not_auto_scored": (
            "Question G is qualitative and is reported for a human reader, never scored."
        ),
        "rule": (
            "REPLICATED when every primary question (A-E) agrees; NO_GO when the metric's "
            "own coherence failed (A or E); REVISE otherwise."
        ),
    }


def assert_report_covers_every_question(report: dict[str, Any]) -> None:
    """Every frozen question is answered, and no extra question was added.

    Raises:
        QuestionError: on a missing or unexpected question key.
    """
    answered = set(report.get("questions", {}))
    expected = set(QUESTION_KEYS)
    missing = sorted(expected - answered)
    extra = sorted(answered - expected)
    if missing or extra:
        raise QuestionError(
            f"replication report must answer exactly the frozen questions -- missing: "
            f"{missing or 'none'}; unexpected: {extra or 'none'}"
        )


def build_answers(frame: pd.DataFrame, artifacts: Any, play_level_mean: float) -> dict[str, Any]:
    """Answer all seven frozen questions and classify the package."""
    answers = {
        "A_population_and_centering": question_a(frame, play_level_mean),
        "B_opportunity_heterogeneity": question_b(frame),
        "C_totals_vs_rate": question_c(frame),
        "D_reliever_single_play_dominance": question_d(frame),
        "E_rate_precision": question_e(frame),
        "F_persistence": question_f(artifacts),
        "G_real_play_sanity_check": question_g(frame),
    }
    report = {"questions": answers, "classification": classify(answers)}
    assert_report_covers_every_question(report)
    return report
