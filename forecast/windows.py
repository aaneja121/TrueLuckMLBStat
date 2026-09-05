"""Contact Forecast R1: strictly chronological hitter-window construction.

This module builds the temporal dataset and NOTHING else -- no features, no
targets beyond the two the design specifies, no model. It is deliberately
the first thing written and the first thing tested, because every leakage
claim the study makes rests on the index arithmetic here being right.

## The construction

For each (batter, season), eligible batted balls are ordered strictly
chronologically and indexed `1..N` (`bbe_index`). For each cutoff `K` and
horizon `H`:

    history window = bbe_index in [1, K]        <- features may read ONLY this
    target window  = bbe_index in [K+1, K+H]    <- targets read ONLY this

A hitter-season enters the cell only when `N >= K + H`. The two windows are
disjoint and ordered by construction, and `assert_windows_causal` re-proves
that on the built table rather than trusting the arithmetic above.

Windows are counted in BATTED BALLS, not calendar days. Two hitters at
`K = 100` sit at different dates, and their target windows span different
stretches of the season and different league run environments. That is what
"first 100 eligible BBE" means and is the intended design, but it is a real
property of the results: `build_windows` therefore records the calendar
dates of every window boundary so the report can show the spread instead of
leaving a reader to assume the windows are contemporaneous.

## The two targets

Both are rates per 100 eligible batted balls over the TARGET window only:

  - `target_realized_rv_per_100` (**Target A**): actual run value. What a
    user ultimately experiences as future production.
  - `target_deserved_rv_per_100` (**Target B**): expected run value under
    the frozen Contact Luck contact model. Persistence of underlying contact
    quality, less contaminated by future defensive and outcome noise.

Target B is read from the input ledger's `baseline_expected_contact_run_value`
column (E0). This
module never computes it, never trains anything, and never touches the
contact model -- `forecast.score_development_seasons` produces that column
under the walk-forward rule and hands it here already computed.

## Which batted balls count: production's `resolved` denominator

An eligible batted ball whose `events` is `field_error` or `fielders_choice`
carries a NULL `outcome_class`: Version 0.1 deliberately refuses to guess the
batter-runner's fate on those (`mlb_luck_score.eligibility`'s
`_AMBIGUOUS_OUTCOME_EVENTS`), so they have no observed run value and no
Contact Luck. Production already excludes them from the per-100 denominator
-- `mlb_luck_score.scoring.aggregate_attribution` builds
`eligible_batted_balls` as `sum(resolved)`, where `resolved` is exactly
"observed contact-result run value is available."

`select_scored_eligible_events` applies that same rule here, so a rate this
package calls "runs per 100 eligible batted balls" is the SAME quantity
production publishes under that name. It is ~1.2% of eligible rows in
2021-2024 and ~1.2% of the 2026 production ledger. Excluding them also
renumbers `bbe_index`, which is why the exclusion must happen BEFORE
`order_eligible_events` and is reported as counts rather than applied
silently.

This module does not re-implement eligibility, which lives in
`mlb_luck_score.eligibility` and nowhere else; it reads the null
`outcome_class` that module already produces.

## Ordering and the doubleheader assumption

`forecast_config.CHRONOLOGICAL_SORT_KEY` orders by `game_date`, then
`game_pk`, then `at_bat_number`, then `pitch_number`. Within a single
calendar day containing a doubleheader, the day's two games are ordered by
ascending `game_pk`. MLB assigns doubleheader game_pks in schedule order, so
this is almost always right, but it is an ASSUMPTION rather than a verified
fact and it is load-bearing only for windows whose boundary falls between
the two games of such a day. `count_doubleheader_boundary_windows` counts
exactly those windows so the report can state the exposure instead of
hand-waving it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast.forecast_config import (
    CHRONOLOGICAL_SORT_KEY,
    FORECAST_CUTOFFS,
    FORECAST_HORIZONS,
    RATE_SCALE,
    assert_forecast_seasons_allowed,
)


class ForecastWindowError(ValueError):
    """Raised when window construction inputs or outputs violate the design."""


#: Columns every per-batted-ball ledger must carry to enter this module.
#:
#: These are production's OWN internal names for the three CONTACT-STAGE
#: quantities (`mlb_luck_score.scoring.attribution_ledger`): Rc, E0, and
#: Rc - E0. They are deliberately NOT the production PLAY-LEDGER names
#: (`observed_run_value`/`expected_run_value`/`contact_luck_runs`), because
#: those columns hold Rf and Rf - E0 -- the full telescoping quantity
#: including batter-runner advancement, which is what contactluck.com
#: publishes as Contact Luck. Reusing the play-ledger names for
#: contact-stage values would be a silent redefinition of the published
#: metric (RESEARCH_RULES.md, "Never silently redefine the Luck Score").
#:
#: `contact_result_surprise` is required even though no target uses it
#: directly: `assert_ledger_schema` verifies the identity on it, which is
#: how this package proves it received a genuine frozen-methodology ledger
#: rather than an arbitrary frame with the right column names.
REQUIRED_LEDGER_COLUMNS: tuple[str, ...] = (
    "event_id",
    "batter",
    "season",
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "observed_contact_result_run_value",
    "baseline_expected_contact_run_value",
    "contact_result_surprise",
)

#: Columns of the window table `build_windows` returns.
WINDOW_COLUMNS: tuple[str, ...] = (
    "window_id",
    "batter",
    "season",
    "cutoff",
    "horizon",
    "n_eligible_bbe_season",
    "history_start_index",
    "history_end_index",
    "target_start_index",
    "target_end_index",
    "history_start_date",
    "history_end_date",
    "target_start_date",
    "target_end_date",
    "target_realized_rv_per_100",
    "target_deserved_rv_per_100",
)

#: Absolute tolerance for the contact-stage identity
#: `contact_result_surprise == observed_contact_result_run_value -
#: baseline_expected_contact_run_value`. Loose
#: enough for float round-trips through parquet, far tighter than any
#: difference a redefinition would produce.
_IDENTITY_ATOL = 1e-6


def assert_ledger_schema(ledger: pd.DataFrame) -> None:
    """Fail loudly on any input defect that would silently corrupt a window.

    Checks, in order: required columns present; a unique non-null
    `event_id`; no nulls anywhere in the chronological sort key; finite
    non-null run values; and the contact-stage identity
    `contact_result_surprise == observed_contact_result_run_value -
    baseline_expected_contact_run_value` (Rc - E0).

    The identity check is the important one. A ledger that fails it is not
    the frozen Contact Luck methodology, whatever its column names say, and
    every "deserved" result computed from it would be meaningless.

    Raises:
        ForecastWindowError: On any of the above.
    """
    missing = [c for c in REQUIRED_LEDGER_COLUMNS if c not in ledger.columns]
    if missing:
        raise ForecastWindowError(f"Ledger is missing required column(s): {missing}")

    if ledger.empty:
        raise ForecastWindowError("Ledger is empty -- there are no batted balls to window")

    if ledger["event_id"].isna().any():
        raise ForecastWindowError("Ledger contains null event_id values")
    if not ledger["event_id"].is_unique:
        duplicated = int(ledger["event_id"].duplicated().sum())
        raise ForecastWindowError(
            f"Ledger contains {duplicated} duplicated event_id value(s); a batted ball "
            "counted twice would corrupt every window index downstream"
        )

    for column in CHRONOLOGICAL_SORT_KEY:
        if ledger[column].isna().any():
            n_null = int(ledger[column].isna().sum())
            raise ForecastWindowError(
                f"Ledger has {n_null} null value(s) in ordering column {column!r}; "
                "chronological order cannot be established"
            )

    value_columns = (
        "observed_contact_result_run_value",
        "baseline_expected_contact_run_value",
        "contact_result_surprise",
    )
    for column in value_columns:
        values = pd.to_numeric(ledger[column], errors="coerce")
        if values.isna().any():
            n_bad = int(values.isna().sum())
            raise ForecastWindowError(
                f"Ledger has {n_bad} null/non-numeric value(s) in {column!r}. Every "
                "batted ball entering a window must be scored -- drop unscored rows "
                "upstream and record the count, never carry them in as NaN"
            )
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ForecastWindowError(f"Ledger has non-finite value(s) in {column!r}")

    residual = (
        ledger["observed_contact_result_run_value"].to_numpy(dtype=float)
        - ledger["baseline_expected_contact_run_value"].to_numpy(dtype=float)
        - ledger["contact_result_surprise"].to_numpy(dtype=float)
    )
    worst = float(np.max(np.abs(residual))) if residual.size else 0.0
    if worst > _IDENTITY_ATOL:
        raise ForecastWindowError(
            "Contact-stage identity violated: contact_result_surprise != "
            "observed_contact_result_run_value - baseline_expected_contact_run_value "
            f"(max absolute residual {worst:.3e} > {_IDENTITY_ATOL:.0e}). "
            "This ledger is not the frozen Contact Luck methodology."
        )

    assert_forecast_seasons_allowed(sorted(ledger["season"].astype(int).unique()))


def select_scored_eligible_events(
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop unresolved batted balls, matching production's per-100 denominator.

    Must run BEFORE `order_eligible_events`: dropping rows renumbers
    `bbe_index`, so applying it afterwards would leave gaps in the index and
    silently shift every window boundary.

    Args:
        ledger: Eligible batted balls, carrying `outcome_class` (which is
            null for `field_error`/`fielders_choice` -- see the module
            docstring) plus the run-value columns.

    Returns:
        `(kept, report)`. `kept` holds only rows with a resolved outcome and
        finite run values; `report` counts what was dropped and why, so the
        leakage/exclusion audit can state the exclusion rather than have a
        reader infer it from a row count that does not add up.

    Raises:
        ForecastWindowError: If `outcome_class` is absent -- its nullity IS
            the rule, so a frame without it cannot be filtered correctly and
            must not be silently passed through.
    """
    if "outcome_class" not in ledger.columns:
        raise ForecastWindowError(
            "Ledger has no outcome_class column; the resolved-batted-ball rule "
            "(production's per-100 denominator) cannot be applied without it"
        )

    n_input = int(len(ledger))
    resolved = ledger["outcome_class"].notna()
    value_columns = [
        "observed_contact_result_run_value",
        "baseline_expected_contact_run_value",
        "contact_result_surprise",
    ]
    present = [c for c in value_columns if c in ledger.columns]
    scored = resolved.copy()
    for column in present:
        scored &= pd.to_numeric(ledger[column], errors="coerce").notna()

    kept = ledger.loc[scored].copy()
    report = {
        "n_input_rows": n_input,
        "n_unresolved_outcome_class": int((~resolved).sum()),
        "n_dropped_for_missing_run_value": int((resolved & ~scored).sum()),
        "n_kept": int(len(kept)),
    }
    if not len(kept):
        raise ForecastWindowError(
            f"Every one of {n_input} row(s) was dropped as unresolved or unscored"
        )
    return kept, report


def order_eligible_events(ledger: pd.DataFrame) -> pd.DataFrame:
    """Sort chronologically and assign a 1-based `bbe_index` per hitter-season.

    Args:
        ledger: A per-batted-ball frame carrying `REQUIRED_LEDGER_COLUMNS`.
            Already filtered to eligible batted balls -- this module does not
            apply eligibility, which lives in `mlb_luck_score.eligibility`
            and nowhere else.

    Returns:
        A copy sorted by `(batter, season, *CHRONOLOGICAL_SORT_KEY)` with
        `game_date` parsed to `datetime64[ns]` and an added `bbe_index`
        column running `1..N` within each `(batter, season)`.

    Raises:
        ForecastWindowError: If the schema check fails, if `game_date` will
            not parse, or if the sort key does not uniquely order a
            hitter-season (which would make `bbe_index` arbitrary).
    """
    assert_ledger_schema(ledger)

    ordered = ledger.copy()
    parsed = pd.to_datetime(ordered["game_date"], errors="coerce")
    if parsed.isna().any():
        n_bad = int(parsed.isna().sum())
        raise ForecastWindowError(f"{n_bad} game_date value(s) could not be parsed as dates")
    ordered["game_date"] = parsed

    group_key = ["batter", "season"]
    duplicated = ordered.duplicated(subset=group_key + list(CHRONOLOGICAL_SORT_KEY))
    if duplicated.any():
        raise ForecastWindowError(
            f"{int(duplicated.sum())} row(s) share a (batter, season, {', '.join(CHRONOLOGICAL_SORT_KEY)}) "
            "key, so chronological order within those hitter-seasons is ambiguous"
        )

    ordered = ordered.sort_values(group_key + list(CHRONOLOGICAL_SORT_KEY)).reset_index(drop=True)
    ordered["bbe_index"] = ordered.groupby(group_key, sort=False).cumcount() + 1
    return ordered


def _cumulative_lookup(ordered: pd.DataFrame) -> pd.DataFrame:
    """Per (batter, season, bbe_index): cumulative run values through that index.

    `cum_observed`/`cum_deserved` at index `i` are the sums over events
    `1..i`, so any window sum is one subtraction: events `K+1..K+H` sum to
    `cum[K + H] - cum[K]`.
    """
    group_key = ["batter", "season"]
    lookup = ordered[[*group_key, "bbe_index", "game_date"]].copy()
    grouped = ordered.groupby(group_key, sort=False)
    lookup["cum_observed"] = grouped["observed_contact_result_run_value"].cumsum().to_numpy()
    lookup["cum_deserved"] = grouped["baseline_expected_contact_run_value"].cumsum().to_numpy()
    return lookup


def build_windows(
    ordered: pd.DataFrame,
    *,
    cutoffs: tuple[int, ...] = FORECAST_CUTOFFS,
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
) -> pd.DataFrame:
    """Enumerate every (hitter-season, cutoff, horizon) window with its targets.

    Args:
        ordered: Output of `order_eligible_events`.
        cutoffs: Observation cutoffs `K`, in eligible batted balls.
        horizons: Forward horizons `H`, in eligible batted balls.

    Returns:
        One row per included window with `WINDOW_COLUMNS`. A hitter-season
        is included in a cell only when it has at least `K + H` eligible
        batted balls, so the same hitter-season legitimately appears in
        several cells with overlapping histories -- that is why every
        bootstrap in this study clusters on `batter` rather than treating
        rows as independent.

    Raises:
        ForecastWindowError: If `ordered` lacks `bbe_index`, if any cutoff
            or horizon is not a positive integer, or if the built table
            fails `assert_windows_causal`.
    """
    if "bbe_index" not in ordered.columns:
        raise ForecastWindowError("`ordered` has no bbe_index -- call order_eligible_events first")
    for name, values in (("cutoffs", cutoffs), ("horizons", horizons)):
        bad = [v for v in values if not isinstance(v, int | np.integer) or v < 1]
        if bad:
            raise ForecastWindowError(f"{name} must be positive integers; got {bad}")

    group_key = ["batter", "season"]
    season_counts = (
        ordered.groupby(group_key, sort=False)["bbe_index"]
        .max()
        .rename("n_eligible_bbe_season")
        .reset_index()
    )
    lookup = _cumulative_lookup(ordered)

    frames: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        for horizon in horizons:
            required = cutoff + horizon
            cell = season_counts[season_counts["n_eligible_bbe_season"] >= required].copy()
            if cell.empty:
                continue

            cell["cutoff"] = int(cutoff)
            cell["horizon"] = int(horizon)
            cell["history_start_index"] = 1
            cell["history_end_index"] = int(cutoff)
            cell["target_start_index"] = int(cutoff) + 1
            cell["target_end_index"] = int(required)

            at_first = _merge_at_index(cell, lookup, 1, suffix="first")
            at_cutoff = _merge_at_index(at_first, lookup, int(cutoff), suffix="cutoff")
            at_target_start = _merge_at_index(
                at_cutoff, lookup, int(cutoff) + 1, suffix="target_start"
            )
            cell = _merge_at_index(at_target_start, lookup, int(required), suffix="target_end")

            cell["history_start_date"] = cell["game_date_first"]
            cell["history_end_date"] = cell["game_date_cutoff"]
            cell["target_start_date"] = cell["game_date_target_start"]
            cell["target_end_date"] = cell["game_date_target_end"]

            observed_sum = cell["cum_observed_target_end"] - cell["cum_observed_cutoff"]
            deserved_sum = cell["cum_deserved_target_end"] - cell["cum_deserved_cutoff"]
            cell["target_realized_rv_per_100"] = RATE_SCALE * observed_sum / float(horizon)
            cell["target_deserved_rv_per_100"] = RATE_SCALE * deserved_sum / float(horizon)

            cell["window_id"] = (
                cell["batter"].astype(str)
                + "-"
                + cell["season"].astype(str)
                + f"-K{int(cutoff)}-H{int(horizon)}"
            )
            frames.append(cell[list(WINDOW_COLUMNS)])

    if not frames:
        return pd.DataFrame(columns=list(WINDOW_COLUMNS))

    windows = pd.concat(frames, ignore_index=True)
    windows = windows.sort_values(["season", "cutoff", "horizon", "batter"]).reset_index(drop=True)
    assert_windows_causal(windows)
    return windows


def _merge_at_index(
    frame: pd.DataFrame, lookup: pd.DataFrame, index: int, *, suffix: str
) -> pd.DataFrame:
    """Attach `lookup`'s values at a specific `bbe_index`, suffixed."""
    at_index = lookup[lookup["bbe_index"] == index].drop(columns=["bbe_index"])
    at_index = at_index.rename(
        columns={
            "game_date": f"game_date_{suffix}",
            "cum_observed": f"cum_observed_{suffix}",
            "cum_deserved": f"cum_deserved_{suffix}",
        }
    )
    merged = frame.merge(at_index, on=["batter", "season"], how="left")
    if merged[f"game_date_{suffix}"].isna().any():
        n_missing = int(merged[f"game_date_{suffix}"].isna().sum())
        raise ForecastWindowError(
            f"{n_missing} hitter-season(s) cleared the count bar but have no batted ball "
            f"at bbe_index {index}; the index is not contiguous"
        )
    return merged


def assert_windows_causal(windows: pd.DataFrame) -> None:
    """Re-prove, on the built table, that no window can see its own future.

    This is the study's central leakage assertion and it deliberately does
    not trust `build_windows`' arithmetic: it re-derives every relationship
    from the emitted columns. Checks that history starts at the hitter's
    first batted ball, that history and target index ranges are disjoint and
    correctly ordered, that each window's span matches its declared cutoff
    and horizon, that the target fits inside the season, and that no target
    window begins before its history window ends in CALENDAR time either.

    Raises:
        ForecastWindowError: On any violation, naming the offending windows.
    """
    if windows.empty:
        return

    def _fail(mask: pd.Series[bool], message: str) -> None:
        if bool(mask.any()):
            offenders = windows.loc[mask, "window_id"].head(5).tolist()
            raise ForecastWindowError(f"{message} ({int(mask.sum())} window(s); e.g. {offenders})")

    _fail(windows["history_start_index"] != 1, "History window does not start at bbe_index 1")
    _fail(
        windows["history_end_index"] != windows["cutoff"],
        "History window does not end exactly at the cutoff",
    )
    # The overlap check runs BEFORE the structural ones below: an overlap is
    # the substantive finding (a target the features can see), and reporting
    # it as a mere off-by-one in window bookkeeping would bury the lede.
    _fail(
        windows["target_start_index"] <= windows["history_end_index"],
        "LEAKAGE: target window overlaps the history window",
    )
    _fail(
        windows["target_start_index"] != windows["history_end_index"] + 1,
        "Target window does not begin immediately after the history window",
    )
    _fail(
        windows["target_end_index"] != windows["target_start_index"] + windows["horizon"] - 1,
        "Target window length does not match its declared horizon",
    )
    _fail(
        windows["target_end_index"] > windows["n_eligible_bbe_season"],
        "Target window extends past the hitter-season's last batted ball",
    )
    _fail(
        windows["history_start_date"] > windows["history_end_date"],
        "History window ends before it starts in calendar time",
    )
    _fail(
        windows["target_start_date"] < windows["history_end_date"],
        "LEAKAGE: target window starts before the history window ends in calendar time",
    )
    _fail(
        windows["target_start_date"] > windows["target_end_date"],
        "Target window ends before it starts in calendar time",
    )

    for column in ("target_realized_rv_per_100", "target_deserved_rv_per_100"):
        values = windows[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ForecastWindowError(f"{column} contains non-finite value(s)")


def history_events(ordered: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Every batted ball a feature builder may read at `cutoff`.

    The single supported way to obtain pre-cutoff rows. Feature code must
    never slice `ordered` itself: routing every slice through here is what
    makes "features saw only events 1..K" checkable in one place rather than
    re-argued at each feature.

    Args:
        ordered: Output of `order_eligible_events`.
        cutoff: The observation cutoff `K`.

    Returns:
        Rows with `bbe_index <= cutoff`, carrying an added `cutoff` column.

    Raises:
        ForecastWindowError: If `cutoff` is not a positive integer or
            `ordered` has no `bbe_index`.
    """
    if "bbe_index" not in ordered.columns:
        raise ForecastWindowError("`ordered` has no bbe_index -- call order_eligible_events first")
    if not isinstance(cutoff, int | np.integer) or cutoff < 1:
        raise ForecastWindowError(f"cutoff must be a positive integer; got {cutoff!r}")

    history = ordered[ordered["bbe_index"] <= int(cutoff)].copy()
    history["cutoff"] = int(cutoff)
    return history


def summarize_inclusion(
    ordered: pd.DataFrame,
    *,
    cutoffs: tuple[int, ...] = FORECAST_CUTOFFS,
    horizons: tuple[int, ...] = FORECAST_HORIZONS,
) -> pd.DataFrame:
    """Characterize the survivorship condition each cell imposes.

    A hitter-season enters a cell only if it has `K + H` batted balls, which
    conditions on FUTURE playing time. Both compared predictors share the
    condition so it does not bias the deserved-versus-realized comparison,
    but it does bound what the results generalize to, and a reader is
    entitled to see how far.

    Returns one row per (season, cutoff, horizon) with: how many
    hitter-seasons reached the cutoff (and so would have received a
    prediction in live use), how many were included, the exclusion rate, and
    the mean realized rate through the cutoff for included versus excluded
    hitter-seasons -- the last pair being the direct evidence of whether the
    hitters who drop out differ from those who survive.
    """
    group_key = ["batter", "season"]
    counts = ordered.groupby(group_key, sort=False)["bbe_index"].max().rename("n").reset_index()
    lookup = _cumulative_lookup(ordered)

    rows: list[dict[str, object]] = []
    for cutoff in cutoffs:
        at_cutoff = lookup[lookup["bbe_index"] == int(cutoff)][[*group_key, "cum_observed"]].rename(
            columns={"cum_observed": "cum_observed_cutoff"}
        )
        reached = counts[counts["n"] >= int(cutoff)].merge(at_cutoff, on=group_key, how="left")
        reached["cutoff_realized_rv_per_100"] = (
            RATE_SCALE * reached["cum_observed_cutoff"] / float(cutoff)
        )
        for horizon in horizons:
            required = int(cutoff) + int(horizon)
            for season in sorted(reached["season"].unique()):
                season_rows = reached[reached["season"] == season]
                included = season_rows[season_rows["n"] >= required]
                excluded = season_rows[season_rows["n"] < required]
                n_reached = int(len(season_rows))
                rows.append(
                    {
                        "season": int(season),
                        "cutoff": int(cutoff),
                        "horizon": int(horizon),
                        "n_reached_cutoff": n_reached,
                        "n_included": int(len(included)),
                        "n_excluded": int(len(excluded)),
                        "exclusion_rate": (
                            float(len(excluded)) / n_reached if n_reached else float("nan")
                        ),
                        "included_mean_cutoff_realized_rv_per_100": (
                            float(included["cutoff_realized_rv_per_100"].mean())
                            if len(included)
                            else float("nan")
                        ),
                        "excluded_mean_cutoff_realized_rv_per_100": (
                            float(excluded["cutoff_realized_rv_per_100"].mean())
                            if len(excluded)
                            else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def count_doubleheader_boundary_windows(
    ordered: pd.DataFrame,
    *,
    cutoffs: tuple[int, ...] = FORECAST_CUTOFFS,
) -> pd.DataFrame:
    """Count windows whose cutoff boundary depends on the doubleheader assumption.

    The `game_pk`-ascending tie-break (see the module docstring) only
    matters when the events at `bbe_index` `K` and `K + 1` fall on the SAME
    calendar day but in DIFFERENT games -- there, and only there, does the
    assumed within-day game order decide which side of the cutoff a batted
    ball lands on. Reported per (season, cutoff) so the leakage audit can
    state the exposure as a number.
    """
    group_key = ["batter", "season"]
    slim = ordered[[*group_key, "bbe_index", "game_date", "game_pk"]]

    rows: list[dict[str, object]] = []
    for cutoff in cutoffs:
        at_cutoff = slim[slim["bbe_index"] == int(cutoff)].rename(
            columns={"game_date": "date_at_cutoff", "game_pk": "game_at_cutoff"}
        )
        after_cutoff = slim[slim["bbe_index"] == int(cutoff) + 1].rename(
            columns={"game_date": "date_after_cutoff", "game_pk": "game_after_cutoff"}
        )
        pairs = at_cutoff.drop(columns=["bbe_index"]).merge(
            after_cutoff.drop(columns=["bbe_index"]), on=group_key, how="inner"
        )
        pairs["boundary_inside_doubleheader"] = (
            pairs["date_at_cutoff"] == pairs["date_after_cutoff"]
        ) & (pairs["game_at_cutoff"] != pairs["game_after_cutoff"])
        for season in sorted(pairs["season"].unique()):
            season_pairs = pairs[pairs["season"] == season]
            n_pairs = int(len(season_pairs))
            n_affected = int(season_pairs["boundary_inside_doubleheader"].sum())
            rows.append(
                {
                    "season": int(season),
                    "cutoff": int(cutoff),
                    "n_hitter_seasons_reaching_cutoff": n_pairs,
                    "n_boundary_inside_doubleheader": n_affected,
                    "share_boundary_inside_doubleheader": (
                        float(n_affected) / n_pairs if n_pairs else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)
