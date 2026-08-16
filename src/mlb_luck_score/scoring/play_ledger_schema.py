"""Version 1.4.0 (Play Explorer foundation): the canonical per-play ledger schema.

Phase 1's inspection found that Contact Luck's per-play scoring state
(`mlb_luck_score.scoring.attribution_ledger.build_attribution_ledger`'s
returned DataFrame, plus the contact model's own per-row probability
vector) already exists in memory during every prospective/evaluation
scoring run, but is never persisted -- only player-season aggregates are
written to disk. This module defines the schema a FUTURE persisted
per-play artifact (`play_ledger.parquet`) must satisfy; `play_ledger_
export.py` is the read-only projection that builds a DataFrame against it
from already-scored values; `play_ledger_metadata.py` is the companion
FILE-level (not per-row) provenance contract -- see that module for why
`play_ledger_version` and other file-level facts live there, not repeated
on every row here.

This module computes nothing and reimplements no scoring formula -- it only
declares column names/dtypes/nullability and validates that a candidate
DataFrame satisfies the contract. See CLAUDE.md's "Never silently redefine
the Luck Score" rule: `contact_luck_runs` here must always equal `observed_
run_value - expected_run_value` for every resolved row, and this module's
own `validate_play_ledger` checks that identity directly rather than
trusting the caller.

## Phase 2 contract amendment: canonical scoring truth, not presentation

An earlier draft included `batter_name`/`batter_team`/`opponent_team` as
canonical row fields. Removed: those values only exist after a
presentation overlay applied AFTER scoring (the same async name-resolution/
game-metadata-join pattern `prospective.prospective_player_names` already
uses for the public leaderboard), so requiring them on the canonical
ledger would make canonical persistence depend on overlay timing --
exactly the kind of scoring-vs-presentation conflation CLAUDE.md's
Version 1.1.1 fix ("player name resolution is presentation-only") already
established the precedent against. The canonical ledger now carries only
`batter_id` (a stable scoring identifier, never overlay-dependent) plus
`game_pk`/`at_bat_number`/`pitch_number`/`game_date`/`season`/`play_id`. A
future browser artifact may
enrich rows with name/team labels by joining the SAME overlays the rest of
the site already uses, at the presentation layer, never altering any
scoring value. (Separately investigated and NOT added here: `home_team`/
`away_team` are genuine RAW, non-overlay Statcast columns already present
on the scoring dataframe -- a team CODE, unlike a team NAME, could in
principle be derived deterministically via `inning_topbot` without any
overlay-timing dependency. Reported per your instruction, not added, since
it is not in the approved field list below.)

## Version 2.0: `observed_run_value`/`contact_luck_runs` corrected to Rf - E0

The real 2026-08-15 canonical production snapshot (Version 1.0 of this
schema, the FIRST naturally-generated snapshot to carry a play ledger) was
independently reconciled against that same snapshot's published
`public_score` and FAILED for 261 of 630 batters on Total Contact Luck Runs
and Runs/100 (BBE and Scored Games both matched exactly for every batter).
Root cause: Version 1.0 sourced `observed_run_value`/`contact_luck_runs`
from the attribution ledger's `observed_contact_result_run_value`/`contact_
result_surprise` columns -- `Rc - E0`, the raw Version 0.2 contact-luck
quantity -- which `attribution_ledger.py`'s own module docstring already
documented as NOT the same as the full telescoping-identity quantity `Rf -
E0` whenever a play has nonzero defensive/advancement execution
contribution. `aggregate_attribution.aggregate_to_batter_season` (and
therefore every published `total_contact_luck_runs`/`contact_luck_runs_
per_100` figure) has ALWAYS summed `Rf - E0`, never `Rc - E0`. Phase 2's
development-data reconciliation never caught this because it deliberately
used a contact-only ledger (no outfield/infield/advancement models
trained), where `Rc - E0` and `Rf - E0` coincide trivially -- see `tests/
test_attribution_ledger.py::test_full_ledger_fixture_has_zero_naturally_
diverging_rows` for why even a full-four-model SYNTHETIC fixture can
accidentally mask this same class of bug, and the deterministic fixture
next to it for the actual regression.

Version 2.0 corrects this: `observed_run_value = Rf` (`ledger["observed_
final_run_value"]`), `expected_run_value = E0` (unchanged, `ledger
["baseline_expected_contact_run_value"]`), `contact_luck_runs = Rf - E0`
(`ledger["final_result_surprise"]`, added to `attribution_ledger.py`
specifically for this fix -- see that module's docstring). The five `p_*`
probabilities are UNCHANGED (still the same already-computed frozen contact
probabilities, zero new inference). No model fitting, scoring formula,
component-attribution formula, or `public_score` aggregation changed --
this is purely a play-ledger PROJECTION fix: `public_score` was always
correct, and always used `Rf - E0`.

### `outcome_class` represents Rc, not Rf -- and that is correct, unchanged

Audited directly (`mlb_luck_score.eligibility`'s `ELIGIBLE_EVENTS_V0_1`/
`_DIRECT_HIT_EVENTS`/`_AMBIGUOUS_OUTCOME_EVENTS`): `outcome_class` is
derived SOLELY from Statcast's raw `events` field -- the officially
recorded contact result of the batted-ball event itself (out/single/
double/triple/home_run), completely independent of what happens on the
bases afterward. It is Rc-level by definition and was NEVER intended to
represent Rf. For the large majority of rows (no advancement modeled, or
advancement modeled but the batter-runner's real final base matches the
recorded hit type exactly), `Rc == Rf` and `outcome_class` already
describes both. For a minority of advancement-MODELED rows, the batter-
runner's true final base (`mlb_luck_score.eligibility.add_advancement_
eligibility`'s own native `batter_final_base` column, one of `ADVANCEMENT_
LABELS` -- e.g. `advanced_to_second` for a recorded `single`) can
legitimately differ from what the recorded hit type alone implies, which
is exactly what makes `Rf != Rc` for that row. This is NOT a data error --
`outcome_class` ("what was officially recorded") and `observed_run_value`/
`contact_luck_runs` ("what the full accounting system computed, including
realized advancement") are two different, both-correct pieces of
information about the same play.

`batter_final_base`/`ADVANCEMENT_LABELS` IS the system's existing native
representation of the true final outcome for the subset of rows where it
can diverge -- but it uses a different vocabulary (base-count labels, not
out/single/double/triple/home_run) and only exists for advancement-eligible
rows, and per the v1.4.0 scope decision it is an advancement-component
field, deliberately NOT exposed on the canonical play ledger (see "Do not
expose defensive or advancement component fields," CLAUDE.md's Version
1.4.0 task). The smallest scientifically correct schema adjustment is
therefore: keep `outcome_class` exactly as-is (unchanged name, unchanged
values, unchanged nullability), documented precisely as the Rc-level
recorded-contact-result classification, and fix the ACTUAL internal-
consistency defect this uncovered -- `_validate_contact_luck_identity`'s
Version 1.0 cross-check against `mlb_luck_score.scoring.contact_luck.
compute_raw_contact_luck_runs` implicitly assumed `contact_luck_runs`
reconstructs from `outcome_class` via `DEFAULT_RUN_VALUE_MAP` (i.e. that it
equals `Rc - E0`), which is no longer true in Version 2.0 for advancement-
modeled rows with real advancement execution. That specific cross-check is
REMOVED (see `_validate_contact_luck_identity`'s own docstring) -- the
self-referential identity `contact_luck_runs == observed_run_value -
expected_run_value` remains fully enforced and is the actual defining
guarantee.

## Why `p_out`..`p_home_run`/`expected_run_value`/`outcome_class`/
## `observed_run_value`/`contact_luck_runs` are nullable together

Phase 2's empirical investigation (real 2021-2024 development data, the
live, unmodified `mlb_luck_score.scoring.aggregate_attribution.
aggregate_to_batter_season`) confirmed: `is_eligible=True` rows can have
`outcome_class=None` (Version 0.1's `fielders_choice`/`field_error`
ambiguous-outcome category -- see `mlb_luck_score.eligibility`), and the
published `eligible_batted_balls` denominator on every player-aggregate
page counts ONLY outcome-RESOLVED rows (`aggregate_attribution.py`'s
`resolved = ledger["observed_contact_result_run_value"].notna()`), not
every `is_eligible` row. `attribution_ledger.build_attribution_ledger`
itself already establishes the pattern this schema follows: never silently
drop an eligible-but-unresolved row -- keep it, with an explicit status.

An earlier draft of this module assumed only the RESULT/CONTACT-LUCK
fields (`outcome_class`/`observed_run_value`/`contact_luck_runs`) would be
null for such a row, reasoning that the contact model's own expectation is
computed from `launch_speed`/`launch_angle`/etc. alone, before any outcome
is known, so it should stay meaningful regardless of resolution. That
reasoning is correct about WHEN the contact model runs, but empirically
wrong about what `build_attribution_ledger` actually RETAINS: its own
final step nulls `baseline_expected_contact_run_value` (this schema's
`expected_run_value`) together with the result fields, even though it was
well-defined moments earlier in that same function. Because this exporter
copies `expected_run_value` directly from that ledger column rather than
recomputing it (Decision 4: persist the already-scored value, never
recompute), the nullable group here had to widen to match what the ledger
actually produces: `p_out`..`p_home_run`, `expected_run_value`,
`outcome_class`, `observed_run_value`, and `contact_luck_runs` are all
null together, on every row. Only the CONTACT section (`launch_speed`/
`launch_angle`/`bb_type`/`spray_angle_approx`/`hit_distance_sc`/`stand`),
sourced from the pre-ledger `df` rather than the ledger itself, remains
populated for an eligible-but-unresolved play.

Consumers (a future Play Explorer search index, a reconciliation test)
must filter on `outcome_class.notna()` to reproduce the published
`eligible_batted_balls` count, never on `play_id` presence alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_luck_score.config import CLASS_ORDER

__all__ = [
    "PLAY_LEDGER_VERSION",
    "PLAY_LEDGER_COLUMNS",
    "PLAY_LEDGER_DTYPES",
    "PLAY_LEDGER_NULLABLE_COLUMNS",
    "PLAY_LEDGER_REQUIRED_NON_NULL_COLUMNS",
    "PROBABILITY_COLUMNS",
    "RESULT_LINKED_COLUMNS",
    "PlayLedgerValidationError",
    "validate_play_ledger",
]

#: Bump on any schema change (new/renamed/retyped column, changed
#: nullability rule, changed validation rule, or -- new in Version 2.0 --
#: a changed SEMANTIC MEANING of an existing column) -- independent of
#: `mlb_luck_score.config`'s `score_version`/model versions, exactly
#: mirroring `demo_counterfactual_grid_version`/`demo_fixture_version`'s
#: existing precedent (see `dashboard/demo_counterfactual_grid.json`/
#: `dashboard/demo_fixture.json`).
#:
#: "1.0" -> "2.0" (Phase 3.1): `observed_run_value`/`contact_luck_runs`
#: changed from `Rc`/`Rc - E0` to `Rf`/`Rf - E0` -- see module docstring's
#: "Version 2.0: observed_run_value/contact_luck_runs corrected to Rf - E0"
#: section. Column NAMES/dtypes/nullability are unchanged from 1.0; only
#: the meaning of two already-named fields changed, which is exactly why a
#: version bump (not a silent 1.0 patch) is required -- see CLAUDE.md
#: "Never silently redefine the Luck Score." The ONE real-production
#: v1.0 snapshot (2026-08-15, sealed, immutable, archived in R2) is
#: PERMANENTLY a v1.0 artifact and must never be reinterpreted as v2.0 --
#: any future Play Explorer/browser-artifact generator MUST explicitly
#: require `play_ledger_version == PLAY_LEDGER_VERSION` (or an allow-list
#: of known-compatible versions) before treating a snapshot's play ledger
#: as consumable, exactly like `prospective.prospective_manifest.
#: verify_v1_seal_unchanged`'s precedent for cross-version compatibility
#: checks elsewhere in this codebase. A future component-attribution
#: addition (explicitly OUT of v1.4.0's schema -- see module docstring)
#: would bump this again, not silently widen "2.0"'s contract.
PLAY_LEDGER_VERSION = "2.0"

#: `p_<class>` columns, in `CLASS_ORDER` order -- the contact model's own
#: ordered 5-class probability vector, unchanged from `mlb_luck_score.
#: config.CLASS_ORDER` (never redefined here; see module docstring).
PROBABILITY_COLUMNS: tuple[str, ...] = tuple(f"p_{c}" for c in CLASS_ORDER)

#: `at_bat_number`/`pitch_number` are the two native Statcast identity
#: fields `play_id` is derived FROM (`play_id = f"{game_pk}-{at_bat_number}-
#: {pitch_number}"`, `mlb_luck_score.data.clean_batted_balls.build_event_id`)
#: but were originally omitted here, forcing any downstream consumer to
#: parse them back out of the `play_id` string. Phase 2.5 amendment: retain
#: them natively -- both are already present on the pre-ledger `df` (see
#: `play_ledger_export.py`'s `_REQUIRED_DF_COLUMNS`), so this costs nothing
#: and avoids re-deriving structured data from a string encoding. `play_id`
#: itself is UNCHANGED (same format, still the canonical row key).
_IDENTITY_COLUMNS: tuple[str, ...] = (
    "play_id",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "game_date",
    "season",
    "batter_id",
)
_CONTACT_COLUMNS: tuple[str, ...] = (
    "stand",
    "launch_speed",
    "launch_angle",
    "bb_type",
    "spray_angle_approx",
    "hit_distance_sc",
)
#: `is_scored` -- a STATUS flag DERIVED from existing ledger state
#: (`observed_run_value.notna()`), not a new eligibility rule and not a
#: second source of truth: `validate_play_ledger` asserts it matches
#: `outcome_class.notna()` exactly on every row. Exists so a UI/reconciliation
#: consumer never has to infer resolution from several nullable columns --
#: see module docstring and `resolved_rows` below.
_STATUS_COLUMNS: tuple[str, ...] = ("is_scored",)
_RESULT_COLUMNS: tuple[str, ...] = ("outcome_class", "observed_run_value")
_EXPECTATION_COLUMNS: tuple[str, ...] = (*PROBABILITY_COLUMNS, "expected_run_value")
_CONTACT_LUCK_COLUMNS: tuple[str, ...] = ("contact_luck_runs",)

#: The full canonical column order. Deliberately EXCLUDES (per the approved
#: Phase 1/Phase 2 decisions): `batter_name`/`batter_team`/`opponent_team`
#: (presentation overlays, not canonical scoring truth -- see module
#: docstring); a `favorable: bool` field (display sign must be derived from
#: `contact_luck_runs`, never a separately-stored and potentially-stale
#: boolean -- a `contact_luck_runs == 0.0` row must never be silently forced
#: into either sign); per-play component attribution (`contact_component`/
#: `defensive_execution_component`/`advancement_execution_component`/
#: `unexplained_residual_component`) and their confidence/reason-code status
#: strings -- reserved for a future `play_ledger_version` bump, not v1.4.0;
#: raw Statcast `des` text (pending the Phase 1-identified site-wide
#: data-attribution gap); any internal feature-engineering intermediate
#: (hang time, alignment, geometry features) not part of the public
#: contract anywhere else in this codebase either; and `play_ledger_
#: version` itself, which is FILE-level metadata now (see `play_ledger_
#: metadata.py`), not repeated on every row.
PLAY_LEDGER_COLUMNS: tuple[str, ...] = (
    *_IDENTITY_COLUMNS,
    *_CONTACT_COLUMNS,
    *_STATUS_COLUMNS,
    *_RESULT_COLUMNS,
    *_EXPECTATION_COLUMNS,
    *_CONTACT_LUCK_COLUMNS,
)

#: pandas dtype string per column -- nullable numeric/string extension
#: dtypes ("Int64"/"string"/"Float64") where a column may be null, plain
#: numpy dtypes where it may not. `game_date` is stored as ISO-format
#: `string` (not `datetime64`) for simple, lossless Parquet/JSON
#: round-tripping without a timezone-handling decision this schema doesn't
#: need to make.
PLAY_LEDGER_DTYPES: dict[str, str] = {
    "play_id": "string",
    "game_pk": "Int64",
    "at_bat_number": "Int64",
    "pitch_number": "Int64",
    "game_date": "string",
    "season": "Int64",
    "batter_id": "Int64",
    "stand": "string",
    "launch_speed": "Float64",
    "launch_angle": "Float64",
    "bb_type": "string",
    "spray_angle_approx": "Float64",
    "hit_distance_sc": "Float64",
    "is_scored": "boolean",
    "outcome_class": "string",
    "observed_run_value": "Float64",
    **{c: "Float64" for c in PROBABILITY_COLUMNS},
    "expected_run_value": "Float64",
    "contact_luck_runs": "Float64",
}

#: Columns permitted to be null on a genuine, individual row.
#: `RESULT_LINKED_COLUMNS` below is the enforced null-together invariant
#: for the scoring group; `is_scored` (a `boolean`, never null -- see
#: `_STATUS_COLUMNS`) is the intended way to check that group's resolution
#: without touching a nullable column directly.
#:
#: IMPORTANT, empirically discovered during Phase 2 implementation (see
#: `tests/test_play_ledger_export.py`): an earlier draft of this schema
#: assumed the EXPECTATION fields (`p_out`..`p_home_run`, `expected_run_
#: value`) stay populated for an eligible-but-unresolved play, reasoning
#: that the contact model predicts from `launch_speed`/`launch_angle`/etc.
#: alone, before any outcome is known. That reasoning is correct about
#: WHEN the contact model runs, but `attribution_ledger.build_attribution_
#: ledger` itself does not preserve that distinction: its own final step
#: (`attribution_ledger.py`, the `for series in (rc, e0, ...): series.loc[
#: unresolved] = np.nan` loop) explicitly nulls `baseline_expected_contact_
#: run_value` (this schema's `expected_run_value`) ALONG WITH the result
#: fields, even though `e0` was well-defined moments earlier. This exporter
#: reads `expected_run_value` directly from that ledger column (Decision 4:
#: "the exporter itself should persist the already-scored value," never
#: recompute) rather than recomputing it from `contact_proba` (which is
#: NOT nulled) for unresolved rows -- so to stay faithful to "copy, never
#: recompute," this schema's null-together group had to widen to match
#: what the ledger actually produces, not what the schema originally
#: assumed. See the Phase 2 report for the alternative considered
#: (recompute `expected_run_value`/keep probabilities populated from
#: `contact_proba` directly for unresolved rows) and why it was not chosen
#: without explicit sign-off.
#:
#: A SECOND, independent empirical finding (Phase 2, real 2024 development
#: data via `demo/measure_play_ledger_scale.py`): raw Statcast contact
#: measurements themselves can be missing on an `is_eligible=True` row --
#: `mlb_luck_score.eligibility.compute_eligibility`'s own `missing_contact_
#: data` flag exists exactly for this (379 of 123,980 real 2024-eligible
#: rows, ~0.3%: 379 missing `launch_speed`, 316 missing `launch_angle`/
#: `hit_distance_sc`, 2 missing `bb_type`/`spray_angle_approx`). This is
#: UNRELATED to outcome resolution -- 361 of those 379 rows have a
#: perfectly resolved `outcome_class` (296 outs, 61 singles, 3 home runs, 1
#: triple), because `observed_run_value` depends only on `outcome_class`
#: via `DEFAULT_RUN_VALUE_MAP`, never on contact measurements. The frozen
#: contact model's own training pipeline includes an imputer (confirmed:
#: `predict_proba_ordered` returns a valid, non-null probability
#: distribution for these rows despite the missing raw inputs) -- this
#: schema does not change or second-guess that existing, already-shipped
#: production behavior; it only reflects, honestly, that the RAW
#: measurement was absent, by leaving the CONTACT field null rather than
#: fabricating a value or silently exposing the model's internal imputed
#: value as if it were an observed measurement. `stand` is the one CONTACT
#: field observed to always be present and stays non-nullable.
PLAY_LEDGER_NULLABLE_COLUMNS: frozenset[str] = frozenset(
    {
        "launch_speed",
        "launch_angle",
        "bb_type",
        "spray_angle_approx",
        "hit_distance_sc",
        *PROBABILITY_COLUMNS,
        "expected_run_value",
        "outcome_class",
        "observed_run_value",
        "contact_luck_runs",
    }
)

#: Every other column must be non-null on every row.
PLAY_LEDGER_REQUIRED_NON_NULL_COLUMNS: tuple[str, ...] = tuple(
    c for c in PLAY_LEDGER_COLUMNS if c not in PLAY_LEDGER_NULLABLE_COLUMNS
)

#: `p_out`..`p_home_run`, `expected_run_value`, `outcome_class`, `observed_
#: run_value`, `contact_luck_runs` must be null together on every row --
#: never a partial resolution. Wider than just the "RESULT"/"CONTACT LUCK"
#: schema sections because `attribution_ledger.build_attribution_ledger`
#: nulls the entire per-play ledger row (including the pre-outcome
#: expectation) for an unresolved play, not just the post-outcome fields --
#: see the docstring on `PLAY_LEDGER_NULLABLE_COLUMNS` above for why. Only
#: the CONTACT section (`launch_speed`/`launch_angle`/`bb_type`/...),
#: sourced from `df` directly rather than the ledger, remains populated for
#: an unresolved row.
RESULT_LINKED_COLUMNS: tuple[str, ...] = (
    *PROBABILITY_COLUMNS,
    "expected_run_value",
    "outcome_class",
    "observed_run_value",
    "contact_luck_runs",
)

_PROBABILITY_SUM_TOLERANCE = 1e-3
#: Matches `mlb_luck_score.scoring.attribution_ledger.IDENTITY_ATOL` -- the
#: same tolerance the live ledger's own accounting-identity check uses, not
#: a separately invented number.
_CONTACT_LUCK_IDENTITY_ATOL = 1e-9


class PlayLedgerValidationError(ValueError):
    """Raised when a candidate play-ledger DataFrame violates the canonical schema."""


def validate_play_ledger(df: pd.DataFrame) -> None:
    """Validate `df` against the canonical `PLAY_LEDGER_COLUMNS` contract.

    Checks (in order): every required column is present; `play_id` is
    non-null and unique; every `PLAY_LEDGER_REQUIRED_NON_NULL_COLUMNS`
    column is non-null on every row; `RESULT_LINKED_COLUMNS` are null
    together (never a partial resolution); every probability column is
    finite and in [0, 1] and the five sum to ~1.0 on every row; and, for
    every row with a resolved outcome, `contact_luck_runs` reconciles
    EXACTLY to `observed_run_value - expected_run_value` within
    `_CONTACT_LUCK_IDENTITY_ATOL` -- this is the "never silently redefine
    the Luck Score" guardrail (CLAUDE.md) applied to this new artifact.
    NOTE (Version 2.0, Phase 3.1): this identity check is deliberately
    self-referential (`observed_run_value - expected_run_value ==
    contact_luck_runs`, using only this schema's own three fields), NOT a
    cross-check against `mlb_luck_score.scoring.contact_luck.compute_raw_
    contact_luck_runs` -- see this module's docstring for why that
    specific cross-check was removed in Version 2.0.

    Raises:
        PlayLedgerValidationError: on any violation, with the exact
            offending column/row count named (never a generic message).
    """
    missing = [c for c in PLAY_LEDGER_COLUMNS if c not in df.columns]
    if missing:
        raise PlayLedgerValidationError(f"play ledger is missing column(s): {missing}")

    if df["play_id"].isna().any():
        n = int(df["play_id"].isna().sum())
        raise PlayLedgerValidationError(f"{n} row(s) have a null play_id")
    dup_mask = df["play_id"].duplicated(keep=False)
    if dup_mask.any():
        dupes = df.loc[dup_mask, "play_id"].unique().tolist()
        raise PlayLedgerValidationError(
            f"{len(dupes)} duplicate play_id value(s) found (first 10: {dupes[:10]})"
        )

    for col in PLAY_LEDGER_REQUIRED_NON_NULL_COLUMNS:
        null_count = int(df[col].isna().sum())
        if null_count:
            raise PlayLedgerValidationError(
                f"column {col!r} must be non-null on every row, found {null_count} null value(s)"
            )

    _validate_is_scored_matches_resolution(df)
    _validate_outcome_class_resolution_equivalence(df)
    _validate_result_linked_columns(df)
    _validate_probabilities(df)
    _validate_contact_luck_identity(df)


def _validate_is_scored_matches_resolution(df: pd.DataFrame) -> None:
    """`is_scored` is a DERIVED status flag, not a second source of truth --
    it must exactly equal `observed_run_value.notna()` on every row, never
    drift from it (e.g. via a stale copy or an independently-set value).

    `observed_run_value.notna()` -- not `outcome_class.notna()` -- is the
    PRIMARY definition (Phase 2.5 amendment): it is the exact same column
    `mlb_luck_score.scoring.aggregate_attribution.aggregate_to_batter_
    season` itself uses to define the published `eligible_batted_balls`
    denominator (`resolved = ledger["observed_contact_result_run_value"].
    notna()`). See `_validate_outcome_class_resolution_equivalence` below
    for the separate, explicit proof that `outcome_class.notna()` agrees
    with this under the currently supported scoring contract -- a genuine
    future divergence between the two must surface as its own named
    failure, not be silently absorbed into this check.
    """
    expected = df["observed_run_value"].notna()
    actual = df["is_scored"].astype("boolean").fillna(False)
    mismatch = expected.to_numpy() != actual.to_numpy()
    if mismatch.any():
        bad_ids = df.loc[mismatch, "play_id"].tolist()
        raise PlayLedgerValidationError(
            f"{int(mismatch.sum())} row(s) have is_scored != observed_run_value.notna() -- "
            f"is_scored must be a pure derivation, never an independent value "
            f"(first 10 play_id: {bad_ids[:10]})"
        )


def _validate_outcome_class_resolution_equivalence(df: pd.DataFrame) -> None:
    """Explicit invariant (Phase 2.5 amendment) for the currently supported
    scoring contract: `observed_run_value.notna() == outcome_class.notna()`
    on every row. `is_scored` is defined against `observed_run_value` (the
    same column `aggregate_to_batter_season` uses), so this check exists to
    prove that choice is equivalent to the `outcome_class`-based definition
    an earlier draft used -- not to silently assume it. If a future scoring
    contract ever legitimately decouples the two (e.g. a resolved outcome
    with no run-value mapping), this check must be revisited explicitly,
    not relaxed by deleting it.
    """
    outcome_resolved = df["outcome_class"].notna()
    value_resolved = df["observed_run_value"].notna()
    mismatch = outcome_resolved.to_numpy() != value_resolved.to_numpy()
    if mismatch.any():
        bad_ids = df.loc[mismatch, "play_id"].tolist()
        raise PlayLedgerValidationError(
            f"{int(mismatch.sum())} row(s) have outcome_class.notna() != "
            f"observed_run_value.notna() -- these must be equivalent under the "
            f"currently supported scoring contract (first 10 play_id: {bad_ids[:10]})"
        )


def _validate_result_linked_columns(df: pd.DataFrame) -> None:
    null_flags = df[list(RESULT_LINKED_COLUMNS)].isna()
    all_null = null_flags.all(axis=1)
    all_non_null = ~null_flags.any(axis=1)
    partial = ~(all_null | all_non_null)
    if partial.any():
        bad_ids = df.loc[partial, "play_id"].tolist()
        raise PlayLedgerValidationError(
            f"{int(partial.sum())} row(s) have a PARTIAL resolution of "
            f"{RESULT_LINKED_COLUMNS} -- these three columns must be null together or "
            f"non-null together on every row (first 10 play_id: {bad_ids[:10]})"
        )


def _validate_probabilities(df: pd.DataFrame, *, atol: float = _PROBABILITY_SUM_TOLERANCE) -> None:
    # Probabilities are null together with outcome_class/observed_run_value/
    # contact_luck_runs for an unresolved play (see PLAY_LEDGER_NULLABLE_
    # COLUMNS' docstring) -- `_validate_result_linked_columns` above already
    # confirms they're null/non-null in lockstep, so this check only needs
    # to validate the resolved subset.
    resolved = df["outcome_class"].notna()
    if not resolved.any():
        return
    proba = df.loc[resolved, list(PROBABILITY_COLUMNS)].astype("float64")
    finite = np.isfinite(proba.to_numpy())
    if not finite.all():
        raise PlayLedgerValidationError("probability column(s) contain non-finite value(s)")
    if (proba.to_numpy() < 0).any() or (proba.to_numpy() > 1).any():
        raise PlayLedgerValidationError("probability column(s) contain value(s) outside [0, 1]")
    totals = proba.sum(axis=1)
    bad = ~np.isclose(totals.to_numpy(), 1.0, atol=atol)
    if bad.any():
        bad_ids = df.loc[resolved].loc[bad, "play_id"].tolist()
        raise PlayLedgerValidationError(
            f"{int(bad.sum())} row(s) have probabilities that do not sum to ~1.0 "
            f"(first 10 play_id: {bad_ids[:10]})"
        )


def _validate_contact_luck_identity(
    df: pd.DataFrame, *, atol: float = _CONTACT_LUCK_IDENTITY_ATOL
) -> None:
    """Version 2.0: checks ONLY the self-referential identity
    `contact_luck_runs == observed_run_value - expected_run_value`.

    An earlier (Version 1.0) draft of this check ALSO cross-checked against
    `mlb_luck_score.scoring.contact_luck.compute_raw_contact_luck_runs`
    (which reconstructs `run_value(outcome_class) - E[run_value|proba]`,
    i.e. `Rc - E0`). That cross-check is REMOVED in Version 2.0: `contact_
    luck_runs` is now deliberately sourced from `Rf - E0` (`final_result_
    surprise`, via `play_ledger_export.py`), which no longer equals `Rc -
    E0` for any row with nonzero defensive/advancement execution
    contribution -- `outcome_class` documents the RECORDED contact result
    (Rc), not necessarily the batter-runner's final position after
    advancement (Rf), so reconstructing an "expected contact_luck_runs"
    from `outcome_class` alone is no longer a valid independent check (see
    module docstring's "outcome_class represents Rc, not Rf" section).
    This is not a weakening of the guardrail: the identity checked here is
    the actual DEFINING relationship of `contact_luck_runs`, regardless of
    which quantity (Rc - E0 or Rf - E0) it is defined to equal.
    """
    resolved = df["outcome_class"].notna()
    if not resolved.any():
        return
    sub = df.loc[resolved]
    reconstructed = sub["observed_run_value"].astype("float64") - sub["expected_run_value"].astype(
        "float64"
    )
    mismatch = ~np.isclose(
        sub["contact_luck_runs"].astype("float64").to_numpy(), reconstructed.to_numpy(), atol=atol
    )
    if mismatch.any():
        bad_ids = sub.loc[mismatch, "play_id"].tolist()
        raise PlayLedgerValidationError(
            f"{int(mismatch.sum())} row(s) have contact_luck_runs != observed_run_value - "
            f"expected_run_value (first 10 play_id: {bad_ids[:10]})"
        )


def resolved_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the outcome-resolved (`is_scored=True`) rows -- the exact
    subset that reproduces the published `eligible_batted_balls` denominator
    (see module docstring). Never use `len(df)` alone for that count.

    Filters on `is_scored` (the intended, validated-consistent status flag)
    rather than `outcome_class.notna()` directly -- equivalent by
    construction (`validate_play_ledger` enforces it), but this is the
    preferred consumer-facing entry point per the Phase 2 amendment.
    """
    return df.loc[df["is_scored"].astype("boolean").fillna(False)]
