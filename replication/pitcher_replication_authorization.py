"""Version 0.14: the maintainer's explicit authorization for ONE sealed 2025
use, for the Pitcher Contact Luck replication only.

`RESEARCH_RULES.md` ("The one narrow exception: the sealed Version 1.0 final
evaluation") permits 2025 to enter this repository exactly once, through the
sealed Version 1.0 entry point, which has already been used, and states that
being asked to run a second final evaluation is "a new, separate decision
requiring the user's explicit sign-off, not a natural extension of this one."

This module IS that separate decision, written down.

## Why this is a separate file, and NOT part of the frozen spec

`pitcher_replication_spec.py` was frozen BEFORE this authorization existed --
that ordering is the whole point of a pre-registration, and its
`AUTHORIZATION_STATUS["second_sealed_2025_evaluation_authorized"] = False`
correctly records the state *at freeze time*. It is deliberately not edited
here: editing it would change `spec_content_hash`, invalidate the freeze,
and destroy the evidence that the questions were fixed before the sign-off.

So the authorization is recorded ALONGSIDE the freeze and binds to it by
hash. This module is likewise absent from
`pitcher_replication_freeze.FROZEN_SOURCE_RELATIVE_PATHS`, for the same
reason: a frozen set that included its own authorization could not be
verified before the authorization existed.

Read `AUTHORIZED_FREEZE_CONTENT_HASH` as the answer to "authorization for
WHAT, exactly".

## Binding: this authorization cannot transfer

`authorization_binds_to` requires BOTH the freeze content hash and the spec
content hash to match the values recorded here. If the specification is
amended, or any frozen source file changes, the freeze's hashes change and
this authorization stops applying -- it does not silently follow the new
freeze. A new freeze requires a new, explicit sign-off.

## One-time use

The authorization permits exactly ONE evaluation. That is enforced by
`pitcher_replication_freeze.assert_no_replication_outputs_exist`, which
`assert_ready_for_2025` calls: once the replication namespace holds any
output, readiness fails and a second run is refused. This module records the
semantics; that guard enforces them.

## What this does NOT authorize

See `AUTHORIZATION_EXCLUSIONS`. In particular it does not authorize any 2026
access, any other research line's use of 2025, or any change to the frozen
metric, denominator, presentation rules, or estimators.
"""

from __future__ import annotations

from typing import Any

AUTHORIZATION_VERSION = "1.0.0"

#: The exact freeze this authorization is for. Both hashes must match for
#: the authorization to apply -- see `authorization_binds_to`.
AUTHORIZED_FREEZE_CONTENT_HASH = "58d8235b8d8324d55fe296875c1c8cdbd9857c5375be887f273822d1b1d40714"
AUTHORIZED_SPEC_CONTENT_HASH = "23320d4654b930112575fa7915b77cac618ff567ed3dabe5c314bf00002d0069"

#: The commit the authorized freeze recorded. Informational provenance: the
#: readiness contract does NOT require HEAD to equal this value, because
#: recording the authorization necessarily advances HEAD past the freeze.
AUTHORIZED_FREEZE_REPOSITORY_COMMIT = "6276461e954711ac1998f44394d5dafe152a6994"

AUTHORIZED_AT_UTC = "2026-09-08T16:29:49Z"

AUTHORIZATION_TEXT = (
    "I explicitly authorize ONE additional sealed use of 2025 for the Pitcher Contact "
    "Luck replication study. This authorization is narrow and applies only to the "
    "already-frozen Version 0.14 pitcher replication specification. Scope: 'One-time "
    "held-out 2025 full-season replication of Pitcher Contact Luck under the frozen "
    "Version 0.14 specification.' It permits exactly one evaluation of 2025 for the "
    "questions already preregistered in the frozen spec."
)

AUTHORIZATION_SCOPE = "pitcher_replication_only"

#: Every capability the maintainer explicitly withheld. Recorded so a later
#: reader cannot mistake "2025 was authorized" for "2025 is open".
AUTHORIZATION_EXCLUSIONS: tuple[str, ...] = (
    "changing the play-level metric",
    "retraining the contact model",
    "changing the denominator",
    "changing Starter-like / Reliever-like boundaries",
    "changing the sub-60 display rule",
    "changing cumulative total as the primary quantity",
    "changing per-100 as the secondary quantity",
    "changing the interval estimator",
    "changing resolving-power methodology",
    "changing Question F split-half methodology",
    "adding new metrics after seeing 2025",
    "selecting favorable subsets",
    "retuning thresholds",
    "redesigning the pitcher UI based on 2025 before the result is reported and frozen",
    "reading 2026",
    "touching Contact Forecast",
    "deploying anything",
)

#: The declarations the authorization is recorded with. These are statements
#: of fact at authorization time, not aspirations.
AUTHORIZATION_DECLARATIONS: dict[str, Any] = {
    "one_time_use": True,
    "one_time_use_enforced_by": (
        "pitcher_replication_freeze.assert_no_replication_outputs_exist, called by "
        "assert_ready_for_2025: once any replication output exists, readiness fails and a "
        "second run is refused."
    ),
    "no_2025_outcome_opened_at_authorization_time": True,
    "no_2025_outcome_opened_evidence": (
        "data/pitcher_replication/2025 and outputs/pitcher_replication/v0_14 were both "
        "absent, and the authorized freeze's own pre-outcome attestation records the same "
        "namespaces empty at freeze time."
    ),
    "is_a_second_sealed_2025_evaluation": True,
    "distinct_from_the_earlier_v1_final_evaluation": (
        "This is a SECOND sealed 2025 evaluation, through a second code path, distinct "
        "from the project's Version 1.0 final evaluation "
        "(evaluation/run_v1_final_evaluation.py). It does not reuse, reopen, or depend on "
        "that evaluation's data or seal."
    ),
    "prior_2025_use_does_not_authorize_reuse_elsewhere": (
        "Neither the Version 1.0 final evaluation nor this authorization permits any "
        "OTHER research line to read 2025. Each such use requires its own explicit "
        "sign-off."
    ),
    "does_not_extend_to_2026": (
        "2026 remains prospective scoring territory (Version 1.1) and is untouched by "
        "this authorization. No part of the pitcher replication may read 2026."
    ),
    "2024_remains_development": True,
    "does_not_transfer_to_a_different_freeze": (
        "Bound to AUTHORIZED_FREEZE_CONTENT_HASH and AUTHORIZED_SPEC_CONTENT_HASH. If "
        "either changes, this authorization stops applying and a new sign-off is required."
    ),
}


def authorization_record() -> dict[str, Any]:
    """The complete authorization as a plain, JSON-serializable dict."""
    return {
        "authorization_version": AUTHORIZATION_VERSION,
        "authorization_text": AUTHORIZATION_TEXT,
        "authorized_at_utc": AUTHORIZED_AT_UTC,
        "scope": AUTHORIZATION_SCOPE,
        "authorized_freeze_content_hash": AUTHORIZED_FREEZE_CONTENT_HASH,
        "authorized_spec_content_hash": AUTHORIZED_SPEC_CONTENT_HASH,
        "authorized_freeze_repository_commit": AUTHORIZED_FREEZE_REPOSITORY_COMMIT,
        "exclusions": list(AUTHORIZATION_EXCLUSIONS),
        "declarations": AUTHORIZATION_DECLARATIONS,
    }


def authorization_binds_to(freeze: Any) -> tuple[bool, list[str]]:
    """Does this authorization apply to `freeze`?

    Both the freeze content hash and the spec content hash must match. A
    mismatch in either means the authorized specification is not the one in
    front of us, and the authorization must NOT transfer.

    Returns:
        `(binds, reasons)` -- `reasons` lists every mismatch when it does
        not bind, and is empty when it does.
    """
    reasons: list[str] = []
    actual_freeze_hash = freeze.freeze_content_hash()
    if actual_freeze_hash != AUTHORIZED_FREEZE_CONTENT_HASH:
        reasons.append(
            f"freeze_content_hash mismatch: authorized "
            f"{AUTHORIZED_FREEZE_CONTENT_HASH}, present {actual_freeze_hash}"
        )
    if freeze.spec_content_hash != AUTHORIZED_SPEC_CONTENT_HASH:
        reasons.append(
            f"spec_content_hash mismatch: authorized {AUTHORIZED_SPEC_CONTENT_HASH}, "
            f"present {freeze.spec_content_hash}"
        )
    return (not reasons, reasons)
