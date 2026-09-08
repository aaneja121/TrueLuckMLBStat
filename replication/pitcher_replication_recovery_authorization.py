"""Version 0.14: the maintainer's authorization for ONE technical recovery of
the already-exposed 2025 Pitcher Contact Luck replication.

**This is not a new first look at 2025.** Execution `8edc32d6ca8830ce` opened
and downloaded the held-out season at 2026-09-08T17:00:33Z and failed inside
the frozen scorer's season precondition guard -- before any model was fit,
any prediction produced, any pitcher aggregate computed, any question A-G
answered, or any classification made. The original one-time authorization is
CONSUMED and is never reused here.

## Revision 2 -- this authorization supersedes revision 1

The revision-1 sign-off was cryptographically bound to recovery manifest
revision 1 and correctly became void when the recovery-control
implementation changed. It is superseded here and may not be reused.

**The revision-1 attempt was NOT consumed.** It failed during recovery
readiness -- before `recovery_receipt.json`, before any second read of 2025,
before any model fit, prediction, pitcher aggregation, question A-G or
classification. See incident record sequence 2.

## Ordering, which is the whole point

    recovery code sealed  ->  recovery manifest frozen  ->  authorization granted

This module was written LAST, and deliberately sits outside
`pitcher_replication_recovery.RECOVERY_SOURCE_RELATIVE_PATHS`, so recording
it cannot alter the manifest it authorizes. The manifest is never rebuilt to
accommodate a later sign-off -- exactly as the original freeze was never
rebuilt to accommodate the original one.

## Fail-closed binding

`recovery_authorization_binds_to` requires the live recovery manifest hash to
equal `AUTHORIZED_RECOVERY_MANIFEST_HASH` exactly. If the recovery code, the
2025 artifact bytes, the original receipt, the original execution manifest,
the incident record or the research freeze change, the manifest hash changes
and this authorization stops applying. It never transfers, and a new
sign-off is then required.

## One attempt

The authorization permits exactly ONE recovery execution.
`pitcher_replication_recovery.assert_no_recovery_receipt` enforces that: once
`recovery_receipt.json` exists the authorization is spent, and a further
attempt is refused. A failure after the receipt is written requires a NEW
explicit incident review -- never an automatic retry, and never a third
execution on this sign-off.
"""

from __future__ import annotations

from typing import Any

RECOVERY_AUTHORIZATION_VERSION = "2.0.0"

#: Which recovery manifest revision this sign-off covers.
AUTHORIZED_RECOVERY_MANIFEST_REVISION = 2

#: The superseded revision-1 sign-off. Recorded so the supersession is
#: legible, and so a stale hash can never be mistaken for a live one.
SUPERSEDED_RECOVERY_MANIFEST_HASH = (
    "24e2ea817e741178fefdb5bd844603f67651983b1bd7eea508b9dbc77a5406b6"
)

#: The exact recovery manifest this authorization is for. Read from
#: `artifacts/pitcher_replication/v0_14/recovery_manifest_rev2.json` and verified
#: byte-for-byte before this file was written -- not copied from a prompt.
AUTHORIZED_RECOVERY_MANIFEST_HASH = (
    "b05b8f1b273eebfc26d6a62fba3608fabfb2cf41929c63bef82270dcb1925eb6"
)

#: The exposure this recovery descends from.
ORIGINAL_EXECUTION_ID = "8edc32d6ca8830ce"
ORIGINAL_RECEIPT_SHA256 = "40a236a8b169d8fad016490ebddec86acfe82473aab5e8d0e9f6d30437e238f2"
ORIGINAL_EXECUTION_MANIFEST_HASH = (
    "cb34daf7af93f77fb05fb91acbefd38a183a8f095f8b0e5e0f30a19a7ef0721f"
)
ORIGINAL_AUTHORIZATION_CONTENT_HASH = (
    "4a25b77e21b8df6d11eb382fa23f95b53a35911a2bb9935c83eed912aa7a85b1"
)
#: The append-only incident log as of this authorization. Two records: the
#: original post-ingestion failure (sequence 1) and the recovery-readiness
#: failure that did NOT consume the previous attempt (sequence 2).
INCIDENT_RECORD_SHA256 = "4c15b6fe61562aa0e04afc32f675fd449a0369c26ce53ac27097849749515d4f"
INCIDENT_CHAIN_LATEST_SEQUENCE = 2
INCIDENT_CHAIN_RECORD_COUNT = 2

#: The authoritative research state, unchanged throughout the incident.
FREEZE_CONTENT_HASH = "58d8235b8d8324d55fe296875c1c8cdbd9857c5375be887f273822d1b1d40714"
SPEC_CONTENT_HASH = "23320d4654b930112575fa7915b77cac618ff567ed3dabe5c314bf00002d0069"

CORRECTION_COMMIT = "185f2621de19b75d4489fe83a8cecb93f33e0ca1"
RECOVERY_CONTROL_COMMIT = "ec1f5e303c9fc937894afb092ad0e4ef0c40a36a"
GATE_SEPARATION_COMMIT = "631d5dea82b1c5be638ec8c68047686dcc9f9c2e"

AUTHORIZED_AT_UTC = "2026-09-08T18:58:55Z"

AUTHORIZATION_TEXT = (
    "I explicitly authorize ONE technical recovery attempt for the already-exposed 2025 "
    "Pitcher Contact Luck replication under recovery manifest revision 2. This "
    "authorization supersedes the prior recovery authorization ONLY because that prior "
    "authorization was cryptographically bound to recovery manifest revision 1 and "
    "therefore correctly became invalid when the recovery-control implementation changed. "
    "The prior authorized recovery attempt was NOT consumed: it failed during recovery "
    "readiness, before recovery_receipt.json was created and before any second read of "
    "2025. 2025 remains already exposed from the ORIGINAL failed execution and this "
    "recovery must never be described as a first look.\n\nRevision-1 authorization text, "
    "retained verbatim for the record: 'I explicitly authorize ONE technical recovery "
    "attempt for the already-exposed 2025 Pitcher Contact Luck replication. This is NOT a "
    "new first look at 2025. The original "
    "one-time 2025 authorization was consumed by execution 8edc32d6ca8830ce. That "
    "execution opened/downloaded 2025 but failed before any model fit, prediction, pitcher "
    "aggregation, A-G diagnostic, or replication classification. The failure has been "
    "diagnosed as a pure execution/plumbing defect: the runner passed the full 2021-2024 "
    "development dataframe to a helper whose contract requires pre-filtered TRAIN_SEASONS "
    "only. The frozen intended contract remains: TRAIN_SEASONS = 2021, 2022, 2023; 2024 "
    "must not enter training; 2025 is evaluation only.'"
)

#: Everything the recovery execution MAY do.
AUTHORIZATION_PERMITS: tuple[str, ...] = (
    "reuse the three already-downloaded and hash-verified 2025 artifacts",
    "train the frozen Contact Luck architecture on 2021-2023 only",
    "evaluate 2025 under the frozen Version 0.14 specification",
    "compute the preregistered questions A-G",
    "apply the frozen REPLICATED / REVISE / NO_GO classification",
    "write and seal the prespecified replication outputs",
)

#: Everything explicitly withheld. Recorded so "the recovery was authorized"
#: can never be read as "2025 is open".
AUTHORIZATION_EXCLUSIONS: tuple[str, ...] = (
    "redownload 2025",
    "change any 2025 artifact",
    "add 2024 to training",
    "train on 2025",
    "modify model architecture",
    "modify eligibility",
    "modify attribution",
    "modify run values",
    "modify denominator rules",
    "change Starter-like / Reliever-like boundaries",
    "change the sub-60 display rule",
    "change cumulative total as primary",
    "change per-100 as secondary",
    "change interval or resolving-power estimators",
    "change Question F",
    "add metrics after exposure",
    "choose favorable subsets",
    "change the package-level classification rule",
    "inspect or use 2026",
    "modify Contact Forecast",
    "deploy anything",
)

AUTHORIZATION_DECLARATIONS: dict[str, Any] = {
    "season_2025_already_exposed": True,
    "exposure_note": (
        f"2025 was opened by execution {ORIGINAL_EXECUTION_ID} at 2026-09-08T17:00:33Z. This "
        "recovery is a RESUMPTION after exposure."
    ),
    "is_a_first_look": False,
    "never_describe_as_a_first_look": (
        "This recovery must never be reported, committed, documented or described as a "
        "first look at 2025. The original execution-start receipt must never be deleted or "
        "replaced to make it appear so."
    ),
    "original_authorization_consumed": True,
    "original_authorization_not_reused": (
        f"The original one-time authorization ({ORIGINAL_AUTHORIZATION_CONTENT_HASH}) was "
        f"spent by execution {ORIGINAL_EXECUTION_ID}. It is not consulted by the recovery "
        "gate and may not stand in for this sign-off."
    ),
    "one_recovery_attempt_only": True,
    "one_attempt_enforced_by": (
        "pitcher_replication_recovery.assert_no_recovery_receipt: once recovery_receipt."
        "json exists the authorization is spent and a further attempt is refused."
    ),
    "creates_no_further_retry_entitlement": (
        "If the recovery execution fails after the recovery receipt is written, STOP. A "
        "third execution requires a new explicit incident review."
    ),
    "revision": AUTHORIZED_RECOVERY_MANIFEST_REVISION,
    "supersedes_revision_1_authorization": (
        f"The revision-1 sign-off ({SUPERSEDED_RECOVERY_MANIFEST_HASH}) is SUPERSEDED and "
        "may not be reused. It became void correctly, by its own fail-closed rule, when "
        "the recovery-control implementation changed."
    ),
    "previous_recovery_launch_did_not_consume_the_attempt": (
        "The revision-1 attempt failed during recovery READINESS -- before "
        "recovery_receipt.json, before any second read of 2025, and before any model fit, "
        "prediction, pitcher aggregation, question A-G or classification. Incident record "
        "sequence 2 records this."
    ),
    "no_permission_to_redownload_2025": True,
    "no_2026_authorization": (
        "2026 remains prospective scoring territory and is untouched. Nothing in the "
        "recovery may read, inspect or use it."
    ),
    "recovery_is_technical_not_methodological": True,
    "no_result_was_computed_before_failure": (
        "The guard raised before the eligibility pipeline and before every train_* call. No "
        "component model was fit, no 2025 prediction produced, no pitcher aggregate "
        "computed, no question A-G answered, and no classification made."
    ),
    "binds_by_hash_and_fails_closed": (
        "Bound to AUTHORIZED_RECOVERY_MANIFEST_HASH. If the recovery manifest changes for "
        "any reason -- recovery code, 2025 artifact bytes, the original receipt, the "
        "original execution manifest, the incident record, or the research freeze -- this "
        "authorization stops applying and a new sign-off is required."
    ),
}


def recovery_authorization_record() -> dict[str, Any]:
    """The complete recovery authorization as a plain, JSON-serializable dict."""
    return {
        "recovery_authorization_version": RECOVERY_AUTHORIZATION_VERSION,
        "authorized_recovery_manifest_revision": AUTHORIZED_RECOVERY_MANIFEST_REVISION,
        "superseded_recovery_manifest_hash": SUPERSEDED_RECOVERY_MANIFEST_HASH,
        "incident_chain_latest_sequence": INCIDENT_CHAIN_LATEST_SEQUENCE,
        "incident_chain_record_count": INCIDENT_CHAIN_RECORD_COUNT,
        "gate_separation_commit": GATE_SEPARATION_COMMIT,
        "authorization_text": AUTHORIZATION_TEXT,
        "authorized_at_utc": AUTHORIZED_AT_UTC,
        "authorized_recovery_manifest_hash": AUTHORIZED_RECOVERY_MANIFEST_HASH,
        "original_execution_id": ORIGINAL_EXECUTION_ID,
        "original_receipt_sha256": ORIGINAL_RECEIPT_SHA256,
        "original_execution_manifest_hash": ORIGINAL_EXECUTION_MANIFEST_HASH,
        "original_authorization_content_hash": ORIGINAL_AUTHORIZATION_CONTENT_HASH,
        "incident_record_sha256": INCIDENT_RECORD_SHA256,
        "freeze_content_hash": FREEZE_CONTENT_HASH,
        "spec_content_hash": SPEC_CONTENT_HASH,
        "correction_commit": CORRECTION_COMMIT,
        "recovery_control_commit": RECOVERY_CONTROL_COMMIT,
        "permits": list(AUTHORIZATION_PERMITS),
        "exclusions": list(AUTHORIZATION_EXCLUSIONS),
        "declarations": AUTHORIZATION_DECLARATIONS,
    }


def recovery_authorization_binds_to(manifest_hash: str) -> tuple[bool, list[str]]:
    """Does this authorization apply to the recovery manifest `manifest_hash`?

    Fail-closed: only an exact match authorizes. Any change to the recovery
    manifest -- for any reason -- changes its hash and revokes this sign-off
    rather than silently carrying it forward.

    Returns:
        `(authorized, reasons)`. `reasons` is empty when it binds.
    """
    if manifest_hash == SUPERSEDED_RECOVERY_MANIFEST_HASH:
        return (
            False,
            [
                "that is recovery manifest REVISION 1, whose authorization is superseded "
                "and may not be reused. This sign-off covers revision "
                f"{AUTHORIZED_RECOVERY_MANIFEST_REVISION} "
                f"({AUTHORIZED_RECOVERY_MANIFEST_HASH}) only. The authorization does not "
                "transfer to a different recovery manifest -- obtain a new explicit "
                "sign-off."
            ],
        )
    if manifest_hash != AUTHORIZED_RECOVERY_MANIFEST_HASH:
        return (
            False,
            [
                "recovery manifest hash mismatch: this authorization covers "
                f"{AUTHORIZED_RECOVERY_MANIFEST_HASH}, but the manifest presented is "
                f"{manifest_hash}. The authorization does not transfer to a different "
                "recovery manifest -- obtain a new explicit sign-off."
            ],
        )
    return (True, [])
