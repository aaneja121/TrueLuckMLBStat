"""Pre-flight for the end-of-season resolution pass: check now what can be
checked now, and preserve what cannot be rebuilt.

## Why this runs weeks before the pass

The resolution pass has a seven-condition gate. Three of them cannot be
satisfied before the regular season ends -- a snapshot reaching the final
date, exactly one such snapshot pinned, and the maintainer's authorization
in the moment. The other four are true or false today:

    3. this specification is frozen and verifies with zero drift
    4. the R1, ridge, HGB and h200_spec freezes verify with zero drift
    5. both frozen result manifests verify, and both pending ledgers
       re-hash to what they recorded
    1. (first half) the season end date is recorded and verified

Discovering a drifted freeze on the day is the bad case: the pass is a
prespecified second look, so re-freezing to make it run would be re-freezing
after seeing that something is wrong -- exactly what a freeze exists to make
impossible. Discovering it weeks early is a repair with slack.

This module runs NOTHING new. It calls the resolution runner's own gate
functions, so there is no second implementation of "is the chain intact"
that could drift from the one the pass actually uses.

## Why it also writes a preservation manifest

The sealed prediction ledgers are the only irreplaceable objects in this
study. `predictions_regenerated` is false by design: a prediction made
before its outcome was known cannot be recreated afterward, at any cost,
by anyone. They total a few hundred kilobytes, live under a gitignored
`outputs/` namespace, and exist on exactly one machine.

The repository already treats snapshots this way -- write-once durable
archival, verified by hash -- because a scoring run is expensive to
reproduce. These are not expensive to reproduce; they are IMPOSSIBLE to
reproduce, which is a stronger reason and currently has weaker protection.

This module does not decide where a durable copy lives. It writes the
manifest that lets any copy, anywhere, be proved identical to what was
sealed -- and names exactly which files a copy must contain.

## What this is not

It is not the resolution pass, it opens no outcome, and a clean pre-flight
authorizes nothing. It is a readiness check whose only possible results are
"the parts that can be broken today are not broken" and a list of what
still blocks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from forecast.forecast_config import PROJECT_ROOT

logger = logging.getLogger(__name__)

PREFLIGHT_VERSION = "contact_forecast_resolution_preflight_v1"

#: Where the pre-flight writes. The resolution namespace, because that is
#: what these artifacts describe.
PREFLIGHT_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "forecast_phase2_resolution"

#: The artifacts a durable copy MUST contain, by horizon namespace. These are
#: the sealed predictions and the manifests that prove what they were, plus
#: the errata that explain every recorded difference. Reports are included
#: because a result nobody can read is not preserved.
IRREPLACEABLE: dict[str, tuple[str, ...]] = {
    "outputs/forecast_phase2": (
        "phase2_2026_pending_predictions.parquet",
        "phase2_2026_completed_evaluation.parquet",
        "phase2_2026_predictions.parquet",
        "phase2_provenance_manifest.json",
        "phase2_authorization.json",
        "phase2_2026_snapshot_manifest.json",
        "phase2_2026_metrics.json",
        "phase2_2026_bootstrap.json",
        "phase2_2026_stability.json",
        "phase2_2026_distribution_shift.json",
        "phase2_2026_report.md",
        "phase2_2026_report_erratum.json",
    ),
    "outputs/forecast_phase2_h200": (
        "h200_2026_pending_predictions.parquet",
        "h200_2026_completed_evaluation.parquet",
        "h200_2026_predictions.parquet",
        "h200_2026_result_manifest.json",
        "h200_2026_authorization.json",
        "h200_2026_snapshot_manifest.json",
        "h200_2026_metrics.json",
        "h200_2026_bootstrap.json",
        "h200_2026_stability.json",
        "h200_2026_distribution_shift.json",
        "h200_2026_survivorship_comparison.json",
        "h200_2026_exposure_disclosure.json",
        "h200_spec_freeze_manifest.json",
        "h200_provenance_answer.json",
        "h200_2026_report.md",
        "h200_2026_report_erratum.json",
    ),
    "outputs/forecast_phase2_resolution": (
        "resolution_specification.json",
        "resolution_spec_freeze_manifest.json",
        "resolution_amendment_record.json",
        "resolution_multiplicity_disclosure.json",
        "r1_freeze_manifest.json",
        "ridge_freeze_manifest.json",
        "hgb_freeze_manifest.json",
        "h200_spec_freeze_manifest.json",
    ),
}


class PreflightError(RuntimeError):
    """The pre-flight found something that would block the pass. Raised only
    for conditions that are wrong NOW -- never for a condition that is
    merely not yet satisfiable, which is reported instead."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_freeze_chain() -> dict[str, Any]:
    """Gate conditions 3 and 4, via the runner's own verifier."""
    from forecast.phase2.run_resolution_evaluation import verify_full_chain

    return verify_full_chain()


def check_sealed_ledgers() -> dict[str, Any]:
    """Gate condition 5, via the runner's own loader.

    `load_sealed_predictions` re-hashes each ledger against the result
    manifest that recorded it and refuses on any mismatch, which is exactly
    the check the pass will make -- so a pass here means the pass will not
    fail there for this reason.
    """
    from forecast.phase2.run_resolution_evaluation import load_sealed_predictions

    out: dict[str, Any] = {}
    for horizon_key in ("H100", "H200"):
        sealed = load_sealed_predictions(horizon_key)
        out[horizon_key] = {
            "ledger_sha256": sealed["ledger_sha256"],
            "result_manifest_sha256": sealed["result_manifest_sha256"],
            # Counts, never the frames themselves: this report is written as
            # JSON, and the loader hands back DataFrames.
            "n_pending": int(len(sealed["pending"])),
            "n_first_look_completed": int(len(sealed["first_look_completed"])),
            "horizon": sealed["horizon"],
            "cutoff": sealed["cutoff"],
        }
    return out


def check_season_gate(*, today: date | None = None) -> dict[str, Any]:
    """The three season-blocked conditions, reported rather than raised.

    A pre-flight that failed because the season has not ended yet would be
    reporting the calendar as a defect.
    """
    from forecast.phase2.run_resolution_evaluation import assert_season_has_ended

    today = today or datetime.now(UTC).date()
    try:
        ended = assert_season_has_ended(today=today)
        return {"season_has_ended": True, "detail": ended, "blocked": []}
    except Exception as exc:  # noqa: BLE001 -- the runner's own gate error
        return {
            "season_has_ended": False,
            "detail": str(exc),
            "blocked": [
                "the 2026 regular season has not ended",
                "no snapshot reaches the season end date yet",
                "the maintainer's in-the-moment authorization is given on the day",
            ],
        }


def build_preservation_manifest() -> dict[str, Any]:
    """Hash every irreplaceable artifact, so any copy can be proved identical.

    Missing files are an error, not an omission: a manifest that silently
    described a smaller set than exists would certify an incomplete copy.
    """
    entries: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    total = 0
    for namespace, names in IRREPLACEABLE.items():
        for name in names:
            path = PROJECT_ROOT / namespace / name
            relative = f"{namespace}/{name}"
            if not path.is_file():
                missing.append(relative)
                continue
            size = path.stat().st_size
            total += size
            entries[relative] = {"sha256": _sha256(path), "bytes": size}
    if missing:
        raise PreflightError(
            "irreplaceable artifacts are missing from this machine:\n  "
            + "\n  ".join(missing)
            + "\nThese cannot be regenerated. If they are gone, the resolution pass "
            "cannot run -- restore them from a durable copy before doing anything else."
        )
    return {
        "files": entries,
        "n_files": len(entries),
        "total_bytes": total,
        "why_irreplaceable": (
            "These predictions were sealed before their outcomes were known. A prediction "
            "made after the fact is a different object, so no amount of compute recreates "
            "them. `predictions_regenerated` is false in the resolution specification for "
            "this reason."
        ),
    }


def write_preservation_bundle(destination: Path, *, manifest: dict[str, Any]) -> dict[str, Any]:
    """Write one self-describing archive of every irreplaceable artifact.

    The manifest goes INSIDE the archive as well as beside it, so a copy
    that turns up on some other disk in six months can be verified without
    this repository -- the archive says what it should contain and what each
    file should hash to.

    Deliberately not "sync to R2 for you": where the durable copy lives is a
    decision with credentials and retention attached, and this writes the
    thing that makes any such copy checkable rather than choosing one.
    """
    import tarfile

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for relative in sorted(manifest["files"]):
            archive.add(PROJECT_ROOT / relative, arcname=relative)
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo("resolution_preservation_manifest.json")
        info.size = len(manifest_bytes)
        import io

        archive.addfile(info, io.BytesIO(manifest_bytes))

    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "contains": manifest["n_files"] + 1,
    }


def verify_preservation_bundle(archive_path: Path) -> dict[str, Any]:
    """Prove an archive holds exactly what its own manifest says.

    The check that makes a durable copy worth having: without it, "we have a
    backup" is a belief rather than a fact.
    """
    import tarfile

    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())
        # Membership first: `extractfile` raises KeyError for a name that is
        # not in the archive, and an archive with no manifest is a bundle we
        # cannot check at all -- which is the loudest possible failure, not a
        # stray KeyError from a stdlib call.
        if "resolution_preservation_manifest.json" not in names:
            raise PreflightError(f"{archive_path} carries no preservation manifest")
        manifest_member = archive.extractfile("resolution_preservation_manifest.json")
        if manifest_member is None:
            raise PreflightError(f"{archive_path} has an unreadable preservation manifest")
        manifest = json.loads(manifest_member.read())

        problems: list[str] = []
        for relative, recorded in sorted(manifest["files"].items()):
            if relative not in names:
                problems.append(f"missing: {relative}")
                continue
            member = archive.extractfile(relative)
            if member is None:
                problems.append(f"unreadable: {relative}")
                continue
            if hashlib.sha256(member.read()).hexdigest() != recorded["sha256"]:
                problems.append(f"hash mismatch: {relative}")

    if problems:
        raise PreflightError(
            f"{archive_path} does not match its own manifest:\n  " + "\n  ".join(problems)
        )
    return {"archive": str(archive_path), "files_verified": len(manifest["files"]), "intact": True}


def run_preflight(*, today: date | None = None) -> dict[str, Any]:
    """Everything checkable now, plus what still blocks. Opens no outcome."""
    report: dict[str, Any] = {
        "preflight_version": PREFLIGHT_VERSION,
        "ran_at_utc": datetime.now(UTC).isoformat(),
        "is_the_resolution_pass": False,
        "authorizes_nothing": True,
        "freeze_chain": check_freeze_chain(),
        "sealed_ledgers": check_sealed_ledgers(),
        "season_gate": check_season_gate(today=today),
        "preservation": build_preservation_manifest(),
    }
    report["checkable_conditions_pass"] = bool(
        report["freeze_chain"]["zero_drift_confirmed"] and report["sealed_ledgers"]
    )
    report["still_blocked"] = report["season_gate"]["blocked"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the report and preservation manifest into the resolution namespace.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Also write a self-describing .tar.gz of every irreplaceable artifact to this "
        "path, then verify it against its own manifest. Move it somewhere durable.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        report = run_preflight()
    except Exception as exc:
        print(f"PREFLIGHT FAILED: {exc}")
        return 2

    chain = report["freeze_chain"]
    print(f"[preflight] freeze chain zero drift : {chain['zero_drift_confirmed']}")
    for stage, digest in chain["manifest_sha256"].items():
        print(f"[preflight]   {stage:16s} {digest[:16]}")
    for key, info in report["sealed_ledgers"].items():
        print(
            f"[preflight] sealed ledger {key}: verified, "
            f"n_pending={info['n_pending']}, H={info['horizon']}"
        )
    pres = report["preservation"]
    print(
        f"[preflight] irreplaceable artifacts : {pres['n_files']} files, {pres['total_bytes']} bytes"
    )
    season = report["season_gate"]
    print(f"[preflight] season has ended        : {season['season_has_ended']}")
    if not season["season_has_ended"]:
        print(f"[preflight]   {season['detail']}")
    print(f"[preflight] CHECKABLE CONDITIONS PASS: {report['checkable_conditions_pass']}")
    print("[preflight] This authorizes nothing and is not the resolution pass.")

    if args.write:
        PREFLIGHT_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = PREFLIGHT_OUTPUTS_DIR / "resolution_preflight_report.json"
        manifest_path = PREFLIGHT_OUTPUTS_DIR / "resolution_preservation_manifest.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest_path.write_text(json.dumps(pres, indent=2, sort_keys=True) + "\n")
        print(f"[preflight] wrote {report_path}")
        print(f"[preflight] wrote {manifest_path}")

    if args.bundle is not None:
        written = write_preservation_bundle(args.bundle, manifest=pres)
        verified = verify_preservation_bundle(args.bundle)
        print(
            f"[preflight] bundle {written['path']} "
            f"({written['bytes']} bytes, sha256 {written['sha256'][:16]})"
        )
        print(f"[preflight] bundle verified against its own manifest: {verified['intact']}")
        print("[preflight] MOVE THIS OFF THIS MACHINE -- it is the only copy of the sealed")
        print("[preflight] predictions, and they cannot be regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
