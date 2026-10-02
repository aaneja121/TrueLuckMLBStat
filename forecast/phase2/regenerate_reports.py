"""Re-render the Phase 2 reports from the frozen numerical artifacts.

Presentation only. This module reads the already-written JSON artifacts and
re-renders the markdown from them. It runs no evaluation, fits nothing, opens
no snapshot, touches no parquet, and writes no result manifest -- the original
frozen manifests are preserved exactly as sealed.

Every re-render is recorded as an ERRATUM: the original report hash, the
corrected report hash, the exact text that changed, and a re-verification that
every other artifact in the namespace still hashes to what the frozen manifest
recorded. If any numerical artifact had moved, the erratum would say so and the
re-render would refuse to be described as cosmetic.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forecast.freeze_r1 import hash_file, hash_json
from forecast.phase2.h200_disclosure import render_disclosure_markdown
from forecast.phase2.h200_spec import H200_OUTPUTS_DIR
from forecast.phase2.phase2_config import PHASE2_OUTPUTS_DIR, assert_phase2_path_allowed
from forecast.phase2.phase2_report import render_phase2_report
from forecast.phase2.run_h200_report import render_h200_report

logger = logging.getLogger(__name__)


class ReportRegenerationError(RuntimeError):
    """Raised when a re-render would not be a presentation-only correction."""


def _read(path: Path) -> Any:
    assert_phase2_path_allowed(path)
    return json.loads(path.read_text())


def _text_changes(before: str, after: str) -> list[dict[str, str]]:
    """The exact lines that changed, as removed/added pairs."""
    matcher = difflib.SequenceMatcher(None, before.splitlines(), after.splitlines())
    changes: list[dict[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changes.append(
            {
                "operation": tag,
                "removed": "\n".join(before.splitlines()[i1:i2]),
                "added": "\n".join(after.splitlines()[j1:j2]),
            }
        )
    return changes


def _verify_numerical_artifacts_unchanged(
    outputs_dir: Path, *, manifest_name: str, report_name: str
) -> dict[str, Any]:
    """Re-hash every artifact the frozen manifest recorded, except the report.

    Raises:
        ReportRegenerationError: If any non-report artifact has moved.
    """
    manifest_path = outputs_dir / manifest_name
    if not manifest_path.exists():
        raise ReportRegenerationError(
            f"{manifest_name} is missing; a re-render cannot be checked against the "
            "frozen result without it."
        )
    manifest = _read(manifest_path)
    recorded: dict[str, str] = manifest["artifact_sha256"]

    moved: dict[str, dict[str, str]] = {}
    checked: dict[str, str] = {}
    for name, frozen_hash in recorded.items():
        if name == report_name:
            continue
        path = outputs_dir / name
        if not path.exists():
            moved[name] = {"frozen": frozen_hash, "current": "MISSING"}
            continue
        current = hash_file(path)
        checked[name] = current
        if current != frozen_hash:
            moved[name] = {"frozen": frozen_hash, "current": current}
    if moved:
        raise ReportRegenerationError(
            f"These numerical artifacts have moved since the freeze: {moved}. A report "
            "re-render is only a cosmetic erratum while every number is untouched."
        )
    return {
        "frozen_result_manifest": manifest_name,
        "frozen_result_manifest_sha256": manifest["manifest_sha256"],
        "frozen_result_manifest_preserved": True,
        "n_numerical_artifacts_reverified": len(checked),
        "all_numerical_artifacts_unchanged": True,
        "reverified_artifact_sha256": checked,
    }


def _write_erratum(
    outputs_dir: Path,
    *,
    erratum_name: str,
    report_name: str,
    original_hash: str,
    corrected_hash: str,
    changes: list[dict[str, str]],
    integrity: dict[str, Any],
    reason: str,
) -> Path:
    record: dict[str, Any] = {
        "erratum": "post-result cosmetic correction to the report renderer",
        "scope": "PRESENTATION ONLY",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "report": report_name,
        "original_report_sha256": original_hash,
        "corrected_report_sha256": corrected_hash,
        "reason": reason,
        "exact_text_changed": changes,
        "what_did_not_change": [
            "every computed metric",
            "the prespecified classification",
            "every prediction",
            "every bootstrap result",
            "every model artifact",
            "the underlying frozen result",
        ],
        "statement": (
            "No numerical artifact and no classification changed. The evaluation was "
            "not re-run: this report was re-rendered from the existing frozen "
            "numerical artifacts alone. The original frozen result manifest is "
            "preserved unmodified, and every artifact it recorded other than the "
            "report itself re-hashes to exactly what was sealed."
        ),
        "evaluation_rerun": False,
        "regenerated_from": "the frozen numerical artifacts only",
        "integrity_recheck": integrity,
    }
    record["erratum_sha256"] = hash_json(
        {k: v for k, v in record.items() if k != "recorded_at_utc"}
    )
    path = outputs_dir / erratum_name
    assert_phase2_path_allowed(path)
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
    logger.info("wrote %s", path)
    return path


SIGN_CONVENTION_ERRATUM_REASON = (
    "The displayed sign convention was not derived from the comparison the report "
    "prints. The H=200 report dumped the shared R1 `DELTA_SIGN_CONVENTION` constant "
    "verbatim, which names a different pair (shrunk deserved measured against shrunk "
    "realized) and therefore misnamed the challenger and reference; the H=100 report "
    "named the correct pair but as hardcoded prose that no artifact could contradict. "
    "Both now render the block from the comparison record's own `definition`, so a "
    "report cannot display a convention that disagrees with the delta beside it. The "
    "direction of the subtraction was correct throughout, and every delta, interval "
    "and classification was computed from the correct pair."
)


def regenerate_h100_report(*, outputs_dir: Path = PHASE2_OUTPUTS_DIR) -> dict[str, Any]:
    """Re-render the H=100 Phase 2 report from its frozen artifacts."""
    report_path = outputs_dir / "phase2_2026_report.md"
    original = report_path.read_text()
    original_hash = hash_file(report_path)

    integrity = _verify_numerical_artifacts_unchanged(
        outputs_dir,
        manifest_name="phase2_provenance_manifest.json",
        report_name="phase2_2026_report.md",
    )
    corrected = render_phase2_report(
        authorization=_read(outputs_dir / "phase2_authorization.json"),
        snapshot_manifest=_read(outputs_dir / "phase2_2026_snapshot_manifest.json"),
        metrics=_read(outputs_dir / "phase2_2026_metrics.json"),
        stability=_read(outputs_dir / "phase2_2026_stability.json"),
        shift=_read(outputs_dir / "phase2_2026_distribution_shift.json"),
    )
    report_path.write_text(corrected)
    corrected_hash = hash_file(report_path)

    _write_erratum(
        outputs_dir,
        erratum_name="phase2_2026_report_erratum.json",
        report_name="phase2_2026_report.md",
        original_hash=original_hash,
        corrected_hash=corrected_hash,
        changes=_text_changes(original, corrected),
        integrity=integrity,
        reason=SIGN_CONVENTION_ERRATUM_REASON,
    )
    return {"original": original_hash, "corrected": corrected_hash}


def regenerate_h200_report(*, outputs_dir: Path = H200_OUTPUTS_DIR) -> dict[str, Any]:
    """Re-render the H=200 report from its frozen artifacts."""
    report_path = outputs_dir / "h200_2026_report.md"
    original = report_path.read_text()
    original_hash = hash_file(report_path)

    integrity = _verify_numerical_artifacts_unchanged(
        outputs_dir,
        manifest_name="h200_2026_result_manifest.json",
        report_name="h200_2026_report.md",
    )
    corrected = render_h200_report(
        authorization=_read(outputs_dir / "h200_2026_authorization.json"),
        snapshot_manifest=_read(outputs_dir / "h200_2026_snapshot_manifest.json"),
        metrics=_read(outputs_dir / "h200_2026_metrics.json"),
        stability=_read(outputs_dir / "h200_2026_stability.json"),
        shift=_read(outputs_dir / "h200_2026_distribution_shift.json"),
        survivorship=_read(outputs_dir / "h200_2026_survivorship_comparison.json"),
        disclosure_markdown=render_disclosure_markdown(),
    )
    report_path.write_text(corrected)
    corrected_hash = hash_file(report_path)

    _write_erratum(
        outputs_dir,
        erratum_name="h200_2026_report_erratum.json",
        report_name="h200_2026_report.md",
        original_hash=original_hash,
        corrected_hash=corrected_hash,
        changes=_text_changes(original, corrected),
        integrity=integrity,
        reason=SIGN_CONVENTION_ERRATUM_REASON,
    )
    return {"original": original_hash, "corrected": corrected_hash}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h100-outputs-dir", type=Path, default=PHASE2_OUTPUTS_DIR)
    parser.add_argument("--h200-outputs-dir", type=Path, default=H200_OUTPUTS_DIR)
    args = parser.parse_args(argv)

    for label, result in (
        ("H=100", regenerate_h100_report(outputs_dir=args.h100_outputs_dir)),
        ("H=200", regenerate_h200_report(outputs_dir=args.h200_outputs_dir)),
    ):
        logger.info("%s report %s -> %s", label, result["original"][:16], result["corrected"][:16])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
