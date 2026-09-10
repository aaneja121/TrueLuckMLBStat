"""`forecast/phase2/resolution_preflight.py` -- readiness, not results.

The pre-flight exists so that a drifted freeze or a moved ledger is found
weeks before the resolution pass, not on the morning of it. Re-freezing to
make a pass run is re-freezing after seeing that something is wrong, which
is the one thing a freeze exists to prevent -- so "we checked early" has
real value and "we checked and the check was broken" does not.

These tests therefore cover the pre-flight's own failure modes: that its
report can actually be written, that a missing irreplaceable artifact is an
error rather than a shorter list, that a bundle proves its own contents,
and that it never claims authority it does not have.

Offline and synthetic throughout; the real artifacts are never modified.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from forecast.phase2 import resolution_preflight as rp


class TestTheReportCanActuallyBeWritten:
    """Regression: the first version put pandas DataFrames in the report and
    blew up in `json.dumps` only when `--write` was passed -- so the check
    passed, printed a clean summary, and wrote nothing. A readiness report
    that cannot be serialized is a readiness report nobody has."""

    def test_the_sealed_ledger_summary_is_json_serializable(self) -> None:
        summary = rp.check_sealed_ledgers()
        json.dumps(summary)  # must not raise

    def test_it_records_counts_not_frames(self) -> None:
        summary = rp.check_sealed_ledgers()
        for horizon, info in summary.items():
            assert isinstance(info["n_pending"], int), horizon
            assert isinstance(info["n_first_look_completed"], int), horizon

    def test_the_whole_report_is_json_serializable(self) -> None:
        json.dumps(rp.run_preflight())


class TestItNeverClaimsAuthority:
    def test_the_report_says_it_is_not_the_pass(self) -> None:
        report = rp.run_preflight()
        assert report["is_the_resolution_pass"] is False
        assert report["authorizes_nothing"] is True

    def test_a_clean_preflight_still_reports_the_season_block(self) -> None:
        """Passing every checkable condition must not read as 'ready to run'
        while three conditions remain unsatisfiable."""
        from datetime import date

        report = rp.run_preflight(today=date(2026, 9, 10))
        assert report["checkable_conditions_pass"] is True
        assert report["season_gate"]["season_has_ended"] is False
        assert len(report["still_blocked"]) == 3

    def test_the_season_block_is_reported_not_raised(self) -> None:
        """The calendar is not a defect."""
        from datetime import date

        gate = rp.check_season_gate(today=date(2026, 9, 10))
        assert gate["season_has_ended"] is False
        assert "2026-09-27" in gate["detail"]


class TestThePreservationManifestIsComplete:
    def test_every_irreplaceable_artifact_is_present_and_hashed(self) -> None:
        manifest = rp.build_preservation_manifest()
        assert manifest["n_files"] == sum(len(v) for v in rp.IRREPLACEABLE.values())
        for relative, entry in manifest["files"].items():
            assert len(entry["sha256"]) == 64, relative
            assert entry["bytes"] > 0, relative

    def test_it_covers_both_sealed_pending_ledgers(self) -> None:
        """The two files the whole second look depends on."""
        files = rp.build_preservation_manifest()["files"]
        assert "outputs/forecast_phase2/phase2_2026_pending_predictions.parquet" in files
        assert "outputs/forecast_phase2_h200/h200_2026_pending_predictions.parquet" in files

    def test_a_missing_artifact_is_an_error_not_a_shorter_list(self, monkeypatch) -> None:
        """A manifest that quietly described fewer files would certify an
        incomplete copy as complete."""
        monkeypatch.setitem(
            rp.IRREPLACEABLE, "outputs/forecast_phase2", ("definitely_not_a_real_file.parquet",)
        )
        with pytest.raises(rp.PreflightError, match="cannot be regenerated"):
            rp.build_preservation_manifest()


class TestTheBundleProvesItsOwnContents:
    @pytest.fixture
    def bundle(self, tmp_path: Path) -> Path:
        manifest = rp.build_preservation_manifest()
        path = tmp_path / "sealed.tar.gz"
        rp.write_preservation_bundle(path, manifest=manifest)
        return path

    def test_a_fresh_bundle_verifies(self, bundle: Path) -> None:
        assert rp.verify_preservation_bundle(bundle)["intact"] is True

    def test_it_carries_its_manifest_inside(self, bundle: Path) -> None:
        """So a copy found on another disk in six months can be checked
        without this repository."""
        with tarfile.open(bundle) as archive:
            assert "resolution_preservation_manifest.json" in archive.getnames()

    def test_a_tampered_file_is_detected(self, bundle: Path, tmp_path: Path) -> None:
        bad = tmp_path / "tampered.tar.gz"
        with tarfile.open(bundle) as src, tarfile.open(bad, "w:gz") as dst:
            for member in src.getmembers():
                data = src.extractfile(member).read()
                if member.name.endswith("phase2_2026_pending_predictions.parquet"):
                    data += b"\x00"
                    member.size = len(data)
                dst.addfile(member, io.BytesIO(data))
        with pytest.raises(rp.PreflightError, match="hash mismatch"):
            rp.verify_preservation_bundle(bad)

    def test_a_dropped_file_is_detected(self, bundle: Path, tmp_path: Path) -> None:
        bad = tmp_path / "incomplete.tar.gz"
        with tarfile.open(bundle) as src, tarfile.open(bad, "w:gz") as dst:
            for member in src.getmembers():
                if member.name.endswith("h200_2026_pending_predictions.parquet"):
                    continue
                dst.addfile(member, io.BytesIO(src.extractfile(member).read()))
        with pytest.raises(rp.PreflightError, match="missing"):
            rp.verify_preservation_bundle(bad)

    def test_an_archive_with_no_manifest_is_refused(self, tmp_path: Path) -> None:
        bad = tmp_path / "nomanifest.tar.gz"
        with tarfile.open(bad, "w:gz") as archive:
            data = b"{}"
            info = tarfile.TarInfo("something.json")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        with pytest.raises(rp.PreflightError, match="carries no preservation manifest"):
            rp.verify_preservation_bundle(bad)

    def test_the_bundled_bytes_are_the_real_bytes(self, bundle: Path) -> None:
        """Not just self-consistent -- the same as what is on disk."""
        relative = "outputs/forecast_phase2/phase2_2026_pending_predictions.parquet"
        on_disk = hashlib.sha256((rp.PROJECT_ROOT / relative).read_bytes()).hexdigest()
        with tarfile.open(bundle) as archive:
            bundled = hashlib.sha256(archive.extractfile(relative).read()).hexdigest()
        assert bundled == on_disk


class TestItReusesTheRunnersOwnGate:
    def test_the_freeze_chain_check_is_the_runners(self) -> None:
        """No second implementation of 'is the chain intact' that could drift
        from the one the pass actually uses."""
        source = Path(rp.__file__).read_text()
        assert "from forecast.phase2.run_resolution_evaluation import verify_full_chain" in source
        assert (
            "from forecast.phase2.run_resolution_evaluation import load_sealed_predictions"
            in source
        )

    def test_the_chain_currently_verifies_with_zero_drift(self) -> None:
        chain = rp.check_freeze_chain()
        assert chain["zero_drift_confirmed"] is True
        assert set(chain["manifest_sha256"]) == {
            "r1",
            "ridge",
            "hgb",
            "h200_spec",
            "resolution_spec",
        }
