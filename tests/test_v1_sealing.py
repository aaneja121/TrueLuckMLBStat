"""Contact Luck v1.0, Task 33 category 9: sealing.

Every test uses a temp directory for `seal_path`/`outputs_dir`/`artifacts_dir`
-- none of them touch the real `artifacts/final_evaluation/v1/seal.json`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import run_v1_final_evaluation as rv1


class _FakeManifest:
    repository_commit = "commit-abc123"

    def content_hash(self) -> str:
        return "manifest-hash-abc123"


@pytest.fixture(autouse=True)
def _patch_artifacts_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`write_seal`'s namespace guard (see test_v1_namespace_isolation.py for
    its own dedicated tests) checks `seal_path.parent` against the real
    production `FINAL_EVALUATION_ARTIFACTS_DIR` by default -- these sealing
    -SEMANTICS tests use `tmp_path`-rooted paths freely, so this fixture
    treats `tmp_path` itself as the namespace root for the duration of each
    test here.
    """
    monkeypatch.setattr(rv1, "FINAL_EVALUATION_ARTIFACTS_DIR", tmp_path)


def _write_seal(seal_path: Path, *, defect_fix_of: str | None = None) -> rv1.EvaluationSeal:
    return rv1.write_seal(
        manifest=_FakeManifest(),  # type: ignore[arg-type]
        report={"outcome_classification": {"outcome": "validated_as_frozen"}},
        seal_path=seal_path,
        defect_fix_of=defect_fix_of,
    )


# ---------------------------------------------------------------------------
# Basic write/read round trip
# ---------------------------------------------------------------------------


def test_no_seal_file_reads_as_none(tmp_path: Path) -> None:
    assert rv1.read_existing_seal(tmp_path / "seal.json") is None


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    seal_path = tmp_path / "artifacts" / "final_evaluation" / "v1" / "seal.json"
    written = _write_seal(seal_path)
    read_back = rv1.read_existing_seal(seal_path)
    assert read_back == written


def test_seal_written_only_after_report_content_is_final(tmp_path: Path) -> None:
    """`write_seal` hashes the exact `report` dict it is given -- confirms
    the seal's `report_content_hash` is a function of the report content,
    not a placeholder written before the report was finished.
    """
    seal_path = tmp_path / "seal.json"
    report_a = {"outcome_classification": {"outcome": "validated_as_frozen"}}
    report_b = {"outcome_classification": {"outcome": "final_evaluation_failed"}}
    seal_a = rv1.write_seal(manifest=_FakeManifest(), report=report_a, seal_path=seal_path)  # type: ignore[arg-type]
    seal_path.unlink()
    seal_b = rv1.write_seal(manifest=_FakeManifest(), report=report_b, seal_path=seal_path)  # type: ignore[arg-type]
    assert seal_a.report_content_hash != seal_b.report_content_hash


# ---------------------------------------------------------------------------
# Corrupted/incomplete seals fail safely
# ---------------------------------------------------------------------------


def test_corrupted_json_seal_fails_safely(tmp_path: Path) -> None:
    seal_path = tmp_path / "seal.json"
    seal_path.write_text("{not valid json")
    with pytest.raises(rv1.SealError, match="not valid JSON"):
        rv1.read_existing_seal(seal_path)


def test_incomplete_seal_missing_field_fails_safely(tmp_path: Path) -> None:
    seal_path = tmp_path / "seal.json"
    seal_path.write_text('{"manifest_content_hash": "a"}')
    with pytest.raises(rv1.SealError):
        rv1.read_existing_seal(seal_path)


def test_seal_file_that_is_not_a_json_object_fails_safely(tmp_path: Path) -> None:
    seal_path = tmp_path / "seal.json"
    seal_path.write_text("[1, 2, 3]")
    with pytest.raises(rv1.SealError):
        rv1.read_existing_seal(seal_path)


def test_corrupted_seal_is_never_treated_as_no_prior_seal(tmp_path: Path) -> None:
    """A corrupted seal must not be silently overwritten by a fresh run --
    it should surface as an error, not as `assert_no_unacknowledged_prior_
    seal` quietly returning `None`.
    """
    seal_path = tmp_path / "seal.json"
    seal_path.write_text("{not valid json")
    with pytest.raises(rv1.SealError):
        rv1.assert_no_unacknowledged_prior_seal(seal_path, acknowledge_defect_fix=False)


# ---------------------------------------------------------------------------
# Rerun rejection / defect-fix acknowledgement
# ---------------------------------------------------------------------------


def test_existing_valid_seal_rejects_an_ordinary_rerun(tmp_path: Path) -> None:
    seal_path = tmp_path / "seal.json"
    _write_seal(seal_path)
    with pytest.raises(rv1.SealError, match="acknowledge_defect_fix"):
        rv1.assert_no_unacknowledged_prior_seal(seal_path, acknowledge_defect_fix=False)


def test_acknowledge_defect_fix_without_description_is_rejected(tmp_path: Path) -> None:
    seal_path = tmp_path / "seal.json"
    _write_seal(seal_path)
    with pytest.raises(rv1.SealError, match="non-empty defect_description"):
        rv1.assert_no_unacknowledged_prior_seal(
            seal_path, acknowledge_defect_fix=True, defect_description=""
        )


def test_acknowledge_defect_fix_with_whitespace_only_description_is_rejected(
    tmp_path: Path,
) -> None:
    seal_path = tmp_path / "seal.json"
    _write_seal(seal_path)
    with pytest.raises(rv1.SealError, match="non-empty defect_description"):
        rv1.assert_no_unacknowledged_prior_seal(
            seal_path, acknowledge_defect_fix=True, defect_description="   "
        )


def test_acknowledge_defect_fix_with_description_is_permitted(tmp_path: Path) -> None:
    seal_path = tmp_path / "seal.json"
    original = _write_seal(seal_path)
    returned = rv1.assert_no_unacknowledged_prior_seal(
        seal_path,
        acknowledge_defect_fix=True,
        defect_description="fixed an off-by-one in the ingestion date range",
    )
    assert returned == original


# ---------------------------------------------------------------------------
# Archiving: pre-fix outputs preserved, defect-fix gets its own namespace
# ---------------------------------------------------------------------------


def test_archive_prior_outputs_preserves_originals_without_self_nesting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outputs_dir = tmp_path / "outputs" / "final_evaluation" / "v1"
    artifacts_dir = tmp_path / "artifacts" / "final_evaluation" / "v1"
    archive_root = tmp_path / "artifacts" / "final_evaluation" / "archived_runs"
    outputs_dir.mkdir(parents=True)
    artifacts_dir.mkdir(parents=True)
    (outputs_dir / "public_score_v1_final.json").write_text('{"rows": []}')
    (artifacts_dir / "seal.json").write_text('{"sealed": true}')

    monkeypatch.setattr(rv1, "FINAL_EVALUATION_OUTPUTS_DIR", outputs_dir)
    monkeypatch.setattr(rv1, "FINAL_EVALUATION_ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setattr(rv1, "ARCHIVE_ROOT", archive_root)

    archive_dir = rv1.archive_prior_outputs(outputs_dir=outputs_dir, artifacts_dir=artifacts_dir)

    # Originals untouched.
    assert (outputs_dir / "public_score_v1_final.json").exists()
    assert (artifacts_dir / "seal.json").exists()
    # Archive got real copies.
    assert (archive_dir / "outputs" / "public_score_v1_final.json").exists()
    assert (archive_dir / "artifacts" / "seal.json").exists()
    # Archive lives OUTSIDE artifacts_dir -- no self-nesting.
    assert not archive_dir.is_relative_to(artifacts_dir)


def test_archive_prior_outputs_rejects_paths_outside_their_own_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    narrow_outputs = tmp_path / "outputs_ns"
    narrow_artifacts = tmp_path / "artifacts_ns"
    narrow_outputs.mkdir()
    narrow_artifacts.mkdir()
    monkeypatch.setattr(rv1, "FINAL_EVALUATION_OUTPUTS_DIR", narrow_outputs)
    monkeypatch.setattr(rv1, "FINAL_EVALUATION_ARTIFACTS_DIR", narrow_artifacts)

    # tmp_path itself is OUTSIDE both narrow namespaces.
    with pytest.raises(rv1.NamespaceViolationError):
        rv1.archive_prior_outputs(outputs_dir=tmp_path, artifacts_dir=tmp_path)


def test_defect_fix_seal_records_audit_trail_to_prior_seal(tmp_path: Path) -> None:
    seal_path = tmp_path / "seal.json"
    original = _write_seal(seal_path)
    seal_path.unlink()
    fixed = _write_seal(seal_path, defect_fix_of=original.sealed_at)
    assert fixed.defect_fix_of == original.sealed_at
    assert fixed.defect_fix_of != fixed.sealed_at


def test_seal_path_default_filename() -> None:
    """The namespace placement of `SEAL_PATH` itself is covered by
    `test_v1_namespace_isolation.py::test_seal_path_resolves_inside_
    artifacts_namespace` -- this module's autouse fixture patches
    `FINAL_EVALUATION_ARTIFACTS_DIR`, so it cannot re-check that here.
    """
    assert rv1.SEAL_PATH.name == "seal.json"
