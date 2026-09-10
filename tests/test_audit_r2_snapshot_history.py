"""The read-only R2 snapshot-history audit.

The whole value of this tool is that it CANNOT write. "We did not call the
write path" is an assurance about intent; "there is no write path" is an
assurance about capability, and only the second survives a future edit by
someone who has not read the docstring. So most of these tests are
structural: they assert on the module's source and on its reader's method
set, not on behaviour.

The rest cover failing closed. A provenance audit that reports a partial
archive as complete is worse than no audit, because it would license
skipping a backfill that is actually needed.

Fully offline: the reader is a dict-backed fake, and no test constructs a
real client or reads a credential.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from audit_r2_snapshot_history import (
    READ_ONLY_S3_OPERATIONS,
    STATUS_ALREADY_ARCHIVED,
    STATUS_AMBIGUOUS,
    STATUS_NONE,
    STATUS_RECOVERABLE,
    AuditError,
    ReadOnlyR2Reader,
    _artifact_carries_snapshot,
    audit_r2_date,
    build_report,
    combine_status,
)

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "audit_r2_snapshot_history.py"
WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "audit-r2-snapshot-history.yml"
)
PREFIX = "prospective"
SEASON = 2026


class FakeReader:
    """Dict-backed stand-in with the SAME narrow surface as the real reader.

    Deliberately has no put/delete either: a test double that could write
    would let a write slip into the code under test and still pass.
    """

    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects

    def list_keys(self, prefix: str) -> list[str]:
        return sorted(k for k in self._objects if k.startswith(prefix))

    def get_bytes(self, key: str) -> bytes:
        return self._objects[key]

    def head(self, key: str) -> dict[str, Any]:
        return {"bytes": len(self._objects[key]), "last_modified": "2026-09-09T00:00:00+00:00"}


def _snapshot(date: str, *, complete: bool = True, corrupt_one: bool = False) -> dict[str, bytes]:
    """A synthetic archived snapshot spanning both sides."""
    files = {
        "outputs/public_score.json": b'[{"batter_id": 1}]',
        "outputs/scorecard.json": b'{"rows": 1}',
        "artifacts/manifest.json": json.dumps(
            {
                "data_through_date": date,
                "repository_commit": "a" * 40,
                "generated_at": f"{date}T17:00:00+00:00",
                "snapshot_label": None,
            }
        ).encode(),
    }
    integrity = {rel: hashlib.sha256(b).hexdigest() for rel, b in files.items()}
    objects = {f"{PREFIX}/{SEASON}/{date}/{rel}": data for rel, data in files.items()}
    objects[f"{PREFIX}/{SEASON}/{date}/artifacts/integrity_hashes.json"] = json.dumps(
        integrity
    ).encode()
    if not complete:
        del objects[f"{PREFIX}/{SEASON}/{date}/outputs/scorecard.json"]
    if corrupt_one:
        objects[f"{PREFIX}/{SEASON}/{date}/outputs/public_score.json"] = b"tampered"
    return objects


class TestNoWriteApiExistsAtAll:
    """Capability, not intention."""

    @staticmethod
    def _executable_identifiers() -> set[str]:
        """Every attribute and name the module actually EXECUTES.

        Deliberately not the raw text: this module's docstring explains which
        write operations `archive_snapshot` has and that this one has none,
        and prose describing an absence must not read as the presence.
        """
        tree = ast.parse(MODULE.read_text())
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Name):
                names.add(node.id)
        return names

    @pytest.mark.parametrize(
        "forbidden",
        [
            "put_object",
            "put_object_bytes",
            "upload_file",
            "upload_fileobj",
            "copy_object",
            "delete_object",
            "delete_objects",
            "create_multipart_upload",
            "upload_part",
            "complete_multipart_upload",
        ],
    )
    def test_the_module_executes_no_write_operation(self, forbidden: str) -> None:
        assert forbidden not in self._executable_identifiers(), forbidden

    def test_the_reader_exposes_only_list_get_head(self) -> None:
        public = {
            name
            for name in dir(ReadOnlyR2Reader)
            if not name.startswith("_") and callable(getattr(ReadOnlyR2Reader, name, None))
        }
        assert public == {"list_keys", "get_bytes", "head"}

    def test_only_three_boto3_operations_are_declared(self) -> None:
        assert READ_ONLY_S3_OPERATIONS == ("list_objects_v2", "get_object", "head_object")

    def test_every_s3_call_in_the_source_is_one_of_them(self) -> None:
        """Parses the AST rather than grepping: an attribute call on the s3
        client is what actually reaches R2."""
        tree = ast.parse(MODULE.read_text())
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "_s3"
        }
        assert called <= set(READ_ONLY_S3_OPERATIONS), called

    def test_it_never_constructs_the_write_capable_client(self) -> None:
        """`archive_snapshot.make_r2_client` / `_client_from_env` return a
        client that CAN put. Importing either would put write capability one
        attribute access away."""
        executed = self._executable_identifiers()
        assert "make_r2_client" not in executed
        assert "_client_from_env" not in executed

    def test_it_imports_only_pure_helpers_from_archive_snapshot(self) -> None:
        """Key shaping and the snapshot-directory rule are reused so the audit
        cannot disagree with the archive about where an object lives. Nothing
        else may be imported from that module."""
        tree = ast.parse(MODULE.read_text())
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "archive_snapshot"
            for alias in node.names
        }
        assert imported == {
            "DEFAULT_ARCHIVE_PREFIX",
            "INTEGRITY_HASHES_FILENAME",
            "MANIFEST_FILENAME",
            "_parse_snapshot_dir_names",
            "_r2_key",
        }


class TestItInvokesNoDangerousEntryPoint:
    @pytest.mark.parametrize(
        "forbidden",
        [
            "run_v1_1_2026_scoring",
            "publish_snapshot.sh",
            "dashboard/build.py",
            "wrangler",
            "sync_missing_snapshots",
            "--sync-history",
            "archive_snapshot_to_r2",
            "forecast",
            "PROSPECTIVE_AUTO_DEPLOY",
        ],
    )
    def test_no_scoring_archive_deploy_or_forecast_entry_point(self, forbidden: str) -> None:
        """Executable content again: the docstring names what this tool does
        NOT do, and that sentence must not fail its own test."""
        tree = ast.parse(MODULE.read_text())
        executed = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                executed.add(node.attr)
            elif isinstance(node, ast.Name):
                executed.add(node.id)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # string literals ARE executable content -- a subprocess
                # target or a path would live here
                executed.add(node.value)
        assert not any(forbidden in item for item in executed), forbidden

    def test_the_workflow_is_given_no_cloudflare_deploy_token(self) -> None:
        """Parsed env/run/uses, not the raw file -- the workflow comment says
        it is given no deploy token, and saying so must not read as having
        one."""
        yaml = pytest.importorskip("yaml")
        spec = yaml.safe_load(WORKFLOW.read_text())
        executed = []
        for step in spec["jobs"]["audit"]["steps"]:
            executed += [
                str(step.get("env", {})),
                str(step.get("run", "")),
                str(step.get("uses", "")),
            ]
        assert "CLOUDFLARE_API_TOKEN" not in "\n".join(executed)

    def test_the_workflow_has_no_write_permission(self) -> None:
        yaml = pytest.importorskip("yaml")
        spec = yaml.safe_load(WORKFLOW.read_text())
        assert spec["permissions"] == {"contents": "read", "actions": "read"}

    def test_the_workflow_is_manual_only(self) -> None:
        yaml = pytest.importorskip("yaml")
        spec = yaml.safe_load(WORKFLOW.read_text())
        assert set(spec[True]) == {"workflow_dispatch"}

    def test_the_workflow_requires_dates_explicitly(self) -> None:
        yaml = pytest.importorskip("yaml")
        inputs = yaml.safe_load(WORKFLOW.read_text())[True]["workflow_dispatch"]["inputs"]
        assert inputs["dates"]["required"] is True
        assert "default" not in inputs["dates"]


class TestDatesMustBeGiven:
    def test_no_date_is_refused(self) -> None:
        with pytest.raises(AuditError, match="never guesses a date"):
            build_report(dates=[], season=SEASON, reader=None, repository=None, github_token=None)

    def test_a_malformed_date_is_refused(self) -> None:
        with pytest.raises(AuditError, match="not YYYY-MM-DD"):
            audit_r2_date(FakeReader({}), date="09-09-2026", season=SEASON)


class TestItClassifiesHonestly:
    def test_a_complete_verified_snapshot_is_already_archived(self) -> None:
        record = audit_r2_date(
            FakeReader(_snapshot("2026-09-09")), date="2026-09-09", season=SEASON
        )
        assert record["status"] == STATUS_ALREADY_ARCHIVED
        directory = record["directories"][0]
        assert directory["complete"] is True
        assert directory["all_integrity_hashes_validate"] is True
        assert directory["repository_commit"] == "a" * 40
        assert directory["data_through_date"] == "2026-09-09"
        assert directory["manifest_sha256"]

    def test_no_objects_at_all_is_no_snapshot(self) -> None:
        record = audit_r2_date(FakeReader({}), date="2026-09-10", season=SEASON)
        assert record["status"] == STATUS_NONE
        assert record["any_objects_exist"] is False
        assert record["object_keys"] == []

    def test_a_partial_snapshot_is_ambiguous_never_complete(self) -> None:
        """The failure that would license skipping a needed backfill."""
        record = audit_r2_date(
            FakeReader(_snapshot("2026-09-09", complete=False)),
            date="2026-09-09",
            season=SEASON,
        )
        assert record["status"] == STATUS_AMBIGUOUS
        assert record["directories"][0]["complete"] is False
        assert any("absent" in p for p in record["directories"][0]["problems"])

    def test_a_corrupt_object_is_ambiguous(self) -> None:
        record = audit_r2_date(
            FakeReader(_snapshot("2026-09-09", corrupt_one=True)),
            date="2026-09-09",
            season=SEASON,
        )
        assert record["status"] == STATUS_AMBIGUOUS
        assert record["directories"][0]["all_integrity_hashes_validate"] is False

    def test_a_missing_integrity_file_fails_closed(self) -> None:
        objects = _snapshot("2026-09-09")
        del objects[f"{PREFIX}/{SEASON}/2026-09-09/artifacts/integrity_hashes.json"]
        record = audit_r2_date(FakeReader(objects), date="2026-09-09", season=SEASON)
        assert record["status"] == STATUS_AMBIGUOUS
        assert record["directories"][0]["complete"] is False

    def test_a_malformed_manifest_fails_closed(self) -> None:
        objects = _snapshot("2026-09-09")
        objects[f"{PREFIX}/{SEASON}/2026-09-09/artifacts/manifest.json"] = b"{not json"
        record = audit_r2_date(FakeReader(objects), date="2026-09-09", season=SEASON)
        assert record["status"] == STATUS_AMBIGUOUS
        assert any("not valid JSON" in p for p in record["directories"][0]["problems"])

    def test_an_orphaned_object_is_reported(self) -> None:
        objects = _snapshot("2026-09-09")
        objects[f"{PREFIX}/{SEASON}/2026-09-09/outputs/stray.json"] = b"{}"
        record = audit_r2_date(FakeReader(objects), date="2026-09-09", season=SEASON)
        assert record["directories"][0]["orphaned_objects"] == ["outputs/stray.json"]
        assert record["status"] == STATUS_AMBIGUOUS

    def test_a_labelled_variant_of_the_date_is_found_not_missed(self) -> None:
        """`<date>__refreshed` is a real shape in this archive."""
        objects = {
            k.replace("/2026-09-09/", "/2026-09-09__refreshed/"): v
            for k, v in _snapshot("2026-09-09").items()
        }
        record = audit_r2_date(FakeReader(objects), date="2026-09-09", season=SEASON)
        assert record["snapshot_dir_names"] == ["2026-09-09__refreshed"]
        assert record["status"] == STATUS_ALREADY_ARCHIVED

    def test_two_complete_directories_for_one_date_is_ambiguous(self) -> None:
        """Not 'pick one' -- a human decides which is production."""
        objects = _snapshot("2026-09-09")
        objects.update(
            {
                k.replace("/2026-09-09/", "/2026-09-09__refreshed/"): v
                for k, v in _snapshot("2026-09-09").items()
            }
        )
        record = audit_r2_date(FakeReader(objects), date="2026-09-09", season=SEASON)
        assert record["status"] == STATUS_AMBIGUOUS
        assert "2 complete" in record["detail"]


class TestActionsRecoverability:
    def test_a_manifest_artifact_does_not_count_as_the_snapshot(self) -> None:
        """The publish workflow uploads manifest+integrity, not the scored
        snapshot. Treating metadata as recoverability would wrongly conclude a
        backfill is unnecessary."""
        assert _artifact_carries_snapshot("snapshot-manifest-2026-09-10") is False

    def test_build_output_does_not_count(self) -> None:
        assert _artifact_carries_snapshot("dashboard-dist-2026-09-08") is False

    def test_a_real_snapshot_artifact_counts(self) -> None:
        assert _artifact_carries_snapshot("prospective-snapshot-2026-09-10") is True

    def test_r2_completeness_wins_over_everything(self) -> None:
        assert combine_status({"status": STATUS_ALREADY_ARCHIVED}, {}) == STATUS_ALREADY_ARCHIVED

    def test_a_retained_scored_artifact_makes_a_date_recoverable(self) -> None:
        assert (
            combine_status(
                {"status": STATUS_NONE},
                {"exact_snapshot_recoverable_without_rescoring": True},
            )
            == STATUS_RECOVERABLE
        )

    def test_partial_beats_absent(self) -> None:
        """A partial archive is something a human must look at; reporting it
        as simply absent would hide it."""
        assert combine_status({"status": STATUS_AMBIGUOUS}, {}) == STATUS_AMBIGUOUS


class TestSecretsAreNeverEmitted:
    def test_missing_credentials_are_named_never_valued(self) -> None:
        source = MODULE.read_text()
        block = source.split("def make_read_only_reader", 1)[1].split("\ndef ", 1)[0]
        assert "Names only" in block
        # The error interpolates the NAME list, never os.environ values.
        assert "', '.join(missing)" in block

    def test_the_report_carries_no_credential_fields(self) -> None:
        report = build_report(
            dates=["2026-09-09"],
            season=SEASON,
            reader=FakeReader(_snapshot("2026-09-09")),
            repository=None,
            github_token=None,
        )
        blob = json.dumps(report).upper()
        for secret in ("SECRET", "ACCESS_KEY", "TOKEN", "PASSWORD"):
            assert secret not in blob, secret
        assert report["contains_no_credentials"] is True

    def test_the_report_declares_itself_read_only(self) -> None:
        report = build_report(
            dates=["2026-09-09"],
            season=SEASON,
            reader=FakeReader(_snapshot("2026-09-09")),
            repository=None,
            github_token=None,
        )
        assert report["read_only"] is True
        assert report["s3_operations_used"] == list(READ_ONLY_S3_OPERATIONS)
