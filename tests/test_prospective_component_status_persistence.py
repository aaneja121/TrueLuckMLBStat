"""Version 1.4.2: `component_model_status_raw` must reach DISK.

Phase 8 fixed the outfield gate's stringified boolean at the scoring layer:
`prospective_scoring.train_and_score_2026` normalizes
`component_model_status` onto `component_confidence`'s MODEL_STATUS_*
vocabulary and records the unnormalized gate readings beside it as
`component_model_status_raw`. The production dry run then showed the fix
only half landed: `run_v1_1_2026_scoring.run_prospective_snapshot` copied
just the normalized field into `component_status_summary.json`, so the raw
half was built in memory and thrown away -- the published mapping had no
published input.

These tests are deliberately NOT source-text assertions (Phase 8 already has
those). They run the real `run_prospective_snapshot` persistence path,
against a real synthetic gate file read by the real
`_read_existing_gate_status`, and then read the JSON that was actually
written. They manufacture no dated production snapshot: every root is a
`tmp_path`.

Fully offline -- the network- and model-touching steps are stubbed by
`test_prospective_force_redownload_immutability._stub_pipeline`, which is
reused rather than copied precisely because it already exercises the real
snapshot writer end to end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import run_v1_1_2026_scoring as runner
from run_v1_final_evaluation import _read_existing_gate_status
from snapshot_data import compute_file_sha256

from mlb_luck_score.scoring.component_confidence import (
    MODEL_STATUS_NOT_CALIBRATED,
    normalize_model_status,
)
from test_prospective_force_redownload_immutability import _init_clean_repo, _stub_pipeline

DATA_THROUGH = "2026-04-15"

# The exact key paths `prospective_scoring.train_and_score_2026` reads. The
# outfield comparison file (one candidate, no selection) publishes no
# `gate_summary`; its only verdict field is a BOOLEAN, which is the whole
# origin of the "False" this test is about.
OUTFIELD_STATUS_KEY = (
    "validation_summary",
    "per_candidate",
    "measured_contact_only_v07",
    "passes_basic_validation",
)
GATE_SUMMARY_STATUS_KEY = ("gate_summary", "overall_status")


def _write_gate_files(gate_dir: Path, *, outfield_passes: bool) -> None:
    """A minimal stand-in for `outputs/tables/`, shaped like the real gate
    files at the key paths production actually reads.
    """
    gate_dir.mkdir(parents=True, exist_ok=True)
    (gate_dir / "opportunity_model_comparison_detail.json").write_text(
        json.dumps(
            {
                "validation_summary": {
                    "per_candidate": {
                        "measured_contact_only_v07": {
                            "passes_basic_validation": outfield_passes,
                        }
                    }
                }
            }
        )
    )
    (gate_dir / "infield_opportunity_detail.json").write_text(
        json.dumps({"gate_summary": {"overall_status": "calibrated"}})
    )
    (gate_dir / "advancement_detail.json").write_text(
        json.dumps(
            {"gate_summary": {"overall_status": "calibrated_with_limited_subgroup_evidence"}}
        )
    )


def _report_status_block(gate_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Reads the three gate files exactly as `train_and_score_2026` does and
    assembles the same two report fields it assembles: normalized through the
    project's canonical `normalize_model_status`, raw copied verbatim.
    """
    raw = {
        "outfield": _read_existing_gate_status(
            gate_dir / "opportunity_model_comparison_detail.json",
            status_key=OUTFIELD_STATUS_KEY,
        ),
        "infield": _read_existing_gate_status(
            gate_dir / "infield_opportunity_detail.json", status_key=GATE_SUMMARY_STATUS_KEY
        ),
        "advancement": _read_existing_gate_status(
            gate_dir / "advancement_detail.json", status_key=GATE_SUMMARY_STATUS_KEY
        ),
    }
    report_fields: dict[str, Any] = {
        "component_model_status": {k: normalize_model_status(v) for k, v in raw.items()},
        "component_model_status_raw": dict(raw),
    }
    return report_fields, raw


def _run_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, report_overrides: dict[str, Any]
) -> Path:
    """Runs the real snapshot writer into throwaway roots and returns the
    directory it wrote.
    """
    outputs_root = tmp_path / "outputs" / "prospective" / "v1_1"
    artifacts_root = tmp_path / "artifacts" / "prospective" / "v1_1"
    outputs_root.mkdir(parents=True)
    artifacts_root.mkdir(parents=True)
    repo_root = _init_clean_repo(tmp_path)
    _stub_pipeline(monkeypatch, distinguishing_value=1.0, report_overrides=report_overrides)

    result = runner.run_prospective_snapshot(
        data_through_date=DATA_THROUGH,
        repo_root=repo_root,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
    )
    assert result["status"] == "written"
    return outputs_root / DATA_THROUGH


class TestRawComponentStatusReachesTheSnapshot:
    def test_a_false_outfield_gate_persists_both_the_normalized_and_raw_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The regression itself, at the persisted artifact.

        The outfield gate genuinely returns False here -- it is read out of a
        real file by the real reader -- and the JSON on disk must carry both
        halves of the fact: the normalized verdict AND the reading it was
        derived from.
        """
        gate_dir = tmp_path / "gate_tables"
        _write_gate_files(gate_dir, outfield_passes=False)
        report_fields, raw = _report_status_block(gate_dir)

        # Precondition, not the assertion under test: confirm the synthetic
        # gate file really does reproduce the production reading.
        assert raw["outfield"] == "False"

        snapshot_dir = _run_snapshot(tmp_path, monkeypatch, report_overrides=report_fields)
        persisted = json.loads((snapshot_dir / "component_status_summary.json").read_text())

        assert persisted["component_model_status"]["outfield"] == MODEL_STATUS_NOT_CALIBRATED
        assert persisted["component_model_status_raw"]["outfield"] == raw["outfield"]
        assert persisted["component_model_status_raw"]["outfield"] == "False"

    def test_the_raw_block_is_copied_verbatim_for_every_component(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`raw` means exactly what the gate reported. The writer must not
        map, retype, normalize, or default any of it -- including the two
        components whose raw readings are already valid vocabulary.
        """
        gate_dir = tmp_path / "gate_tables"
        _write_gate_files(gate_dir, outfield_passes=False)
        report_fields, raw = _report_status_block(gate_dir)

        snapshot_dir = _run_snapshot(tmp_path, monkeypatch, report_overrides=report_fields)
        persisted = json.loads((snapshot_dir / "component_status_summary.json").read_text())

        assert persisted["component_model_status_raw"] == raw
        assert persisted["component_model_status"] == {
            "outfield": MODEL_STATUS_NOT_CALIBRATED,
            "infield": "calibrated",
            "advancement": "calibrated_with_limited_subgroup_evidence",
        }

    def test_a_passing_outfield_gate_still_records_its_own_raw_reading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other side of the same boolean. `"True"` normalizes to
        `calibrated`, and the raw reading is still preserved -- the raw field
        is a record of the input, not a failure marker.
        """
        gate_dir = tmp_path / "gate_tables"
        _write_gate_files(gate_dir, outfield_passes=True)
        report_fields, raw = _report_status_block(gate_dir)
        assert raw["outfield"] == "True"

        snapshot_dir = _run_snapshot(tmp_path, monkeypatch, report_overrides=report_fields)
        persisted = json.loads((snapshot_dir / "component_status_summary.json").read_text())

        assert persisted["component_model_status"]["outfield"] == normalize_model_status("True")
        assert persisted["component_model_status_raw"]["outfield"] == "True"

    def test_the_summary_file_is_covered_by_the_generic_integrity_hashing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Adding a field changes the file's sha256, so the recorded hash has
        to be the hash of what was actually written. It is, because the
        writer hashes the directory generically -- no special case was added.
        """
        gate_dir = tmp_path / "gate_tables"
        _write_gate_files(gate_dir, outfield_passes=False)
        report_fields, _ = _report_status_block(gate_dir)

        snapshot_dir = _run_snapshot(tmp_path, monkeypatch, report_overrides=report_fields)
        artifacts_dir = tmp_path / "artifacts" / "prospective" / "v1_1" / DATA_THROUGH
        integrity = json.loads((artifacts_dir / "integrity_hashes.json").read_text())

        summary_path = snapshot_dir / "component_status_summary.json"
        assert integrity["outputs/component_status_summary.json"] == compute_file_sha256(
            summary_path
        )


class TestBackwardCompatibilityWithReportsThatHaveNoRawBlock:
    """Every already-published snapshot was written from a report with no
    `component_model_status_raw`, and other aggregators still produce reports
    without it. The writer must keep working with those, and must not invent
    a raw block for them.
    """

    def test_a_report_without_the_raw_block_still_persists_a_valid_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        snapshot_dir = _run_snapshot(
            tmp_path,
            monkeypatch,
            report_overrides={"component_model_status": {"infield": "calibrated"}},
        )
        persisted = json.loads((snapshot_dir / "component_status_summary.json").read_text())

        assert persisted["component_model_status"] == {"infield": "calibrated"}
        assert "model_selection_winners" in persisted

    def test_the_key_is_omitted_rather_than_written_as_null(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reader must be able to tell "never recorded" from "recorded as
        nothing"; a null would be a third, invented status.
        """
        snapshot_dir = _run_snapshot(
            tmp_path,
            monkeypatch,
            report_overrides={"component_model_status": {"infield": "calibrated"}},
        )
        persisted = json.loads((snapshot_dir / "component_status_summary.json").read_text())

        assert "component_model_status_raw" not in persisted


class TestTheDashboardReadsBothShapes:
    """The renderer is not part of this fix, but it consumes the file that
    changed shape. It must build against a snapshot that carries the new
    field and keep displaying the recorded status exactly as before.
    """

    def test_a_snapshot_carrying_the_raw_block_still_builds_and_renders_status(
        self, tmp_path: Path
    ) -> None:
        import build as dashboard_build

        from dashboard_snapshot_fixtures import default_player_record, write_snapshot

        out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-01",
            data_through_date="2026-01-01",
            snapshot_label=None,
            generated_at="2026-01-02T00:00:00+00:00",
            players=[
                default_player_record(
                    batter_id=1, batter_name="Alice Alpha", score=5.25, lower=1.1, upper=9.4
                )
            ],
        )

        # Rewrite the summary into the post-fix shape (normalized + raw) and
        # re-record its hash, exactly as a real post-fix snapshot would have
        # been written.
        summary_path = out_root / "2026-01-01" / "component_status_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["component_model_status"]["outfield"] = MODEL_STATUS_NOT_CALIBRATED
        summary["component_model_status_raw"] = {
            "outfield": "False",
            "infield": "calibrated",
            "advancement": "calibrated_with_limited_subgroup_evidence",
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        integrity_path = art_root / "2026-01-01" / "integrity_hashes.json"
        integrity = json.loads(integrity_path.read_text())
        integrity["outputs/component_status_summary.json"] = compute_file_sha256(summary_path)
        integrity_path.write_text(json.dumps(integrity, indent=2))

        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-03T00:00:00+00:00",
        )
        # A documented status renders as its public label, with no
        # `status-code` identifier styling -- which is exactly the change
        # `/status/` will show once a post-fix snapshot is published.
        html = (tmp_path / "dist" / "status" / "index.html").read_text()
        assert "<dd>Not calibrated</dd>" in html
        assert '<dd class="status-code">False</dd>' not in html
