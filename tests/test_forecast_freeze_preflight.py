"""The pre-freeze formatting guard: format -> review if changed -> freeze.

The failure being prevented is concrete. A freeze hashes its own source
modules; `make check` runs the canonical formatter afterwards; the manifest
then no longer matches its inputs. Formatting *inside* the freeze command
would be worse -- it would seal a rewrite nobody read in the same operation
that made it.

These tests run the REAL formatter against real files in a temporary project
root. Nothing here touches a frozen artifact, a 2026 outcome, or any
specification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from forecast import freeze_preflight as fp
from forecast.freeze_r1 import hash_file
from forecast.stage_freeze import StageFreezeSpec

UNFORMATTED = "x = {'a':1,   'b':2}\n\n\n\ndef f( a,b ):\n    return a+b\n"
FORMATTED = 'x = {"a": 1, "b": 2}\n\n\ndef f(a, b):\n    return a + b\n'


@pytest.fixture
def project(tmp_path: Path, monkeypatch: Any) -> Path:
    """A temporary project root, for both the preflight and `stage_freeze`.

    `stage_freeze` resolves source modules against its own PROJECT_ROOT and is
    a hashed input of three sealed stages, so it is patched here rather than
    given a parameter it must not grow.
    """
    (tmp_path / "pkg").mkdir()
    monkeypatch.setattr("forecast.stage_freeze.PROJECT_ROOT", tmp_path)
    return tmp_path


def _write(project: Path, name: str, body: str) -> Path:
    path = project / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


# --------------------------------------------------------------------------
# An unformatted freeze input prevents sealing, and names itself
# --------------------------------------------------------------------------


def test_an_unformatted_input_stops_the_freeze(project: Path) -> None:
    _write(project, "pkg/messy.py", UNFORMATTED)
    with pytest.raises(fp.FreezePreflightError, match="Freeze halted"):
        fp.assert_freeze_inputs_formatted(["pkg/messy.py"], stage="demo", project_root=project)


def test_the_formatter_identifies_the_changed_files(project: Path) -> None:
    _write(project, "pkg/messy.py", UNFORMATTED)
    _write(project, "pkg/tidy.py", FORMATTED)
    _write(project, "pkg/also_messy.py", UNFORMATTED)

    result = fp.format_freeze_inputs(
        ["pkg/messy.py", "pkg/tidy.py", "pkg/also_messy.py"], project_root=project
    )
    assert result["changed_files"] == ["pkg/also_messy.py", "pkg/messy.py"]
    assert result["n_changed"] == 2
    assert result["n_source_files"] == 3
    assert result["formatting_was_idempotent"] is False


def test_the_error_names_every_changed_file(project: Path) -> None:
    _write(project, "pkg/messy.py", UNFORMATTED)
    _write(project, "pkg/tidy.py", FORMATTED)
    with pytest.raises(fp.FreezePreflightError) as excinfo:
        fp.assert_freeze_inputs_formatted(
            ["pkg/messy.py", "pkg/tidy.py"], stage="demo", project_root=project
        )
    message = str(excinfo.value)
    assert "pkg/messy.py" in message
    assert "pkg/tidy.py" not in message.split("Review the diff")[0].split("- pkg/messy.py")[1]
    assert "NOTHING WAS SEALED" in message
    assert "rerun" in message


def test_a_clean_input_passes_and_reports_idempotence(project: Path) -> None:
    _write(project, "pkg/tidy.py", FORMATTED)
    result = fp.assert_freeze_inputs_formatted(["pkg/tidy.py"], stage="demo", project_root=project)
    assert result["changed_files"] == []
    assert result["formatting_was_idempotent"] is True


def test_a_missing_freeze_input_is_refused(project: Path) -> None:
    with pytest.raises(fp.FreezePreflightError, match="missing"):
        fp.format_freeze_inputs(["pkg/absent.py"], project_root=project)


# --------------------------------------------------------------------------
# Two invocations: halt, then freeze
# --------------------------------------------------------------------------


def _demo_spec(source: str) -> StageFreezeSpec:
    return StageFreezeSpec(
        stage="demo",
        version="demo_v1",
        conclusion={"verdict": "DEMO", "statement": "A demo stage for the preflight."},
        limitations=({"id": "demo", "statement": "demo"},),
        artifacts=("demo_artifact.json",),
        source_modules=(source,),
        upstream_manifests=(),
        key_results=lambda outputs_dir: json.loads(
            (outputs_dir / "demo_artifact.json").read_text()
        ),
        checks=lambda key_results: None,
    )


def test_no_manifest_is_written_on_the_halted_invocation(project: Path) -> None:
    outputs = project / "out"
    outputs.mkdir()
    (outputs / "demo_artifact.json").write_text(json.dumps({"n": 1}))
    _write(project, "pkg/messy.py", UNFORMATTED)

    with pytest.raises(fp.FreezePreflightError):
        fp.freeze_stage_with_preflight(
            _demo_spec("pkg/messy.py"), outputs_dir=outputs, project_root=project
        )
    assert not (outputs / "demo_freeze_manifest.json").exists()
    assert not (outputs / "demo_frozen_conclusion.md").exists()
    assert sorted(p.name for p in outputs.iterdir()) == ["demo_artifact.json"]


def test_the_second_invocation_freezes_successfully(project: Path) -> None:
    outputs = project / "out"
    outputs.mkdir()
    (outputs / "demo_artifact.json").write_text(json.dumps({"n": 1}))
    source = _write(project, "pkg/messy.py", UNFORMATTED)
    spec = _demo_spec("pkg/messy.py")

    # First invocation: halts, having formatted but sealed nothing.
    with pytest.raises(fp.FreezePreflightError):
        fp.freeze_stage_with_preflight(spec, outputs_dir=outputs, project_root=project)
    assert source.read_text() == FORMATTED
    assert not (outputs / "demo_freeze_manifest.json").exists()

    # Second invocation, after the maintainer has reviewed the diff.
    manifest = fp.freeze_stage_with_preflight(spec, outputs_dir=outputs, project_root=project)
    assert (outputs / "demo_freeze_manifest.json").exists()
    assert manifest["stage"] == "demo"
    assert manifest["source_module_sha256"]["pkg/messy.py"] == hash_file(source)


def test_formatting_is_idempotent_on_the_second_run(project: Path) -> None:
    _write(project, "pkg/messy.py", UNFORMATTED)
    first = fp.format_freeze_inputs(["pkg/messy.py"], project_root=project)
    second = fp.format_freeze_inputs(["pkg/messy.py"], project_root=project)
    third = fp.format_freeze_inputs(["pkg/messy.py"], project_root=project)

    assert first["formatting_was_idempotent"] is False
    assert second["formatting_was_idempotent"] is True
    assert third["formatting_was_idempotent"] is True
    assert second["sha256_after"] == third["sha256_after"] == first["sha256_after"]


def test_a_completed_freeze_verifies_with_zero_drift(project: Path) -> None:
    from forecast.stage_freeze import verify_stage

    outputs = project / "out"
    outputs.mkdir()
    (outputs / "demo_artifact.json").write_text(json.dumps({"n": 1}))
    _write(project, "pkg/messy.py", UNFORMATTED)
    spec = _demo_spec("pkg/messy.py")

    with pytest.raises(fp.FreezePreflightError):
        fp.freeze_stage_with_preflight(spec, outputs_dir=outputs, project_root=project)
    fp.freeze_stage_with_preflight(spec, outputs_dir=outputs, project_root=project)

    # The whole point: formatting already happened, so a later `make check`
    # cannot rewrite a sealed input behind the manifest's back.
    fp.format_freeze_inputs(["pkg/messy.py"], project_root=project)
    result = verify_stage(spec, outputs_dir=outputs)
    assert result["matches_freeze"] is True, result["drift"]


# --------------------------------------------------------------------------
# The real stages
# --------------------------------------------------------------------------


def test_every_real_stage_is_registered() -> None:
    assert set(fp.available_stages()) == {
        "r1",
        "ridge",
        "hgb",
        "h200_spec",
        "resolution_spec",
    }


def test_the_registry_reads_each_stage_declaration_rather_than_restating_it() -> None:
    from forecast.phase2.resolution_spec import resolution_freeze_spec

    sources = fp._stage_sources()
    assert sources["resolution_spec"] == tuple(resolution_freeze_spec().source_modules)
    assert "forecast/phase2/resolution_spec.py" in sources["resolution_spec"]


def test_an_unknown_stage_is_refused() -> None:
    with pytest.raises(fp.FreezePreflightError, match="Unknown stage"):
        fp.preflight_stage("not_a_stage")


@pytest.mark.parametrize("stage", ["r1", "ridge", "hgb", "h200_spec", "resolution_spec"])
def test_the_repository_is_currently_preflight_clean(stage: str) -> None:
    """Every sealed stage's inputs are already formatted, so none would drift."""
    assert fp.preflight_stage(stage)["formatting_was_idempotent"] is True


def test_the_guard_is_not_wired_into_stage_freeze(project: Path) -> None:
    """`stage_freeze.py` is hashed by three sealed stages and must not change.

    The preflight is deliberately a separate module for that reason; this pins
    the constraint so a later refactor does not quietly break the chain.
    """
    source = (Path(__file__).resolve().parents[1] / "forecast" / "stage_freeze.py").read_text()
    assert "freeze_preflight" not in source
    assert "ruff" not in source


def test_the_canonical_formatter_is_the_repository_formatter() -> None:
    assert fp.CANONICAL_FORMATTER[1:] == ("-m", "ruff", "format")


def test_a_failing_formatter_stops_the_freeze(project: Path) -> None:
    _write(project, "pkg/tidy.py", FORMATTED)
    with pytest.raises(fp.FreezePreflightError, match="formatter failed"):
        fp.format_freeze_inputs(
            ["pkg/tidy.py"],
            project_root=project,
            formatter=(fp.CANONICAL_FORMATTER[0], "-c", "import sys; sys.exit(3)"),
        )
