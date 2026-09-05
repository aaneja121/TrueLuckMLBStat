"""Pre-freeze formatting guard: format, review, THEN freeze.

## The failure this exists to prevent

A freeze hashes its own source modules. If the canonical formatter runs AFTER
a stage is sealed -- which is what `make check` does -- it rewrites those files
and the manifest no longer matches its own inputs. That happened twice while
the H=200 and resolution specifications were being built, and both times the
fix was to re-freeze. Re-freezing is cheap before a result exists and
impossible after one is published.

The obvious repair, formatting inside the freeze command, is worse: the
command would rewrite source and seal the rewrite in one uninterrupted
operation, so a maintainer could seal changes nobody read. A freeze is a claim
that a specific reviewed text produced a specific result.

So the order enforced here is:

    format  ->  review if anything changed  ->  freeze

never `freeze -> format -> broken manifest`, and never
`format silently -> freeze unreviewed changes`.

## How it behaves

The preflight runs the repository's canonical formatter over exactly the
source files the freeze will hash. If any file changed, it STOPS and names
them: the freeze does not run, and no manifest is written. The maintainer
reviews the diff -- `git diff` on the named files -- and reruns. On the second
invocation the formatter is idempotent, nothing changes, and the freeze
proceeds against text that has been seen.

## Why this module is not wired into `stage_freeze`

`forecast/stage_freeze.py` is a hashed source module of the ridge, HGB and
H=200 freezes, all of which are sealed and have published results. Editing it
would drift three manifests and break the chain the end-of-season resolution
pass verifies before it may run. This module is therefore new and standalone:
`make` runs it as a separate step before each freeze target, and future stages
can call `freeze_stage_with_preflight` directly.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from forecast.freeze_r1 import R1_SOURCE_MODULES, hash_file

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The repository's canonical formatter, as `make format` invokes it.
CANONICAL_FORMATTER: tuple[str, ...] = (sys.executable, "-m", "ruff", "format")


class FreezePreflightError(RuntimeError):
    """Raised when a freeze must not proceed until its inputs are reviewed."""


def _stage_sources() -> dict[str, tuple[str, ...]]:
    """Every freezable stage's source modules, read from the stage itself.

    Resolved lazily from each stage's own declaration rather than restated, so
    this registry cannot drift from what the freeze will actually hash.
    """
    from forecast.freeze_hgb import HGB_FREEZE_SPEC
    from forecast.freeze_ridge import RIDGE_FREEZE_SPEC
    from forecast.phase2.h200_spec import h200_freeze_spec
    from forecast.phase2.resolution_spec import resolution_freeze_spec

    return {
        "r1": tuple(R1_SOURCE_MODULES),
        "ridge": tuple(RIDGE_FREEZE_SPEC.source_modules),
        "hgb": tuple(HGB_FREEZE_SPEC.source_modules),
        "h200_spec": tuple(h200_freeze_spec().source_modules),
        "resolution_spec": tuple(resolution_freeze_spec().source_modules),
    }


def available_stages() -> tuple[str, ...]:
    """The stage names the preflight knows how to guard."""
    return tuple(sorted(_stage_sources()))


def format_freeze_inputs(
    source_modules: Sequence[str],
    *,
    project_root: Path = PROJECT_ROOT,
    formatter: Sequence[str] = CANONICAL_FORMATTER,
) -> dict[str, Any]:
    """Format exactly the files a freeze will hash, and report what changed.

    Files are compared by content hash before and after, rather than by
    parsing the formatter's output, so the answer does not depend on a tool's
    reporting format.

    Raises:
        FreezePreflightError: If a declared source file is missing, or the
            formatter cannot run.
    """
    paths = [project_root / name for name in source_modules]
    missing = [name for name, path in zip(source_modules, paths, strict=True) if not path.exists()]
    if missing:
        raise FreezePreflightError(
            f"Cannot run the freeze preflight: declared source file(s) missing: {missing}"
        )

    before = {name: hash_file(path) for name, path in zip(source_modules, paths, strict=True)}
    command = [*formatter, *(str(p) for p in paths)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:  # pragma: no cover - depends on the environment
        raise FreezePreflightError(
            f"Could not run the canonical formatter {formatter!r}: {exc}. The preflight "
            "is not optional -- a freeze may not proceed without it."
        ) from exc
    if completed.returncode != 0:
        raise FreezePreflightError(
            f"The canonical formatter failed (exit {completed.returncode}):\n"
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )

    after = {name: hash_file(path) for name, path in zip(source_modules, paths, strict=True)}
    changed = sorted(name for name in before if before[name] != after[name])
    return {
        "formatter": list(formatter),
        "n_source_files": len(paths),
        "changed_files": changed,
        "n_changed": len(changed),
        "formatting_was_idempotent": not changed,
        "sha256_before": before,
        "sha256_after": after,
    }


def assert_freeze_inputs_formatted(
    source_modules: Sequence[str],
    *,
    stage: str,
    project_root: Path = PROJECT_ROOT,
    formatter: Sequence[str] = CANONICAL_FORMATTER,
) -> dict[str, Any]:
    """Stop the freeze if formatting changed anything. Nothing is sealed here.

    The formatter has already written its changes by the time this raises --
    that is the point. The maintainer reviews the diff and reruns; the second
    invocation is idempotent and proceeds.

    Raises:
        FreezePreflightError: If any freeze input was reformatted.
    """
    result = format_freeze_inputs(source_modules, project_root=project_root, formatter=formatter)
    if result["changed_files"]:
        listing = "\n".join(f"  - {name}" for name in result["changed_files"])
        raise FreezePreflightError(
            f"Freeze halted: the canonical formatter changed {result['n_changed']} of "
            f"{result['n_source_files']} source file(s) that the {stage!r} freeze would "
            f"hash:\n{listing}\n\n"
            "The formatting has been applied but NOTHING WAS SEALED and no manifest was "
            "written. Review the diff (`git diff` on the files above), then rerun this "
            "command. The second run formats to a no-op and the freeze proceeds.\n\n"
            "A freeze is a claim that a specific reviewed text produced a specific "
            "result, so it may not seal a rewrite in the same operation that made it."
        )
    logger.info(
        "preflight clean for %s: %d source file(s), formatting idempotent",
        stage,
        result["n_source_files"],
    )
    return result


def preflight_stage(
    stage: str,
    *,
    project_root: Path = PROJECT_ROOT,
    formatter: Sequence[str] = CANONICAL_FORMATTER,
) -> dict[str, Any]:
    """Run the formatting guard for one named stage.

    Raises:
        FreezePreflightError: If the stage is unknown, or its inputs changed.
    """
    sources = _stage_sources()
    if stage not in sources:
        raise FreezePreflightError(
            f"Unknown stage {stage!r}. Known stages: {', '.join(sorted(sources))}"
        )
    return assert_freeze_inputs_formatted(
        sources[stage], stage=stage, project_root=project_root, formatter=formatter
    )


def freeze_stage_with_preflight(
    spec: Any,
    *,
    outputs_dir: Path,
    project_root: Path = PROJECT_ROOT,
    formatter: Sequence[str] = CANONICAL_FORMATTER,
    freeze: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Guard, then freeze. The entry point new stages should use.

    Raises:
        FreezePreflightError: If formatting changed a freeze input. The freeze
            is not attempted and no manifest is written.
    """
    assert_freeze_inputs_formatted(
        spec.source_modules, stage=spec.stage, project_root=project_root, formatter=formatter
    )
    if freeze is None:
        from forecast.stage_freeze import freeze_stage

        seal: Callable[..., dict[str, Any]] = freeze_stage
    else:
        seal = freeze
    return seal(spec, outputs_dir=outputs_dir)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, help="Stage whose freeze inputs to guard.")
    args = parser.parse_args(argv)
    try:
        preflight_stage(args.stage)
    except FreezePreflightError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
