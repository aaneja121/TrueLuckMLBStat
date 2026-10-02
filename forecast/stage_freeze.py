"""Contact Forecast: reusable staged-freeze machinery.

`forecast.freeze_r1` sealed the R1 baseline. Every later stage needs the same
guarantees -- a content hash over artifacts, source and inputs; a conclusion
checked against the numbers it describes; and a provenance chain back to the
stages it builds on -- so the mechanism lives here once instead of being
copied per stage.

`freeze_r1` is deliberately NOT refactored onto this module. It is hashed
inside its own manifest, so editing it would drift the R1 freeze for no
scientific reason. It keeps its own copy; this module borrows only its two
pure hashing helpers.

## The provenance chain

Each stage records the manifest hash of every stage it depends on, and
`freeze_stage` refuses to seal a stage whose upstream manifest has drifted.
A ridge result is only meaningful against the exact R1 package it was measured
on, and an HGB result only against the exact ridge it is compared with, so the
chain is enforced rather than assumed.

## The conclusion is checked, not just stored

Each stage supplies a `checks` callable that re-reads its own key results and
raises if the recorded prose no longer matches them. Conservative wording is
not sufficient on its own once the number behind it has moved.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forecast.forecast_config import (
    FORECAST_OUTPUTS_DIR,
    FORECAST_SEALED_SEASONS,
    PHASE_2_GATE,
    assert_path_outside_forbidden_namespaces,
)
from forecast.freeze_r1 import ForecastFreezeError, hash_file, hash_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StageFreezeSpec:
    """Everything a stage needs to be sealed and later re-verified."""

    stage: str
    version: str
    conclusion: dict[str, Any]
    limitations: tuple[dict[str, str], ...]
    artifacts: tuple[str, ...]
    source_modules: tuple[str, ...]
    #: Manifest filenames of the stages this one builds on, in order.
    upstream_manifests: tuple[str, ...]
    #: Pulls the handful of numbers the conclusion rests on out of the artifacts.
    key_results: Callable[[Path], dict[str, Any]]
    #: Raises if the conclusion no longer describes those numbers.
    checks: Callable[[dict[str, Any]], None]
    #: Extra machine-readable specification recorded with the freeze.
    specification: Callable[[], dict[str, Any]] | None = None

    @property
    def manifest_name(self) -> str:
        return f"{self.stage}_freeze_manifest.json"

    @property
    def conclusion_name(self) -> str:
        return f"{self.stage}_frozen_conclusion.md"


def _hash_tree(paths: dict[str, Path], *, stage: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.exists():
            raise ForecastFreezeError(f"Cannot freeze {stage}: {name} is missing at {path}")
        hashes[name] = hash_file(path)
    return hashes


def _environment() -> dict[str, str]:
    import numpy
    import pandas
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _upstream_chain(spec: StageFreezeSpec, outputs_dir: Path) -> dict[str, Any]:
    """Hash and verify every upstream stage manifest this stage depends on.

    Raises:
        ForecastFreezeError: If an upstream manifest is missing, or if it no
            longer matches its own recorded self-hash.
    """
    chain: dict[str, Any] = {}
    for name in spec.upstream_manifests:
        path = outputs_dir / name
        if not path.exists():
            raise ForecastFreezeError(
                f"Cannot freeze {spec.stage}: upstream manifest {name} is missing. "
                "Each stage must be sealed before the stage that builds on it."
            )
        manifest = json.loads(path.read_text())
        recorded = manifest.get("manifest_sha256")
        recomputed = hash_json(
            {k: v for k, v in manifest.items() if k not in {"frozen_at_utc", "manifest_sha256"}}
        )
        if recorded != recomputed:
            raise ForecastFreezeError(
                f"Upstream manifest {name} does not match its own recorded hash "
                f"(recorded {recorded}, recomputed {recomputed}); the chain is broken."
            )
        chain[name] = {
            "manifest_sha256": recorded,
            "stage_conclusion": manifest.get("conclusion", {}).get("verdict"),
        }
    return chain


def freeze_stage(
    spec: StageFreezeSpec, *, outputs_dir: Path = FORECAST_OUTPUTS_DIR
) -> dict[str, Any]:
    """Seal one stage: hash it, check its conclusion, write its manifest."""
    assert_path_outside_forbidden_namespaces(outputs_dir)
    key_results = spec.key_results(outputs_dir)
    spec.checks(key_results)

    manifest: dict[str, Any] = {
        "stage": spec.stage,
        "freeze_version": spec.version,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "conclusion": spec.conclusion,
        "limitations": [dict(item) for item in spec.limitations],
        "key_results": key_results,
        "provenance_chain": _upstream_chain(spec, outputs_dir),
        "artifact_sha256": _hash_tree(
            {name: outputs_dir / name for name in spec.artifacts}, stage=spec.stage
        ),
        "source_module_sha256": _hash_tree(
            {name: PROJECT_ROOT / name for name in spec.source_modules}, stage=spec.stage
        ),
        "environment": _environment(),
        "sealed_seasons": list(FORECAST_SEALED_SEASONS),
        "phase_2_gate_requirements": list(PHASE_2_GATE),
        "phase_2_entry_point_exists": False,
    }
    if spec.specification is not None:
        specification = spec.specification()
        manifest["specification"] = specification
        manifest["specification_sha256"] = hash_json(specification)

    manifest["manifest_sha256"] = hash_json(
        {k: v for k, v in manifest.items() if k != "frozen_at_utc"}
    )
    (outputs_dir / spec.manifest_name).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (outputs_dir / spec.conclusion_name).write_text(render_stage_conclusion(spec, manifest))
    return manifest


def verify_stage(
    spec: StageFreezeSpec, *, outputs_dir: Path = FORECAST_OUTPUTS_DIR
) -> dict[str, Any]:
    """Re-hash a sealed stage and name anything that drifted."""
    manifest_path = outputs_dir / spec.manifest_name
    if not manifest_path.exists():
        raise ForecastFreezeError(f"No freeze manifest for {spec.stage} at {manifest_path}")
    frozen = json.loads(manifest_path.read_text())

    drift: dict[str, list[str]] = {}
    current_artifacts = _hash_tree(
        {name: outputs_dir / name for name in spec.artifacts}, stage=spec.stage
    )
    current_sources = _hash_tree(
        {name: PROJECT_ROOT / name for name in spec.source_modules}, stage=spec.stage
    )
    for label, recorded, current in (
        ("artifacts", frozen["artifact_sha256"], current_artifacts),
        ("source_modules", frozen["source_module_sha256"], current_sources),
    ):
        changed = [name for name, digest in recorded.items() if current.get(name) != digest]
        if changed:
            drift[label] = sorted(changed)

    for name, recorded in frozen.get("provenance_chain", {}).items():
        upstream_path = outputs_dir / name
        if not upstream_path.exists():
            drift.setdefault("provenance_chain", []).append(f"{name} missing")
            continue
        upstream = json.loads(upstream_path.read_text())
        if upstream.get("manifest_sha256") != recorded["manifest_sha256"]:
            drift.setdefault("provenance_chain", []).append(f"{name} re-frozen since")

    return {
        "stage": spec.stage,
        "frozen_at_utc": frozen["frozen_at_utc"],
        "manifest_sha256": frozen["manifest_sha256"],
        "matches_freeze": not drift,
        "drift": drift,
    }


def render_stage_conclusion(spec: StageFreezeSpec, manifest: dict[str, Any]) -> str:
    """A one-page frozen conclusion, generated from the manifest."""
    conclusion = manifest["conclusion"]
    lines = [
        f"# Contact Forecast -- {spec.stage} stage FROZEN",
        "",
        f"Frozen {manifest['frozen_at_utc']}.",
        f"Manifest SHA-256: `{manifest['manifest_sha256']}`.",
        "",
        "## Conclusion",
        "",
        f"**{conclusion['verdict']}**",
        "",
        conclusion["statement"],
        "",
    ]
    if conclusion.get("focal_comparison"):
        lines += ["### The focal comparison", "", conclusion["focal_comparison"], ""]
    if conclusion.get("must_not_be_described_as"):
        lines += ["### This result must NOT be described as", ""]
        lines += [f"- {item}" for item in conclusion["must_not_be_described_as"]]
        lines.append("")
    if conclusion.get("preserved_negative_results"):
        lines += ["### Negative results preserved", ""]
        lines += [f"- {item}" for item in conclusion["preserved_negative_results"]]
        lines.append("")
    lines += [
        "## Key results",
        "",
        "```json",
        json.dumps(manifest["key_results"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    lines += ["## Limitations", ""]
    for item in manifest["limitations"]:
        lines += [f"- **{item['id']}** -- {item['statement']}", ""]
    if manifest.get("provenance_chain"):
        lines += ["## Provenance chain", ""]
        for name, entry in sorted(manifest["provenance_chain"].items()):
            lines.append(
                f"- `{name}` -- verdict {entry['stage_conclusion']}, "
                f"manifest `{entry['manifest_sha256'][:16]}...`"
            )
        lines.append("")
    lines += [
        "## Season protection (unchanged)",
        "",
        f"Sealed seasons: {manifest['sealed_seasons']}. "
        f"Phase 2 entry point exists: {manifest['phase_2_entry_point_exists']}.",
        "",
    ]
    return "\n".join(lines) + "\n"
