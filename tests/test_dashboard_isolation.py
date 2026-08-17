"""Contact Luck v1.2 dashboard: structural read-only-boundary tests.

Statically parses every module under `dashboard/` (never imports/executes
the scoring pipeline to check this -- an import-based check could itself
trigger the very thing it's supposed to catch) and asserts none of them
import model-training modules, prospective-scoring orchestration,
final-evaluation scoring, or Statcast-download code, and none reference the
sealed 2025 final-evaluation namespace at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

DASHBOARD_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "dashboard"

_BANNED_IMPORT_PREFIXES = (
    "mlb_luck_score",  # the model-training/scoring package
    "prospective_scoring",  # Version 1.1 scoring orchestration
    "run_v1_1_2026_scoring",  # Version 1.1 CLI entry point
    "run_v1_final_evaluation",  # Version 1.0 sealed final-evaluation entry point
    "v1_final_evaluation_manifest",
    "prospective_ingestion",  # performs the real Statcast download
    "pybaseball",  # the Statcast download library itself
    "build_demo_fixture",  # Version 1.3.0 demo-fixture generator (demo/) -- trains a real model
    "build_counterfactual_grid",  # Version 1.3.1 counterfactual-grid generator (demo/) -- also trains a real model
    "build_play_explorer_dev_ledger",  # Version 1.4.0 Phase 4 (demo/) -- trains real models on 2021-2023/2024 data
    "measure_play_ledger_full_stack_reconciliation",  # Version 1.4.0 Phase 3.1 (demo/) -- also trains real models
    # Version 1.4.0 Phase 4 (demo/): the Play Explorer browser-artifact
    # generator itself never trains a model, but it DOES import
    # `mlb_luck_score.scoring.play_ledger_schema`/`play_ledger_metadata`
    # (pure validation, no training) -- banned by NAME here anyway so a
    # future `dashboard/*.py` importing it can't reach `mlb_luck_score`
    # transitively without this AST-based check (which only inspects each
    # dashboard file's own direct import names) catching it.
    "build_play_explorer_fixture",
)

_BANNED_SOURCE_SUBSTRINGS = (
    "final_evaluation",
    "SEALED_V1",
    "download_statcast",
    "download_development_data",
)


def _dashboard_py_files() -> list[Path]:
    return sorted(p for p in DASHBOARD_SOURCE_ROOT.rglob("*.py") if "dist" not in p.parts)


def _imported_module_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_dashboard_has_python_files_to_check() -> None:
    assert len(_dashboard_py_files()) >= 5


def test_no_dashboard_module_imports_scoring_or_download_code() -> None:
    violations = []
    for path in _dashboard_py_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        imported = _imported_module_names(tree)
        for banned in _BANNED_IMPORT_PREFIXES:
            if banned in imported:
                violations.append(
                    f"{path.relative_to(DASHBOARD_SOURCE_ROOT.parent)} imports '{banned}'"
                )
    assert not violations, (
        "Dashboard code must never import scoring/download modules:\n" + "\n".join(violations)
    )


def test_no_dashboard_module_references_sealed_2025_namespace() -> None:
    violations = []
    for path in _dashboard_py_files():
        text = path.read_text()
        for banned in _BANNED_SOURCE_SUBSTRINGS:
            if banned in text:
                violations.append(
                    f"{path.relative_to(DASHBOARD_SOURCE_ROOT.parent)} references '{banned}'"
                )
    assert not violations, (
        "Dashboard code must never reference the sealed 2025 namespace:\n" + "\n".join(violations)
    )


def test_dashboard_config_defines_no_final_evaluation_paths() -> None:
    import dashboard_config as dc

    for name in dir(dc):
        assert "final_evaluation" not in name.lower(), name
