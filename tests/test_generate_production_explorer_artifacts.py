"""Version 1.4.0 Phase 5: tests for `scripts/
generate_production_explorer_artifacts.py`, the production wiring between a
real Version 1.1 prospective snapshot's canonical outputs and the Play
Explorer browser-artifact generator.

Builds synthetic `outputs/`+`artifacts/` snapshot directory pairs shaped
like a real one (reusing `dashboard_snapshot_fixtures.write_snapshot` for
the manifest/integrity/public_score machinery, then extending it with a
`play_ledger.parquet`/`play_ledger_metadata.json` pair the same way
`tests/test_play_explorer_fixture_generator.py` does) -- never touches this
repo's real `outputs/prospective/`/`artifacts/prospective/`, never the
network, never trains a model.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import generate_production_explorer_artifacts as gpe
import pandas as pd
import pytest
from snapshot_data import compute_file_sha256

from dashboard_snapshot_fixtures import default_player_record, write_snapshot
from mlb_luck_score.scoring.play_ledger_metadata import build_play_ledger_metadata
from mlb_luck_score.scoring.play_ledger_schema import PLAY_LEDGER_COLUMNS, PLAY_LEDGER_VERSION

_OBSERVED = -0.25491574127584554
_EXPECTED = 1.290656107968476


def _resolved_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "play_id": "700001-10-3",
        "game_pk": 700001,
        "at_bat_number": 10,
        "pitch_number": 3,
        "game_date": "2026-06-01",
        "season": 2026,
        "batter_id": 12345,
        "stand": "L",
        "launch_speed": 107.6,
        "launch_angle": 33.0,
        "bb_type": "fly_ball",
        "spray_angle_approx": -5.9,
        "hit_distance_sc": 413.0,
        "is_scored": True,
        "outcome_class": "out",
        "observed_run_value": _OBSERVED,
        "p_out": 0.05692444081479338,
        "p_single": 0.0020191686532899677,
        "p_double": 0.019210357764228986,
        "p_triple": 0.003476039197145578,
        "p_home_run": 0.9183699935705423,
        "expected_run_value": _EXPECTED,
    }
    row["contact_luck_runs"] = row["observed_run_value"] - row["expected_run_value"]
    row.update(overrides)
    return row


def _ledger_df(*rows: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(list(rows))[list(PLAY_LEDGER_COLUMNS)]


def _add_ledger_to_snapshot(
    out_dir: Path,
    art_dir: Path,
    *,
    batter_id: int = 12345,
    data_through_date: str = "2026-08-16",
    play_ledger_version: str | None = PLAY_LEDGER_VERSION,
    tamper_ledger_after_hashing: bool = False,
) -> None:
    """Extends an already-`write_snapshot()`-built directory pair with a
    real `play_ledger.parquet`/`play_ledger_metadata.json`, and folds their
    hashes into the ALREADY-WRITTEN `integrity_hashes.json` -- exactly
    mirroring the real R2 snapshot's flat integrity_hashes.json covering
    both leaderboard/scorecard JSONs AND the canonical ledger.
    """
    df = _ledger_df(_resolved_row(batter_id=batter_id))
    ledger_path = out_dir / "play_ledger.parquet"
    df.to_parquet(ledger_path, index=False)

    metadata = build_play_ledger_metadata(
        df,
        season=2026,
        score_version="0.12.0",
        model_versions={"contact": "baseline_v02"},
        data_through_date=data_through_date,
    )
    if play_ledger_version is None:
        del metadata["play_ledger_version"]
    else:
        metadata["play_ledger_version"] = play_ledger_version
    metadata_path = out_dir / "play_ledger_metadata.json"
    metadata_path.write_text(json.dumps(metadata))

    if tamper_ledger_after_hashing:
        ledger_hash = compute_file_sha256(ledger_path)
        metadata_hash = compute_file_sha256(metadata_path)
        # Tamper AFTER hashing so the recorded hash goes stale, mirroring
        # dashboard_snapshot_fixtures.write_snapshot's "bad_hash" mode.
        df_tampered = _ledger_df(_resolved_row(batter_id=batter_id, play_id="700001-99-9"))
        df_tampered.to_parquet(ledger_path, index=False)
    else:
        ledger_hash = compute_file_sha256(ledger_path)
        metadata_hash = compute_file_sha256(metadata_path)

    integrity_path = art_dir / "integrity_hashes.json"
    integrity_hashes = json.loads(integrity_path.read_text())
    integrity_hashes["outputs/play_ledger.parquet"] = ledger_hash
    integrity_hashes["outputs/play_ledger_metadata.json"] = metadata_hash
    integrity_path.write_text(json.dumps(integrity_hashes, indent=2))


def _write_full_snapshot(
    tmp_path: Path,
    *,
    directory_name: str = "2026-08-16",
    data_through_date: str = "2026-08-16",
    snapshot_label: str | None = None,
    batter_id: int = 12345,
    batter_name: str = "Test Player",
    play_ledger_version: str | None = PLAY_LEDGER_VERSION,
    omit_ledger: bool = False,
    tamper_ledger_after_hashing: bool = False,
    omit_public_score: bool = False,
) -> tuple[Path, Path]:
    outputs_root = tmp_path / "outputs"
    artifacts_root = tmp_path / "artifacts"
    write_snapshot(
        outputs_root,
        artifacts_root,
        directory_name=directory_name,
        data_through_date=data_through_date,
        snapshot_label=snapshot_label,
        generated_at="2026-08-17T13:44:29.657692+00:00",
        players=[
            default_player_record(
                batter_id=batter_id, batter_name=batter_name, score=5.0, lower=1.0, upper=9.0
            )
        ],
    )
    out_dir = outputs_root / directory_name
    art_dir = artifacts_root / directory_name

    if omit_public_score:
        (out_dir / "public_score.json").unlink()
    if not omit_ledger:
        _add_ledger_to_snapshot(
            out_dir,
            art_dir,
            batter_id=batter_id,
            data_through_date=data_through_date,
            play_ledger_version=play_ledger_version,
            tamper_ledger_after_hashing=tamper_ledger_after_hashing,
        )
    return outputs_root, artifacts_root


# ---------------------------------------------------------------------------
# Version safety: fail closed (reuses the generator's own check, transitively)
# ---------------------------------------------------------------------------


class TestVersionSafety:
    def test_v2_ledger_accepted(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = _write_full_snapshot(tmp_path)
        result = gpe.generate_production_explorer_artifacts(
            snapshot_directory_name="2026-08-16",
            output_dir=tmp_path / "explorer_build",
            outputs_root=outputs_root,
            artifacts_root=artifacts_root,
        )
        assert result["play_ledger_version"] == "2.0"
        assert (tmp_path / "explorer_build" / "players.json").exists()

    def test_v1_ledger_rejected(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = _write_full_snapshot(tmp_path, play_ledger_version="1.0")
        with pytest.raises(Exception, match="1.0"):
            gpe.generate_production_explorer_artifacts(
                snapshot_directory_name="2026-08-16",
                output_dir=tmp_path / "explorer_build",
                outputs_root=outputs_root,
                artifacts_root=artifacts_root,
            )
        assert not (tmp_path / "explorer_build").exists()


# ---------------------------------------------------------------------------
# Canonical source integrity: reused from dashboard/snapshot_data.py
# ---------------------------------------------------------------------------


class TestIntegrity:
    def test_snapshot_not_found(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = _write_full_snapshot(tmp_path)
        with pytest.raises(gpe.ProductionExplorerGenerationError, match="No local snapshot"):
            gpe.generate_production_explorer_artifacts(
                snapshot_directory_name="2099-01-01",
                output_dir=tmp_path / "explorer_build",
                outputs_root=outputs_root,
                artifacts_root=artifacts_root,
            )

    def test_tampered_ledger_fails_integrity_check(self, tmp_path: Path) -> None:
        """A play_ledger.parquet whose bytes were changed after the
        recorded hash was computed must be caught by dashboard/
        snapshot_data.py's own hash-verified discovery -- this proves
        generate_production_explorer_artifacts() actually reuses that
        check rather than trusting the file blindly.
        """
        outputs_root, artifacts_root = _write_full_snapshot(
            tmp_path, tamper_ledger_after_hashing=True
        )
        with pytest.raises(gpe.ProductionExplorerGenerationError, match="integrity check"):
            gpe.generate_production_explorer_artifacts(
                snapshot_directory_name="2026-08-16",
                output_dir=tmp_path / "explorer_build",
                outputs_root=outputs_root,
                artifacts_root=artifacts_root,
            )
        assert not (tmp_path / "explorer_build").exists()

    def test_integrity_hashes_not_covering_ledger_is_rejected(self, tmp_path: Path) -> None:
        """integrity_hashes.json existing but NOT listing play_ledger.parquet
        (e.g. a hypothetical older snapshot shape) must be refused, not
        silently treated as "nothing to verify"."""
        outputs_root, artifacts_root = _write_full_snapshot(tmp_path)
        art_dir = artifacts_root / "2026-08-16"
        integrity_path = art_dir / "integrity_hashes.json"
        integrity_hashes = json.loads(integrity_path.read_text())
        del integrity_hashes["outputs/play_ledger.parquet"]
        integrity_path.write_text(json.dumps(integrity_hashes, indent=2))
        with pytest.raises(gpe.ProductionExplorerGenerationError, match="does not cover"):
            gpe.generate_production_explorer_artifacts(
                snapshot_directory_name="2026-08-16",
                output_dir=tmp_path / "explorer_build",
                outputs_root=outputs_root,
                artifacts_root=artifacts_root,
            )

    def test_missing_public_score_is_rejected(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = _write_full_snapshot(tmp_path, omit_public_score=True)
        # public_score.json is no longer covered by integrity_hashes.json either
        # (write_snapshot already hashed it before we deleted it), so this
        # exercises the plain missing-file guard.
        art_dir = artifacts_root / "2026-08-16"
        integrity_hashes = json.loads((art_dir / "integrity_hashes.json").read_text())
        del integrity_hashes["outputs/public_score.json"]
        (art_dir / "integrity_hashes.json").write_text(json.dumps(integrity_hashes, indent=2))
        # Recompute the manifest hash entry isn't affected; just re-verify the
        # snapshot still otherwise passes integrity for the files it does list.
        with pytest.raises(gpe.ProductionExplorerGenerationError, match="public_score"):
            gpe.generate_production_explorer_artifacts(
                snapshot_directory_name="2026-08-16",
                output_dir=tmp_path / "explorer_build",
                outputs_root=outputs_root,
                artifacts_root=artifacts_root,
            )


# ---------------------------------------------------------------------------
# Same-snapshot names, never the network
# ---------------------------------------------------------------------------


class TestBatterNames:
    def test_names_sourced_from_public_score_json(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = _write_full_snapshot(
            tmp_path, batter_id=999, batter_name="Same Snapshot Hitter"
        )
        gpe.generate_production_explorer_artifacts(
            snapshot_directory_name="2026-08-16",
            output_dir=tmp_path / "explorer_build",
            outputs_root=outputs_root,
            artifacts_root=artifacts_root,
        )
        catalog = json.loads((tmp_path / "explorer_build" / "players.json").read_text())
        assert catalog == [
            {"batter_id": 999, "batter_name": "Same Snapshot Hitter", "play_count": 1}
        ]

    def test_module_never_imports_network_or_mlb_api_code(self) -> None:
        """Structural guarantee mirroring `tests/test_dashboard_isolation.py`
        and `test_play_explorer_fixture_generator.py`'s own no-training-code
        checks: this script must never resolve names via a fresh MLB Stats
        API call -- only the same-snapshot public_score.json.
        """
        source = Path(gpe.__file__).read_text()
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        banned = {"requests", "urllib", "download_player_names", "fetch_player_names", "pybaseball"}
        assert not (imported & banned), imported & banned
        assert "fetch_player_names(" not in source


# ---------------------------------------------------------------------------
# Output location: never the canonical snapshot
# ---------------------------------------------------------------------------


class TestOutputLocation:
    def test_output_never_written_into_canonical_snapshot_dirs(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = _write_full_snapshot(tmp_path)
        out_dir = outputs_root / "2026-08-16"
        art_dir = artifacts_root / "2026-08-16"

        def _snapshot_files() -> dict[str, bytes]:
            files: dict[str, bytes] = {}
            for root in (out_dir, art_dir):
                for p in root.rglob("*"):
                    if p.is_file():
                        files[str(p)] = p.read_bytes()
            return files

        before = _snapshot_files()
        explorer_build_dir = tmp_path / "explorer_build" / "2026-08-16"
        gpe.generate_production_explorer_artifacts(
            snapshot_directory_name="2026-08-16",
            output_dir=explorer_build_dir,
            outputs_root=outputs_root,
            artifacts_root=artifacts_root,
        )
        after = _snapshot_files()
        assert before == after, (
            "generating Explorer artifacts must never mutate the canonical snapshot"
        )
        assert not str(explorer_build_dir).startswith(str(outputs_root))
        assert not str(explorer_build_dir).startswith(str(artifacts_root))
        assert (explorer_build_dir / "players.json").exists()


# ---------------------------------------------------------------------------
# Artifact counts reconcile / source ledger sha256 preserved
# ---------------------------------------------------------------------------


class TestArtifactReconciliation:
    def test_counts_and_source_sha256_reconcile(self, tmp_path: Path) -> None:
        outputs_root, artifacts_root = _write_full_snapshot(tmp_path)
        result = gpe.generate_production_explorer_artifacts(
            snapshot_directory_name="2026-08-16",
            output_dir=tmp_path / "explorer_build",
            outputs_root=outputs_root,
            artifacts_root=artifacts_root,
        )
        explore_metadata = json.loads(
            (tmp_path / "explorer_build" / "explore-metadata.json").read_text()
        )
        expected_sha256 = compute_file_sha256(outputs_root / "2026-08-16" / "play_ledger.parquet")
        assert explore_metadata["source_play_ledger_sha256"] == expected_sha256
        assert explore_metadata["play_count"] == result["row_count"]
        assert explore_metadata["player_count"] == result["player_count"]
        assert explore_metadata["game_count"] == result["game_count"]
        assert explore_metadata["play_ledger_version"] == "2.0"
        assert explore_metadata["season"] == 2026
        assert explore_metadata["data_through_date"] == "2026-08-16"


class TestCLI:
    def test_main_resolves_snapshot_dir_name_from_data_through_and_label(
        self, tmp_path: Path
    ) -> None:
        outputs_root, artifacts_root = _write_full_snapshot(
            tmp_path,
            directory_name="2026-08-16__refreshed",
            data_through_date="2026-08-16",
            snapshot_label="refreshed",
        )
        output_dir = tmp_path / "explorer_build"
        exit_code = gpe.main(
            [
                "--data-through",
                "2026-08-16",
                "--snapshot-label",
                "refreshed",
                "--output-dir",
                str(output_dir),
                "--outputs-root",
                str(outputs_root),
                "--artifacts-root",
                str(artifacts_root),
            ]
        )
        assert exit_code == 0
        assert (output_dir / "players.json").exists()
