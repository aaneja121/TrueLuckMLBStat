"""Contact Luck v1.1: isolated namespace tests.

Verifies every 2026 path resolves beneath the isolated `data/prospective/
2026`, `outputs/prospective/v1_1`, or `artifacts/prospective/v1_1` roots,
that path traversal or a development-cache/sealed-2025 path is rejected, and
that a development runner cannot discover prospective data placed only in
the isolated namespace.
"""

from __future__ import annotations

from pathlib import Path

import prospective_config as pc
import pytest

from mlb_luck_score.config import RAW_DATA_DIR, development_raw_path, game_metadata_path
from mlb_luck_score.data.join_venue_metadata import JoinVenueMetadataError, load_game_metadata


def test_default_raw_dir_is_under_data_prospective_2026() -> None:
    assert pc.PROSPECTIVE_RAW_DIR.is_relative_to(pc.PROJECT_ROOT / "data")
    assert pc.PROSPECTIVE_RAW_DIR.parts[-3:] == ("data", "prospective", "2026")


def test_default_outputs_dir_is_under_outputs_prospective_v1_1() -> None:
    assert pc.PROSPECTIVE_OUTPUTS_DIR.parts[-3:] == ("outputs", "prospective", "v1_1")


def test_default_artifacts_dir_is_under_artifacts_prospective_v1_1() -> None:
    assert pc.PROSPECTIVE_ARTIFACTS_DIR.parts[-3:] == ("artifacts", "prospective", "v1_1")


def test_raw_statcast_path_resolves_inside_the_namespace() -> None:
    path = development_raw_path(pc.PROSPECTIVE_RAW_DIR, 2026)
    assert path.is_relative_to(pc.PROSPECTIVE_RAW_DIR)


def test_namespace_guard_accepts_a_path_within_the_root(tmp_path: Path) -> None:
    root = tmp_path / "prospective" / "2026"
    root.mkdir(parents=True)
    nested = root / "subdir" / "file.parquet"
    pc._assert_within_namespace(nested, root, label="test")  # must not raise


def test_namespace_guard_rejects_a_sibling_development_cache_path(tmp_path: Path) -> None:
    root = tmp_path / "prospective" / "2026"
    root.mkdir(parents=True)
    outside = tmp_path / "raw"
    outside.mkdir()
    with pytest.raises(pc.NamespaceViolationError):
        pc._assert_within_namespace(outside, root, label="test")


def test_namespace_guard_rejects_path_traversal_escape(tmp_path: Path) -> None:
    root = tmp_path / "prospective" / "2026"
    root.mkdir(parents=True)
    traversal = root / ".." / ".." / "raw"
    with pytest.raises(pc.NamespaceViolationError):
        pc._assert_within_namespace(traversal, root, label="test")


# ---------------------------------------------------------------------------
# Refusing the sealed Version 1.0 namespaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sealed_root",
    [pc.SEALED_V1_RAW_DIR, pc.SEALED_V1_OUTPUTS_DIR, pc.SEALED_V1_ARTIFACTS_DIR],
)
def test_assert_not_sealed_v1_namespace_rejects_each_sealed_root(sealed_root: Path) -> None:
    with pytest.raises(pc.SealedNamespaceAccessError):
        pc.assert_not_sealed_v1_namespace(sealed_root / "some_file.parquet", label="test")


def test_assert_not_sealed_v1_namespace_accepts_the_prospective_namespace() -> None:
    pc.assert_not_sealed_v1_namespace(
        pc.PROSPECTIVE_RAW_DIR / "statcast_2026_regular_season.parquet", label="test"
    )  # must not raise


def test_assert_not_sealed_v1_namespace_accepts_unrelated_paths(tmp_path: Path) -> None:
    pc.assert_not_sealed_v1_namespace(tmp_path / "anything.json", label="test")  # must not raise


# ---------------------------------------------------------------------------
# A development runner cannot discover prospective-only data
# ---------------------------------------------------------------------------


def test_development_file_discovery_cannot_find_prospective_data(tmp_path: Path) -> None:
    """2026 game metadata written ONLY into the isolated prospective
    namespace must be invisible to a development runner pointed at the
    development raw cache -- mirrors Version 1.0's own equivalent test.
    """
    isolated_root = tmp_path / "prospective" / "2026"
    isolated_root.mkdir(parents=True)
    isolated_path = game_metadata_path(isolated_root, 2026)
    isolated_path.write_text("not a real parquet file, presence is what matters")

    dev_raw_dir = tmp_path / "raw"
    dev_raw_dir.mkdir()

    with pytest.raises(JoinVenueMetadataError, match="Missing game-metadata file"):
        load_game_metadata(dev_raw_dir, [2026])


def test_real_development_raw_dir_has_no_2026_files() -> None:
    """Sanity check against the REAL repository `data/raw` directory (not a
    tmp_path fixture): confirms this construction task never accidentally
    wrote a real 2026 file into the development cache.
    """
    if not RAW_DATA_DIR.exists():
        return
    real_2026_files = list(RAW_DATA_DIR.glob("*2026*"))
    assert real_2026_files == []
