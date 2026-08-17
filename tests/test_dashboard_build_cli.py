"""Version 1.4.0 Phase 5: regression tests for `dashboard/build.py`'s CLI
entry point (`main()`), specifically the fail-closed Play Explorer default.

THE key regression this task adds: a normal production invocation (no
`--explore-artifacts-dir` flag, exactly what `scripts/publish_snapshot.sh`
used to run before this task and what a bare `dashboard/build.py` still
runs if someone forgets the flag) must never silently publish the
committed, bounded `dashboard/explore_fixture/` development fixture. This
runs the REAL `main()` (not a hand-simulated stand-in) against a synthetic
snapshot, so it proves actual behavior, not just an assertion about it.
"""

from __future__ import annotations

from pathlib import Path

import build as dashboard_build

from dashboard_snapshot_fixtures import default_player_record, write_snapshot

REAL_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "dashboard" / "explore_fixture"


def _seed_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
    write_snapshot(
        out_root,
        art_root,
        directory_name="2026-01-01",
        data_through_date="2026-01-01",
        snapshot_label=None,
        generated_at="2026-01-01T00:00:00+00:00",
        players=[
            default_player_record(
                batter_id=1, batter_name="Alice Alpha", score=5.0, lower=1.0, upper=9.0
            )
        ],
    )
    return out_root, art_root


class TestResolveExplorePaths:
    def test_none_in_none_out(self) -> None:
        assert dashboard_build._resolve_explore_paths(None) == (
            None,
            None,
            None,
            None,
            None,
            None,
        )

    def test_directory_resolves_to_six_expected_paths(self, tmp_path: Path) -> None:
        players_path, players_dir, games_dir, metadata_path, showcase_path, sensitivity_dir = (
            dashboard_build._resolve_explore_paths(tmp_path)
        )
        assert players_path == tmp_path / "players.json"
        assert players_dir == tmp_path / "players"
        assert games_dir == tmp_path / "games"
        assert metadata_path == tmp_path / "explore-metadata.json"
        assert showcase_path == tmp_path / "showcase.json"
        assert sensitivity_dir == tmp_path / "showcase-sensitivity"


class TestCLIFailsClosedByDefault:
    def test_main_with_no_flags_never_enables_explorer(self, tmp_path: Path) -> None:
        """THE regression: `main([])` -- exactly what a bare
        `dashboard/build.py` invocation runs -- must build with the
        Explorer disabled, even though this repo checkout has a real,
        valid, committed `dashboard/explore_fixture/` sitting right there
        on disk. There is no code path in `main()` that can reach it
        without an explicit `--explore-artifacts-dir` flag.
        """
        assert (REAL_FIXTURE_DIR / "players.json").exists(), (
            "test precondition: the real committed fixture must exist for this to be a "
            "meaningful regression proof"
        )
        out_root, art_root = _seed_snapshot(tmp_path)
        result = dashboard_build.main(
            [],
            outputs_root=out_root,
            artifacts_root=art_root,
            out_dir=tmp_path / "dist",
        )
        assert not (tmp_path / "dist" / "explore").exists()
        assert not (tmp_path / "dist" / "plays").exists()
        homepage = (tmp_path / "dist" / "index.html").read_text()
        assert "Explore Plays" not in homepage
        assert "Alice Alpha" in homepage  # every other page still builds fine
        assert result.manifest.player_count == 1

    def test_main_with_explicit_fixture_dir_enables_explorer(self, tmp_path: Path) -> None:
        """The flip side: local/dev use of the committed fixture remains
        possible, but only via an EXPLICIT flag."""
        out_root, art_root = _seed_snapshot(tmp_path)
        dashboard_build.main(
            ["--explore-artifacts-dir", str(REAL_FIXTURE_DIR)],
            outputs_root=out_root,
            artifacts_root=art_root,
            out_dir=tmp_path / "dist",
        )
        assert (tmp_path / "dist" / "explore" / "index.html").exists()
        assert (tmp_path / "dist" / "explore" / "players.json").exists()
        homepage = (tmp_path / "dist" / "index.html").read_text()
        assert "Explore Plays" in homepage

    def test_build_dashboard_function_defaults_also_disable_explorer(self, tmp_path: Path) -> None:
        """Not just the CLI wrapper -- build_dashboard() itself has no
        fixture-pointing default at the Python level either, so no future
        caller can reintroduce the implicit fallback by calling the
        function directly instead of through main()."""
        out_root, art_root = _seed_snapshot(tmp_path)
        result = dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        assert not (tmp_path / "dist" / "explore").exists()
        assert result.manifest.player_count == 1
