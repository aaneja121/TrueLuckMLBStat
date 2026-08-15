"""Contact Luck v1.3.0 dashboard: end-to-end `/demo/` page build tests.

Builds against synthetic snapshot fixtures AND a synthetic demo fixture
(never the real committed `dashboard/demo_fixture.json` -- that file's own
correctness is covered by `tests/test_demo_fixture.py`) -- never touches
this repo's real `outputs/prospective`/`artifacts/prospective` directories
or the real demo fixture, and never invokes any model-training/scoring code.
"""

from __future__ import annotations

from pathlib import Path

import build as dashboard_build
import demo_content as dc
import pytest

from dashboard_snapshot_fixtures import default_player_record, write_snapshot
from test_dashboard_demo_content import _example, _write_fixture


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
                batter_id=1, batter_name="Alice Alpha", score=5.25, lower=1.1, upper=9.4
            )
        ],
    )
    return out_root, art_root


def _seed_demo_fixture(tmp_path: Path) -> Path:
    # Precise real values from the committed fixture (see
    # tests/test_demo_fixture.py) -- reconcile exactly, and round to the
    # same "-1.55"/"+0.71" runs display strings the page renders.
    return _write_fixture(
        tmp_path,
        [
            _example(
                example_id="hard_contact_out",
                expected_run_value=1.290656107968476,
                observed_run_value=-0.25491574127584554,
                contact_luck_runs=-1.5455718492443216,
            ),
            _example(
                example_id="weak_contact_single",
                expected_run_value=-0.2507498706080466,
                observed_run_value=0.4632655930819298,
                contact_luck_runs=0.7140154636899765,
            ),
        ],
    )


class TestDemoPageBuild:
    def test_demo_index_html_is_generated(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        demo_fixture_path = _seed_demo_fixture(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        assert (tmp_path / "dist" / "demo" / "index.html").exists()

    def test_both_examples_and_their_key_numbers_are_present(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        demo_fixture_path = _seed_demo_fixture(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "demo" / "index.html").read_text()
        assert 'data-example-id="hard_contact_out"' in html
        assert 'data-example-id="weak_contact_single"' in html
        assert "-1.55 runs" in html
        assert "+0.71 runs" in html
        assert "The contact was worth more than the recorded outcome." in html
        assert (
            "The recorded outcome was worth more than the contact was expected to produce." in html
        )

    def test_ball_sign_class_matches_contact_luck_sign_and_never_leaks_pre_reveal(
        self, tmp_path: Path
    ) -> None:
        """The ball's favorable/unfavorable sign class must be present in the
        static markup (so CSS alone can gate it -- see static/style.css) but
        the `demo-reality-revealed` ancestor class that activates its color
        must NEVER appear in server-rendered HTML -- it is purely JS-runtime
        state added by static/demo.js at Stage 3, or this whole "no early
        reveal" guarantee would be void from page load.
        """
        out_root, art_root = _seed_snapshot(tmp_path)
        demo_fixture_path = _seed_demo_fixture(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "demo" / "index.html").read_text()
        assert "demo-reality-revealed" not in html

        hard_card, weak_card = html.split('data-example-id="weak_contact_single"', 1)
        assert 'class="demo-ball demo-ball-unfavorable"' in hard_card
        assert 'class="demo-ball demo-ball-favorable"' in weak_card

    def test_field_animation_is_labeled_illustrative(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        demo_fixture_path = _seed_demo_fixture(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "demo" / "index.html").read_text()
        assert "Illustrative animation" in html
        assert "not a reconstruction of the actual ball flight or defender positioning" in html

    def test_demo_nav_link_present_on_every_page(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        demo_fixture_path = _seed_demo_fixture(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        for page in (
            "index.html",
            "methodology/index.html",
            "status/index.html",
            "demo/index.html",
        ):
            html = (tmp_path / "dist" / page).read_text()
            assert 'href="/demo/"' in html

    def test_demo_active_page_marks_the_demo_nav_link(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        demo_fixture_path = _seed_demo_fixture(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "demo" / "index.html").read_text()
        assert '<a href="/demo/" class="active">Demo</a>' in html

    def test_build_fails_loudly_when_demo_fixture_is_missing(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        with pytest.raises(dc.DemoContentError):
            dashboard_build.build_dashboard(
                out_dir=tmp_path / "dist",
                outputs_root=out_root,
                artifacts_root=art_root,
                demo_fixture_path=tmp_path / "does_not_exist.json",
                build_timestamp="2026-01-01T12:00:00+00:00",
            )

    def test_build_is_deterministic_given_a_fixed_timestamp(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        demo_fixture_path = _seed_demo_fixture(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist_a",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist_b",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        html_a = (tmp_path / "dist_a" / "demo" / "index.html").read_text()
        html_b = (tmp_path / "dist_b" / "demo" / "index.html").read_text()
        assert html_a == html_b

    def test_normal_leaderboard_build_is_unaffected_by_the_demo_page(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        demo_fixture_path = _seed_demo_fixture(tmp_path)
        result = dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "index.html").read_text()
        assert "Alice Alpha" in html
        assert result.manifest.player_count == 1
