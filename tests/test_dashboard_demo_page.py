"""Contact Luck v1.3.0 dashboard: end-to-end `/demo/` page build tests.

Builds against synthetic snapshot fixtures AND a synthetic demo fixture
(never the real committed `dashboard/demo_fixture.json` -- that file's own
correctness is covered by `tests/test_demo_fixture.py`) -- never touches
this repo's real `outputs/prospective`/`artifacts/prospective` directories
or the real demo fixture, and never invokes any model-training/scoring code.
"""

from __future__ import annotations

import re
from pathlib import Path

import build as dashboard_build
import demo_content as dc
import demo_counterfactual_content as dcc
import pytest

from dashboard_snapshot_fixtures import default_player_record, write_snapshot
from test_dashboard_counterfactual_content import _both_plays, _write_grid
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


def _seed_counterfactual_grid(tmp_path: Path) -> Path:
    # A tiny synthetic grid, distinct from the real committed one -- these
    # tests only need "a structurally valid grid exists," not real model
    # probabilities (that's covered end-to-end by
    # tests/test_counterfactual_grid.py against the real committed file).
    return _write_grid(tmp_path, _both_plays())


class TestDemoPageBuild:
    def test_demo_index_html_is_generated(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        demo_fixture_path = _seed_demo_fixture(tmp_path)
        counterfactual_grid_path = _seed_counterfactual_grid(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            demo_counterfactual_grid_path=counterfactual_grid_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        assert (tmp_path / "dist" / "demo" / "index.html").exists()

    def test_both_examples_and_their_key_numbers_are_present(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        demo_fixture_path = _seed_demo_fixture(tmp_path)
        counterfactual_grid_path = _seed_counterfactual_grid(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            demo_counterfactual_grid_path=counterfactual_grid_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "demo" / "index.html").read_text()
        assert 'data-example-id="hard_contact_out"' in html
        assert 'data-example-id="weak_contact_single"' in html
        # Redesign Phase 1 numeric primitive: signed Contact Luck / run
        # values render with U+2212 MINUS SIGN, not ASCII hyphen-minus,
        # so they align in a tabular-figure column. Asserted with the
        # real glyph so a silent regression to "-" fails here.
        assert "\u22121.55 runs" in html
        assert "-1.55 runs" not in html
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
        counterfactual_grid_path = _seed_counterfactual_grid(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            demo_counterfactual_grid_path=counterfactual_grid_path,
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
        counterfactual_grid_path = _seed_counterfactual_grid(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            demo_counterfactual_grid_path=counterfactual_grid_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "demo" / "index.html").read_text()
        # Phase 7 rewrote the caption; the disclaimer it carries is what
        # this test protects, so both halves of the claim are asserted.
        assert "Illustrative ball flight" in html
        assert "Not a reconstruction of the real flight" in html
        assert "where the" in html and "fielders stood" in html

    def test_demo_nav_link_present_on_every_page(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        demo_fixture_path = _seed_demo_fixture(tmp_path)
        counterfactual_grid_path = _seed_counterfactual_grid(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            demo_counterfactual_grid_path=counterfactual_grid_path,
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
        counterfactual_grid_path = _seed_counterfactual_grid(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            demo_counterfactual_grid_path=counterfactual_grid_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "demo" / "index.html").read_text()
        # Redesign Phase 2 moved the active-route marker from a bare
        # `class="active"` to `is-active` PLUS `aria-current="page"` -- the
        # state is now in the accessibility tree as well as the stylesheet.
        # Asserted at the contract ("/demo/ is the one route marked current")
        # rather than at one spelling of the markup.
        nav = html.split('<nav class="site-nav"', 1)[1].split("</nav>", 1)[0]
        assert nav.count('aria-current="page"') == 1
        current = [line for line in nav.splitlines() if 'aria-current="page"' in line]
        assert current and "/demo/" in current[0] and "How It Works" in current[0]
        assert "is-active" in current[0]

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
        counterfactual_grid_path = _seed_counterfactual_grid(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist_a",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            demo_counterfactual_grid_path=counterfactual_grid_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist_b",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            demo_counterfactual_grid_path=counterfactual_grid_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        html_a = (tmp_path / "dist_a" / "demo" / "index.html").read_text()
        html_b = (tmp_path / "dist_b" / "demo" / "index.html").read_text()
        assert html_a == html_b

    def test_normal_leaderboard_build_is_unaffected_by_the_demo_page(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        demo_fixture_path = _seed_demo_fixture(tmp_path)
        counterfactual_grid_path = _seed_counterfactual_grid(tmp_path)
        result = dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            demo_counterfactual_grid_path=counterfactual_grid_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "index.html").read_text()
        assert "Alice Alpha" in html
        assert result.manifest.player_count == 1


class TestSimulatorSectionBuild:
    """Contact Luck v1.3.1: 'Try It Yourself' counterfactual simulator
    section -- built against the SAME synthetic snapshot/fixture as
    `TestDemoPageBuild` above, plus a synthetic counterfactual grid (never
    the real committed `dashboard/demo_counterfactual_grid.json` -- that
    file's own scientific correctness is covered end-to-end by
    `tests/test_counterfactual_grid.py`).
    """

    def _build(self, tmp_path: Path) -> str:
        out_root, art_root = _seed_snapshot(tmp_path)
        demo_fixture_path = _seed_demo_fixture(tmp_path)
        counterfactual_grid_path = _seed_counterfactual_grid(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            demo_counterfactual_grid_path=counterfactual_grid_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        return (tmp_path / "dist" / "demo" / "index.html").read_text()

    def test_demo_index_html_still_builds_with_the_simulator_section(self, tmp_path: Path) -> None:
        html = self._build(tmp_path)
        # Phase 7 set the heading in sentence case, matching every other
        # heading on the site. The section, not its casing, is the contract.
        assert "Try it yourself" in html
        assert 'data-role="counterfactual-simulator"' in html

    def test_existing_v1_3_0_walkthrough_content_remains_present(self, tmp_path: Path) -> None:
        """Requirement #21: adding the simulator must not remove or alter
        the existing two-play walkthrough.
        """
        html = self._build(tmp_path)
        # Phase 7 retitled the route to match its own nav label ("How It
        # Works"); the walkthrough itself is what must survive.
        assert "How Contact Luck works" in html
        assert 'data-role="demo-play-all"' in html
        assert html.count('class="demo-card"') == 2
        # Redesign Phase 1 numeric primitive: signed Contact Luck / run
        # values render with U+2212 MINUS SIGN, not ASCII hyphen-minus,
        # so they align in a tabular-figure column. Asserted with the
        # real glyph so a silent regression to "-" fails here.
        assert "\u22121.55 runs" in html
        assert "-1.55 runs" not in html
        assert "+0.71 runs" in html

    def test_both_reference_plays_are_selectable(self, tmp_path: Path) -> None:
        """Requirement #16. Attribute order/whitespace inside the button tag
        is a template-formatting detail, not part of the contract -- match
        with a whitespace-tolerant regex rather than an exact substring.
        """
        html = self._build(tmp_path)
        assert re.search(
            r'data-role="simulator-select-play"\s+data-example-id="hard_contact_out"', html
        )
        assert re.search(
            r'data-role="simulator-select-play"\s+data-example-id="weak_contact_single"', html
        )

    def test_ev_and_la_sliders_are_present_as_index_based_range_inputs(
        self, tmp_path: Path
    ) -> None:
        """Requirement #17/#19: sliders exist; per the amendment, the DOM
        `value`/`max` are array INDICES (populated by JS from the fetched
        grid), not raw EV/LA -- the static markup only needs min="0" as a
        placeholder since real bounds depend on which play is selected.
        """
        html = self._build(tmp_path)
        assert 'data-role="simulator-ev-slider"' in html
        assert 'data-role="simulator-la-slider"' in html
        assert 'type="range"' in html

    def test_outcome_controls_exist_for_all_five_classes(self, tmp_path: Path) -> None:
        """Requirement #18."""
        html = self._build(tmp_path)
        for cls in ("out", "single", "double", "triple", "home_run"):
            assert re.search(
                rf'data-role="simulator-outcome-button"\s+data-outcome-class="{cls}"', html
            )

    def test_current_ev_la_and_expected_rv_placeholders_are_represented(
        self, tmp_path: Path
    ) -> None:
        """Requirement #19."""
        html = self._build(tmp_path)
        assert 'data-role="simulator-ev-value"' in html
        assert 'data-role="simulator-la-value"' in html
        assert 'data-role="simulator-expected-rv"' in html

    def test_ceteris_paribus_ui_copy_is_present_and_not_physics_language(
        self, tmp_path: Path
    ) -> None:
        html = self._build(tmp_path)
        flat = " ".join(html.split())
        # Phase 7 wording. Both halves of the ceteris-paribus contract are
        # still stated: everything else is held at the reference play's own
        # values, and nothing here is a recorded play.
        assert "stays fixed at the reference play's own values" in flat
        assert "hypothetical contact, not a recorded play" in flat
        for banned_phrase in (
            "physics simulator",
            "simulated trajectory",
            "what this ball would actually travel",
            "predicted distance",
            "exact flight path",
            "exact spray chart",
            "real trajectory reconstruction",
        ):
            assert banned_phrase not in html.lower()

    def test_reset_control_and_grid_fetch_script_are_present(self, tmp_path: Path) -> None:
        html = self._build(tmp_path)
        assert 'data-role="simulator-reset"' in html
        assert "demo_simulator.js" in html

    def test_simulator_field_visualization_is_present_and_labeled_illustrative(
        self, tmp_path: Path
    ) -> None:
        html = self._build(tmp_path)
        assert 'data-role="simulator-field-svg"' in html
        assert 'data-role="simulator-field-path"' in html
        assert 'data-role="simulator-field-ball"' in html
        assert "Illustrative field view, driven by the two sliders" in html
        assert "not where a ball would physically land" in html

    def test_counterfactual_grid_json_is_copied_into_dist_demo(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        demo_fixture_path = _seed_demo_fixture(tmp_path)
        counterfactual_grid_path = _seed_counterfactual_grid(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            demo_fixture_path=demo_fixture_path,
            demo_counterfactual_grid_path=counterfactual_grid_path,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        copied = tmp_path / "dist" / "demo" / "counterfactual-grid.json"
        assert copied.exists()
        assert copied.read_text() == counterfactual_grid_path.read_text()

    def test_build_fails_loudly_when_counterfactual_grid_is_missing(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        demo_fixture_path = _seed_demo_fixture(tmp_path)
        with pytest.raises(dcc.CounterfactualContentError):
            dashboard_build.build_dashboard(
                out_dir=tmp_path / "dist",
                outputs_root=out_root,
                artifacts_root=art_root,
                demo_fixture_path=demo_fixture_path,
                demo_counterfactual_grid_path=tmp_path / "does_not_exist.json",
                build_timestamp="2026-01-01T12:00:00+00:00",
            )

    def test_build_is_deterministic_with_the_simulator_section(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        demo_fixture_path = _seed_demo_fixture(tmp_path)
        counterfactual_grid_path = _seed_counterfactual_grid(tmp_path)
        for label in ("dist_a", "dist_b"):
            dashboard_build.build_dashboard(
                out_dir=tmp_path / label,
                outputs_root=out_root,
                artifacts_root=art_root,
                demo_fixture_path=demo_fixture_path,
                demo_counterfactual_grid_path=counterfactual_grid_path,
                build_timestamp="2026-01-01T12:00:00+00:00",
            )
        html_a = (tmp_path / "dist_a" / "demo" / "index.html").read_text()
        html_b = (tmp_path / "dist_b" / "demo" / "index.html").read_text()
        assert html_a == html_b
