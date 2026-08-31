"""Redesign Phase 2 -- the application shell.

The shell is `base.html` plus its stylesheet and script: skip link, header,
navigation, global search combobox, page grid and footer. Everything asserted
here is a *contract* the later phases build on, not a styling snapshot -- these
tests should survive Phase 3+ reskinning the routes inside the shell.

Structural HTML assertions are made against a real build (every emitted route,
not just the homepage), because the failure mode they guard is "the shell
regressed on one template that overrode a block."

Rendered geometry -- 44px targets, no two-row nav, the mobile sheet -- is not
assertable here: this repository has no browser-automation test dependency, and
those are verified with the globally configured Playwright MCP (see
`CLAUDE.md` § Verification before claiming done).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import build as dashboard_build
import pytest

from dashboard_snapshot_fixtures import default_player_record, write_snapshot

STYLE_CSS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "style.css"
APP_JS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "app.js"
BASE_HTML = Path(__file__).resolve().parents[1] / "dashboard" / "templates" / "base.html"


def _players() -> list[dict]:
    return [
        default_player_record(
            batter_id=1, batter_name="Alice Alpha", score=5.25, lower=1.1, upper=9.4, bbe=310
        ),
        default_player_record(
            batter_id=2, batter_name="Bob Beta", score=-3.5, lower=-7.2, upper=1.3, bbe=280
        ),
        default_player_record(
            batter_id=3,
            batter_name="Carl Gamma",
            score=1.0,
            lower=-4.0,
            upper=6.0,
            bbe=40,
            qualification_status="small_sample",
        ),
    ]


@pytest.fixture(scope="module")
def built_site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("shell")
    out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
    for day in ("2026-01-01", "2026-01-02"):
        write_snapshot(
            out_root,
            art_root,
            directory_name=day,
            data_through_date=day,
            snapshot_label=None,
            generated_at=f"{day}T00:00:00+00:00",
            players=_players(),
        )
    dist = tmp_path / "dist"
    dashboard_build.build_dashboard(
        out_dir=dist,
        outputs_root=out_root,
        artifacts_root=art_root,
        build_timestamp="2026-01-02T12:00:00+00:00",
    )
    return dist


def _all_pages(dist: Path) -> list[Path]:
    return sorted(dist.rglob("index.html"))


def _strip_css_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


class TestSkipLink:
    """Phase 2 completion criterion: a skip link on every route."""

    def test_every_route_opens_with_a_skip_link_to_main(self, built_site: Path) -> None:
        pages = _all_pages(built_site)
        assert len(pages) >= 5
        for page in pages:
            html = page.read_text()
            assert '<a class="skip-link" href="#main-content">' in html, page

    def test_skip_link_is_the_first_focusable_element_on_every_route(
        self, built_site: Path
    ) -> None:
        # The DOM order of focusables is the tab order here: nothing in the
        # shell sets a positive tabindex (which would be its own defect).
        focusable = re.compile(r"<(a|button|input|select|textarea|summary)\b[^>]*>", re.I)
        for page in _all_pages(built_site):
            body = page.read_text().split("<body>", 1)[1]
            first = focusable.search(body)
            assert first is not None, page
            assert 'class="skip-link"' in first.group(0), (page, first.group(0))

    def test_skip_link_target_exists_and_is_focusable(self, built_site: Path) -> None:
        for page in _all_pages(built_site):
            html = page.read_text()
            assert 'id="main-content"' in html, page
            # Without `tabindex="-1"` the browser moves the scroll position but
            # leaves focus where it was, so the next Tab returns to the header.
            assert re.search(r'<main[^>]*\btabindex="-1"', html), page

    def test_skip_link_is_never_display_none(self) -> None:
        css = _strip_css_comments(STYLE_CSS.read_text())
        block = css.split(".skip-link {", 1)[1].split("}", 1)[0]
        assert "display: none" not in block
        assert "transform" in block  # moved off-canvas, still in the tab order

    def test_skip_link_reveals_on_plain_focus_not_only_focus_visible(self) -> None:
        css = _strip_css_comments(STYLE_CSS.read_text())
        assert ".skip-link:focus," in css or ".skip-link:focus\n" in css


class TestNavigation:
    def test_every_route_is_reachable_from_every_page(self, built_site: Path) -> None:
        for page in _all_pages(built_site):
            html = page.read_text()
            nav = html.split('<nav class="site-nav"', 1)[1].split("</nav>", 1)[0]
            for label in ("Leaderboard", "How It Works", "Methodology", "Data &amp; status"):
                assert label in nav, (page, label)

    def test_active_route_is_marked_with_aria_current_and_a_class(self, built_site: Path) -> None:
        home = (built_site / "index.html").read_text()
        nav = home.split('<nav class="site-nav"', 1)[1].split("</nav>", 1)[0]
        assert nav.count('aria-current="page"') == 1
        assert 'class="is-active" aria-current="page"' in nav

    def test_active_state_is_not_carried_by_colour_alone(self) -> None:
        css = _strip_css_comments(STYLE_CSS.read_text())
        block = css.split(".site-nav-list a.is-active {", 1)[1].split("}", 1)[0]
        # A second, non-colour channel: the rule under the route.
        assert "border-bottom-color" in block
        assert "font-weight" in block

    def test_mobile_disclosure_is_a_named_control_with_expanded_state(
        self, built_site: Path
    ) -> None:
        html = (built_site / "index.html").read_text()
        toggle = html.split('<button class="site-nav-toggle"', 1)[1].split("</button>", 1)[0]
        assert 'aria-expanded="false"' in toggle
        assert 'aria-controls="site-nav-list"' in toggle
        assert ">Menu" in toggle  # a word, not an icon

    def test_nav_works_without_javascript(self, built_site: Path) -> None:
        html = (built_site / "index.html").read_text()
        noscript = html.split("<noscript>", 1)[1].split("</noscript>", 1)[0]
        assert ".site-nav-list[hidden]" in noscript
        assert ".site-nav-toggle { display: none; }" in noscript

    def test_nav_targets_declare_the_minimum_tap_area(self) -> None:
        css = _strip_css_comments(STYLE_CSS.read_text())
        block = css.split(".site-nav-list a {", 1)[1].split("}", 1)[0]
        assert "min-height: var(--target-min)" in block
        root = css.split(":root {", 1)[1].split("}", 1)[0]
        assert "--target-min: 44px" in root


class TestGlobalSearchCombobox:
    def test_input_is_a_labelled_combobox(self, built_site: Path) -> None:
        html = (built_site / "index.html").read_text()
        assert '<label class="global-search-label" for="global-player-search">' in html
        field = html.split('id="global-player-search"', 1)[1].split(">", 1)[0]
        assert 'role="combobox"' in field
        assert 'aria-expanded="false"' in field
        assert 'aria-controls="global-player-search-listbox"' in field
        assert 'aria-autocomplete="list"' in field

    def test_placeholder_is_not_the_label(self, built_site: Path) -> None:
        html = (built_site / "index.html").read_text()
        # A placeholder may exist as a format hint, but the programmatic name
        # must come from the <label> (accessibility.md #5).
        assert 'for="global-player-search"' in html
        assert 'aria-label="Find a player by name"' not in html

    def test_results_container_is_a_listbox(self, built_site: Path) -> None:
        html = (built_site / "index.html").read_text()
        listbox = html.split('id="global-player-search-listbox"', 1)[1].split(">", 1)[0]
        assert 'role="listbox"' in listbox
        assert "aria-label=" in listbox

    def test_search_is_present_on_every_route(self, built_site: Path) -> None:
        for page in _all_pages(built_site):
            assert 'data-role="global-player-search"' in page.read_text(), page

    def test_script_implements_the_keyboard_contract(self) -> None:
        js = APP_JS.read_text()
        for token in (
            "aria-activedescendant",
            '"ArrowDown"',
            '"ArrowUp"',
            '"Enter"',
            '"Escape"',
            'setAttribute("role", "option")',
            'setAttribute("aria-selected"',
        ):
            assert token in js, token

    def test_empty_state_is_not_a_selectable_option(self) -> None:
        js = APP_JS.read_text()
        empty = js.split("if (!matches.length) {", 1)[1].split("setExpanded(true);", 1)[0]
        assert 'setAttribute("role", "presentation")' in empty
        assert 'role", "option' not in empty


class TestSearchIndexMarksUnrankedPlayers:
    def test_index_carries_a_build_time_ranked_flag(self, built_site: Path) -> None:
        html = (built_site / "index.html").read_text()
        raw = html.split('id="player-index-data">', 1)[1].split("</script>", 1)[0]
        index = {row["batter_name"]: row for row in json.loads(raw)}
        assert index["Alice Alpha"]["ranked"] is True
        # Searchable, present, and marked -- never suppressed.
        assert index["Carl Gamma"]["ranked"] is False

    def test_rank_semantics_are_resolved_in_python_not_javascript(self) -> None:
        js = APP_JS.read_text()
        assert "qualification_status" not in js
        assert "p.ranked === false" in js


class TestPageGrid:
    def test_header_main_and_footer_share_one_horizontal_anchor(self, built_site: Path) -> None:
        for page in _all_pages(built_site):
            html = page.read_text()
            assert '<div class="site-header-inner page-shell">' in html, page
            assert '<main class="page-shell"' in html, page
            assert '<div class="site-footer-inner page-shell">' in html, page

    def test_both_measures_are_declared_as_tokens(self) -> None:
        css = _strip_css_comments(STYLE_CSS.read_text())
        root = css.split(":root {", 1)[1].split("}", 1)[0]
        assert "--measure-data: 1200px" in root
        assert "--measure-prose: 68ch" in root
        block = css.split(".page-shell {", 1)[1].split("}", 1)[0]
        assert "max-width: var(--measure-data)" in block

    def test_full_bleed_bands_bleed_by_exactly_the_gutter_token(self) -> None:
        # The gutter changes at mobile; a hardcoded -20px bleed would leave a
        # 4px inset at 390 once it did.
        css = _strip_css_comments(STYLE_CSS.read_text())
        # `.page-hero` was retired in Phase 3 (the homepage band and its 3px
        # decorative amber border); `.player-header` is the remaining
        # full-bleed band until Phase 4 rebuilds it.
        for selector in (".player-header {",):
            block = css.split(selector, 1)[1].split("}", 1)[0]
            assert "calc(-1 * var(--gutter))" in block, selector
            assert "-20px" not in block, selector


class TestLinkSemantics:
    """`docs/design/color.md`: the diverging pair is reserved for the sign of a
    point estimate. An ordinary link may not borrow favorable blue."""

    def test_default_link_is_ink_not_favorable_blue(self) -> None:
        css = _strip_css_comments(STYLE_CSS.read_text())
        block = css.split("\na {", 1)[1].split("}", 1)[0]
        assert "var(--fav-text)" not in block
        assert "var(--text-primary)" in block

    def test_link_affordance_survives_without_colour(self) -> None:
        css = _strip_css_comments(STYLE_CSS.read_text())
        block = css.split("\na {", 1)[1].split("}", 1)[0]
        assert "text-decoration: underline" in block

    def test_no_semantic_hue_is_used_as_a_link_or_shell_colour(self) -> None:
        css = _strip_css_comments(STYLE_CSS.read_text())
        for selector in (
            ".site-nav-list a {",
            ".site-wordmark {",
            ".site-footer-provenance a {",
        ):
            block = css.split(selector, 1)[1].split("}", 1)[0]
            for banned in ("--fav-text", "--fav-mark", "--unfav-text", "--unfav-mark"):
                assert banned not in block, (selector, banned)


class TestAmberIsReserved:
    """`docs/design/color.md`: amber means 'the frozen official record'. It
    marks the active route, the data-through badge, the official-rank column,
    the zero spine and the focus ring -- and nothing else."""

    def test_no_decorative_accent_on_every_heading(self) -> None:
        css = _strip_css_comments(STYLE_CSS.read_text())
        assert "h2::before" not in css

    def test_data_through_uses_the_accent_text_grade(self) -> None:
        css = _strip_css_comments(STYLE_CSS.read_text())
        block = css.split(".data-through-value {", 1)[1].split("}", 1)[0]
        assert "var(--accent-amber-text)" in block

    def test_data_through_is_not_a_status_pill(self) -> None:
        # `--radius-pill` is reserved for qualification status and the
        # Favorable/Unfavorable badge (docs/design/layout.md § Geometry).
        css = _strip_css_comments(STYLE_CSS.read_text())
        block = css.split(".data-through {", 1)[1].split("}", 1)[0]
        assert "radius-pill" not in block
        assert "999px" not in block

    def test_wordmark_carries_no_accent_ornament(self, built_site: Path) -> None:
        html = (built_site / "index.html").read_text()
        header = html.split("<header", 1)[1].split("</header>", 1)[0]
        assert "seam-mark" not in header


class TestProvenanceAndFooter:
    def test_data_through_date_is_machine_readable_and_human_readable(
        self, built_site: Path
    ) -> None:
        for page in _all_pages(built_site):
            html = page.read_text()
            assert '<time class="data-through-value" datetime="2026-01-02"' in html, page
            assert "Data through" in html, page

    def test_footer_keeps_the_retrospective_framing_verbatim(self, built_site: Path) -> None:
        expected = (
            "Contact Luck is a retrospective description of realized outcomes, not a measure of"
        )
        for page in _all_pages(built_site):
            assert expected in page.read_text(), page

    def test_footer_routes_to_methodology_and_status(self, built_site: Path) -> None:
        html = (built_site / "index.html").read_text()
        footer = html.split("<footer", 1)[1].split("</footer>", 1)[0]
        assert "methodology/" in footer
        assert "status/" in footer

    def test_footer_prose_is_held_to_the_prose_measure(self, built_site: Path) -> None:
        html = (built_site / "index.html").read_text()
        assert 'class="site-footer-framing prose"' in html
        css = _strip_css_comments(STYLE_CSS.read_text())
        block = css.split(".prose {", 1)[1].split("}", 1)[0]
        assert "var(--measure-prose)" in block

    def test_footer_is_not_a_card(self) -> None:
        css = _strip_css_comments(STYLE_CSS.read_text())
        block = css.split(".site-footer {", 1)[1].split("}", 1)[0]
        assert "border-top" in block
        assert "background" not in block
        assert "border-radius" not in block
        assert "box-shadow" not in block


class TestChevronCollisionResolved:
    """Phase 1 gave the open outward chevron a meaning: 'the interval continues
    past this edge'. A decorative chevron on the same page is a collision."""

    def test_player_header_ornament_is_gone_from_the_template(self) -> None:
        html = (
            Path(__file__).resolve().parents[1] / "dashboard" / "templates" / "player.html"
        ).read_text()
        assert "player-header-mark" not in html

    def test_player_header_ornament_is_gone_from_the_stylesheet(self) -> None:
        css = _strip_css_comments(STYLE_CSS.read_text())
        assert ".player-header-mark" not in css

    def test_no_decorative_chevron_renders_on_a_player_page(self, built_site: Path) -> None:
        """Phase 4 replaced `.player-header` with the verdict article. The
        rule is unchanged and now reaches further: the chevron has a MEANING
        in the scale vocabulary ("the interval continues past this edge"),
        so no decorative one may appear anywhere above the figure that uses
        it as a legend."""
        html = (built_site / "players" / "1" / "index.html").read_text()
        header = html.split('<article class="player-verdict">', 1)[1].split(
            'class="player-hero ', 1
        )[0]
        assert "<svg" not in header
        assert "cl-scale-edge" not in header

    def test_the_open_chevron_still_means_a_clipped_interval(self) -> None:
        css = _strip_css_comments(STYLE_CSS.read_text())
        assert ".interval-clip-caret" in css


class TestReducedMotion:
    def test_non_essential_motion_is_disabled_under_reduced_motion(self) -> None:
        css = _strip_css_comments(STYLE_CSS.read_text())
        assert "@media (prefers-reduced-motion: reduce)" in css
        block = css.split("@media (prefers-reduced-motion: reduce) {", 1)[1]
        assert "transition-duration: 0.01ms !important" in block


class TestShellDoesNotCompute:
    """`CLAUDE.md` rule 6 -- the dashboard displays, it never computes."""

    def test_shell_template_derives_no_score_rank_or_interval(self) -> None:
        html = BASE_HTML.read_text()
        for banned in ("score", "rank", "interval", "probability"):
            assert f"{{{{ {banned}" not in html, banned
