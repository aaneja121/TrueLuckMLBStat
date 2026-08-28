"""Redesign Phase 5 -- the Explore route.

Reuses `tests/test_dashboard_explore_build.py`'s fixture seeding (a synthetic,
sharded Explorer artifact set, never the real committed one) and adds the
guarantees Phase 5 introduces:

  * one combobox implementation, shared with the header;
  * the URL as the single source of truth for the selection, in and out;
  * one `run_value` domain, computed at build time over every published
    play, with zero on the shared `--cl-zero` and its own visible ticks;
  * a results table whose axis, all-plays strip and per-row marks are the
    same table column, so they cannot disagree at any width;
  * the preserved behaviours: sharded fetch, the empty state that hides the
    table AND the count, the one-way reveal, fail-closed when the Explorer
    is disabled.

There is no JS runtime here, so client behaviour is verified structurally
against the actual shipped `dist/static/explore.js` -- the same pattern this
suite already uses for `play.js`. The interaction itself (typing, arrowing,
Enter, Back/Forward, filtering) was exercised in a real browser; see
`outputs/figures/phase5_explore_2026-08-28/README.md`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import build as dashboard_build
import pytest

from test_dashboard_explore_build import (
    _build,
    _index_row,
    _seed_explore_fixture,
    _seed_snapshot,
    _showcase_row,
)

APP_JS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "app.js"
EXPLORE_JS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "explore.js"
CSS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "style.css"


def _flat(html: str) -> str:
    """Template source wraps at ~100 columns and Jinja escapes `'`, so a
    naive substring check on rendered prose is unreliable."""
    return re.sub(r"\s+", " ", html.replace("&#39;", "'").replace("&amp;", "&"))


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One build shared by every read-only assertion in this module.

    Nine plays across two hitters, spanning both signs and both a
    measured and an unmeasured batted ball, so the scale has a real domain
    and the missing-measurement path is exercised.
    """
    tmp_path = tmp_path_factory.mktemp("explore")
    out_root, art_root = _seed_snapshot(tmp_path)
    rows = [
        _index_row(play_id="700001-1-1", contact_luck_runs=1.40, outcome_class="home_run"),
        _index_row(play_id="700001-2-1", contact_luck_runs=0.62, outcome_class="double"),
        _index_row(play_id="700001-3-1", contact_luck_runs=0.05, outcome_class="single"),
        _index_row(play_id="700001-4-1", contact_luck_runs=-0.31, outcome_class="out"),
        _index_row(
            play_id="700001-5-1",
            contact_luck_runs=-1.10,
            outcome_class="out",
            launch_speed=None,
            launch_angle=None,
        ),
        _index_row(
            play_id="700001-6-1",
            batter_id=54321,
            batter_name="Other Hitter",
            contact_luck_runs=0.90,
            outcome_class="triple",
        ),
        _index_row(
            play_id="700001-7-1",
            batter_id=54321,
            batter_name="Other Hitter",
            contact_luck_runs=-0.44,
            outcome_class="out",
        ),
    ]
    showcase = [
        _showcase_row(
            play_id="700001-1-1",
            rank=1,
            group="favorable",
            expected_run_value=0.17,
            observed_run_value=1.57,
            contact_luck_runs=1.40,
        ),
        _showcase_row(
            play_id="700001-5-1",
            rank=1,
            group="unfavorable",
            outcome_class="out",
            launch_speed=None,
            launch_angle=None,
            expected_run_value=0.17,
            observed_run_value=-0.93,
            contact_luck_runs=-1.10,
        ),
    ]
    players_path, players_dir, games_dir, metadata_path, showcase_path = _seed_explore_fixture(
        tmp_path, index_rows=rows, showcase_rows=showcase
    )
    _build(
        tmp_path,
        out_root,
        art_root,
        players_path,
        players_dir,
        games_dir,
        metadata_path,
        showcase_path,
    )
    return tmp_path / "dist"


def _explore(site: Path) -> str:
    return (site / "explore" / "index.html").read_text()


def _scale(site: Path) -> dict[str, float]:
    html = _explore(site)
    blob = html.split('id="explore-run-value-scale">', 1)[1].split("</script>", 1)[0]
    return json.loads(blob)


class TestSharedCombobox:
    """The route used to ship a second, weaker picker. There is now one
    implementation and both surfaces drive it."""

    def test_the_combobox_factory_is_defined_once_and_exported(self) -> None:
        app = APP_JS.read_text()
        assert app.count("function createCombobox(") == 1
        assert "window.ContactLuck.createCombobox = createCombobox" in app

    def test_explore_drives_the_shared_factory_and_defines_no_picker_of_its_own(self) -> None:
        js = EXPLORE_JS.read_text()
        assert "window.ContactLuck.createCombobox" in js
        assert "function createCombobox(" not in js
        # The legacy dropdown of <button>s is gone, not merely restyled.
        assert "search-results-dropdown" not in js
        assert "search-result-item" not in js

    def test_the_legacy_dropdown_styles_are_gone_too(self) -> None:
        css = CSS.read_text()
        rules = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
        assert ".search-results-dropdown" not in rules
        assert ".search-result-item" not in rules
        assert ".explore-player-search-wrap" not in rules
        assert ".explore-change-player" not in rules

    def test_the_picker_carries_the_full_combobox_contract(self, site: Path) -> None:
        html = _explore(site)
        picker = html.split('class="explore-picker"', 1)[1].split("</div>\n\n", 1)[0]
        assert 'role="combobox"' in picker
        assert 'aria-expanded="false"' in picker
        assert 'aria-controls="explore-player-search-listbox"' in picker
        assert 'aria-autocomplete="list"' in picker
        assert 'role="listbox"' in picker
        # A real, associated label -- never a placeholder standing in for one.
        assert '<label class="explore-picker-label" for="explore-player-search"' in html
        assert 'role="status"' in picker

    def test_the_factory_implements_the_keyboard_contract(self) -> None:
        app = APP_JS.read_text()
        factory = app.split("function createCombobox(", 1)[1].split("\n  function initGlobal", 1)[0]
        for key in ('"ArrowDown"', '"ArrowUp"', '"Home"', '"End"', '"Enter"', '"Escape"', '"Tab"'):
            assert key in factory, key
        assert 'setAttribute("aria-activedescendant"' in factory
        assert 'removeAttribute("aria-activedescendant")' in factory
        # Outside click closes.
        assert 'document.addEventListener("click"' in factory

    def test_an_unmatched_query_is_announced_and_is_not_selectable(self) -> None:
        app = APP_JS.read_text()
        factory = app.split("function createCombobox(", 1)[1].split("\n  function initGlobal", 1)[0]
        empty = factory.split("if (!matches.length)", 1)[1].split("return;", 1)[0]
        assert 'setAttribute("role", "presentation")' in empty
        assert "announce(" in empty

    def test_unqualified_hitters_are_selectable_where_the_catalog_carries_them(
        self, site: Path
    ) -> None:
        """The published Explorer catalog has no qualification concept at
        all -- every hitter with a scored play is in it, and the picker
        offers all of them."""
        catalog = json.loads((site / "explore" / "players.json").read_text())
        assert len(catalog) == 2
        js = EXPLORE_JS.read_text()
        assert "combobox.setItems(players)" in js


class TestUrlState:
    def test_the_url_is_read_on_load_and_on_history_navigation(self) -> None:
        js = EXPLORE_JS.read_text()
        assert 'get("batter")' in js
        assert "function applyUrl()" in js
        assert 'window.addEventListener("popstate", applyUrl)' in js
        # Applied once the catalog is in hand, so a deep link resolves.
        assert "applyUrl();" in js.split("function initExplorePage", 1)[1]

    def test_selecting_writes_the_selection_back_to_the_url(self) -> None:
        js = EXPLORE_JS.read_text()
        push = js.split("function pushSelection(", 1)[1].split("\n    }", 1)[0]
        assert 'url.searchParams.set("batter"' in push
        assert "window.history.pushState" in push
        # Back/Forward stay honest because the address, not the click, is
        # what actually changes the selection.
        assert "applyUrl();" in push

    def test_the_url_is_the_authority_and_the_click_only_changes_it(self) -> None:
        """The plan's engineering guard: derive the in-memory selection from
        the URL, never the reverse."""
        js = EXPLORE_JS.read_text()
        assert "onSelect: pushSelection" in js
        # showSelection is reached through applyUrl, not directly from the
        # combobox callback.
        combobox_block = js.split("combobox = factory({", 1)[1].split("});", 1)[0]
        assert "showSelection" not in combobox_block

    def test_a_repeat_selection_adds_no_history_entry(self) -> None:
        js = EXPLORE_JS.read_text()
        push = js.split("function pushSelection(", 1)[1].split("\n    }", 1)[0]
        assert "if (url.href !== window.location.href)" in push

    def test_an_unknown_or_malformed_batter_clears_rather_than_breaking(self) -> None:
        js = EXPLORE_JS.read_text()
        apply_block = js.split("function applyUrl()", 1)[1].split("\n    }", 1)[0]
        assert "clearSelection()" in apply_block
        # No fetch is attempted for a hitter who is not in the catalog.
        assert "fetch(" not in apply_block

    def test_the_player_page_link_and_this_route_agree_on_the_parameter(self, site: Path) -> None:
        player = (site / "players" / "1" / "index.html").read_text()
        if "/explore/?batter=" in player:
            assert "/explore/?batter=1" in player
        assert 'get("batter")' in EXPLORE_JS.read_text()


class TestRunValueScale:
    def test_the_domain_is_computed_once_at_build_time_and_shipped_as_data(
        self, site: Path
    ) -> None:
        scale = _scale(site)
        assert scale["domain_min"] < 0 < scale["domain_max"]
        assert scale["unit_label"] == "Contact Luck on the play, runs"

    def test_the_domain_covers_every_published_play(self, site: Path) -> None:
        rows = []
        for path in (site / "explore" / "players").glob("*.json"):
            rows.extend(json.loads(path.read_text()))
        values = [r["contact_luck_runs"] for r in rows]
        scale = _scale(site)
        assert scale["domain_min"] <= min(values)
        assert scale["domain_max"] >= max(values)

    def test_zero_lands_on_the_shared_zero_locus(self, site: Path) -> None:
        """Invariant Z. The run-value scale is a DIFFERENT domain from
        `league_per_100`, but it renders zero at the same fraction of its
        field, which is what lets the two live on one site."""
        html = _explore(site)
        cl_zero = float(re.search(r"--cl-zero: ([\d.]+)%", html).group(1)) / 100
        scale = _scale(site)
        span = scale["domain_max"] - scale["domain_min"]
        assert (-scale["domain_min"]) / span == pytest.approx(cl_zero, abs=1e-6)

    def test_the_scale_declares_its_own_domain_in_ticks_and_in_words(self, site: Path) -> None:
        """The anti-fake-alignment guard: a figure that borrows the shared
        zero while hiding its own domain is the deception the guard exists
        to prevent."""
        html = _explore(site)
        ticks = re.findall(r'class="cl-axis-tick[^"]*"[^>]*>([^<]+)<', html)
        assert len(ticks) >= 3
        assert any("+0" in t for t in ticks)
        assert "Contact Luck on the play, runs" in html
        flat = _flat(html)
        assert "published plays. Every hitter is drawn on it." in flat

    def test_the_browser_never_derives_a_domain_of_its_own(self) -> None:
        """A per-hitter autoscale would put two hitters' plays on two
        different scales and move zero off the spine."""
        js = EXPLORE_JS.read_text()
        assert 'document.getElementById("explore-run-value-scale")' in js
        pct = js.split("function scalePct(", 1)[1].split("\n  }", 1)[0]
        assert "scale.domain_max - scale.domain_min" in pct
        # No min/max over the loaded rows anywhere near the scale code.
        assert "Math.min.apply" not in pct
        assert "Math.max.apply" not in pct

    def test_a_value_outside_the_domain_clips_rather_than_widening_it(self) -> None:
        js = EXPLORE_JS.read_text()
        pct = js.split("function scalePct(", 1)[1].split("\n  }", 1)[0]
        assert "if (fraction < 0) fraction = 0;" in pct
        assert "if (fraction > 1) fraction = 1;" in pct

    def test_the_scale_is_not_forced_onto_quantities_that_are_not_run_values(self) -> None:
        """Exit velocity, launch angle, dates and categorical results get no
        zero-centred treatment -- they are not zero-centred quantities."""
        js = EXPLORE_JS.read_text()
        for fn in ("evLaText", "outcomeLabel"):
            body = js.split("function " + fn + "(", 1)[1].split("\n  }", 1)[0]
            assert "scalePct" not in body
            assert "cl-scale" not in body

    def test_no_new_fourth_zero_centred_scale_is_introduced(self, site: Path) -> None:
        html = _explore(site)
        assert html.count('id="explore-run-value-scale"') == 1


class TestResultsArchitecture:
    def test_the_axis_the_strip_and_the_rows_are_one_table_column(self, site: Path) -> None:
        """Registration by construction. An axis in a sibling element would
        be a second coordinate space and a second zero x on one page."""
        html = _explore(site)
        thead = html.split("<thead>", 1)[1].split("</thead>", 1)[0]
        assert 'class="cl-axis-row"' in thead
        assert 'class="explore-distribution-row"' in thead
        assert thead.count('class="col-play-verdict"') == 3
        assert "cl-axis-ticks" in thead

    def test_the_verdict_is_one_cell(self) -> None:
        """docs/design/tables.md rule 1: the mark and the numeral are one
        statement, and the field is the FIRST track so no row's field can
        start further right than another's."""
        js = EXPLORE_JS.read_text()
        build_row = js.split("function buildRow(", 1)[1].split("\n    }", 1)[0]
        verdict = build_row.split("verdictTd", 1)[1]
        assert verdict.index("explore-play-field") < verdict.index("explore-play-value")
        css = CSS.read_text()
        block = css.split(".explore-plays tbody .col-play-verdict {", 1)[1].split("}", 1)[0]
        assert "grid-template-columns: minmax(0, 1fr)" in block

    def test_every_column_comes_from_the_existing_artifact_contract(self, site: Path) -> None:
        html = _explore(site)
        heads = re.findall(r'<th scope="col"[^>]*>(.*?)</th>', html, flags=re.S)
        text = " ".join(re.sub(r"<[^>]+>", " ", h) for h in heads)
        for label in ("Play", "Recorded result", "Contact", "Expected RV", "Contact Luck"):
            assert label in text, label
        # No team, no position, no field this artifact set does not carry.
        for absent in ("Team", "Position", "Pitcher", "Inning"):
            assert absent not in text

    def test_each_row_offers_a_path_to_its_play(self) -> None:
        js = EXPLORE_JS.read_text()
        assert 'return "/plays/?id=" + encodeURIComponent(playId);' in js
        build_row = js.split("function buildRow(", 1)[1].split("\n    }", 1)[0]
        assert "playUrl(row.play_id)" in build_row

    def test_sign_is_carried_by_the_glyph_as_well_as_the_colour(self) -> None:
        js = EXPLORE_JS.read_text()
        signed = js.split("function formatSigned(", 1)[1].split("\n  }", 1)[0]
        assert '(value >= 0 ? "+" : "")' in signed
        assert 'replace("-", MINUS)' in signed

    def test_the_results_table_needs_no_horizontal_scroll_container(self, site: Path) -> None:
        """The baseline wrapped a seven-column nowrap table in a bare
        `.overflow-x` with no min-width contract (guardrails 6). The row
        transforms at mobile instead."""
        html = _explore(site)
        results = html.split('class="explore-results"', 1)[1]
        assert "overflow-x" not in results
        css = CSS.read_text()
        assert "table.explore-results-table" not in css


class TestFiltersAndSort:
    def test_every_control_has_a_real_visible_label(self, site: Path) -> None:
        html = _explore(site)
        controls = html.split('class="explore-controls"', 1)[1].split("</div>\n\n", 1)[0]
        for control_id in ("explore-outcome-filter", "explore-luck-filter", "explore-sort"):
            assert f'<label class="explore-control-label" for="{control_id}"' in controls
            assert f'id="{control_id}"' in controls

    def test_the_preserved_filters_and_sorts_all_survive(self, site: Path) -> None:
        html = _explore(site)
        for value in ("home_run", "triple", "double", "single", "out"):
            assert f'value="{value}"' in html
        for value in ("all", "favorable", "unfavorable"):
            assert f'value="{value}"' in html
        js = EXPLORE_JS.read_text()
        for key in ("most_favorable", "most_unfavorable", "newest", "hardest_hit"):
            assert key in js

    def test_a_reset_exists_and_appears_only_when_a_filter_is_set(self) -> None:
        js = EXPLORE_JS.read_text()
        assert "function filtersActive()" in js
        assert "resetBtn.hidden = !filtersActive();" in js

    def test_filtering_narrows_the_strip_without_removing_the_excluded_plays(self) -> None:
        """The point of the strip is to show what the filter is taking out,
        so an excluded play is recessed, never deleted."""
        js = EXPLORE_JS.read_text()
        render = js.split("function renderDistribution(", 1)[1].split("\n    }", 1)[0]
        assert "selectedRows.forEach" in render
        assert "is-filtered-out" in render
        css = CSS.read_text()
        block = css.split(".explore-distribution-mark.is-filtered-out {", 1)[1].split("}", 1)[0]
        assert "display: none" not in block
        assert "visibility" not in block

    def test_the_result_count_is_announced(self, site: Path) -> None:
        html = _explore(site)
        assert 'role="status" data-role="explore-results-status"' in html
        js = EXPLORE_JS.read_text()
        assert "resultsStatus.textContent" in js


class TestEmptyAndErrorStates:
    def test_the_zero_result_state_hides_the_table_and_the_count_together(self) -> None:
        """Preserved invariant (docs/design/guardrails.md)."""
        js = EXPLORE_JS.read_text()
        empty = js.split("if (!filtered.length) {", 1)[1].split("return;", 1)[0]
        assert "resultsEl.hidden = true;" in empty
        assert "countEl.hidden = true;" in empty
        assert "emptyEl.hidden = false;" in empty

    def test_the_zero_result_state_keeps_the_hitter_and_the_controls(self, site: Path) -> None:
        js = EXPLORE_JS.read_text()
        empty = js.split("if (!filtered.length) {", 1)[1].split("return;", 1)[0]
        for erased in ("selectedRoot.hidden", "clearSelection", "selectedRows = []"):
            assert erased not in empty
        assert "Clear them to see all of this hitter's published plays." in _flat(_explore(site))

    def test_a_hitter_with_no_published_plays_is_described_precisely(self, site: Path) -> None:
        """Never "no plays" or "no Contact Luck" -- only what the currently
        published Explorer data supports. Same wording as the player page."""
        js = EXPLORE_JS.read_text()
        assert "aren't currently available in Play Explorer." in js
        for overclaim in ("has no plays", "no batted balls", "no Contact Luck"):
            assert overclaim not in js

    def test_the_no_selection_state_is_short(self, site: Path) -> None:
        html = _explore(site)
        prompt = html.split('data-role="explore-prompt"', 1)[1].split("</p>", 1)[0]
        words = len(re.sub(r"<[^>]+>", " ", prompt).split())
        assert words <= 15, prompt

    def test_a_failed_fetch_says_what_happened_and_leaves_the_page_usable(self) -> None:
        js = EXPLORE_JS.read_text()
        assert "The rest of the site still works." in js
        assert "The rest of the page still works." in js
        assert "Try again in a moment." in js


class TestPreservedBehaviour:
    def test_the_sharded_fetch_contract_is_intact(self, site: Path) -> None:
        """`players.json` up front, then ONE `players/<id>.json`. A
        monolithic index measured ~29.3 MiB at full-season scale, over
        Cloudflare Pages' 25 MiB per-asset limit."""
        js = EXPLORE_JS.read_text()
        assert 'var PLAYERS_URL = "players.json";' in js
        select_fn = js.split("function selectPlayer(", 1)[1].split("\n    }", 1)[0]
        assert 'fetch("players/"' in select_fn
        assert js.count('fetch("players/"') == 1
        assert "search-index.json" not in js
        assert not (site / "explore" / "search-index.json").exists()

    def test_no_per_player_file_is_fetched_at_page_load(self) -> None:
        js = EXPLORE_JS.read_text()
        load = js.split("fetch(PLAYERS_URL)", 1)[1]
        assert 'fetch("players/"' not in load

    def test_the_show_all_reveal_stays_one_way(self) -> None:
        js = EXPLORE_JS.read_text()
        reveal = js.split("if (rest.length && revealBtn) {", 1)[1].split("\n    }", 1)[0]
        assert 'revealBtn.textContent = "Show all " + sorted.length;' in reveal
        assert "revealBtn.hidden = true;" in reveal
        # Nothing already on screen is re-rendered or re-ordered.
        assert "innerHTML" not in reveal
        assert "sort(" not in reveal

    def test_the_reveal_moves_focus_to_the_newly_revealed_set(self) -> None:
        """The trigger vanishes, so without this focus falls to <body>."""
        js = EXPLORE_JS.read_text()
        reveal = js.split("if (rest.length && revealBtn) {", 1)[1].split("\n    }", 1)[0]
        assert "target.focus()" in reveal

    def test_the_explorer_stays_fail_closed_when_disabled(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        assert not (tmp_path / "dist" / "explore").exists()
        home = (tmp_path / "dist" / "index.html").read_text()
        assert "/explore/" not in home

    def test_nothing_on_this_route_is_recomputed_in_the_browser(self) -> None:
        """CLAUDE.md rule 6. Positions are layout percentages against a
        build-supplied domain; every displayed value is copied from the
        fetched row."""
        js = EXPLORE_JS.read_text()
        for forbidden in ("predict", "Math.exp", "Math.log", "* 100 /", "/ bbe"):
            assert forbidden not in js
        build_row = js.split("function buildRow(", 1)[1].split("\n    }", 1)[0]
        assert "row.contact_luck_runs" in build_row
        assert "row.expected_run_value" in build_row
        # No arithmetic on a displayed scoring value.
        assert "contact_luck_runs -" not in build_row
        assert "contact_luck_runs +" not in build_row


class TestPageStructureAndAccessibility:
    def test_one_h1_and_no_skipped_heading_levels(self, site: Path) -> None:
        html = _explore(site)
        levels = [int(m) for m in re.findall(r"<h([1-6])[ >]", html)]
        assert levels.count(1) == 1
        assert set(levels) <= {1, 2, 3}

    def test_no_duplicate_ids(self, site: Path) -> None:
        ids = re.findall(r'\sid="([^"]+)"', _explore(site))
        assert len(ids) == len(set(ids)), sorted(i for i in ids if ids.count(i) > 1)

    def test_every_aria_reference_resolves(self, site: Path) -> None:
        html = _explore(site)
        ids = set(re.findall(r'\sid="([^"]+)"', html))
        for attr in ("aria-controls", "aria-labelledby", "aria-describedby"):
            for value in re.findall(rf'{attr}="([^"]+)"', html):
                for ref in value.split():
                    assert ref in ids, f"{attr}={ref}"

    def test_the_strip_is_an_image_with_a_text_equivalent(self, site: Path) -> None:
        html = _explore(site)
        assert 'class="cl-scale-field explore-distribution-field"' in html
        assert 'role="img"' in html
        js = EXPLORE_JS.read_text()
        label = js.split('distributionField.setAttribute(\n        "aria-label",', 1)[1]
        assert "runs of Contact Luck" in label
        assert "listed in the rows below" in label

    def test_the_axis_rail_is_presentational(self, site: Path) -> None:
        """Every value it labels is printed as text in the row it scales."""
        html = _explore(site)
        axis_row = html.split('class="cl-axis-row"', 1)[1].split("</tr>", 1)[0]
        assert axis_row.lstrip().startswith('aria-hidden="true"'), axis_row[:60]
        assert "cl-axis-tick" in axis_row

    def test_marks_are_decoration_and_the_statement_is_text(self) -> None:
        js = EXPLORE_JS.read_text()
        build_row = js.split("function buildRow(", 1)[1].split("\n    }", 1)[0]
        assert 'field.setAttribute("aria-hidden", "true")' in build_row
        assert "value.textContent = formatSigned(row.contact_luck_runs)" in build_row

    def test_no_images_and_no_svg_text(self, site: Path) -> None:
        html = _explore(site)
        assert "<img" not in html
        assert "<svg" not in html
        assert "<text" not in html

    def test_controls_meet_the_minimum_target_size(self) -> None:
        css = CSS.read_text()
        for selector in (
            ".explore-picker-input {",
            ".explore-control-select {",
            ".explore-reset {",
            ".explore-showcase-reveal {",
            ".explore-picker-option {",
        ):
            block = css.split(selector, 1)[1].split("}", 1)[0]
            assert "var(--target-min)" in block, selector


class TestShowcase:
    def test_the_showcase_is_a_ruled_list_not_a_card_grid(self, site: Path) -> None:
        """Guardrails 2, 9 and 13: the sign colours encode the sign, they do
        not decorate a card border."""
        html = _explore(site)
        assert 'class="explore-showcase-list"' in html
        css = CSS.read_text()
        assert ".showcase-card" not in css
        assert ".showcase-cards" not in css
        assert "border-top-color: var(--diverging-positive)" not in css
        item = css.split(".explore-showcase-item {", 1)[1].split("}", 1)[0]
        assert "border:" not in item
        assert "border-radius" not in item
        assert "background" not in item

    def test_the_showcase_draws_on_the_same_scale_as_the_results(self, site: Path) -> None:
        html = _explore(site)
        showcase = html.split('class="explore-showcase"', 1)[1]
        assert showcase.count("explore-showcase-axis") == 2
        assert "cl-axis-tick" in showcase
        css = CSS.read_text()
        block = css.split(".explore-showcase-axis,\n.explore-showcase-item {", 1)[1].split("}", 1)[
            0
        ]
        # Same leading tracks as the table, so both put zero at one x.
        assert "32rem" in block

    def test_each_showcase_play_states_why_it_is_here_and_links_out(self) -> None:
        js = EXPLORE_JS.read_text()
        item = js.split("function buildShowcaseItem(", 1)[1].split("\n  }", 1)[0]
        assert "expected " in item
        assert "actual " in item
        assert "playUrl(row.play_id)" in item
        assert "formatSigned(row.contact_luck_runs)" in item

    def test_the_showcase_follows_the_hitter_flow(self, site: Path) -> None:
        """Phase 5 ordering: picker, then results, then the editorial set."""
        html = _explore(site)
        assert html.index('data-role="explore-player-search"') < html.index(
            'data-role="explore-selected"'
        )
        assert html.index('data-role="explore-selected"') < html.index(
            'data-role="showcase-section"'
        )


class TestCopy:
    def test_no_em_dashes_in_the_rendered_page(self, site: Path) -> None:
        """docs/design/typography.md § Product voice."""
        html = _explore(site)
        body = re.sub(r"<head.*?</head>", "", html, flags=re.S)
        body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
        assert "—" not in body

    def test_no_banned_phrases(self, site: Path) -> None:
        html = _explore(site).lower()
        for phrase in (
            "deserved hits",
            "true talent luck",
            "guaranteed regression",
            "should have produced",
            "defense-independent",
            "statistically significant player",
        ):
            assert phrase not in html

    def test_the_retrospective_framing_reaches_this_route(self, site: Path) -> None:
        html = _flat(_explore(site))
        assert "retrospective description of realized outcomes" in html
        assert "not a projection of future performance" in html

    def test_the_provenance_note_is_kept_and_is_one_short_statement(self, site: Path) -> None:
        html = _flat(_explore(site))
        assert "observed_run_value" in html
        assert "expected_run_value" in html
        assert "/status/#play-data-source" in _explore(site)


class TestFixtureProductionBoundary:
    def test_the_committed_fixture_is_never_the_default_source(self, tmp_path: Path) -> None:
        """A bare `build.py` ships the Explorer disabled. Nothing here can
        pick up `dashboard/explore_fixture/` by accident."""
        out_root, art_root = _seed_snapshot(tmp_path)
        result = dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        assert result.out_dir.exists()
        assert not (tmp_path / "dist" / "explore").exists()

    def test_the_scale_is_read_off_the_given_artifacts_not_hardcoded(self, tmp_path: Path) -> None:
        """Two different artifact sets must produce two different domains,
        which is what proves the domain comes from the data in hand rather
        than from a constant that would be wrong on production."""
        out_root, art_root = _seed_snapshot(tmp_path)
        narrow = _seed_explore_fixture(
            tmp_path / "a", index_rows=[_index_row(contact_luck_runs=0.20)]
        )
        wide = _seed_explore_fixture(
            tmp_path / "b", index_rows=[_index_row(contact_luck_runs=3.10)]
        )
        domains = []
        for i, paths in enumerate((narrow, wide)):
            _build(tmp_path, out_root, art_root, *paths, out_dir=tmp_path / f"dist{i}")
            html = (tmp_path / f"dist{i}" / "explore" / "index.html").read_text()
            blob = html.split('id="explore-run-value-scale">', 1)[1].split("</script>", 1)[0]
            domains.append(json.loads(blob)["domain_max"])
        assert domains[0] < domains[1]
