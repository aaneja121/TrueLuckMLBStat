"""Redesign Phase 3 -- the Zero Spine leaderboard.

What is asserted here is the *contract* the leaderboard now rests on: one
shared percentage basis, one verdict cell, real tab and sort semantics,
labelled filters, and the preserved ranking invariants. Styling is not
asserted; Phase 8 may reskin any of this without touching a test.

Rendered geometry -- spine registration in real pixels, rows above the fold,
the mobile transformation -- is verified with the globally configured
Playwright MCP, because this repository has no browser-automation test
dependency (`CLAUDE.md` § Verification before claiming done). The build-time
half of registration IS assertable, and is: every row's percentages come from
one `ZeroScale`, and there is exactly one zero locus on the page.
"""

from __future__ import annotations

import re
from pathlib import Path

import build as dashboard_build
import pytest

from dashboard_snapshot_fixtures import default_player_record, write_snapshot

STYLE_CSS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "style.css"
APP_JS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "app.js"

LONGEST_REAL_NAME = "Christian Encarnacion-Strand"


def _players() -> list[dict]:
    return [
        # A strong positive, a strong negative, a near-zero negative whose
        # interval crosses zero, a wide interval, and one unqualified player.
        default_player_record(
            batter_id=1, batter_name="Pete Crow-Armstrong", score=7.62, lower=3.15, upper=11.81
        ),
        default_player_record(
            batter_id=2, batter_name="Salvador Perez", score=-6.41, lower=-9.63, upper=-3.23
        ),
        default_player_record(
            batter_id=3, batter_name="Jake Burger", score=-0.06, lower=-4.10, upper=3.98
        ),
        default_player_record(
            batter_id=4, batter_name=LONGEST_REAL_NAME, score=0.19, lower=-3.30, upper=3.91
        ),
        default_player_record(
            batter_id=5,
            batter_name="Unqualified Ulysses",
            score=12.0,
            lower=-30.0,
            upper=54.0,
            bbe=6,
            qualification_status="small_sample",
        ),
    ]


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("lb")
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


@pytest.fixture(scope="module")
def home(site: Path) -> str:
    return (site / "index.html").read_text()


def _css() -> str:
    return re.sub(r"/\*.*?\*/", "", STYLE_CSS.read_text(), flags=re.DOTALL)


def _tab_names(html: str) -> list[str]:
    """The accessible name of each ranking tab: its text content with markup
    stripped, which is what the name is computed from."""
    buttons = re.findall(r'role="tab"[^>]*>(.*?)</button>', html, flags=re.DOTALL)
    return [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", b)).strip() for b in buttons]


def _fields(html: str, table_id: str) -> list[dict[str, float]]:
    table = html.split(f'id="lb-{table_id}"', 1)[1].split("</table>", 1)[0]
    body = table.split("<tbody>", 1)[1]
    out = []
    for style in re.findall(r'class="cl-scale-field[^"]*"[^>]*style="([^"]+)"', body):
        out.append(
            {
                "lo": float(re.search(r"--cl-lo:\s*([\d.]+)%", style).group(1)),
                "hi": float(re.search(r"--cl-hi:\s*([\d.]+)%", style).group(1)),
                "pt": float(re.search(r"--cl-pt:\s*([\d.]+)%", style).group(1)),
            }
        )
    return out


class TestZeroSpineRegistration:
    def test_the_page_declares_exactly_one_zero_locus(self, home: str) -> None:
        loci = set(re.findall(r"--cl-zero:\s*([\d.]+%)", home))
        assert len(loci) == 1

    def test_no_row_carries_its_own_zero(self, home: str) -> None:
        for table_id in ("favorable", "unfavorable"):
            body = home.split(f'id="lb-{table_id}"', 1)[1].split("</table>", 1)[0]
            assert "--cl-zero" not in body.split("<tbody>", 1)[1]

    def test_every_row_carries_all_three_positions(self, home: str) -> None:
        for table_id in ("favorable", "unfavorable"):
            fields = _fields(home, table_id)
            assert fields
            for f in fields:
                assert 0.0 <= f["lo"] <= 100.0
                assert 0.0 <= f["hi"] <= 100.0
                assert f["lo"] <= f["pt"] <= f["hi"]

    def test_both_rankings_use_the_same_scale(self, home: str) -> None:
        """The same hitter appears in both tables at the same position."""
        fav = home.split('id="lb-favorable"', 1)[1].split("</table>", 1)[0]
        unfav = home.split('id="lb-unfavorable"', 1)[1].split("</table>", 1)[0]

        def field_for(table: str, name: str) -> str:
            start = table.find(f'data-player-name="{name}"')
            assert start != -1
            return re.search(
                r'class="cl-scale-field[^"]*"[^>]*style="([^"]+)"', table[start:]
            ).group(1)

        for name in ("Pete Crow-Armstrong", "Salvador Perez"):
            assert field_for(fav, name) == field_for(unfav, name)

    def test_the_axis_lives_inside_the_table(self, home: str) -> None:
        """Registration by construction, not by arithmetic.

        The axis ticks and the distribution strip must be table rows in the
        verdict column, sharing the rows' grid tracks. A sibling element whose
        offset is calculated from column widths rendered 26px wrong.
        """
        table = home.split('id="lb-favorable"', 1)[1].split("</tbody>", 1)[0]
        head = table.split("<thead>", 1)[1]
        assert 'class="cl-axis-row"' in head
        assert 'class="cl-distribution-row"' in head
        assert "cl-axis-figure" in head

    def test_axis_and_rows_share_one_grid_track(self) -> None:
        """Ticks, distribution strip and every row's field are the SAME track
        of the same grid -- which is what makes them register by construction
        rather than by arithmetic that happens to agree."""
        css = _css()

        def track(selector: str) -> str:
            block = css.split(selector, 1)[1].split("}", 1)[0]
            return re.search(r"grid-column:\s*([^;]+);", block).group(1).strip()

        assert (
            track(".cl-axis-figure {")
            == track(".cl-distribution {")
            == track(".verdict-inner .cl-scale-field {")
            == "1"
        )

    def test_no_geometry_is_derived_by_arithmetic(self) -> None:
        css = _css()
        assert "--cl-field-offset" not in css
        assert "--cl-field-width" not in css


class TestVerdictCell:
    def test_point_estimate_and_interval_share_one_cell(self, home: str) -> None:
        body = home.split('id="lb-favorable"', 1)[1].split("<tbody>", 1)[1]
        cell = body.split('<td class="col-verdict', 1)[1].split("</td>", 1)[0]
        assert "verdict-value" in cell
        assert "cl-scale-field" in cell
        assert "verdict-interval" in cell

    def test_the_interval_is_never_behind_a_toggle(self, home: str) -> None:
        assert "<details" not in home.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
        body = home.split('id="lb-favorable"', 1)[1].split("<tbody>", 1)[1]
        rows = body.count("<tr data-official-rank=")
        assert body.count('class="verdict-interval num">') == rows

    def test_the_interval_survives_the_narrow_layout_in_the_a11y_tree(self) -> None:
        """Hidden visually where the column cannot hold it -- never removed.

        `display: none` would take the uncertainty numbers out of the
        accessibility tree, and intervals shipping alongside every point
        estimate is a product commitment, not a column.
        """
        css = _css()
        narrow = css.split("@media (max-width: 1023px) {", 1)[1].split("\n}", 1)[0]
        block = narrow.split(".verdict-interval {", 1)[1].split("}", 1)[0]
        assert "display: none" not in block
        assert "clip-path" in block

    def test_every_value_carries_an_explicit_sign_and_a_true_minus(self, home: str) -> None:
        body = home.split('id="lb-favorable"', 1)[1].split("<tbody>", 1)[1].split("</tbody>", 1)[0]
        values = re.findall(r'class="verdict-value num">([^<]+)<', body)
        assert values
        for v in values:
            assert v[0] in "+−", v
            assert "-" not in v  # ASCII hyphen never reaches a displayed value

    def test_sign_follows_the_point_estimate_not_the_interval(self, home: str) -> None:
        """Jake Burger is -0.06 with an interval spanning zero: unfavorable
        colouring, no de-emphasis, identical treatment to any other row."""
        body = home.split('id="lb-favorable"', 1)[1].split("<tbody>", 1)[1]
        start = body.find('data-player-name="Jake Burger"')
        row = body[start : start + 2000]
        assert "is-unfavorable" in row
        assert "cl-scale-unfavorable" in row

    def test_no_treatment_weakens_an_interval_that_crosses_zero(self) -> None:
        """Guardrails anti-pattern 17. The mark that draws an interval, and
        the mark that draws its point estimate, carry no fade, dash, italic
        or filter -- and nothing keys off whether the interval spans zero."""
        css = _css()
        for selector in (".cl-scale-interval {", ".cl-scale-point {", ".verdict-value {"):
            block = css.split(selector, 1)[1].split("}", 1)[0]
            for banned in ("opacity", "font-style", "dashed", "filter:"):
                assert banned not in block, (selector, banned)
        # No rule anywhere selects on "the interval crosses zero".
        assert "crosses-zero" not in css
        assert "data-crosses" not in css


class TestValuePlacement:
    """Gate G3 is decided: V1 ships, V2 is gone.

    V1 is one fixed-width numeral box after the plot field, right-aligned at
    a constant x. V2 -- sign-aware margins on either side of the field -- was
    rejected because a mixed-sign sort scatters the numerals across two
    columns. See `docs/design/tables.md` § Value placement.
    """

    def test_the_rejected_variant_left_nothing_behind(self) -> None:
        css = _css()
        js = APP_JS.read_text()
        for leftover in ("value-placement", "--lb-lead", "data-sign"):
            assert leftover not in css, leftover
            assert leftover not in js, leftover

    def test_the_numeral_is_one_fixed_width_box_after_the_field(self) -> None:
        css = _css()
        value = css.split("\n.verdict-value {", 1)[1].split("}", 1)[0]
        assert "grid-column: 2" in value
        assert "text-align: right" in value

    def test_the_field_is_the_first_track_so_nothing_can_shift_it(self) -> None:
        """The failure this guards is subtle and fatal: anything reserved to
        the LEFT of the plot field moves that row's field off the shared
        percentage basis, and the spine registers against nothing.
        """
        css = _css()
        # The grid declaration, not one of the later per-breakpoint overrides.
        inner = re.search(r"\n\.verdict-inner \{([^}]*grid-template-columns[^}]*)\}", css).group(1)
        tracks = inner.split("grid-template-columns:", 1)[1].split(";", 1)[0].split()
        assert tracks[0] == "minmax(0,"
        assert (
            "grid-column: 1" in css.split(".verdict-inner .cl-scale-field {", 1)[1].split("}", 1)[0]
        )

    def test_no_value_sits_at_an_interval_endpoint(self) -> None:
        css = _css()
        value = css.split(".verdict-value {", 1)[1].split("}", 1)[0]
        assert "--cl-pt" not in value
        assert "--cl-hi" not in value
        assert "--cl-lo" not in value


class TestRankingTabs:
    def test_the_two_rankings_are_real_tabs(self, home: str) -> None:
        assert 'role="tablist"' in home
        assert home.count('role="tab"') == 2
        assert home.count('role="tabpanel"') == 2
        assert 'aria-selected="true"' in home
        assert 'aria-selected="false"' in home

    def test_no_display_none_radio_pattern_survives(self, home: str) -> None:
        assert "tab-radio" not in home
        assert 'type="radio"' not in home
        assert ".tab-radio" not in _css()

    def test_both_rankings_begin_at_rank_one(self, home: str) -> None:
        for table_id in ("favorable", "unfavorable"):
            body = home.split(f'id="lb-{table_id}"', 1)[1].split("<tbody>", 1)[1]
            first = re.search(r'<tr data-official-rank="(\d+)"', body).group(1)
            assert first == "1"

    def test_neither_ranking_is_called_worst(self, home: str) -> None:
        """The frozen `public_labels` string is the tab's ACCESSIBLE NAME at
        every width. Mobile shortens what is seen (`.ranking-tab-lead`) and
        clips the rest, so the name is still assembled from the whole
        string -- assert on the assembled name, not on raw markup."""
        assert "worst" not in home.lower()
        assert _tab_names(home) == [
            "Most favorable realized luck",
            "Least favorable outcomes relative to expectation",
        ]

    def test_the_shortened_mobile_label_is_a_leading_substring(self, home: str) -> None:
        """WCAG 2.5.3: what a speech user says has to be what they see. The
        visible mobile label is the FRONT of the accessible name, never a
        paraphrase of it."""
        leads = re.findall(r'class="ranking-tab-lead">([^<]+)<', home)
        assert leads == ["Most favorable", "Least favorable"]
        for lead, name in zip(leads, _tab_names(home), strict=True):
            assert name.startswith(lead)

    def test_the_clipped_label_tail_is_never_display_none(self) -> None:
        """`display: none` would take the tail out of the accessibility tree
        and shorten the ranking's name to "Most favorable" -- a different
        claim. It is clipped instead, the same technique `.visually-hidden`
        uses (guardrails anti-pattern 7)."""
        mobile = _css().split("@media (max-width: 767px) {", 1)[1]
        block = mobile.split(".ranking-tab-rest {", 1)[1].split("}", 1)[0]
        assert "display: none" not in block
        assert "clip-path" in block

    def test_keyboard_navigation_is_implemented(self) -> None:
        js = APP_JS.read_text()
        tabs = js.split("function initRankingTabs()", 1)[1].split(
            "function initLeaderboardSort", 1
        )[0]
        for key in ('"ArrowRight"', '"ArrowLeft"', '"Home"', '"End"', "tabIndex"):
            assert key in tabs, key


class TestSorting:
    def test_sortable_headers_are_buttons_with_aria_sort(self, home: str) -> None:
        head = home.split('id="lb-favorable"', 1)[1].split("</thead>", 1)[0]
        headers = re.findall(r"<th[^>]*data-sort-key[^>]*>", head)
        assert len(headers) == 4
        for th in headers:
            assert "aria-sort=" in th
        assert head.count('class="sort-button"') == 4

    def test_no_click_handler_on_a_bare_header(self) -> None:
        js = APP_JS.read_text()
        assert 'th.addEventListener("click"' not in js
        assert "[data-role='sort']" in js

    def test_rank_restores_official_order_and_never_reverses(self, home: str) -> None:
        """The flagged direction ambiguity: a reversible Rank column produced
        an order that looks like a ranking but is not the official one."""
        head = home.split('id="lb-favorable"', 1)[1].split("</thead>", 1)[0]
        rank_th = re.search(r"<th[^>]*data-sort-key=\"officialRank\"[^>]*>", head).group(0)
        assert 'data-sort-restores="official"' in rank_th
        assert 'aria-sort="ascending"' in rank_th

        js = APP_JS.read_text()
        sort = js.split("function initLeaderboardSort()", 1)[1]
        assert "restores" in sort
        assert 'dir = "asc"' in sort

    def test_sorted_state_has_three_simultaneous_expressions(self, home: str) -> None:
        js = APP_JS.read_text()
        sort = js.split("function initLeaderboardSort()", 1)[1]
        # 1. aria-sort on the header
        assert 'setAttribute("aria-sort"' in sort
        # 2. the visible label, verbatim
        assert "Sorted view. The official Contact Luck ranking is preserved" in js
        # 3. the official accent leaving the rank column
        assert 'setAttribute("data-sorted", "")' in sort
        assert 'removeAttribute("data-sorted")' in sort
        assert "table.leaderboard:not([data-sorted]) td.rank-cell" in _css()

    def test_the_sorted_label_is_absent_on_the_default_view(self, home: str) -> None:
        label = home.split('id="sorted-label-favorable"', 1)[1].split("</p>", 1)[0]
        assert label.strip() == ">" or label.strip().endswith(">")
        assert "Sorted view" not in home.split("<tbody>", 1)[0]

    def test_sort_changes_are_announced(self, home: str) -> None:
        assert 'role="status" id="sort-status-favorable"' in home
        js = APP_JS.read_text()
        assert "status.textContent" in js


class TestFilters:
    def test_each_filter_has_a_real_label(self, home: str) -> None:
        for table_id in ("favorable", "unfavorable"):
            assert f'<label class="leaderboard-filter-label" for="filter-{table_id}">' in home
            field = home.split(f'id="filter-{table_id}"', 1)[1].split(">", 1)[0]
            assert "placeholder=" in field  # a hint, in addition to the label

    def test_filter_ids_are_unique(self, home: str) -> None:
        ids = re.findall(r'id="(filter-[a-z]+)"', home)
        assert sorted(ids) == ["filter-favorable", "filter-unfavorable"]

    def test_filter_results_are_announced_and_have_an_empty_state(self, home: str) -> None:
        assert 'role="status" id="filter-status-favorable"' in home
        assert 'data-role="filter-empty"' in home
        js = APP_JS.read_text()
        assert "hitters match" in js


class TestColumnArchitecture:
    def test_the_derived_column_is_gone(self, home: str) -> None:
        """Interval width is recoverable from the interval already drawn, and
        it was the column silently clipped at every desktop width."""
        assert "Interval width" not in home
        # Still available to sorting, because the datum is not lost.
        assert "data-interval-width=" in home

    def test_identity_and_verdict_never_shed(self) -> None:
        """The tiers themselves are never hidden at any width. (A column's
        secondary sub-label is apparatus, not the column, and may go.)"""
        css = _css()
        for selector in (".col-rank", ".col-player", ".col-verdict", ".col-evidence"):
            pattern = re.escape(selector) + r"(?![\w-])[^{]*\{([^}]*)\}"
            for block in re.findall(pattern, css):
                assert "display: none" not in block, selector

    def test_the_scroll_container_declares_a_min_width_contract(self, home: str) -> None:
        """`.overflow-x` without one is guardrails anti-pattern 6."""
        assert 'class="leaderboard-scroll overflow-x"' in home
        block = _css().split("table.leaderboard {", 1)[1].split("}", 1)[0]
        assert "min-width:" in block

    def test_evidence_compresses_rather_than_disappearing(self, home: str) -> None:
        body = home.split('id="lb-favorable"', 1)[1].split("<tbody>", 1)[1]
        cell = body.split('<td class="col-evidence', 1)[1].split("</td>", 1)[0]
        assert "BBE" in cell
        assert "Scored Games" in cell
        assert "runs" in cell


class TestPreservedRankingInvariants:
    def test_the_official_leaderboard_stays_qualified_only(self, home: str) -> None:
        assert "Unqualified Ulysses" not in home.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
        # ...but they remain findable in the global search index.
        assert "Unqualified Ulysses" in home

    def test_official_rank_is_carried_by_the_row_not_recomputed(self, home: str) -> None:
        js = APP_JS.read_text()
        # The rank travels on the row as data, and the sort key names it in
        # the markup -- the script only ever reads `row.dataset[key]`.
        assert "data-official-rank" in home
        assert 'data-sort-key="officialRank"' in home
        assert "row.dataset[key]" not in js  # it is `a.dataset[key]`/`b.dataset[key]`
        assert "dataset[key]" in js
        # Sorting reorders rows already rendered from the snapshot: it reads
        # `data-*` attributes and never touches a scoring input.
        for banned in (
            "expected_run_value",
            "runs_per_100",
            "lower_95",
            "upper_95",
            "qualification_status",
        ):
            assert banned not in js, banned

    def test_unqualified_players_are_not_de_emphasised_anywhere(self) -> None:
        css = _css()
        assert ".unqualified" not in css
        assert "small-sample" not in css


class TestMobileTransformation:
    def test_the_row_becomes_a_ruled_grid_not_a_squeezed_table(self) -> None:
        css = _css()
        mobile = css.split("@media (max-width: 767px) {", 1)[1]
        block = mobile.split(".leaderboard tbody tr {", 1)[1].split("}", 1)[0]
        assert "display: grid" in block
        assert "grid-template-areas" in block
        assert '"rank name"' in block

    def test_the_name_owns_a_full_line_at_the_specified_size(self) -> None:
        css = _css()
        mobile = css.split("@media (max-width: 767px) {", 1)[1]
        block = mobile.split(".leaderboard .player-link {", 1)[1].split("}", 1)[0]
        assert "font-size: var(--fs-400)" in block  # 16px
        assert "font-weight: 600" in block

    def test_the_desktop_axis_rail_is_hidden_specifically_enough(self) -> None:
        """A bare `.cl-axis-row` loses to `.leaderboard thead tr` and left a
        collapsed tick rail stacked on an empty strip at 390."""
        css = _css()
        mobile = css.split("@media (max-width: 767px) {", 1)[1]
        assert ".leaderboard thead tr.cl-axis-row" in mobile
        assert ".leaderboard thead tr.cl-distribution-row" in mobile

    def test_sorting_stays_operable_when_columns_stop_existing(self) -> None:
        css = _css()
        mobile = css.split("@media (max-width: 767px) {", 1)[1]
        head = mobile.split(".leaderboard th {", 1)[1].split("}", 1)[0]
        assert "display: block" in head
        assert "display: none" not in head


class TestMobileHomepageHierarchy:
    """The correction that moved the first hitter from 574px to 421px at 390.

    Every assertion here is about WHICH CONTENT OWNS A VERTICAL BAND before
    the table. None of it is allowed to be bought with smaller type, a
    shorter row or a smaller target -- the tests below pin that too.
    """

    def test_the_reorder_wrapper_is_inert_above_the_breakpoint(self, home: str) -> None:
        """`display: contents` means the wrapper has no box, so the approved
        desktop hierarchy renders exactly as it did before it existed."""
        assert 'class="home-leaderboard"' in home
        css = _css()
        base = css.split(".home-leaderboard { ", 1)[1].split("}", 1)[0]
        assert "display: contents" in base

    def test_the_scale_caption_moves_below_the_table_at_mobile(self) -> None:
        """Above 768 it captions a visible tick rail and a 124-mark strip.
        At 390 both are hidden, so pre-table it captions nothing on screen."""
        mobile = _css().split("@media (max-width: 767px) {", 1)[1]
        block = mobile.split(".home-leaderboard {", 1)[1].split("}", 1)[0]
        assert "display: flex" in block
        order = {
            name: int(
                re.search(rf"\.home-leaderboard > \.{name} \{{ order: (\d+)", mobile).group(1)
            )
            for name in ("ranking", "leaderboard-note", "cl-axis-head")
        }
        assert order["ranking"] < order["cl-axis-head"]
        assert order["leaderboard-note"] < order["cl-axis-head"]

    def test_the_caption_claim_is_true_in_both_positions(self, home: str) -> None:
        """It sits ABOVE the rows on desktop and BELOW them at mobile, so it
        cannot say "every row below" any more."""
        caption = home.split('class="cl-axis-caption"', 1)[1].split("</p>", 1)[0]
        assert "Every row in this table is drawn on it" in caption
        assert "below" not in caption

    def test_nothing_operable_above_the_first_row_is_under_the_target_floor(
        self,
    ) -> None:
        """`--target-min` is 44px and EVERY control above the first hitter
        now meets it. Three were short: the disclosure (32px), the ranking
        tabs (40px) and the sort buttons (24.5px); the name filter (30px)
        was the last. None of them was bought by shrinking something else.
        """
        mobile = _css().split("@media (max-width: 767px) {", 1)[1]
        for selector in (
            ".lede-detail > summary {",
            ".ranking-tab {",
            ".leaderboard th .sort-button {",
            ".leaderboard-filter-input {",
        ):
            block = mobile.split(selector, 1)[1].split("}", 1)[0]
            assert "min-height: var(--target-min)" in block, selector

    def test_the_filter_target_is_real_height_not_an_overlapping_hit_area(
        self,
    ) -> None:
        """An oversized hit area over a 30px field would have reached into
        the sort buttons 4px below it, and a tap landing on the wrong
        control is worse than a small one. The field itself is 44px."""
        mobile = _css().split("@media (max-width: 767px) {", 1)[1]
        block = mobile.split(".leaderboard-filter-input {", 1)[1].split("}", 1)[0]
        assert "min-height: var(--target-min)" in block
        assert "::before" not in block
        assert "position: absolute" not in block

    def test_the_correction_buys_no_space_from_type_or_rows(self) -> None:
        """The prohibited ways to hit the target: smaller body type, a
        shorter row, a tighter row grammar. None of them are in this block."""
        mobile = _css().split("@media (max-width: 767px) {", 1)[1]
        row = mobile.split(".leaderboard tbody tr {", 1)[1].split("}", 1)[0]
        assert "padding: var(--sp-2) 0" in row  # unchanged row rhythm
        name = mobile.split(".leaderboard .player-link {", 1)[1].split("}", 1)[0]
        assert "font-size: var(--fs-400)" in name  # still 16px
        lede = mobile.split("\n  .lede {", 1)[1].split("}", 1)[0]
        assert "font-size" not in lede

    def test_the_sort_rail_is_one_row_of_real_headers(self, home: str) -> None:
        """The visual compaction is a gap, not a demotion: four `<th
        scope="col">` with `aria-sort` survive it."""
        mobile = _css().split("@media (max-width: 767px) {", 1)[1]
        row = mobile.split(".leaderboard thead tr {", 1)[1].split("}", 1)[0]
        assert "gap: 0 var(--sp-2)" in row
        head = home.split('id="lb-favorable"', 1)[1].split("</thead>", 1)[0]
        assert head.count('scope="col"') == 4
        assert head.count("aria-sort=") == 4


class TestNoTinyAxisLabels:
    def test_the_leaderboard_emits_no_svg(self, home: str) -> None:
        assert "<svg" not in home.split("<main", 1)[1]

    def test_axis_ticks_are_html_text_at_a_real_size(self, home: str) -> None:
        assert 'class="cl-axis-tick' in home
        block = _css().split(".cl-axis-tick {", 1)[1].split("}", 1)[0]
        assert "font-size: var(--fs-200)" in block  # 12px, the declared floor


class TestColourCleanup:
    def test_the_decorative_amber_hero_border_is_gone(self, home: str) -> None:
        assert "page-hero" not in home
        assert ".page-hero" not in _css()

    def test_field_green_is_no_longer_page_furniture(self, home: str) -> None:
        assert "page-hero-mark" not in home
        assert "homepage-proof-card" not in home
        css = _css()
        assert ".page-hero-mark" not in css
        assert ".homepage-proof-card" not in css

    def test_the_worked_examples_kept_their_educational_content(self, home: str) -> None:
        assert "Expected run value" in home
        assert "Contact Luck on this ball" in home
        assert "worked" in home

    def test_amber_marks_the_official_record_and_the_zero_spine(self) -> None:
        css = _css()
        rank = css.split("table.leaderboard:not([data-sorted]) td.rank-cell {", 1)[1].split("}", 1)[
            0
        ]
        assert "var(--accent-amber-text)" in rank
        spine = css.split(".leaderboard .cl-scale-field::before {", 1)[1].split("}", 1)[0]
        assert "var(--accent-amber-mark)" in spine


class TestPlayerPortrait:
    """The leaderboard's player portraits.

    The acceptance requirement is FRAMING, not presence: a portrait that
    renders but clips a chin is a failure. The framing here is a property of
    the fit rather than of a tuned crop -- a square source under
    `object-fit: contain` cannot be cropped on any axis -- so what is
    assertable offline is exactly the set of declarations that guarantee it.
    Rendered proof at 1440 / 1280 / 390 is Playwright's job, per this file's
    header.
    """

    def _portrait_css(self) -> str:
        return _css().split(".player-portrait {", 1)[1].split("}", 1)[0]

    def test_every_ranked_row_carries_its_own_mlbam_portrait(self, home: str) -> None:
        for table_id in ("favorable", "unfavorable"):
            body = home.split(f'id="lb-{table_id}"', 1)[1].split("</table>", 1)[0]
            body = body.split("<tbody>", 1)[1]
            rows = body.count("<tr ")
            srcs = re.findall(r'class="player-portrait-img" src="([^"]+)"', body)
            assert len(srcs) == rows
            # Keyed by MLBAM id -- the same key the player page is keyed on.
            assert all(re.search(r"/v1/people/\d+/headshot/silo/current$", s) for s in srcs)

    def test_the_fit_is_contain_and_the_portrait_is_never_cropped(self) -> None:
        """`cover` crops to fill -- on a head-and-shoulders source that takes
        the chin and jaw first. `contain` cannot crop on either axis."""
        img = _css().split(".player-portrait-img {", 1)[1].split("}", 1)[0]
        assert "object-fit: contain" in img
        assert "cover" not in img
        assert "object-position: center bottom" in img
        assert "object-fit: cover" not in _css()

    def test_the_portrait_is_not_cropped_into_a_circle(self) -> None:
        assert "border-radius" not in self._portrait_css()
        assert "border-radius" not in _css().split(".player-portrait-img {", 1)[1].split("}", 1)[0]

    def test_the_box_is_a_fixed_reservation_so_no_layout_depends_on_it(self) -> None:
        """`DESIGN.md` rule 9: identity is name-first and imagery may never
        move the layout. A declared box means a missing, failed or slow
        portrait changes no row height, no name x, and no column width."""
        block = self._portrait_css()
        assert "width: 34px" in block
        assert "height: 38px" in block
        assert "flex: 0 0 auto" in block

    def test_portraits_are_lazy_and_carry_intrinsic_dimensions(self, home: str) -> None:
        imgs = re.findall(r"<img class=\"player-portrait-img\".*?>", home, flags=re.DOTALL)
        assert imgs
        for img in imgs:
            assert 'loading="lazy"' in img
            assert 'decoding="async"' in img
            assert 'width="34"' in img and 'height="34"' in img

    def test_the_portrait_is_decorative_in_the_accessibility_tree(self, home: str) -> None:
        """The name is right beside it: a portrait that announced itself
        would double every row for a screen reader."""
        cell = home.split('<td class="col-player">', 1)[1].split("</td>", 1)[0]
        assert 'aria-hidden="true"' in cell
        assert 'alt=""' in cell
        assert not re.search(r'alt="[^"]+"', cell)

    def test_a_missing_portrait_falls_back_to_initials_not_a_broken_image(self, home: str) -> None:
        """Three layers: the row is complete with the name alone, the CDN
        serves its own neutral silhouette for a player it has no photo of,
        and a failed REQUEST reveals the initials the markup already
        carries."""
        assert "d_people:generic:headshot:silo:current.png" in home
        cell = home.split('<td class="col-player">', 1)[1].split("</td>", 1)[0]
        assert 'data-initials="PC"' in cell
        css = _css()
        assert ".player-portrait.is-missing::after { display: flex; }" in css
        assert "content: attr(data-initials)" in css
        js = APP_JS.read_text()
        assert "initPortraitFallback" in js
        assert "player-portrait-img" in js

    def test_initials_never_exceed_two_letters_and_skip_generational_suffixes(self) -> None:
        assert dashboard_build.player_initials("Vladimir Guerrero Jr.") == "VG"
        assert dashboard_build.player_initials("Pete Crow-Armstrong") == "PC"
        assert dashboard_build.player_initials("Ronald Acuña Jr.") == "RA"
        assert dashboard_build.player_initials("Ichiro") == "I"
        assert dashboard_build.player_initials("") == ""

    def test_the_third_party_request_is_declared_on_this_route_only(self, site: Path) -> None:
        """The site's first external dependency, opted into by the one page
        that uses it (docs/design/information-architecture.md § Imagery)."""
        home = (site / "index.html").read_text()
        assert f'rel="preconnect" href="{dashboard_build.HEADSHOT_ORIGIN}"' in home
        for route in ("methodology", "status", "demo"):
            other = (site / route / "index.html").read_text()
            assert "preconnect" not in other
            assert dashboard_build.HEADSHOT_ORIGIN not in other

    def test_the_portrait_never_grows_a_row_past_the_density_ceiling(self) -> None:
        """34px of art in a 28px content box would have added 6px to every
        row. The negative block margins spend the contain slack instead, so
        the row grows 3px and stays under the 44px ceiling
        (`docs/design/tables.md`); measured at 1440: 37px -> 40px."""
        assert "margin-block: -4px -3px" in self._portrait_css()

    def test_portrait_and_name_are_one_vertically_centred_unit(self, home: str) -> None:
        cell = home.split('<td class="col-player">', 1)[1].split("</td>", 1)[0]
        assert cell.index("player-portrait") < cell.index("player-link")
        identity = _css().split(".player-identity {", 1)[1].split("}", 1)[0]
        assert "display: flex" in identity
        assert "align-items: center" in identity

    def test_the_identity_column_pays_for_the_portrait_not_the_name(self) -> None:
        """`--lb-player` grew by exactly the portrait's footprint (34px box +
        an 8px gap = 2.625rem) at both table bands, so the NAME keeps the
        width it had before portraits existed. Without that compensation, 26
        of 154 names wrapped to a second line at 800 and rows went to 50.5px
        -- past the 44px ceiling."""
        css = _css()
        root = css.split(":root {", 1)[1].split("}", 1)[0]
        assert "--lb-player: 17.625rem" in root  # was 15rem
        narrow = css.split("@media (max-width: 1023px) {", 1)[1].split("}", 1)[0]
        assert "--lb-player: 13.625rem" in narrow  # was 11rem

    def test_the_identity_unit_survives_the_mobile_transformation(self) -> None:
        mobile = _css().split("@media (max-width: 767px) {", 1)[1]
        assert ".leaderboard .player-portrait {" in mobile
        assert ".leaderboard .player-identity { align-items: center; }" in mobile
