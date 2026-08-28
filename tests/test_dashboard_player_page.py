"""Redesign Phase 4 -- the player page.

What is asserted here is the *contract*: the hero is drawn on the identical
canonical scale as this hitter's leaderboard row, the component
decomposition earns its own scale by declaring it, the off-scale vocabulary
distinguishes "the interval continues" from "the estimate is not on this
scale", the trend keeps its declared exception and pays for it, and the
qualification states preserve their product invariants. Styling is not
asserted; a later phase may reskin any of this without touching a test.

Rendered geometry -- pixel registration across routes, mobile structure,
label sizes after layout -- is verified with the globally configured
Playwright MCP, because this repository has no browser-automation test
dependency (`CLAUDE.md` § Verification before claiming done). The build-time
half is assertable, and is asserted.
"""

from __future__ import annotations

import re
from pathlib import Path

import build as dashboard_build
import pytest
import visuals as v

from dashboard_snapshot_fixtures import default_player_record, write_snapshot

STYLE_CSS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "style.css"
EXPLORE_JS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "explore.js"

# Chosen to exercise every state the page has to survive, against the real
# canonical domain the qualified five below produce.
QUALIFIED_FAVORABLE = 1
QUALIFIED_UNFAVORABLE = 2
NEAR_ZERO = 3
SMALL_SAMPLE = 5
PROVISIONAL = 6
POINT_OFF_SCALE_HIGH = 7
INTERVAL_CLIPPED_BOTH = 8


def _players() -> list[dict]:
    return [
        default_player_record(
            batter_id=QUALIFIED_FAVORABLE,
            batter_name="Pete Crow-Armstrong",
            score=7.62,
            lower=3.15,
            upper=11.81,
        ),
        default_player_record(
            batter_id=QUALIFIED_UNFAVORABLE,
            batter_name="Salvador Perez",
            score=-6.41,
            lower=-9.63,
            upper=-3.23,
        ),
        default_player_record(
            batter_id=NEAR_ZERO, batter_name="Jake Burger", score=-0.06, lower=-4.10, upper=3.98
        ),
        default_player_record(
            batter_id=4, batter_name="Middle Of The Pack", score=0.19, lower=-3.30, upper=3.91
        ),
        default_player_record(
            batter_id=SMALL_SAMPLE,
            batter_name="Unqualified Ulysses",
            score=-3.98,
            lower=-9.88,
            upper=1.58,
            bbe=60,
            qualification_status="small_sample",
        ),
        # More eligible batted balls than the #1 ranked hitter, and still
        # unranked. The baseline told every unqualified hitter they had too
        # few batted balls; for this status that is simply false.
        default_player_record(
            batter_id=PROVISIONAL,
            batter_name="Provisional Priya",
            score=-1.52,
            lower=-4.80,
            upper=1.76,
            bbe=325,
            qualification_status="provisionally_qualified",
        ),
        default_player_record(
            batter_id=POINT_OFF_SCALE_HIGH,
            batter_name="Offscale Olivia",
            score=17.8,
            lower=5.58,
            upper=29.65,
            bbe=40,
            qualification_status="small_sample",
        ),
        default_player_record(
            batter_id=INTERVAL_CLIPPED_BOTH,
            batter_name="Wide Wanda",
            score=-2.5,
            lower=-24.84,
            upper=19.48,
            bbe=30,
            qualification_status="small_sample",
        ),
    ]


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("player")
    out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
    # Three snapshots so the trend has something to draw, with a deliberate
    # gap between 01-02 and 01-04 that must never be interpolated.
    for day in ("2026-01-01", "2026-01-02", "2026-01-04"):
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
        build_timestamp="2026-01-04T12:00:00+00:00",
    )
    return dist


def _page(site: Path, batter_id: int) -> str:
    return (site / "players" / str(batter_id) / "index.html").read_text()


def _css() -> str:
    return re.sub(r"/\*.*?\*/", "", STYLE_CSS.read_text(), flags=re.DOTALL)


def _player_css() -> str:
    """Just the Phase 4 player-page block.

    Bounded by SELECTORS, not the section comments, because `_css()` has
    already stripped the comments out. The lower bound is `.player-header`,
    the first rule of the retained legacy block that `/plays/` and
    `/explore/` still render (Phases 5 and 6) -- those rules predate the
    type scale and would otherwise be attributed to this page. If that block
    is ever removed, this raises rather than silently widening its scope."""
    css = _css()
    assert ".player-header {" in css, "legacy player-header block moved; rebound _player_css"
    return css.split(".player-breadcrumb {", 1)[1].split(".player-header {", 1)[0]


def _flat(html: str) -> str:
    """Collapsed whitespace with entity apostrophes restored. Template
    source wraps at 100 columns and Jinja escapes `'` to `&#39;`, so a
    prose assertion written the way the sentence reads would otherwise fail
    on wherever the line happened to break."""
    return re.sub(r"\s+", " ", html.replace("&#39;", "'").replace("&amp;", "&"))


def _field_vars(html: str, marker: str) -> dict[str, float]:
    """The three inline percentages on the first `.cl-scale-field` after
    `marker` -- the entire coordinate contract a mark rests on."""
    tail = html.split(marker, 1)[1]
    style = re.search(r'class="cl-scale-field[^"]*"[^>]*style="([^"]+)"', tail, re.DOTALL)
    assert style is not None, f"no scale field after {marker!r}"
    return {
        key: float(re.search(rf"--cl-{key}:\s*([\d.]+)%", style.group(1)).group(1))
        for key in ("lo", "hi", "pt")
    }


def _bar_rows(html: str) -> list[dict[str, str]]:
    body = html.split('<table class="component-decomposition"', 1)[1].split("</tbody>", 1)[0]
    return [
        {"label": label, "value": value, "status": status}
        for label, value, status in re.findall(
            r'<th scope="row" class="col-component">(.*?)</th>.*?'
            r'class="col-component-value num">(.*?)</td>.*?'
            r'class="col-component-status">(.*?)</td>',
            body,
            flags=re.DOTALL,
        )
    ]


class TestSharedDomainRegistration:
    """Invariant D across routes. The hero and the leaderboard row for one
    hitter are two renderings of ONE measurement; if their percentages can
    diverge, the site is quietly drawing two different scales."""

    @pytest.mark.parametrize("batter_id", [QUALIFIED_FAVORABLE, QUALIFIED_UNFAVORABLE, NEAR_ZERO])
    def test_hero_and_leaderboard_row_carry_identical_percentages(
        self, site: Path, batter_id: int
    ) -> None:
        home = (site / "index.html").read_text()
        row_table = home.split('id="lb-favorable"', 1)[1]
        row = row_table.split(f'href="/players/{batter_id}/"', 1)[1]
        hero = _field_vars(_page(site, batter_id), 'class="player-mark-row"')
        assert hero == _field_vars(row, "<td")

    def test_the_page_declares_exactly_one_zero_locus(self, site: Path) -> None:
        html = _page(site, QUALIFIED_FAVORABLE)
        assert len(set(re.findall(r"--cl-zero:\s*([\d.]+%)", html))) == 1

    def test_the_hero_never_carries_its_own_domain(self, site: Path) -> None:
        """A per-page domain is how the baseline moved zero on unqualified
        pages. Nothing on this page may recompute one."""
        for batter_id in (QUALIFIED_FAVORABLE, SMALL_SAMPLE, POINT_OFF_SCALE_HIGH):
            html = _page(site, batter_id)
            hero_block = html.split('class="player-hero ', 1)[1].split("</div>", 1)[0]
            assert "--cl-zero:" not in hero_block

    def test_the_league_ground_is_the_leaderboards_own_distribution(self, site: Path) -> None:
        # The homepage renders the strip once per ranking tab, so compare
        # against one table's worth.
        home = (site / "index.html").read_text().split('id="lb-favorable"', 1)[1]
        pattern = r'class="cl-distribution-mark[^"]*"\s*style="left: ([\d.]+%)"'
        marks = re.findall(pattern, home.split("</thead>", 1)[0])
        player = re.findall(pattern, _page(site, QUALIFIED_FAVORABLE))
        assert marks == player and len(marks) > 1


class TestPrimaryVerdict:
    def test_the_score_carries_its_unit_and_direction_in_text(self, site: Path) -> None:
        """The mark is `aria-hidden`; if the unit is not attached to the
        number, a screen reader gets a bare figure."""
        html = _page(site, QUALIFIED_FAVORABLE)
        score = html.split('class="player-score num">', 1)[1].split("</p>", 1)[0]
        assert "+7.62" in score
        assert "runs per 100 eligible batted balls" in score
        assert "favorable" in score

    def test_the_interval_is_never_behind_a_toggle(self, site: Path) -> None:
        for batter_id in (QUALIFIED_FAVORABLE, SMALL_SAMPLE, INTERVAL_CLIPPED_BOTH):
            html = _page(site, batter_id)
            interval = html.split('class="player-score-interval"', 1)[1].split("</p>", 1)[0]
            assert "95% interval" in interval
            assert "<details" not in html.split('class="player-hero ', 1)[1].split("</div>")[0]

    def test_the_numeral_anchor_keeps_it_on_the_field(self, site: Path) -> None:
        """Anchored at its own `--cl-pt`, a centred numeral runs off the
        plot near either edge. The ANCHOR changes, never the position."""
        anchors = {}
        for batter_id in (QUALIFIED_FAVORABLE, QUALIFIED_UNFAVORABLE, POINT_OFF_SCALE_HIGH):
            html = _page(site, batter_id)
            anchors[batter_id] = re.search(r'data-anchor="(\w+)"', html).group(1)
        assert anchors[POINT_OFF_SCALE_HIGH] == "end"
        assert anchors[QUALIFIED_FAVORABLE] == "center"

    def test_sign_follows_the_point_estimate_even_when_the_interval_crosses_zero(
        self, site: Path
    ) -> None:
        html = _page(site, NEAR_ZERO)
        assert "is-unfavorable" in html.split('class="player-hero ', 1)[1][:40]
        assert "cl-scale-unfavorable" in html.split('class="player-mark-row"', 1)[1][:400]

    def test_no_interval_is_ever_de_emphasised_for_crossing_zero(self) -> None:
        """Guardrail 17 is a product invariant, not a style preference. The
        page has no code path that varies a mark by interval interpretation."""
        template = (
            Path(__file__).resolve().parents[1] / "dashboard" / "templates" / "player.html"
        ).read_text()
        assert "interval_interpretation" not in template

    def test_the_supporting_sample_is_a_stat_line_not_tiles(self, site: Path) -> None:
        html = _page(site, QUALIFIED_FAVORABLE)
        assert "stat-card" not in html
        assert 'class="player-sample"' in html
        for label in ("Eligible BBE", "Scored games", "Total Contact Luck Runs"):
            assert label in html

    def test_scored_games_keeps_its_disambiguation(self, site: Path) -> None:
        html = _page(site, QUALIFIED_FAVORABLE)
        assert "not official MLB games played" in html


class TestOffScaleVocabulary:
    """Two shapes, two meanings, and the solid one wins on its own edge."""

    def test_an_off_scale_point_draws_no_dot(self, site: Path) -> None:
        field = _page(site, POINT_OFF_SCALE_HIGH).split('class="player-mark-row"', 1)[1]
        field = field.split("</div>", 1)[0]
        assert 'data-point-offscale="above"' in field
        assert "cl-scale-point" not in field
        assert "is-high is-point" in field

    def test_a_clipped_interval_draws_a_chevron_and_keeps_its_dot(self, site: Path) -> None:
        field = _page(site, INTERVAL_CLIPPED_BOTH).split('class="player-mark-row"', 1)[1]
        field = field.split("</div>", 1)[0]
        assert 'data-clipped="both"' in field
        assert "cl-scale-point" in field
        assert "is-low is-interval" in field
        assert "is-high is-interval" in field

    def test_the_true_numbers_survive_every_clip(self, site: Path) -> None:
        for batter_id in (POINT_OFF_SCALE_HIGH, INTERVAL_CLIPPED_BOTH):
            html = _page(site, batter_id)
            assert 'class="player-offscale-note"' in html
            assert "the actual point estimate and interval" in _flat(html)

    def test_an_unclipped_page_carries_no_off_scale_note(self, site: Path) -> None:
        assert "player-offscale-note" not in _page(site, QUALIFIED_FAVORABLE)

    def test_the_two_edge_marks_are_visually_distinct(self) -> None:
        """Both fill and scale differ, so the distinction survives greyscale
        -- the same relationship `render_interval_bar_svg` draws."""
        css = _css()
        interval = css.split(".cl-scale-edge.is-interval {", 1)[1].split("}", 1)[0]
        point = css.split(".cl-scale-edge.is-point {", 1)[1].split("}", 1)[0]
        assert "background: none" in interval
        assert "border-top" in interval and "border-left" in interval
        assert "solid transparent" in point
        assert "currentColor" in css.split(".cl-scale-edge.is-point.is-high", 1)[1][:80]


class TestQualificationStates:
    def test_a_qualified_player_gets_both_ranks_as_answers(self, site: Path) -> None:
        html = _page(site, QUALIFIED_FAVORABLE)
        assert "Most favorable realized luck" in html
        assert "Least favorable outcomes relative to expectation" in html
        assert "#1" in html
        assert "qualified hitters" in html

    def test_neither_ranking_is_ever_called_worst(self, site: Path) -> None:
        for batter_id in (QUALIFIED_FAVORABLE, QUALIFIED_UNFAVORABLE):
            assert "worst" not in _page(site, batter_id).lower()

    @pytest.mark.parametrize("batter_id", [SMALL_SAMPLE, PROVISIONAL, POINT_OFF_SCALE_HIGH])
    def test_an_unranked_player_emits_no_rank(self, site: Path, batter_id: int) -> None:
        html = _page(site, batter_id)
        assert 'class="player-standing"' not in html
        assert 'class="player-standing-unranked"' in html
        assert "No official rank" in html

    @pytest.mark.parametrize("batter_id", [SMALL_SAMPLE, PROVISIONAL])
    def test_an_unranked_player_keeps_a_score_and_an_interval(
        self, site: Path, batter_id: int
    ) -> None:
        html = _page(site, batter_id)
        assert 'class="player-score num"' in html
        assert "95% interval" in html
        assert 'class="cl-scale-field' in html

    def test_an_unranked_player_is_never_de_emphasised(self) -> None:
        """No fade, no mute, no warning colour. `--text-primary` on that
        paragraph is a preserved product invariant, not a style choice."""
        css = _css()
        block = css.split(".player-standing-unranked {", 1)[1].split("}", 1)[0]
        assert "color: var(--text-primary)" in block
        assert "opacity" not in block
        assert "--unfav" not in block and "--accent-amber" not in block

    def test_the_reason_for_no_rank_is_specific_to_the_status(self, site: Path) -> None:
        """`provisionally_qualified` is about the PROVENANCE of the value,
        not the size of the sample -- this hitter has 325 eligible batted
        balls, more than the #1 ranked one. The baseline told them they had
        too few."""
        provisional = _flat(_page(site, PROVISIONAL))
        small = _flat(_page(site, SMALL_SAMPLE))
        assert "provisional or limited-evidence" in provisional
        assert "minimum number of eligible batted balls" not in provisional
        assert "minimum number of eligible batted balls" in small
        assert "provisional or limited-evidence" not in small.split("player-sample", 1)[0]

    def test_every_unqualified_status_has_its_own_explanation(self) -> None:
        """The four non-qualified values of the frozen public-score schema.
        Listed here rather than imported because no test of the dashboard
        may pull in scoring code either; if the schema grows a fifth, this
        fails and someone writes the sentence rather than shipping the
        generic fallback."""
        expected = {
            "small_sample",
            "insufficient_component_coverage",
            "provisionally_qualified",
            "not_reportable",
        }
        assert set(dashboard_build._QUALIFICATION_EXPLANATIONS) == expected
        for status, text in dashboard_build._QUALIFICATION_EXPLANATIONS.items():
            assert text.strip().endswith("."), status


class TestComponentDecomposition:
    def test_the_components_sum_to_the_headline(self, site: Path) -> None:
        rows = _bar_rows(_page(site, QUALIFIED_FAVORABLE))
        parts = [r for r in rows if r["label"] != "Total Contact Luck"]
        total = next(r for r in rows if r["label"] == "Total Contact Luck")

        def num(text: str) -> float:
            return float(text.replace("−", "-").replace("+", ""))

        assert len(parts) == 4
        assert sum(num(r["value"]) for r in parts) == pytest.approx(num(total["value"]), abs=0.02)

    def test_every_displayed_component_value_carries_a_status(self, site: Path) -> None:
        for row in _bar_rows(_page(site, QUALIFIED_FAVORABLE)):
            assert row["status"].strip()

    def test_the_status_shares_a_row_with_the_value_it_qualifies(self, site: Path) -> None:
        """The baseline shipped values and statuses as two disconnected
        tables; nothing tied a row in one to a row in the other."""
        html = _page(site, QUALIFIED_FAVORABLE)
        assert html.count('<table class="component-decomposition"') == 1
        rows = _bar_rows(html)
        contact = next(r for r in rows if r["label"] == "Contact")
        assert contact["status"] == "Calibrated"

    def test_the_residual_declares_that_it_has_no_model_status(self, site: Path) -> None:
        """Inventing one would misrepresent it: the residual is what the
        three modelled components do not account for."""
        rows = _bar_rows(_page(site, QUALIFIED_FAVORABLE))
        residual = next(r for r in rows if r["label"] == "Unexplained residual")
        assert "Not modelled" in residual["status"]

    def test_defensive_execution_carries_both_of_its_models_statuses(self, site: Path) -> None:
        html = _page(site, QUALIFIED_FAVORABLE)
        disclosure = html.split("Technical reason codes", 1)[1]
        assert "Outfield defense + Infield defense" in disclosure

    def test_the_component_scale_declares_its_own_domain(self, site: Path) -> None:
        """The anti-fake-alignment guard: a figure that borrows the shared
        zero while hiding its own domain is exactly the deception the Zero
        Spine rules forbid."""
        html = _page(site, QUALIFIED_FAVORABLE)
        axis = html.split('class="component-axis-row"', 1)[1].split("</tr>", 1)[0]
        ticks = re.findall(r'class="cl-axis-tick[^"]*"[^>]*>([^<]+)<', axis)
        assert len(ticks) >= 3
        assert any("+0" in t for t in ticks)
        assert "Runs / 100 eligible BBE" in _flat(html.split("component-figure", 1)[0])

    def test_the_component_scale_lands_on_the_shared_zero(self) -> None:
        scale = v.ZeroScale.from_values(
            "component_per_100", "Runs / 100", [3.8, 1.6, 0.3, -0.5], zero_fraction=0.4589
        )
        assert scale.zero_fraction == pytest.approx(0.4589, abs=1e-9)

    def test_a_component_bar_is_not_drawn_as_an_interval(self, site: Path) -> None:
        """A component value has no interval in the snapshot. Drawing it
        with the interval's vocabulary would claim one."""
        body = _page(site, QUALIFIED_FAVORABLE).split('class="component-decomposition"', 1)[1]
        assert "cl-scale-bar" in body
        assert "cl-scale-interval" not in body
        assert "cl-scale-point" not in body


class TestComponentScaleMath:
    """`ZeroScale.from_values` is the machinery that lets a non-league scale
    sit under the same rule. It may extend a domain; it may never crop one."""

    @pytest.mark.parametrize(
        "values",
        [
            [3.0, 1.0, 0.5],  # all positive
            [-3.0, -1.0, -0.5],  # all negative
            [4.0, -2.0, 0.1],  # mixed
            [0.0],  # degenerate
        ],
    )
    @pytest.mark.parametrize("zero_fraction", [0.1, 0.4589, 0.5, 0.9])
    def test_zero_lands_exactly_on_the_target(
        self, values: list[float], zero_fraction: float
    ) -> None:
        scale = v.ZeroScale.from_values("t", "u", values, zero_fraction=zero_fraction)
        assert scale.zero_fraction == pytest.approx(zero_fraction, abs=1e-9)

    @pytest.mark.parametrize(
        "values", [[3.0, 1.0, 0.5], [-3.0, -1.0, -0.5], [4.0, -2.0, 0.1], [0.0]]
    )
    def test_no_value_is_ever_cropped_out_of_the_field(self, values: list[float]) -> None:
        scale = v.ZeroScale.from_values("t", "u", values, zero_fraction=0.4589)
        for value in values:
            assert scale.contains(value)
            assert 0.0 <= scale.fraction_of(value) <= 1.0

    @pytest.mark.parametrize("zero_fraction", [0.0, 1.0, -0.1, 1.2])
    def test_an_impossible_zero_fraction_is_refused(self, zero_fraction: float) -> None:
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            v.ZeroScale.from_values("t", "u", [1.0], zero_fraction=zero_fraction)


class TestRendererAgreement:
    """The Phase 1 plan's cross-renderer check, now that both renderers are
    live: renderer A (CSS percentages) and renderer B (build-time SVG) must
    place zero identically for the same scale, or the site draws two zeros."""

    def test_a_and_b_agree_on_the_zero_locus(self) -> None:
        scale = v.ZeroScale.from_intervals(
            "league_per_100", "Runs / 100", [(-9.63, -3.23), (3.15, 11.81), (-4.1, 3.98)]
        )
        svg = v.render_interval_bar_svg(
            point=0.0, lower=-1.0, upper=1.0, domain=scale.domain, compact=False
        )
        width = float(re.search(r'viewBox="0 0 (\d+)', svg).group(1))
        zero_x = float(re.search(r'class="interval-zero-line" x1="([\d.]+)"', svg).group(1))
        assert zero_x / width == pytest.approx(scale.zero_fraction, abs=5e-4)


class TestSeasonTrend:
    """Gate G2. The y-domain is this player's own -- a declared exception --
    and every compensation the exception was approved with is checked here."""

    def test_the_player_specific_domain_is_retained(self, site: Path) -> None:
        html = _page(site, POINT_OFF_SCALE_HIGH)
        summary = _flat(html.split('class="trend-caption"', 1)[1])
        assert "The vertical scale is this player's own" in summary
        assert "it is not the leaderboard's scale" in summary

    def test_the_league_domain_is_never_forced_onto_the_trend(self, site: Path) -> None:
        """A trend drawn on the league domain would put every player's line
        flat in the middle of an empty chart. The tick values must be this
        player's, so two players with different ranges get different axes."""
        wide = _trend_ticks(_page(site, INTERVAL_CLIPPED_BOTH))
        narrow = _trend_ticks(_page(site, NEAR_ZERO))
        assert wide != narrow

    @pytest.mark.parametrize(
        "batter_id", [QUALIFIED_FAVORABLE, NEAR_ZERO, SMALL_SAMPLE, INTERVAL_CLIPPED_BOTH]
    )
    def test_zero_is_always_drawn_labelled_and_in_range(self, site: Path, batter_id: int) -> None:
        html = _page(site, batter_id)
        assert 'class="trend-gridline is-zero"' in html
        ticks = _trend_ticks(html)
        assert any(t.strip() in {"+0.0", "+0.00"} for t in ticks)

    @pytest.mark.parametrize(
        "batter_id", [QUALIFIED_FAVORABLE, NEAR_ZERO, SMALL_SAMPLE, INTERVAL_CLIPPED_BOTH]
    )
    def test_at_least_three_real_tick_values(self, site: Path, batter_id: int) -> None:
        """Not just the endpoints -- an axis labelled only at its extremes
        is the failure this criterion exists to catch."""
        assert len(_trend_ticks(_page(site, batter_id))) >= 3

    def test_one_precision_for_the_whole_axis(self, site: Path) -> None:
        for batter_id in (QUALIFIED_FAVORABLE, NEAR_ZERO, INTERVAL_CLIPPED_BOTH):
            ticks = _trend_ticks(_page(site, batter_id))
            decimals = {len(t.split(".")[1]) for t in ticks if "." in t}
            assert len(decimals) == 1, ticks

    def test_the_unit_is_on_the_figure_not_only_in_prose(self, site: Path) -> None:
        html = _page(site, QUALIFIED_FAVORABLE)
        unit = html.split('class="trend-unit"', 1)[1].split("</p>", 1)[0]
        assert "Runs / 100" in unit

    def test_the_figure_contains_no_text_in_a_scaled_coordinate_space(self, site: Path) -> None:
        """Anti-pattern 11. The baseline's 640x220 viewBox rendered at
        350x120 on a phone and put its 10px labels at ~5.5 CSS px. Text is
        no longer inside a coordinate space at all, so the class of bug is
        gone rather than re-tuned."""
        html = _page(site, QUALIFIED_FAVORABLE)
        marks = html.split('class="trend-marks"', 1)[1].split("</svg>", 1)[0]
        assert "<text" not in marks
        assert "font-size" not in marks
        assert 'aria-hidden="true"' in html.split('class="trend-marks"', 1)[0][-120:]

    def test_the_league_range_is_drawn_only_where_it_falls_in_view(self, site: Path) -> None:
        """Clamping a reference rule to the plot edge asserts the boundary
        is somewhere it is not."""
        narrow = _page(site, NEAR_ZERO)
        wide = _page(site, INTERVAL_CLIPPED_BOTH)
        assert narrow.count('class="trend-league-rule"') < wide.count('class="trend-league-rule"')
        assert "of this view" in wide

    def test_the_caption_never_contradicts_the_drawing(self, site: Path) -> None:
        for batter_id in (QUALIFIED_FAVORABLE, NEAR_ZERO, SMALL_SAMPLE, INTERVAL_CLIPPED_BOTH):
            html = _page(site, batter_id)
            if "trend-band-note" not in html:
                continue
            drawn = html.count('class="trend-league-rule"')
            note = html.split("trend-band-note", 1)[1].split("</p>", 1)[0]
            if drawn == 0:
                assert "Neither edge falls on the chart" in note
            elif drawn == 1:
                assert "Its other edge lies beyond this view" in note
            else:
                assert "rules mark" in note

    def test_nothing_in_the_trend_borrows_the_zero_spine(self, site: Path) -> None:
        """A figure on a different scale must not look like it shares the
        leaderboard's. No `--cl-zero`, no amber rule, no shared axis chrome."""
        html = _page(site, QUALIFIED_FAVORABLE)
        figure = html.split('class="trend-figure"', 1)[1].split("</figure>", 1)[0]
        assert "--cl-zero" not in figure
        assert "cl-axis-tick" not in figure
        assert "cl-scale-field" not in figure
        css = _css()
        for selector in (".trend-plot {", ".trend-gridline {", ".trend-league-rule {"):
            block = css.split(selector, 1)[1].split("}", 1)[0]
            assert "--cl-zero" not in block
            assert "accent-amber" not in block

    def test_every_point_states_value_date_and_unit(self, site: Path) -> None:
        html = _page(site, QUALIFIED_FAVORABLE)
        table = html.split("Snapshot values", 1)[1]
        assert "2026-01-01" in html  # the marker titles carry the raw date
        assert "January 1, 2026" in table
        assert "Runs / 100" in table
        assert "Eligible BBE" in table

    def test_a_missing_snapshot_is_never_interpolated(self, site: Path) -> None:
        """The gap between 01-02 and 01-04 is real. Three stored snapshots
        must produce three markers, never four."""
        html = _page(site, QUALIFIED_FAVORABLE)
        assert html.count('class="trend-dot') == 3
        assert "2026-01-03" not in html

    def test_a_single_snapshot_refuses_to_draw_a_trend(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        tmp = tmp_path_factory.mktemp("one")
        out_root, art_root = tmp / "outputs", tmp / "artifacts"
        write_snapshot(
            out_root,
            art_root,
            directory_name="2026-01-01",
            data_through_date="2026-01-01",
            snapshot_label=None,
            generated_at="2026-01-01T00:00:00+00:00",
            players=_players(),
        )
        dist = tmp / "dist"
        dashboard_build.build_dashboard(
            out_dir=dist,
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        html = _page(dist, QUALIFIED_FAVORABLE)
        assert "trend-plot" not in html
        assert "a trend needs at least two" in html

    def test_build_trend_figure_refuses_fewer_than_two_points(self) -> None:
        scale = v.ZeroScale.from_intervals("league_per_100", "Runs / 100", [(-1.0, 1.0)])
        assert v.build_trend_figure([], league_scale=scale) is None


def _trend_ticks(html: str) -> list[str]:
    axis = html.split('class="trend-axis"', 1)[1].split("</div>", 1)[0]
    return re.findall(r'class="trend-y-tick[^"]*"[^>]*>([^<]+)<', axis)


class TestPlayToPlayPath:
    def test_no_plays_link_is_emitted_when_the_explorer_is_disabled(self, site: Path) -> None:
        """Fail-closed: a bare build ships the Play Explorer disabled, and a
        link that predictably 404s is worse than no link."""
        html = _page(site, QUALIFIED_FAVORABLE)
        assert "/explore/?batter=" not in html
        assert "not part of the currently published Play Explorer data" in _flat(html)

    def test_the_player_page_is_not_a_dead_end(self, site: Path) -> None:
        """Even with no plays published it offers a route onward, which the
        baseline page did not."""
        html = _page(site, QUALIFIED_FAVORABLE)
        assert "player-next" in html
        assert "/demo/" in html.split("player-next", 1)[1]

    def test_explore_accepts_the_batter_parameter_the_link_emits(self) -> None:
        js = EXPLORE_JS.read_text()
        assert 'get("batter")' in js
        assert "selectPlayer(match)" in js

    def test_explore_ignores_an_unknown_batter_without_erroring(self) -> None:
        js = EXPLORE_JS.read_text()
        block = js.split('get("batter")', 1)[1].split("})", 1)[0]
        assert "if (match) selectPlayer(match)" in block


class TestPageStructureAndAccessibility:
    def test_one_h1_and_no_skipped_heading_levels(self, site: Path) -> None:
        for batter_id in (QUALIFIED_FAVORABLE, SMALL_SAMPLE, POINT_OFF_SCALE_HIGH):
            html = _page(site, batter_id)
            levels = [int(m) for m in re.findall(r"<h([1-6])[ >]", html)]
            assert levels.count(1) == 1
            assert set(levels) <= {1, 2}

    def test_the_marks_are_decoration_and_the_statement_is_text(self, site: Path) -> None:
        html = _page(site, QUALIFIED_FAVORABLE)
        mark = html.split('class="player-mark-row"', 1)[1].split("</div>", 1)[0]
        assert 'aria-hidden="true"' in mark
        assert 'class="player-axis-row" aria-hidden="true"' in html

    def test_the_league_distribution_is_a_labelled_figure(self, site: Path) -> None:
        html = _page(site, QUALIFIED_FAVORABLE)
        label = re.search(r'class="cl-distribution"[^>]*aria-label="([^"]+)"', html)
        assert label is not None
        assert "qualified hitters" in label.group(1)
        assert "runs per 100 eligible batted balls" in label.group(1)

    def test_identity_is_name_first_and_the_id_is_apparatus(self, site: Path) -> None:
        html = _page(site, QUALIFIED_FAVORABLE)
        assert '<h1 class="player-name">Pete Crow-Armstrong</h1>' in html
        apparatus = html.split('class="player-apparatus"', 1)[1].split("</p>", 1)[0]
        assert "MLBAM" in apparatus

    def test_no_layout_depends_on_imagery_or_team_context(self, site: Path) -> None:
        html = _page(site, QUALIFIED_FAVORABLE)
        assert "<img" not in html
        assert "headshot" not in html.lower()
        assert "team" not in html.lower().split("<footer", 1)[0]

    def test_operable_controls_meet_the_target_floor(self) -> None:
        css = _css()
        for selector in (
            ".player-disclosure > summary {",
            ".player-breadcrumb a,\n.player-next-link a {",
        ):
            block = css.split(selector, 1)[1].split("}", 1)[0]
            assert "min-height: var(--target-min)" in block

    def test_the_skip_target_does_not_draw_a_box_around_the_page(self) -> None:
        """`[tabindex]:focus-visible` is (0,2,0) and was beating
        `main:focus-visible` (0,1,1), so the skip link's destination drew a
        2px amber outline around the whole page."""
        css = _css()
        assert "main[tabindex]:focus-visible { outline: none; box-shadow: none; }" in css

    def test_no_arbitrary_font_sizes_were_reintroduced(self) -> None:
        """Every size on this page comes from the nine-step scale."""
        sizes = re.findall(r"font-size:\s*([^;]+);", _player_css())
        assert sizes
        assert all("var(--fs-" in size for size in sizes), [
            s for s in sizes if "var(--fs-" not in s
        ]

    def test_reason_codes_wrap_but_prose_does_not(self) -> None:
        """Anti-pattern 5: `overflow-wrap: anywhere` is scoped to the
        elements that actually hold unbreakable identifiers."""
        assert ".component-reasons .reason-codes { overflow-wrap: anywhere; }" in _css()
        assert _player_css().count("overflow-wrap") == 1


class TestBannedLanguage:
    def test_no_banned_phrase_reaches_a_player_page(self, site: Path) -> None:
        banned = (
            "deserved hits",
            "true talent luck",
            "guaranteed regression",
            "should have produced",
            "defense-independent",
            "statistically significant player",
        )
        for batter_id in (QUALIFIED_FAVORABLE, SMALL_SAMPLE, POINT_OFF_SCALE_HIGH):
            html = _page(site, batter_id).lower()
            for phrase in banned:
                assert phrase not in html

    def test_the_retrospective_limitation_is_on_every_player_page(self, site: Path) -> None:
        for batter_id in (QUALIFIED_FAVORABLE, SMALL_SAMPLE):
            html = _flat(_page(site, batter_id))
            assert "retrospective description of realized outcomes" in html
            assert "not a projection of future performance" in html

    def test_the_interval_is_never_framed_as_significance(self, site: Path) -> None:
        html = _flat(_page(site, NEAR_ZERO))
        assert "does not indicate whether the underlying effect" in html
        assert "not significant" not in html.lower()
