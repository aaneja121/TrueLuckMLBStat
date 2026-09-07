"""Redesign Phase 6 -- the individual play page.

Reuses `tests/test_dashboard_explore_build.py`'s fixture seeding (a synthetic,
sharded Explorer artifact set, never the real committed one) and adds the
guarantees Phase 6 introduces:

  * the SAME `run_value` domain as /explore/, by construction -- one scale
    object rendered into both templates, so there cannot be two;
  * one figure carrying expected, observed and the gap between them, on that
    domain, with zero on the shared `--cl-zero`;
  * Zero Spine discipline: nothing else on the page gets an amber rule;
  * probabilities as a bounded 0-100% instrument, not a zero-centred one;
  * the preserved behaviours: one static shell, play-id validation before
    any fetch, the per-game shard, the controlled not-found state, and a
    sensitivity section that stays hidden rather than half-built.

There is no JS runtime here, so client behaviour is verified structurally
against the actual shipped `dist/static/play.js` -- the pattern this suite
already uses for play.js and explore.js. The rendering itself (six real
fixture plays, the overlap cases, the sensitivity module, the navigation
round trip) was exercised in a browser; see
`outputs/figures/phase6_play_2026-08-28/README.md`.
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
)

APP_JS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "app.js"
PLAY_JS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "play.js"
EXPLORE_JS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "explore.js"
WHATIF_JS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "showcase_whatif.js"
CSS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "style.css"


def _flat(html: str) -> str:
    return re.sub(r"\s+", " ", html.replace("&#39;", "'").replace("&amp;", "&"))


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One build shared by every read-only assertion here. The rows span
    both signs and both a measured and an unmeasured batted ball, so the
    run-value domain is real and the missing-measurement path exists."""
    tmp_path = tmp_path_factory.mktemp("play")
    out_root, art_root = _seed_snapshot(tmp_path)
    rows = [
        _index_row(play_id="700001-1-1", contact_luck_runs=1.40, outcome_class="home_run"),
        _index_row(play_id="700001-2-1", contact_luck_runs=-1.10, outcome_class="out"),
        _index_row(
            play_id="700001-3-1",
            contact_luck_runs=0.02,
            outcome_class="single",
            launch_speed=None,
            launch_angle=None,
        ),
    ]
    paths = _seed_explore_fixture(tmp_path, index_rows=rows)
    _build(tmp_path, out_root, art_root, *paths)
    return tmp_path / "dist"


def _play(site: Path) -> str:
    return (site / "plays" / "index.html").read_text()


def _explore(site: Path) -> str:
    return (site / "explore" / "index.html").read_text()


def _scale(html: str) -> dict[str, float]:
    blob = html.split('id="explore-run-value-scale">', 1)[1].split("</script>", 1)[0]
    return json.loads(blob)


class TestSharedRunValueDomain:
    """Section 23 of the brief: there must not be two independently derived
    `run_value` scales. `build_dashboard` builds ONE `ZeroScale` and renders
    the same dict and the same JSON blob into both templates, so this holds
    by construction rather than by two call sites agreeing."""

    def test_the_play_page_and_explore_carry_byte_identical_scales(self, site: Path) -> None:
        play_blob = (
            _play(site).split('id="explore-run-value-scale">', 1)[1].split("</script>", 1)[0]
        )
        explore_blob = (
            _explore(site).split('id="explore-run-value-scale">', 1)[1].split("</script>", 1)[0]
        )
        assert play_blob == explore_blob

    def test_both_routes_print_the_same_domain_and_the_same_ticks(self, site: Path) -> None:
        def ticks(html: str) -> list[str]:
            return re.findall(r'class="cl-axis-tick[^"]*"[^>]*style="left: ([^"]+)"', html)

        play_ticks = ticks(_play(site))
        # Explore renders the rail three times (results axis + two showcase
        # groups); the play page renders it once. The POSITIONS must match.
        explore_ticks = ticks(_explore(site))
        assert play_ticks
        assert explore_ticks[: len(play_ticks)] == play_ticks

    def test_zero_lands_on_the_shared_zero_locus(self, site: Path) -> None:
        html = _play(site)
        cl_zero = float(re.search(r"--cl-zero: ([\d.]+)%", html).group(1)) / 100
        scale = _scale(html)
        span = scale["domain_max"] - scale["domain_min"]
        assert (-scale["domain_min"]) / span == pytest.approx(cl_zero, abs=1e-6)

    def test_only_one_scale_object_is_built(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "dashboard" / "build.py").read_text()
        assert source.count('v.ZeroScale.from_values(\n            "run_value"') == 1
        assert source.count("run_value_scale_json=run_value_scale_json") == 2
        assert source.count("run_value=run_value_context") == 2

    def test_the_browser_never_derives_a_domain_of_its_own(self) -> None:
        js = PLAY_JS.read_text()
        assert "runValuePct" in js
        assert "domain_min" not in js
        assert "domain_max" not in js
        app = APP_JS.read_text()
        pct = app.split("function runValuePct(", 1)[1].split("\n  }", 1)[0]
        assert "scale.domain_min" in pct
        assert "if (fraction < 0) fraction = 0;" in pct
        assert "if (fraction > 1) fraction = 1;" in pct


class TestSharedHelpers:
    """Section 26: the signed-number primitive was carried verbatim by three
    scripts. It lives once now, and nothing re-copies it."""

    def test_format_signed_is_defined_once_across_every_script(self) -> None:
        app = APP_JS.read_text()
        assert app.count("function formatSigned(") == 1
        assert "window.ContactLuck.formatSigned = formatSigned" in app
        for path in (PLAY_JS, EXPLORE_JS, WHATIF_JS):
            assert "function formatSigned(" not in path.read_text(), path.name

    def test_the_scale_helpers_are_defined_once(self) -> None:
        app = APP_JS.read_text()
        assert app.count("function runValueScale(") == 1
        assert app.count("function runValuePct(") == 1
        for path in (PLAY_JS, EXPLORE_JS):
            src = path.read_text()
            assert "function runValueScale(" not in src, path.name
            assert "function runValuePct(" not in src, path.name

    def test_the_sign_primitive_is_unchanged(self) -> None:
        app = APP_JS.read_text()
        signed = app.split("function formatSigned(", 1)[1].split("\n  }", 1)[0]
        assert '(value >= 0 ? "+" : "")' in signed
        assert 'replace("-", MINUS)' in signed


class TestPageArchitecture:
    def test_the_shell_embeds_no_play_specific_data(self, site: Path) -> None:
        """Preserved Phase 4.1 contract: ONE static shell for every play."""
        html = _play(site)
        assert "700001-1-1" not in html
        assert "Test Player" not in html
        assert "data-play-id" not in html
        assert "data-game-pk" not in html

    def test_the_analytical_sequence_is_in_order(self, site: Path) -> None:
        """Verdict, then the figure that produces it, then the contact, then
        what the model expected of that contact, then sensitivity."""
        html = _play(site)
        order = [
            'data-role="play-luck"',
            'class="play-gap"',
            'id="contact-heading"',
            'id="expectation-heading"',
            'data-role="play-whatif-section"',
        ]
        positions = [html.index(marker) for marker in order]
        assert positions == sorted(positions), list(zip(order, positions, strict=False))

    def test_the_contact_facts_are_a_stat_line_not_tiles(self, site: Path) -> None:
        """Guardrails anti-pattern 14, and the baseline shipped four
        `.stat-card` boxes here."""
        html = _play(site)
        assert 'class="play-contact"' in html
        assert "stat-card" not in html
        assert "stat-grid" not in html
        css = CSS.read_text()
        block = css.split(".play-contact-item {", 1)[1].split("}", 1)[0]
        assert "border:" not in block
        assert "border-radius" not in block
        assert "background" not in block

    def test_every_contact_fact_the_artifact_supports_is_shown(self, site: Path) -> None:
        html = _play(site)
        for label in ("Exit velocity", "Launch angle", "Batted-ball type", "Recorded result"):
            assert label in html, label
        for absent in ("Team", "Position", "Pitcher", "Inning", "Headshot"):
            assert absent not in html

    def test_the_legacy_explore_class_coupling_is_gone(self, site: Path) -> None:
        """Section 26: the play page used `.explore-loading` for its own
        loading state. It has its own class now, and the stylesheet has no
        orphan rule left behind."""
        html = _play(site)
        assert "explore-loading" not in html
        assert 'class="play-status"' in html
        rules = re.sub(r"/\*.*?\*/", "", CSS.read_text(), flags=re.S)
        assert ".explore-loading" not in rules


class TestGapFigure:
    def test_the_figure_carries_expected_observed_and_the_gap(self, site: Path) -> None:
        html = _play(site)
        assert 'data-role="play-gap-expected-mark"' in html
        assert 'data-role="play-gap-observed-mark"' in html
        assert 'data-role="play-gap-connector"' in html
        assert 'data-role="play-expected-rv"' in html
        assert 'data-role="play-observed-rv"' in html

    def test_the_two_marks_are_distinguishable_without_colour(self) -> None:
        """Shape and a named label, not red versus blue."""
        css = CSS.read_text()
        expected = css.split(".play-gap-mark.is-expected {", 1)[1].split("}", 1)[0]
        observed = css.split(".play-gap-mark.is-observed {", 1)[1].split("}", 1)[0]
        # Hollow ring versus solid dot, and different sizes.
        assert "border:" in expected
        assert "background: var(--page)" in expected
        assert "background: currentColor" in observed
        assert "border:" not in observed
        assert "18px" in expected
        assert "12px" in observed
        html = (
            Path(__file__).resolve().parents[1] / "dashboard" / "templates" / "play.html"
        ).read_text()
        assert "Expected" in html
        assert "Observed" in html

    def test_a_coincident_pair_still_renders_as_two_marks(self) -> None:
        """The ring is deliberately LARGER than the dot and drawn beneath
        it, so a play whose expected and observed values nearly match shows
        a ring around a dot rather than one mark hiding the other. Neither
        position is ever nudged."""
        css = CSS.read_text()
        expected = css.split(".play-gap-mark.is-expected {", 1)[1].split("}", 1)[0]
        observed = css.split(".play-gap-mark.is-observed {", 1)[1].split("}", 1)[0]
        assert "z-index: 2" in expected
        assert "z-index: 3" in observed
        js = PLAY_JS.read_text()
        gap = js.split("function renderGap(", 1)[1].split("\n  }", 1)[0]
        # Positions come straight from the scale; nothing offsets a mark.
        assert "runValuePct(scale, expected)" in gap
        assert "runValuePct(scale, observed)" in gap

    def test_the_connector_spans_whichever_mark_is_lower(self) -> None:
        """Correct in both directions without the template knowing which of
        the two is on the left."""
        css = CSS.read_text()
        block = css.split(".play-gap-connector {", 1)[1].split("}", 1)[0]
        assert "min(var(--play-expected), var(--play-observed))" in block
        assert "max(var(--play-expected), var(--play-observed))" in block

    def test_the_labels_cannot_collide(self, site: Path) -> None:
        """Expected sits above the field and observed below it, so however
        close the marks are the two values stay readable."""
        html = _play(site)
        expected_at = html.index('class="play-gap-caption is-expected"')
        field_at = html.index('class="cl-scale-field play-gap-field"')
        observed_at = html.index('class="play-gap-caption is-observed"')
        assert expected_at < field_at < observed_at

    def test_the_figure_declares_its_own_domain_in_ticks_and_in_words(self, site: Path) -> None:
        """The anti-fake-alignment guard: this is not `league_per_100`."""
        html = _play(site)
        ticks = re.findall(r'class="cl-axis-tick[^"]*"[^>]*>([^<]+)<', html)
        assert len(ticks) >= 3
        assert any("+0" in t for t in ticks)
        flat = _flat(html)
        assert "Run value, runs." in flat
        assert "The same scale as Play Explorer" in flat

    def test_the_accessible_description_states_all_three_numbers(self) -> None:
        js = PLAY_JS.read_text()
        gap = js.split("function renderGap(", 1)[1].split("\n  }", 1)[0]
        label = gap.split('"aria-label",', 1)[1]
        assert "The model expected" in label
        assert "the play actually produced" in label
        assert "The gap between them is" in label
        # Direction in words, not only in colour.
        assert "in the hitter's favor" in label
        assert "against the hitter" in label


class TestZeroSpineDiscipline:
    def test_only_zero_centred_quantities_carry_a_spine(self) -> None:
        """Section 12: an amber rule means a zero-centred quantity. Exit
        velocity, launch angle, probabilities, dates and categories are not
        zero-centred and get none.

        The scan covers the stylesheet from the play-page block to the end,
        so every component added after it is checked too. The allowed set is
        an explicit list, not a count: adding a spine to something new means
        adding it here and saying which zero-centred quantity it draws.

        - `.play-gap-field` -- expected vs. observed run value on one play.
        - `.pitcher-board-table` -- cumulative Contact Luck allowed, runs
          (Version 0.13.1 local prototype).
        - `.pitcher-mark-row` -- the same quantity on a pitcher card. It
          shares one grouped rule with `.pitcher-league-row`, the population
          strip directly beneath it, which the regex reports under the first
          selector of the pair; both draw the same axis and the same zero.

        All three are signed Contact Luck quantities whose zero is a real
        zero, and all three register at `--cl-zero` (Invariant Z).
        """
        css = CSS.read_text()
        block = css.split("/* ══ The play page (redesign Phase 6)", 1)[1]
        spined = re.findall(r"(\.[\w-]+)[^{]*::before \{[^}]*--cl-zero", block)
        assert spined == [
            ".play-gap-field",
            ".pitcher-board-table",
            ".pitcher-mark-row",
        ], spined

    def test_probabilities_are_not_drawn_on_a_zero_centred_scale(self, site: Path) -> None:
        html = _play(site)
        prob_block = html.split('class="play-probabilities"', 1)[1].split("</table>", 1)[0]
        assert "cl-scale" not in prob_block
        assert "--cl-zero" not in prob_block
        css = CSS.read_text()
        for selector in (".play-prob-track {", ".play-prob-fill {"):
            rule = css.split(selector, 1)[1].split("}", 1)[0]
            assert "--cl-zero" not in rule

    def test_the_contact_line_carries_no_signs_and_no_scale(self, site: Path) -> None:
        html = _play(site)
        contact = html.split('class="play-contact"', 1)[1].split("</dl>", 1)[0]
        assert "cl-scale" not in contact
        js = PLAY_JS.read_text()
        for fn in ("evText", "laText", "bbTypeLabel"):
            body = js.split("function " + fn + "(", 1)[1].split("\n  }", 1)[0]
            assert "formatSigned" not in body
            assert '"+"' not in body


class TestProbabilities:
    def test_every_class_keeps_its_value_and_its_frozen_order(self) -> None:
        js = PLAY_JS.read_text()
        assert 'var CLASS_ORDER = ["out", "single", "double", "triple", "home_run"];' in js
        render = js.split("function renderProbabilities(", 1)[1].split("\n  }\n", 1)[0]
        assert "CLASS_ORDER.forEach" in render
        assert 'play["p_" + cls]' in render

    def test_a_tiny_probability_is_not_rounded_to_zero(self) -> None:
        """5e-10 must not print as "0.0%", which would read as impossible."""
        js = PLAY_JS.read_text()
        assert 'pct < 0.05 && pct > 0 ? "<0.1%"' in js

    def test_the_recorded_result_is_marked_in_text_not_only_in_weight(self) -> None:
        js = PLAY_JS.read_text()
        assert 'tag.textContent = "what happened"' in js
        assert 'tr.className = "is-observed"' in js

    def test_no_donut_gauge_or_third_hue(self) -> None:
        css = CSS.read_text()
        block = css.split("/* ══ The play page (redesign Phase 6)", 1)[1]
        for banned in ("conic-gradient", "radial-gradient", "border-radius: 50%"):
            assert banned not in block, banned
        fill = css.split(".play-prob-fill {", 1)[1].split("}", 1)[0]
        # Neutral ink: a probability has no sign (guardrails 9).
        assert "--fav-" not in fill
        assert "--unfav-" not in fill
        assert "--diverging" not in fill

    def test_the_lede_states_the_dominant_expectation_in_plain_words(self) -> None:
        js = PLAY_JS.read_text()
        assert '"The model gave "' in js
        assert '" the best chance at "' in js
        for jargon in ("predictive distribution", "classified outcome", "posterior"):
            assert jargon not in js


class TestFieldDiagram:
    def test_it_is_hidden_when_the_play_carries_no_location(self) -> None:
        """A ball drawn at home plate would read as a measurement."""
        js = PLAY_JS.read_text()
        field = js.split("function renderField(", 1)[1].split("\n  }", 1)[0]
        assert "if (!hasLocation)" in field
        assert "wrap.hidden = true;" in field

    def test_it_is_labelled_as_approximate(self) -> None:
        js = PLAY_JS.read_text()
        assert "Roughly where it landed" in js
        assert "not a reconstruction of the ball flight" in js

    def test_it_stays_subordinate_to_the_run_value_figure(self, site: Path) -> None:
        html = _play(site)
        assert html.index('class="play-gap"') < html.index('class="play-field"')
        css = CSS.read_text()
        block = css.split(".play-field {", 1)[1].split("}", 1)[0]
        assert "max-width: 300px" in block

    def test_the_svg_holds_no_text_and_is_presentational(self, site: Path) -> None:
        html = _play(site)
        svg = html.split("<svg", 1)[1].split("</svg>", 1)[0]
        assert "<text" not in svg
        assert 'aria-hidden="true"' in svg


class TestSensitivity:
    def test_the_section_ships_hidden(self, site: Path) -> None:
        """A play without a sensitivity artifact never reveals it; there is
        no disabled or half-built simulator state."""
        html = _play(site)
        section = html.split('data-role="play-whatif-section"', 1)[1].split(">", 1)[0]
        assert "hidden" in section

    def test_it_sits_below_the_real_play_accounting(self, site: Path) -> None:
        html = _play(site)
        assert html.index('data-role="play-luck"') < html.index('data-role="play-whatif-section"')
        assert html.index('class="play-gap"') < html.index('data-role="play-whatif-section"')

    def test_it_never_writes_the_real_play_values(self) -> None:
        """It moves the model's expectation for a hypothetical contact and
        nothing else. It must never touch the recorded result, the observed
        run value or the Contact Luck headline."""
        js = WHATIF_JS.read_text()
        for role in ("play-luck", "play-observed-rv", "play-expected-rv", "play-outcome"):
            assert role not in js, role

    def test_it_uses_the_same_probability_vocabulary_as_the_page(self) -> None:
        js = WHATIF_JS.read_text()
        for label in ('"Out"', '"Single"', '"Double"', '"Triple"', '"Home run"'):
            assert label in js, label

    def test_its_copy_is_not_model_documentation(self, site: Path) -> None:
        flat = _flat(_play(site))
        assert (
            "Move exit velocity or launch angle and the model's expectation moves with it." in flat
        )
        for jargon in ("sensitivity analysis", "robustness", "perturbation", "predictive"):
            assert jargon not in flat, jargon


class TestNotFound:
    def test_the_shell_carries_the_controlled_state(self, site: Path) -> None:
        html = _play(site)
        assert 'data-role="play-not-found"' in html
        assert "Play not found" in html

    def test_it_offers_a_route_back_and_shows_no_analytics(self, site: Path) -> None:
        html = _play(site)
        block = html.split('data-role="play-not-found"', 1)[1].split("</div>", 1)[0]
        assert "/explore/" in block
        assert "cl-scale" not in block
        assert "play-gap" not in block

    def test_it_claims_no_failure_it_cannot_know_about(self, site: Path) -> None:
        flat = _flat(_play(site))
        note = flat.split("Play not found</h1>", 1)[1].split("</p>", 1)[0]
        for overclaim in ("error", "failed", "broken", "server", "model"):
            assert overclaim not in note.lower(), overclaim

    def test_every_failure_mode_lands_in_the_same_state(self) -> None:
        js = PLAY_JS.read_text()
        collapsed = re.sub(r"\s+", " ", js)
        # missing id, malformed id, fetch failure, id absent from its game
        assert collapsed.count("showNotFound(page)") >= 4
        # Phase 8: `if (!match)` became `if (!gamePk)` when the play-id
        # contract moved to app.js. Same guard, same landing state.
        assert re.search(r"if \(!gamePk\)\s*\{.*?showNotFound\(page\);\s*return;", collapsed)


class TestNavigationContinuity:
    def test_the_back_link_keeps_the_hitter(self) -> None:
        """A reader who arrived from a hitter's plays returns to that
        hitter, not to a cold Explore."""
        js = PLAY_JS.read_text()
        assert '"/explore/?batter=" + encodeURIComponent(String(play.batter_id))' in js

    def test_a_route_to_the_season_page_exists(self) -> None:
        js = PLAY_JS.read_text()
        assert '"/players/" + encodeURIComponent(String(play.batter_id)) + "/"' in js

    def test_both_routes_out_are_full_size_targets(self) -> None:
        css = CSS.read_text()
        block = css.split(".play-breadcrumb a {", 1)[1].split("}", 1)[0]
        assert "var(--target-min)" in block

    def test_the_page_is_not_overfilled_with_navigation(self, site: Path) -> None:
        """Two routes out of the article, both in one place. The footer's
        own links are the shell's, not this page's."""
        html = _play(site)
        article = html.split('data-role="play-body"', 1)[1].split("</article>", 1)[0]
        assert article.count("<a ") == 2


class TestPreservedBehaviour:
    def test_the_play_id_is_validated_before_any_fetch(self) -> None:
        js = PLAY_JS.read_text()
        assert "window.location.search" in js
        assert 'params.get("id")' in js
        # Phase 8: the play-id contract moved into app.js
        # (`window.ContactLuck.gamePkFromPlayId`) so the two /plays/ scripts
        # stop carrying two copies of the same regex. These assertions now
        # pin the CONTRACT -- derive game_pk, then bail before any fetch --
        # rather than the identifier that used to implement it.
        assert js.index("gamePkFromPlayId") < js.index("fetch(")
        assert js.index("if (!gamePk)") < js.index("fetch(")

    def test_only_the_one_game_shard_is_fetched(self) -> None:
        js = PLAY_JS.read_text()
        assert 'fetch("players.json")' not in js
        assert "/explore/players/" not in js
        assert "/explore/games/" in js
        assert js.count("fetch(") == 1

    def test_nothing_is_recomputed_in_the_browser(self) -> None:
        js = PLAY_JS.read_text()
        for forbidden in ("predict", "Math.exp", "Math.log"):
            assert forbidden not in js
        render = js.split("function renderPlay(", 1)[1].split("\n  }", 1)[0]
        assert "play.contact_luck_runs" in render
        assert "play.expected_run_value" in render
        assert "play.observed_run_value" in render
        # No arithmetic on a displayed scoring value.
        assert "observed_run_value -" not in render
        assert "expected_run_value +" not in render


class TestPageStructureAndAccessibility:
    def test_one_visible_h1_and_a_coherent_hierarchy(self, site: Path) -> None:
        """The shell carries two `h1`s -- the play's and the not-found
        state's -- but exactly one is ever shown, because the other's
        container ships `hidden` and only one is revealed."""
        html = _play(site)
        body = html.split("<main", 1)[1]
        assert body.count("<h1") == 2
        assert 'data-role="play-not-found" hidden' in html
        assert 'data-role="play-body" hidden' in html
        levels = [int(m) for m in re.findall(r"<h([1-6])[ >]", body)]
        assert set(levels) <= {1, 2}

    def test_no_duplicate_ids(self, site: Path) -> None:
        ids = re.findall(r'\sid="([^"]+)"', _play(site))
        assert len(ids) == len(set(ids)), sorted(i for i in ids if ids.count(i) > 1)

    def test_every_aria_reference_resolves(self, site: Path) -> None:
        html = _play(site)
        ids = set(re.findall(r'\sid="([^"]+)"', html))
        for attr in ("aria-controls", "aria-labelledby", "aria-describedby"):
            for value in re.findall(rf'{attr}="([^"]+)"', html):
                for ref in value.split():
                    assert ref in ids, f"{attr}={ref}"

    def test_the_gap_figure_is_an_image_with_a_text_equivalent(self, site: Path) -> None:
        html = _play(site)
        field = html.split('data-role="play-gap-field"', 1)[1].split(">", 1)[0]
        assert 'role="img"' in field
        assert "aria-label" in field

    def test_no_images_and_no_text_inside_a_scaled_svg(self, site: Path) -> None:
        """Continuing the Phase 4/5 rule: essential text never lives inside
        a responsively scaled coordinate space."""
        html = _play(site)
        assert "<img" not in html
        assert "<text" not in html
        assert html.count("<svg") == 1

    def test_every_control_is_labelled(self, site: Path) -> None:
        html = _play(site)
        for control_id in ("whatif-ev-slider", "whatif-la-slider"):
            assert f'<label for="{control_id}"' in html
            assert f'id="{control_id}"' in html

    def test_interactive_targets_meet_the_minimum(self) -> None:
        css = CSS.read_text()
        for selector in (".play-breadcrumb a {", ".play-whatif-reset {", ".play-whatif-slider {"):
            block = css.split(selector, 1)[1].split("}", 1)[0]
            assert "var(--target-min)" in block, selector

    def test_the_mobile_row_keeps_every_value(self) -> None:
        """Section 18: nothing sheds, and the run-value figure keeps its
        axis, both marks and both labelled values."""
        css = CSS.read_text()
        page_block = css.split("/* ══ The play page (redesign Phase 6)", 1)[1]
        mobile = page_block.split("@media (max-width: 767px) {", 1)[1].split("/* At 320", 1)[0]
        assert "display: none" not in mobile
        assert ".play-contact,\n  .play-whatif-rv {" in mobile
        # The figure and its axis are untouched at mobile: no rule in the
        # block narrows, hides or sheds any part of them.
        assert "play-gap-field" not in mobile
        assert "cl-axis-tick" not in mobile


class TestCopy:
    def test_no_em_dashes_in_the_rendered_page(self, site: Path) -> None:
        """docs/design/typography.md § Product voice. The missing-value
        glyph in a numeric cell is the one allowed use, and it lives in
        play.js rather than in the shell."""
        html = _play(site)
        body = re.sub(r"<head.*?</head>", "", html, flags=re.S)
        body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
        assert "—" not in body

    def test_no_em_dashes_in_the_runtime_prose(self) -> None:
        js = PLAY_JS.read_text()
        prose = re.sub(r"//[^\n]*", "", js)
        assert prose.count("—") == 1, "only the missing-value glyph"
        assert 'var EM_DASH = "—";' in js

    def test_no_banned_phrases(self, site: Path) -> None:
        html = (_play(site) + PLAY_JS.read_text()).lower()
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
        flat = _flat(_play(site))
        assert "retrospective description of realized outcomes" in flat
        assert "not a projection of future performance" in flat

    def test_the_observed_run_value_distinction_survives(self, site: Path) -> None:
        """The baseline spent a two-sentence callout on it; one sentence
        keeps the qualification without the lecture."""
        flat = _flat(_play(site))
        assert "baserunner advancement" in flat
        assert "differ from what the recorded result alone would suggest" in flat

    def test_the_equation_is_stated_in_words_not_notation(self) -> None:
        js = PLAY_JS.read_text()
        assert '"The model expected "' in js
        assert '" gap is the Contact Luck on the play."' in js


class TestFixtureProductionBoundary:
    def test_a_bare_build_ships_no_play_route_at_all(self, tmp_path: Path) -> None:
        out_root, art_root = _seed_snapshot(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        assert not (tmp_path / "dist" / "plays").exists()
        assert not (tmp_path / "dist" / "explore").exists()

    def test_the_domain_follows_the_artifacts_it_is_given(self, tmp_path: Path) -> None:
        """Two artifact sets must produce two domains, which is what proves
        the play page's scale comes from the data in hand rather than from a
        constant that would be wrong on production."""
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
            html = (tmp_path / f"dist{i}" / "plays" / "index.html").read_text()
            domains.append(_scale(html)["domain_max"])
        assert domains[0] < domains[1]
