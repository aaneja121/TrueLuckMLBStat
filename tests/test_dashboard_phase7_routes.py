"""Redesign Phase 7: `/demo/`, `/methodology/` and `/status/`.

Builds against synthetic snapshot fixtures only (never this repo's real
`outputs/prospective`/`artifacts/prospective`), and never invokes any
model-training or scoring code. `mlb_luck_score.scoring.public_labels` is
imported for its frozen strings and its banned-phrase checker -- that is a
test importing scoring, which is allowed; the rule that no module under
`dashboard/` may do so is covered by `tests/test_dashboard_isolation.py`.
"""

from __future__ import annotations

import html as html_module
import re
from pathlib import Path

import build as dashboard_build
import pytest

from dashboard_snapshot_fixtures import default_player_record, write_snapshot
from mlb_luck_score.scoring import public_labels
from test_dashboard_counterfactual_content import _both_plays, _write_grid
from test_dashboard_demo_content import _example, _write_fixture

STYLE_CSS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "style.css"
DEMO_SIMULATOR_JS = (
    Path(__file__).resolve().parents[1] / "dashboard" / "static" / "demo_simulator.js"
)

#: `<time datetime>` keeps the machine value, so the raw ISO string must
#: still be somewhere in the markup -- it just must not be what a reader
#: sees. This finds it in visible text only.
_ISO_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def _visible_text(html: str) -> str:
    """Rendered text with markup, `<head>`, scripts, styles and comments
    removed.

    Copy assertions are about what a reader sees, so they are made against
    this rather than against raw HTML, where an attribute value, a `<title>`
    or a Jinja comment could satisfy them by accident.
    """
    body = html[html.index("<body>") :] if "<body>" in html else html
    without_code = re.sub(r"<(script|style)\b.*?</\1>", " ", body, flags=re.DOTALL | re.I)
    without_comments = re.sub(r"<!--.*?-->", " ", without_code, flags=re.DOTALL)
    stripped = re.sub(r"<[^>]+>", " ", without_comments)
    # Entities are unescaped, or `&mdash;` in prose would slip past the
    # product-voice check by never matching the character it renders as.
    return " ".join(html_module.unescape(stripped).split())


def _main_text(html: str) -> str:
    """Visible text of the route's own `<main>`, without the shared header
    and footer. The footer prints the build timestamp as a raw ISO string on
    every route; that is the Phase 2 shell, not this phase's copy.
    """
    start = html.index('<main class="page-shell"')
    return _visible_text(html[start : html.index("</main>", start)] + "")


def _css_without_comments() -> str:
    """Comments in this stylesheet quote CSS and class names, so a name
    check has to run against declarations only."""
    return re.sub(r"/\*.*?\*/", "", STYLE_CSS.read_text(), flags=re.DOTALL)


def _seed(tmp_path: Path) -> tuple[Path, Path]:
    out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
    players = [
        default_player_record(
            batter_id=1, batter_name="Alice Alpha", score=5.25, lower=1.1, upper=9.4
        ),
        default_player_record(
            batter_id=2,
            batter_name="Bob Beta",
            score=-3.10,
            lower=-8.2,
            upper=1.4,
            qualification_status="small_sample",
        ),
    ]
    # Two snapshots on one date, so the history has a superseded row as well
    # as a published one, plus a second date.
    write_snapshot(
        out_root,
        art_root,
        directory_name="2026-01-01",
        data_through_date="2026-01-01",
        snapshot_label=None,
        generated_at="2026-01-02T00:00:00+00:00",
        players=players,
    )
    write_snapshot(
        out_root,
        art_root,
        directory_name="2026-01-01__refreshed",
        data_through_date="2026-01-01",
        snapshot_label="refreshed",
        generated_at="2026-01-02T09:30:00+00:00",
        players=players,
    )
    write_snapshot(
        out_root,
        art_root,
        directory_name="2026-01-02",
        data_through_date="2026-01-02",
        snapshot_label=None,
        generated_at="2026-01-03T14:05:00+00:00",
        players=players,
    )
    # 2025-12-30 leaves 2025-12-31 with no snapshot, mirroring the real
    # 2026-08-07 gap the status page has to render honestly.
    write_snapshot(
        out_root,
        art_root,
        directory_name="2025-12-30",
        data_through_date="2025-12-30",
        snapshot_label=None,
        generated_at="2025-12-31T11:00:00+00:00",
        players=players,
    )
    return out_root, art_root


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("phase7")
    out_root, art_root = _seed(tmp_path)
    dashboard_build.build_dashboard(
        out_dir=tmp_path / "dist",
        outputs_root=out_root,
        artifacts_root=art_root,
        demo_fixture_path=_write_fixture(
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
        ),
        demo_counterfactual_grid_path=_write_grid(tmp_path, _both_plays()),
        build_timestamp="2026-01-03T18:00:00+00:00",
    )
    return tmp_path / "dist"


def _demo(site: Path) -> str:
    return (site / "demo" / "index.html").read_text()


def _methodology(site: Path) -> str:
    return (site / "methodology" / "index.html").read_text()


def _status(site: Path) -> str:
    return (site / "status" / "index.html").read_text()


ROUTES = ("demo", "methodology", "status")


# ══ Status ═══════════════════════════════════════════════════════════════


class TestStatusProvenance:
    def test_the_lead_answers_currency_and_trust_in_one_line(self, site: Path) -> None:
        text = _visible_text(_status(site))
        assert "Data through Jan. 2, 2026, from snapshot 2026-01-02" in text
        assert "Integrity verified" in text
        assert "2 players scored, 1 officially ranked" in text

    def test_timestamps_are_reformatted_for_a_reader_and_kept_for_a_machine(
        self, site: Path
    ) -> None:
        """The baseline defect. A 32-character ISO timestamp is an
        unbreakable token that collapsed the history table's last two
        columns to ~31px; design principle 8 says format the data instead of
        engineering a layout that survives it. The machine value is not
        lost -- it moves into `<time datetime>`.
        """
        html = _status(site)
        assert 'datetime="2026-01-03T14:05:00+00:00"' in html
        assert "Jan. 3, 2026, 14:05 UTC" in _main_text(html)
        assert not _ISO_TIMESTAMP.search(_main_text(html)), (
            "a raw ISO timestamp reached visible text; it belongs in `<time datetime>` only"
        )

    def test_the_snapshot_history_is_a_record_list_not_a_squeezed_table(self, site: Path) -> None:
        """One dom, correct at every width: each snapshot is a `<dl>` of
        label/value pairs, so the pairing survives for a screen reader
        whether the layout is showing columns or records. The visible column
        head is `aria-hidden`, so nothing is announced twice.
        """
        html = _status(site)
        history = html[html.index('class="record-list snapshot-history"') :]
        assert '<div class="record-head" aria-hidden="true">' in history
        snapshot_rows = len(re.findall(r'<li class="record-row(?: is-current)?">', history))
        assert snapshot_rows == 4
        assert history.count('<div class="record-cell">') == 20  # 4 records x 5 fields
        assert "<table" not in history

    def test_exactly_one_history_row_is_marked_current(self, site: Path) -> None:
        """Amber means "the frozen official record", so it marks the one
        snapshot this build was rendered from, not every published row."""
        html = _status(site)
        assert html.count('class="record-row is-current"') == 1
        assert "Published, current" in html
        assert "Superseded" in html

    def test_a_date_with_no_snapshot_is_stated_rather_than_skipped(self, site: Path) -> None:
        """A missing date is a provenance fact. Skipping it leaves a reader
        unable to tell an intentional gap from a rendering bug, and the page
        never interpolates one away. It asserts no cause, because the
        dashboard does not know one.
        """
        html = _status(site)
        assert '<li class="record-row is-gap">' in html
        assert "No valid snapshot for Dec. 31, 2025." in _main_text(html)
        # An absence, not a failure: no accent, no alarm.
        gap = html[html.index('class="record-row is-gap"') :][:400]
        assert "status-accent" not in gap
        assert "FAILED" not in gap

    def test_superseded_snapshots_are_still_listed(self, site: Path) -> None:
        html = _status(site)
        assert "2026-01-01__refreshed" in html
        assert ">2026-01-01<" in html

    def test_model_version_status_and_selection_are_joined_into_one_row(self, site: Path) -> None:
        """The snapshot keys these three under vocabularies that do not line
        up (`infield_defense_expected_winner` vs `infield`). The baseline
        shipped them as disconnected lists, so a reader could not tell which
        status belonged to which version.
        """
        html = _status(site)
        models = html[html.index('class="record-list status-models"') : html.index("Sources and")]
        assert "Infield defense" in models
        assert "infield_hgb_v08" in models
        assert "Limited subgroup evidence" in models
        assert "Near-wall specialist" not in models  # not in this fixture

    def test_an_undocumented_status_value_is_shown_verbatim(self, site: Path) -> None:
        """The snapshot records `outfield: "False"`, which is not one of the
        documented statuses. The dashboard displays; it does not
        reinterpret. It is shown as recorded, in the identifier register so
        it never reads as a label this site chose.
        """
        html = _status(site)
        assert '<dd class="status-code">False</dd>' in html

    def test_qualification_counts_use_public_status_labels(self, site: Path) -> None:
        text = _visible_text(_status(site))
        assert "Small sample" in text
        assert "Qualified" in text
        assert "small_sample" not in text

    def test_status_copy_is_terse(self, site: Path) -> None:
        text = _visible_text(_status(site))
        for verbose in (
            "The currently published snapshot contains",
            "is dependent upon",
            "Provenance for the snapshot currently shown on this site",
        ):
            assert verbose not in text


# ══ Methodology ══════════════════════════════════════════════════════════


class TestMethodologyStructure:
    SECTIONS = (
        "definition",
        "inputs",
        "expected",
        "observed",
        "contact-luck",
        "aggregation",
        "uncertainty",
        "qualification",
        "interpretation",
        "limitations",
        "provenance",
    )

    def test_every_section_has_an_id_and_an_index_entry_in_the_same_order(self, site: Path) -> None:
        html = _methodology(site)
        index_order = re.findall(r'<li><a href="#([a-z-]+)">', html)
        assert index_order == list(self.SECTIONS)
        heading_order = re.findall(r'<h2 id="([a-z-]+)">', html)
        assert heading_order == list(self.SECTIONS)

    def test_the_status_page_can_link_straight_to_qualification(self, site: Path) -> None:
        assert 'href="/methodology/#qualification"' in _status(site)
        assert 'id="qualification"' in _methodology(site)

    def test_the_core_relationship_is_stated_as_an_equation(self, site: Path) -> None:
        text = _visible_text(_methodology(site))
        assert "Observed run value" in text
        assert "Expected run value" in text
        assert "Contact Luck" in text
        # Every term is glossed in words as well as named.
        assert "what the play produced" in text
        assert "what the contact was worth" in text
        assert "the gap, in runs" in text

    def test_the_equation_operators_are_readable_by_a_screen_reader(self, site: Path) -> None:
        """`&minus;` and `=` are glyphs. Each is `aria-hidden` with a word
        beside it, so the equation is not announced as three bare nouns."""
        html = _methodology(site)
        assert '<span class="visually-hidden">minus</span>' in html
        assert '<span class="visually-hidden">equals</span>' in html


class TestMethodologyClaims:
    def test_per_100_is_never_confused_with_a_per_play_run_value(self, site: Path) -> None:
        """The single reading error this metric invites."""
        text = _visible_text(_methodology(site))
        assert "not the run value of any one play" in text
        assert "runs per 100 eligible batted balls across the season so far" in text

    def test_the_interval_is_described_and_never_claimed_as_significance(self, site: Path) -> None:
        text = _visible_text(_methodology(site))
        assert "95% interval" in text
        assert "resamples games" in text
        assert "entirely above zero, overlapping zero, or entirely below zero" in text
        assert "never faded, muted, dashed or marked as an error" in text
        assert "is not a finding of" in text
        for significance_claim in (
            "statistically significant",
            "not significant",
            "no real effect",
            "proves",
        ):
            assert significance_claim not in text.lower()

    def test_qualification_is_not_reduced_to_a_small_sample_for_everyone(self, site: Path) -> None:
        """Phase 4's finding: three of the four unranked statuses are not
        about sample size at all, and a provisionally qualified hitter can
        hold more eligible batted balls than the top-ranked one.
        """
        text = _visible_text(_methodology(site))
        for status_label in (
            "Not reportable",
            "Small sample",
            "Insufficient component coverage",
            "Provisionally qualified",
            "Qualified",
        ):
            assert status_label in text
        assert "at least 200 eligible batted balls and 100 scored games" in text
        assert "can hold more eligible batted balls than the top-ranked one" in text

    def test_the_contact_model_input_claim_is_scoped_to_the_contact_model(self, site: Path) -> None:
        """A batter value DOES reach a later stage: `sprint_speed` is an
        input to the infield-opportunity feature set. Saying "nothing about
        the batter reaches the model" would be an overclaim, so the page
        scopes the claim and names the exception.
        """
        text = _visible_text(_methodology(site))
        assert "The contact model has no input for who was batting" in text
        assert "sprint speed" in text

    def test_unqualified_players_are_reported_never_hidden(self, site: Path) -> None:
        text = _visible_text(_methodology(site))
        assert "keeps a page, a score and an interval" in text

    def test_the_retrospective_framing_reaches_this_route(self, site: Path) -> None:
        text = _visible_text(_methodology(site))
        assert public_labels.RETROSPECTIVE_LIMITATION in text
        assert "Contact Luck makes no forecast" in text

    def test_the_frozen_public_definitions_are_used_verbatim(self, site: Path) -> None:
        text = _visible_text(_methodology(site))
        assert public_labels.CONTACT_LUCK_RUNS_DEFINITION in text
        assert public_labels.POSITIVE_VALUE_DEFINITION in text
        assert public_labels.NEGATIVE_VALUE_DEFINITION in text

    def test_no_component_is_ever_offered_as_a_ranking(self, site: Path) -> None:
        text = _visible_text(_methodology(site))
        assert "no component is ever ranked" in text


# ══ Demo ═════════════════════════════════════════════════════════════════


class TestDemoTeaching:
    def test_the_interactive_walkthrough_comes_before_any_long_explanation(
        self, site: Path
    ) -> None:
        """Section 8: the instrument appears early. The page's own note
        about what the two plays are sits BELOW the plays, not in front of
        them, and the simulator follows.
        """
        html = _demo(site)
        assert html.index('class="demo-grid"') < html.index('class="demo-note prose"')
        assert html.index('class="demo-note prose"') < html.index('class="demo-simulator"')

    def test_probabilities_get_the_phase_6_treatment(self, site: Path) -> None:
        """A bounded 0-100% quantity has no sign, so it gets a neutral bar
        on a full-width track: no donut, no gauge, and neither sign colour.
        """
        html = _demo(site)
        assert html.count('<table class="demo-probabilities">') == 3  # 2 plays + simulator
        assert 'class="demo-prob-track"' in html
        assert 'class="demo-prob-fill"' in html
        assert "what happened" in html

    def test_no_probability_bar_is_painted_in_a_sign_colour(self) -> None:
        css = _css_without_comments()
        block = css[css.index(".demo-prob-track {") : css.index(".demo-rv-line {")]
        for sign_token in ("--fav-", "--unfav-", "--diverging-positive", "--diverging-negative"):
            assert sign_token not in block, (
                f"`{sign_token}` reached a probability bar. Blue and red are reserved for the "
                "sign of a point estimate, and a probability has no sign."
            )

    def test_the_shared_probability_bar_list_is_also_neutral(self) -> None:
        """`.demo-prob-bars`/`.demo-prob-row` are emitted by
        static/showcase_whatif.js for the PLAY page's sensitivity module.
        They were filled with `--diverging-positive` -- the blue that means
        "favorable" -- on a quantity with no sign. That module has no
        committed fixture data, so it had never been rendered.
        """
        css = _css_without_comments()
        block = css[css.index(".demo-prob-bar-fill {") : css.index(".simulator-prob-bar-fill")]
        assert "background: var(--text-muted);" in block
        assert "--diverging" not in block

    def test_the_payoff_carries_its_sign_in_words_as_well_as_colour(self, site: Path) -> None:
        html = _demo(site)
        assert "Unfavorable Contact Luck" in html
        assert "Favorable Contact Luck" in html
        # Redesign Phase 1 numeric primitive: U+2212, never an ASCII hyphen.
        assert "−1.55 runs" in html
        assert "-1.55 runs" not in html
        assert "+0.71 runs" in html

    def test_the_simulator_is_framed_as_hypothetical_throughout(self, site: Path) -> None:
        text = _visible_text(_demo(site))
        assert "hypothetical contact, not a recorded play" in text
        assert "If the result had been" in text
        assert "not where a ball would physically land" in text

    def test_the_reset_control_matches_the_play_page_wording(self, site: Path) -> None:
        assert "Reset to recorded values" in _demo(site)

    def test_the_reference_play_selector_is_a_pressed_button_group_not_a_tablist(
        self, site: Path
    ) -> None:
        """The baseline used `role="tablist"`/`role="tab"` on buttons that
        control no tabpanel and implement no arrow-key behaviour. They pick
        which play's fixed context the sliders run against, which is what a
        pressed-button group is for.
        """
        html = _demo(site)
        selector = html[html.index('class="demo-play-selector"') :][:600]
        assert 'role="group"' in selector
        assert 'aria-pressed="false"' in selector
        assert "tablist" not in html
        assert 'role="tab"' not in html

    def test_the_simulator_uses_the_one_shared_signed_number_formatter(self) -> None:
        """Section 6: Demo duplicated `formatSigned`. It now reads the
        shared `window.ContactLuck.formatSigned`, with a local fallback so a
        missing app.js degrades to a working simulator.
        """
        js = DEMO_SIMULATOR_JS.read_text()
        assert "window.ContactLuck && window.ContactLuck.formatSigned" in js
        assert js.count("function formatSigned(") == 0
        # The contract the shared helper must keep for this page.
        assert '"\\u2212"' in js

    def test_every_slider_and_button_keeps_a_programmatic_label(self, site: Path) -> None:
        html = _demo(site)
        assert '<label for="simulator-ev-slider">' in html
        assert '<label for="simulator-la-slider">' in html
        assert 'aria-label="Hypothetical result"' in html
        assert 'aria-label="Reference play"' in html


# ══ Cross-route ══════════════════════════════════════════════════════════


class TestLegacyCleanup:
    RETIRED_CLASSES = (
        "stat-grid",
        "stat-card",
        "callout",
        "page-subtitle",
        "seam-divider",
        "seam-mark",
        "demo-luck-badge",
        "simulator-play-button",
        "simulator-outcome-button",
        "snapshot-type-",
    )

    def test_generic_stat_tiles_are_gone_from_the_stylesheet(self) -> None:
        """A stat that belongs to a stat line is never a bordered tile
        (guardrails anti-pattern 14). After Phase 7 the generic
        `.stat-grid`/`.stat-card` pair has no definition left to reach for.
        """
        css = _css_without_comments()
        for retired in self.RETIRED_CLASSES:
            assert f".{retired}" not in css, f"`.{retired}` is still defined in style.css"

    def test_no_route_still_renders_a_retired_class(self, site: Path) -> None:
        """Class names only. `data-role="simulator-outcome-button"` and its
        siblings are JavaScript hooks and deliberately unchanged -- renaming
        a hook buys nothing and breaks the one contract these files have.
        """
        for route in ROUTES:
            html = (site / route / "index.html").read_text()
            rendered = {
                token for value in re.findall(r'class="([^"]*)"', html) for token in value.split()
            }
            for retired in self.RETIRED_CLASSES:
                assert not any(cls.startswith(retired) for cls in rendered), (
                    f"`{retired}` still rendered on /{route}/"
                )

    def test_no_coloured_top_border_survives_on_a_card(self) -> None:
        """Both cards on /demo/ carried one: field green on the walkthrough,
        amber on the simulator. Borders are hairline; the two 2px exceptions
        in the system are the focus ring and the zero spine.
        """
        css = STYLE_CSS.read_text()
        demo_block = css[css.index("/* ══ /demo/ ·") : css.index("/* ══ /methodology/")]
        assert "border-top: 3px" not in demo_block
        assert "3px solid" not in demo_block

    def test_the_simulator_field_no_longer_borrows_the_official_record_accent(self) -> None:
        """Amber means "the frozen official record". The simulator's live
        ball and flight path are the opposite of a record.
        """
        css = _css_without_comments()
        block = css[css.index(".simulator-field-path {") : css.index(".demo-onward {")]
        assert "amber" not in block
        assert "fill: var(--text-primary);" in block


class TestProductVoice:
    def test_no_em_dash_reaches_visible_prose(self, site: Path) -> None:
        """docs/design/typography.md § Product voice. The missing-value
        glyph is the one allowed use, and it only appears alone in a value
        position, never inside a sentence.
        """
        for route in ROUTES:
            tokens = _main_text((site / route / "index.html").read_text()).split()
            for token in tokens:
                if "—" not in token:
                    continue
                assert token == "—", f"em dash inside prose on /{route}/: {token!r}"
        # /methodology/ is pure prose and has no value position at all, so it
        # carries none even as the glyph.
        assert "—" not in _main_text(_methodology(site))

    def test_no_banned_phrase_reaches_any_of_the_three_routes(self, site: Path) -> None:
        for route in ROUTES:
            text = _main_text((site / route / "index.html").read_text())
            assert public_labels.check_text_for_banned_phrases(text) == []

    def test_no_ai_assistant_register_or_interface_narration(self, site: Path) -> None:
        for route in ROUTES:
            text = _main_text((site / route / "index.html").read_text()).lower()
            for tic in (
                "in other words",
                "note that",
                "it's important to",
                "it is important to",
                "the question here is",
                "let's",
                "we've designed",
                "this page is designed to",
                "as you can see",
                "simply put",
            ):
                assert tic not in text, f"`{tic}` on /{route}/"

    def test_each_route_has_exactly_one_h1(self, site: Path) -> None:
        for route in ROUTES:
            html = (site / route / "index.html").read_text()
            assert html.count("<h1") == 1, route

    def test_the_page_title_uses_the_scale_and_not_an_off_scale_size(self, site: Path) -> None:
        for route in ROUTES:
            assert '<h1 class="route-title">' in (site / route / "index.html").read_text()
        css = STYLE_CSS.read_text()
        block = css[css.index(".route-title {") :][: css[css.index(".route-title {") :].index("}")]
        assert "var(--fs-700)" in block


class TestSupportingRoutesStayThatWay:
    def test_none_of_the_three_draws_a_zero_spine(self, site: Path) -> None:
        """An amber rule means a zero-centred quantity. Nothing on these
        three routes is one, so none of them borrows the Zero Spine's
        vocabulary (docs/design/dataviz.md § What gets no spine).
        """
        for route in ROUTES:
            html = (site / route / "index.html").read_text()
            assert "cl-scale-field" not in html
            assert "cl-axis-tick" not in html

    def test_the_dashboard_still_only_displays(self, site: Path) -> None:
        """No score, rank, interval or probability may be derived here. The
        one number Phase 7 added to a template is a count of rows the build
        already loaded.
        """
        for route in ROUTES:
            html = (site / route / "index.html").read_text()
            assert "mlb_luck_score" not in html
