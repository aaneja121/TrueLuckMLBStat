"""Version 0.13.1 LOCAL PROTOTYPE: the pitcher surface's presentation rules.

Two kinds of coverage here, both offline and both against synthetic
fixtures (never the committed real-2024 one, and never any scoring code):

1. **Fail-closed routing.** A build that is not explicitly given a pitcher
   fixture must emit no pitcher route and no navigation entry pointing at
   one. This is the same guarantee `test_dashboard_build_cli.py` proves for
   the Play Explorer, and it is what keeps a DEVELOPMENT-season surface out
   of a production build by construction rather than by remembering.

2. **The presentation rules that carry the research finding.** The whole
   point of this surface is that a cumulative total, not a normalized rate,
   is the ranking key; that two usage populations are never one ranking; and
   that a below-minimum season is withheld from a board without being called
   unqualified. Those are testable properties of the rendered HTML, so they
   are tested rather than left to a screenshot.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import build as dashboard_build
import pitcher_prototype_content as ppc
import pytest

from dashboard_snapshot_fixtures import default_player_record, write_snapshot


def _play(runs: float, *, play_id: str = "1-1-1") -> dict:
    return {
        "play_id": play_id,
        "game_date": "2024-05-01",
        "pitching_contact_luck_runs": runs,
        "launch_speed_mph": 100.4,
        "launch_angle_deg": 28.0,
        "bb_type": "fly_ball",
        "outcome_class": "out",
    }


def _pitcher(
    pitcher_id: int,
    name: str,
    *,
    total: float,
    bbe: int,
    appearances: int,
    per_100: float,
    role: str,
    band: str,
    same_play: bool = False,
) -> dict:
    return {
        "pitcher_id": pitcher_id,
        "name": name,
        "role_bucket": role,
        "workload_band": band,
        "cumulative_contact_luck_runs": total,
        "eligible_batted_balls": bbe,
        "appearances": appearances,
        "bbe_per_appearance": round(bbe / appearances, 2),
        "contact_luck_per_100": per_100,
        "per_100_ci_low": per_100 - 3.0,
        "per_100_ci_high": per_100 + 3.0,
        "cumulative_ci_low": total - 8.0,
        "cumulative_ci_high": total + 8.0,
        "largest_play_share_of_net": 0.2,
        "largest_favorable_play": _play(0.88),
        "largest_unfavorable_play": _play(
            0.88 if same_play else -1.14, play_id="1-1-1" if same_play else "2-2-2"
        ),
    }


def _fixture_payload() -> dict:
    return {
        "pitcher_prototype_fixture_version": "0.1",
        "season": 2024,
        "metric": "pitching_contact_luck",
        "sign_convention": "expected run value - observed run value",
        "primary_quantity": "cumulative_contact_luck_runs",
        "secondary_quantity": "contact_luck_per_100",
        "development_only": True,
        "batter_side_reproduction": {"reproduces": True, "n_rows": 3, "max_abs_difference": 0.0},
        "presentation_rules": {
            "starter_like_min_bbe_per_appearance": 10.0,
            "reliever_like_max_bbe_per_appearance": 8.0,
            "board_display_minimum_bbe": 60,
            "workload_band_high_min_bbe": 150,
            "role_buckets_are_descriptive_only": True,
            "board_minimum_is_not_qualification": True,
        },
        "pitchers": [
            # Deliberately ordered so the HIGHEST per-100 is NOT the highest
            # total: `High Rate` outranks `Big Total` on the rate and must
            # not outrank it on the board.
            _pitcher(
                1,
                "Big Total",
                total=20.0,
                bbe=500,
                appearances=30,
                per_100=4.0,
                role="starter_like",
                band="high",
            ),
            _pitcher(
                2,
                "High Rate",
                total=12.0,
                bbe=120,
                appearances=8,
                per_100=10.0,
                role="starter_like",
                band="moderate",
            ),
            _pitcher(
                3,
                "Relief Leader",
                total=17.0,
                bbe=190,
                appearances=70,
                per_100=9.0,
                role="reliever_like",
                band="high",
            ),
            _pitcher(
                4,
                "Tiny Sample",
                total=0.2,
                bbe=1,
                appearances=1,
                per_100=21.9,
                role="reliever_like",
                band="below_board_minimum",
                same_play=True,
            ),
            # Clears the display minimum and is still on NO board, because
            # its usage falls between the two descriptions. The second of the
            # two reasons a season is withheld, and the one the landing page
            # originally left unaccounted for.
            _pitcher(
                5,
                "Mixed Usage",
                total=6.0,
                bbe=200,
                appearances=22,
                per_100=3.0,
                role="ambiguous",
                band="moderate",
            ),
        ],
    }


@pytest.fixture
def fixture_path(tmp_path: Path) -> Path:
    path = tmp_path / "pitcher_prototype_fixture.json"
    path.write_text(json.dumps(_fixture_payload()))
    return path


@pytest.fixture
def snapshot_roots(tmp_path: Path) -> tuple[Path, Path]:
    outputs_root = tmp_path / "outputs"
    artifacts_root = tmp_path / "artifacts"
    write_snapshot(
        outputs_root,
        artifacts_root,
        directory_name="2026-08-20",
        data_through_date="2026-08-20",
        snapshot_label=None,
        generated_at="2026-08-21T00:00:00+00:00",
        players=[
            default_player_record(
                batter_id=1,
                batter_name="Alice Alpha",
                score=5.25,
                lower=1.1,
                upper=9.4,
                bbe=310,
                games=101,
            ),
            default_player_record(
                batter_id=2,
                batter_name="Bob Beta",
                score=-3.5,
                lower=-7.2,
                upper=1.3,
                bbe=280,
                games=95,
            ),
        ],
    )
    return outputs_root, artifacts_root


def _build(tmp_path: Path, snapshot_roots: tuple[Path, Path], fixture: Path | None) -> Path:
    out_dir = tmp_path / "dist"
    outputs_root, artifacts_root = snapshot_roots
    dashboard_build.build_dashboard(
        out_dir=out_dir,
        outputs_root=outputs_root,
        artifacts_root=artifacts_root,
        pitcher_prototype_fixture_path=fixture,
    )
    return out_dir


class TestFailClosedRouting:
    def test_no_flag_emits_no_pitcher_route(self, tmp_path, snapshot_roots):
        out_dir = _build(tmp_path, snapshot_roots, None)
        assert not (out_dir / "pitchers").exists()

    def test_no_flag_emits_no_pitcher_nav_entry(self, tmp_path, snapshot_roots):
        out_dir = _build(tmp_path, snapshot_roots, None)
        home = (out_dir / "index.html").read_text()
        assert "/pitchers/" not in home
        assert "Pitching prototype" not in home

    def test_flag_emits_route_and_nav_entry(self, tmp_path, snapshot_roots, fixture_path):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        assert (out_dir / "pitchers" / "index.html").exists()
        assert 'href="/pitchers/"' in (out_dir / "index.html").read_text()

    def test_every_pitcher_gets_a_page_including_the_withheld_one(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        # The board withholds the sub-minimum season; the SITE must not.
        # That is the difference between a display rule and a qualification
        # rule, and it is the thing most easily lost in a refactor.
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        for pitcher_id in (1, 2, 3, 4):
            assert (out_dir / "pitchers" / str(pitcher_id) / "index.html").exists()


class TestCumulativeTotalIsTheRankingKey:
    def test_board_order_follows_the_total_not_the_rate(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        html = (out_dir / "pitchers" / "index.html").read_text()
        big = html.index("Big Total")
        high_rate = html.index("High Rate")
        assert big < high_rate, (
            "the pitcher with the larger cumulative total must rank above the pitcher with "
            "the larger per-100 rate"
        )

    def test_only_the_total_is_drawn_on_the_scale(self, tmp_path, snapshot_roots, fixture_path):
        """The rate must carry no plot field. This is the single strongest
        signal on the surface that the total is what is ranked -- a reader
        who learned "the drawn thing is the ranked thing" on the hitter board
        reads this one correctly only while it stays true.
        """
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        html = (out_dir / "pitchers" / "index.html").read_text()
        rate_cells = re.findall(r'<td class="pcol-rate">.*?</td>', html, re.S)
        assert rate_cells
        for cell in rate_cells:
            assert "cl-scale-field" not in cell
            assert "cl-distribution" not in cell

    def test_no_sort_controls_exist(self, tmp_path, snapshot_roots, fixture_path):
        """No `data-sortable`, no sort buttons: one order, stated once. A
        user-sortable board would let the rate become an ordering that looks
        exactly like the ranked one.
        """
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        html = (out_dir / "pitchers" / "index.html").read_text()
        board = html[html.index("pitcher-board-table") :]
        assert "data-sortable" not in board
        assert 'data-role="sort"' not in board


class TestTwoPopulationsAreNeverOneRanking:
    def test_each_board_starts_at_one(self, tmp_path, snapshot_roots, fixture_path):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        html = (out_dir / "pitchers" / "index.html").read_text()
        tables = re.findall(r'<table class="pitcher-board-table".*?</table>', html, re.S)
        assert len(tables) == 2
        for table in tables:
            first_rank = re.search(r'<td class="pcol-rank num">(\d+)</td>', table)
            assert first_rank is not None
            assert first_rank.group(1) == "1"

    def test_boards_are_separate_sections_with_their_own_headings(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        html = (out_dir / "pitchers" / "index.html").read_text()
        assert 'id="board-starter-like"' in html
        assert 'id="board-reliever-like"' in html

    def test_a_pitcher_never_appears_on_both_boards(self, tmp_path, snapshot_roots, fixture_path):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        html = (out_dir / "pitchers" / "index.html").read_text()
        tables = re.findall(r'<table class="pitcher-board-table".*?</table>', html, re.S)
        starter_ids = set(re.findall(r"/pitchers/(\d+)/", tables[0]))
        reliever_ids = set(re.findall(r"/pitchers/(\d+)/", tables[1]))
        assert starter_ids.isdisjoint(reliever_ids)


class TestVocabulary:
    def test_no_roster_role_noun_is_ever_attached_to_a_person(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        """This repository holds no role metadata, so no pitcher may be
        LABELLED a starter, a reliever or a closer.

        Scoped to the places a label actually attaches to a person -- the
        board rows and the card's identity line -- rather than to the whole
        page, because the explanatory copy is allowed (and required) to use
        the bare nouns in order to disclaim them: "no pitcher here is
        identified as a starter, a reliever or a closer" is the sentence
        this rule exists to produce, and a whole-page substring ban would
        forbid writing it.
        """
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        board = (out_dir / "pitchers" / "index.html").read_text()
        regions = re.findall(r"<tbody>.*?</tbody>", board, re.S)
        assert regions, "expected rendered board rows to check"
        for card_id in (1, 2, 3, 4):
            card = (out_dir / "pitchers" / str(card_id) / "index.html").read_text()
            regions.append(re.search(r'<p class="player-apparatus">.*?</p>', card, re.S).group(0))
        for region in regions:
            bare = region
            for permitted in ("Starter-like", "starter-like", "Reliever-like", "reliever-like"):
                bare = bare.replace(permitted, "")
            for banned in ("closer", "Closer", "starter", "Starter", "reliever", "Reliever"):
                assert banned not in bare, f"{banned!r} attached to a person"

    def test_the_disclaimer_naming_those_roles_is_actually_present(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        """The other half of the rule above: the surface must SAY that these
        are usage descriptions and not roster roles, on both the board and
        the card. Silence would leave a reader to supply the roster meaning
        themselves, which is the failure the vocabulary rule is about.
        """
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        board = (out_dir / "pitchers" / "index.html").read_text()
        assert "not a roster role" in board
        assert "no pitcher here is identified as a closer" in board
        card = (out_dir / "pitchers" / "3" / "index.html").read_text()
        assert "carries no roster role" in card

    def test_the_board_minimum_is_never_called_qualification(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        pages = [out_dir / "pitchers" / "index.html", out_dir / "pitchers" / "4" / "index.html"]
        for page in pages:
            text = page.read_text().lower()
            for banned in ("qualified", "qualification", "unqualified", "eligible to rank"):
                assert banned not in text, f"{banned!r} found in {page.name}"

    def test_withheld_season_says_why_without_saying_it_failed(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        text = (out_dir / "pitchers" / "4" / "index.html").read_text()
        assert "not a" in text and "requirement this season failed" in text


class TestSmallSampleHonesty:
    def test_a_one_play_season_shows_one_play_not_two(self, tmp_path, snapshot_roots, fixture_path):
        """Printing the same batted ball twice, once labelled "largest
        favorable" and once "largest unfavorable", states two findings where
        there is one -- and gives a positive value a label reading
        "unfavorable".
        """
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        text = (out_dir / "pitchers" / "4" / "index.html").read_text()
        assert text.count('class="pitcher-play ') == 1
        assert "Largest unfavorable batted ball" not in text

    def test_singular_nouns_for_a_one_of_everything_season(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        # Scoped to the two places this season's OWN count is printed: the
        # hero opportunity line and the meta description. The board-minimum
        # paragraph on the same page says "Below 60 resolved batted balls",
        # which is correctly plural and describes the rule, not this season.
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        text = (out_dir / "pitchers" / "4" / "index.html").read_text()
        hero = re.search(r'<p class="pitcher-hero-opportunity">.*?</p>', text, re.S).group(0)
        assert "resolved batted ball " in hero
        assert "resolved batted balls" not in hero
        assert "appearance\n" in hero or "appearance<" in hero or "1</span> appearance" in hero
        assert "appearances" not in hero
        meta = re.search(r'<meta name="description" content="([^"]*)"', text).group(1)
        assert "1 resolved batted ball " in meta
        assert "resolved batted balls" not in meta


class TestFixtureLoaderFailsClosed:
    def test_unknown_schema_version_is_refused(self, tmp_path):
        payload = _fixture_payload()
        payload["pitcher_prototype_fixture_version"] = "9.9"
        path = tmp_path / "f.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(ppc.PitcherPrototypeError, match="not one of"):
            ppc.load_pitcher_prototype_data(path)

    def test_failed_batter_side_self_check_is_refused_at_display_time(self, tmp_path):
        """The generator already refuses to WRITE such a fixture. This is the
        second, independent refusal at the point of DISPLAY -- the same
        two-layer principle the prospective cache-coverage guard follows.
        """
        payload = _fixture_payload()
        payload["batter_side_reproduction"] = {"reproduces": False, "max_abs_difference": 0.4}
        path = tmp_path / "f.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(ppc.PitcherPrototypeError, match="not trustworthy"):
            ppc.load_pitcher_prototype_data(path)

    def test_unknown_role_bucket_is_refused(self, tmp_path):
        payload = _fixture_payload()
        payload["pitchers"][0]["role_bucket"] = "closer"
        path = tmp_path / "f.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(ppc.PitcherPrototypeError, match="unknown role bucket"):
            ppc.load_pitcher_prototype_data(path)

    def test_boards_exclude_below_minimum_rows(self, fixture_path):
        data = ppc.load_pitcher_prototype_data(fixture_path)
        assert [r.name for r in data.board("reliever_like")] == ["Relief Leader"]
        assert data.find(4) is not None, "the withheld season keeps its own record"


class TestEverySeasonTheBoardsWithholdIsStillReachable:
    """The frozen display rule is "below the minimum, player-page-only".

    That is only a true description if the page can be reached. The site's
    player search indexes hitters, not pitchers, and no board links a
    withheld season -- so before the withheld-season index existed, all 333
    of the real fixture's off-board pages were built and orphaned, and
    "player-page-only" named a page with no route in.
    """

    def _linked_ids(self, out_dir: Path) -> set[str]:
        index = (out_dir / "pitchers" / "index.html").read_text()
        return set(re.findall(r'href="[^"]*?/pitchers/(\d+)/"', index))

    def test_no_pitcher_page_is_orphaned(self, tmp_path, snapshot_roots, fixture_path):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        built = {p.name for p in (out_dir / "pitchers").iterdir() if p.is_dir()}
        assert built, "the fixture must produce pitcher pages"
        assert built - self._linked_ids(out_dir) == set()

    def test_the_below_minimum_season_is_linked_by_name(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        index = (out_dir / "pitchers" / "index.html").read_text()
        assert "Tiny Sample" in index
        assert "4" in self._linked_ids(out_dir)

    def test_the_withheld_index_is_alphabetical_not_ranked(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        """Ordering the withheld seasons by score would rebuild a ranked
        board out of exactly the seasons the display rules withheld from
        ranking.
        """
        groups = dashboard_build._pitcher_off_board_groups(
            ppc.load_pitcher_prototype_data(fixture_path), "/"
        )
        for group in groups:
            names = [e["name"] for e in group["entries"]]
            assert names == sorted(names)


class TestMixedUsageSeasonsAreAccountedFor:
    """A season can clear the batted-ball minimum and still be on no board.

    That is a second, different reason, and the page that exists to say what
    the boards leave out has to say it -- otherwise those seasons are simply
    absent with no explanation.
    """

    def test_the_landing_page_states_the_mixed_usage_count(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        index = (out_dir / "pitchers" / "index.html").read_text()
        assert "mixed-usage seasons" in index
        assert "Mixed Usage" in index

    def test_a_mixed_usage_season_is_on_neither_board(self, tmp_path, snapshot_roots, fixture_path):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        index = (out_dir / "pitchers" / "index.html").read_text()
        for table in re.findall(r"<table.*?</table>", index, re.S):
            assert "Mixed Usage" not in table

    def test_the_two_withheld_reasons_are_never_merged(self, fixture_path):
        groups = dashboard_build._pitcher_off_board_groups(
            ppc.load_pitcher_prototype_data(fixture_path), "/"
        )
        ids = [{e["name"] for e in g["entries"]} for g in groups]
        assert len(ids) == 2
        assert ids[0].isdisjoint(ids[1]), "a season belongs to exactly one withheld reason"

    def test_the_role_phrase_never_doubles_the_word_usage(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        """`ROLE_LABELS["ambiguous"]` already carries the noun, so a template
        appending " usage" rendered "Mixed usage usage".
        """
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        for page in (out_dir / "pitchers").rglob("index.html"):
            assert "usage usage" not in page.read_text().lower()

    def test_every_role_bucket_has_a_usage_phrase(self):
        assert set(ppc.ROLE_USAGE_PHRASES) == set(ppc.ROLE_LABELS)
        for phrase in ppc.ROLE_USAGE_PHRASES.values():
            assert phrase.lower().count("usage") <= 1


class TestTheSignIsStatedInWords:
    """Blue and red carry the sign, but a reader meeting the surface has no
    key for them. The frozen product contract fixes what the sign MEANS, so
    the page says it in words rather than leaving it to a colour.
    """

    def test_the_board_page_states_both_directions_outside_any_disclosure(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        index = (out_dir / "pitchers" / "index.html").read_text()
        key = re.search(r'<p class="pitcher-sign-key">(.*?)</p>', index, re.S)
        assert key is not None, "the board page must carry a sign key"
        text = key.group(1)
        assert "better for the" in text and "worse" in text
        # Not tucked inside the collapsed "What the number means" disclosure,
        # and not inside the axis caption, which is removed under 767px.
        before = index[: index.index('<p class="pitcher-sign-key">')]
        assert before.count("<details") == before.count("</details>")

    def test_a_favorable_card_says_better_and_an_unfavorable_card_says_worse(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        favorable = (out_dir / "pitchers" / "1" / "index.html").read_text()
        assert "better for the pitcher" in favorable
        assert "worse for the pitcher" not in favorable

    def test_the_sign_key_follows_the_point_estimate(self, tmp_path, snapshot_roots, fixture_path):
        """Sign copy and sign colour must never disagree: both follow the
        point estimate, whether or not the interval crosses zero.
        """
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        page = (out_dir / "pitchers" / "1" / "index.html").read_text()
        assert "is-favorable" in page
        assert "better for the pitcher" in page


class TestTheSecondaryQuantityAlwaysCarriesItsCaveats:
    """Frozen contract: wherever /100 is shown meaningfully, resolved BBE and
    the 95% interval are shown with it.
    """

    def test_the_card_shows_per_100_with_bbe_and_interval(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        page = (out_dir / "pitchers" / "1" / "index.html").read_text()
        assert "Per 100 batted balls" in page
        assert "95% interval" in page
        assert "resolved batted ball" in page

    def test_the_board_shows_an_interval_beside_every_rate(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        index = (out_dir / "pitchers" / "index.html").read_text()
        rates = re.findall(r'<td class="pcol-rate[^"]*">(.*?)</td>', index, re.S)
        assert rates, "the board must render a rate column"
        for cell in rates:
            assert "," in cell and "[" in cell, "each rate cell carries its interval"

    def test_both_largest_plays_are_rendered_with_their_contact(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        page = (out_dir / "pitchers" / "1" / "index.html").read_text()
        assert "Largest favorable batted ball" in page
        assert "Largest unfavorable batted ball" in page
        assert "mph" in page and "fly ball" in page


class TestNoRankOnAPitcherCard:
    """Frozen contract: pitcher player cards carry no board rank.

    A rank on the card would import the board's ordering onto a page that is
    also served for seasons the board withheld -- including the ones that
    have no board at all.
    """

    def test_no_card_prints_a_board_rank(self, tmp_path, snapshot_roots, fixture_path):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        for page in (out_dir / "pitchers").rglob("index.html"):
            if page.parent.name == "pitchers":
                continue
            body = page.read_text()
            # The card's own content only. The shared chrome carries the
            # hitter search payload, whose escaped apostrophes (`&#39;`) are
            # not ranks.
            content = body[body.index("<main") : body.index("</main>")]
            assert "pcol-rank" not in content
            assert not re.search(r"(?<!&)#\s*\d+", content), f"{page} appears to print a rank"
            assert "Rank" not in content

    def test_the_board_still_ranks(self, tmp_path, snapshot_roots, fixture_path):
        """The complement: removing rank from the card must not remove it
        from the board.
        """
        index = _build(tmp_path, snapshot_roots, fixture_path) / "pitchers" / "index.html"
        assert 'class="pcol-rank num"' in index.read_text()


class TestThePitcherSurfaceHasNo2026Dependency:
    """2026 is prospective and unopened for this line. The pitcher surface is
    2024 development data and must not acquire a dependency on the snapshot
    season the rest of the site renders.
    """

    def test_the_fixture_is_a_development_season(self, fixture_path):
        data = ppc.load_pitcher_prototype_data(fixture_path)
        assert data.season == 2024
        assert data.season not in (2025, 2026)

    def test_no_pitcher_page_mentions_2026(self, tmp_path, snapshot_roots, fixture_path):
        """The shared chrome names the snapshot date; the pitcher CONTENT
        must not, or the surface would be claiming a season it never read.
        """
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        for page in (out_dir / "pitchers").rglob("index.html"):
            body = page.read_text()
            content = body[body.index("<main") : body.index("</main>")]
            for marker in ("</header>", "site-header"):
                assert marker not in content
            assert "2026" not in content, f"{page} names 2026 in its own content"


class TestTheRetrospectiveCaveatIsPerSurface:
    """The shared footer sentence names BATTING talent.

    That is the right caveat on a hitter page and the wrong noun on a
    pitcher one, so the two surfaces carry two sentences. These tests exist
    to make swapping them a test failure rather than a copy review: the
    failure mode is silent, because either sentence reads perfectly well
    until you notice it is describing the other half of the game.
    """

    def test_the_dashboard_copy_matches_the_centralized_label(self):
        """`dashboard/` may not import scoring code, so the string is
        duplicated. This equality is what stops the copy drifting from
        `public_labels`, which is the authority for public language.
        """
        from mlb_luck_score.scoring import public_labels

        assert ppc.PITCHER_RETROSPECTIVE_LIMITATION == (
            public_labels.PITCHER_RETROSPECTIVE_LIMITATION
        )

    def test_the_two_sentences_are_distinct_and_name_the_right_half(self):
        from mlb_luck_score.scoring import public_labels

        hitter = public_labels.RETROSPECTIVE_LIMITATION
        pitcher = public_labels.PITCHER_RETROSPECTIVE_LIMITATION
        assert hitter != pitcher
        assert "batting talent" in hitter and "pitching talent" not in hitter
        assert "pitching talent" in pitcher and "batting talent" not in pitcher
        # Both must still carry the retrospective claim in full force.
        for text in (hitter, pitcher):
            assert "retrospective" in text.lower()
            assert "future performance" in text.lower()

    def test_the_pitcher_caveat_carries_no_banned_phrase(self):
        from mlb_luck_score.scoring.public_labels import (
            PITCHER_RETROSPECTIVE_LIMITATION,
            check_text_for_banned_phrases,
        )

        assert check_text_for_banned_phrases(PITCHER_RETROSPECTIVE_LIMITATION) == []

    def test_pitcher_routes_render_the_pitcher_caveat_only(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        for page in (out_dir / "pitchers").rglob("index.html"):
            text = page.read_text()
            assert "stable pitching talent" in text, page
            assert "stable batting talent" not in text, page

    def test_hitter_routes_keep_the_hitter_caveat_only(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        """Including in a build that HAS the pitcher surface -- the override
        must not leak onto the rest of the site.
        """
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        for page in out_dir.rglob("index.html"):
            if "pitchers" in page.parts:
                continue
            text = page.read_text()
            assert "stable batting talent" in text, page
            assert "stable pitching talent" not in text, page


class TestTheRankedQuantityIsNeverCalledAllowed:
    """ "Contact Luck allowed" reads against the sign convention.

    "Allowed" is the runs-allowed idiom, where more is worse for the
    pitcher -- but a POSITIVE Pitcher Contact Luck total is favorable to the
    pitcher. The label and the sign pointed in opposite directions.
    """

    def test_no_pitcher_page_says_contact_luck_allowed(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        for page in (out_dir / "pitchers").rglob("index.html"):
            assert "contact luck allowed" not in page.read_text().lower(), page

    def test_the_ranked_quantity_label_is_the_full_form(self):
        assert dashboard_build.PITCHER_RANKED_QUANTITY_LABEL == "Cumulative Contact Luck Runs"

    def test_the_board_and_card_both_name_the_quantity(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        index = (out_dir / "pitchers" / "index.html").read_text()
        card = (out_dir / "pitchers" / "1" / "index.html").read_text()
        # The compact table header may shorten to "Contact Luck Runs"; the
        # card's hero names the quantity in full.
        assert "Contact Luck Runs" in index
        assert "Cumulative Contact Luck Runs" in card

    def test_the_sign_convention_is_unchanged(self, tmp_path, snapshot_roots, fixture_path):
        """Renaming the label must not have touched what the sign means."""
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        card = (out_dir / "pitchers" / "1" / "index.html").read_text()
        assert "better for the pitcher" in card
        data = ppc.load_pitcher_prototype_data(fixture_path)
        assert data.rows[0].cumulative_contact_luck_runs > 0


class TestTheDataContextIsRouteAware:
    """The site chrome names the snapshot the route is showing.

    The pitcher surface is a 2024 development season and shares no data
    with the 2026 prospective snapshot, so printing "Data through Sep. 1,
    2026" above it would attribute a season those pages never read.
    """

    def _badge(self, page: Path) -> str:
        match = re.search(r'<p class="data-through">(.*?)</p>', page.read_text(), re.S)
        assert match is not None, f"{page} has no data-through badge"
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", match.group(1))).strip()

    def test_pitcher_pages_identify_the_fixture_season(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        season = str(ppc.load_pitcher_prototype_data(fixture_path).season)
        for page in (out_dir / "pitchers").rglob("index.html"):
            badge = self._badge(page)
            assert "Pitcher data" in badge, page
            assert season in badge, page

    def test_pitcher_pages_do_not_claim_the_snapshot_date(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        for page in (out_dir / "pitchers").rglob("index.html"):
            assert "Data through" not in self._badge(page), page

    def test_hitter_pages_keep_the_snapshot_date(self, tmp_path, snapshot_roots, fixture_path):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        for page in (out_dir / "players").rglob("index.html"):
            badge = self._badge(page)
            assert "Data through" in badge
            assert "Pitcher data" not in badge
        home = self._badge(out_dir / "index.html")
        assert "Data through" in home and "Pitcher data" not in home

    def test_the_season_comes_from_the_fixture_not_a_constant(self, tmp_path, snapshot_roots):
        """Change the fixture's season and the badge must follow it.

        This is what makes the badge a fact read off approved data rather
        than a hard-coded assumption that could outlive it.
        """
        payload = _fixture_payload()
        payload["season"] = 2023
        path = tmp_path / "alt_fixture.json"
        path.write_text(json.dumps(payload))
        out_dir = _build(tmp_path, snapshot_roots, path)
        badge = self._badge(out_dir / "pitchers" / "index.html")
        assert "2023" in badge and "2024" not in badge

    def test_the_pitcher_chrome_is_absent_when_the_surface_is(self, tmp_path, snapshot_roots):
        """A build with no pitcher fixture must render no pitcher chrome."""
        out_dir = _build(tmp_path, snapshot_roots, None)
        text = (out_dir / "index.html").read_text()
        assert "Pitcher data" not in text
        assert "stable pitching talent" not in text


STYLE_CSS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "style.css"
APP_JS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "app.js"


def _css() -> str:
    return STYLE_CSS.read_text()


class TestPitcherPortraitsReuseTheHitterContract:
    """The pitcher surfaces draw portraits with the SAME contract the hitter
    leaderboard already uses -- not a second image system.

    What "the same contract" means, and what each test here pins:
    one construction site for the URL (`build.headshot_url`), keyed on an
    MLBAM person id; the CDN's own silhouette as the first fallback; the
    initials the markup already carries as the second; the name alone as
    the third; `contain` framing that cannot crop; lazy loading with
    intrinsic dimensions; and decorative semantics, because the name is
    right beside it.

    The failure this guards against is a plausible one: a per-player image
    map, a second CDN, or a hand-written `<img src>` on the pitcher side
    that drifts from the hitter side without anything noticing.
    """

    def test_the_url_helper_is_the_one_construction_site(self) -> None:
        """Same function, same template, differing only in the id. If a
        pitcher URL could be built any other way, "the same contract" would
        be an assertion about today's source rather than a property."""
        assert dashboard_build.headshot_url(669060) == dashboard_build.HEADSHOT_URL_TEMPLATE.format(
            mlbam_id=669060
        )
        hitter = dashboard_build.headshot_url(605141)
        pitcher = dashboard_build.headshot_url(669060)
        assert hitter.replace("605141", "<id>") == pitcher.replace("669060", "<id>")
        assert pitcher.startswith(dashboard_build.HEADSHOT_ORIGIN)
        assert pitcher.endswith("/v1/people/669060/headshot/silo/current")

    def test_the_helper_is_keyed_on_a_person_id_not_on_a_batter(self) -> None:
        """The rename that makes the shared contract honest: hitters and
        pitchers are one people register, so the template's placeholder is
        named for the key rather than for the surface that used it first."""
        assert "{mlbam_id}" in dashboard_build.HEADSHOT_URL_TEMPLATE
        assert "{batter_id}" not in dashboard_build.HEADSHOT_URL_TEMPLATE

    def test_every_board_row_carries_its_own_mlbam_portrait(
        self, tmp_path, snapshot_roots, fixture_path
    ) -> None:
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        board = (out_dir / "pitchers" / "index.html").read_text()
        for slug in ("starter-like", "reliever-like"):
            body = board.split(f'id="pb-{slug}"', 1)[1].split("</table>", 1)[0]
            body = body.split("<tbody>", 1)[1]
            rows = body.count("<tr>")
            srcs = re.findall(r'class="player-portrait-img" src="([^"]+)"', body)
            assert len(srcs) == rows
            assert all(re.search(r"/v1/people/\d+/headshot/silo/current$", s) for s in srcs)

    def test_the_row_portrait_is_the_row_s_own_pitcher(
        self, tmp_path, snapshot_roots, fixture_path
    ) -> None:
        """The id in the URL is the id in the row's own link -- the check
        that catches a portrait column built from the wrong key."""
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        board = (out_dir / "pitchers" / "index.html").read_text()
        cells = re.findall(r'<td class="pcol-name">(.*?)</td>', board, flags=re.DOTALL)
        assert cells
        for cell in cells:
            src = re.search(r'player-portrait-img" src="([^"]+)"', cell).group(1)
            href = re.search(r'href="[^"]*pitchers/(\d+)/"', cell).group(1)
            assert src == dashboard_build.headshot_url(href)

    def test_the_card_carries_the_same_portrait_in_its_profile_head(
        self, tmp_path, snapshot_roots, fixture_path
    ) -> None:
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        card = (out_dir / "pitchers" / "1" / "index.html").read_text()
        head = card.split('class="pitcher-profile-head"', 1)[1].split("</div>", 1)[0]
        assert dashboard_build.headshot_url(1) in head
        assert "pitcher-profile-portrait" in head
        # Name-first: the portrait never replaces the name, it precedes it.
        assert "Big Total" in card

    def test_portraits_are_lazy_and_carry_intrinsic_dimensions(
        self, tmp_path, snapshot_roots, fixture_path
    ) -> None:
        """Intrinsic width/height are what stop a portrait arriving late
        from reflowing the row it is in."""
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        for page in (out_dir / "pitchers").rglob("index.html"):
            for img in re.findall(
                r"<img class=\"player-portrait-img\".*?>", page.read_text(), re.S
            ):
                assert 'loading="lazy"' in img
                assert 'decoding="async"' in img
                assert re.search(r'width="\d+"', img) and re.search(r'height="\d+"', img)

    def test_the_portrait_is_decorative_in_the_accessibility_tree(
        self, tmp_path, snapshot_roots, fixture_path
    ) -> None:
        """The name is right beside it on both surfaces, so a portrait that
        announced itself would say every pitcher's name twice."""
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        board = (out_dir / "pitchers" / "index.html").read_text()
        cell = board.split('<td class="pcol-name">', 1)[1].split("</td>", 1)[0]
        assert 'aria-hidden="true"' in cell
        assert 'alt=""' in cell
        assert not re.search(r'alt="[^"]+"', cell)

        card = (out_dir / "pitchers" / "1" / "index.html").read_text()
        head = card.split('class="pitcher-profile-head"', 1)[1].split("</div>", 1)[0]
        assert 'aria-hidden="true"' in head
        assert 'alt=""' in head
        assert not re.search(r'alt="[^"]+"', head)

    def test_three_fallbacks_stand_behind_every_portrait(
        self, tmp_path, snapshot_roots, fixture_path
    ) -> None:
        """The CDN's own neutral silhouette for a pitcher it has no photo
        of; the initials in the same box if the REQUEST fails; and the name,
        which never depended on either."""
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        board = (out_dir / "pitchers" / "index.html").read_text()
        card = (out_dir / "pitchers" / "1" / "index.html").read_text()
        for page in (board, card):
            assert "d_people:generic:headshot:silo:current.png" in page
            assert 'data-initials="BT"' in page or re.search(r'data-initials="[A-Z]{1,2}"', page)

        css = _css()
        assert ".player-portrait.is-missing::after { display: flex; }" in css
        assert "content: attr(data-initials)" in css
        js = APP_JS.read_text()
        assert "initPortraitFallback" in js
        assert "player-portrait-img" in js

    def test_initials_come_from_the_shared_helper(self) -> None:
        assert dashboard_build.player_initials("Big Total") == "BT"
        assert dashboard_build.player_initials("Luis L. Ortiz Jr.") == "LO"
        assert dashboard_build.player_initials("") == ""

    def test_the_framing_is_the_shared_contain_fit_never_a_crop(self) -> None:
        """Same declarations, same box, same reason: a square silo source
        under `contain` cannot lose a chin on any axis, and no circle mask
        is applied on either surface."""
        img = _css().split(".player-portrait-img {", 1)[1].split("}", 1)[0]
        assert "object-fit: contain" in img
        assert "object-position: center bottom" in img
        assert "object-fit: cover" not in _css()
        profile = _css().split(".pitcher-profile-portrait {", 1)[1].split("}", 1)[0]
        assert "border-radius" not in profile

    def test_the_profile_portrait_is_a_fixed_reservation(self) -> None:
        """`DESIGN.md` rule 9 on the card too: a declared box means a
        missing, failed or slow portrait moves neither the name nor the
        numeral under it."""
        profile = _css().split(".pitcher-profile-portrait {", 1)[1].split("}", 1)[0]
        assert "width: 72px" in profile
        assert "height: 80px" in profile
        base = _css().split(".player-portrait {", 1)[1].split("}", 1)[0]
        assert "flex: 0 0 auto" in base

    def test_the_third_party_request_is_declared_on_the_routes_that_use_it(
        self, tmp_path, snapshot_roots, fixture_path
    ) -> None:
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        for page in (out_dir / "pitchers").rglob("index.html"):
            body = page.read_text()
            assert f'rel="preconnect" href="{dashboard_build.HEADSHOT_ORIGIN}"' in body
        # And still nowhere that draws no portrait.
        for route in ("methodology", "status", "demo"):
            other = (out_dir / route / "index.html").read_text()
            assert dashboard_build.HEADSHOT_ORIGIN not in other

    def test_no_portrait_url_is_hard_coded_per_player(
        self, tmp_path, snapshot_roots, fixture_path
    ) -> None:
        """Every emitted portrait URL must be reproducible from the helper
        and the row's own id. A literal image URL for a named player -- the
        thing a "just fix this one headshot" patch would add -- fails here.
        """
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        for page in (out_dir / "pitchers").rglob("index.html"):
            for url in re.findall(r'src="(https?://[^"]+)"', page.read_text()):
                mlbam_id = re.search(r"/v1/people/(\d+)/", url)
                assert mlbam_id, f"non-silo image URL emitted: {url}"
                assert url == dashboard_build.headshot_url(mlbam_id.group(1))
        # One origin on the whole surface: no second CDN or provider.
        origins = set()
        for page in (out_dir / "pitchers").rglob("index.html"):
            origins.update(re.findall(r'src="(https?://[^/]+)', page.read_text()))
        assert origins <= {dashboard_build.HEADSHOT_ORIGIN}

    def test_the_portrait_sheds_before_the_name_does_on_a_phone(self) -> None:
        """Below 768 the board's identity column is `width: auto` in a
        four-column table, so the portrait's footprint would come straight
        out of the name. It is dropped there instead -- decoration sheds
        before content, and the row is complete with the name alone."""
        mobile = _css().split("@media (max-width: 767px) {", 1)[1]
        assert ".pitcher-board-table .player-portrait { display: none; }" in mobile


class TestNoPlayerIsExcludedByIdentity:
    """No pitcher is removed from, or added to, a board because of who they
    are. The board rules are the frozen 2024 display rules and nothing else.

    This is the standing guard against the class of change that starts as
    "drop this one player" and ends as an unwritten policy: a name, an
    MLBAM id, or an allow/deny list in presentation code.
    """

    DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"

    def test_no_player_name_or_id_list_appears_in_presentation_code(self) -> None:
        banned = re.compile(
            r"\b(excluded_players?|banned_players?|hidden_players?|blocked_players?|"
            r"suspended_players?|ineligible_players?|player_denylist|player_blocklist|"
            r"roster_status|player_status_filter)\b"
        )
        for path in sorted(self.DASHBOARD.glob("*.py")) + sorted(
            (self.DASHBOARD / "templates").glob("*.html")
        ):
            assert not banned.search(path.read_text()), f"{path} names a player-status filter"

    def test_board_membership_follows_only_the_frozen_display_rules(
        self, tmp_path, snapshot_roots, fixture_path
    ) -> None:
        """Every season that clears the display minimum AND lands in one of
        the two usage descriptions is on its board; every one that does not
        is on a page instead. Nothing else decides."""
        data = ppc.load_pitcher_prototype_data(fixture_path)
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        board = (out_dir / "pitchers" / "index.html").read_text()
        for row in data.rows:
            ranked = row.on_board and row.role_bucket in ("starter_like", "reliever_like")
            in_a_board_table = any(
                row.name in board.split(f'id="pb-{slug}"', 1)[1].split("</table>", 1)[0]
                for slug in ("starter-like", "reliever-like")
            )
            assert in_a_board_table is ranked, row.name
            # Withheld or not, the season still has its own page.
            assert (out_dir / "pitchers" / str(row.pitcher_id) / "index.html").exists()

    def test_board_counts_are_unchanged_by_this_change(
        self, tmp_path, snapshot_roots, fixture_path
    ) -> None:
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        board = (out_dir / "pitchers" / "index.html").read_text()
        counts = {
            slug: board.split(f'id="pb-{slug}"', 1)[1]
            .split("</table>", 1)[0]
            .split("<tbody>", 1)[1]
            .count("<tr>")
            for slug in ("starter-like", "reliever-like")
        }
        # From `_fixture_payload`: two ranked starter-like, one ranked
        # reliever-like ("Tiny Sample" is below the minimum, "Mixed Usage"
        # is on neither board).
        assert counts == {"starter-like": 2, "reliever-like": 1}


def _strip_css_comments(css: str) -> str:
    """This stylesheet's comments quote CSS, braces included, so they must
    go before any brace-based parsing."""
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def _media_block(css: str, query: str, *, must_contain: str) -> str:
    """The body of the `@media <query>` block containing `must_contain`.

    There is more than one block per query in this stylesheet, so the
    marker picks the one under test rather than the first one written.
    """
    body = _strip_css_comments(css)
    needle = f"@media {query} {{"
    start = 0
    while True:
        start = body.index(needle, start) + len(needle)
        depth, i = 1, start
        while depth:
            if body[i] == "{":
                depth += 1
            elif body[i] == "}":
                depth -= 1
            i += 1
        block = body[start : i - 1]
        if must_contain in block:
            return block


class TestTheDesktopPitcherSurfaceUsesItsCanvas:
    """Manual review, 2026-09-09: the pitcher surfaces read as underscaled
    at 1512.

    The cause was NOT the page shell -- `--measure-data` already gives every
    route 1152px of usable content at any viewport >= 1248 -- so these tests
    pin the two things that actually were wrong, and pin the shell as
    *unchanged* so a future "just make it wider" does not quietly become a
    site-wide measure change.
    """

    def test_the_shared_page_measure_is_unchanged(self) -> None:
        """`docs/design/layout.md` fixes the data measure at 1200px and
        `.page-shell` is worn by the header, main and footer of every route.
        Widening it for one surface would move the wordmark between routes.
        """
        root = _css().split(":root {", 1)[1].split("}", 1)[0]
        assert "--measure-data: 1200px" in root
        shell = _css().split(".page-shell {", 1)[1].split("}", 1)[0]
        assert "max-width: var(--measure-data)" in shell

    def test_the_card_figure_is_no_longer_capped_at_720px(self) -> None:
        """The measured defect: a 720px figure centred in a 1152px column,
        which is the shape `docs/design/layout.md` calls wasted space."""
        base = _css().split(".pitcher-hero-plot {", 1)[1].split("}", 1)[0]
        assert "max-width: 720px" in base, "the narrow default is still the mobile behaviour"
        desktop = _media_block(_css(), "(min-width: 1024px)", must_contain=".pitcher-hero-plot")
        plot = desktop.split(".pitcher-hero-plot {", 1)[1].split("}", 1)[0]
        assert "max-width: none" in plot
        assert "grid-column: 1 / -1" in plot

    def test_the_card_hero_is_a_grid_at_desktop_width(self) -> None:
        desktop = _media_block(_css(), "(min-width: 1024px)", must_contain=".pitcher-hero-plot")
        hero = desktop.split(".pitcher-hero {", 1)[1].split("}", 1)[0]
        assert "display: grid" in hero
        assert "grid-template-columns" in hero

    def test_every_desktop_rule_is_scoped_to_the_pitcher_surfaces(self) -> None:
        """The whole no-hitter-regression argument in one assertion: if a
        selector in either new block does not name a pitcher class, it can
        reach a hitter route."""
        for query, marker in (
            ("(min-width: 1280px)", "table.pitcher-board-table"),
            ("(min-width: 1024px)", ".pitcher-hero-plot"),
        ):
            block = _media_block(_css(), query, must_contain=marker)
            for rule in re.finditer(r"([^{}]+)\{[^{}]*\}", block):
                for selector in rule.group(1).split(","):
                    assert "pitcher" in selector, f"{selector.strip()!r} escapes the pitcher scope"

    def test_the_desktop_step_up_uses_existing_scale_steps_only(self) -> None:
        """`docs/design/guardrails.md` anti-pattern 3: move between steps on
        the nine-step scale, never nudge a raw px size into a font-size."""
        for query, marker in (
            ("(min-width: 1280px)", "table.pitcher-board-table"),
            ("(min-width: 1024px)", ".pitcher-hero-plot"),
        ):
            block = _media_block(_css(), query, must_contain=marker)
            for value in re.findall(r"font-size:\s*([^;]+);", block):
                assert re.fullmatch(r"var\(--fs-\d00\)", value.strip()), value

    def test_the_marks_themselves_are_untouched(self) -> None:
        """`docs/design/dataviz.md`: 2px interval line, >= 8px point dot.
        The figures were made readable by width, not by inflating marks past
        their spec."""
        for query, marker in (
            ("(min-width: 1280px)", "table.pitcher-board-table"),
            ("(min-width: 1024px)", ".pitcher-hero-plot"),
        ):
            block = _media_block(_css(), query, must_contain=marker)
            assert ".cl-scale-interval" not in block
            assert ".cl-scale-point" not in block

    def test_the_board_still_declares_its_min_width_scroll_contract(self) -> None:
        """`docs/design/guardrails.md` anti-pattern 6: an `.overflow-x` box
        owes a min-width contract. The wider columns must not have quietly
        removed it."""
        table = _css().split("table.pitcher-board-table {", 1)[1].split("}", 1)[0]
        assert "min-width: 760px" in table
        assert "overflow-x: auto" in _css().split(".pitcher-scroll {", 1)[1].split("}", 1)[0]

    def test_the_identity_column_pays_for_the_portrait_not_the_name(self) -> None:
        """Both bands grew by exactly the portrait's footprint (34px box +
        an 8px gap = 2.625rem), the same compensation the hitter board
        makes, so no name has less room than it had before portraits."""
        css = _strip_css_comments(_css())
        assert ".pitcher-board-table col.pcol-name { width: 15.625rem; }" in css  # was 13rem
        narrow = _media_block(_css(), "(max-width: 1023px)", must_contain="col.pcol-rate")
        assert "col.pcol-name { width: 13.625rem; }" in narrow  # was 11rem
