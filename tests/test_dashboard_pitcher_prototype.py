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
