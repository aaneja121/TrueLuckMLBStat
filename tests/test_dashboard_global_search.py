"""The global player search, across BOTH published surfaces.

The search index is the one place hitters and pitchers meet, so it is the
one place a whole class of mistake can happen quietly: publishing a pitcher
season nobody authorized, routing a pitcher result at a season-less URL, or
silently collapsing a person who holds both a hitter page and a pitcher page
into whichever one happened to sort first.

Everything asserted here is decided at BUILD TIME and rendered into
`#player-index-data`. That is deliberate (`CLAUDE.md` rule 6): what a result
is, what it is called and where it points are product facts, and none of
them is computed in JavaScript. The browser-side matcher, keyboard contract
and accent folding are covered by
`tests/test_global_search_normalization.py`; what this file pins is the
index those behaviours operate on.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import build as dashboard_build
import pytest

from test_dashboard_pitcher_season import (
    _build,
    _fixture_payload,
    fixture_path,
    snapshot_roots,
)

__all__ = ["fixture_path", "snapshot_roots"]  # re-exported fixtures

APP_JS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "app.js"


def _index(out_dir: Path) -> list[dict]:
    html = (out_dir / "index.html").read_text()
    raw = re.search(r'id="player-index-data">(.*?)</script>', html, re.S).group(1)
    return json.loads(raw)


def _of_kind(entries: list[dict], kind: str) -> list[dict]:
    return [e for e in entries if e["kind"] == kind]


class TestPitchersAreFindable:
    def test_every_published_pitcher_is_in_the_index(self, tmp_path, snapshot_roots, fixture_path):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        pitchers = _of_kind(_index(out_dir), "pitcher")
        expected = {p["pitcher_id"] for p in _fixture_payload()["pitchers"]}
        assert {e["mlbam_id"] for e in pitchers} == expected

    def test_a_pitcher_is_findable_by_name(self, tmp_path, snapshot_roots, fixture_path):
        entries = _index(_build(tmp_path, snapshot_roots, fixture_path))
        match = [e for e in entries if e["name"] == "Big Total" and e["kind"] == "pitcher"]
        assert len(match) == 1

    def test_a_pitcher_is_findable_by_mlbam_id(self, tmp_path, snapshot_roots, fixture_path):
        """The id path is an exact string compare in the browser, so what has
        to be true here is that the id is present and is the MLBAM one."""
        entries = _index(_build(tmp_path, snapshot_roots, fixture_path))
        match = [e for e in entries if e["mlbam_id"] == 1 and e["kind"] == "pitcher"]
        assert len(match) == 1
        assert match[0]["url"].endswith("/pitchers/2024/1/")

    def test_a_withheld_season_is_still_findable(self, tmp_path, snapshot_roots, fixture_path):
        """A pitcher-season the boards withhold keeps a page, so it must keep
        a way in. Search is the only route to it besides the off-board index."""
        entries = _index(_build(tmp_path, snapshot_roots, fixture_path))
        # "Tiny Sample" is below the display minimum and on no board.
        assert any(e["name"] == "Tiny Sample" and e["kind"] == "pitcher" for e in entries)


class TestPitcherResultsRouteCorrectly:
    def test_every_pitcher_url_is_season_scoped(self, tmp_path, snapshot_roots, fixture_path):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        for entry in _of_kind(_index(out_dir), "pitcher"):
            assert re.fullmatch(r"/pitchers/\d{4}/\d+/", entry["url"]), entry

    def test_every_pitcher_url_resolves_to_a_built_page(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        """A search result that 404s is worse than one that does not exist."""
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        for entry in _of_kind(_index(out_dir), "pitcher"):
            assert (out_dir / entry["url"].strip("/") / "index.html").exists(), entry

    def test_hitter_urls_are_unchanged(self, tmp_path, snapshot_roots, fixture_path):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        for entry in _of_kind(_index(out_dir), "hitter"):
            assert re.fullmatch(r"/players/\d+/", entry["url"]), entry


class TestOnlyAuthorizedSeasonsAreSearchable:
    """The index is derived from the datasets the publication gate already
    approved, so an unauthorized season cannot reach it by any route. These
    tests exist because "derived from" is an implementation detail that a
    later refactor could quietly replace with a second season list.
    """

    def test_only_authorized_seasons_appear(self, tmp_path, snapshot_roots, fixture_path):
        from dashboard_config import PITCHER_PUBLIC_SEASONS

        entries = _index(_build(tmp_path, snapshot_roots, fixture_path))
        seasons = {e["season"] for e in _of_kind(entries, "pitcher")}
        assert seasons <= set(PITCHER_PUBLIC_SEASONS)
        assert seasons == {2024}

    @pytest.mark.parametrize("season", [2025, 2026])
    def test_a_sealed_or_unopened_season_never_reaches_the_index(
        self, season, tmp_path, snapshot_roots
    ):
        """Synthetic payload with a `season` field -- no 2025 or 2026 data is
        read. The build refuses before an index could be built at all."""
        payload = _fixture_payload()
        payload["season"] = season
        path = tmp_path / f"fixture_{season}.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(dashboard_build.DashboardBuildError):
            _build(tmp_path, snapshot_roots, path)

    def test_no_index_entry_names_a_forbidden_season(self, tmp_path, snapshot_roots, fixture_path):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        blob = json.dumps(_index(out_dir))
        for forbidden in ("2025", "2026"):
            assert forbidden not in blob, forbidden

    def test_a_build_with_no_pitcher_surface_indexes_no_pitchers(self, tmp_path, snapshot_roots):
        entries = _index(_build(tmp_path, snapshot_roots, None))
        assert _of_kind(entries, "pitcher") == []
        assert _of_kind(entries, "hitter")


class TestDualIdentityIsNeverResolvedSilently:
    """A position player who pitched holds both a hitter page and a pitcher
    page under one MLBAM id. Both are real; neither is the answer.
    """

    def _dual_site(self, tmp_path, snapshot_roots):
        """A fixture whose pitcher id 1 collides with a real hitter id."""
        payload = _fixture_payload()
        payload["pitchers"][0]["pitcher_id"] = 1
        payload["pitchers"][0]["name"] = "Alice Alpha"  # the snapshot's hitter #1
        path = tmp_path / "dual_fixture.json"
        path.write_text(json.dumps(payload))
        return _build(tmp_path, snapshot_roots, path)

    def test_both_surfaces_appear_as_separate_results(self, tmp_path, snapshot_roots):
        entries = _index(self._dual_site(tmp_path, snapshot_roots))
        both = [e for e in entries if e["mlbam_id"] == 1]
        assert {e["kind"] for e in both} == {"hitter", "pitcher"}
        assert len(both) == 2

    def test_the_two_results_point_at_different_pages(self, tmp_path, snapshot_roots):
        entries = _index(self._dual_site(tmp_path, snapshot_roots))
        urls = {e["kind"]: e["url"] for e in entries if e["mlbam_id"] == 1}
        assert urls["hitter"] == "/players/1/"
        assert urls["pitcher"] == "/pitchers/2024/1/"

    def test_each_result_says_which_surface_it_is(self, tmp_path, snapshot_roots):
        entries = _index(self._dual_site(tmp_path, snapshot_roots))
        labels = {e["kind"]: e["label"] for e in entries if e["mlbam_id"] == 1}
        assert labels["hitter"] == "Hitter"
        assert labels["pitcher"] == "Pitcher · 2024"

    def test_the_hitter_result_sorts_first(self, tmp_path, snapshot_roots):
        """Deterministic, and the hitter surface is the site's primary one --
        but BOTH are present, which is the point."""
        entries = _index(self._dual_site(tmp_path, snapshot_roots))
        both = [e for e in entries if e["mlbam_id"] == 1]
        assert [e["kind"] for e in both] == ["hitter", "pitcher"]


class TestEveryResultDeclaresItsSurface:
    def test_no_entry_is_unlabelled(self, tmp_path, snapshot_roots, fixture_path):
        for entry in _index(_build(tmp_path, snapshot_roots, fixture_path)):
            assert entry["label"], entry
            assert entry["kind"] in {"hitter", "pitcher"}, entry

    def test_a_pitcher_label_carries_its_season(self, tmp_path, snapshot_roots, fixture_path):
        """Unlabelled "Pitcher" would go ambiguous the day a second season is
        published; the season is part of a pitcher-season's identity."""
        entries = _index(_build(tmp_path, snapshot_roots, fixture_path))
        for entry in _of_kind(entries, "pitcher"):
            assert entry["label"] == f"Pitcher · {entry['season']}"

    def test_the_label_is_resolved_at_build_time_not_in_javascript(self):
        """The renderer places the label; it must not compose one. A `kind`
        string turned into prose in JS is a product decision made in the
        wrong half of the repository."""
        js = APP_JS.read_text()
        init = js.split("function initGlobalPlayerSearch()", 1)[1].split("\n  }\n", 1)[0]
        assert "p.label" in init
        assert '"Pitcher' not in init
        assert '"Hitter' not in init


class TestHitterSearchBehaviourIsPreserved:
    def test_every_hitter_in_the_snapshot_is_still_indexed(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        out_dir = _build(tmp_path, snapshot_roots, fixture_path)
        hitters = _of_kind(_index(out_dir), "hitter")
        assert {e["mlbam_id"] for e in hitters} == {1, 2}

    def test_unqualified_hitters_are_still_marked_not_suppressed(
        self, tmp_path, snapshot_roots, fixture_path
    ):
        """A preserved product invariant: an unranked hitter keeps a page,
        stays findable, and is MARKED rather than hidden."""
        entries = _index(_build(tmp_path, snapshot_roots, fixture_path))
        hitters = _of_kind(entries, "hitter")
        assert all("ranked" in e for e in hitters)

    def test_pitcher_entries_carry_no_rank_semantics(self, tmp_path, snapshot_roots, fixture_path):
        """`ranked` is a hitter qualification concept. A pitcher-season has a
        board position, which is not the same thing and is deliberately not
        shown on its card -- so it must not leak into search either."""
        entries = _index(_build(tmp_path, snapshot_roots, fixture_path))
        for entry in _of_kind(entries, "pitcher"):
            assert "ranked" not in entry, entry
