"""Global-search hotfix: tests for `dashboard/static/app.js`'s top-right
site-navigation player search matching -- case-insensitive, diacritic
-insensitive, whitespace-normalized (`normalizeSearchText`), matching the
semantics already shipped for Play Explorer's own search
(`dashboard/static/explore.js`, see `tests/test_explore_search_normalization.py`),
while leaving displayed names (`render()`'s `p.batter_name`) and player
links (`p.url`/`p.batter_id`) exactly as fetched, accents included.

Runs the ACTUAL shipped `normalizeSearchText` function body under Node (no
DOM/jsdom needed -- it's a pure string function with no `document`
reference), extracted from the real file rather than reimplemented in
Python, so these tests can never silently drift from what the browser
actually executes. Mirrors `tests/test_explore_search_normalization.py`'s
own extraction/execution approach exactly.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

APP_JS_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "app.js"


def _extract_js_function(source: str, name: str) -> str:
    """Extract `function <name>(...) { ... }` from `source` by brace
    -depth counting -- robust to reformatting, unlike a fixed-indent
    regex. Identical helper to the one already proven in
    `tests/test_explore_search_normalization.py`.
    """
    marker = f"function {name}("
    start = source.index(marker)
    brace_start = source.index("{", start)
    depth = 0
    i = brace_start
    while i < len(source):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
        i += 1
    raise AssertionError(
        f"could not find a matching closing brace for function {name}() in {source!r}"
    )


@pytest.fixture(scope="module")
def normalize_search_text() -> Callable[[str], str]:
    source = APP_JS_PATH.read_text(encoding="utf-8")
    fn_source = _extract_js_function(source, "normalizeSearchText")

    def call(text: str) -> str:
        script = (
            fn_source
            + "\nprocess.stdout.write(JSON.stringify(normalizeSearchText(process.argv[1])));"
        )
        result = subprocess.run(
            ["node", "-e", script, text],
            capture_output=True,
            text=True,
            check=True,
        )
        value = json.loads(result.stdout)
        assert isinstance(value, str)
        return value

    return call


class TestNormalizeSearchText:
    def test_jose_ramirez_ascii_query(self, normalize_search_text: Callable[[str], str]) -> None:
        assert normalize_search_text("Jose Ramirez") == "jose ramirez"

    def test_lowercase_query(self, normalize_search_text: Callable[[str], str]) -> None:
        assert normalize_search_text("jose ramirez") == "jose ramirez"

    def test_uppercase_query(self, normalize_search_text: Callable[[str], str]) -> None:
        assert normalize_search_text("JOSE RAMIREZ") == "jose ramirez"

    def test_real_accented_display_name(self, normalize_search_text: Callable[[str], str]) -> None:
        assert normalize_search_text("José Ramírez") == "jose ramirez"

    def test_extra_internal_leading_trailing_whitespace(
        self, normalize_search_text: Callable[[str], str]
    ) -> None:
        assert normalize_search_text("  Jose   Ramirez  ") == "jose ramirez"

    def test_all_five_required_variants_produce_the_identical_key(
        self, normalize_search_text: Callable[[str], str]
    ) -> None:
        variants = [
            "José Ramírez",
            "Jose Ramirez",
            "JOSE RAMIREZ",
            "jose ramirez",
            "  Jose   Ramirez",
        ]
        keys = {normalize_search_text(v) for v in variants}
        assert keys == {"jose ramirez"}, keys

    def test_empty_and_whitespace_only_input(
        self, normalize_search_text: Callable[[str], str]
    ) -> None:
        assert normalize_search_text("") == ""
        assert normalize_search_text("   ") == ""

    def test_regression_ronald_acuna_jr(self, normalize_search_text: Callable[[str], str]) -> None:
        """The second real accented name called out in the hotfix task,
        distinct from Jose Ramirez -- also carries a non-letter ". "
        suffix ("Jr."), proving the fix strips combining accent marks
        without disturbing ordinary punctuation.
        """
        assert normalize_search_text("Ronald Acuña Jr.") == "ronald acuna jr."
        assert normalize_search_text("Ronald Acuna Jr.") == "ronald acuna jr."
        assert normalize_search_text("RONALD ACUNA JR.") == "ronald acuna jr."
        assert normalize_search_text("  Ronald   Acuna   Jr.  ") == "ronald acuna jr."


class TestMatchingBehaviorEndToEnd:
    """Exercises the SAME accent-folded substring predicate
    `initGlobalPlayerSearch()`'s input listener uses
    (`normalizeSearchText(p.batter_name).indexOf(normalizeSearchText(
    input.value)) !== -1`), proving a query actually finds a catalog
    entry -- not just that two normalized strings happen to be equal.
    """

    def test_every_required_query_variant_matches_the_real_catalog_name(
        self, normalize_search_text: Callable[[str], str]
    ) -> None:
        catalog_name = "José Ramírez"
        normalized_name = normalize_search_text(catalog_name)
        for query in [
            "Jose Ramirez",
            "jose ramirez",
            "JOSE RAMIREZ",
            "José Ramírez",
            "  Jose   Ramirez  ",
            "jose",
            "ramirez",
            "RAMIREZ",
        ]:
            normalized_query = normalize_search_text(query)
            assert normalized_query in normalized_name, (query, normalized_query, normalized_name)

    def test_regression_ronald_acuna_jr_matches_as_substring(
        self, normalize_search_text: Callable[[str], str]
    ) -> None:
        normalized_name = normalize_search_text("Ronald Acuña Jr.")
        for query in ["Acuna", "acuna", "ACUNA", "Ronald Acuna", "  acuna  "]:
            assert normalize_search_text(query) in normalized_name

    def test_unrelated_query_does_not_match(
        self, normalize_search_text: Callable[[str], str]
    ) -> None:
        normalized_name = normalize_search_text("José Ramírez")
        assert normalize_search_text("Ronald Acuna") not in normalized_name

    def test_substring_search_still_works_for_partial_names(
        self, normalize_search_text: Callable[[str], str]
    ) -> None:
        normalized_name = normalize_search_text("Christian Yelich")
        assert normalize_search_text("chris") in normalized_name
        assert normalize_search_text("yelich") in normalized_name
        assert normalize_search_text("xyz") not in normalized_name


class TestPlayerIdAndLinkRetained:
    """The matching predicate and normalization only ever touch
    `p.batter_name` -- `p.url`/`p.batter_id`, which drive the rendered
    link's `href` and the numeric-ID matching path, must never be
    normalized or otherwise mutated by the search fix."""

    def test_matches_filter_only_normalizes_batter_name_not_id_or_url(self) -> None:
        """The matcher moved into the shared combobox factory in Phase 5;
        the rule it enforces is unchanged -- fold accents on the NAME only,
        and compare the id as an exact string."""
        source = APP_JS_PATH.read_text(encoding="utf-8")
        search_source = _extract_js_function(source, "search")
        assert "normalizeSearchText(item.batter_name)" in search_source
        assert "normalizeSearchText(item.url)" not in search_source
        assert "normalizeSearchText(item.batter_id)" not in search_source
        assert "String(item.batter_id) === q" in search_source

    def test_render_uses_original_url_and_id_verbatim(self) -> None:
        source = APP_JS_PATH.read_text(encoding="utf-8")
        init_source = _extract_js_function(source, "initGlobalPlayerSearch")
        assert "a.href = p.url;" in init_source
        assert "p.batter_id" in init_source
        assert "normalizeSearchText" not in init_source


class TestDisplayNameUnaffected:
    """`render()` is the only function that renders a name into the DOM --
    it must never call `normalizeSearchText`, so a displayed/autocompleted
    name always keeps its original accents (`p.batter_name` verbatim, the
    same same-snapshot presentation-name overlay `dashboard/content.py`'s
    `build_player_index` already provides)."""

    def test_render_never_normalizes_the_display_name(self) -> None:
        """`renderOption` is the only thing that writes a name into the DOM
        for this surface, and it lives in `initGlobalPlayerSearch`."""
        source = APP_JS_PATH.read_text(encoding="utf-8")
        init_source = _extract_js_function(source, "initGlobalPlayerSearch")
        assert "normalizeSearchText" not in init_source
        assert "p.batter_name" in init_source

    def test_normalize_search_text_defined_exactly_once(self) -> None:
        source = APP_JS_PATH.read_text(encoding="utf-8")
        assert source.count("function normalizeSearchText(") == 1

    def test_the_matcher_normalizes_both_the_query_and_every_candidate_name(self) -> None:
        source = APP_JS_PATH.read_text(encoding="utf-8")
        search_source = _extract_js_function(source, "search")
        assert search_source.count("normalizeSearchText(") == 2
        assert ".toLowerCase()" not in search_source

    def test_explore_shares_this_implementation_rather_than_copying_it(self) -> None:
        """Redesign Phase 5 consolidated the two player pickers onto one
        combobox, so this function exists ONCE. It used to be duplicated in
        explore.js and pinned byte-for-byte; a shared implementation is the
        stronger version of the same guarantee, and this asserts the copy
        has not crept back.
        """
        explore_js_path = APP_JS_PATH.parent / "explore.js"
        app_source = APP_JS_PATH.read_text(encoding="utf-8")
        explore_source = explore_js_path.read_text(encoding="utf-8")
        assert app_source.count("function normalizeSearchText(") == 1
        assert "function normalizeSearchText(" not in explore_source
        assert "window.ContactLuck.normalizeSearchText = normalizeSearchText" in app_source
