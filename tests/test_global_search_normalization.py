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
        source = APP_JS_PATH.read_text(encoding="utf-8")
        init_source = _extract_js_function(source, "initGlobalPlayerSearch")
        assert "normalizeSearchText(p.batter_name)" in init_source
        assert "normalizeSearchText(p.url)" not in init_source
        assert "normalizeSearchText(p.batter_id)" not in init_source
        assert "String(p.batter_id) === q" in init_source

    def test_render_uses_original_url_and_id_verbatim(self) -> None:
        source = APP_JS_PATH.read_text(encoding="utf-8")
        init_source = _extract_js_function(source, "initGlobalPlayerSearch")
        render_source = _extract_js_function(init_source, "render")
        assert "a.href = p.url;" in render_source
        assert "p.batter_id" in render_source
        assert "normalizeSearchText" not in render_source


class TestDisplayNameUnaffected:
    """`render()` is the only function that renders a name into the DOM --
    it must never call `normalizeSearchText`, so a displayed/autocompleted
    name always keeps its original accents (`p.batter_name` verbatim, the
    same same-snapshot presentation-name overlay `dashboard/content.py`'s
    `build_player_index` already provides)."""

    def test_render_never_normalizes_the_display_name(self) -> None:
        source = APP_JS_PATH.read_text(encoding="utf-8")
        init_source = _extract_js_function(source, "initGlobalPlayerSearch")
        render_source = _extract_js_function(init_source, "render")
        assert "normalizeSearchText" not in render_source
        assert "p.batter_name" in render_source

    def test_normalize_search_text_defined_exactly_once(self) -> None:
        source = APP_JS_PATH.read_text(encoding="utf-8")
        assert source.count("function normalizeSearchText(") == 1

    def test_global_search_input_listener_normalizes_both_the_query_and_every_candidate_name(
        self,
    ) -> None:
        source = APP_JS_PATH.read_text(encoding="utf-8")
        init_source = _extract_js_function(source, "initGlobalPlayerSearch")
        assert init_source.count("normalizeSearchText(") == 2
        assert ".toLowerCase()" not in init_source

    def test_normalize_search_text_matches_explore_js_semantics_byte_for_byte(self) -> None:
        """The two independent, unbundled <script> files intentionally
        duplicate this pure function (see app.js's own docstring comment
        on why) -- this pins them to stay identical rather than silently
        drifting apart on a future edit to only one of the two files.
        """
        explore_js_path = APP_JS_PATH.parent / "explore.js"
        app_source = APP_JS_PATH.read_text(encoding="utf-8")
        explore_source = explore_js_path.read_text(encoding="utf-8")
        app_fn = _extract_js_function(app_source, "normalizeSearchText")
        explore_fn = _extract_js_function(explore_source, "normalizeSearchText")
        assert app_fn == explore_fn, (app_fn, explore_fn)
