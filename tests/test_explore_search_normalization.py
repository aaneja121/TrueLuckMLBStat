"""Version 1.4.0 search hotfix: tests for `dashboard/static/explore.js`'s
Play Explorer player-name search matching -- case-insensitive, diacritic
-insensitive, whitespace-normalized (`normalizeSearchText`), while leaving
displayed names (`playerLabel`) exactly as fetched, accents included.

Runs the ACTUAL shipped `normalizeSearchText` function body under Node (no
DOM/jsdom needed -- it's a pure string function with no `document`
reference), extracted from the real file rather than reimplemented in
Python, so these tests can never silently drift from what the browser
actually executes. Node is a system dependency already required for this
repo's frontend tooling; this adds no new dependency.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

EXPLORE_JS_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "explore.js"


def _extract_js_function(source: str, name: str) -> str:
    """Extract `function <name>(...) { ... }` from `source` by brace
    -depth counting -- robust to reformatting, unlike a fixed-indent
    regex.
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
    source = EXPLORE_JS_PATH.read_text(encoding="utf-8")
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
        """A second real accented name from the production player catalog
        (2026-08-16 snapshot), distinct from Jose/Jose Ramirez -- also
        carries a non-letter ". " suffix ("Jr."), proving the fix strips
        combining accent marks without disturbing ordinary punctuation.
        """
        assert normalize_search_text("Ronald Acuña Jr.") == "ronald acuna jr."
        assert normalize_search_text("ronald acuna jr.") == "ronald acuna jr."
        assert normalize_search_text("RONALD ACUNA JR.") == "ronald acuna jr."
        assert normalize_search_text("  Ronald   Acuna   Jr.  ") == "ronald acuna jr."


class TestMatchingBehaviorEndToEnd:
    """Exercises the SAME accent-folded substring predicate
    `onSearchInput()` uses (`normalizeSearchText(name).indexOf(
    normalizeSearchText(query)) !== -1`), proving a query actually finds an
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


class TestDisplayNameUnaffected:
    """`playerLabel()` is the only function that renders a name into the
    DOM -- it must never call `normalizeSearchText`, so a displayed name
    always keeps its original accents (`entry.batter_name` verbatim, per
    `demo/build_play_explorer_fixture.py`'s presentation-overlay
    contract)."""

    def test_player_label_never_normalizes_the_display_name(self) -> None:
        source = EXPLORE_JS_PATH.read_text(encoding="utf-8")
        player_label_source = _extract_js_function(source, "playerLabel")
        assert "normalizeSearchText" not in player_label_source
        assert "entry.batter_name" in player_label_source

    def test_normalize_search_text_defined_exactly_once(self) -> None:
        source = EXPLORE_JS_PATH.read_text(encoding="utf-8")
        assert source.count("function normalizeSearchText(") == 1

    def test_on_search_input_normalizes_both_the_query_and_every_candidate_name(self) -> None:
        source = EXPLORE_JS_PATH.read_text(encoding="utf-8")
        on_search_input_source = _extract_js_function(source, "onSearchInput")
        assert on_search_input_source.count("normalizeSearchText(") == 2
        assert ".toLowerCase()" not in on_search_input_source
