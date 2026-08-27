"""Regression coverage for the site-wide mobile horizontal-overflow fix.

Root cause (found via real-browser viewport measurement during v1.3.0 QA):
`.component-table`'s reason-codes column and `.status-table`'s
snapshot-type column contain single long, unbroken snake_case tokens (e.g.
"calibrated_with_limited_subgroup_evidence") with no space/hyphen for the
browser's default line-breaking to use; neither table is wrapped in an
`.overflow-x` scroll container the way `table.leaderboard` already is. The
`<code>` reason codes on the methodology page hit the same problem. Auto
table layout then sized the column -- and the page -- wider than the
viewport at every width up to and including 768px.

WHY THIS TEST CHANGED (redesign Phase 1, 2026-08-26)
----------------------------------------------------
This file previously asserted the *mechanism* of the v1.3.0 fix: a single
`body { overflow-wrap: anywhere }` rule. That mechanism turned out to be
harmful, and the assertion was pinning the harm in place.

`overflow-wrap: anywhere` on `body` applies to every element on the site,
including player names, and "break at any character" is exactly what it
does when a long name meets a narrow column. The 2026-08-26 baseline audit
measured "Pete Crow-Armstro / ng" and "Kyle Schwarb / er" in the
leaderboard's 79px Player column at 390px. Breaking a person's name
mid-word is a correctness defect, and `docs/design/guardrails.md`
anti-pattern 5 bans the rule for that reason.

Coverage is NOT weakened: it is aimed one level lower, at the contract
rather than at one implementation of it. The contract is unchanged in
substance:

  1. Machine-like identifiers stay breakable (the actual overflow source).
  2. Human names never break mid-word (what the old rule got wrong).
  3. No page's scrollWidth exceeds its viewport.

(1) and (2) are asserted here. (3) remains a real-browser check -- this
repository has no browser-automation test dependency, so it is verified
with the globally configured Playwright MCP at 320/375/390/768/1400px on
the leaderboard, demo, methodology, player and status pages, in both
themes, with every `<details>` forced open.
"""

from __future__ import annotations

import re
from pathlib import Path

STYLE_CSS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "style.css"

#: The elements that actually hold unbreakable machine tokens. Each must be
#: able to break anywhere, or the v1.3.0 overflow bug returns.
IDENTIFIER_BEARING_SELECTORS = (
    "code",
    ".component-table td",
    ".status-table td",
    '[class*="snapshot-type-"]',
)


def _strip_comments(css: str) -> str:
    """Comments in this stylesheet quote CSS (including braces), so they
    must go before any brace-based parsing."""
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def _rule_for(css: str, selector: str) -> str:
    """The declaration block of the first rule whose selector list contains
    `selector` as a whole comma-separated entry."""
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", _strip_comments(css)):
        selectors = [s.strip() for s in match.group(1).split(",")]
        if selector in selectors:
            return match.group(2)
    return ""


def test_body_does_not_break_words_anywhere() -> None:
    """`body { overflow-wrap: anywhere }` must NOT come back.

    It is the rule that broke player names mid-word; see this module's
    docstring. If a future change needs document-level overflow protection,
    it must be scoped to identifier-bearing elements instead.
    """
    css = _strip_comments(STYLE_CSS.read_text())
    match = re.search(r"(?m)^body\s*\{[^}]*\}", css)
    assert match, "expected a top-level `body { ... }` rule in style.css"
    assert "overflow-wrap" not in match.group(0), (
        "`body` must not set `overflow-wrap`: it applies to every element on the site, "
        "including player names, and produced 'Pete Crow-Armstro / ng' at 390px. "
        "Scope the rule to elements holding machine tokens (see "
        "IDENTIFIER_BEARING_SELECTORS) instead."
    )


def test_identifier_bearing_elements_stay_breakable() -> None:
    """The actual overflow source must still be able to wrap anywhere.

    This is the half of the v1.3.0 contract that was correct, kept intact.
    """
    css = STYLE_CSS.read_text()
    for selector in IDENTIFIER_BEARING_SELECTORS:
        block = _rule_for(css, selector)
        assert block, f"expected a CSS rule whose selector list includes `{selector}`"
        assert "overflow-wrap: anywhere" in block, (
            f"`{selector}` holds long unbroken snake_case tokens (e.g. "
            "'calibrated_with_limited_subgroup_evidence') and must stay breakable, or "
            "auto table layout sizes the column -- and the page -- wider than the "
            "viewport at every width up to 768px."
        )


def test_player_names_are_protected_from_mid_word_breaks() -> None:
    """Identity text wraps between words or not at all.

    Declared explicitly rather than left to the initial value, so that
    re-introducing a permissive ancestor rule cannot silently break names
    again -- which is precisely how the original defect arose.
    """
    css = STYLE_CSS.read_text()
    block = _rule_for(css, ".player-link")
    assert block, "expected a CSS rule whose selector list includes `.player-link`"
    assert "overflow-wrap: normal" in block, (
        "player names must be protected from mid-word breaking; see "
        "docs/design/guardrails.md anti-pattern 5."
    )
    assert "word-break: normal" in block
