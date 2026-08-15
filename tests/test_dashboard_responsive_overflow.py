"""Contact Luck v1.3.0 QA follow-up: regression coverage for the site-wide
mobile horizontal-overflow fix.

Root cause (found via real-browser viewport measurement, not screenshot
inspection -- see the task that produced this fix): `.component-table`'s
reason-codes column and `.status-table`'s snapshot-type column contain
single long, unbroken snake_case tokens (e.g.
"calibrated_with_limited_subgroup_evidence") with no space/hyphen for the
browser's default line-breaking to use; neither table is wrapped in an
`.overflow-x` scroll container the way `table.leaderboard` already is. The
`<code>` reason codes on the methodology page hit the same problem. The fix
is a single shared `overflow-wrap: anywhere` rule on `body` in
`dashboard/static/style.css` -- this test only confirms that rule stays in
place; the actual cross-page/cross-width verification (no page's
`document.documentElement.scrollWidth` exceeds its viewport width, at
320/375/390/768/1400px, on the leaderboard, demo, methodology, player, and
status pages, in both light and dark mode, with every `<details>` element
forced open) was done with a real browser (Playwright/Chromium) during
manual QA -- this repository has no browser-automation test dependency, so
that check is not re-run automatically here.
"""

from __future__ import annotations

import re
from pathlib import Path

STYLE_CSS = Path(__file__).resolve().parents[1] / "dashboard" / "static" / "style.css"


def test_body_rule_sets_overflow_wrap_anywhere() -> None:
    css = STYLE_CSS.read_text()
    match = re.search(r"\bbody\s*\{[^}]*\}", css, re.DOTALL)
    assert match, "expected a top-level `body { ... }` rule in style.css"
    assert "overflow-wrap: anywhere" in match.group(0), (
        "the body-level `overflow-wrap: anywhere` rule is the root-cause fix for the "
        "site-wide mobile overflow bug (long unbroken tokens in table cells / <code> "
        "reason codes forcing the page wider than the viewport) -- do not remove it "
        "without an equivalent replacement, and re-verify with a real browser at "
        "320/375/390/768px on the leaderboard, demo, methodology, player, and status pages."
    )
