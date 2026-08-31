# Phase 3 Zero Spine Leaderboard Handoff

PROJECT
Contact Luck / TrueLuckMLBStat

BRANCH
ui-redesign-v2

TASK
Phase 3 — Homepage + Leaderboard Zero Spine redesign.

CURRENT STATE
Phase 1 and Phase 2 are complete and committed.
Phase 3 implementation is complete and verified but final design cleanup is not complete.
Phase 4 has NOT started.

PHASE 3 IMPLEMENTED
- Homepage hierarchy rebuilt.
- Leaderboard rebuilt as Rank · Player · Verdict · Sample.
- Per-row SVGs removed from leaderboard.
- CSS-positioned Zero Spine marks implemented.
- Axis ticks are HTML text at readable size.
- Verdict combines exact value, plot field, and interval.
- Official favorable/unfavorable rankings preserved.
- Accessible tabs implemented.
- Accessible sortable headers implemented with aria-sort.
- Sorted-view state preserved.
- Filters labelled/accessibly implemented.
- Mobile leaderboard structurally transforms instead of shrinking desktop table.
- Two value-placement variants V1/V2 currently exist behind development-only query/localStorage switching.

VERIFICATION
make check:
1906 passed, 8 skipped.

Zero registration at 1440:
axis zero tick = 628.45
distribution strip = 628.45
all 124 row fields = 628.45
one unique x position.

Density:
13 rows above fold at 1440x900.
Row height approximately 36.5px.

No horizontal overflow in reported width/theme sweep.

IMPORTANT DESIGN DECISIONS ALREADY MADE

1. VALUE PLACEMENT
Choose V1 permanently.

V1 = one fixed-width consistently aligned exact-value column.

Reason:
V2 looks attractive when rows are mostly one sign, but becomes difficult to scan when sorting by BBE or another mixed-sign field. The Zero Spine should provide visual/distributional reading; the fixed numeral column should provide exact-value scanning.

Required cleanup:
- remove V2 development switching
- remove V2-only CSS/JS/state
- retain documentation explaining why V2 was rejected

2. MOBILE ZERO SPINE
Accept segmented mobile zero registration.

The mobile invariant is:
every VERDICT field registers zero at the same x-position.

The mobile invariant is NOT:
the amber line must remain physically continuous through player identity/evidence rows.

Do not draw the spine through player names just to preserve literal continuity.

3. DISTRIBUTION STRIP
NO FINAL DECISION YET.

Keep it unchanged until human screenshot review.
Do not remove or substantially redesign it yet.

It currently uses only about 40% of the full canonical field because the league distribution is concentrated inside the wider canonical domain.

SCREENSHOTS TO REVIEW
- p3-1440-nearzero.png
- p3-1440-v1-fav.png
- p3-1440-v1-sortbbe.png
- p3-390-v1-fav.png

Other useful captures:
- p3-1440-v2-sortbbe.png
- p3-1440-dark.png
- p3-1440-worked.png

OPEN DESIGN CONCERN
Mobile homepage preamble remains tall:
tbody starts around 574px at 390px width, leaving roughly 3 leaderboard rows initially visible.

Do not change this until screenshot review unless explicitly instructed.

FILES MODIFIED DURING PHASE 3
- dashboard/build.py
- dashboard/templates/_macros.html
- dashboard/templates/index.html
- dashboard/static/style.css
- dashboard/static/app.js
- tests/test_dashboard_leaderboard.py
- tests/test_dashboard_build.py
- tests/test_dashboard_shell.py

DO NOT REPEAT
- repo reconnaissance
- baseline UI audit
- design audit
- Zero Spine direction exploration
- Phase 1 design-foundation reasoning
- Phase 2 shell analysis
- V1/V2 investigation

DO NOT BEGIN
Phase 4 player-page redesign.

EXACT NEXT ACTION
1. Read project instructions and this handoff.
2. Inspect current git status/diff only; do not redo prior audits.
3. Apply the approved V1 cleanup:
   - remove V2 development switching
   - remove V2-only CSS/JS/state
   - document the V1 decision
4. Document segmented mobile zero registration as intentional.
5. Leave the distribution strip unchanged pending human screenshot review.
6. Run targeted leaderboard tests.
7. Run full make check if reasonable.
8. Verify V1 desktop/mobile after cleanup.
9. STOP before Phase 4.
