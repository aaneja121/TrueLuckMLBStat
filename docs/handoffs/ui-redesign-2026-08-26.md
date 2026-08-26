# Handoff — UI redesign, 2026-08-26

## PROJECT

Contact Luck (https://contactluck.com) — MLB batted-ball metric. Research pipeline in
`src/mlb_luck_score/`; static display-only dashboard in `dashboard/`.
Routers: `CLAUDE.md` → `PRODUCT.md`, `CONTEXT.md`, `ARCHITECTURE.md`, `RESEARCH_RULES.md`.

## CURRENT BRANCH

`ui-redesign-v2` — based on the cream/amber live-site design. The dark broadcast/Statcast
redesign on `backup-pre-skill-redesign` is **not** the baseline and must not be restored.

## CURRENT OBJECTIVE

Redesign the dashboard's visual and product design. Phases 1 (baseline capture) and 2
(design audit + design system) are done. Phase 3 (design directions) has **not** started.

## COMPLETED WORK

1. **Baseline audit** — rendered and inspected all routes at 1440 / 1180 / 390 via the
   global Playwright MCP; interactions, edge cases, console/network, a11y, contrast.
2. **Product + visual design audit** — classified every defect as design-system /
   information-architecture / component / responsive / accessibility / isolated bug.
3. **`DESIGN.md`** written as the canonical design source of truth, and conceptually
   tested against 9 pages, which refined 5 of its rules.

## FILES CREATED/CHANGED

- `DESIGN.md` (repo root) — **625 lines / ~36 KB. Too large; see EXACT NEXT ACTION.**
- `outputs/figures/ui_baseline_2026-08-26/` — 26 PNGs + `README.md` (gitignored).
- `docs/handoffs/ui-redesign-2026-08-26.md` — this file.

**No application code was modified.** No CSS, HTML/templates, JS, or assets. Verify with
`git status` before doing anything else.

## CURRENT DESIGN THESIS

A scorekeeper's sheet that a statistician has audited: a ruled, fixed-column page where
every number is entered against a **shared zero line**, and every claim carries its
uncertainty and provenance in the same breath. Full text: `DESIGN.md` § Visual thesis.

Reference world (three, one borrowing each): baseball scorebook, statistical abstract,
measuring instrument. Prohibited: Baseball Savant, ESPN, team colours, novelty baseball.

## KEY DESIGN DECISIONS

Summarised only — the reasoning and measured values live in `DESIGN.md`.

1. Grotesk for everything measured; **serif for everything argued** (deliberate inversion).
2. **Two grades per semantic colour** — text ≥4.5:1, mark ≥3:1. Root cause of all contrast
   failures: a chart palette was adopted as a UI palette.
3. Amber accent means **"the frozen official record"**; removed on user re-sort.
4. **The verdict cell is one cell** — point estimate + interval on the shared zero axis.
5. **Three admissible cards**; everything else gets rules and space. Card sets cap at six.
6. **Four column tiers with declared shedding order** (Identity/Verdict never shed).
7. **Format the data rather than engineering around it.**
8. Three radii, a 9-step type scale, an 8 px spacing scale, two measures per page.

## BASELINE AUDIT LOCATION

`outputs/figures/ui_baseline_2026-08-26/` — its `README.md` carries the build identity,
per-file route/viewport index, and the measured baseline facts. Gitignored
(`.gitignore:21`), so it never becomes tracked content.

Build under audit: dashboard v1.4.1, snapshot `2026-08-14`, commit `05dbf24`.
Serve with `cd dashboard/dist && python3 -m http.server 8000` (a pre-existing server may
already hold port 8000 — check before starting one).

## IMPORTANT PRODUCT INVARIANTS

Full list: `DESIGN.md` § Preserved product invariants. The ones most easily broken by a
redesign:

- Sign colour follows the **point estimate's sign only** — an interval crossing zero is
  rendered identically to one that does not. No fading, muting, or "not significant" cue.
- The **league-wide shared interval domain already works** (zero at 0.463/0.464 of width on
  both the leaderboard and every player page). Do not introduce per-chart auto-scaling.
- Two independent official rankings, both starting at #1, both using their frozen
  `public_labels` strings. Never "worst players."
- "Sorted view — not the official Contact Luck ranking" on any re-sort, never by default.
- Unqualified players keep a page, score and interval; no rank; never de-emphasised.
- The dashboard displays; it never computes.

## KNOWN UI/ACCESSIBILITY DEFECTS

Do not re-derive these. Evidence and measurements are in the baseline `README.md` and
`DESIGN.md`.

- `/status/` snapshot-history table: last two columns 31 px wide, one character per line,
  682 px rows, 8363 px page. Reproduces at all three widths.
- Leaderboard "Interval width" clipped at **every** desktop width (1158 px table in a
  1100 px box). Root cause: `.overflow-x` with no `min-width` contract.
- Mobile player names break mid-word — global `overflow-wrap: anywhere` on `body`.
- Trend chart labels render at ~5.5 CSS px on mobile (640×220 viewBox at 350×120).
- Leaderboard tabs are `display:none` radios — absent from the a11y tree and tab order.
- Sortable `<th>` have no `role`, `tabindex`, or `aria-sort`.
- Light-theme contrast fails on four tokens (dark theme passes everywhere).
- No skip link; leaderboard filter inputs unlabelled; search lacks combobox semantics;
  play-page field diagram unlabelled; nav targets 90×25 px.
- First leaderboard row below the fold at every width (1 row at 1440×900, 0 at 1180×820).
- No player → Explore link, and Explore's selection is not URL-addressable.
- No favicon declared anywhere → `/favicon.ico` 404 on first load.

## UNRESOLVED DECISIONS

1. **Player imagery / team context.** Team, position and headshots are absent from the data
   model. Adding MLB's headshot CDN would be the site's first external request. **Resolve
   as permitted-but-optional** — see EXACT NEXT ACTION.
2. **Web-font budget.** Two self-hosted families, ~60–80 KB. The grotesk/serif split works
   with system faces if declined. Owner has not ruled.

## DO NOT REPEAT

- Do not re-run the browser reconnaissance. Use the saved screenshots and measurements.
- Do not re-derive the defect list, the contrast numbers, or the CSS inventory.
- Do not restore the `backup-pre-skill-redesign` dark design.
- Do not run training, prospective scoring, snapshot generation, publishing, deployment,
  frozen-input sync, or production explorer artifact generation.
- Playwright note: the headed Chrome periodically stops compositing (screenshots time out
  while `evaluate` still works). Recover by cycling `Browser.setWindowBounds`
  minimized → normal, or by closing and reopening the page.

## EXACT NEXT ACTION

The next session must, in order:

1. **Refactor the oversized `DESIGN.md` using progressive disclosure.** 625 lines is too
   large to load for routine frontend work. Reduce the root file to a short, high-signal
   entry point — thesis, principles, and a table of pointers — and move the detailed
   sections into focused companion files (e.g. `docs/design/typography.md`,
   `docs/design/color.md`, `docs/design/tables.md`, `docs/design/responsive.md`,
   `docs/design/accessibility.md`). Layer it so a reader loads only what the task needs.
2. **Preserve the useful content.** This is a reorganisation, not a rewrite. Every measured
   value, computed hex, contrast ratio, anti-pattern and invariant must survive with its
   evidence intact. Do not re-litigate decisions or drop rationale.
3. **Resolve player imagery as permitted-but-optional.** Change it from a blocking open
   question into a stated design rule: identity is name-first and no layout may depend on
   imagery; headshots and team context are permitted if and when the data model supports
   them, and must degrade cleanly to name-only. Remove it from unresolved decisions.
4. **Update stale pointers.** `CLAUDE.md` says "`DESIGN.md` (once it exists)";
   `ARCHITECTURE.md` says "A `DESIGN.md` will be added by the redesign phase." Both are now
   false. Keep `AGENTS.md` in sync with `CLAUDE.md`, per `CLAUDE.md`'s routing rule.
5. **Verify no application code changed.** Run `git status` and confirm the diff touches
   only Markdown — no CSS, HTML/templates, JS, or visual assets.
6. **Then stop.** Do not propose the three redesign directions, do not fix any defect, and
   do not modify the interface.
