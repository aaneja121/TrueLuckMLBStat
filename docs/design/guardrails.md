# Design guardrails — anti-patterns and preserved invariants

The two prohibition lists. **Load this file with every dashboard change**, whatever else the
task needs. Entry point: `DESIGN.md`.

The anti-patterns are style prohibitions drawn from the measured baseline. The invariants
are *behaviour*, not styling — several are product commitments that outrank any visual
argument (`PRODUCT.md`).

---

## Anti-patterns

Explicitly prohibited. Each is drawn from the measured baseline.

1. **A decorative accent applied uniformly to every `<h2>`.** `h2::before` currently gives
   every section the same amber tick regardless of importance — decoration masquerading as
   structure.
2. **Boxing a page section** in a bordered rounded rectangle so it "looks like a section."
3. **Nudging a font size** by 0.02 rem instead of using a scale step. No new sizes.
4. **Adding a border-radius value.** Three exist; that is the set.
5. **`overflow-wrap: anywhere` on `body`** or on any ancestor of prose or player names.
   Scope it to the elements that actually hold unbreakable identifiers (`code`, reason
   codes, snapshot labels).
6. **`.overflow-x` without a `min-width` contract and a visible scroll affordance.**
7. **`display: none` on a control that must remain operable.**
8. **A placeholder used as a label.**
9. **Reusing blue or red for anything that is not the sign of a point estimate.**
10. **A third or fourth non-semantic hue.** One accent; the field green is illustration-only.
11. **SVG text below 12 CSS px** after viewBox scaling.
12. **Count-up / odometer animation on a statistic.**
13. **Gradients, glows, or drop shadows on data marks or surfaces.**
14. **A stat rendered as a bordered tile** when it belongs to a stat line.
15. **A hero section whose only job is to restate the page title** before the actual data.
16. **Emoji as UI iconography.**
17. **Any visual treatment that makes an interval crossing zero look weaker** — fading,
    muting, dashing, italics, a "not significant" marker. This is a product invariant, not a
    style preference.
18. **Ranking or leaderboarding an individual component**, or any phrasing on the
    `BANNED_PHRASES` list.

---

## Preserved product invariants

The redesign must not change any of these. They are behaviour, not styling.

- Two independent official rankings, both starting at #1, both using their frozen
  `public_labels` strings. Never "worst players."
- Official rank semantics: competition ranking on `contact_luck_runs_per_100` alone;
  interval endpoints never reorder anyone; the Rank column always restores the frozen order.
- "Sorted view — not the official Contact Luck ranking" appears on any user re-sort and
  never on the default view.
- Sign colour follows the **point estimate's sign only**, identically whether or not the
  interval crosses zero.
- Unqualified players keep a page, a score, and an interval, receive no rank, and are never
  visually de-emphasised. Their status chip is neutral, not a warning.
- Intervals ship next to every point estimate, never behind a toggle.
- **The league-wide shared interval domain**, identical on the leaderboard and every player
  page (zero at 0.463/0.464 of width), so scores stay comparable across routes.
- Explore's empty state hides the table and the result count together.
- The "Play not found" state and its route back to Explore.
- Retrospective / not-predictive framing in the hero, per-page callouts, and footer.
- Dark mode.
- The one-way "Show all 12" reveal (its trigger hides after use).
- Every existing useful interaction — sort, filter, search, tab, disclosure, simulator —
  unless a deliberately designed superior replacement is specified in this design system.
- The dashboard displays; it never computes.

Also preserve what already passes accessibility review: `lang="en"`, no duplicate ids, no
`<img>` (so no alt debt), correct landmarks, and a consistent visible focus ring.
