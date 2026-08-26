# Accessibility

Hard requirements, satisfied by the primitives rather than by a later pass. Entry point:
`DESIGN.md`. Read with `docs/design/color.md` (the two contrast grades),
`docs/design/interaction.md` (focus) and `docs/design/tables.md` (tabs, sortable headers).

1. **Contrast.** Text ≥ 4.5:1 against **its own** surface — page, card, *and* recessed
   control surfaces are all checked, because the recessed plane is the worst case. Marks
   ≥ 3:1. Two grades per semantic colour (see `docs/design/color.md`). SVG axis labels are
   text and are held to the text grade.
2. **Non-colour semantics.** Every Contact Luck value carries an explicit sign glyph; every
   headline value also carries the word Favorable / Unfavorable.
3. **Tabs are buttons.** `role="tablist"` / `role="tab"` / `aria-selected`, keyboard
   operable. The `display:none` radio + `<label>` pattern is banned — it removes the
   leaderboard's ranking switch from both the accessibility tree and the tab order.
4. **Sortable headers are buttons.** `<th aria-sort><button>`; sort changes announced via a
   polite live region.
5. **Every input has a programmatic label.** `<label>` or `aria-label`. A placeholder is
   never a label. Two identical unlabelled filter inputs currently ship on the leaderboard.
6. **Combobox semantics** as specified in `docs/design/information-architecture.md`
   § Navigation and search.
7. **Skip link** first in the tab order on every page.
8. **Focus** always visible, per `docs/design/interaction.md`.
9. **Touch targets ≥ 44 × 44** for anything tapped. Nav links are 90 × 25 px today.
10. **Meaningful SVG is labelled;** decorative SVG is `aria-hidden`.
11. **One `<h1>` per page**, no skipped heading levels, landmarks on every route.
12. `prefers-reduced-motion: reduce` disables all non-essential motion.

Preserve what already passes: `lang="en"`, no duplicate ids, no `<img>` (so no alt debt),
correct landmarks, and a consistent visible focus ring.

The measured baseline failures behind each of these are recorded in
`outputs/figures/ui_baseline_2026-08-26/README.md` and summarised in
`docs/handoffs/ui-redesign-2026-08-26.md` § Known UI/accessibility defects. Do not
re-derive them.
