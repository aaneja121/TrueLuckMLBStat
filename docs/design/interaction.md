# Interaction and motion

State treatments and the motion budget. Entry point: `DESIGN.md`.
Read with `docs/design/accessibility.md` (focus, tabs, live regions).

---

## States

| State | Treatment |
|---|---|
| **Hover** | Row: recessed surface tint. Link: underline thickens. **No lift, no shadow, no scale.** |
| **Focus** | 2 px accent ring, 2 px offset, plus a contrasting outer hairline so it reads on any surface. Never removed, never replaced by a background change alone. |
| **Selected** | Tab: accent underline + `aria-selected`. Sorted column: direction glyph + `aria-sort`. Both carry a text or shape channel, never colour alone. |
| **Sorted** | The three simultaneous expressions in `docs/design/tables.md` § Rules 5, plus a polite live region announcing the new order. |
| **Loading** | A labelled placeholder in the destination's own shape. No spinners on a static site; no skeleton shimmer. |
| **Empty** | States what matched nothing and what to change. Preserve "No plays match these filters." — it hides the table *and* the result count, which is correct. |
| **Error** | Says what happened and offers the way out. Preserve the "Play not found" state and its link back to Explore. |
| **Disclosure** | `<details>`/`<summary>` for technical reason codes. One-way reveals ("Show all 12") hide their trigger after use — preserve. |

---

## Motion

Motion is allowed only where the movement *is* the information.

- **Permitted:** the `/demo/` ball-flight animation (it teaches the metric); row re-ordering
  on sort (position change is the payload); the one-way reveal of a hidden set.
- **Prohibited:** entrance/scroll-reveal animation, hover lift or scale on cards, parallax,
  skeleton shimmer, ambient background motion, and number count-ups.
- Durations 120–200 ms, ease-out. Everything above respects `prefers-reduced-motion`.
