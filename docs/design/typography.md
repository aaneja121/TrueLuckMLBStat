# Typography

Type families, the size scale, and numeric setting. Entry point: `DESIGN.md`.
Read with `docs/design/color.md` (ink tokens) and `docs/design/tables.md` (where the
numeric rules are exercised hardest).

---

## Roles

Two families, split along the product's own fault line: **grotesk for everything measured,
serif for everything argued.** The serif is not a display face — it carries the careful,
hedged prose the product cannot afford to have skimmed ("retrospective, not a projection").
This inverts the usual serif-display/sans-body pairing on purpose.

| Role | Family | Notes |
|---|---|---|
| Brand / page title | Grotesk, condensed cut, 600 | Sparing: wordmark and `h1` only |
| Column heads, eyebrows, status codes | Grotesk, condensed, 600, uppercase, +0.06em | Labels a measured quantity |
| UI / navigation / controls | Grotesk, 400–500 | |
| **Numbers and tables** | Grotesk, tabular lining figures | The product's primary typographic surface |
| Body / explanation / methodology | **Serif**, 400 | Prose measure, comfortable at length |
| Metadata, captions, footnotes | Grotesk, 400, `--ink-muted` | Never below 12 px |
| Identifiers (`play_id`, reason codes, hashes) | Mono | Identifiers only — **never statistics** |

## Font budget — open

Recommended: **IBM Plex Sans / Plex Sans Condensed / Plex Mono** (true tabular figures,
instrument register, open licence, not Inter) with **Source Serif 4** for prose. Self-hosted
`woff2` subsets. The site currently loads **zero** web fonts and makes zero external
requests; budget two families at ~60–80 KB total, and ship a system fallback stack that
preserves the grotesk/serif split (`ui-sans-serif` / `ui-serif`).

**The owner has not ruled on this budget.** It is not blocking: if the web fonts are
declined, the grotesk/serif split still holds using system faces, and every other rule in
this file is unaffected.

---

## Scale

Replaces ~26 ad-hoc sizes, of which five sat inside a 0.06 rem band (0.82/0.84/0.85/0.86/
0.88 rem) — differences no reader perceives and every alignment breaks on.

| Token | px | Use |
|---|---|---|
| `--fs-100` | 11 | Column heads, status codes (uppercase only) |
| `--fs-200` | 12 | Metadata, captions, footnotes — **floor for any text** |
| `--fs-300` | 14 | Table body, UI labels |
| `--fs-400` | 16 | Body prose |
| `--fs-500` | 18 | Lead prose, emphasised table figures |
| `--fs-600` | 22 | Section heads |
| `--fs-700` | 30 | Page title |
| `--fs-800` | 44 | Player hero score |
| `--fs-900` | 64 | Reserved: the single largest number on a page |

No step may be added between two existing steps. Mobile: `--fs-700`/`800`/`900` step down
one; `--fs-400` never drops below 16 px; `--fs-100` never drops below 11 px.

Mobile `h1` is capped at **24 px**, not 26. Measured: "Christian Encarnacion-Strand" is
404 px at 30 px/600 and 350 px at 26 px/600 against 350 px of available width at 390 px —
26 px lands exactly on the limit with no margin for a wider face or a longer future name.

Line height: 1.15 display, 1.3 tables, 1.6 prose. Letter-spacing: `-0.01em` at `--fs-700`
and above, `+0.06em` on uppercase micro-labels, `0` everywhere else.

---

## Numeric typography

- `font-variant-numeric: tabular-nums lining-nums` is a **primitive** (`.num`), applied
  once, not re-declared per component (currently declared ad hoc in 10 places).
- Contact Luck values always carry an explicit sign: `+7.62`, `−6.41`, `±0.00`. The sign is
  the **non-colour channel** for the semantic pair.
- Use the true minus **U+2212**, never a hyphen, so `−6.41` and `+7.62` align in a column.
- Precision is fixed per quantity, never per context: per-100 → 2 dp; total runs → 1 dp;
  counts → integer; probabilities → 1 dp percent.
- Intervals render one scale step below their point estimate, in brackets, on the same unit.
