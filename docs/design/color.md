# Color

Semantic colour, the two-grade rule, and the full token table. Entry point: `DESIGN.md`.
Read with `docs/design/accessibility.md` (contrast is checked against the recessed plane,
which is the worst case) and `docs/design/dataviz.md` (marks vs. axis text).

---

## The finding this system exists to fix

The site adopted the `dataviz` skill's **chart palette as its UI palette**. Those hexes are
specified as *mark* and *axis* colours (3:1 floor); the site uses them as *text* at 12–17 px
(4.5:1 required). Measured against all three light surfaces (`--page #f8f6f0`,
`--surface #fefdfa`, `--surface-recessed #f1ede1`):

| Token | Light contrast (page / surface / recessed) | Dark contrast |
|---|---|---|
| muted `#898781` | 3.32 / 3.53 / **3.07** | 5.44 / 4.62 / 5.11 ✔ |
| favorable `#2a78d6` | 4.09 / 4.34 / **3.77** | 5.37 / 4.56 / 5.05 ✔ |
| unfavorable `#e34948` | 3.66 / 3.89 / **3.38** | 6.05 / 5.14 / 5.69 ✔ |
| accent `#b06f16` | 3.79 / 4.02 / **3.50** | 9.36 / 7.95 / 8.80 ✔ |

**Dark mode already passes AA on every token; only the light theme fails.** That is the
signature of a dark theme that was designed and a light theme that inherited chart hexes
untouched.

---

## The rule

**Every semantic colour has two grades: a text grade (≥ 4.5:1 on all three surfaces of its
theme) and a mark grade (≥ 3:1).** Text never uses a mark grade.

---

## Tokens

| Role | Light — text | Light — mark | Dark — text | Dark — mark |
|---|---|---|---|---|
| Favorable (positive sign) | `#256bbf` (4.56–5.25) | `#2a78d6` | `#3987e5` (4.56–5.37) | `#3987e5` |
| Unfavorable (negative sign) | `#d22120` (4.51–5.19) | `#e34948` | `#e66767` (5.14–6.05) | `#e66767` |
| Official / accent | `#975f13` (4.53–5.21) | — | `#e0aa57` (7.95–9.36) | — |
| Ink primary | `#0b0b0b` | — | `#ffffff` | — |
| Ink secondary | `#52514e` | — | `#c3c2b7` | — |
| Ink muted (incl. **chart axis labels**) | `#6d6b66` (4.55–5.23) | — | `#898781` (4.62–5.44) | — |
| Page / surface / recessed | `#f8f6f0` / `#fefdfa` / `#f1ede1` | | `#0d0c0a` / `#211e19` / `#17140f` | |
| Gridline / baseline / hairline border | `#e3dfd3` / `#c3c2b7` / `rgba(11,11,11,.12)` | | `#322f29` / `#3a3834` / `rgba(255,255,255,.12)` | |

Neutral / uncertainty: the interval line takes the **same** semantic colour as its point
estimate (never a separate "uncertainty grey," which would read as a hedge). Uncertainty is
carried by *length on the shared axis*, not by hue.

---

## Rules

- **The diverging pair is reserved for the sign of a point estimate.** No decorative
  element, border, illustration, link, or state may use blue or red. (`PRODUCT.md` §3.)
- **One non-semantic accent: amber, meaning "the frozen official record."** It marks the
  active nav item, the "Data through" badge, the official-rank column, the zero spine and
  the focus ring. When a table is user-sorted, the accent is **removed** from the rank
  column — the departure from official order is signalled by absence plus the explicit
  "Sorted view" label, never by a second colour.
- **`--accent-field` (green) is demoted to an illustration-only token.** It may appear
  inside field diagrams and nowhere else. It currently colours card top-borders on the home
  and demo pages, competing with the semantic pair on the same visual device.
- **Colour is never the only signal.** Every coloured value ships with an explicit sign
  glyph and, where the value is a headline, the word Favorable / Unfavorable.
- Links: ink primary with a 1 px underline at 0.12em offset; the accent is not a link
  colour. This removes the last non-semantic use of blue.
