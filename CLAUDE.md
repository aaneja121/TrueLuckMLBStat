# CLAUDE.md -- how to work in this repository

Contact Luck: an MLB batted-ball metric (research pipeline in `src/mlb_luck_score/`) plus
a static public dashboard (`dashboard/`, https://contactluck.com).

This file is a router. It stays short on purpose — load only what your task needs.

## Read before you act

| Situation | Read |
|---|---|
| Anything under `src/mlb_luck_score/`, `prospective/`, `evaluation/`, `demo/`, `scripts/`, `notebooks/` | **`RESEARCH_RULES.md` in full, first.** Non-negotiable. |
| Product intent, page purpose, user framing, copy | `PRODUCT.md` |
| Domain/metric terminology | `CONTEXT.md` |
| Where code lives, data flow, build/test commands | `ARCHITECTURE.md` |
| Frontend/design work | `DESIGN.md` — a router itself; it points into `docs/design/*.md`. Load `docs/design/guardrails.md` for any dashboard change. Then `ARCHITECTURE.md`'s `dashboard/` section |
| Full version history and validation results | `README.md` (large — read sections, never the whole file) |

`AGENTS.md` is the same router for Codex and other cross-agent tooling — keep the two in
sync when routing changes.

## Hard rules (these apply always, no exceptions)

1. **2025 is sealed.** Never tune, iterate, select features, or run analysis on 2025
   results. Never add 2025 to any development season list or downloader. The one narrow
   sealed-evaluation exception is documented in `RESEARCH_RULES.md`; if you think you
   need it, ask first.
2. **2026 may be scored, never tuned against.** Prospective scoring only.
3. **Never commit datasets, secrets, `.venv/`, model artifacts, or anything under
   `data/`, `artifacts/`, `outputs/`.** Run `git status` before every git operation.
4. **Never `git push`, create/modify a remote, deploy, or touch CI** without the user
   asking in that moment. No `git push --force`, `reset --hard`, `checkout --`,
   `clean -f`, or `branch -D` unless explicitly requested right then.
5. **Never silently redefine the score, its formula, its run values, or a threshold.**
   That is a redefinition to be called out, not a bug fix.
6. **The dashboard displays; it never computes.** No module under `dashboard/` may import
   `mlb_luck_score` or any scoring/training/download code, and no score, rank, interval,
   or probability may be derived in a template or in JS.
7. **Say so before any network download.** Full-season downloads are large and slow.

## Working style

- **Search before reading.** grep/glob to locate; never read the repo (or `README.md`)
  broadly by default.
- **Inspect before editing.** Read the real file and its tests first.
- **Prefer small, reviewable diffs.** No unrelated refactors.
- **Preserve current behavior** unless the task is to change it. If a change alters what
  users see or what a number means, say so explicitly.
- **Load only relevant skills.** Purely visual work does not need the ML/research rules,
  and research work does not need the design skills.
- **Use these docs as external memory.** When you learn something durable about the
  project, put it in the right doc above rather than re-deriving it next session.

## Public-facing copy

All user-facing language for the metric is centralized in
`mlb_luck_score.scoring.public_labels`, including a banned-phrase list. Do not invent a
new definition, a new positive/negative explanation, or a stronger claim in a template.
See `PRODUCT.md`'s principles and `CONTEXT.md`'s terminology.

## Verification before claiming done

```bash
make check   # ruff format + ruff check + mypy + pytest  (ruff/mypy cover src+tests only)
```

Run it and fix failures before reporting done, or explain clearly why you couldn't. The
test suite must stay fully offline.

**Substantial frontend work is not done until it has been rendered and looked at** —
~1440, ~1280 and ~390 widths, keyboard focus visible, clean console. Never assert visual
correctness from source.

Use the **globally configured Microsoft Playwright MCP** for this. It is the canonical
browser capability here — never add a repo-local Playwright install, browser dependency,
or dev-server skill. `dashboard/dist/` is gitignored and can be stale; see
`ARCHITECTURE.md` for the safe rebuild command before judging the rendered site.

Never claim something was tested unless it actually ran; report real output, including
failures.
