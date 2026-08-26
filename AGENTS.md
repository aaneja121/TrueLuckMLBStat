# AGENTS.md -- router for AI coding agents

**Contact Luck** is an MLB batted-ball metric: how much more or less favorable a batter's
realized contact outcomes were than expected, in runs. The repository has two halves — a
Python research/scoring pipeline (`src/mlb_luck_score/`, `prospective/`, `evaluation/`)
and a static public dashboard (`dashboard/`, live at https://contactluck.com).

This file is a router, not a rulebook. Load only the document your task needs; do not read
the repository broadly by default. (`CLAUDE.md` is the equivalent router for Claude Code —
keep the two in sync when routing changes.)

## Where to look

| Your task | Read |
|---|---|
| Modeling, scoring, data, seasons — anything under `src/mlb_luck_score/`, `prospective/`, `evaluation/`, `demo/`, `scripts/`, `notebooks/` | **`RESEARCH_RULES.md`, in full, before editing.** Required. |
| Product intent, page purpose, user framing, user-facing copy | `PRODUCT.md` |
| Metric and baseball terminology | `CONTEXT.md` |
| Repo structure, data flow, build/run/test commands | `ARCHITECTURE.md` |
| Frontend / visual design | `DESIGN.md` (once it exists), then `ARCHITECTURE.md`'s `dashboard/` section |
| Version history and validation results | `README.md` — large; read sections, never the whole file |

Repository documentation is the authoritative source of durable project context. If you
learn something durable, put it in the right document above rather than re-deriving it.

## Non-negotiables

1. **2025 is a sealed evaluation season.** Never tune, iterate, select features, or run
   analysis on it. **2026 may be scored, never tuned against.** Details in
   `RESEARCH_RULES.md`; if you think you need an exception, ask.
2. **Never commit datasets, secrets, `.venv/`, model artifacts, or anything under
   `data/`, `artifacts/`, `outputs/`.** Run `git status` before any git operation.
3. **Never push, deploy, change a remote, or touch CI** unless asked in that moment. No
   destructive git operations (`push --force`, `reset --hard`, `clean -f`, `branch -D`).
4. **The dashboard displays; it never computes.** No module under `dashboard/` may import
   scoring, training, or download code, and no score, rank, interval, or probability may
   be derived in a template or in JS.
5. **Never silently redefine the score, a formula, a run value, or a threshold** — that is
   a redefinition to call out, not a bug fix. User-facing metric language is centralized in
   `mlb_luck_score.scoring.public_labels`, including a banned-phrase list.
6. **Preserve existing behavior** unless changing it is the task. Prefer small, reviewable
   diffs; no unrelated refactors. Inspect the real file and its tests before editing.
7. **Say so before any network download.** The test suite must stay fully offline.

## Verification before claiming done

```bash
make check   # ruff format + ruff check + mypy + pytest  (ruff/mypy cover src+tests only)
```

Run it and fix failures, or explain clearly why you couldn't. Never claim something was
tested unless it actually ran; report real output, including failures.

**Substantial UI work is not done until it has been rendered and looked at** — ~1440,
~1280 and ~390 widths, keyboard focus visible, clean console. Never assert visual
correctness from source. Use the globally configured **Microsoft Playwright MCP** (or your
host's equivalent browser capability); do not add a repo-local Playwright install or
browser dependency. `dashboard/dist/` is gitignored and can be stale — see
`ARCHITECTURE.md` for the safe rebuild command before judging the rendered site.
