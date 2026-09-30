# Handoff — Contact Forecast resolution, v1.2 geometry, 2027 projection (2026-09-29)

Branch: **`predictive-contact-forecast`** (local only; nothing pushed). Tree clean at
`876a50b`. `main` has only the 2026 publish pipeline work; all of the below lives on this
branch. Read `RESEARCH_RULES.md` in full before any research work.

## Done this session

1. **2026 season published.** The 13:37 UTC cron dropped 2026-09-28, so a manual publish ran for
   `data_through=2026-09-27` (run 36476087332, success, deployed). The late scheduled run that
   followed failed with exit 2, expected to be the write-once archive conflict; the log was not
   read. From 2026-09-28 onward the no-completed-games guard (exit 3) keeps the loop quiet.
2. **Contact Forecast resolution pass: done for both horizons** (commit `6783bf8` fixed the
   runner: per-horizon output dirs, a one-run guard, and a missing-features bug in the
   distribution-shift diagnostic). Results: `outputs/forecast_phase2_resolution/h100/`, `h200/`.
   Both classified **no evidence of incremental improvement** (Model D vs shrunk deserved;
   incremental cohort n=31 / n=48). The first runs were interrupted by the bug after metrics were
   written; the rerun reproduced them exactly. Interrupted copies with sha256 are in
   `*/interrupted_run_1/`. Shrunk deserved beat shrunk realized on point MAE in all cohorts, but
   that comparison was not pre-registered for 2026 (R1 on 2023–24 dev *did* pre-register it:
   −3.9% MAE, CI excluded zero).
3. **Plans written and decided** (`docs/plans/`):
   - `2027_projection_model_plan.md`: inputs **2026 only** (no 2025 authorization needed),
     target **contact stage**. Only 2022→23 and 2023→24 are valid dev pairs.
   - `v1_2_park_weather_plan.md`: park **and** weather treated as context, not luck (Option B).
     Airport (ASOS) wind is not in-park wind (Oracle), so the wind term must be proven per venue.
     Dome: wind 0, climate null, altitude still applies.
4. **Park geometry human review: done.** Worksheet: `docs/reviews/park_geometry_review_2026-09-29.md`.
   Rules decided: measured over posted; ±22.5° = team-labelled power alley, else the team's
   left/right-center (A); team figure beats an unsourced "measured" figure (B). Oracle RF stays
   24 ft (a 25 ft correction was approved, then withdrawn in favour of the official MLB.com figure).
5. **v1.2 geometry table built** (`876a50b`): `src/mlb_luck_score/data/park_geometry_v12.py`
   + `tests/test_park_geometry_v12.py` (102 tests). It imports the frozen table and applies the
   approved overrides with sources. `review_status` describes the wall DISTANCE only. **Not wired
   into scoring.** `make check`: 3536 passed, 8 skipped.

## Never do

- Edit `src/mlb_luck_score/data/park_geometry.py`: it is a frozen v1.0 input (32-file manifest).
  Corrections go only in `park_geometry_v12.py`.
- Choose a model or a rule using the 2026 resolution result (2026 is scored, never tuned on).
- Read 2025: sealed. The projection plan no longer needs it.
- Mark a geometry point maintainer-reviewed that the maintainer did not approve.

## Open questions for the maintainer

1. **Fenway CF:** official guide 389 vs stored 390. Not approved; left 390, unreviewed.
2. **Dodger alley heights:** 4.5 ft to the bullpens, 8 ft between them, but which side the ±22.5°
   points fall on is unsourced. Left empty.
3. **Sequencing:** v1.2 adoption comes before the projection model (projections should use v1.2
   deserved values).
4. **Push or PR** the branch? Not done; ask first.

## Next steps, in order

1. **Carry adjustment (v1.2 plan, approach step 1).** Choose and cite a published physical
   relationship for air density / wind → carry distance *before* writing code; record it in the
   plan. Use TDD; synthetic data only in tests.
2. **Per-venue wind check** on 2021–24 development data only: does station wind relate to in-park
   carry at each venue? Where it can't be shown, the wind term is off for that venue, documented.
3. **Freeze the v1.2 evaluation design** (pre-registered adoption criteria: log loss with a
   game-clustered CI, the v0.7D three-way gate by venue, perturbation checks) before computing
   any v1.2 metric.
4. Then the projection-model spec freeze (see the plan's sequencing).

## Also outstanding on `main`

- `PRODUCT.md` says the model uses "venue"; the `venue` feature is 100% null (raw Statcast has no
  such column). Documentation fix.
- CI warnings: Node 20 actions deprecated; `ubuntu-latest` moves to Ubuntu 26 on 2026-10-19.

## Personal (not in the repo)

The maintainer's interview study sheet is at `~/Documents/contact_luck_study_sheet.md`. Never
add it to the repository.
