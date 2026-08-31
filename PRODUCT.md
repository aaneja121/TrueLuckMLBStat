# Product

**Contact Luck** (https://contactluck.com) is a public MLB batted-ball analytics site. It
answers one question: *for a given batted ball — or a whole season of them — was the
result better or worse than what the contact itself was worth?*

The metric is **Contact Luck Runs per 100 eligible batted balls**. It compares the run
value the frozen model **expected** from a fair batted ball (from exit velocity, launch
angle, approximate spray direction, batted-ball type, venue) against the run value that
**actually** occurred, and sums the gap across a player's season to date.

- **Positive** = realized outcomes were more favorable than expected.
- **Negative** = realized outcomes were less favorable than expected.

Every score ships with a game-clustered 95% interval, because at season-to-date sample
sizes most players' Contact Luck is not distinguishable from zero — and the site says so
rather than hiding it.

The product surface is a **static site** (`dashboard/dist/`) rebuilt from immutable daily
scoring snapshots. It never scores, trains, or fetches anything at request time; every
number a visitor sees was computed offline and is copied verbatim into the page.

# Primary audience

1. **Analytically-curious baseball fans** — people who already know xwOBA/xBA exist and
   want to see *whose* results ran hot or cold, at the season and the individual-play
   level. This is the largest group and the one the visual design is for.
2. **Fantasy / betting-adjacent readers** looking for hitters whose surface stats may
   be running ahead of or behind their contact. The site deliberately refuses to make
   the regression promise they may be looking for (see Non-goals).
3. **Analysts and researchers** who want the decomposition, the component status labels,
   the intervals, and the provenance page.
4. **The maintainer**, who uses the Data & status page to confirm what shipped.

A first-time visitor should be able to read the leaderboard's top row and correctly
explain what the number means within about fifteen seconds.

# Core user questions

- Who has had the most (and least) favorable batted-ball luck this season?
- Is *this specific hitter* running hot or cold, by how much, and how sure can we be?
- Has that changed over the season so far?
- What actually happened on a given batted ball, and how big was the gap?
- What does this metric measure — and what does it *not* measure?
- Where does the data come from, and how current is it?

# Core workflows

| Route | Page | What it does |
|---|---|---|
| `/` | **Leaderboard** | Two ranked tables — "Most favorable realized luck" and "Least favorable outcomes relative to expectation" — plus an explainer, a season-to-date stat line, and a small "proof strip" of real example plays. Sortable/filterable client-side; the frozen official rank is always recoverable. |
| `/players/<batter_id>/` | **Player page** | One hitter's headline score + interval, total runs, eligible BBE, scored games, provisional share, component decomposition with status labels, and a season-to-date trend chart. |
| `/explore/` | **Explore Plays** | Editorial "Showcase Plays" (biggest favorable/unfavorable breaks), then a player-first browse: pick a hitter, then filter/sort *their* individual scored batted balls. |
| `/plays/?id=<play_id>` | **Play page** | One batted ball: EV, launch angle, batted-ball type, recorded result, the frozen model's five-class outcome probabilities, Expected RV → observed RV, and the resulting Contact Luck. Showcase plays add a "What if the contact were different?" sensitivity slider. |
| `/demo/` | **How It Works** | A 30-second animated walkthrough of two real 2024 batted balls, plus a "Try It Yourself" EV/launch-angle counterfactual simulator. The conversion path for a confused first-time visitor. |
| `/methodology/` | **Methodology** | Definitions, how a score is built, and limitations, in plain language. |
| `/status/` | **Data & status** | Snapshot provenance, per-component model status, and snapshot history. |

A global player search lives in the site header on every page.

# Value proposition

- **One honest number.** Runs per 100 is additive, in real run units, and directional —
  not an opaque 0–100 index.
- **Uncertainty is shown, not buried.** The interval is rendered next to the point
  estimate on every ranked row, never behind a toggle.
- **It goes all the way down to the play.** Most public "luck" stats stop at the season
  aggregate; here a visitor can walk from a leaderboard row to the exact batted ball and
  see the model's probability distribution for it.
- **Provenance is a feature.** The snapshot, its date coverage, and each component's
  calibration status are visible to anyone who wants them.

# Product principles

1. **Never overclaim.** Contact Luck is retrospective. It is not talent, not a
   projection, and not a regression promise. Banned public phrasings are enforced in
   code (`mlb_luck_score.scoring.public_labels.BANNED_PHRASES`).
2. **Uncertainty is presented, never used as a visual verdict.** An interval that crosses
   zero is rendered exactly like one that doesn't — no fading, muting, or de-emphasis.
3. **Sign color is reserved.** Blue = favorable, red = unfavorable, based on the sign of
   the point estimate only. Decorative accents must never reuse those colors.
4. **Never "worst players."** The negative leaderboard is "Least favorable outcomes
   relative to expectation."
5. **The dashboard displays; it never computes.** No score, rank, interval, or
   probability is derived in a template, in JS, or at request time.
6. **Provisional means labeled.** Components that are not fully calibrated are shown with
   their status and their share of the player's score, not hidden or silently included.
7. **Small samples are reported, never ranked.** Below the qualification threshold a
   player keeps their own page but receives no official rank.
8. **Fail closed.** A missing/invalid snapshot fails the build rather than falling back
   to stale or partial data.

# Non-goals

- **Not a projection or regression tool.** No "expect X to regress" claims.
- **Not a talent metric.** Split-half reliability is low by design — this describes
  realized luck, not skill.
- **Not a defensive- or baserunning-rating product.** Component values are displayed but
  there is deliberately no leaderboard for any individual component.
- **Not a live/in-game product.** Snapshots are daily and only for fully completed
  slates.
- **Not a general Statcast browser.** Only *eligible fair batted balls* are scored;
  strikeouts, walks, HBP and similar never appear.
- **No accounts, no personalization, no server.** The deliverable is static files on a
  CDN.
- **No 2025 data as product data.** 2025 is the sealed final-evaluation season; the live
  product shows 2026 prospective scoring only.
