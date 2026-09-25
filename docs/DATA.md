# Data

What Contact Luck consumes, what it deliberately excludes, the limits of the public
feeds it is built on, and the licensing/reproducibility terms that follow from that.

Companion documents: [`RESEARCH_LOG.md`](RESEARCH_LOG.md) (model development history),
[`OPERATIONS.md`](OPERATIONS.md) (deployment and archival).

## Eligible events (Version 0.1)

Centralized in `mlb_luck_score.eligibility.ELIGIBLE_EVENTS_V0_1`:

```
field_out, force_out, grounded_into_double_play, double_play,
fielders_choice, fielders_choice_out, single, double, triple, home_run,
field_error, sac_bunt, sac_fly, sac_fly_double_play
```

This is a **documented Version 0.1 research choice, not a permanent taxonomy**. Two of
these event types (`fielders_choice`, `field_error`) have an ambiguous batter-runner
resulting base in public Statcast data; rows with those events are kept in the cleaned
table but excluded from model training with a specific reason -- no label is ever
invented for them.

## Excluded events (Version 0.1)

Strikeouts, walks, hit-by-pitches, catcher interference, foul balls/tips, dropped third
strikes, wild pitches, passed balls, stolen bases, pickoffs, balks, fan/outside
interference, discretionary base awards, identified player-rule violations, and rows
lacking required contact variables (`launch_speed`, `launch_angle`).

## Public-data limitations

All data comes from public Statcast fields exposed via `pybaseball`. Some fields
(alignment classifications, bat speed, sprint speed) are only available for a subset of
seasons/players and are treated as optional, not required.

## Approximate spray-direction limitations

`hc_x`/`hc_y` are Statcast's raw visualization coordinates, not verified physical spray
angles. `spray_angle_approx` (and everything derived from it: `spray_sector`, `is_pull`,
`is_center`, `is_opposite_field`) uses a widely-used community approximation formula that
has **not** been independently verified against Baseball Savant's internal geometry.
Treat these as directional signals, not precise measurements.

## Data licensing and redistribution caution

Statcast data is provided publicly by MLB Advanced Media / Baseball Savant via
`pybaseball`. Venue/roof/surface game metadata (Version 0.3) is provided publicly by the
MLB Stats API (`statsapi.mlb.com`, no key required). This repository does not redistribute
any downloaded data (see `.gitignore`); downloaded files stay local under `data/raw/`,
`data/interim/`, and `data/processed/`, all git-ignored. Review MLB Advanced Media/
Baseball Savant's and the MLB Stats API's terms before redistributing any derived data
outside this repository.

## Reproducibility

- Deterministic random seeds are set where applicable (`RANDOM_SEED = 42` in
  `train_contact_model.py`).
- Every saved model artifact is accompanied by a metadata JSON recording model version,
  training seasons, feature list, class order, training timestamp, and package versions.
- `event_id` is built deterministically from stable identifiers
  (`game_pk-at_bat_number-pitch_number`), not a random hash.
