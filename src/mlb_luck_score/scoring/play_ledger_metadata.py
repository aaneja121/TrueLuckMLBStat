"""Version 1.4.0 (Play Explorer foundation): FILE-level play-ledger provenance.

Phase 2 contract amendment: `play_ledger_version` and other facts that
describe the ARTIFACT as a whole (not any individual play) were removed
from `play_ledger_schema.PLAY_LEDGER_COLUMNS` -- repeating an identical
constant value on 100K+ rows wastes space for zero benefit, exactly the
kind of thing `dashboard/demo_counterfactual_grid.json`'s own top-level
`model_configuration`/`counterfactual_semantics` block (as opposed to a
per-cell field) already established as this codebase's convention. This
module builds that companion metadata dict, meant to be written alongside
`play_ledger.parquet` as a small sidecar (e.g. `play_ledger_metadata.
json`) -- never inside the Parquet file's own rows.

## What is deliberately NOT included here, and why

- **No file SHA-256.** `scripts/archive_snapshot.py`/`prospective/
  run_v1_1_2026_scoring.py`'s existing `integrity_hashes.json` mechanism
  already generically hashes every file under a snapshot's `outputs/`
  directory (Phase 1's inspection confirmed this: `discover_local_
  snapshot_files`/the `output_hashes` dict comprehension both enumerate
  files dynamically, no hardcoded list) -- once `play_ledger.parquet` is
  written into a real snapshot directory (Phase 3), it is automatically
  covered by that mechanism with zero code change there. A second,
  independently-computed hash living inside THIS metadata file would be
  redundant at best and a silent-drift risk at worst if the two hashing
  code paths ever diverged.
- **No `generated_at`/timestamp.** Mirrors `dashboard/demo_counterfactual_
  grid.json`'s own byte-deterministic precedent (that artifact's docstring
  explains the same reasoning) -- `build_play_ledger_metadata`'s output
  must be identical for identical inputs, verified by `tests/
  test_play_ledger_metadata.py`.
- **No `snapshot_directory_name`.** The play ledger file simply lives
  INSIDE that directory once persisted (Phase 3) -- the enclosing
  snapshot's own `manifest.json` already names it; duplicating the name
  here is exactly the kind of "provenance already better represented by
  the enclosing prospective snapshot manifest" the amendment says to
  avoid.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from mlb_luck_score.config import CLASS_ORDER
from mlb_luck_score.scoring.play_ledger_schema import PLAY_LEDGER_VERSION
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP

__all__ = [
    "PlayLedgerMetadataError",
    "build_play_ledger_metadata",
    "compute_scoring_config_fingerprint",
    "validate_play_ledger_metadata",
]


class PlayLedgerMetadataError(ValueError):
    """Raised when a candidate play-ledger metadata dict is invalid or
    inconsistent with its companion play-ledger DataFrame."""


def compute_scoring_config_fingerprint() -> str:
    """SHA-256 over canonical JSON of `{class_order, run_value_map}` -- a
    small, independently re-derivable reproducibility fingerprint (anyone
    can recompute this offline from live `mlb_luck_score.config`/
    `mlb_luck_score.scoring.run_values` constants and confirm it matches),
    mirroring `demo/build_counterfactual_grid.py`'s own `generator_config_
    fingerprint` pattern. Changes if and only if `CLASS_ORDER` or
    `DEFAULT_RUN_VALUE_MAP` change -- i.e. exactly when a play ledger
    produced under the old fingerprint would no longer be reproducible
    under the current scoring configuration.
    """
    canonical = json.dumps(
        {"class_order": list(CLASS_ORDER), "run_value_map": dict(DEFAULT_RUN_VALUE_MAP)},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_play_ledger_metadata(
    df: pd.DataFrame,
    *,
    season: int,
    score_version: str,
    model_versions: Mapping[str, str],
    data_through_date: str | None = None,
) -> dict[str, Any]:
    """Build the file-level metadata dict for a validated play-ledger `df`.

    Args:
        df: A play-ledger DataFrame already satisfying `play_ledger_schema.
            validate_play_ledger` (this function does not itself validate
            row-level correctness -- see `validate_play_ledger_metadata`
            for the cross-check this module DOES perform).
        season: The single season this ledger covers.
        score_version: Verbatim from the same run's player-aggregate
            output (e.g. `public_score_table`'s own `score_version`) --
            never redefined here.
        model_versions: Verbatim, e.g. the same dict `public_score_table`/
            the snapshot manifest already records.
        data_through_date: The snapshot's `data_through_date` when this
            ledger is produced as part of a real prospective run (Phase 3).
            `None` for an offline/development-data validation run (see
            `demo/measure_play_ledger_scale.py`) -- there is no real
            "data-through date" for a fixed historical season, and this
            field must never be fabricated to look like one.

    Returns:
        A plain dict, JSON-serializable, deterministic for identical
        inputs (no timestamp -- see module docstring).
    """
    row_count = int(len(df))
    scored_row_count = int(df["is_scored"].astype("boolean").fillna(False).sum())
    unresolved_row_count = row_count - scored_row_count

    return {
        "play_ledger_version": PLAY_LEDGER_VERSION,
        "season": int(season),
        "data_through_date": data_through_date,
        "row_count": row_count,
        "scored_row_count": scored_row_count,
        "unresolved_row_count": unresolved_row_count,
        "score_version": score_version,
        "model_versions": dict(model_versions),
        "scoring_config_fingerprint": compute_scoring_config_fingerprint(),
    }


def validate_play_ledger_metadata(metadata: Mapping[str, Any], df: pd.DataFrame) -> None:
    """Cross-check that `metadata`'s row-count fields exactly match `df` --
    catches a metadata/data drift (e.g. a metadata dict built against a
    different ledger than the one actually written) rather than trusting
    the caller.

    Raises:
        PlayLedgerMetadataError: on any mismatch, or a missing required key.
    """
    required_keys = (
        "play_ledger_version",
        "season",
        "data_through_date",
        "row_count",
        "scored_row_count",
        "unresolved_row_count",
        "score_version",
        "model_versions",
        "scoring_config_fingerprint",
    )
    missing = [k for k in required_keys if k not in metadata]
    if missing:
        raise PlayLedgerMetadataError(f"play ledger metadata is missing key(s): {missing}")

    actual_row_count = int(len(df))
    actual_scored = int(df["is_scored"].astype("boolean").fillna(False).sum())
    actual_unresolved = actual_row_count - actual_scored

    if metadata["row_count"] != actual_row_count:
        raise PlayLedgerMetadataError(
            f"metadata row_count={metadata['row_count']!r} != actual len(df)={actual_row_count!r}"
        )
    if metadata["scored_row_count"] != actual_scored:
        raise PlayLedgerMetadataError(
            f"metadata scored_row_count={metadata['scored_row_count']!r} != "
            f"actual is_scored sum={actual_scored!r}"
        )
    if metadata["unresolved_row_count"] != actual_unresolved:
        raise PlayLedgerMetadataError(
            f"metadata unresolved_row_count={metadata['unresolved_row_count']!r} != "
            f"actual={actual_unresolved!r}"
        )

    expected_fingerprint = compute_scoring_config_fingerprint()
    if metadata["scoring_config_fingerprint"] != expected_fingerprint:
        raise PlayLedgerMetadataError(
            f"metadata scoring_config_fingerprint={metadata['scoring_config_fingerprint']!r} "
            f"does not match the CURRENT live CLASS_ORDER/DEFAULT_RUN_VALUE_MAP "
            f"(expected {expected_fingerprint!r}) -- this metadata was produced under a "
            "different scoring configuration than the one now running"
        )
