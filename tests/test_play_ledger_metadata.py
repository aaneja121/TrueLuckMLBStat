"""Unit tests for the Version 1.4.0 play-ledger FILE-level metadata contract
(`play_ledger_metadata.py`).

Pure hand-built DataFrames -- no model training, no real data, no network.
Only `is_scored` is exercised by this module's functions, so fixtures here
are intentionally minimal (not full `PLAY_LEDGER_COLUMNS`-shaped rows; see
`tests/test_play_ledger_schema.py`/`tests/test_play_ledger_export.py` for
the full-schema contract).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mlb_luck_score.scoring.play_ledger_metadata import (
    PlayLedgerMetadataError,
    build_play_ledger_metadata,
    compute_scoring_config_fingerprint,
    validate_play_ledger_metadata,
)
from mlb_luck_score.scoring.play_ledger_schema import PLAY_LEDGER_VERSION


def _df(is_scored: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({"is_scored": pd.array(is_scored, dtype="boolean")})


def test_build_play_ledger_metadata_basic_fields():
    df = _df([True, True, False])
    metadata = build_play_ledger_metadata(
        df,
        season=2024,
        score_version="0.2",
        model_versions={"contact": "v0.2"},
    )
    assert metadata["play_ledger_version"] == PLAY_LEDGER_VERSION
    assert metadata["season"] == 2024
    assert metadata["data_through_date"] is None
    assert metadata["row_count"] == 3
    assert metadata["scored_row_count"] == 2
    assert metadata["unresolved_row_count"] == 1
    assert metadata["score_version"] == "0.2"
    assert metadata["model_versions"] == {"contact": "v0.2"}
    assert isinstance(metadata["scoring_config_fingerprint"], str)


def test_build_play_ledger_metadata_data_through_date_passthrough():
    df = _df([True])
    metadata = build_play_ledger_metadata(
        df,
        season=2026,
        score_version="0.2",
        model_versions={},
        data_through_date="2026-08-14",
    )
    assert metadata["data_through_date"] == "2026-08-14"


def test_metadata_row_counts_all_unresolved():
    df = _df([False, False])
    metadata = build_play_ledger_metadata(df, season=2024, score_version="0.2", model_versions={})
    assert metadata["row_count"] == 2
    assert metadata["scored_row_count"] == 0
    assert metadata["unresolved_row_count"] == 2


def test_metadata_has_no_timestamp_field():
    """Must stay deterministic for identical inputs -- no `generated_at` or
    any other wall-clock-derived field (mirrors `dashboard/demo_
    counterfactual_grid.json`'s existing byte-deterministic precedent)."""
    df = _df([True, False])
    metadata = build_play_ledger_metadata(df, season=2024, score_version="0.2", model_versions={})
    for key in metadata:
        assert "time" not in key.lower()
        assert "generated" not in key.lower()
        assert "timestamp" not in key.lower()


def test_metadata_deterministic_across_repeated_calls():
    df = _df([True, True, False])
    first = build_play_ledger_metadata(
        df, season=2024, score_version="0.2", model_versions={"contact": "v0.2"}
    )
    second = build_play_ledger_metadata(
        df, season=2024, score_version="0.2", model_versions={"contact": "v0.2"}
    )
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_scoring_config_fingerprint_deterministic():
    assert compute_scoring_config_fingerprint() == compute_scoring_config_fingerprint()


def test_scoring_config_fingerprint_is_sha256_hex():
    fingerprint = compute_scoring_config_fingerprint()
    assert len(fingerprint) == 64
    int(fingerprint, 16)  # raises ValueError if not valid hex


def test_validate_play_ledger_metadata_passes_for_consistent_metadata():
    df = _df([True, True, False])
    metadata = build_play_ledger_metadata(df, season=2024, score_version="0.2", model_versions={})
    validate_play_ledger_metadata(metadata, df)  # no raise


def test_validate_play_ledger_metadata_raises_on_missing_key():
    df = _df([True])
    metadata = build_play_ledger_metadata(df, season=2024, score_version="0.2", model_versions={})
    del metadata["row_count"]
    with pytest.raises(PlayLedgerMetadataError, match="missing key"):
        validate_play_ledger_metadata(metadata, df)


def test_validate_play_ledger_metadata_raises_on_row_count_mismatch():
    df = _df([True, False])
    metadata = build_play_ledger_metadata(df, season=2024, score_version="0.2", model_versions={})
    metadata["row_count"] = 999
    with pytest.raises(PlayLedgerMetadataError, match="row_count"):
        validate_play_ledger_metadata(metadata, df)


def test_validate_play_ledger_metadata_raises_on_scored_row_count_mismatch():
    df = _df([True, False])
    metadata = build_play_ledger_metadata(df, season=2024, score_version="0.2", model_versions={})
    metadata["scored_row_count"] = 999
    with pytest.raises(PlayLedgerMetadataError, match="scored_row_count"):
        validate_play_ledger_metadata(metadata, df)


def test_validate_play_ledger_metadata_raises_on_unresolved_row_count_mismatch():
    df = _df([True, False])
    metadata = build_play_ledger_metadata(df, season=2024, score_version="0.2", model_versions={})
    metadata["unresolved_row_count"] = 999
    with pytest.raises(PlayLedgerMetadataError, match="unresolved_row_count"):
        validate_play_ledger_metadata(metadata, df)


def test_validate_play_ledger_metadata_raises_on_fingerprint_mismatch():
    df = _df([True])
    metadata = build_play_ledger_metadata(df, season=2024, score_version="0.2", model_versions={})
    metadata["scoring_config_fingerprint"] = "not-the-real-fingerprint"
    with pytest.raises(PlayLedgerMetadataError, match="scoring_config_fingerprint"):
        validate_play_ledger_metadata(metadata, df)


def test_validate_play_ledger_metadata_recomputed_from_a_different_but_consistent_df():
    """A metadata dict built against one ledger df must not spuriously pass
    validation against an inconsistent second df."""
    df_a = _df([True, True])
    df_b = _df([True, False, False])
    metadata_a = build_play_ledger_metadata(
        df_a, season=2024, score_version="0.2", model_versions={}
    )
    with pytest.raises(PlayLedgerMetadataError):
        validate_play_ledger_metadata(metadata_a, df_b)
