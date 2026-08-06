"""Contact Luck v1.0, Task 33 category 2: dedicated-entry-point enforcement.

Verifies 2025 is rejected by every development downloader/runner, that
`allow_final_evaluation=True` cannot be reached through a development CLI,
and that only `run_v1_final_evaluation.run_pre_evaluation_guards` can mint a
`FinalEvaluationAuthorization` that the low-level ingestion functions accept.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import run_v1_final_evaluation as rv1

from mlb_luck_score.config import ProtectedSeasonError
from mlb_luck_score.data.clean_development_data import clean_development_data
from mlb_luck_score.data.download_development_data import download_development_data
from mlb_luck_score.data.download_game_metadata import download_game_metadata
from mlb_luck_score.data.download_sprint_speed import download_sprint_speed
from mlb_luck_score.scoring.run_public_score import build_arg_parser as public_score_arg_parser
from mlb_luck_score.scoring.run_season_aggregation import build_arg_parser as season_agg_arg_parser

# ---------------------------------------------------------------------------
# Development downloaders/runners reject 2025
# ---------------------------------------------------------------------------


def test_download_development_data_rejects_2025(tmp_path: Path) -> None:
    with pytest.raises(ProtectedSeasonError):
        download_development_data([2025], tmp_path, dry_run=True)


def test_download_game_metadata_rejects_2025(tmp_path: Path) -> None:
    with pytest.raises(ProtectedSeasonError):
        download_game_metadata([2025], tmp_path, dry_run=True)


def test_download_sprint_speed_rejects_2025(tmp_path: Path) -> None:
    with pytest.raises(ProtectedSeasonError):
        download_sprint_speed([2025], tmp_path, dry_run=True)


def test_clean_development_data_rejects_2025_without_flag(tmp_path: Path) -> None:
    with pytest.raises(ProtectedSeasonError):
        clean_development_data([2025], tmp_path)


def test_download_development_data_allows_2025_only_with_explicit_flag(tmp_path: Path) -> None:
    """The escape hatch exists on the low-level function signature (for
    `run_v1_final_evaluation` NOT to use it -- it uses `download_statcast_
    range` directly instead, see module docstring), but this confirms the
    flag genuinely gates it rather than being decorative.
    """
    results = download_development_data([2025], tmp_path, dry_run=True, allow_final_evaluation=True)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Development scoring/aggregation runners have no escape hatch at all
# ---------------------------------------------------------------------------


def test_season_aggregation_cli_has_no_allow_final_evaluation_flag() -> None:
    parser = season_agg_arg_parser()
    option_strings = {opt for action in parser._actions for opt in action.option_strings}
    assert "--allow-final-evaluation" not in option_strings


def test_public_score_cli_has_no_allow_final_evaluation_flag() -> None:
    parser = public_score_arg_parser()
    option_strings = {opt for action in parser._actions for opt in action.option_strings}
    assert "--allow-final-evaluation" not in option_strings


# ---------------------------------------------------------------------------
# Only run_pre_evaluation_guards can mint a usable authorization
# ---------------------------------------------------------------------------


def test_bare_bool_authorization_is_rejected() -> None:
    with pytest.raises(rv1.FinalEvaluationError, match="FinalEvaluationAuthorization"):
        rv1._require_authorization(True, "test_caller")


def test_none_authorization_is_rejected() -> None:
    with pytest.raises(rv1.FinalEvaluationError):
        rv1._require_authorization(None, "test_caller")


def test_fabricated_authorization_object_is_rejected() -> None:
    """Constructing the dataclass directly (not via `_mint_authorization`)
    produces a token that was never added to `_ISSUED_AUTHORIZATION_TOKENS`.
    """
    fake = rv1.FinalEvaluationAuthorization(
        token="fabricated-token-not-issued",
        manifest_content_hash="deadbeef",
        granted_at="2025-01-01T00:00:00+00:00",
    )
    with pytest.raises(rv1.FinalEvaluationError, match="fabricated"):
        rv1._require_authorization(fake, "test_caller")


def test_genuinely_minted_authorization_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeManifest:
        def content_hash(self) -> str:
            return "abc123"

    auth = rv1._mint_authorization(_FakeManifest())  # type: ignore[arg-type]
    # Must not raise.
    rv1._require_authorization(auth, "test_caller")


def test_ingestion_functions_reject_unauthorized_calls(tmp_path: Path) -> None:
    fake = rv1.FinalEvaluationAuthorization(
        token="still-fabricated", manifest_content_hash="x", granted_at="y"
    )
    with pytest.raises(rv1.FinalEvaluationError):
        rv1.ingest_2025_raw_statcast(authorization=fake, output_dir=tmp_path)
    with pytest.raises(rv1.FinalEvaluationError):
        rv1.ingest_2025_game_metadata(authorization=fake, output_dir=tmp_path)
    with pytest.raises(rv1.FinalEvaluationError):
        rv1.ingest_2025_sprint_speed(authorization=fake, output_dir=tmp_path)
    with pytest.raises(rv1.FinalEvaluationError):
        rv1.build_2025_evaluation_dataset(authorization=fake, raw_dir=tmp_path)
