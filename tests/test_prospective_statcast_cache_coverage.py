"""Contact Luck v1.1.2: coverage-aware raw Statcast cache validation.

Regression coverage for the incident this fixes: `data/prospective/2026/
statcast_2026_regular_season.parquet` was downloaded once and then silently
reused across three later `--data-through` requests (2026-08-06, the first
2026-08-08 attempt) whose actual coverage had moved on -- the old guard only
asked "does the file exist", never "does it actually cover what was
requested". Every test here is fully offline: real Statcast/schedule fetch
functions are never called, only injected synthetic ones (`tests/conftest.py`
blocks real network sockets globally as a backstop).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import prospective_ingestion as pi
import pytest

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_raw_parquet(path: Path, game_dates: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"game_date": game_dates}).to_parquet(path, index=False)


def _write_matching_provenance(
    raw_path: Path,
    provenance_path: Path,
    *,
    requested_start: str,
    requested_end: str,
    retrieved_at: str = "2026-08-06T00:00:00+00:00",
) -> pi.RawStatcastCacheProvenance:
    """Builds provenance genuinely FROM the raw file's own real content (real
    hash, real observed dates) -- exactly what `ingest_2026_raw_statcast`
    does after a real download, so these fixtures are internally consistent
    by construction unless a test deliberately corrupts them afterward.
    """
    provenance = pi.build_raw_statcast_provenance(
        raw_path,
        requested_start_date=requested_start,
        requested_end_date=requested_end,
        retrieved_at=retrieved_at,
    )
    pi.write_raw_statcast_provenance(provenance, provenance_path)
    return provenance


def _completed_game(date_str: str, *, status: str = "Final", game_pk: int = 1) -> dict[str, object]:
    return {"game_pk": game_pk, "game_date": date_str, "status_detailed_state": status}


def _fetch_fn(games: list[dict[str, object]]):
    def _fn(start_date: str, end_date: str) -> list[dict[str, object]]:
        return games

    return _fn


def _raising_fetch_fn():
    """Used to prove a code path never actually calls the network-facing
    completed-dates fetcher (e.g. the force_redownload/forward-coverage
    short circuits)."""

    def _fn(start_date: str, end_date: str) -> list[dict[str, object]]:
        raise AssertionError(
            "completed_dates_fetch_fn should not have been called on this code path"
        )

    return _fn


# ---------------------------------------------------------------------------
# 1. cache through Aug 5 + request Aug 8 => automatic refresh
# ---------------------------------------------------------------------------


def test_cache_behind_requested_cutoff_refreshes_automatically(tmp_path: Path) -> None:
    raw_path = tmp_path / "statcast_2026_regular_season.parquet"
    provenance_path = pi._raw_statcast_provenance_path(raw_path)
    _write_raw_parquet(raw_path, ["2026-08-04", "2026-08-05"])
    _write_matching_provenance(
        raw_path, provenance_path, requested_start="2026-03-25", requested_end="2026-08-05"
    )

    completed = [
        _completed_game("2026-08-04"),
        _completed_game("2026-08-05", game_pk=2),
        _completed_game("2026-08-06", game_pk=3),
        _completed_game("2026-08-07", game_pk=4),
        _completed_game("2026-08-08", game_pk=5),
    ]
    result = pi.evaluate_raw_statcast_cache(
        raw_path=raw_path,
        provenance_path=provenance_path,
        requested_start_date="2026-03-25",
        requested_end_date="2026-08-08",
        force_redownload=False,
        completed_dates_fetch_fn=_fetch_fn(completed),
    )
    assert result.decision == pi.CACHE_DECISION_REFRESHED
    assert result.reason == pi.REFRESH_REASON_FORWARD_COVERAGE
    assert result.missing_completed_game_dates == ["2026-08-06", "2026-08-07", "2026-08-08"]


# ---------------------------------------------------------------------------
# 2. cache through Aug 8 + request Aug 8 => reuse
# ---------------------------------------------------------------------------


def test_cache_exactly_matching_requested_cutoff_reuses(tmp_path: Path) -> None:
    raw_path = tmp_path / "statcast_2026_regular_season.parquet"
    provenance_path = pi._raw_statcast_provenance_path(raw_path)
    game_dates = ["2026-03-25", "2026-08-06", "2026-08-07", "2026-08-08"]
    _write_raw_parquet(raw_path, game_dates)
    _write_matching_provenance(
        raw_path, provenance_path, requested_start="2026-03-25", requested_end="2026-08-08"
    )

    completed = [_completed_game(d) for d in game_dates]
    result = pi.evaluate_raw_statcast_cache(
        raw_path=raw_path,
        provenance_path=provenance_path,
        requested_start_date="2026-03-25",
        requested_end_date="2026-08-08",
        force_redownload=False,
        completed_dates_fetch_fn=_fetch_fn(completed),
    )
    assert result.decision == pi.CACHE_DECISION_REUSED
    assert result.reason == pi.REUSE_REASON_VERIFIED
    assert result.missing_completed_game_dates == []


# ---------------------------------------------------------------------------
# 3. recent-but-stale cache => refresh (age never grants validity)
# ---------------------------------------------------------------------------


def test_freshly_retrieved_but_insufficient_coverage_still_refreshes(tmp_path: Path) -> None:
    raw_path = tmp_path / "statcast_2026_regular_season.parquet"
    provenance_path = pi._raw_statcast_provenance_path(raw_path)
    _write_raw_parquet(raw_path, ["2026-08-05"])
    # retrieved "just now" -- but coverage only reaches 2026-08-05.
    _write_matching_provenance(
        raw_path,
        provenance_path,
        requested_start="2026-03-25",
        requested_end="2026-08-05",
        retrieved_at="2026-08-09T23:59:59+00:00",
    )

    result = pi.evaluate_raw_statcast_cache(
        raw_path=raw_path,
        provenance_path=provenance_path,
        requested_start_date="2026-03-25",
        requested_end_date="2026-08-08",
        completed_dates_fetch_fn=_fetch_fn(
            [_completed_game("2026-08-05"), _completed_game("2026-08-08", game_pk=2)]
        ),
    )
    assert result.decision == pi.CACHE_DECISION_REFRESHED
    assert result.reason == pi.REFRESH_REASON_FORWARD_COVERAGE


# ---------------------------------------------------------------------------
# 4. old-but-coverage-complete cache => reuse (age never revokes validity)
# ---------------------------------------------------------------------------


def test_old_retrieval_timestamp_with_complete_coverage_still_reuses(tmp_path: Path) -> None:
    raw_path = tmp_path / "statcast_2026_regular_season.parquet"
    provenance_path = pi._raw_statcast_provenance_path(raw_path)
    game_dates = ["2026-03-25", "2026-08-05"]
    _write_raw_parquet(raw_path, game_dates)
    _write_matching_provenance(
        raw_path,
        provenance_path,
        requested_start="2026-03-25",
        requested_end="2026-08-05",
        retrieved_at="2026-08-06T00:00:00+00:00",  # "old" relative to any later check
    )

    completed = [_completed_game(d) for d in game_dates]
    result = pi.evaluate_raw_statcast_cache(
        raw_path=raw_path,
        provenance_path=provenance_path,
        requested_start_date="2026-03-25",
        requested_end_date="2026-08-05",
        completed_dates_fetch_fn=_fetch_fn(completed),
    )
    assert result.decision == pi.CACHE_DECISION_REUSED


# ---------------------------------------------------------------------------
# 5. later max date with a missing completed-game date INSIDE the range => failure
# (the exact shape of the real incident: a later date's presence must not
# hide an earlier internal gap)
# ---------------------------------------------------------------------------


def test_internal_gap_is_detected_even_though_max_date_reaches_cutoff(tmp_path: Path) -> None:
    raw_path = tmp_path / "statcast_2026_regular_season.parquet"
    provenance_path = pi._raw_statcast_provenance_path(raw_path)
    # Cache has the season-start date AND the cutoff date, but is missing
    # 2026-08-06 and 2026-08-07 entirely -- exactly the real incident shape.
    _write_raw_parquet(raw_path, ["2026-03-25", "2026-08-08"])
    _write_matching_provenance(
        raw_path, provenance_path, requested_start="2026-03-25", requested_end="2026-08-08"
    )

    completed = [
        _completed_game("2026-03-25"),
        _completed_game("2026-08-06", game_pk=2),
        _completed_game("2026-08-07", game_pk=3),
        _completed_game("2026-08-08", game_pk=4),
    ]
    result = pi.evaluate_raw_statcast_cache(
        raw_path=raw_path,
        provenance_path=provenance_path,
        requested_start_date="2026-03-25",
        requested_end_date="2026-08-08",
        completed_dates_fetch_fn=_fetch_fn(completed),
    )
    assert result.decision == pi.CACHE_DECISION_REFRESHED
    assert result.reason == pi.REFRESH_REASON_COVERAGE_GAP
    assert result.missing_completed_game_dates == ["2026-08-06", "2026-08-07"]


# ---------------------------------------------------------------------------
# 6. legitimate league off-day gap (e.g. All-Star break) is allowed
# ---------------------------------------------------------------------------


def test_off_days_with_no_completed_games_are_not_treated_as_gaps(tmp_path: Path) -> None:
    raw_path = tmp_path / "statcast_2026_regular_season.parquet"
    provenance_path = pi._raw_statcast_provenance_path(raw_path)
    # The cache has NO rows for 2026-07-13/14/15 (All-Star break) -- that
    # must never be flagged, since no completed games happened then.
    game_dates = ["2026-07-12", "2026-07-16"]
    _write_raw_parquet(raw_path, game_dates)
    _write_matching_provenance(
        raw_path, provenance_path, requested_start="2026-07-12", requested_end="2026-07-16"
    )

    completed = [
        _completed_game("2026-07-12"),
        # 07-13/14/15: zero games at all (not even postponed) -- an off day.
        _completed_game("2026-07-16", game_pk=2),
    ]
    result = pi.evaluate_raw_statcast_cache(
        raw_path=raw_path,
        provenance_path=provenance_path,
        requested_start_date="2026-07-12",
        requested_end_date="2026-07-16",
        completed_dates_fetch_fn=_fetch_fn(completed),
    )
    assert result.decision == pi.CACHE_DECISION_REUSED
    assert result.missing_completed_game_dates == []


def test_postponed_and_cancelled_games_never_count_as_missing_coverage(tmp_path: Path) -> None:
    raw_path = tmp_path / "statcast_2026_regular_season.parquet"
    provenance_path = pi._raw_statcast_provenance_path(raw_path)
    game_dates = ["2026-04-01"]
    _write_raw_parquet(raw_path, game_dates)
    _write_matching_provenance(
        raw_path, provenance_path, requested_start="2026-04-01", requested_end="2026-04-02"
    )

    completed = [
        _completed_game("2026-04-01"),
        _completed_game("2026-04-02", status="Postponed", game_pk=2),
    ]
    result = pi.evaluate_raw_statcast_cache(
        raw_path=raw_path,
        provenance_path=provenance_path,
        requested_start_date="2026-04-01",
        requested_end_date="2026-04-02",
        completed_dates_fetch_fn=_fetch_fn(completed),
    )
    assert result.decision == pi.CACHE_DECISION_REUSED


# ---------------------------------------------------------------------------
# 7. corrupted/missing provenance => never silently reused
# ---------------------------------------------------------------------------


def test_missing_raw_file_refreshes(tmp_path: Path) -> None:
    raw_path = tmp_path / "statcast_2026_regular_season.parquet"
    provenance_path = pi._raw_statcast_provenance_path(raw_path)
    result = pi.evaluate_raw_statcast_cache(
        raw_path=raw_path,
        provenance_path=provenance_path,
        requested_start_date="2026-03-25",
        requested_end_date="2026-08-08",
        completed_dates_fetch_fn=_raising_fetch_fn(),
    )
    assert result.decision == pi.CACHE_DECISION_REFRESHED
    assert result.reason == pi.REFRESH_REASON_NO_CACHE_FILE


def test_raw_file_present_but_provenance_sidecar_missing_refreshes(tmp_path: Path) -> None:
    raw_path = tmp_path / "statcast_2026_regular_season.parquet"
    provenance_path = pi._raw_statcast_provenance_path(raw_path)
    _write_raw_parquet(raw_path, ["2026-08-05"])
    # No provenance sidecar written at all.
    result = pi.evaluate_raw_statcast_cache(
        raw_path=raw_path,
        provenance_path=provenance_path,
        requested_start_date="2026-03-25",
        requested_end_date="2026-08-05",
        completed_dates_fetch_fn=_raising_fetch_fn(),
    )
    assert result.decision == pi.CACHE_DECISION_REFRESHED
    assert result.reason == pi.REFRESH_REASON_NO_PROVENANCE


def test_corrupted_provenance_json_refreshes_never_raises_uncaught(tmp_path: Path) -> None:
    raw_path = tmp_path / "statcast_2026_regular_season.parquet"
    provenance_path = pi._raw_statcast_provenance_path(raw_path)
    _write_raw_parquet(raw_path, ["2026-08-05"])
    provenance_path.write_text("{not valid json")

    result = pi.evaluate_raw_statcast_cache(
        raw_path=raw_path,
        provenance_path=provenance_path,
        requested_start_date="2026-03-25",
        requested_end_date="2026-08-05",
        completed_dates_fetch_fn=_raising_fetch_fn(),
    )
    assert result.decision == pi.CACHE_DECISION_REFRESHED
    assert result.reason == pi.REFRESH_REASON_MALFORMED_PROVENANCE


def test_provenance_missing_required_fields_refreshes(tmp_path: Path) -> None:
    raw_path = tmp_path / "statcast_2026_regular_season.parquet"
    provenance_path = pi._raw_statcast_provenance_path(raw_path)
    _write_raw_parquet(raw_path, ["2026-08-05"])
    provenance_path.write_text(json.dumps({"row_count": 1}))  # missing every other field

    result = pi.evaluate_raw_statcast_cache(
        raw_path=raw_path,
        provenance_path=provenance_path,
        requested_start_date="2026-03-25",
        requested_end_date="2026-08-05",
        completed_dates_fetch_fn=_raising_fetch_fn(),
    )
    assert result.decision == pi.CACHE_DECISION_REFRESHED
    assert result.reason == pi.REFRESH_REASON_MALFORMED_PROVENANCE


def test_read_raw_statcast_provenance_raises_on_malformed_rather_than_silently_returning_none(
    tmp_path: Path,
) -> None:
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text("{not valid json")
    with pytest.raises(pi.RawStatcastProvenanceError):
        pi.read_raw_statcast_provenance(provenance_path)


def test_read_raw_statcast_provenance_returns_none_only_when_truly_absent(tmp_path: Path) -> None:
    assert pi.read_raw_statcast_provenance(tmp_path / "does_not_exist.json") is None


# ---------------------------------------------------------------------------
# 8. raw-file hash mismatch => failure/refresh
# ---------------------------------------------------------------------------


def test_hash_mismatch_between_raw_file_and_provenance_refreshes(tmp_path: Path) -> None:
    raw_path = tmp_path / "statcast_2026_regular_season.parquet"
    provenance_path = pi._raw_statcast_provenance_path(raw_path)
    _write_raw_parquet(raw_path, ["2026-08-05"])
    _write_matching_provenance(
        raw_path, provenance_path, requested_start="2026-03-25", requested_end="2026-08-05"
    )
    # Mutate the raw file AFTER provenance was written -- e.g. a partial
    # write, manual edit, or filesystem corruption. Hash no longer matches.
    _write_raw_parquet(raw_path, ["2026-08-05", "2026-08-06"])

    result = pi.evaluate_raw_statcast_cache(
        raw_path=raw_path,
        provenance_path=provenance_path,
        requested_start_date="2026-03-25",
        requested_end_date="2026-08-05",
        completed_dates_fetch_fn=_raising_fetch_fn(),
    )
    assert result.decision == pi.CACHE_DECISION_REFRESHED
    assert result.reason == pi.REFRESH_REASON_HASH_MISMATCH


# ---------------------------------------------------------------------------
# 9. --force-redownload still forces a refresh
# ---------------------------------------------------------------------------


def test_force_redownload_refreshes_even_a_perfectly_valid_cache(tmp_path: Path) -> None:
    raw_path = tmp_path / "statcast_2026_regular_season.parquet"
    provenance_path = pi._raw_statcast_provenance_path(raw_path)
    game_dates = ["2026-03-25", "2026-08-05"]
    _write_raw_parquet(raw_path, game_dates)
    _write_matching_provenance(
        raw_path, provenance_path, requested_start="2026-03-25", requested_end="2026-08-05"
    )

    result = pi.evaluate_raw_statcast_cache(
        raw_path=raw_path,
        provenance_path=provenance_path,
        requested_start_date="2026-03-25",
        requested_end_date="2026-08-05",
        force_redownload=True,
        completed_dates_fetch_fn=_raising_fetch_fn(),  # force short-circuits before any fetch
    )
    assert result.decision == pi.CACHE_DECISION_REFRESHED
    assert result.reason == pi.REFRESH_REASON_FORCE_REDOWNLOAD


# ---------------------------------------------------------------------------
# 10. immutable prior snapshot directories are never modified by a cache refresh
# ---------------------------------------------------------------------------


def _init_clean_git_repo(root: Path) -> Path:
    import subprocess

    repo_root = root / "repo"
    repo_root.mkdir()
    env_args = [
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
    ]
    subprocess.run(["git", *env_args, "init", "-q"], cwd=repo_root, check=True)
    (repo_root / "frozen.py").write_text("X = 1\n")
    subprocess.run(["git", *env_args, "add", "frozen.py"], cwd=repo_root, check=True)
    subprocess.run(["git", *env_args, "commit", "-q", "-m", "init"], cwd=repo_root, check=True)
    return repo_root


def test_raw_cache_refresh_never_touches_a_pre_existing_snapshot_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = _init_clean_git_repo(tmp_path)
    authorization = pi.run_prospective_guards(repo_root=repo_root)

    raw_dir = tmp_path / "raw"
    snapshot_marker = (
        tmp_path / "outputs" / "prospective" / "v1_1" / "2026-08-05" / "public_score.json"
    )
    snapshot_marker.parent.mkdir(parents=True)
    snapshot_marker.write_text('{"already": "sealed"}')
    before_mtime = snapshot_marker.stat().st_mtime
    before_content = snapshot_marker.read_bytes()

    fake_df = pd.DataFrame(
        {
            "game_date": ["2026-08-06"],
            "game_pk": [1],
            "at_bat_number": [1],
            "pitch_number": [1],
            "batter": [1],
        }
    )
    monkeypatch.setattr(pi, "download_statcast_range", lambda *a, **kw: fake_df)
    monkeypatch.setattr(
        pi,
        "assert_raw_statcast_schema_compatible",
        lambda raw: {
            "missing_required_columns": [],
            "incompatible_numeric_columns": [],
            "extra_columns_present": [],
        },
    )
    # `_assert_within_namespace` requires `output_dir` to resolve inside the
    # REAL `PROSPECTIVE_RAW_DIR` -- patch that constant to this test's tmp
    # dir rather than weakening the guard itself.
    monkeypatch.setattr(pi, "PROSPECTIVE_RAW_DIR", raw_dir)

    pi.ingest_2026_raw_statcast(
        authorization=authorization,
        data_through_date="2026-08-06",
        season_start_date="2026-08-06",
        output_dir=raw_dir,
        completed_dates_fetch_fn=_fetch_fn([_completed_game("2026-08-06")]),
    )

    assert (raw_dir / "statcast_2026_regular_season.parquet").exists()  # refresh happened
    assert snapshot_marker.stat().st_mtime == before_mtime
    assert snapshot_marker.read_bytes() == before_content


# ---------------------------------------------------------------------------
# 11. final pre-scoring assertion catches stale data even if an upstream
# cache decision is incorrectly mocked -- it never reads that decision
# ---------------------------------------------------------------------------


def test_final_assertion_is_independent_of_an_incorrect_upstream_cache_decision(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "statcast_2026_regular_season.parquet"
    provenance_path = pi._raw_statcast_provenance_path(raw_path)
    _write_raw_parquet(raw_path, ["2026-08-05"])
    _write_matching_provenance(
        raw_path, provenance_path, requested_start="2026-08-05", requested_end="2026-08-06"
    )

    # An upstream cache-validation call using a BROKEN fetch_fn (returns no
    # completed games at all) wrongly concludes "reuse, nothing missing".
    misleading_validation = pi.evaluate_raw_statcast_cache(
        raw_path=raw_path,
        provenance_path=provenance_path,
        requested_start_date="2026-08-05",
        requested_end_date="2026-08-06",
        completed_dates_fetch_fn=_fetch_fn([]),
    )
    assert misleading_validation.decision == pi.CACHE_DECISION_REUSED  # the WRONG verdict

    # The actual dataset about to be scored is genuinely missing 2026-08-06.
    scoring_df = pd.DataFrame({"game_date": ["2026-08-05"]})

    # A correct fetch_fn, supplied ONLY to the final assertion -- it never
    # sees or reads `misleading_validation` at all.
    correct_fetch_fn = _fetch_fn(
        [_completed_game("2026-08-05"), _completed_game("2026-08-06", game_pk=2)]
    )

    with pytest.raises(pi.CoverageContractViolationError, match="2026-08-06"):
        pi.assert_scoring_dataset_satisfies_coverage_contract(
            scoring_df,
            season_start_date="2026-08-05",
            data_through_date="2026-08-06",
            completed_dates_fetch_fn=correct_fetch_fn,
        )


def test_final_assertion_passes_when_scoring_dataset_genuinely_has_full_coverage() -> None:
    scoring_df = pd.DataFrame({"game_date": ["2026-08-05", "2026-08-06"]})
    result = pi.assert_scoring_dataset_satisfies_coverage_contract(
        scoring_df,
        season_start_date="2026-08-05",
        data_through_date="2026-08-06",
        completed_dates_fetch_fn=_fetch_fn(
            [_completed_game("2026-08-05"), _completed_game("2026-08-06", game_pk=2)]
        ),
    )
    assert result["missing_completed_game_dates"] == []


def test_final_assertion_handles_an_empty_scoring_dataset_without_crashing() -> None:
    empty_df = pd.DataFrame({"game_date": pd.array([], dtype="object")})
    with pytest.raises(pi.CoverageContractViolationError):
        pi.assert_scoring_dataset_satisfies_coverage_contract(
            empty_df,
            season_start_date="2026-08-05",
            data_through_date="2026-08-05",
            completed_dates_fetch_fn=_fetch_fn([_completed_game("2026-08-05")]),
        )


# ---------------------------------------------------------------------------
# completed_mlb_game_dates / fetch_schedule_game_statuses_range: unit checks
# ---------------------------------------------------------------------------


def test_completed_mlb_game_dates_excludes_non_final_and_postponed_statuses() -> None:
    games = [
        _completed_game("2026-08-01", status="Final"),
        _completed_game("2026-08-02", status="In Progress", game_pk=2),
        _completed_game("2026-08-03", status="Postponed", game_pk=3),
        _completed_game("2026-08-04", status="Suspended", game_pk=4),
    ]
    result = pi.completed_mlb_game_dates("2026-08-01", "2026-08-04", fetch_fn=_fetch_fn(games))
    assert result == {"2026-08-01"}


def test_fetch_schedule_game_statuses_range_chunks_and_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def _fake_get_json(url: str, params: dict[str, object], **kwargs: object) -> dict[str, object]:
        calls.append((str(params["startDate"]), str(params["endDate"])))
        return {
            "dates": [
                {
                    "date": params["startDate"],
                    "games": [{"gamePk": 1, "status": {"detailedState": "Final"}}],
                }
            ]
        }

    monkeypatch.setattr(pi, "_get_json_with_retries", _fake_get_json)
    games = pi.fetch_schedule_game_statuses_range("2026-03-25", "2026-08-08", chunk_days=30)
    assert len(calls) > 1  # a ~135-day range with 30-day chunks spans multiple requests
    assert all(g["status_detailed_state"] == "Final" for g in games)
