"""Contact Luck v1.1: data-through-date completeness guard.

Do not permit a `--data-through` date containing games still in progress.
Every test injects a SYNTHETIC `fetch_fn` -- never a real MLB Stats API call
(the offline suite blocks real network access globally; see
`tests/conftest.py`).
"""

from __future__ import annotations

import prospective_ingestion as pi
import pytest


def _status(game_pk: int, status: str) -> dict[str, object]:
    return {"game_pk": game_pk, "game_date": "2026-04-15", "status_detailed_state": status}


def _fetch(games: list[dict[str, object]]):
    def _fn(date_str: str) -> list[dict[str, object]]:
        return games

    return _fn


# ---------------------------------------------------------------------------
# All-final date is complete
# ---------------------------------------------------------------------------


def test_all_final_games_is_complete() -> None:
    games = [_status(1, "Final"), _status(2, "Final"), _status(3, "Game Over")]
    result = pi.check_data_through_date_completeness("2026-04-15", fetch_fn=_fetch(games))
    assert result.is_complete is True
    assert result.total_games_on_date == 3
    assert result.incomplete_games == []


def test_off_day_with_zero_games_is_complete() -> None:
    result = pi.check_data_through_date_completeness("2026-04-15", fetch_fn=_fetch([]))
    assert result.is_complete is True
    assert result.total_games_on_date == 0


# ---------------------------------------------------------------------------
# In-progress games block
# ---------------------------------------------------------------------------


def test_in_progress_game_is_incomplete() -> None:
    games = [_status(1, "Final"), _status(2, "In Progress")]
    result = pi.check_data_through_date_completeness("2026-04-15", fetch_fn=_fetch(games))
    assert result.is_complete is False
    assert len(result.incomplete_games) == 1
    assert result.incomplete_games[0]["game_pk"] == 2


@pytest.mark.parametrize(
    "status", ["In Progress", "Warmup", "Pre-Game", "Delayed", "Delayed Start", "Scheduled"]
)
def test_various_not_yet_final_statuses_are_incomplete(status: str) -> None:
    games = [_status(1, status)]
    result = pi.check_data_through_date_completeness("2026-04-15", fetch_fn=_fetch(games))
    assert result.is_complete is False


def test_unrecognized_status_is_treated_as_incomplete_fail_safe() -> None:
    """Fail-safe by design -- an unrecognized status string is never assumed
    final."""
    games = [_status(1, "Something MLB Has Never Returned Before")]
    result = pi.check_data_through_date_completeness("2026-04-15", fetch_fn=_fetch(games))
    assert result.is_complete is False


# ---------------------------------------------------------------------------
# Postponed/cancelled games: excluded from the completeness requirement,
# but always recorded for provenance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["Postponed", "Cancelled"])
def test_postponed_or_cancelled_game_does_not_block_the_date(status: str) -> None:
    games = [_status(1, "Final"), _status(2, status)]
    result = pi.check_data_through_date_completeness("2026-04-15", fetch_fn=_fetch(games))
    assert result.is_complete is True
    assert result.incomplete_games == []
    assert len(result.postponed_or_cancelled_games) == 1
    assert result.postponed_or_cancelled_games[0]["game_pk"] == 2


def test_a_date_with_only_a_postponed_game_and_nothing_else_is_complete() -> None:
    games = [_status(1, "Postponed")]
    result = pi.check_data_through_date_completeness("2026-04-15", fetch_fn=_fetch(games))
    assert result.is_complete is True


# ---------------------------------------------------------------------------
# Suspended games: documented rule -- treated like in-progress, NOT postponed
# ---------------------------------------------------------------------------


def test_suspended_game_blocks_the_date_like_in_progress() -> None:
    games = [_status(1, "Suspended")]
    result = pi.check_data_through_date_completeness("2026-04-15", fetch_fn=_fetch(games))
    assert result.is_complete is False
    assert len(result.incomplete_games) == 1
    assert result.postponed_or_cancelled_games == []


def test_suspended_status_is_not_classified_as_postponed_or_cancelled() -> None:
    assert "Suspended" not in pi.POSTPONED_OR_CANCELLED_STATUS_VALUES
    assert "Suspended" not in pi.COMPLETED_GAME_STATUS_VALUES


# ---------------------------------------------------------------------------
# assert_data_through_date_is_complete: fails loudly, never silently shifts
# the requested date
# ---------------------------------------------------------------------------


def test_assert_passes_through_the_result_when_complete() -> None:
    games = [_status(1, "Final")]
    result = pi.assert_data_through_date_is_complete("2026-04-15", fetch_fn=_fetch(games))
    assert result.is_complete is True


def test_assert_raises_incomplete_data_through_date_error_when_incomplete() -> None:
    games = [_status(1, "In Progress")]
    with pytest.raises(pi.IncompleteDataThroughDateError, match="2026-04-15"):
        pi.assert_data_through_date_is_complete("2026-04-15", fetch_fn=_fetch(games))


def test_assert_error_message_names_the_observed_statuses() -> None:
    games = [_status(1, "Suspended")]
    with pytest.raises(pi.IncompleteDataThroughDateError, match="Suspended"):
        pi.assert_data_through_date_is_complete("2026-04-15", fetch_fn=_fetch(games))


def test_assert_never_returns_a_different_date_than_requested() -> None:
    """The guard fails rather than silently substituting an earlier date --
    confirmed by checking the raised exception message references ONLY the
    requested date, never a computed fallback."""
    games = [_status(1, "In Progress")]
    with pytest.raises(pi.IncompleteDataThroughDateError) as exc_info:
        pi.assert_data_through_date_is_complete("2026-08-06", fetch_fn=_fetch(games))
    assert "2026-08-06" in str(exc_info.value)
    assert "Choose an earlier --data-through date" in str(exc_info.value)


# ---------------------------------------------------------------------------
# fetch_schedule_game_statuses parsing (network-shaped but not network-called)
# ---------------------------------------------------------------------------


def test_fetch_schedule_game_statuses_parses_the_mlb_api_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_get_json(url: str, params: dict[str, object], **kwargs: object) -> dict[str, object]:
        assert params["startDate"] == "2026-04-15"
        assert params["endDate"] == "2026-04-15"
        return {
            "dates": [
                {
                    "games": [
                        {"gamePk": 111, "status": {"detailedState": "Final"}},
                        {"gamePk": 222, "status": {"detailedState": "In Progress"}},
                    ]
                }
            ]
        }

    monkeypatch.setattr(pi, "_get_json_with_retries", _fake_get_json)
    games = pi.fetch_schedule_game_statuses("2026-04-15")
    assert games == [
        {"game_pk": 111, "game_date": "2026-04-15", "status_detailed_state": "Final"},
        {"game_pk": 222, "game_date": "2026-04-15", "status_detailed_state": "In Progress"},
    ]
