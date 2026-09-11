"""Contact Luck v1.1: the "nothing new to score" guard.

A `--data-through` date on which NO game was actually completed -- the day
after the regular season ends, a league-wide off day, or a date whose entire
slate was postponed -- carries no new play. Scoring it anyway produces a
snapshot whose contents are identical to the previous game date's, under a
new snapshot directory name and a new archive key: a duplicate, not an
observation.

This matters most at season end. `check_data_through_date_completeness`
treats a date with zero scheduled games as trivially complete (there is
nothing unfinished about a day with no baseball), and both schedule fetches
filter `gameType="R"`, so the postseason is invisible to this pipeline.
Without the guard below, every day after the final regular-season game
passes every existing check and writes another duplicate snapshot.

Every test injects a SYNTHETIC `fetch_fn` -- never a real MLB Stats API call
(the offline suite blocks real network access globally; see
`tests/conftest.py`).
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import prospective_config as pc
import prospective_ingestion as pi
import pytest
import run_v1_1_2026_scoring as runner


def _game(game_pk: int, status: str, date_str: str = "2026-09-28") -> dict[str, object]:
    return {"game_pk": game_pk, "game_date": date_str, "status_detailed_state": status}


def _fetch(games: list[dict[str, object]]):
    def _fn(date_str: str) -> list[dict[str, object]]:
        return games

    return _fn


def _result(games: list[dict[str, object]], date_str: str = "2026-09-28"):
    return pi.check_data_through_date_completeness(date_str, fetch_fn=_fetch(games))


# ---------------------------------------------------------------------------
# completed_games_on_date counts genuinely-final games only
# ---------------------------------------------------------------------------


def test_completed_games_on_date_counts_final_games() -> None:
    result = _result([_game(1, "Final"), _game(2, "Game Over"), _game(3, "Completed Early")])
    assert result.completed_games_on_date == 3


def test_completed_games_on_date_excludes_postponed_and_cancelled() -> None:
    result = _result([_game(1, "Final"), _game(2, "Postponed"), _game(3, "Cancelled")])
    assert result.completed_games_on_date == 1


def test_completed_games_on_date_is_zero_for_a_date_with_no_games() -> None:
    result = _result([])
    assert result.completed_games_on_date == 0


# ---------------------------------------------------------------------------
# The guard itself
# ---------------------------------------------------------------------------


def test_a_date_with_completed_games_passes_the_guard() -> None:
    result = _result([_game(1, "Final"), _game(2, "Postponed")])
    pi.assert_data_through_date_has_completed_games(result)


def test_a_date_with_no_scheduled_games_is_refused() -> None:
    """The season-end case: the day after the last regular-season game."""
    result = _result([])
    with pytest.raises(pi.NoCompletedGamesToScoreError) as excinfo:
        pi.assert_data_through_date_has_completed_games(result)
    assert "2026-09-28" in str(excinfo.value)


def test_a_fully_postponed_slate_is_refused() -> None:
    result = _result([_game(1, "Postponed"), _game(2, "Cancelled")])
    with pytest.raises(pi.NoCompletedGamesToScoreError):
        pi.assert_data_through_date_has_completed_games(result)


def test_the_guard_is_a_subclass_of_prospective_error() -> None:
    """`main()` catches ProspectiveError; this must be reachable by it."""
    assert issubclass(pi.NoCompletedGamesToScoreError, pi.ProspectiveError)


# ---------------------------------------------------------------------------
# Manifest-schema stability
# ---------------------------------------------------------------------------


def test_completed_games_on_date_stays_out_of_to_dict() -> None:
    """`to_dict()` flows into every snapshot manifest's `schema_checks`, and
    manifests are content-hashed to decide whether a rerun is an idempotent
    no-op or a conflict. A NEW KEY here would change the deterministic hash
    of every already-archived snapshot's date-completeness block, turning a
    legitimate rerun of an archived date into a SnapshotConflictError. This
    is why `completed_games_on_date` is a derived property rather than a
    dataclass field.
    """
    result = _result([_game(1, "Final")])
    assert "completed_games_on_date" not in result.to_dict()
    assert set(result.to_dict()) == {
        "data_through_date",
        "is_complete",
        "total_games_on_date",
        "incomplete_games",
        "postponed_or_cancelled_games",
        "checked_at",
    }


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


def test_runner_refuses_before_downloading_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard must fire BEFORE ingestion. A full-season Statcast download
    on every off-season morning is the expensive half of this defect.
    """

    def _boom(**kwargs: object) -> None:
        raise AssertionError("ingestion must not run when there are no completed games")

    monkeypatch.setattr(
        runner, "run_prospective_guards", lambda **kw: SimpleNamespace(token="t", granted_at="now")
    )
    monkeypatch.setattr(
        runner, "assert_data_through_date_is_complete", lambda *a, **kw: _result([])
    )
    monkeypatch.setattr(runner, "ingest_2026_raw_statcast", _boom)
    monkeypatch.setattr(runner, "ingest_2026_game_metadata", _boom)
    monkeypatch.setattr(runner, "ingest_2026_sprint_speed", _boom)

    with pytest.raises(pi.NoCompletedGamesToScoreError):
        runner.run_prospective_snapshot(data_through_date="2026-09-28")


def test_main_maps_the_guard_to_its_own_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exit 3 means "nothing to score", distinct from exit 2 ("this run
    failed"), so the publish loop can stop cleanly instead of alarming.
    """

    def _raise(**kwargs: object) -> None:
        raise pi.NoCompletedGamesToScoreError("no completed games")

    monkeypatch.setattr(runner, "run_prospective_snapshot", _raise)
    assert runner.main(["--data-through", "2026-09-28"]) == 3


def test_main_still_returns_two_for_an_ordinary_guard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(**kwargs: object) -> None:
        raise pi.IncompleteDataThroughDateError("a game is still in progress")

    monkeypatch.setattr(runner, "run_prospective_snapshot", _raise)
    assert runner.main(["--data-through", "2026-09-28"]) == 2


# ---------------------------------------------------------------------------
# The recorded season-END date, and the cross-check against it
# ---------------------------------------------------------------------------


class TestRecordedSeasonEndDate:
    """The explicit second layer beside the schedule-derived guard above.

    It is a CROSS-CHECK, never a blind gate. A hardcoded end date that simply
    refused every later request would silently skip a rainout makeup played
    after the finale -- so this fires only when the schedule and the recorded
    constant genuinely DISAGREE, which means the constant is stale.
    """

    def test_the_constants_are_recorded_together(self) -> None:
        """Date, source citation and verification date move as a unit -- see
        the same rule for the season-START constants.
        """
        assert date(2026, 9, 27) == pc.PROSPECTIVE_2026_SEASON_END_DATE
        assert pc.PROSPECTIVE_2026_SEASON_END_VERIFIED is True
        assert pc.PROSPECTIVE_2026_SEASON_END_VERIFIED_AT == "2026-09-11"
        assert "2026-09-27" in pc.PROSPECTIVE_2026_SEASON_END_SOURCE
        assert "maintainer-provided citation" in pc.PROSPECTIVE_2026_SEASON_END_SOURCE

    def test_the_recorded_end_is_after_the_recorded_start(self) -> None:
        assert pc.PROSPECTIVE_2026_SEASON_END_DATE > pc.PROSPECTIVE_2026_SEASON_START_DATE

    def test_the_finale_itself_passes(self) -> None:
        result = _result([_game(1, "Final", "2026-09-27")], "2026-09-27")
        pi.assert_data_through_date_agrees_with_recorded_season_end(result)

    def test_an_ordinary_in_season_date_passes(self) -> None:
        result = _result([_game(1, "Final", "2026-07-04")], "2026-07-04")
        pi.assert_data_through_date_agrees_with_recorded_season_end(result)

    def test_games_completed_after_the_recorded_finale_are_a_loud_failure(self) -> None:
        """A makeup played after the recorded finale, or simply a wrong
        constant. Either way the recorded fact is stale and a human must
        update it -- never silently skipped, never silently scored.
        """
        result = _result([_game(1, "Final", "2026-09-28")], "2026-09-28")
        with pytest.raises(pi.RecordedSeasonEndDateStaleError) as excinfo:
            pi.assert_data_through_date_agrees_with_recorded_season_end(result)
        message = str(excinfo.value)
        assert "2026-09-28" in message
        assert "2026-09-27" in message
        assert "PROSPECTIVE_2026_SEASON_END_DATE" in message

    def test_a_date_after_the_finale_with_no_games_is_not_this_guards_problem(self) -> None:
        """That is the ordinary off-season case, owned by the schedule-derived
        guard and its clean exit 3 -- this one must stay silent about it.
        """
        result = _result([], "2026-10-15")
        pi.assert_data_through_date_agrees_with_recorded_season_end(result)

    def test_the_stale_constant_error_is_a_prospective_error(self) -> None:
        assert issubclass(pi.RecordedSeasonEndDateStaleError, pi.ProspectiveError)


def test_runner_rejects_games_after_the_recorded_finale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner, "run_prospective_guards", lambda **kw: SimpleNamespace(token="t", granted_at="now")
    )
    monkeypatch.setattr(
        runner,
        "assert_data_through_date_is_complete",
        lambda *a, **kw: _result([_game(1, "Final", "2026-09-28")], "2026-09-28"),
    )
    with pytest.raises(pi.RecordedSeasonEndDateStaleError):
        runner.run_prospective_snapshot(data_through_date="2026-09-28")


def test_a_stale_season_end_constant_is_exit_two_not_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exit 3 means "nothing to score, stop quietly". A stale recorded fact is
    a real failure that must be seen, so it keeps the generic failure code.
    """

    def _raise(**kwargs: object) -> None:
        raise pi.RecordedSeasonEndDateStaleError("stale")

    monkeypatch.setattr(runner, "run_prospective_snapshot", _raise)
    assert runner.main(["--data-through", "2026-09-28"]) == 2
