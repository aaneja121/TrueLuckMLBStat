"""Fail-fast guards for the one-shot 2027 v1.3 comparison (RESEARCH_RULES.md, "2027: a
pre-registered test season"). Synthetic schedule data only; never the network."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import v1_3_2027_config as cfg

START, END = date(2027, 3, 24), date(2027, 9, 26)


def _games(statuses: list[tuple[str, str]]):
    def fetch(start: str, end: str):
        return [
            {"game_pk": i, "game_date": d, "status_detailed_state": s}
            for i, (d, s) in enumerate(statuses)
            if start <= d <= end
        ]

    return fetch


# --- recorded season dates ---------------------------------------------------------


def test_start_date_is_the_cited_official_date() -> None:
    assert cfg.SEASON_2027_START_DATE == START
    assert "mlb.com/news/mlb-2027-schedule-released" in cfg.SEASON_2027_START_SOURCE
    assert cfg.SEASON_2027_START_VERIFIED_AT == "2026-10-01"


def test_end_date_is_not_recorded_yet() -> None:
    # Recorded only after the season, with a citation (same rule as 2026).
    assert cfg.SEASON_2027_END_DATE is None


def test_unrecorded_end_date_blocks_the_run() -> None:
    with pytest.raises(cfg.SeasonNotRecordedError):
        cfg.assert_season_end_recorded(None, None, None)


@pytest.mark.parametrize(
    ("source", "verified_at"), [("", "2027-09-30"), ("https://x", ""), (None, "2027-09-30")]
)
def test_end_date_needs_source_and_verification(source, verified_at) -> None:
    with pytest.raises(cfg.SeasonNotRecordedError):
        cfg.assert_season_end_recorded(END, source, verified_at)


def test_recorded_end_date_passes() -> None:
    assert cfg.assert_season_end_recorded(END, "https://www.mlb.com/schedule", "2027-09-30") == END


# --- season completeness -------------------------------------------------------------------


def test_complete_season_passes_and_counts() -> None:
    fetch = _games(
        [("2027-03-24", "Final"), ("2027-06-01", "Postponed"), ("2027-09-26", "Game Over")]
    )
    report = cfg.check_season_complete(START, END, fetch_fn=fetch, today=date(2027, 10, 1))
    assert report["n_final"] == 2 and report["n_postponed_or_cancelled"] == 1


@pytest.mark.parametrize("status", ["Suspended", "In Progress", "Scheduled", "Something New"])
def test_any_unfinished_game_blocks(status: str) -> None:
    fetch = _games([("2027-03-24", "Final"), ("2027-07-04", status)])
    with pytest.raises(cfg.SeasonIncompleteError):
        cfg.check_season_complete(START, END, fetch_fn=fetch, today=date(2027, 10, 1))


def test_a_game_completed_after_the_recorded_end_blocks() -> None:
    fetch = _games([("2027-03-24", "Final"), ("2027-09-28", "Final")])
    with pytest.raises(cfg.SeasonIncompleteError, match="after the recorded"):
        cfg.check_season_complete(START, END, fetch_fn=fetch, today=date(2027, 10, 15))


def test_running_before_the_end_date_blocks() -> None:
    fetch = _games([("2027-03-24", "Final")])
    with pytest.raises(cfg.SeasonIncompleteError):
        cfg.check_season_complete(START, END, fetch_fn=fetch, today=date(2027, 9, 20))


def test_a_season_with_no_final_games_blocks() -> None:
    with pytest.raises(cfg.SeasonIncompleteError):
        cfg.check_season_complete(START, END, fetch_fn=_games([]), today=date(2027, 10, 1))


# --- one run, own namespace ---------------------------------------------------------------------


def test_second_run_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "comparison_2027.json"
    cfg.assert_not_already_run(out)
    out.write_text("{}")
    with pytest.raises(cfg.AlreadyRunError):
        cfg.assert_not_already_run(out)


def test_namespaces_are_inside_the_gitignored_prospective_roots() -> None:
    root = cfg.PROJECT_ROOT
    assert root / "data" / "prospective" / "2027" / "v1_3" == cfg.RAW_DIR
    assert root / "outputs" / "prospective" / "v1_3" == cfg.OUTPUTS_DIR
    assert root / "artifacts" / "prospective" / "v1_3" == cfg.ARTIFACTS_DIR
    assert cfg.COMPARISON_PATH.parent == cfg.OUTPUTS_DIR


def test_paths_outside_the_namespace_are_refused() -> None:
    cfg.assert_in_namespace(cfg.RAW_DIR / "statcast_2027.parquet")
    for bad in (
        cfg.PROJECT_ROOT / "data" / "processed" / "x.parquet",
        cfg.RAW_DIR / ".." / ".." / ".." / "processed" / "x.parquet",
        cfg.PROJECT_ROOT / "data" / "prospective" / "2026" / "x.parquet",
        cfg.PROJECT_ROOT / "data" / "final_evaluation" / "2025" / "x.parquet",
    ):
        with pytest.raises(cfg.NamespaceViolationError):
            cfg.assert_in_namespace(bad)


def test_2027_never_enters_the_frozen_season_lists() -> None:
    from mlb_luck_score import config

    assert 2027 not in config.DEVELOPMENT_SEASONS
    assert 2027 not in config.MLB_REGULAR_SEASON_DATE_RANGES
    assert config.PROSPECTIVE_SEASONS == (2026,)


def test_run_guards_check_everything_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cfg, "assert_clean_tree", lambda repo_root: calls.append("clean"))
    fetch = _games([("2027-03-24", "Final"), ("2027-09-26", "Final")])
    report = cfg.run_guards(
        end_date=END,
        end_source="https://www.mlb.com/schedule",
        end_verified_at="2027-09-30",
        comparison_path=tmp_path / "comparison_2027.json",
        fetch_fn=fetch,
        today=date(2027, 10, 1),
    )
    assert calls == ["clean"]
    assert report["season_end_date"] == "2027-09-26" and report["schedule"]["n_final"] == 2


def test_run_guards_refuse_a_dirty_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def dirty(repo_root: Path) -> None:
        raise cfg.WorkingTreeNotCleanError("dirty")

    monkeypatch.setattr(cfg, "assert_clean_tree", dirty)
    with pytest.raises(cfg.WorkingTreeNotCleanError):
        cfg.run_guards(
            end_date=END,
            end_source="s",
            end_verified_at="v",
            comparison_path=tmp_path / "c.json",
            fetch_fn=_games([("2027-03-24", "Final")]),
            today=date(2027, 10, 1),
        )
