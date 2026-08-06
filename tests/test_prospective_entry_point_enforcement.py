"""Contact Luck v1.1: dedicated entry point CLI surface + authorization
-token enforcement, mirroring `tests/test_v1_entry_point_enforcement.py`'s
style for Version 1.0.

None of the tests here are ABOUT the working-tree-cleanliness guard itself
(see `tests/test_prospective_working_tree_guard.py` for those, which use
their own synthetic git repos) -- the autouse fixture below bypasses
`run_prospective_guards`'s (now-blocking) `assert_clean_working_tree` call so
these tests aren't incidentally sensitive to whether the REAL repository
working tree happens to be clean while the suite runs.
"""

from __future__ import annotations

import prospective_ingestion as pi
import pytest
import run_v1_1_2026_scoring as runner
from prospective_config import PROSPECTIVE_RAW_DIR

from mlb_luck_score.config import RAW_DATA_DIR


@pytest.fixture(autouse=True)
def _bypass_working_tree_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pi, "assert_clean_working_tree", lambda repo_root: None)


# ---------------------------------------------------------------------------
# CLI surface: exactly the three documented flags, no tuning surface
# ---------------------------------------------------------------------------


def test_cli_exposes_exactly_the_three_documented_flags() -> None:
    parser = runner.build_arg_parser()
    option_strings = {
        opt
        for action in parser._actions
        for opt in action.option_strings
        if opt.startswith("--") and opt != "--help"
    }
    assert option_strings == {"--data-through", "--snapshot-label", "--force-redownload"}


@pytest.mark.parametrize(
    "forbidden_flag",
    [
        "--model",
        "--model-version",
        "--calibration",
        "--calibrate",
        "--feature-set",
        "--features",
        "--threshold-set",
        "--threshold",
        "--allow-final-evaluation",
        "--class-weight",
    ],
)
def test_cli_never_exposes_a_tuning_or_selection_flag(forbidden_flag: str) -> None:
    parser = runner.build_arg_parser()
    option_strings = {opt for action in parser._actions for opt in action.option_strings}
    assert forbidden_flag not in option_strings


def test_data_through_is_required() -> None:
    parser = runner.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_threshold_set_is_hardcoded_never_a_cli_input() -> None:
    assert runner.DEFAULT_THRESHOLD_SET == "primary"
    parser = runner.build_arg_parser()
    args = parser.parse_args(["--data-through", "2026-04-15"])
    assert not hasattr(args, "threshold_set")


# ---------------------------------------------------------------------------
# ProspectiveAuthorization: unforgeable-in-process token
# ---------------------------------------------------------------------------


def test_authorization_object_is_required_not_a_bare_bool() -> None:
    with pytest.raises(pi.ProspectiveError):
        pi._require_authorization(True, "test_fn")  # type: ignore[arg-type]


def test_fabricated_token_is_rejected() -> None:
    forged = pi.ProspectiveAuthorization(
        token="not-a-real-token", granted_at="2026-01-01T00:00:00Z"
    )
    with pytest.raises(pi.ProspectiveError):
        pi._require_authorization(forged, "test_fn")


def test_minted_authorization_is_accepted() -> None:
    auth = pi.run_prospective_guards()
    pi._require_authorization(auth, "test_fn")  # must not raise


def test_each_guard_run_mints_a_distinct_token() -> None:
    auth_a = pi.run_prospective_guards()
    auth_b = pi.run_prospective_guards()
    assert auth_a.token != auth_b.token


# ---------------------------------------------------------------------------
# Ingestion functions refuse a namespace-violating output_dir/raw_dir
# ---------------------------------------------------------------------------


def test_ingest_raw_statcast_rejects_output_dir_outside_prospective_namespace() -> None:
    from prospective_config import NamespaceViolationError

    auth = pi.run_prospective_guards()
    with pytest.raises(NamespaceViolationError):
        pi.ingest_2026_raw_statcast(
            authorization=auth, data_through_date="2026-04-15", output_dir=RAW_DATA_DIR
        )


def test_ingest_game_metadata_rejects_output_dir_outside_prospective_namespace() -> None:
    from prospective_config import NamespaceViolationError

    auth = pi.run_prospective_guards()
    with pytest.raises(NamespaceViolationError):
        pi.ingest_2026_game_metadata(
            authorization=auth, data_through_date="2026-04-15", output_dir=RAW_DATA_DIR
        )


def test_ingest_sprint_speed_rejects_output_dir_outside_prospective_namespace() -> None:
    from prospective_config import NamespaceViolationError

    auth = pi.run_prospective_guards()
    with pytest.raises(NamespaceViolationError):
        pi.ingest_2026_sprint_speed(authorization=auth, output_dir=RAW_DATA_DIR)


def test_build_scoring_dataset_rejects_raw_dir_outside_prospective_namespace() -> None:
    from prospective_config import NamespaceViolationError

    auth = pi.run_prospective_guards()
    with pytest.raises(NamespaceViolationError):
        pi.build_2026_scoring_dataset(authorization=auth, raw_dir=RAW_DATA_DIR)


# ---------------------------------------------------------------------------
# Refuses any path resolving inside a sealed Version 1.0 namespace
# ---------------------------------------------------------------------------


def test_run_prospective_snapshot_rejects_sealed_outputs_root() -> None:
    from prospective_config import SEALED_V1_OUTPUTS_DIR, SealedNamespaceAccessError

    with pytest.raises(SealedNamespaceAccessError):
        runner.run_prospective_snapshot(
            data_through_date="2026-04-15", outputs_root=SEALED_V1_OUTPUTS_DIR
        )


def test_run_prospective_snapshot_rejects_sealed_artifacts_root() -> None:
    from prospective_config import SEALED_V1_ARTIFACTS_DIR, SealedNamespaceAccessError

    with pytest.raises(SealedNamespaceAccessError):
        runner.run_prospective_snapshot(
            data_through_date="2026-04-15", artifacts_root=SEALED_V1_ARTIFACTS_DIR
        )


def test_ingestion_module_default_raw_dir_matches_prospective_config() -> None:
    assert pi.PROSPECTIVE_RAW_DIR == PROSPECTIVE_RAW_DIR
