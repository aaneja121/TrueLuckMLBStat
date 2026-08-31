"""Phase 8: regression coverage for the two production-blocking data
questions raised by the redesign, plus the global timestamp treatment.

Both questions were investigated at the data-generation layer, not from the
rendered UI. Neither turned out to be a scoring defect. These tests pin the
two findings so a future reader cannot "fix" either one back into a bug.

Fully offline: the only real artifact touched is the committed development
fixture `dashboard/explore_fixture`, which ships in the repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import build as dashboard_build
import pytest

from mlb_luck_score.scoring.attribution_ledger import FINAL_BASE_RUN_VALUE_MAP
from mlb_luck_score.scoring.component_confidence import (
    MODEL_STATUS_NOT_CALIBRATED,
    MODEL_STATUS_VALUES,
    normalize_model_status,
)
from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPLORE_FIXTURE = REPO_ROOT / "dashboard" / "explore_fixture"
PROSPECTIVE_SCORING = REPO_ROOT / "prospective" / "prospective_scoring.py"


def _explore_paths() -> dict[str, Path]:
    """The same mapping `build.py`'s own `--explore-artifacts-dir` performs."""
    return {
        "explore_players_path": EXPLORE_FIXTURE / "players.json",
        "explore_players_dir": EXPLORE_FIXTURE / "players",
        "explore_games_dir": EXPLORE_FIXTURE / "games",
        "explore_metadata_path": EXPLORE_FIXTURE / "explore-metadata.json",
        "explore_showcase_path": EXPLORE_FIXTURE / "showcase.json",
    }


def _fixture_plays() -> list[dict]:
    rows: list[dict] = []
    for shard in sorted((EXPLORE_FIXTURE / "games").glob("*.json")):
        rows += json.loads(shard.read_text())
    return rows


# ══ Blocker A — observed run value vs. the recorded outcome class ═════════


class TestObservedRunValueIsWholePlay:
    """`outcome_class` is Rc (what was officially recorded);
    `observed_run_value` is Rf (what the full accounting produced, realized
    baserunner advancement included). They are two different, both-correct
    facts about one play, and `play_ledger_schema`'s Version 2.0 notes say so
    at length -- Version 1.0 sourced Rc and FAILED reconciliation against the
    published `public_score` for 261 of 630 batters.

    So a row where the two disagree is the system working, not a defect. It
    must never be "corrected" in the renderer, in the exporter, or by
    reconciling `observed_run_value` back onto `DEFAULT_RUN_VALUE_MAP`.
    """

    def test_the_fixture_still_contains_a_diverging_row(self) -> None:
        """If this ever reaches zero the regression below stops proving
        anything, and the fixture needs a diverging play put back."""
        diverging = [
            r
            for r in _fixture_plays()
            if r.get("outcome_class")
            and r.get("observed_run_value") is not None
            and abs(r["observed_run_value"] - DEFAULT_RUN_VALUE_MAP[r["outcome_class"]]) > 1e-9
        ]
        assert diverging, (
            "no play in dashboard/explore_fixture has observed_run_value != "
            "DEFAULT_RUN_VALUE_MAP[outcome_class]; the Rc/Rf distinction is no longer covered"
        )

    def test_every_diverging_row_lands_on_a_real_final_base_value(self) -> None:
        """The distinguishing evidence between "legitimate Rf" and "a
        mapping bug".

        Rf for an advancement-modeled row is
        `FINAL_BASE_RUN_VALUE_MAP[batter_final_base]`, and that map REUSES
        `DEFAULT_RUN_VALUE_MAP` re-indexed by final base count rather than
        defining new numbers. So a legitimate diverging row is bit-exactly
        equal to one of the five run values -- just not the one its recorded
        outcome class names. A corrupt row would be an arbitrary float.
        """
        for row in _fixture_plays():
            observed = row.get("observed_run_value")
            if observed is None:
                continue
            assert any(abs(observed - v) < 1e-12 for v in FINAL_BASE_RUN_VALUE_MAP.values()), (
                f"{row['play_id']}: observed_run_value={observed!r} is not any "
                "FINAL_BASE_RUN_VALUE_MAP value, which a real Rf always is"
            )

    def test_the_self_referential_identity_holds_on_every_scored_row(self) -> None:
        """`contact_luck_runs == observed_run_value - expected_run_value` is
        the actual defining guarantee (the Rc-reconstruction cross-check was
        deliberately REMOVED in schema Version 2.0 because it assumed
        `contact_luck_runs` rebuilds from `outcome_class`, which stopped
        being true for advancement-modeled rows).
        """
        for row in _fixture_plays():
            if row.get("contact_luck_runs") is None:
                continue
            gap = row["observed_run_value"] - row["expected_run_value"]
            assert abs(gap - row["contact_luck_runs"]) < 1e-9, row["play_id"]

    def test_a_diverging_row_survives_the_build_verbatim(self, tmp_path: Path) -> None:
        """The dashboard displays; it never reconciles. A diverging play must
        reach the published per-game shard with the value the ledger gave it.
        """
        out_root, art_root = _seed_snapshot(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            **_explore_paths(),
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        source = {r["play_id"]: r for r in _fixture_plays()}
        published: dict[str, dict] = {}
        for shard in sorted((tmp_path / "dist" / "explore" / "games").glob("*.json")):
            for row in json.loads(shard.read_text()):
                published[row["play_id"]] = row

        diverging = [
            pid
            for pid, r in source.items()
            if r.get("outcome_class")
            and r.get("observed_run_value") is not None
            and abs(r["observed_run_value"] - DEFAULT_RUN_VALUE_MAP[r["outcome_class"]]) > 1e-9
        ]
        assert diverging
        for pid in diverging:
            assert published[pid]["observed_run_value"] == source[pid]["observed_run_value"]
            assert published[pid]["outcome_class"] == source[pid]["outcome_class"]
            assert published[pid]["contact_luck_runs"] == source[pid]["contact_luck_runs"]

    def test_the_play_page_explains_the_distinction_in_words(self, tmp_path: Path) -> None:
        """A reader who meets a diverging play must be told why the observed
        value can disagree with the recorded result, or the page looks wrong
        even though it is right.
        """
        out_root, art_root = _seed_snapshot(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            **_explore_paths(),
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        play_html = " ".join((tmp_path / "dist" / "plays" / "index.html").read_text().split())
        assert "baserunner advancement" in play_html
        assert "differ from what the recorded result alone would suggest" in play_html

        methodology = " ".join(
            (tmp_path / "dist" / "methodology" / "index.html").read_text().split()
        )
        assert "It counts the whole play, baserunner advancement included" in methodology


# ══ Blocker B — component_model_status carried a stringified boolean ══════


class TestComponentModelStatusVocabulary:
    """`_read_existing_gate_status` returns `str(node)` of whatever its
    configured key path finds, and the three components' gate files do not
    agree on a type: infield and advancement expose a status string at
    `gate_summary.overall_status`, while the outfield comparison file (one
    candidate, no selection) exposes only the boolean
    `validation_summary.per_candidate.measured_contact_only_v07.
    passes_basic_validation`.

    So `component_model_status.outfield` published the literal string
    "False" in a field documented as carrying the MODEL_STATUS_* vocabulary,
    while the per-play layer -- which routes the SAME raw value through
    `normalize_model_status` -- recorded `not_calibrated` for the same model
    in the same snapshot. One fact, two representations, one of them wrong.
    """

    def test_the_boolean_gate_reading_normalizes_to_not_calibrated(self) -> None:
        """Not an invented mapping: `normalize_model_status` already names
        this exact input in its docstring, and never guesses `calibrated`.
        """
        assert normalize_model_status("False") == MODEL_STATUS_NOT_CALIBRATED
        assert normalize_model_status("not_available_missing_file:x.json") == "unavailable"

    def test_normalization_is_idempotent_on_a_real_status(self) -> None:
        """Infield and advancement already supply valid vocabulary tokens, so
        routing all three through the normalizer must not disturb them.
        """
        for status in MODEL_STATUS_VALUES:
            assert normalize_model_status(status) == status

    def test_prospective_scoring_normalizes_component_model_status(self) -> None:
        """Structural, because running real prospective scoring is a network
        -and-model operation this offline suite cannot perform. It pins the
        two things that actually regressed: the published field is
        normalized, and the unnormalized gate reading is still recorded
        beside it so the mapping stays auditable.
        """
        source = PROSPECTIVE_SCORING.read_text()
        published = source[source.index('"component_model_status": {') :][:400]
        for component in ("outfield", "infield", "advancement"):
            assert f'"{component}": normalize_model_status({component}_status_raw)' in published

        raw_block = source[source.index('"component_model_status_raw": {') :][:400]
        for component in ("outfield", "infield", "advancement"):
            assert f'"{component}": {component}_status_raw,' in raw_block

    def test_the_status_page_still_shows_whatever_a_snapshot_recorded(self, tmp_path: Path) -> None:
        """The renderer is NOT part of the fix. An already-published
        immutable snapshot keeps its recorded value, and the dashboard shows
        it verbatim in the identifier register rather than reinterpreting it.
        """
        out_root, art_root = _seed_snapshot(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            build_timestamp="2026-01-01T12:00:00+00:00",
        )
        html = (tmp_path / "dist" / "status" / "index.html").read_text()
        # `dashboard_snapshot_fixtures` seeds the real snapshot's own
        # `outfield: "False"`, so this asserts the honest-display behaviour.
        assert '<dd class="status-code">False</dd>' in html


def _seed_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    from dashboard_snapshot_fixtures import default_player_record, write_snapshot

    out_root, art_root = tmp_path / "outputs", tmp_path / "artifacts"
    write_snapshot(
        out_root,
        art_root,
        directory_name="2026-01-01",
        data_through_date="2026-01-01",
        snapshot_label=None,
        generated_at="2026-01-02T00:00:00+00:00",
        players=[
            default_player_record(
                batter_id=1, batter_name="Alice Alpha", score=5.25, lower=1.1, upper=9.4
            )
        ],
    )
    return out_root, art_root


# ══ The global build timestamp ═══════════════════════════════════════════


class TestBuildTimestampIsFormattedEverywhere:
    """Phase 7 formatted `/status/`'s timestamps at the source and left the
    footer printing a raw ISO string on all eight routes -- one product, two
    treatments of the same kind of value.
    """

    @pytest.fixture(scope="class")
    def site(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        tmp_path = tmp_path_factory.mktemp("phase8_footer")
        out_root, art_root = _seed_snapshot(tmp_path)
        dashboard_build.build_dashboard(
            out_dir=tmp_path / "dist",
            outputs_root=out_root,
            artifacts_root=art_root,
            **_explore_paths(),
            build_timestamp="2026-01-03T18:07:00+00:00",
        )
        return tmp_path / "dist"

    def test_every_route_prints_a_reader_facing_build_time(self, site: Path) -> None:
        for page in sorted(site.rglob("index.html")):
            html = page.read_text()
            assert "Jan. 3, 2026, 18:07 UTC" in html, page

    def test_no_route_prints_a_raw_iso_build_timestamp(self, site: Path) -> None:
        """The machine value stays in `<time datetime>`, never in prose."""
        for page in sorted(site.rglob("index.html")):
            html = page.read_text()
            assert "generated 2026-01-03T18:07:00+00:00" not in html, page
            assert 'datetime="2026-01-03T18:07:00+00:00"' in html, page
