"""`scripts/verify_dashboard_artifact.py` -- the promotion gate.

This is the last check between a built artifact and the public site, so
what matters is not that it passes a good artifact but that it REFUSES
every shape of bad one. A gate that only ever says yes is decoration.

The cases below are drawn from real ways a `dist` can look deployable and
not be: it is a different build than the operator thinks; it is older than
what is already live (the incident this gate exists for); it disagrees with
itself because two stages of the build saw different snapshots; it
publishes a pitcher season nobody authorized; it is truncated.

Fully offline and synthetic -- every artifact here is assembled in tmp_path
and the live-site manifest is injected, never fetched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from verify_dashboard_artifact import (
    ArtifactVerificationError,
    verify_artifact,
)

COMMIT = "303b5f3dd8f58c1e58847f8d9c3718f1198d96e3"
LIVE = {"data_through_date": "2026-09-08"}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    """A well-formed artifact: 2 hitters, 1 pitcher season, 2 games."""
    dist = tmp_path / "dist"
    _write(
        dist / "data" / "dashboard_build_manifest.json",
        json.dumps(
            {
                "data_through_date": "2026-09-08",
                "repository_commit": COMMIT,
                "pitcher_seasons": [2024],
                "pitcher_count": 2,
                "player_count": 2,
                "qualified_count": 1,
                "snapshot_manifest_hash_reference": "abc",
            }
        ),
    )
    _write(
        dist / "explore" / "explore-metadata.json",
        json.dumps(
            {
                "data_through_date": "2026-09-08",
                "play_count": 500,
                "game_count": 2,
                "player_count": 2,
            }
        ),
    )
    _write(dist / "explore" / "players.json", "[]")
    for game in ("1", "2"):
        _write(dist / "explore" / "games" / f"{game}.json", "{}")
    for player in ("1", "2"):
        _write(dist / "explore" / "players" / f"{player}.json", "{}")
        _write(dist / "players" / player / "index.html", "<html></html>")
    _write(dist / "pitchers" / "2024" / "index.html", "<html></html>")
    for pitcher in ("10", "11"):
        _write(dist / "pitchers" / "2024" / pitcher / "index.html", "<html></html>")
    for shell in ("index.html", "methodology/index.html", "status/index.html"):
        _write(dist / shell, "<html></html>")
    _write(dist / "static" / "style.css", "body{}")
    _write(dist / "static" / "app.js", "//")
    return dist


def _verify(dist: Path, **overrides):
    kwargs = {
        "expected_data_through": "2026-09-08",
        "expected_repository_commit": COMMIT,
        "live_manifest": LIVE,
        "seasons": (2024,),
    }
    kwargs.update(overrides)
    return verify_artifact(dist, **kwargs)


class TestAGoodArtifactPasses:
    def test_it_accepts_a_complete_consistent_artifact(self, artifact: Path) -> None:
        report = _verify(artifact)
        assert report["data_through_date"] == "2026-09-08"
        assert report["pitcher_seasons"] == [2024]

    def test_exact_operator_counts_are_accepted_when_they_match(self, artifact: Path) -> None:
        report = _verify(
            artifact,
            expected_counts={
                "player_count": 2,
                "qualified_count": 1,
                "pitcher_count": 2,
                "play_count": 500,
                "game_count": 2,
            },
        )
        assert report["play_count"] == 500


class TestItRefusesTheWrongBuild:
    def test_a_different_date_than_the_operator_named(self, artifact: Path) -> None:
        with pytest.raises(ArtifactVerificationError, match="data_through_date is"):
            _verify(artifact, expected_data_through="2026-09-07")

    def test_a_different_commit_than_the_operator_named(self, artifact: Path) -> None:
        with pytest.raises(ArtifactVerificationError, match="repository_commit is"):
            _verify(artifact, expected_repository_commit="0" * 40)

    @pytest.mark.parametrize(
        "key,wrong",
        [
            ("player_count", 999),
            ("qualified_count", 999),
            ("pitcher_count", 999),
            ("play_count", 999),
            ("game_count", 999),
        ],
    )
    def test_any_operator_count_mismatch(self, artifact: Path, key, wrong) -> None:
        with pytest.raises(ArtifactVerificationError, match=key):
            _verify(artifact, expected_counts={key: wrong})


class TestItRefusesARollback:
    """The incident this gate exists for."""

    def test_an_artifact_older_than_production_is_refused(self, artifact: Path) -> None:
        with pytest.raises(ArtifactVerificationError, match="ROLLBACK"):
            _verify(artifact, live_manifest={"data_through_date": "2026-09-09"})

    def test_the_same_date_as_production_is_allowed(self, artifact: Path) -> None:
        """A feature-only promotion republishes the SAME hitter date on
        purpose -- that is the whole point, and must not be blocked."""
        assert _verify(artifact, live_manifest={"data_through_date": "2026-09-08"})

    def test_a_newer_artifact_is_allowed(self, artifact: Path) -> None:
        assert _verify(artifact, live_manifest={"data_through_date": "2026-09-01"})

    def test_an_unreadable_live_site_refuses_rather_than_guesses(self, artifact: Path) -> None:
        """If we cannot tell what production serves, we cannot rule out a
        rollback, so we do not proceed."""
        with pytest.raises(ArtifactVerificationError, match="could not read the live site"):
            _verify(artifact, live_manifest=None)

    def test_a_rollback_can_be_forced_deliberately(self, artifact: Path, capsys) -> None:
        report = _verify(
            artifact, live_manifest={"data_through_date": "2026-09-09"}, allow_rollback=True
        )
        assert report["data_through_date"] == "2026-09-08"
        assert "ROLLBACK" in capsys.readouterr().out


class TestItRefusesAnArtifactThatDisagreesWithItself:
    """An operator can only confirm what they already believe. These catch
    a build that is wrong in a way nobody thought to type in."""

    def test_explore_describing_a_different_snapshot(self, artifact: Path) -> None:
        meta = artifact / "explore" / "explore-metadata.json"
        data = json.loads(meta.read_text())
        data["data_through_date"] = "2026-09-01"
        meta.write_text(json.dumps(data))
        with pytest.raises(ArtifactVerificationError, match="two different snapshots"):
            _verify(artifact)

    def test_explore_game_count_not_matching_the_files_on_disk(self, artifact: Path) -> None:
        (artifact / "explore" / "games" / "2.json").unlink()
        with pytest.raises(ArtifactVerificationError, match="game files are present"):
            _verify(artifact)

    def test_explore_player_count_not_matching_the_manifest(self, artifact: Path) -> None:
        meta = artifact / "explore" / "explore-metadata.json"
        data = json.loads(meta.read_text())
        data["player_count"] = 3
        meta.write_text(json.dumps(data))
        with pytest.raises(ArtifactVerificationError, match="player_count"):
            _verify(artifact)

    def test_manifest_player_count_not_matching_the_pages_built(self, artifact: Path) -> None:
        import shutil

        shutil.rmtree(artifact / "players" / "2")
        with pytest.raises(ArtifactVerificationError, match="hitter pages built"):
            _verify(artifact)

    def test_manifest_pitcher_count_not_matching_the_cards_built(self, artifact: Path) -> None:
        import shutil

        shutil.rmtree(artifact / "pitchers" / "2024" / "11")
        with pytest.raises(ArtifactVerificationError, match="pitcher card pages built"):
            _verify(artifact)


class TestItRefusesUnauthorizedPitcherSeasons:
    def test_a_declared_season_outside_the_gate(self, artifact: Path) -> None:
        manifest = artifact / "data" / "dashboard_build_manifest.json"
        data = json.loads(manifest.read_text())
        data["pitcher_seasons"] = [2024, 2025]
        manifest.write_text(json.dumps(data))
        with pytest.raises(ArtifactVerificationError, match="unauthorized pitcher season"):
            _verify(artifact)

    def test_a_route_for_a_season_outside_the_gate(self, artifact: Path) -> None:
        """Routes on disk are checked independently of what the manifest
        declares -- a route nobody declared is still publicly reachable."""
        _write(artifact / "pitchers" / "2026" / "index.html", "<html></html>")
        with pytest.raises(ArtifactVerificationError, match="not authorized"):
            _verify(artifact)

    def test_declared_seasons_and_routes_must_agree(self, artifact: Path) -> None:
        manifest = artifact / "data" / "dashboard_build_manifest.json"
        data = json.loads(manifest.read_text())
        data["pitcher_seasons"] = []
        manifest.write_text(json.dumps(data))
        with pytest.raises(ArtifactVerificationError, match="routes exist for"):
            _verify(artifact)

    def test_the_real_shipped_gate_authorizes_2024_only(self) -> None:
        from verify_dashboard_artifact import authorized_pitcher_seasons

        assert authorized_pitcher_seasons() == (2024,)


class TestItRefusesATruncatedArtifact:
    @pytest.mark.parametrize(
        "relative",
        ["index.html", "static/app.js", "explore/players.json", "status/index.html"],
    )
    def test_a_missing_required_file(self, artifact: Path, relative) -> None:
        (artifact / relative).unlink()
        with pytest.raises(ArtifactVerificationError, match="not a complete site"):
            _verify(artifact)

    def test_a_zero_byte_file(self, artifact: Path) -> None:
        """A half-written upload looks present until you check its size."""
        (artifact / "static" / "style.css").write_text("")
        with pytest.raises(ArtifactVerificationError, match="truncated"):
            _verify(artifact)

    def test_a_missing_directory_entirely(self, tmp_path: Path) -> None:
        with pytest.raises(ArtifactVerificationError, match="not a directory"):
            _verify(tmp_path / "nope")


class TestItNeverRepairs:
    def test_a_refused_artifact_is_left_byte_for_byte_alone(self, artifact: Path) -> None:
        before = {p: p.read_bytes() for p in sorted(artifact.rglob("*")) if p.is_file()}
        with pytest.raises(ArtifactVerificationError):
            _verify(artifact, expected_data_through="2026-01-01")
        after = {p: p.read_bytes() for p in sorted(artifact.rglob("*")) if p.is_file()}
        assert before == after


PROMOTE_WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "promote-dashboard-artifact.yml"
)
PUBLISH_WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github" / "workflows" / "publish-prospective.yml"
)


def _workflow(path: Path) -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(path.read_text())


class TestThePromotionWorkflowCannotDoAnythingElse:
    """The promotion workflow's safety is structural, not procedural: it
    does not decline to score, archive or rebuild -- it contains no way to.

    These assert on the workflow file itself, because "we never call the
    scoring script" is only true for as long as nobody adds a step that
    does, and that is exactly the kind of edit that looks harmless in
    review.
    """

    def _steps(self) -> list[dict]:
        return _workflow(PROMOTE_WORKFLOW)["jobs"]["promote"]["steps"]

    def _all_run_text(self) -> str:
        return "\n".join(s.get("run", "") for s in self._steps())

    def _executable_content(self) -> str:
        """Everything the runner actually EXECUTES -- `run:` bodies, `uses:`
        and every `env:` value. Deliberately not the raw file: the header
        comment names the scripts this workflow does not call, and prose
        explaining an absence must not read as the presence."""
        parts = []
        for step in self._steps():
            parts.append(step.get("run", ""))
            parts.append(str(step.get("uses", "")))
            parts.append(str(step.get("env", {})))
            parts.append(str(step.get("with", {})))
        return "\n".join(parts)

    @pytest.mark.parametrize(
        "forbidden",
        [
            "publish_snapshot.sh",
            "run_v1_1_2026_scoring.py",
            "archive_snapshot.py",
            "--sync-history",
            "dashboard/build.py",
            "generate_production_explorer_artifacts.py",
        ],
    )
    def test_it_never_invokes_the_build_or_archive_pipeline(self, forbidden) -> None:
        assert forbidden not in self._executable_content(), forbidden

    def test_it_is_given_no_r2_credentials(self) -> None:
        """It cannot write to R2 because it is never handed the keys."""
        content = self._executable_content()
        for secret in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"):
            assert secret not in content, secret

    def test_it_is_manual_only(self) -> None:
        """No schedule: a promotion is always someone's decision."""
        triggers = _workflow(PROMOTE_WORKFLOW)[True]
        assert set(triggers) == {"workflow_dispatch"}

    def test_it_demands_an_explicit_source_run_and_identity(self) -> None:
        inputs = _workflow(PROMOTE_WORKFLOW)[True]["workflow_dispatch"]["inputs"]
        for required in (
            "source_run_id",
            "artifact_name",
            "expected_data_through",
            "expected_repository_commit",
        ):
            assert inputs[required]["required"] is True, required

    def test_rollback_is_off_by_default(self) -> None:
        inputs = _workflow(PROMOTE_WORKFLOW)[True]["workflow_dispatch"]["inputs"]
        assert inputs["allow_rollback"]["default"] is False

    def test_the_gate_runs_before_wrangler_is_even_installed(self) -> None:
        """Ordering is the guarantee: a failed gate means the deploy step is
        never reached, and `set -e` plus step ordering is what enforces it."""
        names = [s["name"] for s in self._steps()]
        verify = next(i for i, n in enumerate(names) if "Verify the artifact" in n)
        node = next(i for i, n in enumerate(names) if "Node.js" in n)
        deploy = next(i for i, n in enumerate(names) if "Deploy the verified" in n)
        assert verify < node < deploy

    def test_the_source_run_is_checked_before_the_artifact_is_downloaded(self) -> None:
        names = [s["name"] for s in self._steps()]
        confirm = next(i for i, n in enumerate(names) if "Confirm the source run" in n)
        download = next(i for i, n in enumerate(names) if "Download the artifact" in n)
        assert confirm < download

    def test_it_deploys_the_downloaded_directory_verbatim(self) -> None:
        """The bytes inspected must be the bytes shipped -- no copy, no
        rewrite, no rebuild between the gate and the deploy."""
        run_text = self._all_run_text()
        assert "npx wrangler pages deploy promote-dist --project-name=contact-luck" in run_text
        verify_step = next(s for s in self._steps() if "Verify the artifact" in s["name"])
        assert "--dist-dir promote-dist" in verify_step["run"]

    def test_it_downloads_by_run_id(self) -> None:
        download = next(s for s in self._steps() if "Download the artifact" in s["name"])
        assert download["with"]["run-id"] == "${{ inputs.source_run_id }}"

    def test_it_shares_the_publish_loop_s_concurrency_group(self) -> None:
        """Both end at `wrangler pages deploy` for one project, so they must
        queue rather than race -- two runs deploying different artifacts is
        how a stale build wins by finishing last."""
        promote = _workflow(PROMOTE_WORKFLOW)["concurrency"]
        publish = _workflow(PUBLISH_WORKFLOW)["concurrency"]
        assert promote["group"] == publish["group"]
        assert promote["cancel-in-progress"] is False

    def test_the_publish_workflow_is_unchanged_by_this_feature(self) -> None:
        """The promotion path is additive: the normal loop still owns
        scoring, archiving and its own deploy."""
        text = PUBLISH_WORKFLOW.read_text()
        assert "publish_snapshot.sh" in text
        assert "promote-dist" not in text
