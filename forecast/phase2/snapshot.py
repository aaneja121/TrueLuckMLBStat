"""Contact Forecast Phase 2: pin one 2026 snapshot and adapt it, read-only.

## One pinned snapshot, verified before any metric

`resolve_snapshot` picks the newest complete prospective snapshot that can
support the evaluation, verifies every output file against the snapshot's own
`integrity_hashes.json`, and records the pin. The whole evaluation then reads
that one snapshot. Nothing here downloads, refreshes, regenerates or writes a
production snapshot, and nothing deploys.

A snapshot qualifies only if it carries a `play_ledger.parquet` -- the
per-batted-ball table the windows are built from. Season-aggregate outputs
alone cannot support a 100-BBE window, so a snapshot without a play ledger is
skipped with a recorded reason rather than worked around.

## The contact-stage adapter

Production's play ledger carries the FULL TELESCOPING quantities under
`observed_run_value` / `expected_run_value` / `contact_luck_runs` -- that is
Rf and Rf - E0, which is what contactluck.com publishes. R1 and every frozen
stage use the CONTACT STAGE: Rc, E0 and Rc - E0. Those are different
quantities and the difference is load-bearing, so this module never reuses the
production columns.

Instead it recomputes the contact stage from the ledger's own frozen contact
model output, which the snapshot already carries:

    E0 = sum(p(class) * run_value(class))   over the ledger's p_* columns
    Rc = run_value(observed outcome_class)
    Rc - E0

using `mlb_luck_score.scoring.weather_attribution.
compute_expected_run_value_vectorized` and
`mlb_luck_score.scoring.run_values.DEFAULT_RUN_VALUE_MAP` -- the same two
frozen pieces `forecast.score_development_seasons` used to build the
development ledgers. Nothing is retrained, refit or redesigned, and no
production scoring module is modified: this is a read-only projection of
values the snapshot already contains.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast.phase2.phase2_config import (
    PROJECT_ROOT,
    Phase2PathError,
    assert_phase2_path_allowed,
    assert_phase2_seasons_allowed,
)

logger = logging.getLogger(__name__)

PROSPECTIVE_OUTPUTS_ROOT = PROJECT_ROOT / "outputs" / "prospective" / "v1_1"
PROSPECTIVE_ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "prospective" / "v1_1"

#: Files a snapshot must carry to support this evaluation.
REQUIRED_SNAPSHOT_FILES: tuple[str, ...] = (
    "play_ledger.parquet",
    "play_ledger_metadata.json",
)

#: The contact-stage column names every frozen stage uses.
OBSERVED_COLUMN = "observed_contact_result_run_value"
DESERVED_COLUMN = "baseline_expected_contact_run_value"
SURPRISE_COLUMN = "contact_result_surprise"

#: Production play-ledger columns this module must never reuse as if they were
#: contact-stage values. Present only so their misuse can be refused.
FULL_TELESCOPING_COLUMNS: tuple[str, ...] = (
    "observed_run_value",
    "expected_run_value",
    "contact_luck_runs",
)


class Phase2SnapshotError(RuntimeError):
    """Raised when no usable pinned 2026 snapshot can be resolved."""


@dataclass(frozen=True)
class PinnedSnapshot:
    """One immutable 2026 snapshot, pinned for the whole evaluation."""

    label: str
    data_through_date: str
    season: int
    outputs_dir: Path
    artifacts_dir: Path
    manifest: dict[str, Any]
    play_ledger_metadata: dict[str, Any]
    integrity_hashes: dict[str, str]
    verified_files: dict[str, str]

    def as_record(self) -> dict[str, Any]:
        return {
            "snapshot_label": self.label,
            "data_through_date": self.data_through_date,
            "prospective_season": self.season,
            "outputs_dir": str(self.outputs_dir),
            "artifacts_dir": str(self.artifacts_dir),
            "score_version": self.manifest.get("score_version"),
            "repository_commit": self.manifest.get("repository_commit"),
            "working_tree_clean_at_generation": self.manifest.get("working_tree_clean"),
            "model_versions": self.manifest.get("model_versions"),
            "play_ledger_metadata": self.play_ledger_metadata,
            "integrity_hashes_recorded": self.integrity_hashes,
            "integrity_hashes_verified": self.verified_files,
            "integrity_verified": True,
            "read_only": True,
            "regenerated_or_overwritten": False,
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_snapshots() -> list[str]:
    """Every snapshot directory name, newest last, that is not a refresh variant."""
    if not PROSPECTIVE_OUTPUTS_ROOT.exists():
        return []
    return sorted(
        p.name
        for p in PROSPECTIVE_OUTPUTS_ROOT.iterdir()
        if p.is_dir() and not p.name.endswith("__refreshed")
    )


def resolve_snapshot(*, label: str | None = None) -> PinnedSnapshot:
    """Pin the newest snapshot that can support the evaluation, and verify it.

    Args:
        label: Pin a specific snapshot directory instead of the newest usable
            one. Provided so a later re-run can reproduce this exact
            evaluation, not so a snapshot can be shopped for.

    Raises:
        Phase2SnapshotError: If no snapshot carries a play ledger, if the
            requested snapshot is missing, or if any file fails its recorded
            integrity hash.
    """
    candidates = available_snapshots()
    if not candidates:
        raise Phase2SnapshotError(
            f"No prospective snapshots found under {PROSPECTIVE_OUTPUTS_ROOT}. Phase 2 "
            "does not generate one -- stop and report rather than improvising a data path."
        )

    ordered = [label] if label else list(reversed(candidates))
    skipped: list[dict[str, str]] = []
    for name in ordered:
        outputs_dir = PROSPECTIVE_OUTPUTS_ROOT / name
        artifacts_dir = PROSPECTIVE_ARTIFACTS_ROOT / name
        if not outputs_dir.exists():
            raise Phase2SnapshotError(f"Requested snapshot {name!r} does not exist")
        missing = [f for f in REQUIRED_SNAPSHOT_FILES if not (outputs_dir / f).exists()]
        if missing:
            skipped.append({"snapshot": name, "reason": f"missing {missing}"})
            continue
        if not (artifacts_dir / "manifest.json").exists():
            skipped.append({"snapshot": name, "reason": "no manifest"})
            continue
        pinned = _verify_snapshot(name, outputs_dir, artifacts_dir)
        if skipped:
            logger.info("skipped %d newer snapshot(s): %s", len(skipped), skipped)
        return pinned

    raise Phase2SnapshotError(
        "No prospective snapshot carries a per-batted-ball play ledger, so no "
        f"100-BBE window can be built. Skipped: {skipped}. Phase 2 stops here rather "
        "than generating a snapshot or improvising a new data path."
    )


def _verify_snapshot(name: str, outputs_dir: Path, artifacts_dir: Path) -> PinnedSnapshot:
    """Check every recorded output hash before the snapshot is used."""
    for path in (outputs_dir, artifacts_dir):
        assert_phase2_path_allowed(path)

    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    season = int(manifest["prospective_season"])
    assert_phase2_seasons_allowed(season)

    integrity_path = artifacts_dir / "integrity_hashes.json"
    if not integrity_path.exists():
        raise Phase2SnapshotError(f"Snapshot {name!r} has no integrity_hashes.json")
    recorded = json.loads(integrity_path.read_text())

    # Keys are recorded as "<outputs|artifacts>/<filename>", relative to this
    # snapshot's own two directories.
    roots = {"outputs": outputs_dir, "artifacts": artifacts_dir}
    verified: dict[str, str] = {}
    mismatched: list[str] = []
    for relative, expected in sorted(recorded.items()):
        prefix, _, filename = relative.partition("/")
        root = roots.get(prefix)
        if root is None:
            mismatched.append(f"{relative}: unrecognised root {prefix!r}")
            continue
        target = root / filename
        if not target.exists():
            mismatched.append(f"{relative}: file missing")
            continue
        actual = _file_sha256(target)
        if actual != expected:
            mismatched.append(f"{relative}: {actual[:12]} != recorded {expected[:12]}")
            continue
        verified[relative] = actual
    if mismatched:
        raise Phase2SnapshotError(
            f"Snapshot {name!r} failed its own integrity hashes: {mismatched}. "
            "The pinned data is not the data that was generated."
        )

    return PinnedSnapshot(
        label=str(manifest.get("snapshot_label") or name),
        data_through_date=str(manifest["data_through_date"]),
        season=season,
        outputs_dir=outputs_dir,
        artifacts_dir=artifacts_dir,
        manifest=manifest,
        play_ledger_metadata=json.loads((outputs_dir / "play_ledger_metadata.json").read_text()),
        integrity_hashes=recorded,
        verified_files=verified,
    )


def load_contact_stage_ledger(snapshot: PinnedSnapshot) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Project the pinned play ledger onto the frozen CONTACT-STAGE quantities.

    Read-only. Recomputes E0 from the ledger's own frozen contact-model
    probabilities and Rc from its recorded outcome, using the same two frozen
    production helpers the development ledgers were built with. Nothing is
    retrained and no production module is modified.

    Returns:
        `(ledger, audit)` where `ledger` carries the forecast schema and
        `audit` records the projection, including a check that the derived
        contact stage is NOT the production full-telescoping quantity.
    """
    from mlb_luck_score.config import CLASS_ORDER
    from mlb_luck_score.scoring.run_values import DEFAULT_RUN_VALUE_MAP
    from mlb_luck_score.scoring.weather_attribution import (
        compute_expected_run_value_vectorized,
    )

    path = snapshot.outputs_dir / "play_ledger.parquet"
    assert_phase2_path_allowed(path)
    raw = pd.read_parquet(path)
    assert_phase2_seasons_allowed(sorted(int(s) for s in raw["season"].unique()))

    probability_columns = [f"p_{cls}" for cls in CLASS_ORDER]
    missing = [c for c in probability_columns if c not in raw.columns]
    if missing:
        raise Phase2SnapshotError(
            f"The pinned play ledger lacks contact-model probability column(s) "
            f"{missing}, so the contact stage cannot be derived read-only."
        )

    # The prospective ledger uses pandas NULLABLE dtypes (Int64/Float64/string).
    # They propagate `pd.NA` rather than NaN and make `groupby(...).cumsum()`
    # fail on an object column downstream, which is the hazard RESEARCH_RULES.md
    # flags under "pd.NA breaks more than SimpleImputer". Everything is coerced
    # to plain numpy dtypes here, once, at the boundary.
    proba = raw[probability_columns].astype("float64")
    proba.columns = list(CLASS_ORDER)
    resolved = raw["outcome_class"].notna().to_numpy(dtype=bool)

    deserved = compute_expected_run_value_vectorized(proba).to_numpy(dtype=float)
    observed = (
        raw["outcome_class"].astype(object).map(DEFAULT_RUN_VALUE_MAP).astype("float64")
    ).to_numpy(dtype=float)

    ledger = pd.DataFrame(
        {
            "event_id": raw["play_id"].astype(str).to_numpy(),
            "batter": raw["batter_id"].astype("int64").to_numpy(),
            "season": raw["season"].astype("int64").to_numpy(),
            "game_date": raw["game_date"].astype(str).to_numpy(),
            "game_pk": raw["game_pk"].astype("int64").to_numpy(),
            "at_bat_number": raw["at_bat_number"].astype("int64").to_numpy(),
            "pitch_number": raw["pitch_number"].astype("int64").to_numpy(),
            "stand": raw["stand"].astype(object).to_numpy(),
            "launch_speed": raw["launch_speed"].astype("float64").to_numpy(),
            "launch_angle": raw["launch_angle"].astype("float64").to_numpy(),
            "bb_type": raw["bb_type"].astype(object).to_numpy(),
            "spray_angle_approx": raw["spray_angle_approx"].astype("float64").to_numpy(),
            "outcome_class": raw["outcome_class"].astype(object).to_numpy(),
            OBSERVED_COLUMN: observed,
            DESERVED_COLUMN: deserved,
        }
    )
    ledger[SURPRISE_COLUMN] = ledger[OBSERVED_COLUMN] - ledger[DESERVED_COLUMN]
    # Production nulls every result-linked column on an unresolved play; the
    # development ledgers reproduce that, and so does this.
    for column in (OBSERVED_COLUMN, DESERVED_COLUMN, SURPRISE_COLUMN):
        ledger.loc[~resolved, column] = np.nan

    audit = _projection_audit(raw, ledger, resolved=pd.Series(resolved, index=raw.index))
    return ledger, audit


def _projection_audit(
    raw: pd.DataFrame, ledger: pd.DataFrame, *, resolved: pd.Series
) -> dict[str, Any]:
    """Record what the projection did, and prove it is the contact stage.

    The check that matters: the derived contact-stage surprise must NOT equal
    the production `contact_luck_runs` column, which holds the full
    telescoping Rf - E0. If they matched, the projection would have quietly
    reproduced the published metric instead of the research quantity.
    """
    comparison: dict[str, Any] = {}
    if "contact_luck_runs" in raw.columns:
        both = resolved & raw["contact_luck_runs"].notna()
        derived = ledger.loc[both, SURPRISE_COLUMN].to_numpy(dtype=float)
        published = raw.loc[both, "contact_luck_runs"].to_numpy(dtype=float)
        differences = np.abs(derived - published)
        comparison = {
            "n_compared": int(both.sum()),
            "identical_to_published_full_telescoping": bool(
                differences.size and float(differences.max()) < 1e-12
            ),
            "mean_absolute_difference": (float(differences.mean()) if differences.size else None),
            "correlation": (
                float(np.corrcoef(derived, published)[0, 1]) if differences.size > 1 else None
            ),
            "note": (
                "The contact stage (Rc - E0) is DIFFERENT from the published full "
                "telescoping quantity (Rf - E0). A max difference of zero would mean "
                "the projection had reproduced the published metric by mistake."
            ),
        }
    return {
        "method": (
            "read-only projection: E0 recomputed from the snapshot's own frozen "
            "contact-model probabilities via compute_expected_run_value_vectorized; "
            "Rc from the recorded outcome_class via DEFAULT_RUN_VALUE_MAP"
        ),
        "frozen_helpers_used": [
            "mlb_luck_score.scoring.weather_attribution.compute_expected_run_value_vectorized",
            "mlb_luck_score.scoring.run_values.DEFAULT_RUN_VALUE_MAP",
        ],
        "production_scoring_code_modified": False,
        "model_retrained_or_redesigned": False,
        "full_telescoping_columns_reused": False,
        "n_rows": int(len(ledger)),
        "n_resolved": int(resolved.sum()),
        "n_unresolved_excluded_from_denominator": int((~resolved).sum()),
        "contact_stage_versus_published_full_telescoping": comparison,
    }


def assert_no_full_telescoping_columns(frame: pd.DataFrame) -> None:
    """Refuse a frame that smuggled a production Rf column into the study.

    Raises:
        Phase2PathError: If any full-telescoping column is present.
    """
    offending = sorted(set(frame.columns) & set(FULL_TELESCOPING_COLUMNS))
    if offending:
        raise Phase2PathError(
            f"Full-telescoping column(s) {offending} reached the Phase 2 ledger. The "
            "forecast research uses the contact stage (Rc - E0), never the published "
            "Rf - E0 quantity."
        )
