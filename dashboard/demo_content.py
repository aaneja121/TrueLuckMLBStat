"""Contact Luck v1.3.0: view-model for the `/demo/` ("Why Contact Luck
Exists") page.

Reads ONLY the committed `dashboard/demo_fixture.json` -- produced once,
offline, by `demo/build_demo_fixture.py` (which imports `mlb_luck_score`
freely; this module must never do that -- see `tests/
test_dashboard_isolation.py`). Every value here is copied verbatim from that
file; the only computation this module performs is a pure-arithmetic
reconciliation check on the numbers already in the fixture (no scoring
formula is reimplemented, no probability model is invoked).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DemoContentError",
    "DemoExample",
    "DemoPageData",
    "load_demo_page_data",
]

_RECONCILIATION_TOLERANCE = 1e-6

#: Display order/labels for the 5-class outcome distribution shown in Stage
#: 2. Matches `mlb_luck_score.config.CLASS_ORDER` (duplicated here, not
#: imported -- see this module's docstring on the read-only boundary); a
#: mismatch between this tuple and the fixture's own `probabilities` keys is
#: caught by `_validate_probabilities` below (it validates every key
#: actually present in the fixture, regardless of this display list).
CLASS_DISPLAY_ORDER: tuple[str, ...] = ("out", "single", "double", "triple", "home_run")
CLASS_DISPLAY_LABELS: dict[str, str] = {
    "out": "Out",
    "single": "1B",
    "double": "2B",
    "triple": "3B",
    "home_run": "HR",
}


class DemoContentError(Exception):
    """Raised when the committed demo fixture is missing, malformed, or internally
    inconsistent -- never a reason to fabricate a placeholder example.
    """


@dataclass(frozen=True)
class DemoExample:
    example_id: str
    narrative_label: str
    contact_description: str
    batter_name: str
    batter_team: str
    opponent_team: str
    game_date: str
    description: str
    exit_velocity_mph: float
    launch_angle_deg: float
    spray_angle_deg: float
    bb_type: str
    hit_distance_ft: float
    probabilities: dict[str, float]
    expected_run_value: float
    outcome_class: str
    outcome_label: str
    observed_run_value: float
    contact_luck_runs: float
    explanation: str
    probability_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class DemoPageData:
    examples: list[DemoExample]
    model_configuration: dict[str, Any] = field(default_factory=dict)
    fixture_version: str = ""


def _validate_probabilities(probabilities: dict[str, float], example_id: str) -> None:
    for cls, p in probabilities.items():
        if not isinstance(p, int | float) or p != p or p in (float("inf"), float("-inf")):
            raise DemoContentError(f"{example_id}: probability for {cls!r} is not finite ({p!r})")
        if p < 0.0 or p > 1.0:
            raise DemoContentError(f"{example_id}: probability for {cls!r} out of [0, 1] ({p!r})")
    total = sum(probabilities.values())
    if abs(total - 1.0) > 1e-3:
        raise DemoContentError(f"{example_id}: probabilities sum to {total!r}, expected ~1.0")


_EXPLANATION_FAVORABLE = (
    "The recorded outcome was worth more than the contact was expected to produce."
)
_EXPLANATION_UNFAVORABLE = "The contact was worth more than the recorded outcome."


def _explanation_for(contact_luck_runs: float) -> str:
    return _EXPLANATION_FAVORABLE if contact_luck_runs >= 0 else _EXPLANATION_UNFAVORABLE


def _build_example(raw: dict[str, Any]) -> DemoExample:
    example_id = raw.get("example_id", "<unknown>")
    provenance = raw["provenance"]
    contact = raw["contact"]
    expectation = raw["expectation"]
    reality = raw["reality"]

    probabilities = expectation["probabilities"]
    _validate_probabilities(probabilities, example_id)

    expected_run_value = float(expectation["expected_run_value"])
    observed_run_value = float(reality["observed_run_value"])
    contact_luck_runs = float(raw["contact_luck_runs"])
    gap = abs((observed_run_value - expected_run_value) - contact_luck_runs)
    if gap > _RECONCILIATION_TOLERANCE:
        raise DemoContentError(
            f"{example_id}: observed_run_value - expected_run_value does not reconcile with "
            f"contact_luck_runs (gap={gap!r}, tolerance={_RECONCILIATION_TOLERANCE!r})"
        )

    probability_rows = [
        {
            "outcome_class": cls,
            "label": CLASS_DISPLAY_LABELS.get(cls, cls),
            "probability": probabilities[cls],
            "probability_pct": probabilities[cls] * 100.0,
            "is_observed": cls == reality["outcome_class"],
        }
        for cls in CLASS_DISPLAY_ORDER
        if cls in probabilities
    ]

    return DemoExample(
        example_id=example_id,
        narrative_label=raw["narrative_label"],
        contact_description=raw["contact_description"],
        batter_name=provenance["batter_name"],
        batter_team=provenance["batter_team"],
        opponent_team=provenance["opponent_team"],
        game_date=provenance["game_date"],
        description=provenance["description"],
        exit_velocity_mph=float(contact["exit_velocity_mph"]),
        launch_angle_deg=float(contact["launch_angle_deg"]),
        spray_angle_deg=float(contact["spray_angle_deg"]),
        bb_type=contact["bb_type"],
        hit_distance_ft=float(contact["hit_distance_ft"]),
        probabilities=probabilities,
        expected_run_value=expected_run_value,
        outcome_class=reality["outcome_class"],
        outcome_label=reality["outcome_label"],
        observed_run_value=observed_run_value,
        contact_luck_runs=contact_luck_runs,
        explanation=_explanation_for(contact_luck_runs),
        probability_rows=probability_rows,
    )


#: The exact set of example ids the `/demo/` page is built around -- see
#: `demo/build_demo_fixture.py`'s `EXAMPLE_SPECS`. Enforced here so a
#: malformed or partially-edited fixture fails the build loudly rather than
#: silently rendering a page with the wrong (or missing) examples.
REQUIRED_EXAMPLE_IDS = ("hard_contact_out", "weak_contact_single")


def load_demo_page_data(path: Path) -> DemoPageData:
    """Load and validate the committed demo fixture at `path`.

    Raises:
        DemoContentError: if the file is missing/malformed, doesn't contain
            exactly `REQUIRED_EXAMPLE_IDS`, has an invalid probability
            distribution, or fails the observed/expected/Contact-Luck
            reconciliation check.
    """
    if not path.exists():
        raise DemoContentError(
            f"Demo fixture not found at {path}. It is committed reference data, produced by "
            "`demo/build_demo_fixture.py` -- it is never regenerated by the dashboard build."
        )
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise DemoContentError(f"failed to parse demo fixture {path}: {exc}") from exc

    raw_examples = raw.get("examples", [])
    found_ids = [e.get("example_id") for e in raw_examples]
    if sorted(found_ids) != sorted(REQUIRED_EXAMPLE_IDS):
        raise DemoContentError(
            f"Demo fixture at {path} has example_ids {found_ids}, expected exactly "
            f"{list(REQUIRED_EXAMPLE_IDS)}"
        )

    examples_by_id = {e["example_id"]: _build_example(e) for e in raw_examples}
    ordered_examples = [examples_by_id[eid] for eid in REQUIRED_EXAMPLE_IDS]

    return DemoPageData(
        examples=ordered_examples,
        model_configuration=raw.get("model_configuration", {}),
        fixture_version=raw.get("demo_fixture_version", ""),
    )
