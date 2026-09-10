"""Gate a built dashboard artifact before it is promoted to production.

This is the last thing that runs between a `dashboard/dist` produced by
some earlier CI run and `wrangler pages deploy`. It builds nothing, scores
nothing, and touches no snapshot: it inspects a directory and decides
whether that directory may become the public site.

It exists because of a specific past incident -- a stale local `dist` was
deployed and rolled production data backward. Every check below is aimed at
one of the ways a directory can look deployable and not be:

* it is a DIFFERENT build than the operator thinks (wrong commit, wrong
  date) -- caught by requiring the operator to state both and matching them;
* it is OLDER than what is already live -- caught by the anti-rollback
  check, which is the one this script exists for;
* it is INTERNALLY inconsistent (the Explore payload describes a different
  snapshot than the manifest, or the file counts contradict the metadata)
  -- caught by cross-checking the artifact against itself rather than
  against operator input, since an operator can only confirm what they
  already believe;
* it publishes a pitcher season nobody authorized -- caught against
  `dashboard_config.PITCHER_PUBLIC_SEASONS`, the same gate the build uses;
* it is TRUNCATED -- caught by the structural checks.

Every failure is fatal and nothing is ever repaired. A promotion step that
"fixed" an artifact would be publishing something no build produced.
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

#: Files without which the artifact is not a site. Deliberately short: this
#: is a truncation check, not an inventory.
REQUIRED_PATHS = (
    "index.html",
    "data/dashboard_build_manifest.json",
    "explore/explore-metadata.json",
    "explore/players.json",
    "static/style.css",
    "static/app.js",
    "methodology/index.html",
    "status/index.html",
)

LIVE_MANIFEST_URL = "https://contactluck.com/data/dashboard_build_manifest.json"


class ArtifactVerificationError(Exception):
    """The artifact must not be promoted. Never raised for anything this
    script could plausibly repair -- it repairs nothing."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ArtifactVerificationError(f"{path} is missing") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactVerificationError(f"{path} is not valid JSON: {exc}") from exc


def fetch_live_manifest(url: str = LIVE_MANIFEST_URL) -> dict[str, Any] | None:
    """The manifest the public site is serving right now, or None if it
    cannot be read. Read-only GET of our own published site.

    Certificate verification is never disabled -- an unverifiable read of
    the live site is exactly as useless as no read, and the caller already
    treats None as "refuse". `certifi`'s bundle is used when the platform's
    own trust store is not configured (the case on some local installs);
    CI's is fine either way.
    """
    context = None
    try:
        import certifi  # noqa: PLC0415 -- optional, resolved at call time

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = None
    try:
        request = Request(url, headers={"User-Agent": "contact-luck-promotion-check"})
        with urlopen(request, timeout=30, context=context) as response:  # noqa: S310
            return json.loads(response.read())
    except Exception:  # noqa: BLE001 -- any failure means "unknown", handled by the caller
        return None


def authorized_pitcher_seasons() -> tuple[int, ...]:
    from dashboard_config import PITCHER_PUBLIC_SEASONS

    return tuple(PITCHER_PUBLIC_SEASONS)


def verify_artifact(
    dist_dir: Path,
    *,
    expected_data_through: str,
    expected_repository_commit: str,
    expected_counts: dict[str, int] | None = None,
    live_manifest: dict[str, Any] | None = None,
    allow_rollback: bool = False,
    seasons: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Raise `ArtifactVerificationError` unless `dist_dir` may be promoted."""
    problems: list[str] = []
    if not dist_dir.is_dir():
        raise ArtifactVerificationError(f"{dist_dir} is not a directory")

    # ── structure ────────────────────────────────────────────────────────
    for relative in REQUIRED_PATHS:
        if not (dist_dir / relative).is_file():
            problems.append(f"required file missing: {relative}")
    if problems:
        raise ArtifactVerificationError(
            "artifact is not a complete site:\n  " + "\n  ".join(problems)
        )

    empty = [
        str(p.relative_to(dist_dir))
        for p in sorted(dist_dir.rglob("*"))
        if p.is_file() and p.stat().st_size == 0
    ]
    if empty:
        problems.append(f"{len(empty)} zero-byte file(s), artifact is truncated: {empty[:5]}")

    manifest = _load_json(dist_dir / "data" / "dashboard_build_manifest.json")
    explore = _load_json(dist_dir / "explore" / "explore-metadata.json")

    # ── identity: is this the build the operator means to promote? ───────
    if manifest.get("data_through_date") != expected_data_through:
        problems.append(
            f"data_through_date is {manifest.get('data_through_date')!r}, "
            f"operator expected {expected_data_through!r}"
        )
    if manifest.get("repository_commit") != expected_repository_commit:
        problems.append(
            f"repository_commit is {manifest.get('repository_commit')!r}, "
            f"operator expected {expected_repository_commit!r}"
        )

    # ── internal consistency: does the artifact agree with itself? ───────
    # An operator can only confirm what they already believe; these checks
    # catch a build that is wrong in a way nobody thought to type in.
    if explore.get("data_through_date") != manifest.get("data_through_date"):
        problems.append(
            f"Explore payload describes {explore.get('data_through_date')!r} but the "
            f"manifest says {manifest.get('data_through_date')!r} -- two different snapshots "
            "in one artifact"
        )
    if explore.get("player_count") != manifest.get("player_count"):
        problems.append(
            f"Explore player_count {explore.get('player_count')} != manifest player_count "
            f"{manifest.get('player_count')}"
        )
    games_on_disk = len(list((dist_dir / "explore" / "games").glob("*.json")))
    if games_on_disk != explore.get("game_count"):
        problems.append(
            f"Explore metadata claims {explore.get('game_count')} games but "
            f"{games_on_disk} game files are present"
        )
    players_on_disk = len(list((dist_dir / "explore" / "players").glob("*.json")))
    if players_on_disk != explore.get("player_count"):
        problems.append(
            f"Explore metadata claims {explore.get('player_count')} players but "
            f"{players_on_disk} player files are present"
        )
    hitter_pages = len(list((dist_dir / "players").glob("*/index.html")))
    if hitter_pages != manifest.get("player_count"):
        problems.append(
            f"manifest player_count {manifest.get('player_count')} != {hitter_pages} "
            "hitter pages built"
        )

    # ── pitcher seasons: only what is authorized, and only what is built ─
    declared = [int(s) for s in manifest.get("pitcher_seasons", [])]
    allowed = authorized_pitcher_seasons() if seasons is None else seasons
    unauthorized = sorted(set(declared) - set(allowed))
    if unauthorized:
        problems.append(
            f"artifact publishes unauthorized pitcher season(s) {unauthorized}; "
            f"PITCHER_PUBLIC_SEASONS is {list(allowed)}"
        )
    pitchers_root = dist_dir / "pitchers"
    if declared and not pitchers_root.is_dir():
        problems.append("manifest declares pitcher seasons but no /pitchers/ route exists")
    if pitchers_root.is_dir():
        on_disk = sorted(int(p.name) for p in pitchers_root.iterdir() if p.name.isdigit())
        for season in on_disk:
            if season not in allowed:
                problems.append(f"/pitchers/{season}/ exists but that season is not authorized")
            if not (pitchers_root / str(season) / "index.html").is_file():
                problems.append(f"/pitchers/{season}/ has no board page")
        if sorted(declared) != on_disk:
            problems.append(
                f"manifest declares pitcher seasons {sorted(declared)} but routes exist for "
                f"{on_disk}"
            )
        cards = len(list(pitchers_root.glob("*/*/index.html")))
        if declared and cards != manifest.get("pitcher_count"):
            problems.append(
                f"manifest pitcher_count {manifest.get('pitcher_count')} != {cards} pitcher "
                "card pages built"
            )

    # ── operator-supplied counts, if any ─────────────────────────────────
    for key, expected in (expected_counts or {}).items():
        actual = explore.get(key) if key in ("play_count", "game_count") else manifest.get(key)
        if actual != expected:
            problems.append(f"{key} is {actual}, operator expected {expected}")

    # ── anti-rollback: the check this script exists for ──────────────────
    live_date = (live_manifest or {}).get("data_through_date")
    if live_manifest is None:
        problems.append(
            "could not read the live site's manifest, so a rollback cannot be ruled out. "
            "Refusing rather than guessing."
        )
    elif live_date and manifest.get("data_through_date", "") < live_date:
        message = (
            f"ROLLBACK: this artifact is data_through {manifest.get('data_through_date')} but "
            f"production is already serving {live_date}. Promoting it would move production "
            "backward."
        )
        if allow_rollback:
            print(f"warning: {message} Proceeding because allow_rollback was set.")
        else:
            problems.append(message)

    if problems:
        raise ArtifactVerificationError(
            "artifact refused for promotion:\n  - " + "\n  - ".join(problems)
        )

    return {
        "data_through_date": manifest["data_through_date"],
        "repository_commit": manifest["repository_commit"],
        "pitcher_seasons": declared,
        "player_count": manifest.get("player_count"),
        "qualified_count": manifest.get("qualified_count"),
        "pitcher_count": manifest.get("pitcher_count"),
        "play_count": explore.get("play_count"),
        "game_count": explore.get("game_count"),
        "snapshot_manifest_hash_reference": manifest.get("snapshot_manifest_hash_reference"),
        "live_data_through_date": live_date,
        "files": sum(1 for p in dist_dir.rglob("*") if p.is_file()),
    }


def _int_or_none(value: str | None) -> int | None:
    return int(value) if value not in (None, "") else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--expected-data-through", required=True)
    parser.add_argument("--expected-repository-commit", required=True)
    for name in ("player-count", "qualified-count", "pitcher-count", "play-count", "game-count"):
        parser.add_argument(f"--expected-{name}", default=None)
    parser.add_argument(
        "--allow-rollback",
        action="store_true",
        help="Permit promoting an artifact OLDER than what production serves. Never set this "
        "to get past a surprise -- only for a deliberate, understood revert.",
    )
    parser.add_argument("--live-manifest-url", default=LIVE_MANIFEST_URL)
    parser.add_argument(
        "--skip-live-check",
        action="store_true",
        help="Offline use only (tests). Disables the anti-rollback check.",
    )
    args = parser.parse_args(argv)

    counts = {
        key.replace("-", "_"): _int_or_none(getattr(args, f"expected_{key.replace('-', '_')}"))
        for key in ("player-count", "qualified-count", "pitcher-count", "play-count", "game-count")
    }
    counts = {k: v for k, v in counts.items() if v is not None}

    live = None if args.skip_live_check else fetch_live_manifest(args.live_manifest_url)
    try:
        report = verify_artifact(
            args.dist_dir,
            expected_data_through=args.expected_data_through,
            expected_repository_commit=args.expected_repository_commit,
            expected_counts=counts,
            live_manifest={} if args.skip_live_check else live,
            allow_rollback=args.allow_rollback,
        )
    except ArtifactVerificationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("[verify_dashboard_artifact] VERIFIED -- artifact may be promoted")
    for key, value in report.items():
        print(f"[verify_dashboard_artifact] {key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
