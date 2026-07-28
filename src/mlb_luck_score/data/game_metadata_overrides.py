"""Hand-reviewed overrides for games the MLB Stats API cannot resolve.

The raw per-season game-metadata cache files produced by
`mlb_luck_score.data.download_game_metadata` are NEVER modified by this
module -- they remain an exact, re-fetchable copy of the API response. This
module documents a small, manually reviewed set of corrections applied on
TOP of that raw data at load time (see
`mlb_luck_score.data.join_venue_metadata.apply_metadata_overrides`), for
specific, verified games the API has a real data gap for.

Known case: the MLB "Field of Dreams" games, played at a temporary ballpark
built adjacent to the actual Field of Dreams movie site in Dyersville, Iowa,
have `venue: {"link": ".../venues/null"}` in the API's `/schedule` response
-- no venue id or name at all (see `download_game_metadata` module
docstring). Both games were played at the same physical site, so they share
one venue_id here.

`venue_id` for this entry is a NEGATIVE SENTINEL (`-1`), chosen so it can
never collide with a real MLB Stats API venue id (which are always
non-negative). Park geometry (fence distances, elevation) and precise
on-site GPS coordinates are NOT populated yet -- see the `notes` field on
the entry below; these are documented extension points for future work, not
implemented here.

Adding an entry here requires a documented, verifiable reason (see
`source_note` on each override) -- this is reviewed reference data, not a
place to silently paper over unexpected join gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Sentinel venue_id reserved for the temporary MLB Field of Dreams ballpark
#: (Dyersville, Iowa), used for both the 2021 and 2022 games played there.
#: Guaranteed never to collide with a real (non-negative) MLB Stats API
#: venue id.
FIELD_OF_DREAMS_VENUE_ID = -1


@dataclass(frozen=True)
class GameMetadataOverride:
    """A single hand-reviewed correction for one `game_pk`.

    Attributes:
        game_pk: The game this override applies to.
        venue_id: Corrected venue identifier (may be a negative sentinel
            for venues with no real MLB Stats API id).
        venue_name: Corrected venue name.
        roof_type: Corrected roof type, if known (`None` if not yet
            reviewed/available).
        surface_type: Corrected surface type, if known.
        is_neutral_site: Whether this game was played at a neutral site.
        latitude / longitude: On-site GPS coordinates -- NOT YET POPULATED,
            a documented extension point for future park-geometry work.
        left_field_ft / center_field_ft / right_field_ft: Fence distances --
            NOT YET POPULATED, a documented extension point.
        elevation_ft: Site elevation -- NOT YET POPULATED, a documented
            extension point (relevant for e.g. altitude/carry effects, as
            with Coors Field).
        source_note: How this override was verified -- required, never
            leave blank.
    """

    game_pk: int
    venue_id: int
    venue_name: str
    roof_type: str | None = None
    surface_type: str | None = None
    is_neutral_site: bool = True
    latitude: float | None = None
    longitude: float | None = None
    left_field_ft: float | None = None
    center_field_ft: float | None = None
    right_field_ft: float | None = None
    elevation_ft: float | None = None
    source_note: str = field(default="")

    def __post_init__(self) -> None:
        if not self.source_note:
            raise ValueError(
                f"GameMetadataOverride for game_pk={self.game_pk} has no source_note -- "
                "every override must document how it was verified."
            )


#: Reviewed overrides, keyed by `game_pk`. See module docstring.
GAME_METADATA_OVERRIDES: dict[int, GameMetadataOverride] = {
    632924: GameMetadataOverride(
        game_pk=632924,
        venue_id=FIELD_OF_DREAMS_VENUE_ID,
        venue_name="MLB Field of Dreams (Dyersville, Iowa)",
        surface_type="Grass",
        is_neutral_site=True,
        source_note=(
            "2021-08-12: Chicago White Sox (home) vs. New York Yankees (away), the "
            "inaugural 'MLB Field of Dreams Game'. Verified 2026-07-27 via "
            "GET https://statsapi.mlb.com/api/v1/schedule"
            "?sportId=1&season=2021&gameType=R&startDate=2021-08-12&endDate=2021-08-12 "
            "-- the API returns this game with venue={'link': '.../venues/null'} (no id "
            "or name). Surface is grass (the temporary field was sodded); roof N/A "
            "(outdoor, no roof). Park geometry (fence distances) and precise GPS "
            "location not yet reviewed/populated -- extension point for future work."
        ),
    ),
    663023: GameMetadataOverride(
        game_pk=663023,
        venue_id=FIELD_OF_DREAMS_VENUE_ID,
        venue_name="MLB Field of Dreams (Dyersville, Iowa)",
        surface_type="Grass",
        is_neutral_site=True,
        source_note=(
            "2022-08-11: Cincinnati Reds (home) vs. Chicago Cubs (away), the second "
            "'MLB Field of Dreams Game', same site as game_pk 632924. Verified "
            "2026-07-27 via GET https://statsapi.mlb.com/api/v1/schedule"
            "?sportId=1&season=2022&gameType=R&startDate=2022-08-11&endDate=2022-08-11 "
            "-- same API gap as 632924 (venue={'link': '.../venues/null'}). Park "
            "geometry and precise GPS location not yet reviewed/populated -- extension "
            "point for future work."
        ),
    ),
}
