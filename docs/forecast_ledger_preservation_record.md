# Where the sealed Contact Forecast prediction ledgers are preserved

Recorded 2026-09-18, nine days before the 2026 regular season ends and the end-of-season
resolution pass becomes runnable.

**A durable copy already existed before this record was written.** A preservation bundle
was placed in iCloud Drive on 2026-09-10, the same day the pre-flight was built. It was
verified on 2026-09-18 and is intact. This record exists because nothing in the
repository said so, which meant the next session had no way to know the copy existed --
and re-derived it as an open risk.

## What is preserved, and why it is irreplaceable

The 36 artifacts named by `forecast.phase2.resolution_preflight.IRREPLACEABLE`: both
horizons' sealed prediction ledgers (H100 and H200), the manifests that prove what those
predictions were, the authorizations, the errata, and the reports. 561,042 bytes.

`predictions_regenerated` is `false` in the resolution specification, and that is not a
performance choice. A prediction made before its outcome was known cannot be recreated
after the fact by anyone, at any cost. Every other artifact in this repository is
expensive to reproduce; these are impossible.

## The copies

Both live in the same iCloud Drive account.

| Created | Path | sha256 of archive |
|---|---|---|
| 2026-09-10 | `~/Library/Mobile Documents/com~apple~CloudDocs/sealed_predictions_preservation.tar.gz` | `2989bf3ed9a0fecdfc4019ec672881d2ddceb7e87e8794b099d01cfb594fd510` |
| 2026-09-18 | `~/Library/Mobile Documents/com~apple~CloudDocs/ContactLuck/forecast-sealed-ledgers/resolution_preservation_bundle_2026-09-18.tar.gz` | `114d950f438361253b2f09df955eba74410fb35571c8a3b3957734fb604e8080` |

Each has a `.sha256` sidecar beside it, and the working copies remain in
`outputs/forecast_phase2_resolution/` on this machine.

**The two archive digests differ while the contents are identical.** gzip records the
source filename and an mtime in its header, so two archives built from byte-identical
inputs at different moments never hash alike. What matters is the *contents*, and those
were compared file-by-file on 2026-09-18: same 36 files, same 561,042 total bytes, no
differing hash. Compare contents, never archive digests, when asking whether two bundles
hold the same sealed set.

## How to verify any copy

Each bundle is self-describing: `resolution_preservation_manifest.json` is written
*inside* the archive as well as beside it, so a copy found on another disk years from now
can be verified without this repository.

```bash
# The check that makes a backup a fact rather than a belief. Needs only this
# repository's forecast package -- no network, no credentials.
.venv/bin/python -c "
from pathlib import Path
from forecast.phase2.resolution_preflight import verify_preservation_bundle
print(verify_preservation_bundle(Path('<path to bundle>')))
"
```

Both copies returned `files_verified: 36, intact: True` on 2026-09-18.

A `shasum -a 256 -c` against a sidecar also works, with one gotcha: the 2026-09-10
sidecar records a **repo-relative** path (`outputs/forecast_phase2_resolution/...`), so
`-c` only resolves from the repository root, not from the iCloud directory where the
archive now sits. The 2026-09-18 sidecar records a bare filename and resolves beside the
archive.

## Drift status

The 2026-09-18 bundle was built from the working copies and compared against the
preservation manifest written by the 2026-09-10 pre-flight: same 36 files, same total
bytes, no hash changed. The 2026-09-10 archive's own contents were then compared against
it and are likewise identical. The sealed ledgers have not moved. The freeze chain (r1,
ridge, hgb, h200_spec, resolution_spec) verified with zero drift on the same run.

## That the iCloud destination is real

Worth recording, because a folder under `~/Library/Mobile Documents/` is an ordinary
local directory when iCloud Drive is off -- in which case a "durable copy" is a second
directory on the same disk. Verified on 2026-09-18: `MOBILE_DOCUMENTS` shows
`Enabled = 1`, `bird` and `fileproviderd` are running, and the container holds live
synced content (Desktop, Documents, Safari).

Note that `brctl status` fails with `Client zone not found` from a terminal without Full
Disk Access, and `mdls` reports `kMDItemIsUbiquitous = (null)` on a freshly written file
until Spotlight indexes it. Neither is evidence of a failed upload, and neither should be
read as one.

## What this is not

- **Not an authorization.** The pre-flight authorizes nothing, and neither does this. The
  resolution gate still has three blocked conditions: the season has not ended, no
  snapshot reaches the season end date, and the maintainer's in-the-moment authorization
  is given on the day.
- **Not a second look.** No outcome was opened, no evaluation run, no snapshot pinned.
- **Not a replacement for the R2 snapshot archive**, which covers a different object
  (daily production scoring snapshots) for a different reason (expensive to reproduce,
  not impossible).

## Known limitation

Two objects, one provider, one account. This is redundancy against accidental deletion
of a single file, not against loss of the account itself -- for that the copies would
have to be independent, and they are not. A third copy outside iCloud (an offline disk,
or the R2 bucket the repository already uses for snapshots) would close that gap for a
245 KB file. If one is added, record it in the table above and verify it with the command
in this file; an unverified copy is a belief, which is the thing this record exists to
avoid.
