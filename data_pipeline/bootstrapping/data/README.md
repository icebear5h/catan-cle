# Compatibility paths

Raw replay payloads now live outside the installable package:

- `artifacts/raw/colonist/replays/`
- `artifacts/staging/colonist/replays/`

The `raw_replays` and `replay_staging` symlinks preserve existing read-only
callers while path-only consumers migrate. New pipeline code must use the
canonical artifact paths and must not write through these compatibility links.
