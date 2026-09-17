# Colonist replay acquisition

This package indexes, captures, validates, and inspects Colonist replay payloads.
It does not produce policy-training examples.

The former observation/action generator was removed because it applied each
target action before formatting the claimed pre-action observation. Decision
datasets must be built from the authoritative verified replay executor.

## Code layout

- `scrapers/colonist_api.py`: leaderboard, profile, and history API client.
- `scrapers/scrape_top_players.py`: builds lightweight candidate indexes.
- `scrapers/replay_playwright_scraper.py`: preferred authenticated replay capture.
- `scrapers/replay_api_scraper.py`: direct API fallback and debugging path.
- `scrapers/build_replay_splits.py`: deterministic leakage-safe split manifests.
- `replay_decoder.py`: standalone raw replay inspection utility.

## Artifact layout

- `artifacts/raw/colonist/indexes/`: candidate game indexes.
- `artifacts/staging/colonist/replays/`: new, unverified, or rejected captures.
- `artifacts/raw/colonist/replays/`: validated canonical replay payloads.
- `artifacts/manifests/colonist/splits/`: split and queue manifests.
- `artifacts/fixtures/catan_board_bench/smoke5/`: frozen visual smoke fixture.

Compatibility symlinks under `data_pipeline/bootstrapping/data/` preserve
existing read-only callers. New code must use the canonical artifact paths.

## Build an index

Run from the repository root:

```bash
python -m data_pipeline.bootstrapping.scrapers.scrape_top_players \
  --mode index \
  --top 100 \
  --games 100 \
  --all-games \
  --game-modes Classic4P,Tournament
```

The default output is
`artifacts/raw/colonist/indexes/4p_games_top100.json`. Use `--index-output` for
a separately named index.

Authenticated or explicit-user history can be indexed with:

```bash
COLONIST_JWT="<token>" \
python -m data_pipeline.bootstrapping.scrapers.scrape_top_players \
  --mode index --me --games 100 --all-games --game-modes all

python -m data_pipeline.bootstrapping.scrapers.scrape_top_players \
  --mode index --username Robijs --games 100 --all-games
```

## Capture replay payloads

The Playwright path is preferred because Colonist may require browser and
Cloudflare session state before the replay endpoint succeeds.

```bash
python -m data_pipeline.bootstrapping.scrapers.replay_playwright_scraper \
  --game-id 228953487 \
  --player-color 1
```

New captures default to `artifacts/staging/colonist/replays/`. A bounded batch:

```bash
python -m data_pipeline.bootstrapping.scrapers.replay_playwright_scraper \
  --max-games 10 \
  --max-attempts 15 \
  --expected-player-count 4 \
  --expected-mode-setting 0 \
  --delay-seconds 40
```

Stop immediately on HTTP 429 or `Retry-After`, cool down before a manually
approved retry, and never run multiple replay scrapers in parallel.

If OAuth refuses the Playwright profile, attach to a user-approved Chrome
instance over CDP. Do not terminate or relaunch the user's browser without
confirmation.

The direct API fallback uses the same artifact defaults:

```bash
COLONIST_JWT="<token>" \
python -m data_pipeline.bootstrapping.scrapers.replay_api_scraper \
  --max-games 10
```

JWT alone may still receive a 403 when browser session state is required.

## Keep pulling in batches

`pull_replays_loop.py` runs bounded batches back to back through a logged-in
Chrome over CDP, rebuilding a deduplicated index (raw, staging, and rejected
captures excluded) before each batch and cooling down between them. It waits for
any running scraper first and halts on its own on a rate limit, an empty batch
(dead session or exhausted index), or the batch cap.

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --user-data-dir="$PWD/.colonist-cdp-profile" --remote-debugging-port=9222 &
# log in once in that window, then:
nohup uv run python -m data_pipeline.bootstrapping.scrapers.pull_replays_loop \
  --batch-size 150 --cooldown-seconds 600 --max-batches 20 > logs/colonist_pull_loop.log 2>&1 &
```

Playwright's bundled Chromium cannot read cookies written by Google Chrome, so a
profile logged in through Chrome must be driven through Chrome via `--cdp-url`.
Per-batch logs are `logs/colonist_scrape_<timestamp>.log`.

Observed limit (2026-09-17): the replay endpoint returned 429 after about 50
requests in 35 minutes at 40s pacing, with no `Retry-After`. The loop therefore
defaults to 45 per batch and an hour between batches; one 429 cools down 90
minutes and retries once, a second consecutive 429 stops the loop.

## Seat ratings

Replay payloads, the history endpoint, and `gameDetails.isRanked` (always false)
carry no ratings, and `/api/profile/{username}` does not exist. The only public
source is the Classic4P leaderboard: every rated player (30k+), 100 per page,
`search` ignored. `annotate_seat_ratings.py` takes one snapshot per day into
`artifacts/raw/colonist/indexes/classic4p_leaderboard_<date>.json` and writes
`artifacts/manifests/colonist/seat_ratings.json` with per-seat rank, rating,
division, and bot flag for every captured replay. Ratings are as of the snapshot,
not at game time. The pull loop runs it after each batch. Rated games are
`gameSettings.eloType == 4` with `gameType == 6` (inferred from the corpus; the
enum is undocumented).

## Validate and promote

Before copying a staged payload into `artifacts/raw/colonist/replays/`, verify:

- the payload has a non-empty event history;
- the player count and mode match the supported base-game contract;
- the game ID is not a duplicate of an existing canonical replay;
- the replay passes the consolidated semantic/resource/trade audit;
- the source hash and acquisition metadata are retained.

Unsupported and malformed payloads stay in staging with an explicit rejection
reason. Never repair a raw payload in place.

## Build split manifests

```bash
python -m data_pipeline.bootstrapping.scrapers.build_replay_splits \
  --index-files artifacts/raw/colonist/indexes/4p_games_top100.json
```

Outputs default to `artifacts/manifests/colonist/splits/`. CatanBoardBench holdout IDs
are excluded before game-level splitting.
