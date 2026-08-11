# Phase 1: Bootstrapping

Learn from expert Colonist.io replays to achieve baseline Catan competency.

## Goal

Generate training data from top-ranked players' games to bootstrap an LLM Catan agent via behavioral cloning.

## Directory Structure

```
bootstrapping/
├── __init__.py              # Module exports
├── README.md                # This file
├── db.py                    # Supabase storage client
├── generate_training_data.py # Main training data generator
├── scrapers/
│   ├── replay_api_scraper.py  # Scrape replays via API
│   ├── colonist_api.py        # Colonist API client
│   ├── colonist_schema.py     # Data type definitions
│   ├── scrape_top_players.py  # Build game index from leaderboards
│   └── 4p_games_top100.json   # Index of 8,495 games
├── tests/
│   └── test_replay_auth.py    # Auth testing utilities
├── deprecated/              # Old approaches (for reference)
└── data/
    └── raw_replays/         # Downloaded replay JSON files
```

## Pipeline

### 1. Build Game Index (Done)

Scrape top 100 players from ranked leaderboards and their game histories.

```bash
cd scrapers
python scrape_top_players.py --mode index --top 100 --games 100 --all-games --game-modes Classic4P,Tournament --index-output 4p_games_top100.json

# Or index histories more likely to be replay-accessible for the current account
COLONIST_JWT="<your-token>" python scrape_top_players.py --mode index --me --games 100 --all-games --game-modes all --index-output 4p_games_me_all.json
python scrape_top_players.py --mode index --username Robijs --games 100 --all-games --index-output 4p_games_robijs.json
```

**Output:** `4p_games_top100.json` with replay candidate IDs and lightweight player/game metadata. Use `--game-modes all` to include other Colonist variants.

### 2. Scrape Replays

Capture replay data through a persistent Playwright browser session. This matches the
working Colonist replay page flow: the first replay API request may be denied while the
browser resolves session/Cloudflare state, then a later request to the same endpoint
returns the replay JSON.

```bash
# First run opens a browser profile at .colonist-playwright-profile/.
# Log in or complete any browser challenge there if prompted.
python replay_playwright_scraper.py \
  --game-id 228953487 \
  --player-color 1 \
  --output-dir ../data/raw_replays

# Batch into staging in small, paced chunks. Validate before promoting files.
python replay_playwright_scraper.py \
  --index-file 4p_games_top100.json \
  --max-games 10 \
  --max-attempts 15 \
  --expected-player-count 4 \
  --expected-mode-setting 0 \
  --delay-seconds 40 \
  --output-dir ../data/replay_staging/base4p

# Stop after any HTTP 429 and cool down before a manually approved retry.
# Do not run multiple scraper processes in parallel.

# If Google/Colonist login refuses the Playwright profile, attach to real Chrome instead.
# First quit Chrome, then relaunch it with local remote debugging enabled:
open -na "Google Chrome" --args \
  --remote-debugging-address=127.0.0.1 \
  --remote-debugging-port=9222 \
  --profile-directory=Default

python replay_playwright_scraper.py \
  --game-id 228953487 \
  --player-color 1 \
  --output-dir ../data/raw_replays \
  --cdp-url http://127.0.0.1:9222

# Direct API fallback/debug path. JWT alone may still 403 if browser session state is required.
COLONIST_JWT="<your-token>" python replay_api_scraper.py --index-file 4p_games_top100.json --max-games 100 --raw --output-dir ../data/raw_replays
```

### 3. Generate Training Data

Convert replays to observation-action pairs.

```bash
# From local files
python generate_training_data.py --input-dir data/raw_replays --output data/training/policy_data.jsonl

# From Supabase
python generate_training_data.py --from-db

# To Supabase
python generate_training_data.py --supabase --no-file
```

**Output format (JSONL):**
```json
{
  "observation": "=== GAME STATE (Turn 5) ===\nPhase: main_game\nYou are: Player 2\n...",
  "action": "BUILD_SETTLEMENT node=51",
  "action_type": "BUILD_SETTLEMENT",
  "action_value": 51,
  "player": 2,
  "turn": 5,
  "phase": "main_game",
  "is_winner": true,
  "game_id": "192418134"
}
```

### 4. Train Model

Use TRL's SFTTrainer for behavioral cloning.

```python
from trl import SFTTrainer
from datasets import load_dataset

# Load training data
dataset = load_dataset("json", data_files="policy_data.jsonl")

# Format for instruction tuning
def format_example(example):
    return f"""Given this Catan game state:
{example['observation']}

What action should you take?
{example['action']}"""

# Train with LoRA
trainer = SFTTrainer(
    model="meta-llama/Llama-3.1-8B",
    train_dataset=dataset,
    formatting_func=format_example,
    peft_config=lora_config,
)
trainer.train()
```

## Supported Action Types

- BUILD_SETTLEMENT, BUILD_CITY, BUILD_ROAD
- ROLL (dice)
- MOVE_ROBBER
- BANK_TRADE, OFFER_TRADE
- BUY_DEV_CARD, PLAY_DEV_CARD

## Requirements

```bash
pip install httpx playwright supabase
python -m playwright install chromium
```

- **Colonist.io Membership** - Replay viewing requires paid membership
- **Playwright browser profile** - Replay scraping uses `.colonist-playwright-profile/`
- **Real Chrome attach** - Use `--cdp-url` when OAuth refuses the Playwright browser
- **JWT Token** - Still useful for authenticated profile/history indexing and direct API debugging

## Metrics

**Target after bootstrapping:**
- 1000+ games processed
- 50k+ training examples
- Win rate >60% vs random baseline
- Reasonable action accuracy on held-out test set
