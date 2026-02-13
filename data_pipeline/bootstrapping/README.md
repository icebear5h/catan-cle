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
python scrape_top_players.py --mode all
```

**Output:** `4p_games_top100.json` with 8,495 games (7,327 unique)

### 2. Scrape Replays

Download replay data using the direct API endpoint.

```bash
# Get JWT from colonist.io cookies (DevTools > Application > Cookies > jwt_colonist.io)
export COLONIST_JWT="<your-token>"

# Scrape to local files
python replay_api_scraper.py --max-games 100

# Or scrape directly to Supabase
python replay_api_scraper.py --max-games 100 --supabase
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
pip install httpx supabase
```

- **Colonist.io Membership** - Replay viewing requires paid membership
- **JWT Token** - Auth cookie expires every ~7 days

## Metrics

**Target after bootstrapping:**
- 1000+ games processed
- 50k+ training examples
- Win rate >60% vs random baseline
- Reasonable action accuracy on held-out test set
