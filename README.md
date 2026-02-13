# Catan Learning Environment (CLE)

RL training framework for teaching LLMs to play Settlers of Catan through self-play and fine-tuning.

**[Training Pipeline Design](docs/architecture/DESIGN.md)** - Architecture, implementation roadmap, and V0 plan.

## Architecture

```
LLM Agents (4 policies) <-> PettingZoo AEC (multi-agent RL) <-> Catan Engine (rules)
                                              ^
                                              |
                                    Real-time Web UI (React)
```

## Components

1. **Catan Engine** (engine/) - Pure game rules
2. **PettingZoo Environment** (cle/env/) - Multi-agent text-based observations
3. **Agents** (cle/agents/) - LLM agents with strategic memory
4. **Evaluation** (cle/eval/) - RL algorithms, metrics, benchmarking
5. **Training** (cle/training/) - Fine-tuning orchestration

See [cle/env/README.md](cle/env/README.md) for detailed environment docs.

## Quick Start

```python
from cle.env import catan_env

# Create multi-agent environment (4 players)
env = catan_env.env(num_players=4)
env.reset()

# PettingZoo AEC game loop
for agent in env.agent_iter():
    obs = env.observe(agent)
    # Semantic text: "Your VP: 0/10, Phase: initial_placement..."

    action = env.game.state.playable_actions[0]  # LLM decides from obs
    env.step(action)
```

## Key Features

- **PettingZoo AEC** - Native multi-agent turn-based environment
- **VLM observations** - Rendered board image + structured text context (dual-channel)
- **4-agent self-play** - All agents learn simultaneously
- **LoRA fine-tuning via GRPO** - Efficient VLM updates with group-relative policy optimization
- **GLM-4.6V-Flash** - 9B VLM with native XML tool calling, 128K context, thinking/non-thinking modes

## Structure

```
engine/               # Game engine (pure rules, no ML deps)
cle/
  ├── agents/           # LLM agents with memory
  ├── env/      # PettingZoo wrapper (multi-agent text obs/actions)
  ├── eval/
  │   ├── game_viewer/  # Flask game viewer server (refactored package)
  │   │   ├── colonist/ # Pure Colonist<->Engine translation (no Flask deps)
  │   │   ├── replay/   # Replay stepping, action matching, navigation
  │   │   ├── live/     # Live game logging and auto-play
  │   │   ├── routes/   # Flask Blueprints for all API endpoints
  │   │   ├── state.py  # ServerState singleton (replaces globals)
  │   │   └── app.py    # Flask factory + __main__ entry point
  │   └── ...           # RL training, metrics, evaluator
  ├── training/         # Fine-tuning orchestration
  └── examples/         # Usage examples
```

## Game Viewer UI

Real-time web interface for watching LLM agents play Catan.

### Quick Start

**Backend (Flask + SocketIO):**
```bash
python -m playground.game_viewer.app
# Server runs on http://localhost:5001
```

**Frontend (React + Vite):**
```bash
cd playground/frontend
npm install
npm run dev
# UI runs on http://localhost:5173
```

### Features
- Real-time hex board rendering with SVG
- Live game state updates via WebSocket
- Player stats (VP, resources, buildings)
- Step-by-step game control
- LLM decision tracking
- Probability dots on number tokens

## Player-to-Player Trading

Full player-to-player trading is now implemented with async state machine:

**How it works:**
1. During their turn (after rolling), players can offer trades via OFFER_TRADE actions
2. Trade offers specify resources to give (1-3 of one type) and request (1-2 of one type)
3. Other players respond sequentially (ACCEPT_TRADE or REJECT_TRADE)
4. Offering player chooses which acceptee to trade with (CONFIRM_TRADE) or cancels
5. Resources are swapped atomically

**State machine:**
- `is_resolving_trade`: bool flag indicating active trade
- `current_trade`: 11-tuple (5 offered + 5 requested + offering_player_index)
- `acceptees`: tuple tracking which players accepted
- Sequential prompts: PLAY_TURN -> DECIDE_TRADE -> DECIDE_ACCEPTEES -> PLAY_TURN
- New trade offer cancels any in-progress trade (auto-reset)

**Backend implementation:**
- Engine generates trade possibilities in [engine/models/actions.py:322](engine/models/actions.py#L322)
- State transitions in [engine/state.py:689-770](engine/state.py#L689-L770)
- Game viewer broadcasts trade state via WebSocket

## Replay Scraper (Training Data Pipeline)

Scrapes expert player replays from Colonist.io for training data bootstrapping.

### Current Status

**Game Index Complete:** 8,495 games from top 100 4P players collected in `4p_games_top100.json`
- 7,327 unique game IDs
- 3,016 wins / 5,479 losses
- Ready for replay scraping

### Requirements

- **Colonist.io Membership** - Replay viewing requires a paid membership
- **JWT Token** - Authentication cookie from your logged-in session

### Setup

```bash
cd data_pipeline/bootstrapping/scrapers
pip install -r requirements.txt
playwright install chromium
```

### Get Your JWT Token

1. Log into colonist.io (with membership)
2. Open DevTools (F12) > Application > Cookies > colonist.io
3. Copy the value of `jwt_colonist.io`
4. Token expires after ~7 days, refresh as needed

### Usage

```bash
# Test API connection (no auth needed)
python scrape_top_players.py --mode test-api

# Scrape single replay via API (recommended)
COLONIST_JWT="<your-token>" python replay_api_scraper.py --game-id 192418134

# Batch scrape from game index
COLONIST_JWT="<your-token>" python replay_api_scraper.py --max-games 100

# Export training data
python scrape_top_players.py --mode export
```

### Pipeline

1. **Leaderboard API** - Fetches top players from ranked leaderboards (done)
2. **History API** - Gets game history for each player (done - 8,495 games indexed)
3. **Replay API** - Direct endpoint: `/api/replay/data-from-game-id?gameId={id}` (needs JWT + membership)
4. **Training Data Generator** - Converts replays to observation-action pairs (done)

### Training Data Generation

```bash
# Generate from single replay (for testing)
python data_pipeline/bootstrapping/generate_training_data.py --single data/raw_replays/192418134.json

# Batch process all replays
python data_pipeline/bootstrapping/generate_training_data.py --input-dir data/raw_replays --output data/training/policy_data.jsonl
```

**Supported action types (with engine param conversions):**
- BUILD_SETTLEMENT, BUILD_CITY, BUILD_ROAD (corner/edge ID mapping)
- ROLL (dice values injected)
- MOVE_ROBBER (tile coordinate only, type 49 or type 11 with pieceEnum=5)
- STEAL (separate action after MOVE_ROBBER, stolen resource from type 14/15 logs)
- DISCARD (engine uses None value)
- MARITIME_TRADE (bank/port, 5-tuple matching)
- OFFER_TRADE, ACCEPT_TRADE, REJECT_TRADE, CONFIRM_TRADE (10/11-tuple formats)
- BUY_DEVELOPMENT_CARD (no value needed)
- PLAY_KNIGHT_CARD, PLAY_ROAD_BUILDING (no value needed)
- PLAY_YEAR_OF_PLENTY (matched via YEAR_OF_PLENTY_RESOURCES, Resource tuple)
- PLAY_MONOPOLY (matched via MONOPOLY_RESOURCE, Resource string)

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

### Replay Data Format (Fully Decoded)

The replay API returns delta-based state changes. Key mappings:

**Tile Types (0-5):**
- 0: Desert, 1: Wood, 2: Brick, 3: Sheep, 4: Wheat, 5: Ore
- (VERIFIED: Road builds use resource 1+2 = Wood+Brick, confirming 1=Wood not Wheat)

**Resource Types (1-5):**
- 1: Wood, 2: Brick, 3: Sheep, 4: Wheat, 5: Ore, 9: Any (3:1 trades)
- (VERIFIED: Tile type N produces resource N, and roads cost resources [1,2])

**Development Card Types (11-15):**
- 11: Knight, 12: Victory Point, 13: Road Building, 14: Monopoly, 15: Year of Plenty

**Action States (Colonist):**
- 0: Main turn (can roll, trade, build)
- 1: Setup - place settlement
- 3: Setup - place road
- 24: Must roll dice
- 27: Must move robber (after 7)
- 28: Must steal from player
- 30-31: Road building card

**Engine Action Prompts:**
- MOVE_ROBBER: Choose tile to place robber (coordinate only)
- STEAL: Choose victim to steal from (after MOVE_ROBBER, if valid targets exist)
- Stolen resource is deterministic from type 14/15 logs during replay

**Building Types:**
- 1: Settlement, 2: City

**Hex Grid Geometry:**
- Corners have (x, y, z) where z=0 is upper vertex, z=1 is lower
- Edges have (x, y, z) where z=0,1,2 are three edge directions
- Corner adjacency and edge-corner connections use specific coordinate patterns

### Coordinate Mapping (Colonist -> Engine)

Complete bidirectional mapping between Colonist coordinate IDs and engine node/edge IDs:

**Corner Mapping (54 corners):**
1. Transform Colonist (x, y) to engine cube coords: rotate 180 CW (3x 60) + reflect_x
2. z=0 -> NodeRef.NORTH, z=1 -> NodeRef.SOUTH
3. Saved in `cle/eval/corner_to_node_map.json`

**Edge Mapping (72 edges):**
1. Same hex transform as corners
2. z=0 -> EdgeRef.NORTHWEST, z=1 -> EdgeRef.WEST, z=2 -> EdgeRef.SOUTHWEST
3. Saved in `cle/eval/edge_to_edge_map.json`

**Verified working:**
- Initial placement phase correctly places settlements/roads at exact replay positions
- Dice rolls injected from replay data for deterministic playback
- All 54 corners and 72 edges mapped and validated
- Player-to-player trades parsed from tradeState events

**Trade Event Parsing:**
- OFFER_TRADE: New trade in `activeOffers` with `offeredResources`/`wantedResources`
- ACCEPT_TRADE: `playerResponses` value = 1
- REJECT_TRADE: `playerResponses` value = 2
- CONFIRM_TRADE: `gameLogState` with type=115 (trade completed)
  - When state machine can't match (e.g., flexible trade was skipped), resources are applied manually
- ALL_REJECT: When all players reject (playerResponses all = 2), offer is nulled in `activeOffers`
  - Engine handles this automatically: when `sum(acceptees) == 0`, trade is cancelled
- Flexible trades (resource 9 = "any") are flagged as `is_flexible` and skipped during replay
- Unmatchable trades don't accelerate dice rolls (skipped without executing fallback actions)

**Trade Replay Limitations (Training Signal Impact):**

The current trade replay is a lossy sequential approximation of Colonist's async trading.
Real trades happen simultaneously (all players see offers, respond in real-time with
social/timing reads). The engine replays them sequentially via a `current_player_index`
swap hack. Known consequences:

- **Resource cascade from skipped trades:** A single skipped flexible trade (resource=9)
  permanently diverges resource state. Every subsequent BUILD/BUY depending on those
  resources also gets skipped. CONFIRM_TRADE patches some back via manual resource
  application, but only when the confirm event exists in the replay data.
- **Incomplete skill coverage:** Training data captures OFFER_TRADE but not
  ACCEPT/REJECT/CONFIRM responses. The LLM learns when to propose trades but not how to
  evaluate incoming offers -- arguably the harder and more strategically important half.
- **No trade context in observations:** The observation formatter doesn't include
  active/pending trade state, so even the OFFER_TRADE actions in training data are
  disconnected from the negotiation state that motivated them.
- **Sequential != async:** Real trading has timing dynamics (quick accepts signal
  desperation, slow responses signal leverage). Completely lost in replay.
- **Net effect:** Games with heavy player trading produce increasingly unreliable training
  signal as the game progresses. Games with mostly maritime/bank trades are reliable.
  A proper fix requires either (a) modeling async trade as a multi-agent sub-episode,
  or (b) filtering training data to exclude post-divergence steps.

**Divergence Handling:**
- When BUILD_ROAD/BUILD_SETTLEMENT/BUILD_CITY can't match (player lacks resources due to skipped trades), action is skipped with a warning instead of executing a random fallback action
- When BUY_DEVELOPMENT_CARD can't match (player lacks 1 sheep + 1 wheat + 1 ore), action is skipped with a warning
- CONFIRM_TRADE uses the engine's playable action directly (not the Colonist trade tuple) to avoid Color enum mismatch errors
- Divergences are logged to the game log for visibility
- Replay properly signals completion via `finished: true` in all response paths

**Game Log Types:**
- 11: Generic action (pieceEnum=5 means robber moved on 7-roll)
- 14: Cards taken FROM victim (cardEnums = stolen resource)
- 15: Cards received BY thief (cardEnums = stolen resource)
- 16: Steal public broadcast (thief/victim, cardBacks hidden)
- 20: Dev card played (cardEnum)
- 21: Year of Plenty resources taken
- 49: Robber moved via Knight (tileInfo)
- 55: Discard on 7 (cardEnums)
- 86: Monopoly steal (cardEnum, amountStolen)
- 115: Player trade completed
- 116: Bank/maritime trade (givenCardEnums, receivedCardEnums)

**Example Event:**
```json
{
  "input": {"deltaS": 2.5},
  "stateChange": {
    "mapState": {
      "tileCornerStates": {"41": {"owner": 1, "buildingType": 1}},
      "tileEdgeStates": {"53": {"owner": 1, "type": 1}}
    },
    "currentState": {"actionState": 3, "completedTurns": 1},
    "playerStates": {"1": {"resourceCards": {"cards": [3, 4, 4]}}},
    "diceState": {"dice1": 4, "dice2": 5}
  }
}
```

### Replay Decoder

Full state reconstruction with valid move generation:

```bash
cd data_pipeline/bootstrapping
python replay_decoder.py data/raw_replays/194335024.json
```

**Features:**
- Reconstructs complete game state from delta events
- Computes valid moves at each state (respecting distance rule, road connections, etc.)
- Maps corners to tiles for resource production
- Generates training data with state-action-validmoves tuples

Data saved to `data_pipeline/bootstrapping/data/training/`

## Supabase Database (Optional)

Cloud-native storage for replays and training data.

### Setup

1. Create a Supabase project at https://supabase.com
2. Go to SQL Editor and run the schema:
```sql
-- Replays table
create table if not exists replays (
    game_id text primary key,
    data jsonb not null,
    metadata jsonb default '{}',
    scraped_at timestamptz default now()
);

-- Training examples table
create table if not exists training_examples (
    id serial primary key,
    game_id text references replays(game_id) on delete cascade,
    event_index int not null,
    turn int not null,
    phase text not null,
    player int not null,
    observation text not null,
    action text not null,
    action_type text not null,
    action_value jsonb,
    is_winner boolean default false,
    time_taken_seconds float,
    created_at timestamptz default now()
);

-- Indexes
create index if not exists idx_training_winner on training_examples(is_winner);
create index if not exists idx_training_action_type on training_examples(action_type);
```

3. Set environment variables (from Project Settings > API):
```bash
export SUPABASE_URL="https://xxx.supabase.co"
export SUPABASE_KEY="eyJ..."
```

### Usage

```bash
# Scrape replays to Supabase
COLONIST_JWT="<token>" python replay_api_scraper.py --max-games 100 --supabase

# Generate training data to Supabase
python generate_training_data.py --supabase --no-file

# Process replays already in Supabase
python generate_training_data.py --from-db

# Export from Supabase to JSONL
python -c "from db import export_to_jsonl; export_to_jsonl('training.jsonl', is_winner=True)"

# Check stats
python db.py stats
```

## V0: VLM Agent with GLM-4.6V-Flash

V0 = "Can a fine-tuned 9B VLM play Catan at above-random level from board renders + text context?"

### Base Model: [GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) (9B)

Chosen for native agentic capabilities at 9B scale:
- **Native XML tool calling** - trained with `<tool_call>` format during SFT and RL, no prompt hacking
- **128K context window** - board image + game history + strategic notes without truncation
- **Thinking mode toggle** - `<think>` for critical decisions, `/nothink` for fast self-play rollouts
- **Trained on GUI agent tasks** - already understands structured visual environments with action spaces
- **GRPO-trained** - model's reward sensitivity shaped by Group Relative Policy Optimization

Paper: [arxiv:2507.01006](https://arxiv.org/abs/2507.01006)

### Observation Design: Dual-Channel (Image + Text)

The VLM receives two channels per decision point. Spatial understanding is embedded in the visual channel - the model sees adjacency, clustering, expansion paths, opponent proximity directly from pixels. Text handles discrete/hidden state only.

**Visual channel (board render):**
- Rendered game board matching Colonist UI style
- Hex tiles with resource colors + number tokens with pip dots
- Settlements/cities/roads colored by player
- Robber position, ports marked
- Valid build locations highlighted (green dots/lines for current player's legal moves)

**Text channel (non-spatial game state):**
```
YOUR RESOURCES: wood=2, brick=1, wheat=3, sheep=0, ore=0
YOUR VP: 3/10 | DEV CARDS: knight x1
OPPONENTS:
  BLUE: 4 VP, 2 settlements, 1 city, 5 resources
  RED: 3 VP, 2 settlements, 4 resources
  WHITE: 2 VP, 2 settlements, 3 resources
VALID ACTIONS: [0] BUILD_ROAD edge=42, [1] BUILD_SETTLEMENT node=17, [2] MARITIME_TRADE 4 wheat -> 1 ore, [3] END_TURN
```

**Why dual-channel:**
- Image captures spatial relationships (what Catan is fundamentally about) - the model doesn't need "node 42 is adjacent to ore hex with 6 pips" in text because it can see that
- Text captures discrete/hidden state that a 9B VLM would unreliably OCR from a screenshot (resource counts, dev cards, enumerated actions)
- GLM-4.6V-Flash was trained on interleaved image-text; this is on-distribution

### Agent Decision Loop: Recall -> Think -> Act

The agent uses GLM's native XML tool calling in a three-phase loop. Strategic memory is recalled FIRST so that reasoning is grounded in retrieved context, not generated from scratch.

```xml
<!-- 1. RECALL: Load strategic context before reasoning -->
<tool_call>recall_strategy
<arg_key>query</arg_key>
<arg_value>current_plan</arg_value>
</tool_call>
--> "Longest road + dev card VP. Need 2 more roads then pivot."

<!-- 2. THINK: Reason with recalled context + board image + text state -->
<think>
My plan is longest road + dev cards. I see blue expanding toward
my ore port - if they settle there I lose city potential.
Road at edge 42 both extends my longest road AND blocks blue.
Two birds one stone.
</think>

<!-- 3. ACT: Execute decision -->
<tool_call>select_action
<arg_key>action_index</arg_key>
<arg_value>0</arg_value>
</tool_call>
```

**Why Recall -> Think -> Act (not Think -> Recall -> Act):**
- Thinking without context leads to hallucinated plans and wasted reasoning tokens
- Recall first means the `<think>` block integrates retrieved strategy with current board state
- Same pattern as RAG but for the agent's own strategic memory across turns

**Agent tools:**

| Tool | Purpose |
|------|---------|
| `recall_strategy` | Retrieve long-term plan, opponent models, win condition |
| `update_strategy` | Persist strategic notes after key events (placement, trades, pivots) |
| `select_action` | Execute a game action by index from valid action list |

### Training Data Strategy

Two sources, combined via a flywheel:

**Source 1: YouTube videos (small volume, high reasoning quality)**

Expert Catan gameplay with commentary = pre-annotated reasoning data.

```
Raw Video -> Whisper transcription -> segment by decision point
     |                                        |
     v                                        v
Gemini snapshots (board frames            Reasoning chunks
when reasoning detected in audio)               |
     |                                        |
     +----------------+------------------------+
                      |
                      v
         Frontier VLM aligns and cleans:
         "Given this board frame and transcript segment,
          extract the strategic reasoning and action taken"
                      |
                      v
          (board_image, reasoning, action) triple
```

Best sources by signal density:
- Tournament commentary (3rd person expert analysis) - very high signal
- Streamer gameplay on Colonist.io (screen shows exact board state) - high volume
- "How I got to top 100" strategy breakdowns - explicit reasoning
- Initial placement is the highest-signal target: every video covers it, every player explains it, and it's the highest-impact decision in the game

**Source 2: Colonist replays (high volume, no reasoning)**

8.5K expert games with actions but no explanations. Generate synthetic reasoning using YouTube-derived examples as few-shot style templates:

```
Here are examples of how expert Catan players reason about decisions:
[3 YouTube-derived examples with board images]

Now explain this expert's decision:
[Replay board render + expert action + top 3 alternatives not chosen]
```

Contrastive framing ("why X instead of Y and Z") forces board-specific reasoning over generic heuristics.

**The flywheel:**
```
YouTube (500-2K genuine reasoning examples)
     |
     v
Few-shot reasoning style seed
     |
     +-- applied across -->  Replays (50K+ decision points)
                                  |
                                  v
                         Combined SFT dataset
                         (real reasoning style, replay scale)
```

**Critical moment oversampling:** Weight training data 5-10x toward high-variance decisions (initial placement, robber on 7, late-game pivots, build vs save vs trade). "Rolled 8, collected wheat, ended turn" teaches nothing.

### Training Pipeline

Following GLM paper's own recipe, adapted for Catan:

```
Colonist Replays --> Board Renders + Action Labels --> SFT (LoRA)
                                                          |
                                                     Policy v1
                                                          |
                                            Self-Play (4x same policy)
                                                          |
                                                  GRPO with rewards:
                                                  - Win/loss (+5/-1)
                                                  - VP gain (+1.0)
                                                  - Production increase (+0.1)
                                                          |
                                                  LoRA Training v2
                                                          |
                                                       LOOP
```

**Phase 1 - SFT**: Behavioral cloning on expert replays with XML tool-call output format
**Phase 2 - GRPO**: Group Relative Policy Optimization with Catan-specific reward verifiers
**Phase 3 - RLCS**: Curriculum sampling - focus training compute on mid-difficulty decisions

### Key Differences from Previous Design

| Aspect | Old V0 | New V0 |
|--------|--------|--------|
| Model | Llama 3.1 8B (text-only) | GLM-4.6V-Flash 9B (VLM) |
| Observation | Text-only semantic description | Board render + structured text |
| Action format | `<reasoning>` + JSON | Native XML `<tool_call>` |
| RL algorithm | PPO | GRPO (no critic network, simpler) |
| Training recipe | Custom | Following GLM paper's SFT -> GRPO -> RLCS pipeline |

## TODOs

- [x] Training data pipeline from Colonist replays
- [x] Supabase cloud storage integration
- [ ] Scrape full game replays with JWT token
- [ ] Server-side board renderer (engine state -> annotated PNG)
- [ ] Replay-to-image pipeline (render each decision point from engine state)
- [ ] YouTube reasoning pipeline (yt-dlp -> Whisper -> Gemini frame extraction -> alignment)
- [ ] Synthetic reasoning generation (YouTube few-shot style applied to replay decision points)
- [ ] Critical moment classifier (tag initial placement, robber, pivots for oversampling)
- [ ] VLM SFT script (LoRA on GLM-4.6V-Flash with Recall->Think->Act format)
- [ ] Strategic memory tool implementation (recall_strategy, update_strategy, select_action)
- [ ] GRPO training with Catan reward verifiers
- [ ] Self-play runner with `/nothink` mode for throughput
- [ ] RLCS curriculum sampling by decision difficulty
- [ ] Evaluation vs random/rule-based baselines
