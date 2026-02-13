# Catan PettingZoo Environment for LLM Training

Multi-agent text-based RL environment for training LLMs to play Settlers of Catan via PPO/DQN with LoRA fine-tuning.

## Architecture

```
LLM Agents (4 policies) <-> PettingZoo AEC (multi-agent RL) <-> Catan Engine (rules)
```

### Components

1. **Catan Engine** (`catanatron/`)
   - Pure game rules and state management
   - Provides `Game`, `State`, `Action` interfaces
   - No AI logic, just game mechanics

2. **PettingZoo Wrapper** (`cle/env/`)
   - `CatanEnv`: PettingZoo AEC environment for turn-based multi-agent
   - Text-based observation/action spaces for LLMs
   - Reward shaping for RL training

3. **Observation Formatter** (`observation_formatter.py`)
   - Converts Catanatron state → semantic text
   - Following FLE (Factorio Learning Environment) pattern
   - Rich strategic context for LLM reasoning

4. **Action Interface** (`action_space.py`)
   - Parses LLM outputs (tool calls, text) → Catanatron actions
   - Validates actions against legal moves
   - Formats action descriptions with strategic context

## Usage

### Basic Example (PettingZoo AEC API)

```python
from cle.env import catan_env

# Create environment (4 agents, turn-based)
env = catan_env.env(num_players=4)

# Reset for new game
env.reset()

# PettingZoo AEC game loop
for agent in env.agent_iter():
    observation = env.observe(agent)
    reward = env.rewards[agent]
    termination = env.terminations[agent]
    truncation = env.truncations[agent]

    if termination or truncation:
        break

    # LLM decides action from text observation
    action = env.game.state.playable_actions[0]  # In practice: LLM output
    env.step(action)
```

### RL Training Loop (Conceptual)

```python
# Initialize LLM policy with LoRA
model = load_llm_with_lora("gpt-4", lora_config)

# PPO training
for episode in range(num_episodes):
    observation, info = env.reset()

    episode_data = []
    while not terminated:
        # LLM generates action from text observation
        action_text = model(observation)
        action = parse_action(action_text, env.game.state.playable_actions)

        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Store trajectory
        episode_data.append((observation, action_text, reward))
        observation = next_obs

    # Update LLM via LoRA using PPO
    update_lora_weights(model, episode_data)
```

## Observation Format

Observations are semantic text descriptions following the FLE pattern:

```
=== GAME STATE (Turn 5) ===

Phase: main_game
Your VP: 3/10
Last dice roll: 8
STATUS: Behind by 1 VP

YOUR BUILDINGS:
  Settlements (2):
    - Node 3: On wheat port (6-wheat, 8-ore)
    - Node 12: Near ore-rich hex (5-ore, 9-wheat)
  Roads (3): 3 connections
    Longest road length: 3

YOUR RESOURCES:
  WOOD: 2
  BRICK: 1
  WHEAT: 3
  Total: 6 cards
  Can afford: settlement, road

OPPONENTS:
  BLUE: 4 VP (2 settlements, 1 city, 4 roads, 5 resources) - threatening
  WHITE: 2 VP (2 settlements, 0 cities, 2 roads, 3 resources) - building
  ORANGE: 3 VP (2 settlements, 0 cities, 3 roads, 4 resources) - building
  Longest road: BLUE (+2 VP)

VALID ACTIONS:
  BUILD_SETTLEMENT: 12 options
    1. Build settlement at node 15 (expands toward ore)
    2. Build settlement at node 22 (wheat access, blocks BLUE)
    ... and 10 more
  BUILD_ROAD: 8 options
  END_TURN: 1 option
```

## Action Format

Actions can be provided as:
1. **Catanatron Action objects** (direct)
2. **Tool calls**: `build_settlement(node=15)`
3. **Natural language**: "Build settlement at node 15"
4. **JSON**: `{"action": "BUILD_SETTLEMENT", "node": 15}`

## Reward Structure

- **+1** per victory point gained
- **+10** for winning the game
- **-10** for losing the game
- TODO: Small penalties for inefficient resource usage

## Multi-Agent Self-Play

For 4-agent self-play training:

```python
# Create 4 LLM agents
agents = [LLMAgent(color) for color in [RED, BLUE, WHITE, ORANGE]]

# Each agent has its own env view
envs = [CatanEnv(all_players, player_color=agent.color) for agent in agents]

# Training loop handles turn-taking
while not terminated:
    current_agent = agents[current_player_index]
    current_env = envs[current_player_index]

    action = current_agent.decide(current_env.game, playable_actions)

    # All envs observe the same game state
    for env in envs:
        obs, reward, terminated, truncated, info = env.step(action)
```

## TODOs

- [ ] Implement action parsing from LLM text outputs
- [ ] Add strategic context to node descriptions (tile probabilities, ports)
- [ ] Implement reward shaping for intermediate strategic goals
- [ ] Add PettingZoo wrapper for native multi-agent support
- [ ] Create RL training scripts (PPO with LoRA)
- [ ] Add evaluation metrics (win rate, avg VP, strategic diversity)
- [ ] Implement observation/action caching for faster inference
