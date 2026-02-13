# CLE Architecture: Strategic LLM Agents for Catan

## Overview

This implements a learning framework for training Claude LLM agents to play Settlers of Catan through self-play with strategic memory and opponent modeling.

## Key Design Principles

1. **Two-Phase Decision Making**
   - **Strategic Foundation** (Initial placement): Deep reasoning about overall game plan
   - **Tactical Execution** (Rest of game): Smaller decisions aligned with strategy

2. **Event-Driven Memory Updates**
   - After EVERY action (by any player), ALL agents update their mental models
   - Agents maintain persistent context throughout the game
   - No history logs - everything compressed into strategy + observations

3. **Opponent Modeling**
   - Agents track mental models of opponents
   - Make predictions about opponent actions
   - Learn to anticipate and counter strategies
   - Prevents strategy collapse through adaptation

## Architecture Components

### 1. Agent Memory (`cle/agents/memory.py`)

**InternalPlan** - Agent's own strategy (ground truth)
- Initial strategy set during placement
- Living todo list with priorities
- Strategic pivots when situation changes
- Private reasoning (can include bluffs)

**PerceivedPlan** - Mental model of opponent (uncertain)
- Observable facts (VPs, buildings, card counts)
- Inferred strategy and todos
- Threat assessment and confidence
- Predictions for training

**AgentMemory** - Complete memory state
- Internal plan
- Opponent perceptions for each player
- Selective event log
- Prediction results for training

### 2. LLM Agent (`cle/agents/llm_agent_impl.py`)

**StrategicLLMAgent** - Claude-powered agent with memory

Key methods:
- `set_initial_strategy()` - Big strategic decision during placement
- `select_action()` - Choose action using full context
- `update_my_plan()` - Update own strategy after acting
- `observe_opponent_action()` - Update opponent model after they act

### 3. Game Orchestrator (`cle/eval/game_orchestrator.py`)

**GameOrchestrator** - Runs games with event-driven updates

Workflow:
1. Initialize game with N LLM agents
2. Run initial placement (strategic planning phase)
3. Main game loop:
   - Agent selects action
   - Execute in game engine
   - **Trigger event**: Acting agent updates own plan
   - **Trigger event**: ALL other agents update opponent models
4. Extract results and trajectories for training

## What Gets Stored in Context

**NOT stored:**
- Turn-by-turn history (too verbose)
- All game events (too noisy)

**STORED:**
- Current strategy and active todos
- Opponent threat assessments and inferred plans
- Selective event log (agent chooses what's important)
- Current game state (board, resources, VPs)

**Compression happens naturally:**
- Strategy = compressed intent
- Opponent models = compressed observations
- Event log = sparse inflection points only

## Training Loop

1. **Self-Play**: 4 agents play full game
2. **Trajectory Collection**: Save (state, action, outcome) for each agent
3. **Fine-Tuning**: Update Claude on winning strategies
4. **Opponent Modeling**: Agents that predict opponents better win more
5. **Population**: Keep checkpoints to prevent collapse

## What's Implemented

✅ Memory structures (InternalPlan, PerceivedPlan, AgentMemory)
✅ LLM agent with strategic planning
✅ Event-driven update framework
✅ Semantic action formatting (rich text descriptions for LLM reasoning)
✅ LLMGameAccumulator (event system that triggers memory updates)
✅ LLM-driven compression (agents decide what's meaningful and extract insights)

## What Needs Implementation

### High Priority

1. **Bridge to Catanatron Game Engine**
   - [ ] Convert Catanatron State → Observation for LLM
   - [ ] Convert Catanatron Action → Tool calls
   - [ ] Create Player adapter (StrategicLLMAgent → Catanatron Player)
   - [ ] Format game state for LLM prompts

2. **Action Tools**
   - [ ] Define all action types as tools (build, trade, dev cards, etc.)
   - [ ] Map tool calls back to Catanatron Actions
   - [ ] Validate action legality

3. **Initial Placement**
   - [ ] Implement settlement placement logic
   - [ ] Strategic analysis of board positions
   - [ ] Convert placements to game actions

### Medium Priority

4. **Observation Formatting**
   - [ ] Format board state (hexes, numbers, ports)
   - [ ] Format player states (resources, buildings, VPs)
   - [ ] Include hidden information constraints

5. **Training Pipeline**
   - [ ] Trajectory storage
   - [ ] Fine-tuning data preparation
   - [ ] Modal.com integration for distributed training

### Nice to Have

6. **Optimizations**
   - [ ] Context compression strategies
   - [ ] Caching for repeated game states
   - [ ] Parallel game execution

7. **Evaluation**
   - [ ] ELO ratings
   - [ ] Strategy clustering
   - [ ] Prediction accuracy tracking

## File Structure

```
cle/
├── agents/
│   ├── memory.py              # ✅ Memory structures
│   ├── llm_agent_impl.py      # ✅ LLM agent with memory
│   └── agent_abc.py           # Base agent interface
├── eval/
│   └── game_orchestrator.py   # ✅ Event-driven game runner
├── env/
│   ├── tools/                 # ⚠️  Action tools (needs implementation)
│   └── protocols/             # Agent-to-agent communication
└── training/
    └── fine_tune.py           # ⚠️  Fine-tuning pipeline (stub)

catanatron/
└── catanatron/
    ├── game.py                # ✅ Game engine
    ├── state.py               # ✅ Game state
    └── models/                # ✅ Actions, board, etc.
```

## Example Game Flow

```python
from cle.eval.game_orchestrator import GameOrchestrator

# Setup
config = {"model": "claude-sonnet-4-5-20250929"}
orchestrator = GameOrchestrator(num_players=4, agent_config=config)

# Run game
orchestrator.initialize_game(seed=42)
results = orchestrator.run_game()

# Results include:
# - Winner
# - Each agent's final memory state
# - Prediction accuracy
# - Trajectories for training
```

## Next Steps

1. **Implement Catanatron bridge** - Connect LLM agent to game engine
2. **Define action tools** - Map all Catan actions to tool calls
3. **Test single game** - Run one complete game end-to-end
4. **Implement training loop** - Collect data and fine-tune
5. **Scale to self-play** - Run 1000s of games in parallel

## Notes

- Gym wrappers removed (not needed for LLM agents)
- Focus on tool-based interaction (Claude's native interface)
- Memory updates triggered by ALL events (fully reactive)
- Agents decide what to remember (selective logging)
- Opponent modeling prevents strategy collapse
