# Turn Planning Architecture (Final Design)

## Core Principle

**One LLM call per turn outputs structured decision with opponent modeling.**

This gives us:
- Clear training signal (plan → outcome)
- Opponent modeling (prevents strategy collapse)
- Strategic reasoning (multi-step thinking)
- Efficient inference (one call per turn)

## Structured Output

### Part 1: Self Assessment & Plan

```
SITUATION: At 6 VP, strong ore production, have resources
GOAL: Build city at node 42
VP_GAIN: 2
REASONING: Doubles production on best spot, most efficient path to 10 VP
ALTERNATIVES: [Build settlement, Buy dev card]
RISKS: [RED might win next turn, Might seven out]
ACTION_SEQUENCE: BUILD_CITY -> END_TURN
```

**Why this matters:**
- Training: (situation, goal, reasoning, outcome)
- Shows strategic thinking
- Clear intent for action sequence

### Part 2: Opponent Modeling

```
OPPONENT: RED
  THREAT_LEVEL: HIGH (7 VP, going for longest road)
  STRATEGY: Building toward longest road
  PREDICTION: Will build road next turn
  NOTES: Need to block or race them

OPPONENT: BLUE
  THREAT_LEVEL: LOW (4 VP, buying dev cards)
  STRATEGY: Dev card focus
  PREDICTION: Will buy dev card
  NOTES: Not immediate threat
```

**Why this matters:**
- Prevents strategy collapse (agents learn to counter each other)
- Training includes opponent prediction accuracy
- Agents learn from predicting other agents
- Models evolve together (co-evolution)

### Part 3: Action Sequence

```
SEQUENCE:
  1. OFFER_TRADE to RED (3 sheep for 1 ore)
     IF accepted:
       BUILD_CITY at node 42
       END_TURN
     IF rejected:
       MARITIME_TRADE (4:1)
       BUILD_CITY at node 42
       END_TURN

FIRST_ACTION: 2 (OFFER_TRADE is index 2 in valid_actions)
```

**Why this matters:**
- Executable plan with branches
- Clear contingencies
- Maps to engine actions

## Training Data Format

From a single turn, we get:

```json
{
  "game_state": {...},
  "turn_decision": {
    "self_plan": {
      "goal": "Build city at node 42",
      "reasoning": "Doubles production, best VP efficiency",
      "expected_vp": 2
    },
    "opponent_models": [
      {
        "color": "RED",
        "threat": "HIGH",
        "prediction": "Build road",
        "actual": "Built road"  // Added after turn executes
      }
    ],
    "action_sequence": "OFFER_TRADE -> BUILD_CITY -> END_TURN",
    "executed_actions": ["OFFER_TRADE", "MARITIME_TRADE", "BUILD_CITY", "END_TURN"]
  },
  "outcome": {
    "vp_gained": 2,
    "plan_succeeded": true,
    "prediction_accuracy": 0.67  // 2/3 opponents predicted correctly
  }
}
```

## Why This Beats Action-by-Action

**Action-by-action:**
```
Turn start -> Pick action -> Execute -> Pick action -> Execute -> ...
- 4 LLM calls per turn
- No strategic coherence
- Hard to attribute success to specific decisions
- No opponent modeling
```

**Upfront plan:**
```
Turn start -> Plan entire turn -> Execute sequence
- 1 LLM call per turn
- Clear strategic intent
- Easy attribution (plan quality -> outcome)
- Opponent predictions can be validated
```

## Implementation Flow

### 1. Turn Start

```python
# Get game state
state = game.state
valid_actions = state.playable_actions

# Build observation
obs = create_observation(state, player_color)

# Show opponent states
opponents = get_opponent_info(state, player_color)

# Prompt LLM
prompt = f"""
GAME STATE:
{obs}

OPPONENTS:
{opponents}

VALID ACTIONS:
{format_actions(valid_actions)}

Plan your turn with opponent analysis.
"""

# Get structured output
turn_decision = llm.generate(prompt)
```

### 2. Parse & Validate

```python
parsed = parse_turn_decision(turn_decision, valid_actions)

# Validate plan is legal
assert parsed.first_action_index < len(valid_actions)
assert all contingencies map to valid future states
```

### 3. Execute Sequence

```python
current_action = valid_actions[parsed.first_action_index]

while not is_turn_over:
    # Execute action
    result = game.step(current_action)

    # Determine next action based on outcome
    if result.requires_choice:  # e.g., trade rejected
        # Follow contingency plan
        outcome_key = get_outcome_key(result)
        next_index = parsed.contingency_plans[outcome_key]
        current_action = get_valid_actions()[next_index]
    else:
        # Deterministic - get next action from plan
        current_action = get_next_planned_action()
```

### 4. Record for Training

```python
training_example = {
    "state": state,
    "plan": parsed.self_plan,
    "opponent_predictions": parsed.opponent_models,
    "executed_actions": actions_taken,
    "outcome": {
        "vp_gained": final_vp - initial_vp,
        "success": final_vp > initial_vp
    }
}

# Later: evaluate prediction accuracy
for opponent_model in parsed.opponent_models:
    actual_action = get_opponent_action(opponent_model.color)
    accuracy = compare(opponent_model.prediction, actual_action)
```

## Opponent Modeling Benefits

### For Training

**Multi-task learning:**
- Primary: Maximize own VP
- Secondary: Predict opponent actions
- Auxiliary: Assess threat levels

**Better representations:**
- Agent learns to encode "what makes a good city spot"
- Also learns to recognize when opponents see good spots
- Shared representation improves both

### For Strategy

**Counter-play:**
- "RED is going for longest road → I should block or race"
- "BLUE has lots of sheep → they want dev cards → robber them"

**Adaptation:**
- Early models: random play
- Mid training: learn building patterns
- Late training: meta-game (counter opponent strategies)

### For Preventing Collapse

**Strategy diversity:**
- Agents that predict opponents better win more
- Must adapt to changing opponent strategies
- Can't just learn one dominant strategy

**Population training:**
- Keep historical checkpoints
- New models train against diverse opponents
- Prevents rock-paper-scissors collapse

## Prompt Template

```
You are {COLOR} playing Settlers of Catan.

=== GAME STATE ===
{formatted_observation}

=== OPPONENTS ===
{opponent_states}

=== VALID ACTIONS ({len(valid_actions)} total) ===
{action_list_with_indices}

=== INSTRUCTIONS ===
Plan your turn using this format:

=== SELF ASSESSMENT ===
SITUATION: <Your current position>
GOAL: <What you want to accomplish>
VP_GAIN: <Expected VP gain>
REASONING: <Why this goal>
ALTERNATIVES: <Other options considered>
RISKS: <What could go wrong>

=== OPPONENT ANALYSIS ===
[For each opponent]
OPPONENT: <Color>
THREAT_LEVEL: <LOW/MEDIUM/HIGH/CRITICAL>
STRATEGY: <What they're doing>
PREDICTION: <What they'll do next turn>
NOTES: <Key observations>

=== ACTION PLAN ===
SEQUENCE: <Your action sequence with branches>
FIRST_ACTION: <index of first action>

=== NOTES FOR NEXT TURN ===
<Brief summary to remember>
```

## Next Steps

1. Implement prompt builder
2. Build parser for structured output
3. Create execution engine
4. Test with real games
5. Validate training data format
6. Start collecting self-play data

This architecture is:
- Simple to implement
- Rich training signal
- Includes opponent modeling
- One call per turn
- Clear strategic reasoning
