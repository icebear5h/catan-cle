# Turn-Based Planning Architecture

## Problem with Action-Based Decisions

**Current approach (broken):**
```
LLM: "What action should I take?"
Engine: "You can: END_TURN, MARITIME_TRADE(...)"
LLM: "I'll end turn" ❌
```

**Why it fails:**
- LLM doesn't see that trading → building is possible IN THE SAME TURN
- Each decision is myopic
- Can't plan multi-step sequences
- Misses obvious opportunities

## Turn-Based Planning Solution

**New approach:**
```
LLM: "What goal should I accomplish this turn?"
Planner: "You can:
  1. Trade 3 sheep for ore, then build city (+2 VP)
  2. Build settlement at node 42 (+1 VP)
  3. End turn and save resources"
LLM: "I'll do #1 - that's 2 VP" ✅
Planner: Executes [MARITIME_TRADE, BUILD_CITY] sequence
```

## Architecture

### 1. Turn Planner (`turn_planner.py`)

Analyzes current state and generates **TurnGoals**:

```python
@dataclass
class TurnGoal:
    goal: str  # "Build city on node 42"
    actions: List[SemanticAction]  # [trade action, build action]
    final_vp: int
    strategic_value: str
```

**Goal categories:**
- **Immediate**: Can do right now with current resources
- **Trade-enabled**: Requires trading first
- **Complex**: Multi-step sequences (dev cards, etc)
- **Conservative**: End turn, prepare for next

### 2. Semantic Actions (`SemanticAction`)

Actions carry meaning, not just data:

```python
@dataclass
class SemanticAction:
    action: Action  # Raw engine action
    description: str  # "Build city at node 42"
    immediate_effect: str  # "Gain 2 VP, double ore production"
    enables: List[str]  # ["Can build dev cards easier"]
    costs: Dict[str, int]  # {"WHEAT": 2, "ORE": 3}
    vp_gain: int
```

**Why this matters:**
- LLM can reason about action consequences
- Prompts include strategic context
- Training data captures WHY actions are good

### 3. Turn-Level Prompts

Prompt asks about TURNS, not ACTIONS:

```
What do you want to accomplish THIS TURN?

Available goals:
1. Build city at node 42
   - Actions: Trade 3 sheep → 1 ore, Build city
   - VP gain: +2
   - Effect: Double production on wheat/ore spot
   - Strategic value: Highest VP per resource spent

2. Build settlement at node 15
   - Actions: Build settlement
   - VP gain: +1
   - Effect: Gain access to wood/brick production
   - Strategic value: Expand to new resources

3. End turn
   - Actions: End turn
   - VP gain: 0
   - Effect: Save resources for next turn
   - Strategic value: Preserve options

Choose goal index: __
```

### 4. Action Sequencing

Once LLM picks a goal, execute the action sequence:

```python
class TurnExecutor:
    def execute_goal(self, goal: TurnGoal):
        for semantic_action in goal.actions:
            # Execute each action in sequence
            self.game.step(semantic_action.action)

            # Store intermediate state for next action
            self.turn_state.update(game.state)
```

### 5. State Persistence

**Between actions in the same turn:**
- Track resources spent/gained
- Update available actions
- Adjust remaining goals

**Example:**
```
Turn start: {WHEAT: 3, ORE: 3, SHEEP: 4}
After trade: {WHEAT: 3, ORE: 4, SHEEP: 1}  ← State updated
After build: {WHEAT: 1, ORE: 1, SHEEP: 1}  ← City built
Turn end
```

## Implementation Plan

### Phase 1: Semantic Actions ✅ (Started)
- [x] Create SemanticAction class
- [ ] Implement semantic wrappers for all action types
- [ ] Extract strategic context (what action enables)
- [ ] Add cost/benefit calculations

### Phase 2: Turn Planning
- [ ] Implement immediate goals (direct actions)
- [ ] Implement trade-enabled goals
- [ ] Implement complex goals (dev cards, multi-step)
- [ ] Add goal ranking/prioritization

### Phase 3: Prompt Integration
- [ ] Update prompts to present goals, not actions
- [ ] Add turn-level strategic context
- [ ] Format goals with semantic information
- [ ] Parse LLM goal selection

### Phase 4: Execution
- [ ] Build turn executor
- [ ] Handle action sequences
- [ ] Manage state between actions
- [ ] Error handling (action becomes invalid mid-turn)

### Phase 5: Training Data
- [ ] Capture turn goals in training examples
- [ ] Store action sequences with reasoning
- [ ] Label with outcomes (did goal succeed?)
- [ ] Format for fine-tuning

## Examples

### Example 1: Trade to Enable Build

**State:**
- Resources: WHEAT=1, ORE=2, SHEEP=4, WOOD=2, BRICK=1
- Want: Build city (need WHEAT=2, ORE=3)

**Turn Planner generates:**
```python
TurnGoal(
    goal="Build city at node 42",
    actions=[
        SemanticAction(
            action=MARITIME_TRADE([SHEEP, SHEEP, SHEEP, None, None, ORE]),
            description="Trade 3 sheep for 1 ore (3:1 port)",
            immediate_effect="Spend 3 sheep, gain 1 ore",
            enables=["Can now afford city"],
            costs={"SHEEP": 3}
        ),
        SemanticAction(
            action=BUILD_CITY(node_id=42),
            description="Build city at node 42",
            immediate_effect="Gain 2 VP, double production",
            enables=["Closer to victory", "More ore per turn"],
            costs={"WHEAT": 2, "ORE": 3}
        )
    ],
    final_vp=2,
    strategic_value="Best available VP gain this turn"
)
```

**LLM sees:**
```
Goal: Build city at node 42 (+2 VP)
Plan: Trade 3 sheep for ore, then build city
Strategic value: Doubles production on your best spot
```

### Example 2: Multi-Step Development

**State:**
- Have: ROAD_BUILDING dev card
- Want: Expand to new settlement location

**Turn Planner generates:**
```python
TurnGoal(
    goal="Expand to high-value settlement spot",
    actions=[
        SemanticAction(
            action=PLAY_ROAD_BUILDING(),
            description="Play road building card",
            immediate_effect="Build 2 free roads",
            enables=["Reach node 50"]
        ),
        SemanticAction(
            action=BUILD_ROAD(edge=(40, 45)),
            description="Build road toward expansion"
        ),
        SemanticAction(
            action=BUILD_ROAD(edge=(45, 50)),
            description="Build road to node 50"
        ),
        SemanticAction(
            action=BUILD_SETTLEMENT(node_id=50),
            description="Build settlement at node 50 (10 pips)",
            immediate_effect="Gain 1 VP, excellent production"
        )
    ],
    final_vp=1,
    strategic_value="High-value expansion using dev card"
)
```

## Benefits

### For Training:
- Captures strategic reasoning at turn level
- Shows how actions compose
- Includes cost/benefit analysis
- Better training signal (goal success/failure)

### For Performance:
- Reduces API calls (one decision per turn, not per action)
- Better planning (sees multi-step opportunities)
- More strategic play (thinks about turn goals)

### For Debugging:
- Clear turn-level intentions
- Can replay turn goals
- Easier to understand agent strategy
- Better evaluation metrics

## Next Steps

1. Finish implementing turn_planner.py
2. Build semantic action wrappers for all action types
3. Update prompts to use turn goals
4. Test with real gameplay
5. Iterate on goal generation heuristics
