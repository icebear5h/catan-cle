# Credit Assignment Under Dice Variance

## The Problem

Catan has significant outcome variance from dice rolls. A strategically optimal decision can lead to bad outcomes (and vice versa) purely due to luck.

**Example**: Player places settlement on 6-ore/8-wheat intersection (excellent numbers). Over 50 rolls, the 6 hits twice and the 8 hits three times (expected: ~7 and ~7). Player loses. Was the placement bad?

No. The placement was correct. The outcome was unlucky.

**The question**: How do we train a model to recognize good decisions independent of noisy outcomes?

## Approaches

### 1. Outcome-Based (Naive)

```
reward = 1.0 if won else 0.0
```

**Pros:**
- Simple, unambiguous ground truth
- Eventually converges with enough data (law of large numbers)

**Cons:**
- Extremely high variance per game
- Requires massive data to average out luck
- May learn superstitions (spurious correlations)

**Variance estimate**: In a 4-player game, random play wins ~25%. A 5% skill edge might mean 30% win rate. Detecting this edge requires hundreds of games per decision type.

### 2. Expected Value (Theoretical)

```
reward = EV(action | board_state)
```

Calculate the "true" expected value of each action given perfect knowledge of probabilities.

**Pros:**
- Removes dice variance entirely
- Rewards correct reasoning

**Cons:**
- EV is only calculable for solo decisions (placement, robber)
- Trading has no well-defined EV (depends on opponent behavior)
- Development card timing depends on hidden information
- Computationally expensive for full game tree

### 3. Expert Matching

```
reward = 1.0 if action == expert_action else 0.0
```

Did the agent do what a strong player did in this situation?

**Pros:**
- Leverages human knowledge
- Removes dice variance (experts also face dice, but made good decisions)
- Computationally cheap

**Cons:**
- Ceiling limited by expert quality
- May not generalize beyond expert distribution
- Assumes experts play optimally (they don't always)

### 4. Hindsight Reweighting

Weight game outcomes by "luck factor":

```
luck_factor = actual_resource_income / expected_resource_income
reward = outcome * (1 / luck_factor)  # Normalize by luck
```

If you won despite bad luck, upweight. If you won with good luck, downweight.

**Pros:**
- Partially decomposes skill from luck
- Uses actual game data

**Cons:**
- Only captures resource luck, not other variance (dev cards, opponent decisions)
- Assumes linear relationship between resources and winning
- May overcorrect

### 5. Counterfactual Reasoning

For each decision, simulate alternative actions and compare expected outcomes.

```
reward = EV(action_taken) - mean(EV(alternative_actions))
```

**Pros:**
- Directly measures decision quality
- Accounts for all legal alternatives

**Cons:**
- Expensive (need simulator)
- EV calculation still has trading problem
- Simulation depth is limited

### 6. Temporal Difference (Game-Level)

Instead of per-action rewards, use TD learning across the game:

```
V(state_t) = V(state_t) + alpha * (V(state_{t+1}) - V(state_t))
```

Learn a value function, then derive action quality from value differences.

**Pros:**
- Naturally handles credit assignment over time
- Well-studied in RL literature

**Cons:**
- Requires accurate value function (chicken-egg)
- Still has variance in final outcome
- Complex to implement

## V0 Recommendation

**Hybrid: Outcome + Expert Matching**

```python
def compute_reward(action, expert_action, game_outcome, is_winner):
    # Base: did you win?
    outcome_reward = 1.0 if is_winner else 0.0

    # Bonus: did you match expert?
    expert_bonus = 0.2 if action == expert_action else 0.0

    return outcome_reward + expert_bonus
```

**Rationale:**
- Outcome reward provides ground truth signal (noisy but real)
- Expert matching reduces variance and bootstraps from human knowledge
- Simple to implement
- Can weight by game outcome (expert match in winning game > expert match in losing game)

**For V1**: Add hindsight luck correction and possibly counterfactual analysis for key decisions (initial placement, robber).

## Open Questions

1. **How similar is "similar enough"?** Expert might place settlement on node 42, agent places on adjacent node 43. Is that a match? Need fuzzy matching for some action types.

2. **What about novel good moves?** Agent finds a move experts never made but is actually better. Pure expert matching penalizes this.

3. **Turn-level vs game-level?** Should reward be per-turn or only at game end? Per-turn is denser but may have wrong signs.

## Experiments to Run

1. Train on pure outcome reward, measure variance in learned policy
2. Train on pure expert matching, measure ceiling
3. Train on hybrid, compare
4. Ablate the expert bonus weight (0.1, 0.2, 0.5)
