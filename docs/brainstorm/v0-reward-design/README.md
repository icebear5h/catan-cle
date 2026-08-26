# V0 Reward Design Decision

## Goal

Define the simplest reward function that:
1. Provides enough signal to learn something useful
2. Can be implemented in < 1 day
3. Doesn't require solving unsolved problems (trading, perfect credit assignment)
4. Has a clear upgrade path to V1+

## Constraints

- Must work with existing Colonist replay data
- Must work with self-play (4 copies of same policy)
- Should not require additional infrastructure (reward model, opponent modeling)

## Decision: Outcome-Weighted SFT

**For SFT (supervised fine-tuning on Colonist replays):**

```python
def compute_sft_weight(game_outcome: str, turn_number: int, total_turns: int) -> float:
    """
    Weight training examples by game outcome.
    Winning game actions get higher weight.
    """
    # Base weight by outcome
    if game_outcome == "win":
        base = 1.0
    elif game_outcome == "loss":
        base = 0.3  # Still learn from losses, but downweight
    else:
        base = 0.5  # Draw or unknown

    # Optional: weight later turns higher (closer to outcome)
    # turn_weight = 0.5 + 0.5 * (turn_number / total_turns)
    # return base * turn_weight

    return base
```

**For self-play:**

```python
def compute_self_play_reward(game_result: dict, player_id: int) -> float:
    """
    Simple outcome reward for self-play.
    """
    if game_result["winner"] == player_id:
        return 1.0
    else:
        # Scale by final VP to give partial credit
        my_vp = game_result["vp"][player_id]
        winner_vp = game_result["vp"][game_result["winner"]]
        return my_vp / winner_vp  # 0.0 to ~0.9 depending on how close
```

## Why This Works (Good Enough)

### 1. Outcome Signal is Real

Yes, it's noisy. Yes, dice variance exists. But over thousands of games, good strategy correlates with winning. The law of large numbers is on our side.

**Key insight**: We're not trying to learn from one game. We're learning from 8,500 games. Dice variance averages out.

### 2. Expert Actions Bootstrap Quality

The SFT phase trains on expert actions. These actions are already filtered by human judgment. We're not learning from random play - we're learning from strong players.

**The reasoning traces add signal**: When we generate reasoning for expert actions, we're injecting strategic knowledge into the training data. The model learns *why* actions are good, not just which actions to take.

### 3. Self-Play Refines

Self-play with outcome rewards reinforces what works. Even with noise, actions that consistently lead to wins get reinforced. Actions that consistently lose get suppressed.

**The 4-copies setup helps**: All players are the same policy. If a strategy is good against itself, it's probably good. No need to model diverse opponents in V0.

### 4. Upgrade Path is Clear

V0 reward is a stepping stone:
- V1: Add reward model (trained on replay data) for per-turn scoring
- V2: Add hindsight luck correction
- V3: Add counterfactual analysis for key decisions

We're not locked in. We're just getting started.

## What We're NOT Doing in V0

| Feature | Why Not V0 |
|---------|-----------|
| Reward model | Requires training separate model, adds complexity |
| Per-turn rewards | Need to define intermediate value function |
| Trading rewards | Unsolved problem, maritime-only is fine |
| Luck correction | Adds complexity, marginal benefit with enough data |
| Expert matching bonus | Complicates loss function, SFT already uses expert data |

## Implementation

### SFT Data Generation

```python
# Sketch only: the authoritative exporter must supply a pre-action packet.

def generate_sft_example(decision_packet: dict) -> dict:
    observation = decision_packet["observation"]
    action = decision_packet["expert_action"]
    reasoning = generate_reasoning(decision_packet)  # LLM call

    outcome = decision_packet["trajectory_outcome"]
    weight = compute_sft_weight(
        outcome,
        decision_packet["decision_index"],
        decision_packet["decision_count"],
    )

    return {
        "input": observation,
        "output": f"<reasoning>\n{reasoning}\n</reasoning>\n<action>{action}</action>",
        "weight": weight
    }
```

### Self-Play Loop

```python
# In self_play.py

def run_self_play_game(policy) -> List[dict]:
    """Run one game, return training examples."""
    env = CatanEnv()
    trajectory = []

    obs = env.reset()
    while not env.done:
        player = env.current_player
        action, reasoning = policy.act(obs[player])

        trajectory.append({
            "player": player,
            "observation": obs[player],
            "action": action,
            "reasoning": reasoning
        })

        obs, _, done, info = env.step(action)

    # Compute rewards based on outcome
    winner = env.winner
    for example in trajectory:
        example["reward"] = compute_self_play_reward(
            {"winner": winner, "vp": env.victory_points},
            example["player"]
        )

    return trajectory
```

### Training

```python
# In training.py

def train_lora(sft_data: List[dict], self_play_data: List[dict]):
    """
    Combine SFT and self-play data for training.
    """
    # SFT data uses weights, self-play uses rewards
    combined = []

    for ex in sft_data:
        combined.append({
            "input": ex["input"],
            "output": ex["output"],
            "weight": ex["weight"]
        })

    for ex in self_play_data:
        combined.append({
            "input": ex["observation"],
            "output": f"<reasoning>\n{ex['reasoning']}\n</reasoning>\n<action>{ex['action']}</action>",
            "weight": ex["reward"]
        })

    # Standard SFT with sample weights
    train_weighted_sft(combined)
```

## Success Criteria

V0 is successful if:

1. **Trains without crashing** - Full loop runs end-to-end
2. **Beats random** - Policy wins > 30% against 3 random players (baseline: 25%)
3. **Learns something** - Actions are not uniform random, show strategic patterns
4. **Iterates** - Self-play data improves policy on subsequent rounds

We're not trying to make a Catan god in V0. We're trying to prove the loop works.

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Dice variance swamps signal | Medium | Use lots of games, outcome weighting |
| Learns superstitions | Medium | Monitor action diversity, sanity check policies |
| Never learns trading | High (by design) | V0 is maritime-only, V1 adds trading |
| SFT ceiling too low | Low | Experts are strong, 8.5K games is decent data |
| Self-play collapse | Medium | Monitor win distribution, add exploration if needed |

## Next Steps After V0

1. **Evaluate thoroughly** - Where does V0 policy fail? What decisions are bad?
2. **Build reward model** - Train on (state, action, outcome) to predict win probability
3. **Add per-turn rewards** - Use reward model for denser signal
4. **Add trading** - Start with simplified protocol, expand later
5. **Scale data** - Use full 8.5K games instead of 1K subset

## Summary

V0 reward is dead simple:
- **SFT**: Train on expert actions, weight by game outcome
- **Self-play**: Reward = 1 if win, scaled by VP otherwise

This is enough to get started. Sophistication comes later.
