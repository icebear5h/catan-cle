# Credit Assignment Proposal: Turn-by-Turn Resource-Aware Scoring

## Problem

REINFORCE with win/loss reward conflates dice luck with decision quality.
With a ~1-2k game budget, we can't rely on volume to average out stochasticity.
We need a per-turn credit signal that reflects **decision quality, not dice luck**.

## Core Idea

At every turn, compute a **Position Score** that captures how good your situation is
independent of how you got there. The **advantage** of each decision is the delta
in position score that the decision caused, minus the delta that luck caused.

```
decision_credit(turn t) = position_score(t+1) - position_score(t) - luck_delta(t)
```

## Position Score (computed per turn, per player)

Simplified: VPs + expected resource income toward your strategy.

### 1. Victory Points
- Raw VP count (settlements, cities, longest road, largest army, dev cards)
- Ground truth objective, always counts

### 2. Strategy-Weighted Expected Resource Income
- For each settlement: sum of pip probabilities on adjacent tiles
- For each city: 2x the pip probability
- Pip probability: number of ways to roll that number / 36
- **Weighted by strategy relevance**: only resources the agent currently needs count fully
- Resources irrelevant to current strategy are discounted to near-zero

```
expected_income(t) = sum over tiles:
    pip_prob(tile.number) * (1 if settlement, 2 if city) * strategy_weight(resource)
```

Strategy weights come from the agent's **strategic notes** (see below).

## Strategic Notes (agent-declared, persistent across turns)

The LLM agent maintains a persistent `strategic_notes` block that carries across turns.
It only rewrites when something meaningful changes (new settlement placed, strategy pivot, etc).

### What the Agent Declares
```
strategic_notes:
  current_strategy: "cities"           # or "longest_road", "dev_cards", "balanced"
  priority_resources: [ORE, WHEAT]     # what I need right now
  next_build: "city"                   # immediate build goal
  secondary_resources: [SHEEP]         # nice to have
  irrelevant_resources: [WOOD, BRICK]  # don't care about these currently
```

### Dual Purpose
1. **Gameplay**: Agent doesn't re-derive its plan every turn. Saves tokens.
   Only updates when board state changes meaningfully (new building, robber moved,
   opponent took longest road, etc.)
2. **Credit assignment**: The priority_resources list directly feeds into luck delta
   and trade evaluation. The system knows what the agent *wanted*, so it can judge
   whether dice/trades helped the actual plan or were irrelevant noise.

### Strategy Weight Function
```python
def strategy_weight(resource, strategic_notes):
    if resource in strategic_notes.priority_resources:
        return 1.0     # fully relevant
    if resource in strategic_notes.secondary_resources:
        return 0.4     # somewhat useful
    if resource in strategic_notes.irrelevant_resources:
        return 0.05    # near-zero, only matters for trades
    return 0.2          # unclassified default
```

## Luck Delta (per-resource, strategy-aware)

Isolates the portion of position change attributable to dice, not decisions.
Computed **per resource** and filtered by what the agent actually needs.

### Per Turn, Per Resource:
```
for each resource R:
    expected_R(t) = sum over tiles producing R:
        pip_prob(tile.number) * (1 if settlement, 2 if city)

    actual_R(t) = amount of R received from dice this turn

    luck_R(t) = actual_R(t) - expected_R(t)

# Total luck delta, weighted by strategy relevance:
luck_delta(t) = sum over resources R:
    luck_R(t) * strategy_weight(R, strategic_notes(t))
```

### Why Per-Resource Matters
- Rolling 3 ore when you're going for longest road: luck_delta ~ 0 (ore is irrelevant)
- Rolling 2 wood when you need wood for roads: luck_delta = positive (genuinely lucky)
- Rolling nothing when you expected wheat for cities: luck_delta = negative (unlucky)

The same dice roll has different luck values depending on the agent's declared strategy.

## Trade Credit (strategy-aware)

Trades are pure decisions with zero luck. Credit is based on whether the trade
moved resources toward the agent's strategic goals.

```python
def trade_credit(gave, received, strategic_notes):
    gave_value = sum(strategy_weight(r, strategic_notes) for r in gave)
    received_value = sum(strategy_weight(r, strategic_notes) for r in received)
    return received_value - gave_value
```

- Trading away irrelevant ore for priority wheat: high positive credit
- Trading away priority wheat for irrelevant wood: high negative credit
- Maritime 4:1 trade of junk for needed resource: moderate positive credit

### Scarcity Modifier
Trade credit is also modulated by board-level scarcity:
- Trading for a resource that's scarce on the board (few tiles, robber blocking)
  gets a bonus -- the agent recognized scarcity and acted on it
- Trading for an abundant resource is less impressive

## Decision Classification

Not all actions are equal. Tag each action by type for credit weighting:

### High-Signal Decisions (full credit)
- **Initial placement**: Settlement + road placement (highest impact in the game)
- **Trading**: Accepting, rejecting, proposing trades (pure decision, zero luck)
- **Building**: What to build and where (resource allocation decision)
- **Dev card play**: When to play knight, monopoly, year of plenty
- **Robber placement**: Where to move robber + who to steal from

### Low-Signal Actions (reduced credit)
- **Rolling dice**: No decision involved (mandatory action)
- **Forced discards on 7**: Some decision, but constrained
- **End turn**: Usually obvious when to stop

### Zero-Signal Events (excluded from credit)
- **Resource collection from dice**: Pure luck
- **Being robbed**: Pure luck (from your perspective)
- **Bank depletion effects**: Environmental, not your decision

## Resource Provenance Tracking

### What to Add to the Engine

Currently `state.py` tracks resources as 5-int freqdecks with no history.
Add a provenance log per player:

```python
# New field on game state
resource_log: dict[Color, list[ResourceEvent]] = {}

@dataclass
class ResourceEvent:
    turn: int
    resource: str          # WOOD, BRICK, etc.
    amount: int            # positive = gained, negative = spent
    source: str            # "dice", "trade_player", "trade_maritime", "dev_card",
                           #  "steal", "build_cost", "discard"
    counterparty: Color | None  # who you traded with, if applicable
    tile_number: int | None     # which dice number produced it, if from dice
```

### Where to Hook In

1. `yield_resources()` (state.py:282) - tag with source="dice", tile_number=roll
2. `apply_confirm_trade()` (state.py:857) - tag with source="trade_player", counterparty
3. `apply_maritime_trade()` (state.py:743) - tag with source="trade_maritime"
4. `buy_dev_card()` (state.py:258) - tag cost with source="build_cost"
5. `build_settlement/city/road()` - tag cost with source="build_cost"
6. `apply_steal()` - tag with source="steal"
7. `apply_discard()` - tag with source="discard"

## Turn-by-Turn Credit Computation

After each game, for every turn t where a player made a decision:

```python
def compute_turn_credit(game_log, player, turn):
    # 1. Position delta
    pos_before = position_score(state_at(turn), player)
    pos_after = position_score(state_at(turn + 1), player)
    pos_delta = pos_after - pos_before

    # 2. Luck delta (resources from dice this turn vs expected)
    actual = resources_from_dice(game_log, player, turn)
    expected = expected_income(state_at(turn), player)
    luck = resource_value(actual) - resource_value(expected)

    # 3. Decision credit = position improvement minus luck contribution
    credit = pos_delta - luck

    # 4. Scale by decision type
    action = action_at(turn, player)
    credit *= decision_weight(action.type)

    return credit
```

## Reward Assignment for REINFORCE

For each game, instead of:
```
reward = +1 if win, -1 if loss (applied to all turns equally)
```

Do:
```
reward(turn t) = alpha * turn_credit(t) + beta * game_outcome
```

Where:
- `alpha` controls how much per-turn credit matters (~0.7)
- `beta` controls how much final outcome matters (~0.3)
- `game_outcome` = +1 win, -1 loss, scaled by luck_adjustment

```
luck_adjustment = 1.0 - sigmoid(total_luck_score)
```

Lucky win = small game_outcome contribution.
Unlucky win = large game_outcome contribution.

## Implementation Priority

### Phase 1: Resource Provenance (engine change)
- Add ResourceEvent logging to all resource flows
- No training impact yet, just data collection
- Validate by replaying games and checking resource accounting balances

### Phase 2: Position Score (new module)
- Implement heuristic position evaluator
- Compute per-turn scores for recorded games
- Validate: does position score correlate with eventual winners?

### Phase 3: Luck Delta (new module)
- Expected income calculator from board state
- Resource valuation function
- Per-turn luck isolation

### Phase 4: Credit Assignment Integration (training change)
- Replace flat win/loss reward with per-turn credit
- A/B test: flat reward vs credit-assigned reward on same game count
- Measure: does credit assignment learn faster in 500 games?

## Open Questions

1. **Strategic notes honesty**: The agent declares its own strategy. What if it declares
   wrong priorities? Over time, bad declarations lead to worse credit scores and get
   selected against -- but early in training this could be noisy. May need to bootstrap
   with a simple heuristic strategy detector for the first N games.
2. **When to trigger notes rewrite**: Too often = wasted tokens. Too rarely = stale strategy.
   Candidates: after building, after being robbed, after a failed trade, after an opponent
   hits a VP milestone. Need to define the trigger set.
3. **Opponent modeling**: Credit should account for opponent strength. Beating strong
   play > beating random. Not addressed here yet.
4. **Dev card stochasticity**: Buying a dev card has variance (knight vs VP).
   Should luck-adjust using deck composition probabilities at time of purchase.
5. **Scarcity computation**: How to efficiently compute board-level resource scarcity
   for the trade credit modifier. Tile counts + robber position + bank depletion.
