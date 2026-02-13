# Trading Reward Design

## The Problem

Trading in Catan is fundamentally different from other decisions because success depends on opponent cooperation, which is:

1. Not calculable from game state alone
2. Influenced by social dynamics, psychology, and metagame
3. Rational for opponents to refuse trades that help the leader

**The paradox**: A "good" trade offer might be one that opponents accept. But opponents accepting means the trade also benefits them. The best trades for you are ones opponents should refuse.

## Trading Types

### 1. Maritime Trading (Solo)
- 4:1 bank trades (always available)
- 3:1 port trades (if you have generic port)
- 2:1 port trades (if you have specific port)

**This is calculable.** No opponent interaction. Pure resource conversion math.

### 2. Player Trading (Social)
- Propose trades to other players
- Accept/reject incoming offers
- Counteroffers and negotiation

**This is not cleanly calculable.** Success depends on:
- Opponent resource needs
- Opponent perception of your win probability
- Social dynamics (grudges, alliances)
- Metagame (reputation from past games)

## Approaches to Trading Rewards

### 1. Skip Player Trading (V0)

Just don't model it. Maritime trades only.

**Pros:**
- Eliminates the problem entirely
- Maritime trading is still strategically interesting
- Simplifies action space

**Cons:**
- Missing huge part of Catan strategy
- Expert replays include trading (data mismatch)
- Eventually need to solve this anyway

**V0 viability**: HIGH. Maritime-only is a reasonable simplification.

### 2. Reward Trade Execution

```
reward = resource_gained_value - resource_given_value
```

Only reward trades that actually happen. Measure the "spread" you captured.

**Pros:**
- Simple to compute
- Rewards extracting value from trades

**Cons:**
- Doesn't reward good offers that got rejected
- Doesn't penalize bad offers that got accepted (maybe opponent made a mistake)
- Resource values are context-dependent (ore worth more early game)

### 3. Reward Trade Attempts (Offer Quality)

Reward based on whether the offer was "reasonable" independent of acceptance.

```
offer_quality = your_resource_EV / their_resource_EV
reward = 1.0 if offer_quality > 1.0 else 0.0  # You asked for more value
```

**Pros:**
- Rewards good offers even if rejected
- Doesn't reward lucky acceptances of bad offers

**Cons:**
- What's "reasonable"? Resource values are dynamic
- Ignores context (desperate need vs nice-to-have)
- Doesn't account for opponent win probability

### 4. Expert Matching for Trades

Did you trade like the expert did?

**Pros:**
- Leverages human judgment on trade quality
- Implicitly captures social dynamics that worked

**Cons:**
- Expert's trades worked in their games with their opponents
- Different opponents = different optimal trades
- May learn to mimic suboptimal patterns

### 5. Outcome Correlation

Correlate trading behavior with game outcomes across many games.

```
reward = correlation(trading_pattern, win_rate)
```

Learn that certain trading patterns (e.g., "trade aggressively early, defensively late") correlate with winning.

**Pros:**
- Data-driven, no manual value function
- Captures patterns that actually work

**Cons:**
- High variance, needs lots of data
- May learn spurious correlations
- Doesn't help with individual trade decisions

### 6. Counterfactual: "Would a rational opponent accept?"

Model opponent decision-making. Reward trades that a rational opponent would accept.

```
opponent_value = value_to_opponent(their_resources, your_offer)
your_value = value_to_you(your_resources, their_offer)

# Good trade = both sides gain (or opponent gains and you gain more)
reward = your_value if opponent_value > 0 else 0
```

**Pros:**
- Grounds trading in game theory
- Can reason about trade feasibility

**Cons:**
- Requires accurate opponent modeling
- "Rational" opponent is not realistic
- Circular: need to know what opponent values, but they value what helps them win

## V0 Recommendation

**Maritime-only trading for self-play, full trading in SFT data.**

```python
# SFT data: include expert trades with outcome labeling
# Self-play: maritime trades only

def get_legal_trades(game_state, player_id):
    trades = []

    # Always include maritime trades
    trades.extend(get_maritime_trades(game_state, player_id))

    # V0: skip player trades in self-play
    # V1+: add player trades

    return trades
```

**Rationale:**
- SFT on expert replays teaches what good trades look like
- Self-play removes the social modeling problem
- Policy learns trading patterns from experts, practices solo decisions in self-play
- Clear upgrade path: add player trading in V1 once we have a working loop

## V1+ Ideas

### Simplified Player Trading

Instead of full negotiation, use a restricted protocol:

1. Active player proposes trade (giving X for Y)
2. Each opponent says accept/reject (no counteroffers)
3. If multiple accept, active player chooses partner

This is still multiplayer but removes negotiation complexity.

### Opponent Modeling

Train a separate "opponent response model":
- Input: game state, trade offer
- Output: P(accept | opponent_i)

Use this to filter trade proposals: only propose trades with P(accept) > threshold.

### Self-Play with Fictitious Play

All 4 agents are the same policy. They learn to trade with each other. Equilibrium emerges.

Problem: may converge to "never trade" equilibrium if trading helps opponents too much.

### Reward Shaping for Trading

Add domain knowledge:
- Reward port access (enables better maritime rates)
- Reward resource diversity (more trade options)
- Penalize resource hoarding (can't use it, could trade it)

## Open Questions

1. **Is maritime-only too limiting?** Some games are won/lost on player trades. Missing this might cap performance.

2. **Can we learn trade value from data?** Maybe train a model: (game_state, trade) -> win_probability_delta. Use this as trade reward.

3. **How do experts decide when to trade?** Analyze Colonist data for trading patterns. Are there heuristics we can extract?

## Experiments to Run

1. Compare win rates: maritime-only vs player-trading (with random acceptance model)
2. Analyze Colonist games: what % of trades are "good" trades (both parties better off)?
3. Cluster trading patterns by player ELO, see if strong players trade differently
