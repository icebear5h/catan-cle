# Reward Design Brainstorm

This folder contains working documents for solving the fundamental reward design problem in Catan RL.

## The Core Tension

We face a chicken-and-egg problem:

1. **Outcome-based rewards** (win/loss, VP gained) are noisy because dice variance swamps strategic signal
2. **EV-based rewards** (expected value calculations) break down for trading because trade success depends on opponent cooperation, which isn't calculable

## Documents

| Document | Problem | Status |
|----------|---------|--------|
| [credit-assignment/](./credit-assignment/README.md) | How to assign credit under dice randomness | WIP |
| [trading-rewards/](./trading-rewards/README.md) | How to reward trading decisions | WIP |
| [v0-reward-design/](./v0-reward-design/README.md) | Final V0 reward function decision | WIP |

## Key Questions

1. **Credit Assignment**: A player makes a "correct" settlement placement on an 8-ore hex, but the 8 never rolls. How do we reward the decision vs the outcome?

2. **Trading Paradox**: A player offers a fair trade that would help them win. Opponents refuse (correctly). Is proposing the trade good or bad? The action's quality depends entirely on opponent response.

3. **Sparse vs Dense**: Win/loss is sparse but true. Intermediate signals are dense but potentially misleading. What's the right mix?

4. **V0 Pragmatism**: What's the simplest reward that still teaches something useful, even if it's not optimal?

## Design Principles

- **Signal over noise**: A noisy reward that correlates with good play beats a clean reward that doesn't
- **Iteration speed**: V0 should be simple enough to implement in days, not weeks
- **Upgradability**: V0 design shouldn't paint us into a corner for V1+
