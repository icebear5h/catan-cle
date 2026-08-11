# Phase 2: Continual Learning

Improve beyond human baseline through self-play and RL.

## Goal

Take the bootstrapped model and:
- Discover novel strategies beyond human play
- Achieve superhuman performance
- Generalize to diverse game scenarios
- Maintain diverse strategy population

## Process

### 1. Self-Play Data Generation (`self_play/`)

**Infrastructure:**
- Run 1000s of games with bootstrapped agents
- Collect full trajectories with outcomes
- Parallel execution on Modal

**Components:**
- `game_runner.py` - Orchestrate parallel games
- `trajectory_collector.py` - Store game data with rewards

**Output:** Self-play trajectories in `data/trajectories/`

### 2. RL Training (`rl/`)

**Algorithm:** PPO (Proximal Policy Optimization)

**Components:**
- `ppo_trainer.py` - Main RL training loop
- `reward_shaping.py` - Reward function design

**Process:**
- Train on self-play outcomes
- Maximize win rate
- Maintain reference policy
- Multi-agent credit assignment

**Output:** Improved model checkpoints in `data/checkpoints/`

### 3. Population Management (`population/`)

**Strategy:**
- Keep diverse set of strong models
- Prevent strategy collapse
- League training against historical best

**Components:**
- `checkpoint_manager.py` - Version control for models
- `elo_tracker.py` - Track relative strength

**Output:** Ranked population of models

### 4. Iteration Loop

```
1. Generate data (self-play with current best)
2. Train new model (PPO on trajectories)
3. Evaluate (vs population)
4. Update population (keep if strong)
5. Repeat
```

## Usage

```bash
# 1. Generate self-play data
python self_play/game_runner.py \
  --model checkpoints/bootstrap.pt \
  --num-games 1000 \
  --parallel 100

# 2. Train with RL
python rl/ppo_trainer.py \
  --data data/trajectories/batch_001 \
  --epochs 4

# 3. Evaluate
python population/elo_tracker.py --tournament

# 4. Iterate
./run_iteration.sh
```

## Metrics

**Target after continual learning:**
- Win rate >70% vs bootstrapped model
- ELO rating >1600 (estimated competitive human)
- Strategy diversity score >0.7
- Discovers 3+ novel strategic patterns

## Budget

- Self-play generation: ~$20-30 per 1000 games
- PPO training: ~$25-35 per iteration
- 3-5 iterations: ~$75-100
- **Total: ~$95-140**

## Key Techniques

**Reward Shaping:**
- Sparse reward (win/loss) + dense signals (VP gained, buildings)
- Exploration bonuses
- Opponent modeling rewards

**Population Diversity:**
- Keep checkpoints from different training stages
- Force diverse strategies via reward variations
- League play prevents meta collapse

**Credit Assignment:**
- Multi-agent RL is hard (4 players)
- Use counterfactual baselines
- Track individual contribution to wins

## Advanced: Curriculum Learning

As continual learning progresses, introduce:
1. Harder opponents (historical best models)
2. Adversarial scenarios (resource scarcity)
3. Board variations (different setups)

This creates a robust, generalizable agent.
