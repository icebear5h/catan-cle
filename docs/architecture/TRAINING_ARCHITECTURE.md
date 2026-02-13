# Training Architecture: Two-Phase Approach

## Overview

Training pipeline split into two distinct phases:
1. **Bootstrapping** - Learn from human experts (imitation learning)
2. **Continual Learning** - Improve through self-play (RL)

## Phase 1: Bootstrapping (Imitation Learning)

**Goal:** Get model to baseline human competency by learning from expert gameplay and strategy content.

**Budget:** ~$100-150 (mostly Claude API for reasoning extraction)

### Step 1: Data Collection (Week 1-2)

**Sources:**
- YouTube strategy videos with commentary
- Blog posts explaining tactics
- Reddit strategy discussions
- Tournament analysis write-ups

**What we scrape:**
- Raw transcripts/text
- Game scenarios described
- Expert reasoning ("I do X because Y")
- Strategic principles

**Output:** ~10k-50k expert reasoning snippets

### Step 2: Reasoning Extraction (Week 2-3)

**Process:**
- Feed scraped content to Claude
- Claude extracts structured reasoning:
  - Game state (if described)
  - Strategic principle being applied
  - Tactical reasoning
  - Action taken
  - Outcome

**Output:** Structured training dataset in format:
```json
{
  "observation": "Your settlements are on 6-wheat, 8-ore, 5-sheep...",
  "strategic_context": "Need ore for cities, wheat production is strong",
  "reasoning": "Building city on wheat/ore gives 2x production...",
  "action": "BUILD_CITY at (3,4)",
  "principle": "Cities are more VP-efficient than settlements"
}
```

### Step 3: Initial Fine-tuning (Week 3-4)

**Model:** Llama 3.3 70B with LoRA

**Training:**
- Supervised fine-tuning on expert reasoning
- Input: Game state + strategic context
- Output: Reasoning + action
- 8-12 hours on Modal A100 (~$10-15)

**Output:** "Bootstrapped model" that thinks like a competitive Catan player

### Step 4: Validation

**Tests:**
- Play 100 games against random baseline
- Play 100 games against heuristic baseline
- Manual review of reasoning quality

**Success criteria:**
- Win rate >80% vs random
- Win rate >60% vs heuristic
- Reasoning mentions relevant strategic principles

## Phase 2: Continual Learning (RL)

**Goal:** Improve beyond human baseline through self-play and discover novel strategies.

**Budget:** ~$100-150 (Modal GPU for self-play + RL training)

### Step 1: Self-Play Data Generation (Week 5-6)

**Process:**
- Run 1000+ games with 4 bootstrapped agents
- Collect full game trajectories
- Store (state, action, reasoning, outcome)

**Infrastructure:**
- Modal serverless functions for parallel games
- Can run 100 games concurrently
- ~$20-30 for 1000 games

**Output:** Self-play dataset with reward signals

### Step 2: RL Training (Week 6-7)

**Algorithm:** PPO (Proximal Policy Optimization) with LoRA

**Training:**
- Use self-play outcomes as reward signal
- Update policy to maximize win rate
- Keep reference policy to prevent collapse
- Multi-agent credit assignment

**Compute:**
- 20-30 hours on A100 (~$25-35)
- Iterative: train, evaluate, collect more data

**Output:** Improved model checkpoint

### Step 3: Population Management

**Strategy:**
- Keep best checkpoints from each iteration
- Test against diverse opponents (prevents overfitting)
- League training: new models vs historical best

**Prevents:**
- Strategy collapse (everyone plays the same)
- Overfitting to current meta
- Forgetting early strategies

### Step 4: Iterative Improvement

**Loop:**
1. Generate data with current best model
2. Train new model on data
3. Evaluate vs population
4. Keep if better, discard if worse
5. Repeat

**Budget allocation:**
- Reserve $50-75 for iterations
- ~3-5 training cycles

## Directory Structure

```
data_pipeline/
├── bootstrapping/
│   ├── scrapers/
│   │   ├── youtube_scraper.py
│   │   ├── blog_scraper.py
│   │   ├── reddit_scraper.py
│   │   └── __init__.py
│   ├── extractors/
│   │   ├── reasoning_extractor.py  # Uses Claude
│   │   ├── game_state_parser.py
│   │   └── __init__.py
│   ├── data/
│   │   ├── raw/              # Raw scraped content
│   │   ├── processed/        # Extracted reasoning
│   │   └── training/         # Final training format
│   └── fine_tune_bootstrap.py
│
├── continual_learning/
│   ├── self_play/
│   │   ├── game_runner.py
│   │   ├── trajectory_collector.py
│   │   └── __init__.py
│   ├── rl/
│   │   ├── ppo_trainer.py
│   │   ├── reward_shaping.py
│   │   └── __init__.py
│   ├── population/
│   │   ├── checkpoint_manager.py
│   │   ├── elo_tracker.py
│   │   └── __init__.py
│   └── data/
│       ├── trajectories/     # Self-play games
│       └── checkpoints/      # Model versions
│
├── schema.py                 # Shared data structures
└── README.md
```

## Cost Breakdown

### Bootstrapping Phase
- Scraping: $0 (runs locally)
- Claude reasoning extraction: $75-100
- Initial fine-tuning: $10-15
- **Total: ~$85-115**

### Continual Learning Phase
- Self-play data generation: $20-30
- PPO training (1st iteration): $25-35
- Additional iterations (3x): $50-75
- **Total: ~$95-140**

### Grand Total: ~$180-255

Leaves $0-70 buffer for:
- Re-runs if something fails
- Hyperparameter tuning
- Extended training if needed

## Success Metrics

### After Bootstrapping
- Can play legal games
- Uses strategic reasoning
- Wins >60% vs heuristic baseline
- Reasoning quality passes human review

### After Continual Learning
- Wins >70% vs bootstrapped model
- Discovers non-obvious strategies
- ELO rating >1600 (estimated)
- Generalizes to different board setups

## Timeline

- **Weeks 1-2:** Data collection (bootstrapping)
- **Weeks 2-3:** Reasoning extraction
- **Weeks 3-4:** Initial fine-tuning + validation
- **Weeks 5-6:** Self-play infrastructure + data gen
- **Weeks 6-7:** RL training (1st iteration)
- **Week 8:** Evaluation + iteration planning

**Total: ~8 weeks to trained model**

## Key Decisions

**Model choice:** Llama 3.3 70B
- Good reasoning ability
- Can run locally for inference
- LoRA fine-tuning is affordable

**Bootstrapping first:**
- Cheaper than pure RL
- Faster convergence
- Learns human strategic principles
- Provides better starting point for RL

**Continual learning second:**
- Discovers strategies beyond human play
- Self-play scales infinitely
- Can iterate over time
- Doesn't rely on external data

## Next Steps

1. Finish scraper implementations
2. Build reasoning extractor with Claude
3. Collect 10k expert reasoning examples
4. Fine-tune bootstrapped model
5. Build self-play infrastructure
6. Run RL training loop
