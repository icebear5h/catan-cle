# Catan Learning - Training Pipeline Design

> This document describes training experiments around the active runtime. The
> runtime source of truth is `CatanSandbox` plus `GameEngine`; historical fixed
> action spaces and free-form action parsers are not part of the architecture.

## Core Architecture

```
+-------------------------------------------------------------------------+
|                         TRAINING PIPELINE                                |
+-------------------------------------------------------------------------+
|                                                                          |
|  +-------------+     +------------------+     +-------------------+      |
|  | Colonist    |---->| Reward Model     |---->| Self-Play Env     |      |
|  | Replays     |     | (Multi-Signal)   |     | (CatanSandbox)    |      |
|  | ~8.5K games |     |                  |     |                   |      |
|  +-------------+     +------------------+     +---------+---------+      |
|                              |                          |                |
|                              | scores actions           | generates      |
|                              v                          v                |
|                       +--------------------------------------+           |
|                       |      High-Signal Decision Buffer     |           |
|                       |  (state, action, reward, metadata)   |           |
|                       +------------------+-------------------+           |
|                                          |                               |
|                                          | filtered by quality           |
|                                          v                               |
|                       +--------------------------------------+           |
|                       |         LoRA Fine-Tuning             |           |
|                       |    (Modal GPU - A100/H100)           |           |
|                       +------------------+-------------------+           |
|                                          |                               |
|                                          | updated weights               |
|                                          v                               |
|                       +--------------------------------------+           |
|                       |      Policy (LLM + LoRA adapter)     |<----------+
|                       +--------------------------------------+   loop    |
|                                                                          |
+--------------------------------------------------------------------------+
```

---

## Engine Capabilities

### What the Engine Provides

The Catan engine (`cle/game_engine/`) is a complete implementation of base game rules:

| Feature | Status | Notes |
|---------|--------|-------|
| Board generation | Done | Random hex placement, ports, number tokens |
| Resource production | Done | Dice rolls, settlement/city collection |
| Building | Done | Roads, settlements, cities with distance rule |
| Development cards | Done | Knight, VP, Road Building, Year of Plenty, Monopoly |
| Robber | Done | 7-roll discard, placement, stealing |
| Longest road / Largest army | Done | Automatic tracking and VP |
| Maritime trading | Done | 4:1 bank, 3:1/2:1 ports |
| Player-to-player trading | Done | Full async state machine (see below) |
| Win detection | Done | First to 10 VP |

### Key Simplification: Engine Handles Legality

The engine exposes only valid moves via `game.state.playable_actions`. The policy never sees illegal moves and cannot make them:

- **No legality learning** - The model doesn't waste capacity learning rules
- **Pure strategy problem** - All training signal goes toward move quality, not move validity
- **Simpler action space** - Variable-size list of legal actions, not fixed action space with masking

### The Trading Problem

Player-to-player trading is the hardest part of Catan to model. Here's why:

**Real Catan trading is:**
- **Async/simultaneous** - All players see offers at once, respond in real-time
- **Social** - Timing, tone, and table talk influence acceptance
- **Multi-party** - Counteroffers, bidding wars, "I'll do it for one less"
- **Implicit** - "Anyone have wheat?" isn't a formal offer but drives trades

**What the engine supports:**
```
Player A: OFFER_TRADE (2 wheat for 1 ore)
     |
     v
Player B: ACCEPT_TRADE or REJECT_TRADE  (sequential, not simultaneous)
Player C: ACCEPT_TRADE or REJECT_TRADE
Player D: ACCEPT_TRADE or REJECT_TRADE
     |
     v
Player A: CONFIRM_TRADE (pick acceptee) or cancel
```

**Why this is lossy for learning:**
1. **Colonist replays are async** - We see the final trade, not the negotiation
2. **No counteroffer data** - "I'll do 2:2 instead" isn't captured
3. **Response context lost** - Why did Player B accept? We don't know their hand
4. **Timing signals gone** - Quick accept = desperate, slow = leverage

**V0 Strategy:** Focus on maritime trades (deterministic) and treat player trades as outcomes in training data. Full trade modeling is V2+.

---

## Phase 1: Reward Model

### Multi-Signal Reward Design

The reward model combines several signals to score any (state, action) pair:

| Signal | Source | Weight | Notes |
|--------|--------|--------|-------|
| **Win Correlation** | Colonist replays | High | Actions that lead to wins score higher |
| **Expert Matching** | Top player actions | Medium | Did experts make this choice in similar states? |
| **VP Delta** | Game state | Medium | Direct VP gains (settlements, cities, dev cards) |
| **Resource Efficiency** | Calculated | Low | Resource conversion rate, waste minimization |
| **Positional Value** | Board analysis | Low | Settlement spots, port access, robber blocking |

### Training Data Structure
```python
@dataclass
class RewardSample:
    game_id: str
    turn_number: int
    state_embedding: List[float]  # or text representation
    action_taken: str
    action_alternatives: List[str]  # other legal actions
    outcome: float  # 1.0 win, 0.0 loss
    vp_at_action: int
    final_vp: int
    expert_action: bool  # from Colonist replay
```

### Architecture Options

**Option A: Bradley-Terry Pairwise**
- Compare (state, action_A) vs (state, action_B)
- Learn: P(A > B | state)
- Pro: Works well for preference learning
- Con: Need to generate comparison pairs

**Option B: Direct Score Regression**
- (state, action) -> reward in [0, 1]
- Trained on outcome-weighted actions
- Pro: Simpler, faster inference
- Con: Harder to calibrate

**Recommendation: Start with Option B, add pairwise later for refinement**

---

## Phase 2: Self-Learning Environment

### Environment Setup (Already Exists)
- Event-driven `CatanSandbox` composition root in `cle/sandbox/`
- Complete privacy-projected context plus an exact ordered legal-action menu
- Four-player standard games, deterministic barriers, and multi-game pooling

### Self-Play Configuration
```python
@dataclass
class SelfPlayConfig:
    num_parallel_games: int = 32
    players_per_game: int = 4

    # Population mixing
    policy_pool_size: int = 4  # Keep last N checkpoints
    latest_policy_prob: float = 0.7  # 70% play latest, 30% play older

    # Exploration
    temperature: float = 1.0  # Action sampling temp
    epsilon_greedy: float = 0.05  # Random action prob

    # Data collection
    min_reward_threshold: float = 0.6  # Only keep high-signal decisions
    buffer_size: int = 100_000
```

### High-Signal Decision Filtering

Not all decisions matter equally. Filter for:

1. **Critical moments**
   - Initial settlement placement (huge impact)
   - Robber placement when multiple options
   - Trade decisions (accept/reject/counter)
   - Development card plays (timing matters)

2. **Divergence from baseline**
   - Actions that differ from random baseline
   - Actions that differ from previous policy version
   - Unexpected moves that led to wins

3. **Reward model confidence**
   - High reward score (> 0.7)
   - Low uncertainty (if using ensemble)

---

## Phase 3: LoRA Fine-Tuning

### Training Approach

**Stage 1: Behavioral Cloning (SFT)**
- Train on Colonist expert replays
- Input: state text -> Output: reasoning + action
- Get a reasonable baseline policy
- V0: ~1K games x ~50 turns = ~50K examples
- V1+: Scale to full 8.5K (~425K examples)

**Stage 2: Reward-Weighted SFT**
- Weight examples by reward model score
- Upweight winning game actions
- Downweight losing game actions
- Simple and stable

**Stage 3: Online RL (Optional)**
- PPO or DPO with reward model
- More complex, potentially better ceiling
- Requires stable reward model first

### Base Model Considerations

| Model | Params | Pros | Cons |
|-------|--------|------|------|
| Llama 3.1 8B | 8B | Well-tested, good LoRA support | Generic |
| Qwen 2.5 7B | 7B | Strong reasoning | Less community tooling |
| Mistral 7B | 7B | Fast, efficient | Weaker at complex reasoning |
| Llama 3.3 70B | 70B | Best quality | Expensive, slow |

**Recommendation: Start with Llama 3.1 8B, upgrade if ceiling is hit**

### LoRA Config
```python
lora_config = LoraConfig(
    r=16,  # rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

---

## Phase 4: Modal Compute Setup

### Infrastructure
```python
# modal_training.py
import modal

app = modal.App("catan-learning")

# GPU image with training deps
training_image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "peft", "trl", "wandb")
)

@app.function(
    gpu="A100",  # or "H100" for faster
    timeout=3600 * 4,  # 4 hour max
    image=training_image
)
def train_lora(config: dict, data_path: str):
    # Training logic here
    pass

@app.function(gpu="A100", image=training_image)
def run_self_play(policy_path: str, num_games: int):
    # Self-play logic here
    pass
```

### Cost Estimates
- A100 on Modal: ~$2.78/hr
- Training run (8B model, 4 epochs): ~2-4 hours = ~$10
- Self-play batch (1000 games): ~30 min = ~$1.50

---

## Implementation Roadmap

### V0: Minimal Viable Loop (Do This First)

The goal is to get the full loop running with the simplest possible version of each component.

```
+-----------------------------------------------------------------+
|  V0 MINIMAL LOOP                                                |
|                                                                 |
|  Colonist Replays --> SFT Dataset --> LoRA Training             |
|        |                   |              |                     |
|        |                   |              v                     |
|        |                   |         Policy v1                  |
|        |                   |              |                     |
|        v                   |              v                     |
|  Win/Loss Labels ----------+----> Self-Play (4x same policy)    |
|  (simple reward)           |              |                     |
|                            |              v                     |
|                            +------ New (state, action) pairs    |
|                                           |                     |
|                                           v                     |
|                                    LoRA Training v2             |
|                                           |                     |
|                                        LOOP                     |
+-----------------------------------------------------------------+
```

**V0 Reward Signal (Dense):**

Pure win/loss is too sparse (~70 turns). V0 uses simple dense rewards:

| Signal | Value | Trigger |
|--------|-------|---------|
| VP gain | +1.0 | Settlement, city, longest road, largest army, VP dev card |
| Production increase | +0.1 | New settlement/city adds production pips |
| Win | +5.0 | Game end |
| Loss | -1.0 | Game end |

This gives ~10-15 reward signals per game instead of 1.

**V0 Components:**

| Component | V0 Implementation | Upgrade Later |
|-----------|-------------------|---------------|
| Reward | Dense (VP + production) + win/loss | Multi-signal model |
| Action format | Reasoning + JSON | Reasoning quality scoring |
| Reasoning source | LLM-generated from expert replays | Self-generated during play |
| Self-play | 4x same policy | Population mixing |
| Decision filter | None (keep all) | High-signal only |
| Base model | Llama 3.1 8B | Larger if needed |

**V0 Learning Phases (Isolated for Debugging):**

Each phase has independent metrics so regressions can be isolated:

| Phase | Method | Success Metric | Failure = |
|-------|--------|----------------|-----------|
| 1. Imitation | SFT on expert actions | Action accuracy vs held-out replays > 40% | Data pipeline or model issue |
| 2. Dense Reward | Weight SFT by reward | Win rate vs random > 60% | Reward signal broken |
| 3. Self-Play | Generate + retrain | Win rate vs Phase 2 checkpoint > 55% | Distribution shift or exploration issue |

If Phase 3 regresses, you know it's not the imitation or reward - it's the self-play loop.

**V0 Steps:**

1. **SFT Dataset from Replays** (~2-3 days)

   **Output Format (Reasoning + Structured Action):**
   ```
   Input: <observation text>
   Output: <reasoning>
   I have wheat/brick/sheep but no ore. The robber is blocking my ore hex.
   Node 42 is on ore (6) + wheat (8) - excellent numbers. Building here
   diversifies my production and sets up for cities. Node 38 has port
   access but worse numbers. Production > flexibility early game.
   </reasoning>
   <action>{"type": "BUILD_SETTLEMENT", "node_id": 42}</action>
   ```

   **Steps:**
   - [ ] Export versioned, perspective-safe pre-action decision packets from
     the authoritative verified replay executor
   - [ ] Prove each expert action belongs to the recorded legal-action menu and
     that no hidden or future state enters the packet
   - [ ] Build reasoning generation pipeline:
     - Input: observation + expert action + game context (VP, turn, outcome)
     - Prompt strong LLM (Claude/GPT-4) to explain WHY this action is good
     - Include strategic concepts: production, position, tempo, risk
   - [ ] V0: Process ~1K games x ~50 turns = ~50K examples
   - [ ] Cost estimate: ~$5-10 for reasoning generation
   - [ ] Include win/loss label for weighting

   **Reasoning Generation Prompt (sketch):**
   ```
   You are analyzing an expert Catan player's decision.

   Game State:
   {observation}

   Expert Action: {action}
   Game Outcome: {win/loss}

   Explain the strategic reasoning behind this action in 2-4 sentences.
   Consider: resource production, board position, development timing,
   opponent threats, trade leverage, victory path.
   ```

2. **Decision Contract** (complete)
   - [x] Present every action as an exact entry in an ordered legal menu
   - [x] Parse a zero-based action index plus typed parameters where required
   - [x] Retry malformed or stale decisions within a configured bound
   - [x] Preserve accepted player context without committing rejected attempts

3. **LoRA Training Script** (~1 day)
   - [ ] Modal function for SFT training
   - [ ] Llama 3.1 8B + LoRA (r=16)
   - [ ] Win-weighted loss (upweight winning game actions)
   - [ ] Output: LoRA adapter checkpoint

4. **Self-Play Runner** (~2 days)
   - [ ] Load policy (base + LoRA adapter)
   - [ ] Run N games with 4 copies of policy
   - [ ] Collect all (observation, action, game_outcome) triples
   - [ ] Save to JSONL for next training round

5. **Evaluation** (~0.5 day)
   - [ ] Win rate vs SimplePlayer (first-action baseline)
   - [ ] Win rate vs WeightedRandomPlayer (rule-based)
   - [ ] Action diversity metrics

6. **Loop It**
   - [ ] Combine self-play data with Colonist data
   - [ ] Re-train LoRA
   - [ ] Measure improvement
   - [ ] Repeat

---

### V1: Add Reward Model (After V0 Works)

- [ ] Train BERT-based reward model on Colonist replays
- [ ] Score = P(win | state, action)
- [ ] Add strategic signals (VP delta, resource efficiency)
- [ ] Use reward model to filter self-play decisions

### V2: Scale and Refine

- [ ] Population-based training (pool of policies)
- [ ] High-signal decision filtering
- [ ] Curriculum learning (easy -> hard opponents)
- [ ] Tune reward model weights

---

## Key Design Decisions

### Action Format: Reasoning + Structured JSON

```
<reasoning>
Natural language strategic thinking. This becomes training signal
for the reasoning model. Can evaluate reasoning quality separately
from action quality.
</reasoning>
<action>{"type": "BUILD_SETTLEMENT", "node_id": 42}</action>
```

**Why this format:**
- Reasoning improves decision quality (CoT effect)
- Reasoning traces are interpretable (debug, analyze)
- Structured JSON ensures reliable parsing
- Can train reward model on reasoning quality
- Enables future distillation of good reasoning patterns

### Reward Model: Small Separate Model (BERT-based)

- Fast inference for scoring during self-play
- Train on: (state, action) -> P(win)
- Later add: reasoning quality scoring
- Keep it separate from policy for clean iteration

---

### Trading: Simplified for V0

**Problem**: Colonist replays have full async trading (offers, counteroffers, multi-party). Full trading is not feasible to model in V0.

**Solution**:
- **SFT data**: Include trades but simplify representation to outcomes
  - Instead of modeling the negotiation, represent as: "Traded 2 wheat for 1 ore with Player 2"
  - Reasoning can explain *why* the trade was good strategically
- **Self-play V0**: Maritime trades only (4:1, 3:1, 2:1 ports)
- **Self-play V1+**: Add simplified player trading (propose -> accept/reject)

### Data Scale: Start Small

- **V0**: ~1K games (~50K turns) for fast iteration
- **V1**: Scale to full 8.5K once loop is validated
- **Reasoning generation cost**: ~$5-10 for 1K games (very manageable)

---

## Open Questions

1. **State representation**: Current text is verbose - worth compressing?
2. **Exploration strategy**: Temperature sampling vs epsilon-greedy vs curiosity?
3. **Reasoning length**: How verbose should reasoning be? Token budget?

---

## Files to Create/Modify

### V0 (In Order)

| File | Purpose |
|------|---------|
| Authoritative replay decision exporter | Build versioned pre-action packets with legal menus |
| Versioned reasoning-data builder | Generate evidence-linked reasoning without future leakage |
| `cle/sandbox/catan/` | Validate and apply exact indexed decisions |
| `modal_app/__init__.py` | **NEW** - Modal app setup |
| `modal_app/training.py` | **NEW** - LoRA training on Modal |
| `cle/training/self_play.py` | Self-play game runner |
| `cle/eval/evaluate.py` | **NEW** - Evaluation against baselines |

### V1+ (After Loop Works)

| File | Purpose |
|------|---------|
| `cle/training/reward_model.py` | Multi-signal reward model |
| `cle/training/decision_buffer.py` | High-signal decision storage |
| `modal_app/self_play.py` | Modal-scaled self-play |
