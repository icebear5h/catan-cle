#!/usr/bin/env bash
# Continued Pretraining: GLM-4.1V-9B-Thinking on Catan Knowledge
#
# Prerequisites:
#   pip install -e ".[pretraining]"
#   export YOUTUBE_API_KEY=your_key  (for YouTube transcript fetching)
#   wandb login  (optional, for training logs)
#
# Usage:
#   ./scripts/run_cpt.sh                    # Full pipeline
#   ./scripts/run_cpt.sh --skip-corpus      # Skip corpus building, just train
#   ./scripts/run_cpt.sh --skip-training    # Only build corpus, don't train

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CORPUS_OUTPUT="$PROJECT_ROOT/artifacts/generated/pretraining/legacy_corpus/catan_corpus.jsonl"
TRAINING_CONFIG="$PROJECT_ROOT/configs/training_configs/catan_cpt.yaml"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/catan-cpt-lora"

SKIP_CORPUS=false
SKIP_TRAINING=false

for arg in "$@"; do
    case $arg in
        --skip-corpus)  SKIP_CORPUS=true ;;
        --skip-training) SKIP_TRAINING=true ;;
        *)              echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "==================================="
echo " Catan Continued Pretraining (CPT)"
echo "==================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check for API keys
if [ -z "${YOUTUBE_API_KEY:-}" ]; then
    echo "WARNING: YOUTUBE_API_KEY not set. YouTube channel fetching will be limited."
    echo "  Set via: export YOUTUBE_API_KEY=your_key"
    echo ""
fi

# Step 1: Build corpus
if [ "$SKIP_CORPUS" = false ]; then
    echo "[1/2] Building Catan knowledge corpus..."
    cd "$PROJECT_ROOT"
    python -m data_pipeline.training.pretraining.build_corpus --output "$CORPUS_OUTPUT"

    if [ ! -f "$CORPUS_OUTPUT" ]; then
        echo "ERROR: Corpus file not created at $CORPUS_OUTPUT"
        exit 1
    fi

    LINES=$(wc -l < "$CORPUS_OUTPUT")
    SIZE=$(du -h "$CORPUS_OUTPUT" | cut -f1)
    echo "Corpus: $LINES documents, $SIZE"
    echo ""
else
    echo "[1/2] Skipping corpus build (--skip-corpus)"
    echo ""
fi

# Step 2: Run training
if [ "$SKIP_TRAINING" = false ]; then
    echo "[2/2] Starting LLaMA-Factory training..."

    if [ ! -f "$CORPUS_OUTPUT" ]; then
        echo "ERROR: No corpus found at $CORPUS_OUTPUT"
        echo "Run without --skip-corpus first."
        exit 1
    fi

    # Use LLaMA-Factory CLI
    cd "$PROJECT_ROOT"
    llamafactory-cli train "$TRAINING_CONFIG"

    echo ""
    echo "Training complete!"
    echo "LoRA adapter saved to: $CHECKPOINT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Merge LoRA: llamafactory-cli export configs/training_configs/catan_cpt.yaml"
    echo "  2. Upload to HuggingFace: huggingface-cli upload <repo> $CHECKPOINT_DIR"
    echo "  3. Update MODELS dict in playground/openrouter_client.py"
else
    echo "[2/2] Skipping training (--skip-training)"
fi

echo ""
echo "Done."
