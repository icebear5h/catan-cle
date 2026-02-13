#!/bin/bash
# Definitive test script for replay divergence testing
# Success = all 373 steps complete without divergence

GAME_ID="194335024"
TOTAL_STEPS=373
SERVER_URL="http://localhost:5001"
RESULTS_FILE="/tmp/divergence_test_results.txt"

echo "========================================"
echo "Replay Divergence Test"
echo "Game ID: $GAME_ID"
echo "Total steps: $TOTAL_STEPS"
echo "========================================"
echo ""

# Load the replay
echo "Loading replay..."
load_result=$(curl -s -X POST "$SERVER_URL/api/load-replay" \
  -H "Content-Type: application/json" \
  -d "{\"game_id\": \"$GAME_ID\"}")

if echo "$load_result" | grep -q "error"; then
  echo "ERROR: Failed to load replay"
  echo "$load_result"
  exit 1
fi

echo "Replay loaded successfully"
echo ""

# Run until divergence
echo "Running replay until divergence is found..."
echo "(This may take a minute...)"
echo ""

divergence_result=$(curl -s -X POST "$SERVER_URL/api/replay-goto-divergence" \
  -H "Content-Type: application/json" \
  -d '{"max_steps": 9999}')

# Parse the result
status=$(echo "$divergence_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', 'unknown'))" 2>/dev/null || echo "error")
event_index=$(echo "$divergence_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('event_index', 0))" 2>/dev/null || echo "0")
steps_taken=$(echo "$divergence_result" | python3 -c "import sys, json; print(json.load(sys.stdin).get('steps_taken', 0))" 2>/dev/null || echo "0")

echo "========================================"
echo "RESULTS"
echo "========================================"
echo "Status: $status"
echo "Steps completed: $event_index / $TOTAL_STEPS"
echo "Engine steps taken: $steps_taken"
echo ""

if [ "$status" = "divergence_found" ]; then
  action_type=$(echo "$divergence_result" | python3 -c "import sys, json; d=json.load(sys.stdin).get('divergence', {}); print(d.get('action_type', 'unknown'))" 2>/dev/null || echo "unknown")

  echo "DIVERGENCE DETECTED at step $event_index"
  echo "Action type: $action_type"
  echo ""
  echo "Mismatch details:"
  echo "$divergence_result" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    divergence = data.get('divergence', {})
    for mismatch in divergence.get('mismatches', []):
        player = mismatch.get('player_idx')
        diff = mismatch.get('diff', {})
        engine = mismatch.get('engine_has', {})
        expected = mismatch.get('expected', {})
        print(f'  Player {player}:')
        print(f'    Engine has: {engine}')
        print(f'    Expected:   {expected}')
        print(f'    Diff:       {diff}')
except:
    pass
" 2>/dev/null

  echo ""
  echo "❌ TEST FAILED - Divergence at step $event_index"
  echo ""

  # Save result
  echo "$(date): FAILED at step $event_index/$TOTAL_STEPS (action: $action_type)" >> "$RESULTS_FILE"
  exit 1

elif [ "$status" = "no_divergence" ]; then
  echo "✅ TEST PASSED - All $event_index steps completed without divergence!"
  echo ""

  if [ "$event_index" -eq "$TOTAL_STEPS" ]; then
    echo "🎉 SUCCESS! Replay is fully aligned with Colonist!"
    echo "$(date): SUCCESS - All $TOTAL_STEPS steps completed" >> "$RESULTS_FILE"
    exit 0
  else
    echo "⚠️  Warning: Only $event_index/$TOTAL_STEPS steps were tested"
    echo "$(date): PARTIAL - $event_index/$TOTAL_STEPS steps completed" >> "$RESULTS_FILE"
    exit 0
  fi

else
  echo "❌ ERROR: Unexpected status: $status"
  echo "$(date): ERROR - unexpected status" >> "$RESULTS_FILE"
  exit 1
fi
