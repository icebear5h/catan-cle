#!/usr/bin/env python3
"""
Test all actions in the replay and report which ones fail.

Unlike the divergence test which stops at first failure, this runs through
ALL actions and reports every failure.
"""

import requests
import json
import sys
from pathlib import Path
from collections import defaultdict

SERVER_URL = "http://localhost:5001"

def colonist_cards_to_freqdeck(card_list):
    """Convert Colonist card IDs to freqdeck format."""
    freqdeck = [0, 0, 0, 0, 0]
    for card in card_list:
        if card == 1:
            freqdeck[0] += 1  # WOOD
        elif card == 2:
            freqdeck[1] += 1  # BRICK
        elif card == 3:
            freqdeck[2] += 1  # SHEEP
        elif card == 4:
            freqdeck[3] += 1  # WHEAT
        elif card == 5:
            freqdeck[4] += 1  # ORE
    return freqdeck

def get_engine_state():
    """Get current engine state from server."""
    resp = requests.get(f"{SERVER_URL}/api/state")
    if resp.status_code == 200:
        return resp.json()
    return None

def step_replay():
    """Execute one replay step."""
    resp = requests.post(f"{SERVER_URL}/api/replay-step")
    if resp.status_code == 200:
        return resp.json()
    return None

def load_replay(game_id="194335024"):
    """Load the replay."""
    resp = requests.post(
        f"{SERVER_URL}/api/load-replay",
        headers={"Content-Type": "application/json"},
        json={"game_id": game_id}
    )
    if resp.status_code == 200:
        return resp.json()
    return None

def main():
    # Load test cases
    test_file = Path(__file__).parent / "replay_test_cases.json"
    with open(test_file) as f:
        test_cases = json.load(f)

    print(f"Testing {len(test_cases)} actions...")
    print()

    # Load replay
    print("Loading replay...")
    result = load_replay()
    if not result:
        print("ERROR: Failed to load replay")
        sys.exit(1)

    print(f"Loaded {result.get('total_events')} events")
    print()

    # Track results
    results = []
    failures_by_type = defaultdict(list)
    res_names = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]

    # Step through each action
    for i, test_case in enumerate(test_cases):
        action_type = test_case['action_type']
        expected_after = test_case['after_resources']

        # Execute step
        step_result = step_replay()
        if not step_result:
            print(f"ERROR: Failed to execute step {i}")
            break

        # Get current state
        state = get_engine_state()
        if not state:
            print(f"ERROR: Failed to get state at step {i}")
            break

        # Check resources
        passed = True
        mismatches = []

        colonist_to_engine = state.get('colonist_color_to_engine_idx', {})

        for colonist_id_str, expected_cards in expected_after.items():
            engine_idx = colonist_to_engine.get(colonist_id_str)
            if engine_idx is None:
                continue

            expected_freq = colonist_cards_to_freqdeck(expected_cards)

            # Get actual resources from engine
            players = state.get('state', {}).get('players', [])
            if engine_idx < len(players):
                actual_freq = players[engine_idx].get('resources', [0,0,0,0,0])

                if actual_freq != expected_freq:
                    passed = False
                    diff = [actual_freq[j] - expected_freq[j] for j in range(5)]
                    diff_str = ", ".join(f"{res_names[j]}:{diff[j]:+d}" for j in range(5) if diff[j] != 0)

                    mismatches.append({
                        'player': engine_idx,
                        'colonist_id': colonist_id_str,
                        'expected': expected_freq,
                        'actual': actual_freq,
                        'diff': diff_str
                    })

        result = {
            'index': i,
            'action_type': action_type,
            'passed': passed,
            'mismatches': mismatches
        }

        results.append(result)

        if not passed:
            failures_by_type[action_type].append(result)

            # Print failure immediately
            print(f"❌ Step {i:3d} {action_type:25s} FAILED")
            for m in mismatches:
                print(f"     Player {m['player']}: {m['diff']}")

        # Check if we've finished
        if step_result.get('finished'):
            print(f"\n✅ Reached end of replay at step {i+1}")
            break

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_passed = sum(1 for r in results if r['passed'])
    total_failed = sum(1 for r in results if not r['passed'])

    print(f"Total tests: {len(results)}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    print()

    if failures_by_type:
        print("Failures by Action Type:")
        for action_type in sorted(failures_by_type.keys()):
            failures = failures_by_type[action_type]
            print(f"  {action_type:30s}: {len(failures):3d} failures")

        print()
        print("First failure for each action type:")
        for action_type in sorted(failures_by_type.keys()):
            first_fail = failures_by_type[action_type][0]
            print(f"\n  {action_type} (step {first_fail['index']}):")
            for m in first_fail['mismatches']:
                print(f"    Player {m['player']}: {m['diff']}")

    return 0 if total_failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
