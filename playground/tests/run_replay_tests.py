#!/usr/bin/env python3
"""
Run unit tests for each extracted replay action.

Tests each action in isolation to identify which action types are broken.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def colonist_cards_to_freqdeck(card_list):
    """Convert Colonist card IDs to freqdeck format."""
    # Colonist card enums: 1=WOOD, 2=BRICK, 3=SHEEP, 4=WHEAT, 5=ORE
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

def run_single_test(test_case, verbose=False):
    """
    Run a single test case.

    Returns: (passed: bool, result_msg: str)
    """
    index = test_case['index']
    action_type = test_case['action_type']
    before = test_case['before_resources']
    after = test_case['after_resources']

    # For now, just check if resources changed correctly
    # We'll implement actual action execution later

    # Calculate expected deltas
    deltas = {}
    for player_id in after.keys():
        before_cards = before.get(str(player_id), [])
        after_cards = after.get(str(player_id), [])

        before_freq = colonist_cards_to_freqdeck(before_cards)
        after_freq = colonist_cards_to_freqdeck(after_cards)

        delta = [after_freq[i] - before_freq[i] for i in range(5)]

        if any(d != 0 for d in delta):
            deltas[player_id] = {
                'before': before_freq,
                'after': after_freq,
                'delta': delta
            }

    if verbose and deltas:
        print(f"\n  Step {index} ({action_type}):")
        res_names = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
        for player_id, d in deltas.items():
            delta_str = ", ".join(f"{res_names[i]}:{d['delta'][i]:+d}" for i in range(5) if d['delta'][i] != 0)
            if delta_str:
                print(f"    Player {player_id}: {delta_str}")

    # For now, just mark all as "pending" since we haven't implemented execution yet
    return ('pending', deltas)

def main():
    test_file = Path(__file__).parent / "replay_test_cases.json"

    if not test_file.exists():
        print(f"Error: {test_file} not found. Run extract_replay_tests.py first.")
        sys.exit(1)

    with open(test_file) as f:
        test_cases = json.load(f)

    print(f"Running {len(test_cases)} test cases...")
    print()

    # Group results by action type
    results_by_type = defaultdict(lambda: {'pending': 0, 'pass': 0, 'fail': 0})
    failed_tests = []

    for test_case in test_cases:
        action_type = test_case['action_type']
        status, deltas = run_single_test(test_case, verbose=False)

        results_by_type[action_type][status] += 1

        if status == 'fail':
            failed_tests.append((test_case, deltas))

    # Print summary by action type
    print("=== Test Results by Action Type ===")
    print(f"{'Action Type':<30} {'Pending':<10} {'Pass':<10} {'Fail':<10}")
    print("-" * 70)

    for action_type in sorted(results_by_type.keys()):
        stats = results_by_type[action_type]
        print(f"{action_type:<30} {stats['pending']:<10} {stats['pass']:<10} {stats['fail']:<10}")

    print()
    print(f"Total: {len(test_cases)} tests")
    print(f"  Pending: {sum(s['pending'] for s in results_by_type.values())}")
    print(f"  Pass: {sum(s['pass'] for s in results_by_type.values())}")
    print(f"  Fail: {sum(s['fail'] for s in results_by_type.values())}")

    print()
    print("Note: Tests are 'pending' because we haven't implemented action execution yet.")
    print("Next step: Implement actual game state execution for each action type.")

if __name__ == "__main__":
    main()
