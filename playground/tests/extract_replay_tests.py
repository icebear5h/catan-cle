#!/usr/bin/env python3
"""
Extract test cases from Colonist replay for unit testing.

For each action in the replay, captures:
- State before action (resources)
- The action itself
- Expected state after action

This allows testing each action in isolation.
"""

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from cle.replay.colonist.event_parser import parse_colonist_events_to_actions


def extract_test_cases(replay_file: Path) -> list[dict[str, object]]:
    """Extract all actions with before/after states."""
    with open(replay_file) as f:
        raw_data = json.load(f)

    events = raw_data["data"]["eventHistory"]["events"]
    initial_state = raw_data["data"].get("eventHistory", {}).get("initialState")
    tile_hex_states = initial_state.get("mapState", {}).get("tileHexStates", {}) if initial_state else {}

    # Parse all actions
    parsed_actions = parse_colonist_events_to_actions(events, tile_hex_states)

    print(f"Extracted {len(parsed_actions)} actions from replay")

    # Track resources through the replay to get "before" state
    # Start with empty resources (setup happens first)
    current_resources: Mapping[str, object] = {}

    test_cases: list[dict[str, object]] = []

    for i, action in enumerate(parsed_actions):
        action_type = action.get('type')

        # Get expected resources AFTER this action (already in parsed data)
        after_resources = action.get('expected_resources', {})

        # Create test case
        test_case: dict[str, object] = {
            'index': i,
            'action_type': action_type,
            'action_data': action,
            'before_resources': dict(current_resources),  # Copy current state
            'after_resources': after_resources,
        }

        test_cases.append(test_case)

        # Update current_resources to after_resources for next iteration
        current_resources = cast(Mapping[str, object], after_resources)

    return test_cases

def group_by_action_type(
    test_cases: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    """Group test cases by action type."""
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for tc in test_cases:
        action_type = str(tc['action_type'])
        if action_type not in grouped:
            grouped[action_type] = []
        grouped[action_type].append(tc)
    return grouped

def main() -> None:
    replay_file = Path(__file__).parents[2] / "data_pipeline/bootstrapping/data/raw_replays/194335024.json"

    print(f"Extracting test cases from {replay_file}")
    test_cases = extract_test_cases(replay_file)

    # Group by action type to see coverage
    grouped = group_by_action_type(test_cases)

    print("\n=== Action Type Coverage ===")
    for action_type, cases in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"{action_type:30s}: {len(cases):3d} test cases")

    # Save test cases to file
    output_file = Path(__file__).parent / "replay_test_cases.json"
    with open(output_file, 'w') as f:
        json.dump(test_cases, f, indent=2)

    print(f"\n✅ Saved {len(test_cases)} test cases to {output_file}")

if __name__ == "__main__":
    main()
