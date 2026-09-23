"""State-independent admission for bounded deterministic semantic plans."""

from __future__ import annotations

import re
from copy import deepcopy

from cle.game_engine.trading import RESOURCE_NAMES
from cle.players.data import ActionCall, JsonValue

MAX_BATCH_ACTIONS = 4
BATCH_TOOLS = frozenset({
    "build_settlement", "build_road", "upgrade_city", "maritime_trade", "end_turn",
})


def validate_batch_actions(
    actions: JsonValue | tuple[ActionCall, ...], *, stored: bool = False,
) -> tuple[ActionCall, ...]:
    """Check the entire syntax, never future legality against an initial menu.

    The detached tuple is private plan data; arguments retain their semantic names
    so later roads, settlements and port rates bind to the updated engine state.
    """
    expected = tuple if stored else list
    if not isinstance(actions, (list, tuple)) or not isinstance(actions, expected) or not 1 <= len(actions) <= MAX_BATCH_ACTIONS:
        raise ValueError(f"actions must contain 1–{MAX_BATCH_ACTIONS} deterministic calls")
    admitted: list[ActionCall] = []
    for index, call in enumerate(actions):
        if not isinstance(call, dict):
            raise ValueError("Each batch action requires only tool and arguments")
        admitted.append(call)
        if set(call) == {"tool"}:
            call = {**call, "arguments": {}}
        if set(call) != {"tool", "arguments"}:
            raise ValueError("Each batch action requires only tool and arguments")
        tool, args = call["tool"], call["arguments"]
        if not isinstance(tool, str) or tool not in BATCH_TOOLS:
            raise ValueError("Batches allow only build_settlement, build_road, upgrade_city, maritime_trade, end_turn")
        if not isinstance(args, dict):
            raise ValueError("Batch arguments must be an object")
        if tool == "end_turn":
            if args or index != len(actions) - 1:
                raise ValueError("end_turn takes no arguments and must be last")
        elif tool == "maritime_trade":
            if set(args) != {"give", "receive"}:
                raise ValueError("maritime_trade requires only give and receive")
            bundles = []
            for field in ("give", "receive"):
                bundle = args[field]
                if not isinstance(bundle, dict) or len(bundle) != 1:
                    raise ValueError("Maritime bundles must name exactly one resource")
                resource, count = next(iter(bundle.items()))
                if not isinstance(resource, str) or resource.upper() not in RESOURCE_NAMES or type(count) is not int:
                    raise ValueError("Maritime resources require named integer counts")
                if count not in ({2, 3, 4} if field == "give" else {1}):
                    raise ValueError("Maritime trade gives 2, 3 or 4 cards and receives one")
                bundles.append(resource.upper())
            if bundles[0] == bundles[1]:
                raise ValueError("Maritime trade must receive a different resource")
        else:
            field = "edge" if tool == "build_road" else "node"
            pattern = r"<E\d{2}_\d{2}>" if field == "edge" else r"<N\d{2}>"
            token = args.get(field)
            if set(args) != {field} or not isinstance(token, str) or not re.fullmatch(pattern, token, flags=re.ASCII):
                raise ValueError(f"{tool} requires one literal {field} token")
    return deepcopy(tuple(admitted))
