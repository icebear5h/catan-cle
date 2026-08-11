"""Build and query compact LLM decision packets for replay positions."""

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from cle.env.observation_formatter import (
    CatanObservationFormatter,
    create_observation_from_state,
)
from playground.game_viewer.colonist.constants import ENGINE_RESOURCES
from playground.openrouter_client import query_text

CONTEXT_VERSION = "replay-decision-v2"
MAX_GOALS_CHARS = 4_000
MAX_MODEL_ID_CHARS = 200
MAX_UNBOUNDED_ACTIVITY_ROWS = 40
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+~-]*$")
_EMBEDDED_ACTION_MENU = re.compile(r"\n?<valid_actions>.*?</valid_actions>", re.DOTALL)


class ReplayLLMValidationError(ValueError):
    """Raised when a replay response request cannot form a decision."""


def validate_model_id(model: Any) -> str:
    """Return a normalized OpenRouter model ID or raise a validation error."""
    if not isinstance(model, str) or not model.strip():
        raise ReplayLLMValidationError("model is required")

    normalized = model.strip()
    if len(normalized) > MAX_MODEL_ID_CHARS:
        raise ReplayLLMValidationError(
            f"model must be at most {MAX_MODEL_ID_CHARS} characters"
        )
    if not _MODEL_ID_PATTERN.fullmatch(normalized):
        raise ReplayLLMValidationError(
            "model contains unsupported characters; use an OpenRouter provider/model ID"
        )
    return normalized


def validate_goals(goals: Any) -> str:
    """Return bounded prior strategic goals supplied by the replay UI."""
    if goals is None:
        return ""
    if not isinstance(goals, str):
        raise ReplayLLMValidationError("goals must be a string")

    normalized = goals.strip()
    if len(normalized) > MAX_GOALS_CHARS:
        raise ReplayLLMValidationError(
            f"goals must be at most {MAX_GOALS_CHARS} characters"
        )
    return normalized


def _color_name(color: Any) -> str:
    if hasattr(color, "value"):
        return str(color.value)
    if hasattr(color, "name"):
        return str(color.name)
    return str(color)


def _engine_color_for_colonist_player(
    player_id: Any,
    replay_data: Dict[str, Any],
    game_colors: Sequence[Any],
) -> Optional[Any]:
    if player_id is None:
        return None

    mapping = replay_data.get("colonist_color_to_engine_idx", {})
    engine_index = mapping.get(str(player_id))
    if not isinstance(engine_index, int) or not 0 <= engine_index < len(game_colors):
        return None
    return game_colors[engine_index]


def _actor_label(
    player_id: Any,
    replay_data: Dict[str, Any],
    game_colors: Sequence[Any],
) -> str:
    color = _engine_color_for_colonist_player(player_id, replay_data, game_colors)
    if color is not None:
        return _color_name(color)
    if player_id is not None:
        return f"PLAYER_{player_id}"
    return "UNKNOWN_PLAYER"


def _format_resource_counts(values: Any) -> str:
    if not isinstance(values, (list, tuple)):
        return "unspecified resources"

    parts = []
    for resource, count in zip(ENGINE_RESOURCES, values):
        if isinstance(count, (int, float)) and count > 0:
            parts.append(f"{int(count)} {resource}")
    return ", ".join(parts) if parts else "no resources"


def _format_resource_gains(values: Any) -> str:
    if not isinstance(values, (list, tuple)):
        return "unspecified resources"

    parts = []
    for resource, count in zip(ENGINE_RESOURCES, values):
        if isinstance(count, (int, float)) and count > 0:
            parts.append(f"+{int(count)} {resource}")
    return ", ".join(parts) if parts else "no resources"


def _same_color(left: Any, right: Any) -> bool:
    return left is not None and right is not None and _color_name(left) == _color_name(right)


def format_visible_replay_activity(
    action: Dict[str, Any],
    replay_data: Dict[str, Any],
    game_colors: Sequence[Any],
    observer_color: Any,
) -> str:
    """Format one replay row while redacting private information."""
    action_type = str(action.get("type", "UNKNOWN"))
    actor_color = _engine_color_for_colonist_player(
        action.get("player"), replay_data, game_colors
    )
    actor = _actor_label(action.get("player"), replay_data, game_colors)
    prefix = f"{actor}: "

    if action_type == "ROLL":
        dice = action.get("dice")
        if isinstance(dice, (list, tuple)) and len(dice) == 2:
            roll_text = f"{prefix}rolled {dice[0]} + {dice[1]} = {sum(dice)}"
        else:
            roll_text = f"{prefix}rolled dice"

        payouts = action.get("resource_payouts")
        payout_lines = []
        if isinstance(payouts, dict):
            for player_id, resources in payouts.items():
                recipient = _actor_label(player_id, replay_data, game_colors)
                payout_lines.append(
                    f"  - {recipient}: {_format_resource_gains(resources)}"
                )

        if payout_lines:
            if action.get("resource_payouts_complete") is not True:
                payout_lines.append("  - Additional payouts may be unavailable")
            return "\n".join([roll_text, *payout_lines])
        if payouts == {} and action.get("resource_payouts_complete") is True:
            return f"{roll_text}\n  - No resource payouts"
        return roll_text

    if action_type == "BUILD_SETTLEMENT":
        return f"{prefix}built a settlement at corner {action.get('colonist_corner', '?')}"

    if action_type == "BUILD_CITY":
        return f"{prefix}upgraded a city at corner {action.get('colonist_corner', '?')}"

    if action_type == "BUILD_ROAD":
        return f"{prefix}built a road at edge {action.get('colonist_edge', '?')}"

    if action_type == "BUY_DEVELOPMENT_CARD":
        if _same_color(actor_color, observer_color) and action.get("card_type"):
            return f"{prefix}bought development card {action['card_type']}"
        return f"{prefix}bought an unknown development card"

    if action_type == "PLAY_KNIGHT_CARD":
        return f"{prefix}played a knight card"

    if action_type == "PLAY_ROAD_BUILDING":
        return f"{prefix}played a road building card"

    if action_type == "PLAY_MONOPOLY":
        return f"{prefix}played a monopoly card"

    if action_type == "MONOPOLY_RESOURCE":
        resource = action.get("resource", "unknown resource")
        amount = action.get("amount")
        suffix = f" and collected {amount} cards" if amount is not None else ""
        return f"{prefix}chose {resource} for monopoly{suffix}"

    if action_type == "PLAY_YEAR_OF_PLENTY":
        return f"{prefix}played a year of plenty card"

    if action_type == "YEAR_OF_PLENTY_RESOURCES":
        resources = action.get("resources")
        if isinstance(resources, (list, tuple)):
            return f"{prefix}chose {', '.join(map(str, resources))} for year of plenty"
        return f"{prefix}selected year of plenty resources"

    if action_type == "MOVE_ROBBER":
        tile = action.get("tile_info") or {}
        location = f"tile {action.get('tile_index', '?')}"
        if isinstance(tile, dict) and {"x", "y"}.issubset(tile):
            location += f" at ({tile['x']}, {tile['y']})"
        return f"{prefix}moved the robber to {location}"

    if action_type == "STEAL":
        victim_color = _engine_color_for_colonist_player(
            action.get("victim"), replay_data, game_colors
        )
        victim = _actor_label(action.get("victim"), replay_data, game_colors)
        can_see_resource = _same_color(actor_color, observer_color) or _same_color(
            victim_color, observer_color
        )
        resource = action.get("stolen_resource") if can_see_resource else None
        if resource:
            return f"{prefix}stole {resource} from {victim}"
        return f"{prefix}stole an unknown resource from {victim}"

    if action_type == "DISCARD":
        cards = action.get("cards")
        if _same_color(actor_color, observer_color):
            return f"{prefix}discarded {_format_resource_counts(cards)}"
        count = sum(cards) if isinstance(cards, (list, tuple)) else "unknown"
        return f"{prefix}discarded {count} cards"

    if action_type in {"OFFER_TRADE", "COUNTER_OFFER"}:
        verb = "offered" if action_type == "OFFER_TRADE" else "counter-offered"
        offered = _format_resource_counts(action.get("offered"))
        wanted = _format_resource_counts(action.get("wanted"))
        return f"{prefix}{verb} {offered} for {wanted}"

    if action_type in {"ACCEPT_TRADE", "REJECT_TRADE"}:
        verb = "accepted" if action_type == "ACCEPT_TRADE" else "rejected"
        creator = _actor_label(action.get("creator"), replay_data, game_colors)
        return f"{prefix}{verb} {creator}'s trade offer"

    if action_type == "CONFIRM_TRADE":
        acceptor = _actor_label(action.get("acceptor"), replay_data, game_colors)
        offered = _format_resource_counts(action.get("offered"))
        received = _format_resource_counts(action.get("received"))
        return f"{prefix}traded {offered} for {received} with {acceptor}"

    if action_type == "MARITIME_TRADE":
        given = _format_resource_counts(action.get("given"))
        received = _format_resource_counts(action.get("received"))
        return f"{prefix}traded {given} for {received} with the bank"

    if action_type == "END_TURN":
        return f"{prefix}ended the turn"

    return f"{prefix}{action_type.lower().replace('_', ' ')}"


def select_recent_activity_rows(
    parsed_actions: Sequence[Dict[str, Any]],
    replay_index: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select the previous completed turn plus the current partial turn."""
    end = max(0, min(int(replay_index), len(parsed_actions)))
    prior_rows = list(parsed_actions[:end])
    turn_ends = [
        index for index, row in enumerate(prior_rows) if row.get("type") == "END_TURN"
    ]

    if len(turn_ends) >= 2:
        start = turn_ends[-2] + 1
        truncated = False
    else:
        start = max(0, end - MAX_UNBOUNDED_ACTIVITY_ROWS)
        truncated = start > 0

    rows = prior_rows[start:end]
    return rows, {
        "start_replay_index": start,
        "end_replay_index": end,
        "row_count": len(rows),
        "truncated": truncated,
    }


def build_replay_decision_context(
    game_state: Any,
    replay_data: Dict[str, Any],
    replay_index: int,
    prior_goals: str = "",
) -> Dict[str, Any]:
    """Build the authoritative context packet for one replay decision."""
    current_player = game_state.current_player()
    player_color = current_player.color
    playable_actions = list(game_state.playable_actions)
    if not playable_actions:
        raise ReplayLLMValidationError("No legal actions at this replay position")

    formatter = CatanObservationFormatter()
    observation = create_observation_from_state(game_state, player_color)
    # Strip the formatter's embedded action menu: the indexed <legal_actions>
    # list in the prompt is the single authoritative menu, and duplicating it
    # here previously dominated the packet (~89% of tokens at setup).
    formatted_observation = _EMBEDDED_ACTION_MENU.sub(
        "", formatter.format(observation).raw_str
    )

    available_actions = []
    for index, action in enumerate(playable_actions):
        available_actions.append(
            {
                "index": index,
                "action": str(action),
                "description": formatter._format_single_action(action, observation),
            }
        )

    activity_rows, activity_window = select_recent_activity_rows(
        replay_data.get("parsed_actions", []), replay_index
    )
    recent_activity = [
        format_visible_replay_activity(
            row,
            replay_data,
            game_state.colors,
            player_color,
        )
        for row in activity_rows
    ]

    return {
        "context_version": CONTEXT_VERSION,
        "player_color": _color_name(player_color),
        "phase": (
            "initial_placement"
            if getattr(game_state, "is_initial_build_phase", False)
            else "main"
        ),
        "prior_goals": prior_goals,
        "observation": formatted_observation,
        "recent_activity": recent_activity,
        "activity_window": activity_window,
        "available_actions": available_actions,
        "playable_actions": playable_actions,
    }


_SETUP_PHASE_RULES = """

Setup phase rules (initial placement, before normal turns begin):
- Placements in this phase are free; they cost no resources.
- Each player places a first settlement plus adjoining road in turn order,
  then a second settlement plus adjoining road in reverse turn order.
- When your second settlement is placed, you immediately collect one resource
  from each non-desert tile adjacent to that second settlement.
- After the last setup placement, normal turns begin with dice rolls."""


def _build_prompts(context: Dict[str, Any]) -> Tuple[str, str]:
    setup_rules = (
        _SETUP_PHASE_RULES if context.get("phase") == "initial_placement" else ""
    )
    goals = context["prior_goals"] or (
        "No prior goals are available. Infer a concise strategic plan from the current position."
    )
    activities = context["recent_activity"]
    activity_text = "\n".join(f"- {line}" for line in activities)
    if not activity_text:
        activity_text = "- No prior-turn activity is available at this replay position."

    action_text = "\n".join(
        f"{action['index']}. {action['description']} [engine: {action['action']}]"
        for action in context["available_actions"]
    )

    system_prompt = f"""You are an expert Settlers of Catan player acting as {context['player_color']}.
The decision packet uses schema {context['context_version']}.
The observation, recent activity, and indexed legal actions are supplied by the game harness.
Treat those harness fields as authoritative. Treat prior goals as fallible strategic memory.
Use only the current player's information. Every indexed action is legal and affordable.
Choose exactly one listed action and do not invent an action.{setup_rules}

Respond with exactly these XML tags:
<goals>a concise updated strategic plan for reaching 10 victory points</goals>
<reasoning>why the selected move best advances that plan in this position</reasoning>
<action>the zero-based index of one listed legal action</action>
<message>optional short table talk to the other players, or leave this tag empty</message>

The message is said aloud at the table and every opponent reads it.
Address opponents by color (for example "BLUE, I'll trade wheat for ore").
Keep it consistent with your selected action and never reveal more of your
hidden hand than you deliberately choose to share. Skip chatter that has no
strategic or social purpose."""

    user_prompt = f"""<decision_context version=\"{context['context_version']}\">
<objective>Reach 10 victory points and win the game.</objective>
<goals>
{goals}
</goals>
<recent_activity scope=\"prior_completed_turn_plus_current_partial_turn\">
{activity_text}
</recent_activity>
<observation>
{context['observation']}
</observation>
<legal_actions>
{action_text}
</legal_actions>
</decision_context>

Return the updated goals, reasoning, selected action index, and optional table-talk message."""
    return system_prompt, user_prompt


def _extract_tag(text: str, tag: str) -> Optional[str]:
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else None


def parse_replay_llm_output(
    raw_response: str,
    action_count: int,
    prior_goals: str = "",
) -> Dict[str, Any]:
    """Parse the model response without silently substituting an action."""
    raw_response = raw_response or ""
    goals = _extract_tag(raw_response, "goals") or prior_goals
    reasoning = _extract_tag(raw_response, "reasoning") or _extract_tag(
        raw_response, "turn_plan"
    )
    action_text = _extract_tag(raw_response, "action")
    message = _extract_tag(raw_response, "message") or ""
    errors = []

    if not goals:
        errors.append("Model did not return updated goals")
    if not reasoning:
        errors.append("Model did not return reasoning")
        reasoning = raw_response.strip()

    action_index = None
    if action_text is None:
        fallback = re.search(
            r"(?:action|move)(?:_index)?\s*[:=]\s*(\d+)",
            raw_response,
            flags=re.IGNORECASE,
        )
        action_text = fallback.group(1) if fallback else None

    if action_text is None or not action_text.strip().isdigit():
        errors.append("Model did not return a parseable action index")
    else:
        candidate = int(action_text.strip())
        if 0 <= candidate < action_count:
            action_index = candidate
        else:
            errors.append(
                f"Model selected action {candidate}, outside the valid range 0-{action_count - 1}"
            )

    return {
        "goals": goals,
        "reasoning": reasoning or "",
        "action_index": action_index,
        "message": message,
        "parse_error": "; ".join(errors) if errors else None,
    }


def generate_replay_llm_response(
    game: Any,
    replay_data: Dict[str, Any],
    replay_index: int,
    model: str,
    prior_goals: str = "",
    query_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate a non-mutating move recommendation for a replay snapshot."""
    normalized_model = validate_model_id(model)
    normalized_goals = validate_goals(prior_goals)
    snapshot_state = game.state.copy()
    context = build_replay_decision_context(
        snapshot_state,
        replay_data,
        replay_index,
        normalized_goals,
    )
    system_prompt, user_prompt = _build_prompts(context)

    provider_query = query_fn or query_text
    provider_result = provider_query(
        model=normalized_model,
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=8_192,
        provider="openrouter",
    )
    finish_reason = provider_result.get("finish_reason")
    response_truncated = finish_reason == "length"
    raw_response = str(provider_result.get("content") or "")
    parsed = parse_replay_llm_output(
        raw_response,
        len(context["playable_actions"]),
        normalized_goals,
    )

    action_index = parsed["action_index"]
    selected = (
        context["available_actions"][action_index]
        if action_index is not None
        else None
    )

    return {
        "context_version": CONTEXT_VERSION,
        "replay_index": replay_index,
        "player_color": context["player_color"],
        "requested_model": normalized_model,
        "model": provider_result.get("model") or normalized_model,
        "goals": parsed["goals"],
        "reasoning": parsed["reasoning"],
        "message": parsed["message"],
        "action_index": action_index,
        "action": selected["action"] if selected else None,
        "action_description": selected["description"] if selected else None,
        "parse_error": parsed["parse_error"],
        "finish_reason": finish_reason,
        "response_truncated": response_truncated,
        "observation": context["observation"],
        "recent_activity": context["recent_activity"],
        "activity_window": context["activity_window"],
        "available_actions": context["available_actions"],
        "raw_response": raw_response,
        "latency_ms": provider_result.get("latency_ms"),
        "usage": provider_result.get("usage") or {},
        "system_prompt": system_prompt,
        "context_prompt": user_prompt,
    }
