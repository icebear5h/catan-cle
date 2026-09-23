"""Pure validation and materialization of player-authored action parameters."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

from cle.game_engine.models.enums import Action, ActionType
from cle.game_engine.models.player import Color
from cle.game_engine.trading import RESOURCE_NAMES, TradeOffer, TradeOfferStatus
from cle.players.action_batches import validate_batch_actions
from cle.players.communication_validation import (
    validate_communication_choice as validate_communication_choice,
)
from cle.players.contracts import (
    CommunicationChoice,
    CommunicationMode,
    PlayerAttempt,
    PlayerChoice,
    PlayerContext,
)
from cle.players.notes import MAX_NOTES_CHARS, validate_notes


def action_from_choice(
    context: PlayerContext,
    choice: PlayerChoice,
    *,
    max_notes_chars: int = MAX_NOTES_CHARS,
) -> Action:
    """Return detached parameters bound to the exact menu; never mutate state.

    Invalid types, values, or menu associations raise ValueError. Affordability
    and current-state legality still require the engine's strict validation.
    """
    if not isinstance(choice, PlayerChoice):
        raise ValueError("Player choice must be a PlayerChoice")
    if not isinstance(choice.batch_actions, tuple):
        raise ValueError("batch_actions must be a tuple")
    if choice.batch_actions:
        validate_batch_actions(choice.batch_actions, stored=True)
        if any(value is not None for value in (
            choice.trade_offer, choice.confirm_if_accepted_by,
            choice.knight_destination, choice.discard_cards,
        )):
            raise ValueError("Batches cannot carry trade, discard or Knight instructions")
    if choice.notes_update is not None:
        try:
            validate_notes(choice.notes_update, max_notes_chars)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"PlayerChoice.notes_update: {exc}") from exc
    if isinstance(choice.action_index, bool) or not isinstance(choice.action_index, int):
        raise ValueError("Action index must be an integer")
    try:
        action = context.action_at(choice.action_index)
    except IndexError as exc:
        raise ValueError(str(exc)) from exc
    if action.color != context.actor:
        raise ValueError("Selected action must belong to the context actor")
    for name in ("game_plan", "rationale", "raw_response", "native_reasoning"):
        if not isinstance(getattr(choice, name), str):
            raise ValueError(f"PlayerChoice.{name} must be a string")
    for name in (
        "model", "parse_warning", "provider_response_id", "provider_request_id",
        "provider_native_finish_reason",
    ):
        value = getattr(choice, name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"PlayerChoice.{name} must be a string or None")
    for name in ("usage", "reasoning_request"):
        value = getattr(choice, name)
        if not isinstance(value, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str)
            for item in value
        ):
            raise ValueError(f"PlayerChoice.{name} must contain (string, value) pairs")
    if not isinstance(choice.native_reasoning_details, tuple):
        raise ValueError("PlayerChoice.native_reasoning_details must be a tuple")
    if choice.latency_ms is not None and (
        isinstance(choice.latency_ms, bool)
        or not isinstance(choice.latency_ms, int)
        or choice.latency_ms < 0
    ):
        raise ValueError("PlayerChoice.latency_ms must be a non-negative integer or None")

    offer = choice.trade_offer
    is_trade = action.action_type in {ActionType.OFFER_TRADE, ActionType.COUNTER_OFFER}
    if offer is not None and not isinstance(offer, TradeOffer):
        raise ValueError("trade_offer must be a TradeOffer")
    if offer is not None and not is_trade:
        raise ValueError("trade_offer is only allowed for OFFER_TRADE or COUNTER_OFFER")
    if is_trade:
        if isinstance(action.value, str):
            if offer is None:
                raise ValueError("Selected trade action requires an exact trade_offer")
            if (
                offer.id is not None
                or offer.created_round is not None
                or not isinstance(offer.willing_by, set)
                or offer.willing_by
                or not isinstance(offer.declined_by, set)
                or offer.declined_by
                or not isinstance(offer.status, TradeOfferStatus)
                or offer.status != TradeOfferStatus.ACTIVE
            ):
                raise ValueError("trade_offer may not override engine-owned lifecycle metadata")
            parent = None
            audience = frozenset(context.observation.opponent_resource_counts)
            if action.action_type == ActionType.COUNTER_OFFER:
                if not action.value.startswith("COUNTER_OFFER:"):
                    raise ValueError("Selected counteroffer menu entry has no parent")
                parent, separator, _ = action.value.removeprefix("COUNTER_OFFER:").rpartition(":")
                if not separator or not parent:
                    raise ValueError("Selected counteroffer menu entry has no parent")
                audience = frozenset({context.observation.turn_player_color})
            if offer.parent_offer_id != parent:
                raise ValueError("trade_offer parent must match the selected menu entry")
            if action.action_type == ActionType.COUNTER_OFFER:
                if offer.audience != audience:
                    raise ValueError("trade_offer audience must match the selected menu entry")
            elif not offer.audience or not offer.audience <= audience:
                # A root offer may target any nonempty subset of the other seats.
                raise ValueError("trade_offer audience must name other participants only")
        elif isinstance(action.value, TradeOffer):
            if offer is not None and offer != action.value:
                raise ValueError("trade_offer cannot replace a concrete menu action")
            offer = action.value
        else:
            raise ValueError("Selected trade action requires a TradeOffer or parameterized menu entry")
        if not isinstance(offer.offered_by, Color) or offer.offered_by != context.actor:
            raise ValueError("trade_offer offerer must match the context actor")
        if not isinstance(offer.audience, frozenset) or any(
            not isinstance(color, Color) for color in offer.audience
        ):
            raise ValueError("trade_offer audience must be a frozenset of player colors")
        if not isinstance(offer.give, tuple) or not isinstance(offer.receive, tuple):
            raise ValueError("TradeOffer bundles must be tuples of five integer counts")
        if offer.parent_offer_id is not None and (
            not isinstance(offer.parent_offer_id, str) or not offer.parent_offer_id
        ):
            raise ValueError("trade_offer parent must be a nonempty string or None")
        # Reconstruct to recheck resource invariants after possible mutation.
        validated = TradeOffer(
            offered_by=offer.offered_by,
            audience=offer.audience,
            give=offer.give,
            receive=offer.receive,
            give_any=offer.give_any,
            receive_any=offer.receive_any,
            parent_offer_id=offer.parent_offer_id,
        )
        if isinstance(action.value, str):
            action = Action(action.color, action.action_type, validated)

    priority = choice.confirm_if_accepted_by
    if priority is not None:
        if action.action_type != ActionType.OFFER_TRADE or offer is None or offer.parent_offer_id is not None:
            raise ValueError("confirm_if_accepted_by is only allowed on a root OFFER_TRADE")
        if context.actor != context.observation.turn_player_color:
            raise ValueError("Only the turn-player proposer may preauthorize confirmation")
        if offer.give_any or offer.receive_any:
            raise ValueError("Preauthorization requires exact terms without wildcards")
        if priority != "ANY" and (
            not isinstance(priority, tuple) or not priority
            or any(not isinstance(color, Color) or color not in offer.audience for color in priority)
            or len(set(priority)) != len(priority)
        ):
            raise ValueError("confirm_if_accepted_by must be ANY or distinct ordered players in the offer audience")

    cards = choice.discard_cards
    if cards is not None:
        if action.action_type != ActionType.DISCARD:
            raise ValueError("discard_cards is only allowed for DISCARD")
        if not isinstance(cards, tuple) or any(
            not isinstance(card, str) or card not in RESOURCE_NAMES for card in cards
        ):
            raise ValueError("discard_cards must be a tuple of named resource cards")
        if len(cards) != context.discard_count:
            raise ValueError(f"Must discard exactly {context.discard_count} resource cards")
        if any(count > context.observation.my_resources.get(card, 0) for card, count in Counter(cards).items()):
            raise ValueError("Cannot discard resource cards the player does not hold")
        if action.value is not None and Counter(action.value) != Counter(cards):
            raise ValueError("discard_cards cannot replace a concrete menu action")
        action = Action(action.color, action.action_type, cards)

    destination = choice.knight_destination
    if destination is not None:
        if action.action_type != ActionType.PLAY_KNIGHT_CARD:
            raise ValueError("knight_destination is only allowed for PLAY_KNIGHT_CARD")
        if not isinstance(destination, tuple) or len(destination) != 3 or any(
            type(coordinate) is not int for coordinate in destination
        ):
            raise ValueError("knight_destination must be a tuple of three integer coordinates")
        if context.observation.board_map is None or destination not in context.observation.board_map.land_tiles:
            raise ValueError("knight_destination must be a land tile")
        if destination == context.observation.robber_position:
            raise ValueError("knight_destination must differ from the current robber tile")
    return deepcopy(action)


def choice_followup_action(
    context: PlayerContext, choice: PlayerChoice | CommunicationChoice,
) -> Action | None:
    """Return the checked requested followup, not a simulation of its execution.

    The caller must stop at an engine victory before applying this action.
    Legacy choices leave the next decision to the player.
    """
    if isinstance(choice, CommunicationChoice) or choice.knight_destination is None:
        return None
    action = action_from_choice(context, choice)
    return Action(action.color, ActionType.MOVE_ROBBER, choice.knight_destination)


def validate_player_attempt(
    context: PlayerContext,
    attempt: PlayerAttempt,
    *,
    max_notes_chars: int = MAX_NOTES_CHARS,
) -> Action | CommunicationChoice:
    """Validate a returned attempt before accessing its choice parameters."""
    if not isinstance(attempt, PlayerAttempt):
        raise ValueError("Player attempt must be a PlayerAttempt")
    if not isinstance(attempt.context_id, str) or attempt.context_id != context.context_id:
        raise ValueError("Player attempt does not match the requested context")
    if attempt.validation_error is not None and not isinstance(attempt.validation_error, str):
        raise ValueError("Player attempt validation_error must be a string or None")
    if attempt.choice is None:
        raise ValueError(attempt.validation_error or "Return one valid PlayerChoice")
    if attempt.validation_error is not None:
        raise ValueError("Player attempt cannot contain both a choice and a validation error")
    if isinstance(attempt.choice, CommunicationChoice):
        if not context.speech_allowed or attempt.choice.mode != CommunicationMode.SAY:
            raise ValueError("This decision requires a game action; standalone speech is unavailable")
        validate_communication_choice(
            attempt.choice, speaker=context.actor,
            participants=(context.actor, *context.observation.opponent_resource_counts),
            max_notes_chars=max_notes_chars,
        )
        if attempt.choice.respondents is None:
            raise ValueError("Standalone speech requires explicit respondents")
        return deepcopy(attempt.choice)
    return action_from_choice(context, attempt.choice, max_notes_chars=max_notes_chars)
