"""Parse Colonist events into action type hints for the engine."""

from .constants import COLONIST_DEV_CARD, COLONIST_RES_TO_ENGINE
from .helpers import colonist_resources_to_tuple


def parse_colonist_events_to_actions(events, tile_hex_states=None):
    """Parse Colonist events into action type hints for our engine.

    Note: Colonist uses delta encoding - only changed values are included.
    We track state across events to get complete dice values and trade states.
    Also tracks Colonist's resource state per player (source of truth for syncing).

    Each action dict includes 'expected_resources': {player_id: [card_ids...]}
    representing the expected state AFTER that action executes.

    Args:
        events: List of Colonist events
        tile_hex_states: Optional dict mapping tile_index -> {type, diceNumber, x, y}
            for matching MOVE_ROBBER actions
    """
    if tile_hex_states is None:
        tile_hex_states = {}
    actions = []
    # Track last known dice values (Colonist sends deltas)
    last_dice = [1, 1]  # Default to (1, 1)

    # Track Colonist's resource state per player (source of truth)
    # Key: player color ID, Value: list of resource card IDs
    colonist_resources = {}  # {player_id: [card_ids...]}

    def snapshot_resources():
        """Create a deep copy of current colonist_resources."""
        return {k: list(v) for k, v in colonist_resources.items()}

    # Track active trades to detect new offers and responses
    # Structure: {trade_id: {full trade data with merged deltas}}
    active_trades = {}
    # Track which players have responded to each trade (to avoid duplicates)
    trade_responses = {}  # {trade_id: {color_id: response_value}}
    # Sequential trade numbering for UI display
    trade_counter = 0
    trade_id_to_number = {}  # {colonist_trade_id: our_trade_number}

    # Track current player to detect turn changes and emit END_TURN
    last_player = None
    # Track action state to only emit END_TURN during main game (not setup)
    # actionState 0 = main turn, 1/3 = setup, 24 = must roll
    is_main_game = False

    for i, event in enumerate(events):
        # Update resource tracking from playerStates FIRST (before detecting actions)
        # This gives us the expected state AFTER this event's action executes
        state_change = event.get("stateChange", {})
        player_states = state_change.get("playerStates", {})

        # Detect turn changes and emit END_TURN (only in main game)
        current_state = state_change.get("currentState", {})
        action_state = current_state.get("actionState")
        if action_state is not None:
            # Main game starts when we see actionState 24 (must roll) or 0 (main turn)
            if action_state in (0, 24):
                is_main_game = True

        current_player = current_state.get("currentTurnPlayerColor")
        if is_main_game and current_player is not None and last_player is not None and current_player != last_player:
            actions.append({
                "index": i,
                "type": "END_TURN",
                "player": last_player,
                "expected_resources": snapshot_resources(),
            })
        if current_player is not None:
            last_player = current_player
        for player_id_str, pstate in player_states.items():
            if "resourceCards" in pstate:
                cards = pstate["resourceCards"].get("cards", [])
                colonist_resources[int(player_id_str)] = cards
        action_info = {"index": i, "type": None, "player": None}

        # Dice roll - merge with last known values
        dice_state = state_change.get("diceState", {})
        if dice_state.get("diceThrown"):
            # Update only the dice that changed (delta encoding)
            if "dice1" in dice_state and dice_state["dice1"] is not None:
                last_dice[0] = dice_state["dice1"]
            if "dice2" in dice_state and dice_state["dice2"] is not None:
                last_dice[1] = dice_state["dice2"]

            actions.append({
                "index": i,
                "type": "ROLL",
                "dice": tuple(last_dice),
                "expected_resources": snapshot_resources(),
            })

        # Building placement (settlement/city)
        map_state = state_change.get("mapState", {})
        corners = map_state.get("tileCornerStates", {})
        if corners:
            for corner_id, data in corners.items():
                btype = data.get("buildingType", 0)
                actions.append({
                    "index": i,
                    "type": "BUILD_SETTLEMENT" if btype == 1 else "BUILD_CITY",
                    "player": data.get("owner"),
                    "colonist_corner": int(corner_id),
                    "expected_resources": snapshot_resources(),
                })
                break

        # Road placement
        edges = map_state.get("tileEdgeStates", {})
        if edges:
            for edge_id, data in edges.items():
                actions.append({
                    "index": i,
                    "type": "BUILD_ROAD",
                    "player": data.get("owner"),
                    "colonist_edge": int(edge_id),
                    "expected_resources": snapshot_resources(),
                })
                break

        # Trade events
        trade_state = state_change.get("tradeState", {})
        active_offers = trade_state.get("activeOffers", {})

        for trade_id, offer_data in active_offers.items():
            if offer_data is None:
                # Trade cancelled or completed - clean up
                if trade_id in active_trades:
                    del active_trades[trade_id]
                if trade_id in trade_responses:
                    del trade_responses[trade_id]
                continue

            # Check if this is a new trade offer (has full data)
            if "creator" in offer_data and "offeredResources" in offer_data and "wantedResources" in offer_data:
                # New trade offer
                active_trades[trade_id] = offer_data.copy()
                trade_responses[trade_id] = {}

                offered, offered_any = colonist_resources_to_tuple(offer_data["offeredResources"])
                wanted, wanted_any = colonist_resources_to_tuple(offer_data["wantedResources"])

                # Check if this is a counter offer
                counter_offer_to = offer_data.get("counterOfferInResponseToTradeId")
                is_counter = counter_offer_to is not None

                # Assign trade number - counter offers inherit parent's number
                if is_counter and counter_offer_to in trade_id_to_number:
                    trade_num = trade_id_to_number[counter_offer_to]
                else:
                    trade_counter += 1
                    trade_num = trade_counter
                trade_id_to_number[trade_id] = trade_num

                action_info = {
                    "index": i,
                    "type": "COUNTER_OFFER" if is_counter else "OFFER_TRADE",
                    "player": offer_data["creator"],
                    "trade_id": trade_id,
                    "trade_num": trade_num,
                    "offered": offered,
                    "wanted": wanted,
                    "trade_tuple": offered + wanted + (offered_any, wanted_any),
                    "is_flexible": bool(offered_any or wanted_any),
                    "is_counter_offer": is_counter,
                    "counter_offer_to": counter_offer_to,
                    "expected_resources": snapshot_resources(),
                }
                actions.append(action_info)

            # Check for response updates (delta)
            elif "playerResponses" in offer_data:
                # Merge into tracked trade
                if trade_id in active_trades:
                    if "playerResponses" not in active_trades[trade_id]:
                        active_trades[trade_id]["playerResponses"] = {}
                    active_trades[trade_id]["playerResponses"].update(offer_data["playerResponses"])

                if trade_id not in trade_responses:
                    trade_responses[trade_id] = {}

                for color_str, response in offer_data["playerResponses"].items():
                    color_id = int(color_str)
                    # Only emit action if response changed (avoid duplicates)
                    if color_id not in trade_responses[trade_id] or trade_responses[trade_id][color_id] != response:
                        trade_responses[trade_id][color_id] = response

                        trade_num = trade_id_to_number.get(trade_id, 0)
                        # Get trade creator from active_trades
                        creator = active_trades.get(trade_id, {}).get("creator")
                        if response == 1:  # Accept
                            action_info = {
                                "index": i,
                                "type": "ACCEPT_TRADE",
                                "player": color_id,
                                "trade_id": trade_id,
                                "trade_num": trade_num,
                                "creator": creator,
                                "expected_resources": snapshot_resources(),
                            }
                            actions.append(action_info)
                        elif response == 2:  # Reject
                            action_info = {
                                "index": i,
                                "type": "REJECT_TRADE",
                                "player": color_id,
                                "trade_id": trade_id,
                                "trade_num": trade_num,
                                "creator": creator,
                                "expected_resources": snapshot_resources(),
                            }
                            actions.append(action_info)

        # Check for completed trades in gameLogState (type 115)
        game_log_state = state_change.get("gameLogState", {})
        for log_id, log_entry in game_log_state.items():
            text = log_entry.get("text", {})
            if text.get("type") == 115:
                # Trade completed - CONFIRM_TRADE
                given, _ = colonist_resources_to_tuple(text.get("givenCardEnums", []))
                received, _ = colonist_resources_to_tuple(text.get("receivedCardEnums", []))
                acceptor = text.get("acceptingPlayerColor")
                creator = text.get("playerColor")

                action_info = {
                    "index": i,
                    "type": "CONFIRM_TRADE",
                    "player": creator,
                    "acceptor": acceptor,
                    "offered": given,
                    "received": received,
                    "trade_tuple": given + received + (acceptor,),
                    "trade_num": trade_counter,
                    "expected_resources": snapshot_resources(),
                }
                actions.append(action_info)

            # Bank/maritime trade (LOG type 116)
            elif text.get("type") == 116:
                player = text.get("playerColor")
                given, _ = colonist_resources_to_tuple(text.get("givenCardEnums", []))
                received, _ = colonist_resources_to_tuple(text.get("receivedCardEnums", []))

                action_info = {
                    "index": i,
                    "type": "MARITIME_TRADE",
                    "player": player,
                    "given": given,
                    "received": received,
                    "expected_resources": snapshot_resources(),
                }
                actions.append(action_info)

            # Discard on 7 (LOG type 55)
            elif text.get("type") == 55:
                player = text.get("playerColor")
                cards, _ = colonist_resources_to_tuple(text.get("cardEnums", []))

                action_info = {
                    "index": i,
                    "type": "DISCARD",
                    "player": player,
                    "cards": cards,
                    "expected_resources": snapshot_resources(),
                }
                actions.append(action_info)

            # Dev card played (LOG type 20)
            elif text.get("type") == 20:
                player = text.get("playerColor")
                card_enum = text.get("cardEnum")
                card_type = COLONIST_DEV_CARD.get(card_enum, f"UNKNOWN_{card_enum}")

                if card_type == "KNIGHT":
                    action_info = {
                        "index": i,
                        "type": "PLAY_KNIGHT_CARD",
                        "player": player,
                        "expected_resources": snapshot_resources(),
                    }
                    actions.append(action_info)
                elif card_type == "ROAD_BUILDING":
                    action_info = {
                        "index": i,
                        "type": "PLAY_ROAD_BUILDING",
                        "player": player,
                        "expected_resources": snapshot_resources(),
                    }
                    actions.append(action_info)
                elif card_type == "MONOPOLY":
                    action_info = {
                        "index": i,
                        "type": "PLAY_MONOPOLY",
                        "player": player,
                        "expected_resources": snapshot_resources(),
                    }
                    actions.append(action_info)
                elif card_type == "YEAR_OF_PLENTY":
                    action_info = {
                        "index": i,
                        "type": "PLAY_YEAR_OF_PLENTY",
                        "player": player,
                        "expected_resources": snapshot_resources(),
                    }
                    actions.append(action_info)

            # Year of Plenty resources taken (LOG type 21)
            elif text.get("type") == 21:
                player = text.get("playerColor")
                cards = text.get("cardEnums", [])
                resource_list = [COLONIST_RES_TO_ENGINE.get(c) for c in cards if c in COLONIST_RES_TO_ENGINE]

                action_info = {
                    "index": i,
                    "type": "YEAR_OF_PLENTY_RESOURCES",
                    "player": player,
                    "resources": resource_list,
                    "expected_resources": snapshot_resources(),
                }
                actions.append(action_info)

            # Monopoly steal (LOG type 86)
            elif text.get("type") == 86:
                player = text.get("playerColor")
                card_enum = text.get("cardEnum")
                amount = text.get("amountStolen", 0)
                resource = COLONIST_RES_TO_ENGINE.get(card_enum)

                action_info = {
                    "index": i,
                    "type": "MONOPOLY_RESOURCE",
                    "player": player,
                    "resource": resource,
                    "amount": amount,
                    "expected_resources": snapshot_resources(),
                }
                actions.append(action_info)

            # Steal from player (LOG type 16)
            elif text.get("type") == 16:
                thief = text.get("playerColorThief")
                victim = text.get("playerColorVictim")

                stolen_resource = None
                for other_log_id, other_log in game_log_state.items():
                    other_text = other_log.get("text", {})
                    other_type = other_text.get("type")
                    if other_type in (14, 15):
                        card_enums = other_text.get("cardEnums", [])
                        if card_enums:
                            stolen_resource = COLONIST_RES_TO_ENGINE.get(card_enums[0])
                            break

                action_info = {
                    "index": i,
                    "type": "STEAL",
                    "player": thief,
                    "victim": victim,
                    "stolen_resource": stolen_resource,
                    "expected_resources": snapshot_resources(),
                }
                actions.append(action_info)

        # Development card bought
        dev_state = state_change.get("mechanicDevelopmentCardsState", {})
        for color_str, player_dev in dev_state.get("players", {}).items():
            bought = player_dev.get("developmentCardsBoughtThisTurn")
            if bought and bought != [] and bought is not None:
                card_enum = bought[-1]
                card_type = COLONIST_DEV_CARD.get(card_enum, "UNKNOWN")
                print(f"[Parse] BUY_DEVELOPMENT_CARD: player={color_str}, bought={bought}, card_enum={card_enum}, card_type={card_type}")
                action_info = {
                    "index": i,
                    "type": "BUY_DEVELOPMENT_CARD",
                    "player": int(color_str),
                    "card_type": card_type,
                    "expected_resources": snapshot_resources(),
                }
                actions.append(action_info)

        # MOVE_ROBBER - detect via mechanicRobberState (not via game log types)
        robber_state = state_change.get("mechanicRobberState", {})
        tile_index = robber_state.get("locationTileIndex")
        if tile_index is not None:
            tile_hex = tile_hex_states.get(str(tile_index), {})
            tile_info = {
                "resourceType": tile_hex.get("type"),
                "diceNumber": tile_hex.get("diceNumber"),
                "x": tile_hex.get("x"),
                "y": tile_hex.get("y"),
            }
            current_state = state_change.get("currentState", {})
            action_state = current_state.get("actionState", {})
            player = None
            if isinstance(action_state, dict):
                player = action_state.get("player")

            action_info = {
                "index": i,
                "type": "MOVE_ROBBER",
                "tile_info": tile_info,
                "tile_index": tile_index,
                "player": player,
                "expected_resources": snapshot_resources(),
            }
            actions.append(action_info)
            print(f"[Parse] MOVE_ROBBER: tile_index={tile_index}, tile_info={tile_info}")

    return actions
