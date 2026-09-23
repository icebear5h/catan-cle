"""Fold one replay event into the running `GameState`."""

from data_pipeline.bootstrapping.replay_decoder._enums import ActionState, BuildingType
from data_pipeline.bootstrapping.replay_decoder._models import GameState
from data_pipeline.bootstrapping.replay_decoder._types import ActionInfo, MoveDetail
from data_pipeline.json_coerce import as_dict, as_float, as_int, as_list
from data_pipeline.json_types import JsonDict


def apply_event(state: GameState, event: JsonDict) -> ActionInfo:
    """Apply a single event to the game state, return action info"""
    state_change = as_dict(event.get('stateChange', {}))
    details: dict[str, MoveDetail] = {}
    action_info: ActionInfo = {
        'delta_s': as_float(as_dict(event.get('input', {})).get('deltaS', 0)),
        'action_type': None,
        'player': None,
        'details': details,
    }

    # Apply map state changes (buildings, roads)
    if 'mapState' in state_change:
        ms = as_dict(state_change['mapState'])

        # Corner states (settlements, cities)
        for cid, raw_corner in as_dict(ms.get('tileCornerStates', {})).items():
            cdata = as_dict(raw_corner)
            corner = state.corners[int(cid)]
            if 'owner' in cdata:
                corner.owner = as_int(cdata['owner'])
                action_info['player'] = corner.owner
            if 'buildingType' in cdata:
                old_type = corner.building_type
                corner.building_type = BuildingType(as_int(cdata['buildingType']))
                if old_type == BuildingType.NONE:
                    action_info['action_type'] = 'build_settlement'
                elif old_type == BuildingType.SETTLEMENT:
                    action_info['action_type'] = 'build_city'
                details['corner'] = int(cid)

        # Edge states (roads)
        for eid, raw_edge in as_dict(ms.get('tileEdgeStates', {})).items():
            edata = as_dict(raw_edge)
            edge = state.edges[int(eid)]
            if 'owner' in edata:
                edge.owner = as_int(edata['owner'])
                action_info['player'] = edge.owner
            if 'type' in edata:
                edge.road_type = as_int(edata['type'])
                action_info['action_type'] = 'build_road'
                details['edge'] = int(eid)

    # Apply current state changes
    if 'currentState' in state_change:
        cs = as_dict(state_change['currentState'])
        if 'currentTurnPlayerColor' in cs:
            state.current_player = as_int(cs['currentTurnPlayerColor'])
        if 'actionState' in cs:
            state.action_state = ActionState(as_int(cs['actionState']))
        if 'completedTurns' in cs:
            state.completed_turns = as_int(cs['completedTurns'])

    # Apply dice state changes
    if 'diceState' in state_change:
        ds = as_dict(state_change['diceState'])
        if 'dice1' in ds and 'dice2' in ds:
            state.dice_value = (as_int(ds['dice1']), as_int(ds['dice2']))
            action_info['action_type'] = 'roll_dice'
            details['dice'] = state.dice_value

    # Apply player state changes
    if 'playerStates' in state_change:
        for pid, raw_player in as_dict(state_change['playerStates']).items():
            pdata = as_dict(raw_player)
            color = int(pid)
            if color not in state.players:
                continue
            player = state.players[color]

            if 'resourceCards' in pdata:
                cards = as_dict(pdata['resourceCards']).get('cards', [])
                player.resources = [as_int(c) for c in as_list(cards)]

            if 'victoryPointsState' in pdata:
                for vptype, vpcount in as_dict(pdata['victoryPointsState']).items():
                    player.victory_points[vptype] = as_int(vpcount)

            if 'bankTradeRatiosState' in pdata:
                for rid, ratio in as_dict(pdata['bankTradeRatiosState']).items():
                    player.bank_trade_ratios[int(rid)] = as_int(ratio)

    # Apply robber state changes
    if 'mechanicRobberState' in state_change:
        rs = as_dict(state_change['mechanicRobberState'])
        if 'locationTileIndex' in rs:
            state.robber_tile = as_int(rs['locationTileIndex'])
            action_info['action_type'] = 'move_robber'
            details['tile'] = state.robber_tile

    return action_info


__all__ = ["apply_event"]
