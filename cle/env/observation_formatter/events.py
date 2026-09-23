"""Historical action-log text, including private-card redaction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.game_engine.trading import TradeCandidate, TradeOffer

if TYPE_CHECKING:
    from . import CatanObservationFormatter, Observation


def format_events(self: CatanObservationFormatter, obs: Observation) -> str:
    """Format recent game events since last observation."""
    if not obs.recent_events:
        return ""
    node_coords = self._node_coords
    lines = ["RECENT EVENTS:"]
    for action in obs.recent_events:
        color_str = action.color.name if hasattr(action.color, 'name') else str(action.color)
        at = action.action_type
        val = action.value
        if at.name == "ROLL":
            if isinstance(val, tuple) and len(val) == 2:
                lines.append(f"  {color_str}: ROLL {val[0]+val[1]} ({val[0]}+{val[1]})")
            else:
                lines.append(f"  {color_str}: ROLL {val}")
        elif at.name == "BUILD_SETTLEMENT":
            lines.append(f"  {color_str}: BUILD_SETTLEMENT at {self._format_node(val, node_coords)}")
        elif at.name == "BUILD_CITY":
            lines.append(f"  {color_str}: BUILD_CITY at {self._format_node(val, node_coords)}")
        elif at.name == "BUILD_ROAD":
            lines.append(f"  {color_str}: BUILD_ROAD at {self._format_edge(val, node_coords)}")
        elif at.name == "BUY_DEVELOPMENT_CARD":
            lines.append(f"  {color_str}: BUY_DEVELOPMENT_CARD")
        elif at.name == "PLAY_KNIGHT_CARD":
            lines.append(f"  {color_str}: PLAY_KNIGHT_CARD")
        elif at.name == "PLAY_YEAR_OF_PLENTY":
            if isinstance(val, tuple) and len(val) == 2:
                r1 = val[0].name if hasattr(val[0], 'name') else str(val[0])
                r2 = val[1].name if hasattr(val[1], 'name') else str(val[1])
                lines.append(f"  {color_str}: PLAY_YEAR_OF_PLENTY {r1}, {r2}")
            else:
                lines.append(f"  {color_str}: PLAY_YEAR_OF_PLENTY {val}")
        elif at.name == "PLAY_MONOPOLY":
            r = val.name if hasattr(val, 'name') else str(val)
            lines.append(f"  {color_str}: PLAY_MONOPOLY {r}")
        elif at.name == "PLAY_ROAD_BUILDING":
            lines.append(f"  {color_str}: PLAY_ROAD_BUILDING")
        elif at.name == "MARITIME_TRADE":
            if isinstance(val, tuple) and len(val) >= 5:
                offered_parts = []
                for i in range(len(val) - 1):
                    if val[i] is not None:
                        r = val[i].name if hasattr(val[i], 'name') else str(val[i])
                        offered_parts.append(r)
                received = val[-1]
                r_name = received.name if hasattr(received, 'name') else str(received)
                count = len(offered_parts)
                if offered_parts:
                    lines.append(f"  {color_str}: MARITIME_TRADE {count} {offered_parts[0]} for 1 {r_name}")
                else:
                    lines.append(f"  {color_str}: MARITIME_TRADE {val}")
            else:
                lines.append(f"  {color_str}: MARITIME_TRADE {val}")
        elif at.name in {"OFFER_TRADE", "COUNTER_OFFER"}:
            if isinstance(val, TradeOffer):
                give = self._format_resource_tuple_with_any(val.give, val.give_any)
                receive = self._format_resource_tuple_with_any(val.receive, val.receive_any)
                parent = f" countering {val.parent_offer_id}" if val.parent_offer_id else ""
                lines.append(
                    f"  {color_str}: {at.name}{parent}, giving {give} for {receive}"
                )
            else:
                lines.append(f"  {color_str}: {at.name} {val}")
        elif at.name == "ACCEPT_TRADE":
            lines.append(f"  {color_str}: WILLING_TO_TRADE on offer {val}")
        elif at.name == "REJECT_TRADE":
            lines.append(f"  {color_str}: DECLINED offer {val}")
        elif at.name == "CONFIRM_TRADE":
            if isinstance(val, TradeCandidate):
                partner = self._color_name(val.counterparty)
                lines.append(f"  {color_str}: CONFIRM_TRADE offer {val.offer_id} with {partner}")
            else:
                partner = val.name if hasattr(val, 'name') else str(val)
                lines.append(f"  {color_str}: CONFIRM_TRADE with {partner}")
        elif at.name == "CANCEL_TRADE":
            lines.append(f"  {color_str}: CANCEL_TRADE")
        elif at.name == "MOVE_ROBBER":
            lines.append(f"  {color_str}: MOVE_ROBBER to {val}")
        elif at.name == "STEAL":
            if isinstance(val, tuple) and len(val) >= 2:
                victim = val[0].name if hasattr(val[0], 'name') else str(val[0])
                lines.append(f"  {color_str}: STEAL from {victim}")
            else:
                lines.append(f"  {color_str}: STEAL {val}")
        elif at.name == "DISCARD":
            if isinstance(val, (list, tuple)):
                lines.append(f"  {color_str}: DISCARD {len(val)} cards")
            else:
                lines.append(f"  {color_str}: DISCARD")
        elif at.name == "END_TURN":
            lines.append(f"  {color_str}: END_TURN")
        else:
            lines.append(f"  {color_str}: {at.name} {val}")
    return "\n".join(lines)
