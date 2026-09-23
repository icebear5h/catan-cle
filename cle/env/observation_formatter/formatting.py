"""Assemble the complete observation text in its historical section order."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.env import observation_formatter

if TYPE_CHECKING:
    from . import CatanObservationFormatter, FormattedObservation, Observation


def format_observation(
    self: CatanObservationFormatter,
    obs: Observation,
    *,
    include_legal_actions: bool = True,
    include_initial_placement_order: bool = True,
    shared: bool = False,
) -> FormattedObservation:
    # Build node coordinate map once for the whole format pass.
    self._node_coords = self._build_node_coordinate_map(obs.board_map)
    board_state = self._format_board_state(obs)
    resources = self._format_resources(obs, shared=shared)
    opponents = self._format_opponents(obs, shared=shared)
    valid_actions = self._format_valid_actions(obs) if include_legal_actions else ""
    strategic_context = self._format_strategic_context(
        obs, include_initial_placement_order=include_initial_placement_order, shared=shared,
    )
    trade_context = self._format_trade_context(obs)
    events_section = self._format_events(obs)
    trade_section = f"\n<trading>\n{trade_context}\n</trading>" if trade_context else ""
    events_block = f"\n<recent_events>\n{events_section}\n</recent_events>" if events_section else ""
    actions_block = (
        f"\n<valid_actions>\n{valid_actions}\n</valid_actions>" if include_legal_actions else ""
    )
    color_name = obs.my_color.value if hasattr(obs.my_color, 'value') else str(obs.my_color)
    raw_str = f"""<game_state turn="{obs.current_turn}" player="{color_name}">{events_block}
<phase_info>
{strategic_context}
</phase_info>
<board_state>
{board_state}
</board_state>
<resources>
{resources}
</resources>
<opponents>
{opponents}
</opponents>{trade_section}{actions_block}
</game_state>""".strip()
    return observation_formatter.FormattedObservation(
        raw_str=raw_str,
        board_state=board_state,
        resources=resources,
        opponents=opponents,
        valid_actions=valid_actions,
        strategic_context=strategic_context,
        trade_context=trade_context,
    )
