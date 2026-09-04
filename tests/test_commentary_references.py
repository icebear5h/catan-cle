import contextlib
import io
import json
from pathlib import Path

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.replay.colonist.coordinates import create_map_from_colonist
from playground.game_viewer.commentary.references import (
    build_corner_index,
    extract_number_mentions,
    ground_text_references,
    resolve_corner_mention,
)


PAIR_REPLAY = Path(
    "artifacts/staging/colonist/replays/242781000.json"
)


def _paired_map():
    raw_data = json.loads(PAIR_REPLAY.read_text(encoding="utf-8"))
    initial_state = raw_data["data"]["eventHistory"]["initialState"]
    with contextlib.redirect_stdout(io.StringIO()):
        return create_map_from_colonist(initial_state)


def _game(catan_map):
    return GameEngine(
        [
            Color.BLUE,
            Color.BLACK,
            Color.ORANGE,
            Color.RED,
        ],
        catan_map=catan_map,
        shuffle_players=False,
    )


def test_extracts_hyphenated_spaced_compact_and_asr_eight_mentions():
    text = "go 8-4-10 then 6 9 3; cleaned 9510, E410, A10 down; 3-1 port; 110%."

    mentions = extract_number_mentions(text)

    assert [mention.surface for mention in mentions] == [
        "8-4-10",
        "6 9 3",
        "9510",
        "E410",
        "A10",
    ]
    assert [mention.number_options for mention in mentions] == [
        ((8, 4, 10),),
        ((6, 9, 3),),
        ((9, 5, 10),),
        ((8, 4, 10),),
        ((8, 10),),
    ]
    assert mentions[-1].direction == "down"
    assert all(text[mention.start : mention.end] == mention.surface for mention in mentions)


def test_rejects_singles_roll_percentages_and_port_ratios():
    text = "Roll a 10. Trade at 3-1. I am 110% sure; game 242781000."

    assert extract_number_mentions(text) == ()


def test_resolves_known_pilot_corners_and_preserves_ambiguity():
    corner_index = build_corner_index(_paired_map())
    expected = {
        "8 4 10": ("unique", [(37, 7)]),
        "6-9-3": ("unique", [(26, 10)]),
        "9510": ("unique", [(41, 23)]),
        "8 4 3": ("ambiguous", [(7, 18), (30, 8)]),
        "8 5 10": ("none", []),
    }

    for source, (status, candidates) in expected.items():
        mention = extract_number_mentions(source)[0]
        result = resolve_corner_mention(corner_index, mention)
        assert result.status == status
        assert [
            (candidate.corner.colonist_corner_id, candidate.corner.engine_node_id)
            for candidate in result.candidates
        ] == candidates


def test_number_order_is_irrelevant_but_source_order_is_preserved():
    corner_index = build_corner_index(_paired_map())

    forward = ground_text_references("8 4 10", corner_index)[0]
    permuted = ground_text_references("10-8-4", corner_index)[0]

    assert forward.mention.number_options == ((8, 4, 10),)
    assert permuted.mention.number_options == ((10, 8, 4),)
    assert forward.candidates == permuted.candidates
    assert forward.mention.surface == "8 4 10"
    assert permuted.mention.surface == "10-8-4"


def test_pair_direction_remains_a_candidate_set_not_a_guess():
    corner_index = build_corner_index(_paired_map())

    result = ground_text_references("play toward 8 10 down", corner_index)[0]

    assert result.mention.direction == "down"
    assert result.status == "ambiguous"
    assert [candidate.corner.colonist_corner_id for candidate in result.candidates] == [
        10,
        11,
        34,
        37,
    ]
    assert len({candidate.corner.screen_coordinate for candidate in result.candidates}) == 4


def test_dynamic_candidate_facts_track_legality_and_occupancy():
    catan_map = _paired_map()
    corner_index = build_corner_index(catan_map)
    game = _game(catan_map)
    mention = extract_number_mentions("8-4-10")[0]

    before = resolve_corner_mention(corner_index, mention, game.state)
    assert before.candidates[0].legal_settlement_now is True
    assert before.candidates[0].occupied_by is None

    settlement = next(
        action
        for action in game.state.playable_actions
        if action.value == before.candidates[0].corner.engine_node_id
    )
    game.step(settlement)

    after = resolve_corner_mention(corner_index, mention, game.state)
    assert after.candidates[0].legal_settlement_now is False
    assert after.candidates[0].occupied_by == "BLUE"
    assert after.candidates[0].building == "SETTLEMENT"
