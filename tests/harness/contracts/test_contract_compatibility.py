import base64
import pickle
from dataclasses import FrozenInstanceError, fields, replace
from typing import Any

import pytest

from cle.players.contracts import PlayerChoice, PlayerContext

_PRE_DISCARD_CHOICE = (
    "gAWVcwAAAAAAAACMFWNsZS5wbGF5ZXJzLmNvbnRyYWN0c5SMDFBsYXllckNob2ljZZSTlCmBlF2U"
    "KEsDTowYcHJlLXJlbWVkaWF0aW9uIHNuYXBzaG90lIwAlIwSPGFjdGlvbj4zPC9hY3Rpb24+lE4p"
    "Tk5oBikpTk5OZWIu"
)
_PRE_KNIGHT_CHOICE = (
    "gASVcgAAAAAAAACMFWNsZS5wbGF5ZXJzLmNvbnRyYWN0c5SMDFBsYXllckNob2ljZZSTlCmBlF2U"
    "KEsCTowTcHJlLWtuaWdodCBzbmFwc2hvdJSMAJSMB2Rpc2NhcmSUTilOTmgGKSlOTk6MBFdPT0SU"
    "jANPUkWUhpRlYi4="
)


def test_real_pre_discard_player_choice_pickle_defaults_appended_slot() -> None:
    restored = pickle.loads(base64.b64decode(_PRE_DISCARD_CHOICE))

    assert restored == PlayerChoice(
        action_index=3,
        game_plan="pre-remediation snapshot",
        raw_response="<action>3</action>",
    )
    assert restored.discard_cards is None
    assert restored.knight_destination is None
    assert replace(restored, action_index=4).discard_cards is None
    assert replace(restored, action_index=4).knight_destination is None
    assert pickle.loads(pickle.dumps(restored)) == restored
    with pytest.raises(FrozenInstanceError):
        restored.discard_cards = ("WOOD",)


def test_current_player_choice_pickle_preserves_discard_bundle() -> None:
    choice = PlayerChoice(0, discard_cards=("WOOD", "WOOD", "ORE", "ORE"))
    assert [
        item.name for item in fields(PlayerChoice)
        if item.name in {"discard_cards", "knight_destination"}
    ] == [
        "discard_cards", "knight_destination",
    ]
    assert pickle.loads(pickle.dumps(choice)) == choice


def test_real_pre_knight_player_choice_pickle_preserves_discard_and_defaults_destination() -> None:
    restored = pickle.loads(base64.b64decode(_PRE_KNIGHT_CHOICE))
    assert restored == PlayerChoice(
        2,
        game_plan="pre-knight snapshot",
        raw_response="discard",
        discard_cards=("WOOD", "ORE"),
    )
    assert restored.knight_destination is None
    assert replace(restored, action_index=3).knight_destination is None
    assert pickle.loads(pickle.dumps(restored)) == restored


def test_current_player_choice_pickle_preserves_knight_destination() -> None:
    choice: Any = PlayerChoice(1, knight_destination=(1, -1, 0))
    assert pickle.loads(pickle.dumps(choice)) == choice
    with pytest.raises(FrozenInstanceError):
        choice.knight_destination = (0, 0, 0)


def test_player_context_appends_only_the_agreed_default_fields() -> None:
    appended = [
        item for item in fields(PlayerContext)
        if item.name in {"recent_messages", "active_commitments", "discard_count"}
    ]
    assert [(item.name, item.default) for item in appended] == [
        ("recent_messages", ()),
        ("active_commitments", ()),
        ("discard_count", 0),
    ]
