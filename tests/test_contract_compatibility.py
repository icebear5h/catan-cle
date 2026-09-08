import base64
from dataclasses import FrozenInstanceError, fields, replace
import pickle

import pytest

from cle.players.contracts import PlayerChoice, PlayerContext


_PRE_DISCARD_CHOICE = (
    "gAWVcwAAAAAAAACMFWNsZS5wbGF5ZXJzLmNvbnRyYWN0c5SMDFBsYXllckNob2ljZZSTlCmBlF2U"
    "KEsDTowYcHJlLXJlbWVkaWF0aW9uIHNuYXBzaG90lIwAlIwSPGFjdGlvbj4zPC9hY3Rpb24+lE4p"
    "Tk5oBikpTk5OZWIu"
)


def test_real_pre_discard_player_choice_pickle_defaults_appended_slot():
    restored = pickle.loads(base64.b64decode(_PRE_DISCARD_CHOICE))

    assert restored == PlayerChoice(
        action_index=3,
        game_plan="pre-remediation snapshot",
        raw_response="<action>3</action>",
    )
    assert restored.discard_cards is None
    assert replace(restored, action_index=4).discard_cards is None
    assert pickle.loads(pickle.dumps(restored)) == restored
    with pytest.raises(FrozenInstanceError):
        restored.discard_cards = ("WOOD",)


def test_current_player_choice_pickle_preserves_discard_bundle():
    choice = PlayerChoice(0, discard_cards=("WOOD", "WOOD", "ORE", "ORE"))
    assert fields(PlayerChoice)[-1].name == "discard_cards"
    assert pickle.loads(pickle.dumps(choice)) == choice


def test_player_context_appends_only_the_agreed_default_fields():
    appended = fields(PlayerContext)[-3:]
    assert [(item.name, item.default) for item in appended] == [
        ("recent_messages", ()),
        ("active_commitments", ()),
        ("discard_count", 0),
    ]
