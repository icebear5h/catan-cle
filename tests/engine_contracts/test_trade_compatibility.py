"""Contracts for relocating trade values without migrating saved windows."""

import base64
import pickle

import pytest

from cle.game_engine import trading
from cle.game_engine.models import trade
from cle.game_engine.models.player import Color

# Protocol-4 snapshot captured from trading.py before the model relocation.
# Includes both deal-key forms, responses, limits and a selected counteroffer.
HISTORICAL_WINDOW = (
    "gASVEQMAAAAAAACMF2NsZS5nYW1lX2VuZ2luZS50cmFkaW5nlIwLVHJhZGVXaW5kb3eUk5QpgZROfZQo"
    "jAJpZJSMCmhpc3RvcmljYWyUjAt0dXJuX3BsYXllcpSMHWNsZS5nYW1lX2VuZ2luZS5tb2RlbHMucGxh"
    "eWVylIwFQ29sb3KUk5SMA1JFRJSFlFKUjAxwYXJ0aWNpcGFudHOUaA1oCowFV0hJVEWUhZRSlGgKjARC"
    "TFVFlIWUUpSHlIwGbGltaXRzlGgAjAtUcmFkZUxpbWl0c5STlCmBlF2UKEsCSwhLAksDSwJlYowFcm91"
    "bmSUSwCMBm9mZmVyc5R9lCiMBnotcm9vdJRoAIwKVHJhZGVPZmZlcpSTlCmBlE59lCiMCm9mZmVyZWRf"
    "YnmUaA2MCGF1ZGllbmNllChoFGgRkZSMBGdpdmWUKEsBSwBLAEsASwB0lIwHcmVjZWl2ZZQoSwBLAEsA"
    "SwBLAXSUjAhnaXZlX2FueZRLAIwLcmVjZWl2ZV9hbnmUSwCMD3BhcmVudF9vZmZlcl9pZJROaAVoHowN"
    "Y3JlYXRlZF9yb3VuZJRLAIwKd2lsbGluZ19ieZSPlChoFGgRkIwLZGVjbGluZWRfYnmUj5SMBnN0YXR1"
    "c5RoAIwQVHJhZGVPZmZlclN0YXR1c5STlIwGYWN0aXZllIWUUpR1hpRijAlhLWNvdW50ZXKUaCApgZRO"
    "fZQoaCNoFGgkKGgNkZRoJihLAEsASwBLAEsCdJRoKGgnaCpLAGgrSwBoLGgeaAVoOWgtSwBoLo+UaDCPlGgy"
    "aDd1hpRidYwSc2VsZWN0ZWRfY2FuZGlkYXRllGgAjA5UcmFkZUNhbmRpZGF0ZZSTlCmBlF2UKGg5aA1o"
    "FGViaDJoAIwRVHJhZGVXaW5kb3dTdGF0dXOUk5SMBG9wZW6UhZRSlIwIY2FwX2hpdHOUSwCMEl9uZXh0"
    "X29mZmVyX251bWJlcpRLA4wLX3NlZW5fZGVhbHOUj5QoKGgLaBJoD4aUaCdLAGgpSwB0lChoEmgLaD1L"
    "AGgnSwB0lJB1hpRiLg=="
)


def test_historical_window_restores_deals_order_and_continuation() -> None:
    window: object = pickle.loads(base64.b64decode(HISTORICAL_WINDOW))
    assert isinstance(window, trading.TradeWindow)
    assert window.limits == trade.TradeLimits(max_active_root_offers=2, max_offers_per_player=2)
    assert (window.status, window.round) == (trade.TradeWindowStatus.OPEN, 0)
    assert list(window.offers) == ["z-root", "a-counter"]
    assert window.executable_candidates() == (
        trade.TradeCandidate("z-root", Color.RED, Color.WHITE),
        trade.TradeCandidate("z-root", Color.RED, Color.BLUE),
        trade.TradeCandidate("a-counter", Color.RED, Color.BLUE),
    )
    assert window.selected_candidate is not None
    assert window.selected_candidate == window.executable_candidates()[-1]
    assert window.selected_candidate.to_payload() == {
        "offer_id": "a-counter", "turn_player": "RED", "counterparty": "BLUE",
    }
    assert window._seen_deals == {
        ("RED", ("BLUE", "WHITE"), (1, 0, 0, 0, 0), 0, (0, 0, 0, 0, 1), 0),
        ("BLUE", "RED", (0, 0, 0, 0, 2), 0, (1, 0, 0, 0, 0), 0),
    }
    root = window.offers["z-root"]
    assert type(root) is trade.TradeOffer
    assert root.to_payload() == {
        "id": "z-root", "offered_by": "RED", "audience": ["BLUE", "WHITE"],
        "give": {"WOOD": 1}, "receive": {"ORE": 1}, "give_any": 0, "receive_any": 0,
        "parent_offer_id": None, "willing_by": ["BLUE", "WHITE"], "declined_by": [],
        "status": "active",
    }
    opposite = trade.TradeOffer(
        Color.RED, frozenset({Color.BLUE}), (1, 0, 0, 0, 0), (0, 0, 0, 0, 2),
    )
    with pytest.raises(ValueError, match="Equivalent offer"):
        window.create_offer(opposite)
    assert window.cap_hits == 0
    created = window.create_offer(opposite, allow_duplicate=True)
    assert created.id == "historical:o3"
    assert opposite.id is None
    window.mark_executed()
    assert window.offers["a-counter"].status is trade.TradeOfferStatus.EXECUTED
    assert root.status is trade.TradeOfferStatus.EXPIRED
    assert window.status is trade.TradeWindowStatus.CLOSED
    serialized = pickle.dumps(window, protocol=4)
    assert b"cle.game_engine.trading" in serialized
    assert b"cle.game_engine.models.trade" not in serialized
    restored: object = pickle.loads(serialized)
    assert isinstance(restored, trading.TradeWindow)
    assert restored == window
    assert restored.offers["z-root"].willing_by is not root.willing_by


@pytest.mark.parametrize("model_type", [
    trade.TradeOfferStatus,
    trade.TradeWindowStatus,
    trade.TradeLimits,
    trade.TradeOffer,
    trade.TradeCandidate,
])
def test_model_exports_keep_historical_pickle_identity(model_type: type[object]) -> None:
    assert getattr(trading, model_type.__name__) is model_type
    assert model_type.__module__ == "cle.game_engine.trading"
    assert pickle.loads(pickle.dumps(model_type)) is model_type
    assert trading.RESOURCE_NAMES is trade.RESOURCE_NAMES
    assert trading.ResourceBundle is trade.ResourceBundle
