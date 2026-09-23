"""Trade offer builder shared by the live game-logging tests."""


from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeOffer


def _offer(*, parent_offer_id: str | None = None) -> TradeOffer:
    return TradeOffer(
        offered_by=Color.RED,
        audience=frozenset({Color.BLUE, Color.WHITE, Color.ORANGE}),
        give=(0, 1, 0, 0, 0),
        receive=(1, 0, 0, 0, 0),
        parent_offer_id=parent_offer_id,
    )
