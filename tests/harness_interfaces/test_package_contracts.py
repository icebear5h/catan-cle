"""Small compatibility captures from the original single-file harness modules."""

import hashlib
import json
import pickle

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness import action_tools, board_surface
from cle.harness.action_tools import arguments, definitions, parser, rendering, trades
from cle.harness.board_surface import contracts, provider, redaction, serialization
from cle.harness.catan_board_surface import ImageBoardPresenter, IndexedTileRowsBoardPresenter
from cle.players.contracts import PlayerContext


def test_board_dataclass_pickles_match_before_conversion() -> None:
    # Same actual seeded board/context as the pre-conversion action-tools capture.
    engine = GameEngine((Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE), seed=9,
                        shuffle_players=False)
    context = PlayerContext(
        context_id="semantic-tools:test", actor=Color.RED,
        turn_number=engine.state.num_turns, phase="main_game",
        observation=engine.observe(Color.RED), events=(),
        legal_actions=tuple(engine.state.playable_actions), prompt_key="main_game",
    )
    text = IndexedTileRowsBoardPresenter().present(context)
    image = ImageBoardPresenter(image_size=512).present(context)
    # Protocol-4 captures bind original module identity, slots/field order,
    # additive defaults, provenance, and exact text/image bytes together.
    captured = (
        (text.provenance, "0e6da04678a837462fb32ac5e2d0751c63bf67e5eb3536809b7b5a4d80dccb8c"),
        (text, "f6edaa8150b3565f6e9f71e1294ce03c665016f066968f21eb358df197a14f83"),
        (image, "567e65b898bae522c114f64b2cc460f7bfe12207782000ad8a25f8327fb273d7"),
    )
    for presentation, expected in captured:
        assert type(presentation).__module__ == "cle.harness.board_surface"
        stored = pickle.dumps(presentation, protocol=4)
        assert hashlib.sha256(stored).hexdigest() == expected
        assert pickle.loads(stored) == presentation


def test_legacy_exports_are_the_package_implementations() -> None:
    action_exports = (
        (arguments, ("_bundle", "_card_counts", "_color", "_offer_id", "_require_arguments",
                     "_resource", "_spatial_values")),
        (definitions, ("_TOOL_TYPES", "_REVERSE_TOOL_TYPES", "SHARED_ACTION_TOOLS")),
        (parser, ("parse_tool_choice",)),
        (rendering, ("render_action_tools", "legal_tool_names", "render_shared_legal_actions",
                     "trade_responder_note")),
        (trades, ("_counter_parent", "_semantic_offer", "_shared_trade_arguments")),
    )
    board_exports = (
        (contracts, ("BOARD_PRESENTATION_SCHEMA", "MAX_BOARD_IMAGE_BYTES",
                     "MAX_BOARD_IMAGE_DIMENSION", "BoardIdentitySpace", "BoardMediaType",
                     "BoardPresentation", "BoardPresentationProvenance", "BoardPresenter",
                     "TextBoardPresentation", "ImageBoardPresentation", "_SHA256",
                     "_sha256_bytes", "_require_sha256")),
        (provider, ("openai_messages_with_board",)),
        (serialization, ("board_presentation_payload",)),
        (redaction, ("sanitize_provider_payload",)),
    )
    for package, exports in ((action_tools, action_exports), (board_surface, board_exports)):
        for module, names in exports:
            for name in names:
                assert getattr(package, name) is getattr(module, name)
    assert hashlib.sha256(json.dumps(action_tools.SHARED_ACTION_TOOLS).encode()).hexdigest() == (
        # Literal text capture, independent of whitespace in the Python source.
        "35e68c5f0a37231da76fe8011b939e83682d552ab373b22fe786874fc58e4937"
    )


def test_redaction_preserves_tuple_keys_and_arbitrary_metadata() -> None:
    marker = object()
    value = {"nested": (marker, b"bytes", ["data:image/png;base64,a"], {3: "data:image/webp;x"})}
    cleaned = board_surface.sanitize_provider_payload(value)
    assert isinstance(cleaned["nested"], tuple)
    assert cleaned["nested"][0] is marker
    assert cleaned["nested"][1:] == (
        b"bytes", ["local-board-image://redacted"], {3: "local-board-image://redacted"},
    )
    assert value["nested"][2] == ["data:image/png;base64,a"]
