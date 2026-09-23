"""Historical checkpoint slots and the sandbox's stable pickle identity."""

import pickle
from dataclasses import replace

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action
from cle.game_engine.models.player import Color
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import PlayerContext
from cle.players.data import ContractSlot
from cle.sandbox import catan
from cle.sandbox.catan import CatanSandbox
from cle.sandbox.contracts import SandboxSnapshot, SandboxStepResult

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _sandbox() -> CatanSandbox:
    return CatanSandbox.create(
        COLORS, {color: FirstLegalPlayer(color) for color in COLORS},
        seed=7, shuffle_players=False,
    )


def test_historical_slots_supply_defaults_and_preserve_getstate_order() -> None:
    saved = _sandbox().snapshot()
    historical: list[ContractSlot] = [saved.engine, saved.player_states]
    restored = object.__new__(SandboxSnapshot)
    restored.__setstate__(historical)
    assert restored.__getstate__() == [*historical, None, False, None, (), None, None, None]
    assert restored.pending_action_batch is None
    assert restored.trade_preauthorization is None

    result = object.__new__(SandboxStepResult)
    result.__setstate__([(), (), ()])
    assert result.__getstate__() == [(), (), (), (), None]
    assert pickle.loads(pickle.dumps(result)) == result
    with pytest.raises(ValueError, match="field count"):
        restored.__setstate__([saved.engine])


@pytest.mark.asyncio
async def test_old_pickle_global_and_checkpoint_continue_identically() -> None:
    assert CatanSandbox.__module__ == "cle.sandbox.catan"
    assert pickle.loads(b"ccle.sandbox.catan\nCatanSandbox\n.") is CatanSandbox
    sandbox = _sandbox()
    await sandbox.step()
    loaded = pickle.loads(pickle.dumps(sandbox))
    assert isinstance(loaded, CatanSandbox)

    saved = sandbox.snapshot()
    historical = object.__new__(SandboxSnapshot)
    historical.__setstate__([saved.engine, saved.player_states])
    loaded.restore(historical)
    actual, expected = await loaded.step(), await sandbox.step()
    # Board maps use object identity; compare the committed continuation contract.
    assert actual.transitions == expected.transitions
    assert actual.attempts == expected.attempts
    assert actual.messages == expected.messages
    assert loaded.game_engine.rng.getstate() == sandbox.game_engine.rng.getstate()
    assert tuple(player.snapshot() for player in loaded.players.values()) == tuple(
        player.snapshot() for player in sandbox.players.values()
    )


@pytest.mark.parametrize("advertise", [False, True])
def test_public_context_helper_wrapper_is_used(
    monkeypatch: pytest.MonkeyPatch, advertise: bool,
) -> None:
    sandbox = _sandbox()
    original = catan.build_decision_context
    calls: list[tuple[GameEngine, Color | None, tuple[Action, ...] | None, int | None]] = []

    def wrapped(
        engine: GameEngine,
        actor: Color | None = None,
        advertised_actions: tuple[Action, ...] | None = None,
        *,
        context_revision: int | None = None,
        allow_terminal: bool = False,
    ) -> PlayerContext:
        calls.append((engine, actor, advertised_actions, context_revision))
        return replace(original(
            engine, actor, advertised_actions,
            context_revision=context_revision, allow_terminal=allow_terminal,
        ), prompt_key="wrapped-context")

    monkeypatch.setattr("cle.sandbox.catan.build_decision_context", wrapped)
    advertised = tuple(sandbox.game_engine.state.playable_actions[:1]) if advertise else None
    context = sandbox.decision_context(Color.RED, advertised)
    assert calls == [(sandbox.game_engine, Color.RED, advertised, sandbox.revision)]
    assert context.prompt_key == "wrapped-context"
    assert context.legal_actions == (advertised or tuple(sandbox.game_engine.state.playable_actions))
