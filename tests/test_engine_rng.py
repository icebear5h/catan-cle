import random

from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import ActionType
from cle.game_engine.models.player import Color


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _game(seed):
    return GameEngine(
        list(COLORS),
        seed=seed,
        shuffle_players=False,
        capture_history=True,
    )


def _advance_initial_placement(game):
    while game.state.is_initial_build_phase:
        game.step(game.state.playable_actions[0])


def _roll_action(game):
    return next(
        action
        for action in game.state.playable_actions
        if action.action_type == ActionType.ROLL
    )


def _board_fingerprint(game):
    return tuple(
        sorted(
            (
                coordinate,
                tile.resource,
                tile.number,
            )
            for coordinate, tile in game.state.board.map.land_tiles.items()
        )
    )


def test_game_does_not_mutate_process_global_random_state():
    random.seed(90210)
    global_state = random.getstate()

    game = _game(41)
    _advance_initial_placement(game)
    game.step(_roll_action(game))

    assert random.getstate() == global_state


def test_same_seed_games_match_when_another_game_is_interleaved():
    first = _game(77)
    unrelated = _game(991)
    second = _game(77)

    assert first.state.colors == second.state.colors
    assert _board_fingerprint(first) == _board_fingerprint(second)
    assert first.state.development_listdeck == second.state.development_listdeck

    while first.state.is_initial_build_phase:
        assert first.state.playable_actions == second.state.playable_actions
        first.step(first.state.playable_actions[0])
        if unrelated.state.is_initial_build_phase:
            unrelated.step(unrelated.state.playable_actions[0])
        second.step(second.state.playable_actions[0])

    first_roll = first.step(_roll_action(first))
    unrelated.step(_roll_action(unrelated))
    second_roll = second.step(_roll_action(second))

    assert first_roll == second_roll
    assert first.state.rng.getstate() == second.state.rng.getstate()


def test_game_copy_has_an_independent_identical_rng_continuation():
    game = _game(1234)
    _advance_initial_placement(game)
    branch = game.copy()

    assert branch.rng is branch.state.rng
    assert branch.rng is not game.rng
    assert branch.rng.getstate() == game.rng.getstate()
    assert branch.state.playable_actions is not game.state.playable_actions

    original_roll = game.step(_roll_action(game))
    branch_roll = branch.step(_roll_action(branch))

    assert original_roll == branch_roll
    assert branch.rng.getstate() == game.rng.getstate()


def test_undo_restores_rng_so_repeating_action_repeats_outcome():
    game = _game(2026)
    _advance_initial_placement(game)
    roll = _roll_action(game)

    first_result = game.step(roll)
    game.undo()
    repeated_result = game.step(roll)

    assert repeated_result == first_result
