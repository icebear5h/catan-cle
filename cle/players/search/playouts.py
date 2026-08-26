import time
import random
import multiprocessing
from collections import Counter

from game_engine.game import GameEngine
from cle.players.legacy import Player

DEFAULT_NUM_PLAYOUTS = 25
USE_MULTIPROCESSING = True
NUM_WORKERS = multiprocessing.cpu_count()

PLAYOUTS_BUDGET = 100


# Single threaded NUM_PLAYOUTS=25 takes ~185.3893163204193 secs on initial placement
#   10.498431205749512 secs to do initial road (3 playable actions)
# Multithreaded, dividing the NUM_PLAYOUTS only (actions serially), takes ~52.22048330307007 secs
#   on intial placement. 4.187309980392456 secs on initial road.
# Multithreaded, on different actions
class GreedyPlayoutsPlayer(Player):
    """For each playable action, play N random playouts."""

    def __init__(self, color, num_playouts=DEFAULT_NUM_PLAYOUTS):
        super().__init__(color)
        self.num_playouts = int(num_playouts)

    def decide(self, game: GameEngine, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        start = time.time()
        # num_playouts = PLAYOUTS_BUDGET // len(playable_actions)
        num_playouts = self.num_playouts

        best_action = None
        max_wins = None
        for action in playable_actions:
            action_applied_game_copy = game.copy()
            action_applied_game_copy.step(action)

            counter = run_playouts(action_applied_game_copy, num_playouts)

            wins = counter[self.color]
            if max_wins is None or wins > max_wins:
                best_action = action
                max_wins = wins

        print(
            f"Greedy took {time.time() - start} secs to decide "
            + f"{len(playable_actions)} at {num_playouts} per action"
        )
        return best_action


def run_playouts(action_applied_game_copy, num_playouts):
    params = []
    for _ in range(num_playouts):
        params.append(action_applied_game_copy)
    if USE_MULTIPROCESSING:
        with multiprocessing.Pool(NUM_WORKERS) as p:
            counter = Counter(p.map(run_playout, params))
    else:
        counter = Counter(map(run_playout, params))
    return counter


def run_playout(action_applied_game_copy):
    game_copy = action_applied_game_copy.copy()
    while game_copy.winning_color() is None and game_copy.state.num_turns < 1000:
        game_copy.step(decide_fn(game_copy.state.playable_actions))
    return game_copy.winning_color()


def decide_fn(playable_actions):
    index = random.randrange(0, len(playable_actions))
    return playable_actions[index]
