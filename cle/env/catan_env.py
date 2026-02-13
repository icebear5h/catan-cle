"""
PettingZoo AEC environment for Catan with LLM-compatible text-based interface.

This wraps the Catanatron game engine in a multi-agent RL interface for training
LLMs via PPO/DQN with LoRA fine-tuning.

Uses PettingZoo's Agent Environment Cycle (AEC) API for turn-based games.
"""

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from gymnasium import spaces
from typing import Any, Dict, List, Optional
from engine.game import Game
from engine.models.player import Player, Color
from engine.models.enums import Action
from cle.env.observation_formatter import (
    CatanObservationFormatter,
    create_observation_from_state,
)


def env(**kwargs):
    """
    Environment factory function for PettingZoo compatibility.

    Usage:
        from cle.env import catan_env
        env = catan_env.env()
    """
    env = raw_env(**kwargs)
    # Add wrappers for better compatibility
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(**kwargs):
    """Create raw CatanEnv without wrappers."""
    return CatanEnv(**kwargs)


class CatanEnv(AECEnv):
    """
    PettingZoo AEC environment for Settlers of Catan.

    Observation Space: Text-based semantic descriptions (following FLE pattern)
    Action Space: Text-based tool calls that map to Catanatron actions
    Reward: Game outcome + strategic milestones

    Multi-agent turn-based game where agents act sequentially.
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "catan_v0",
        "is_parallelizable": False,  # Turn-based, not parallel
    }

    def __init__(
        self,
        players: Optional[List[Player]] = None,
        num_players: int = 4,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize Catan environment.

        Args:
            players: Optional list of Player instances (will create random players if None)
            num_players: Number of players (2-4) if players not provided
            render_mode: Rendering mode for visualization
        """
        super().__init__()

        # Create players if not provided
        if players is None:
            from engine.models.player import SimplePlayer
            colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE][:num_players]
            players = [SimplePlayer(color) for color in colors]

        self.players = players
        self.num_players = len(players)
        self.render_mode = render_mode

        # PettingZoo required attributes
        self.possible_agents = [str(p.color) for p in self.players]
        self.agents = self.possible_agents[:]

        # Text-based spaces for LLM interaction
        # Using gymnasium.spaces.Text for flexibility
        # In practice, observations/actions are just strings
        self._observation_spaces = {
            agent: spaces.Text(max_length=10000) for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Text(max_length=1000) for agent in self.possible_agents
        }

        self.formatter = CatanObservationFormatter()
        self.game: Optional[Game] = None

        # Per-player cursor into state.actions for event history
        self._action_cursors: Dict[str, int] = {agent: 0 for agent in self.possible_agents}

        # Track VP for each agent for reward shaping
        self._last_vps: Dict[str, int] = {agent: 0 for agent in self.possible_agents}

        # PettingZoo state tracking
        self.agent_selection = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """
        Reset the environment for a new episode (new game).

        PettingZoo AEC environments don't return observations on reset.
        Call observe() after reset to get the first observation.
        """
        # Reset agents list
        self.agents = self.possible_agents[:]

        # Initialize new Catanatron game
        self.game = Game(self.players)

        # Reset rewards, terminations, truncations, infos
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Reset VP tracking
        self._last_vps = {agent: 0 for agent in self.agents}

        # Reset action cursors
        self._action_cursors = {agent: 0 for agent in self.agents}

        self.agent_selection = str(self.game.state.current_color())

    def step(self, action: Action):
        """
        Execute one step in the environment.

        Args:
            action: Catanatron Action object (parsing from text happens externally)

        PettingZoo AEC API: step() doesn't return anything.
        Use observe(), rewards, terminations, truncations, infos after step.
        """
        if self.game is None:
            raise RuntimeError("Must call reset() before step()")

        expected_agent = str(self.game.state.current_color())
        if self.agent_selection != expected_agent:
            self.agent_selection = expected_agent

        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            # Agent already terminated, just return without action
            return self._was_dead_step(action)

        # Execute action in Catanatron game
        self.game.execute(action)

        # Compute rewards for all agents
        self._compute_rewards()

        # Check if game is over
        winning_color = self.game.winning_color()
        if winning_color is not None:
            # Game over, terminate all agents
            for agent in self.agents:
                self.terminations[agent] = True

        # Update infos
        for agent in self.agents:
            self.infos[agent] = {
                "num_turns": self.game.state.num_turns,
                "current_player": str(self.game.state.current_color()),
            }

        self.agent_selection = str(self.game.state.current_color())

        # Accumulate rewards
        self._accumulate_rewards()

    def observe(self, agent: str) -> str:
        """
        Get observation for specified agent.

        Args:
            agent: Agent name (color string like "Color.RED")

        Returns:
            Semantic text observation from agent's perspective
        """
        if self.game is None:
            return ""

        # Convert agent name back to Color
        color = self._agent_to_color(agent)

        # Consume new actions since this agent's last observation
        cursor = self._action_cursors.get(agent, 0)
        new_actions = list(self.game.state.actions[cursor:])
        self._action_cursors[agent] = len(self.game.state.actions)

        # Convert state to structured observation
        obs = create_observation_from_state(self.game.state, color, recent_events=new_actions)

        # Format as semantic text
        formatted_obs = self.formatter.format(obs)

        return formatted_obs.raw_str

    def observation_space(self, agent: str):
        """Return observation space for specified agent."""
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        """Return action space for specified agent."""
        return self._action_spaces[agent]

    def _agent_to_color(self, agent: str) -> Color:
        """Convert agent name string to Color enum."""
        # Agent name is like "Color.RED", extract the color
        for player in self.players:
            if str(player.color) == agent:
                return player.color
        raise ValueError(f"Unknown agent: {agent}")

    def _compute_rewards(self):
        """
        Compute rewards for all agents.

        Reward structure:
        - +1 for each VP gained
        - +10 for winning
        - -10 for losing
        """
        from engine.state_functions import get_visible_victory_points

        winning_color = self.game.winning_color()

        for agent in self.agents:
            color = self._agent_to_color(agent)
            current_vp = get_visible_victory_points(self.game.state, color)
            vp_delta = current_vp - self._last_vps[agent]
            self._last_vps[agent] = current_vp

            reward = float(vp_delta)

            # Add win/loss bonus
            if winning_color is not None:
                if winning_color == color:
                    reward += 10.0
                else:
                    reward -= 10.0

            self.rewards[agent] = reward

    def render(self):
        """Render the current game state."""
        if self.game is None:
            return

        if self.render_mode == "human":
            # Print observation for current agent
            obs = self.observe(self.agent_selection)
            print(f"\n{'='*60}")
            print(f"Agent: {self.agent_selection}")
            print(f"{'='*60}")
            print(obs)
        elif self.render_mode == "ansi":
            return self.observe(self.agent_selection)

    def close(self):
        """Clean up resources."""
        self.game = None
        self.agents = []
