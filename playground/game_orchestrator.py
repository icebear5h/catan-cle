"""GameEngine orchestrator for running Catan with LLM agents and event-driven memory updates."""

from typing import List, Dict, Any, Optional
import sys
import os

# Add catanatron to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../catanatron/catanatron"))

from game_engine.game import GameEngine
from game_engine.models.player import Color
from cle.players.legacy import Player
from game_engine.models.enums import Action

from ..agents.llm_agent_impl import StrategicLLMAgent


class GameOrchestrator:
    """Orchestrates Catan games with LLM agents.

    Handles:
    - GameEngine initialization with N agents
    - Event-driven memory updates (after every action)
    - Trajectory collection for training
    - GameEngine result extraction
    """

    def __init__(self, num_players: int = 4, agent_config: Optional[Dict[str, Any]] = None):
        """Initialize orchestrator.

        Args:
            num_players: Number of players (2-6, default 4)
            agent_config: Configuration for LLM agents
        """
        if num_players < 2 or num_players > 6:
            raise ValueError("Number of players must be between 2 and 6")

        self.num_players = num_players
        self.agent_config = agent_config or {}

        # Color assignments
        self.colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE, Color.BROWN, Color.GREEN][:num_players]

        # Agents
        self.agents: List[StrategicLLMAgent] = []

        # GameEngine instance
        self.game: Optional[GameEngine] = None

    def initialize_game(self, seed: Optional[int] = None) -> GameEngine:
        """Initialize a new game with LLM agents.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Initialized GameEngine instance
        """
        # Create LLM agents
        self.agents = [
            StrategicLLMAgent(
                agent_id=i,
                color=color.value,
                config=self.agent_config
            )
            for i, color in enumerate(self.colors)
        ]

        # Create Catanatron player wrappers
        # TODO: Create adapter that bridges StrategicLLMAgent to Catanatron Player interface
        players = self._create_catanatron_players()

        # Initialize game
        self.game = GameEngine(
            colors=[player.color for player in players],
            seed=seed
        )

        return self.game

    def run_game(self) -> Dict[str, Any]:
        """Run a complete game with event-driven memory updates.

        Returns:
            GameEngine results with trajectories and agent memories
        """
        if not self.game:
            raise RuntimeError("GameEngine not initialized. Call initialize_game() first.")

        print(f"Starting game with {self.num_players} LLM agents...")

        # Phase 1: Initial placement (strategic foundation)
        print("\n=== INITIAL PLACEMENT PHASE ===")
        self._run_initial_placement()

        # Phase 2: Main game loop
        print("\n=== MAIN GAME ===")
        turn = 0
        while not self.game.winning_color():
            turn += 1
            print(f"\nTurn {turn}")

            # Get current player
            current_color = self.game.state.current_color()
            current_agent = self._get_agent_by_color(current_color.value)

            print(f"  {current_color.value}'s turn")

            # Agent selects action
            observation = self._get_observation(current_color)
            action = current_agent.select_action(observation)

            print(f"    Action: {action}")

            # Execute action in game
            self.game.step(action)

            # EVENT TRIGGER: Update ALL agents after action
            self._trigger_action_event(current_agent, action)

            # Safety: Stop if too many turns
            if turn > 500:
                print("Turn limit reached, ending game.")
                break

        # GameEngine over
        winner = self.game.winning_color()
        print(f"\n=== GAME OVER ===")
        print(f"Winner: {winner.value if winner else 'None (turn limit)'}")

        # Extract results
        results = self._extract_results()

        return results

    def _run_initial_placement(self):
        """Run initial placement phase with strategic planning."""
        # TODO: Implement initial placement with LLM strategic planning
        # This should call agent.set_initial_strategy() for each agent

        for agent in self.agents:
            print(f"\n{agent.color} placing initial settlements...")

            # Get available positions from game state
            available_positions = self._get_available_positions()

            # Agent plans strategy and chooses placements
            placement_result = agent.set_initial_strategy(
                game_state=self.game.state,
                available_positions=available_positions
            )

            print(f"  Strategy: {placement_result['initial_strategy']}")
            print(f"  Placements: {placement_result['placements']}")
            print(f"  Initial todos: {len(placement_result['todos'])} tasks")

            # Apply placements to game
            # TODO: Convert placement_result to actual game actions and execute

    def _trigger_action_event(self, acting_agent: StrategicLLMAgent, action: Action):
        """Trigger memory updates for ALL agents after an action.

        Args:
            acting_agent: The agent that took the action
            action: The action taken
        """
        # 1. Acting agent updates their own plan
        acting_agent.update_my_plan(action, self.game.state)

        # 2. ALL other agents observe and update opponent models
        for agent in self.agents:
            if agent.color != acting_agent.color:
                agent.observe_opponent_action(
                    opponent_color=acting_agent.color,
                    action=action,
                    game_state=self.game.state
                )

    def _extract_results(self) -> Dict[str, Any]:
        """Extract game results and agent memories for training.

        Returns:
            Dict with:
                - winner: Winning color
                - num_turns: Number of turns
                - agent_memories: Each agent's final memory state
                - trajectories: Action trajectories for fine-tuning
        """
        winner = self.game.winning_color()

        results = {
            "winner": winner.value if winner else None,
            "num_turns": self.game.state.num_turns if hasattr(self.game.state, 'num_turns') else 0,
            "agent_memories": {},
            "trajectories": {},
        }

        # Extract each agent's memory
        for agent in self.agents:
            if agent.memory:
                # Serialize memory
                results["agent_memories"][agent.color] = agent.memory.model_dump()

        return results

    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save game results to file.

        Args:
            results: Results from run_game()
            filepath: Path to save JSON results
        """
        import json

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results saved to {filepath}")

    # ========== Helper Methods ==========

    def _create_catanatron_players(self) -> List[Player]:
        """Create Catanatron Player instances that wrap LLM agents.

        TODO: Implement adapter pattern to bridge StrategicLLMAgent to Catanatron Player
        """
        # Placeholder - needs actual implementation
        return []

    def _get_agent_by_color(self, color: str) -> StrategicLLMAgent:
        """Get agent by color."""
        for agent in self.agents:
            if agent.color == color:
                return agent
        raise ValueError(f"No agent found with color {color}")

    def _get_observation(self, color: Color) -> Any:
        """Get observation for a player.

        TODO: Convert game state to Observation object
        """
        # Placeholder
        return None

    def _get_available_positions(self) -> List[Any]:
        """Get available settlement positions.

        TODO: Extract from game state
        """
        # Placeholder
        return []


def main():
    """Example usage."""
    # Configuration
    config = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 2000,
        "temperature": 1.0,
    }

    # Create orchestrator
    orchestrator = GameOrchestrator(num_players=4, agent_config=config)

    # Initialize and run game
    orchestrator.initialize_game(seed=42)
    results = orchestrator.run_game()

    # Save results
    orchestrator.save_results(results, "game_results.json")

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Winner: {results['winner']}")
    print(f"Turns: {results['num_turns']}")


if __name__ == "__main__":
    main()
