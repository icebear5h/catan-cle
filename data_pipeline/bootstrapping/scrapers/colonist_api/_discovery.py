"""Interactive endpoint probe used when Colonist changes its API surface."""

from data_pipeline.bootstrapping.scrapers.colonist_api._client import ColonistAPI


async def discover_api_endpoints(jwt_token: str | None = None) -> None:
    """
    Helper to discover API endpoints by testing common patterns.
    Run this to find the correct endpoints for your Colonist instance.

    Args:
        jwt_token: Your JWT token from jwt_colonist.io cookie
    """
    async with ColonistAPI(jwt_token=jwt_token) as api:
        # Test leaderboard tabs
        print("Testing /api/leaderboards-tabs/...")
        try:
            tabs = await api.get_leaderboard_tabs()
            print(f"  Found tabs: {tabs}")
        except Exception as e:
            print(f"  Failed: {e}")

        # Test leaderboard
        print("\nTesting leaderboard endpoints...")
        top_player = None
        for leaderboard_type in ["Classic4P", "CitiesAndKnights4P"]:
            try:
                entries = await api.get_leaderboard(leaderboard_type, start=1, end=5)
                print(f"  {leaderboard_type}: {len(entries)} entries")
                if entries:
                    print(f"    Top player: {entries[0].username} (rating: {entries[0].rating})")
                    if not top_player:
                        top_player = entries[0].username
            except Exception as e:
                print(f"  {leaderboard_type}: Failed - {e}")

        # Test game history
        if top_player:
            print(f"\nTesting game history for {top_player}...")
            try:
                games = await api.get_player_games(top_player, limit=5)
                print(f"  Found {len(games)} games")
                for g in games[:3]:
                    print(f"    Game {g.game_id}: {g.mode} - {g.result}")
                    print(f"      Replay: {g.replay_url}")
            except Exception as e:
                print(f"  Failed: {e}")


__all__ = ["discover_api_endpoints"]
