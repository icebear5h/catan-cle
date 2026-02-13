"""
Supabase Database Client for Training Data Pipeline

Tables:
- replays: Raw Colonist replay data (JSONB)
- training_examples: Processed observation-action pairs

Setup:
1. Create Supabase project at https://supabase.com
2. Run the SQL schema (see schema.sql or bottom of this file)
3. Set environment variables:
   - SUPABASE_URL=https://xxx.supabase.co
   - SUPABASE_KEY=eyJ...
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Lazy-load supabase to avoid import errors when not configured
_client = None


def get_client():
    """Get or create Supabase client."""
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY environment variables required.\n"
                "Get these from your Supabase project settings > API."
            )

        from supabase import create_client
        _client = create_client(url, key)
        logger.info(f"Connected to Supabase: {url}")

    return _client


def is_configured() -> bool:
    """Check if Supabase is configured."""
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"))


# ============================================================================
# Replays
# ============================================================================

def save_replay(game_id: str, data: Dict[str, Any], metadata: Optional[Dict] = None) -> bool:
    """
    Save raw replay data to Supabase.

    Args:
        game_id: Colonist game ID
        data: Full API response
        metadata: Optional metadata (player_color, result, etc.)

    Returns:
        True if saved successfully
    """
    try:
        client = get_client()

        record = {
            "game_id": game_id,
            "data": data,
            "metadata": metadata or {},
            "scraped_at": datetime.utcnow().isoformat(),
        }

        result = client.table("replays").upsert(record).execute()
        logger.info(f"Saved replay {game_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to save replay {game_id}: {e}")
        return False


def get_replay(game_id: str) -> Optional[Dict[str, Any]]:
    """Get replay by game ID."""
    try:
        client = get_client()
        result = client.table("replays").select("*").eq("game_id", game_id).execute()

        if result.data:
            return result.data[0]
        return None

    except Exception as e:
        logger.error(f"Failed to get replay {game_id}: {e}")
        return None


def list_replays(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """List replays with pagination."""
    try:
        client = get_client()
        result = (
            client.table("replays")
            .select("game_id, metadata, scraped_at")
            .order("scraped_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return result.data

    except Exception as e:
        logger.error(f"Failed to list replays: {e}")
        return []


def count_replays() -> int:
    """Get total replay count."""
    try:
        client = get_client()
        result = client.table("replays").select("game_id", count="exact").execute()
        return result.count or 0
    except Exception as e:
        logger.error(f"Failed to count replays: {e}")
        return 0


# ============================================================================
# Training Examples
# ============================================================================

def save_training_examples(examples: List[Dict[str, Any]], batch_size: int = 100) -> int:
    """
    Batch insert training examples.

    Args:
        examples: List of TrainingExample.to_dict() results
        batch_size: Insert batch size

    Returns:
        Number of examples saved
    """
    try:
        client = get_client()
        saved = 0

        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            result = client.table("training_examples").insert(batch).execute()
            saved += len(result.data)

        logger.info(f"Saved {saved} training examples")
        return saved

    except Exception as e:
        logger.error(f"Failed to save training examples: {e}")
        return 0


def get_training_examples(
    action_type: Optional[str] = None,
    is_winner: Optional[bool] = None,
    phase: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Query training examples with filters.

    Args:
        action_type: Filter by action type (e.g., "BUILD_SETTLEMENT")
        is_winner: Filter by winner status
        phase: Filter by game phase ("initial_placement", "main_game")
        limit: Max results
        offset: Pagination offset

    Returns:
        List of training examples
    """
    try:
        client = get_client()
        query = client.table("training_examples").select("*")

        if action_type:
            query = query.eq("action_type", action_type)
        if is_winner is not None:
            query = query.eq("is_winner", is_winner)
        if phase:
            query = query.eq("phase", phase)

        result = query.range(offset, offset + limit - 1).execute()
        return result.data

    except Exception as e:
        logger.error(f"Failed to get training examples: {e}")
        return []


def count_training_examples(is_winner: Optional[bool] = None) -> int:
    """Get training example count."""
    try:
        client = get_client()
        query = client.table("training_examples").select("id", count="exact")

        if is_winner is not None:
            query = query.eq("is_winner", is_winner)

        result = query.execute()
        return result.count or 0

    except Exception as e:
        logger.error(f"Failed to count training examples: {e}")
        return 0


def get_training_stats() -> Dict[str, Any]:
    """Get summary statistics for training data."""
    try:
        client = get_client()

        # Total counts
        total = count_training_examples()
        winners = count_training_examples(is_winner=True)

        # Action type distribution
        action_types = {}
        for action in ["BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD", "ROLL",
                       "BANK_TRADE", "OFFER_TRADE", "MOVE_ROBBER", "BUY_DEV_CARD"]:
            result = (
                client.table("training_examples")
                .select("id", count="exact")
                .eq("action_type", action)
                .execute()
            )
            action_types[action] = result.count or 0

        return {
            "total_examples": total,
            "winner_examples": winners,
            "loser_examples": total - winners,
            "action_types": action_types,
            "replay_count": count_replays(),
        }

    except Exception as e:
        logger.error(f"Failed to get training stats: {e}")
        return {}


def delete_examples_for_game(game_id: str) -> int:
    """Delete all training examples for a game (for reprocessing)."""
    try:
        client = get_client()
        result = client.table("training_examples").delete().eq("game_id", game_id).execute()
        return len(result.data)
    except Exception as e:
        logger.error(f"Failed to delete examples for {game_id}: {e}")
        return 0


# ============================================================================
# Export
# ============================================================================

def export_to_jsonl(
    output_file: str,
    is_winner: Optional[bool] = None,
    action_types: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> int:
    """
    Export training examples to JSONL file.

    Args:
        output_file: Output path
        is_winner: Filter by winner status
        action_types: Filter by action types
        limit: Max examples to export

    Returns:
        Number of examples exported
    """
    import json

    try:
        client = get_client()

        offset = 0
        batch_size = 1000
        exported = 0

        with open(output_file, "w") as f:
            while True:
                query = client.table("training_examples").select("*")

                if is_winner is not None:
                    query = query.eq("is_winner", is_winner)
                if action_types:
                    query = query.in_("action_type", action_types)

                result = query.range(offset, offset + batch_size - 1).execute()

                if not result.data:
                    break

                for example in result.data:
                    # Remove internal fields
                    example.pop("id", None)
                    example.pop("created_at", None)
                    f.write(json.dumps(example) + "\n")
                    exported += 1

                if limit and exported >= limit:
                    break

                offset += batch_size

        logger.info(f"Exported {exported} examples to {output_file}")
        return exported

    except Exception as e:
        logger.error(f"Failed to export: {e}")
        return 0


# ============================================================================
# Schema (run this in Supabase SQL editor)
# ============================================================================

SCHEMA_SQL = """
-- Replays table
create table if not exists replays (
    game_id text primary key,
    data jsonb not null,
    metadata jsonb default '{}',
    scraped_at timestamptz default now()
);

-- Training examples table
create table if not exists training_examples (
    id serial primary key,
    game_id text references replays(game_id) on delete cascade,
    event_index int not null,
    turn int not null,
    phase text not null,
    player int not null,
    observation text not null,
    action text not null,
    action_type text not null,
    action_value jsonb,
    is_winner boolean default false,
    time_taken_seconds float,
    created_at timestamptz default now()
);

-- Indexes for common queries
create index if not exists idx_training_game on training_examples(game_id);
create index if not exists idx_training_winner on training_examples(is_winner);
create index if not exists idx_training_action_type on training_examples(action_type);
create index if not exists idx_training_phase on training_examples(phase);

-- Enable Row Level Security (optional, for production)
-- alter table replays enable row level security;
-- alter table training_examples enable row level security;
"""


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "schema":
        print("Run this SQL in your Supabase SQL editor:\n")
        print(SCHEMA_SQL)
    elif len(sys.argv) > 1 and sys.argv[1] == "stats":
        if not is_configured():
            print("Set SUPABASE_URL and SUPABASE_KEY environment variables")
            sys.exit(1)

        stats = get_training_stats()
        print(f"Replays: {stats.get('replay_count', 0)}")
        print(f"Training examples: {stats.get('total_examples', 0)}")
        print(f"  Winners: {stats.get('winner_examples', 0)}")
        print(f"  Losers: {stats.get('loser_examples', 0)}")
        print(f"Action types: {stats.get('action_types', {})}")
    else:
        print("Usage:")
        print("  python db.py schema  - Print SQL schema")
        print("  python db.py stats   - Show database stats")
