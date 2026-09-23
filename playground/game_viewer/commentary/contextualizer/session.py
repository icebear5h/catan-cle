"""Advancing the replay, but only behind a sealed provisional reading."""

import copy
import hashlib
import secrets
from collections.abc import Mapping
from typing import cast

from cle.replay.activity import format_visible_replay_activity
from cle.replay.colonist.types import ActionHint
from cle.replay.contracts import ParsedActions
from cle.replay.runtime.step_executor import replay_step_logic

from ..references import (
    CornerFact,
    build_corner_index,
    ground_text_references,
    load_colonist_corner_mapping,
)
from .evidence import CommentarySessionBase
from .models import (
    BlindContext,
    CommentaryContextError,
    CommentarySpan,
    CommitToken,
    RevealedEvent,
    _color_name,
    _engine_color_for_colonist,
    _PendingContext,
    _state_fingerprint,
    archive,
    engine_state,
)

__all__ = ["CausalCommentarySession"]


class CausalCommentarySession(CommentarySessionBase):
    """Advance a loaded replay only after an agent seals a provisional reading."""

    def _begin_locked(self) -> BlindContext:
        self._validate_loaded_replay()
        if self._pending is not None:
            raise CommentaryContextError(
                "The current context must be committed, revealed, or abandoned first"
            )

        game_state = engine_state(self._state)
        corner_index_key = (
            archive(self._state).get("game_id"),
            id(game_state.board.map),
        )
        if self._corner_index_key != corner_index_key:
            self._corner_index = build_corner_index(game_state.board.map)
            self._corner_index_key = corner_index_key

        narrator = self._narrator_metadata()
        narrator_colonist_color = cast(int | None, narrator.get("colonist_color"))
        narrator_engine_color = _engine_color_for_colonist(
            archive(self._state), game_state, narrator_colonist_color
        )
        current_player_color = _color_name(game_state.current_color())
        narrator_is_current_player = (
            narrator_engine_color is not None
            and narrator_engine_color == current_player_color
        )
        player_perspective = archive(self._state).get("player_perspective")
        include_legality = narrator_is_current_player and (
            game_state.is_initial_build_phase
            or (
                player_perspective is not None
                and str(player_perspective) == str(narrator_colonist_color)
            )
        )

        commentary = tuple(
            CommentarySpan(
                evidence_id=(
                    f"{archive(self._state).get('game_id')}:"
                    f"{self._state.replay_index}:"
                    f"{segment['source_start_index']}-{segment['source_end_index']}:"
                    f"{hashlib.sha256(str(segment['text']).encode('utf-8')).hexdigest()[:12]}"
                ),
                replay_index=self._state.replay_index,
                start_s=float(segment["start_s"]),
                end_s=float(segment["end_s"]),
                text=str(segment["text"]),
                source_start_index=int(segment["source_start_index"]),
                source_end_index=int(segment["source_end_index"]),
                source_segment_count=int(segment["source_segment_count"]),
                references=ground_text_references(
                    str(segment["text"]),
                    # Rebuilt above whenever the layout key changes.
                    cast(tuple[CornerFact, ...], self._corner_index),
                    game_state,
                    include_legality=include_legality,
                ),
            )
            for segment in self._select_evidence()
        )

        context = BlindContext(
            context_id=secrets.token_urlsafe(16),
            game_id=str(archive(self._state).get("game_id")),
            replay_index=self._state.replay_index,
            narrator_username=cast(str | None, narrator.get("username")),
            narrator_colonist_color=narrator_colonist_color,
            narrator_engine_color=narrator_engine_color,
            current_player_color=current_player_color,
            narrator_is_current_player=narrator_is_current_player,
            pairing_verified=bool(
                cast(
                    Mapping[str, object],
                    archive(self._state).get("paired_transcript") or {},
                ).get("verified", False)
            ),
            commentary=commentary,
        )
        self._pending = _PendingContext(
            context=context,
            cursor=self._state.replay_index,
            fingerprint=_state_fingerprint(self._state),
        )
        return context

    def begin(self) -> BlindContext:
        """Return current commentary and grounded candidates without the next action."""
        mutation_lock = getattr(self._state, "replay_mutation_lock", None)
        if mutation_lock is None:
            return self._begin_locked()
        with mutation_lock:
            return self._begin_locked()

    def _commit_locked(
        self, context_id: str, provisional_interpretation: object
    ) -> CommitToken:
        pending = self._pending
        if pending is None or pending.context.context_id != context_id:
            raise CommentaryContextError("Unknown or inactive commentary context")
        if pending.token is not None:
            raise CommentaryContextError("This commentary context is already committed")
        if (
            self._state.replay_index != pending.cursor
            or _state_fingerprint(self._state) != pending.fingerprint
        ):
            raise CommentaryContextError("Replay state changed before commitment")

        token_value = secrets.token_urlsafe(24)
        pending.provisional = copy.deepcopy(provisional_interpretation)
        pending.token = token_value
        self._tokens[token_value] = pending
        return CommitToken(token_value)

    def commit(self, context_id: str, provisional_interpretation: object) -> CommitToken:
        """Seal arbitrary agent output while the replay remains at the blind cursor."""
        mutation_lock = getattr(self._state, "replay_mutation_lock", None)
        if mutation_lock is None:
            return self._commit_locked(context_id, provisional_interpretation)
        with mutation_lock:
            return self._commit_locked(context_id, provisional_interpretation)

    def abandon(self, context_id: str) -> None:
        """Discard an unneeded context without advancing the replay."""
        mutation_lock = getattr(self._state, "replay_mutation_lock", None)
        if mutation_lock is None:
            self._abandon_locked(context_id)
            return
        with mutation_lock:
            self._abandon_locked(context_id)

    def _abandon_locked(self, context_id: str) -> None:
        if self._pending is None or self._pending.context.context_id != context_id:
            raise CommentaryContextError("Unknown or inactive commentary context")
        if self._pending.token is not None:
            self._tokens.pop(self._pending.token, None)
        self._pending = None

    def _reveal_one_locked(self, token: CommitToken) -> RevealedEvent:
        if token.value in self._used_tokens:
            raise CommentaryContextError("Commit token was already revealed")
        pending = self._tokens.get(token.value)
        if pending is None or pending is not self._pending:
            raise CommentaryContextError("Unknown or inactive commit token")
        if (
            self._state.replay_index != pending.cursor
            or _state_fingerprint(self._state) != pending.fingerprint
        ):
            raise CommentaryContextError("Replay state changed after commitment")

        replay_data = archive(self._state)
        game_state = engine_state(self._state)
        action_hint: ActionHint = cast(
            ParsedActions, replay_data["parsed_actions"]
        )[pending.cursor]
        actor_colonist_color = cast(int | None, action_hint.get("player"))
        actor_engine_color = _engine_color_for_colonist(
            replay_data, game_state, actor_colonist_color
        )
        public_summary = format_visible_replay_activity(
            action_hint,
            replay_data,
            game_state.colors,
            observer_color=None,
        )
        colonist_corner_id = action_hint.get("colonist_corner")
        engine_node_id: int | None
        engine_node_id = None
        if isinstance(colonist_corner_id, int):
            engine_node_id = load_colonist_corner_mapping().get(colonist_corner_id)

        result = replay_step_logic(
            self._state,
            lambda: None,
            allow_lookahead=False,
        )
        if isinstance(result, tuple):
            payload, status_code = result
            raise CommentaryContextError(
                f"Replay reveal failed ({status_code}): {payload.get('error', payload)}"
            )
        if self._state.replay_index != pending.cursor + 1:
            raise CommentaryContextError("Replay reveal did not advance exactly one row")

        revealed = RevealedEvent(
            game_id=pending.context.game_id,
            replay_index_before=pending.cursor,
            replay_index_after=self._state.replay_index,
            engine_status=str(result.get("status", "unknown")),
            action_type=str(action_hint.get("type", "UNKNOWN")),
            actor_colonist_color=actor_colonist_color,
            actor_engine_color=actor_engine_color,
            actor_username=self._actor_username(pending.cursor),
            actor_matches_narrator=(
                actor_engine_color is not None
                and pending.context.narrator_engine_color is not None
                and actor_engine_color == pending.context.narrator_engine_color
            ),
            public_summary=public_summary,
            colonist_corner_id=colonist_corner_id,
            engine_node_id=engine_node_id,
            colonist_edge_id=action_hint.get("colonist_edge"),
            colonist_tile_id=cast(int | None, action_hint.get("colonist_tile")),
            compatible_reference_surfaces=self._compatible_surfaces(
                pending.context, colonist_corner_id
            ),
            provisional_interpretation=copy.deepcopy(pending.provisional),
        )
        self._used_tokens.add(token.value)
        self._tokens.pop(token.value, None)
        self._pending = None
        return revealed

    def reveal_one(self, token: CommitToken) -> RevealedEvent:
        """Advance exactly one parsed row and reveal only public confirmation evidence."""
        mutation_lock = getattr(self._state, "replay_mutation_lock", None)
        if mutation_lock is None:
            return self._reveal_one_locked(token)
        with mutation_lock:
            return self._reveal_one_locked(token)
