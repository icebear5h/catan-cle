"""Packet segmentation of global transcript evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Dict, List, Sequence, Tuple, TypedDict

from cle.replay.contracts import as_mapping
from evals.json_types import JsonDict, as_dict, as_int, as_number
from evals.transcript_observation_assembly.anchors import _availability_cursor
from evals.transcript_observation_assembly.job import ObservationAssemblyError
from evals.transcript_observation_assembly.protocol import MAX_PACKET_SECONDS, MAX_PACKET_UTTERANCES
from evals.transcript_reasoning.support import json_value
from playground.game_viewer.replay.transcript import parse_transcript_segments


class PacketAnchor(TypedDict):
    """One causal cursor that closes a packet, and why it is an anchor."""

    replay_index: int
    anchor_kind: str
    decision_ids: List[str]


class PacketSpec(TypedDict):
    """The evidence one packet owns between its previous and current anchor."""

    packet_index: int
    replay_index: int
    previous_replay_index: int
    anchor_kind: str
    decision_ids: Tuple[str, ...]
    utterances: Tuple[JsonDict, ...]


def _mappings(value: object, label: str) -> List[Mapping[str, object]]:
    if not isinstance(value, list):
        raise ObservationAssemblyError(f"{label} must be a list")
    return [as_mapping(item, f"{label} entry") for item in value]


def _seconds(item: JsonDict, field: str) -> float:
    return float(as_number(item[field], f"evidence {field}"))


def _available_index(item: JsonDict) -> int:
    return as_int(item["available_replay_index"], "available_replay_index")


def build_global_evidence(paired_transcript: Mapping[str, object]) -> Tuple[JsonDict, ...]:
    """Reflow the full transcript once and assign each utterance causally."""
    timings = _mappings(
        paired_transcript.get("action_timings", []), "Transcript action timings"
    )
    utterances = parse_transcript_segments(
        _mappings(paired_transcript.get("segments", []), "Transcript segments")
    )
    evidence: List[JsonDict] = []
    for index, utterance in enumerate(utterances):
        fields = as_dict(json_value(utterance, "utterance"), "utterance")
        evidence.append(
            {
                "evidence_id": f"e{index:04d}",
                **fields,
                "available_replay_index": _availability_cursor(
                    _seconds(fields, "end_s"), timings
                ),
            }
        )
    return tuple(evidence)


def _packet_exceeds_limit(items: Sequence[JsonDict]) -> bool:
    if len(items) >= MAX_PACKET_UTTERANCES:
        return True
    if not items:
        return False
    return _seconds(items[-1], "end_s") - _seconds(items[0], "start_s") >= MAX_PACKET_SECONDS


def build_packet_anchors(
    evidence: Sequence[JsonDict],
    decision_ids_by_cursor: Dict[int, Tuple[str, ...]],
    total_events: int,
) -> Tuple[PacketAnchor, ...]:
    """Add bounded public-observation checkpoints between narrator decisions."""
    if total_events < 0:
        raise ValueError("total_events must be nonnegative")
    required = sorted({*decision_ids_by_cursor, total_events})
    if any(cursor < 0 or cursor > total_events for cursor in required):
        raise ObservationAssemblyError("Decision anchor is outside the replay")

    anchors = set(required)
    previous_required = -1
    for required_cursor in required:
        interval_items = [
            item
            for item in evidence
            if previous_required < _available_index(item) <= required_cursor
        ]
        grouped: Dict[int, List[JsonDict]] = defaultdict(list)
        for item in interval_items:
            grouped[_available_index(item)].append(item)
        packet_items: List[JsonDict] = []
        for cursor in sorted(grouped):
            packet_items.extend(grouped[cursor])
            if cursor < required_cursor and _packet_exceeds_limit(packet_items):
                anchors.add(cursor)
                packet_items = []
        previous_required = required_cursor

    records: List[PacketAnchor] = []
    for cursor in sorted(anchors):
        decision_ids = decision_ids_by_cursor.get(cursor, ())
        if decision_ids:
            kind = "decision"
        elif cursor == total_events:
            kind = "complete"
        else:
            kind = "observation"
        records.append(
            {
                "replay_index": cursor,
                "anchor_kind": kind,
                "decision_ids": list(decision_ids),
            }
        )
    return tuple(records)


def build_packet_specs(
    paired_transcript: Mapping[str, object],
    decision_ids_by_cursor: Dict[int, Tuple[str, ...]],
    total_events: int,
) -> Tuple[PacketSpec, ...]:
    """Partition globally reflowed evidence exactly once across causal anchors."""
    evidence = build_global_evidence(paired_transcript)
    anchors = build_packet_anchors(evidence, decision_ids_by_cursor, total_events)
    specs: List[PacketSpec] = []
    previous_cursor = -1
    for packet_index, anchor in enumerate(anchors):
        replay_index = anchor["replay_index"]
        owned = tuple(
            item
            for item in evidence
            if previous_cursor < _available_index(item) <= replay_index
        )
        specs.append(
            {
                "packet_index": packet_index,
                "replay_index": replay_index,
                "previous_replay_index": previous_cursor,
                "anchor_kind": anchor["anchor_kind"],
                "decision_ids": tuple(anchor["decision_ids"]),
                "utterances": owned,
            }
        )
        previous_cursor = replay_index

    assigned_ids = [
        item["evidence_id"] for spec in specs for item in spec["utterances"]
    ]
    expected_ids = [item["evidence_id"] for item in evidence]
    if assigned_ids != expected_ids:
        raise ObservationAssemblyError("Packet anchors did not partition evidence exactly")
    return tuple(specs)

