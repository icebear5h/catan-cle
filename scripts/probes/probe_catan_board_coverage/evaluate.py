"""Probe joining, per-token scoring, and coverage rollups."""

from __future__ import annotations

from collections import defaultdict

from scripts.probes.probe_catan_board_coverage.inventory import question_to_targets
from scripts.probes.probe_catan_board_coverage.model import (
    ROBBER_PROBE_TYPES,
    CoverageMetrics,
    JsonDict,
    PartPayload,
    PartSummary,
    ProbeEntry,
    ProbeResult,
    canonical_category,
    empty_uncovered,
    filter_probe_map,
)

ProbeMap = dict[str, dict[str, list[ProbeEntry]]]

__all__ = [
    "ProbeMap",
    "evaluate_part_token",
    "summarize_part_payloads",
    "summarize_probe",
    "uncovered_for_part",
]


def _response_for(
    responses: dict[str, JsonDict] | None, question_id: str | None
) -> JsonDict | None:
    if responses is None or not question_id:
        return None
    return responses.get(question_id)


def summarize_probe(
    records: list[JsonDict], responses: dict[str, JsonDict] | None = None
) -> ProbeMap:
    """Summarize QA coverage keyed by (part type, part token, probe kind)."""

    by_part: dict[tuple[str, str], dict[str, list[ProbeEntry]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for qa in records:
        raw_id = qa.get("id")
        question_id = str(raw_id) if isinstance(raw_id, str) else None
        response_info = _response_for(responses, question_id)

        score_payload = response_info.get("score", {}) if response_info else None
        if not isinstance(score_payload, dict):
            score_payload = {}
        scored_correct = (
            None if response_info is None else bool(score_payload.get("correct", False))
        )
        had_error = bool(response_info.get("error")) if response_info is not None else False

        entry = ProbeEntry(
            question_id=raw_id,
            category=qa.get("category"),
            canonical_category=canonical_category(str(qa.get("category", ""))),
            answer=qa.get("answer"),
            question=qa.get("question"),
            sample_id=qa.get("sample_id"),
            response=response_info.get("response") if response_info else None,
            correct=scored_correct,
            error=had_error,
        )

        for part_kind, token, probe_kind in question_to_targets(qa):
            by_part[(part_kind, token)][probe_kind].append(entry)

    # keep dict keys stringified as "kind:token" for compact joining by caller
    return {f"{kind}:{token}": dict(probes) for (kind, token), probes in by_part.items()}


def evaluate_part_token(
    probes_by_kind: dict[str, list[ProbeEntry]],
) -> dict[str, ProbeResult]:
    result: dict[str, ProbeResult] = {}

    for probe_kind, records in sorted(probes_by_kind.items()):
        attempted = sum(1 for row in records if row["correct"] is not None)
        correct = sum(1 for row in records if row["correct"] is True)
        incorrect = sum(1 for row in records if row["correct"] is False)
        errors = sum(1 for row in records if row["error"])

        status = "untested"
        if attempted:
            if incorrect:
                status = "partial" if correct else "failed"
            else:
                status = "passed"

        result[probe_kind] = ProbeResult(
            attempted=attempted,
            correct=correct,
            incorrect=incorrect,
            errors=errors,
            question_ids=[row["question_id"] for row in records],
            status=status,
        )

    return result


def summarize_part_payloads(
    part: str,
    part_payloads: dict[str, PartPayload],
    probe_filter: set[str] | None = None,
) -> PartSummary:
    total_parts = len(part_payloads)

    coverage_by_probe: defaultdict[str, CoverageMetrics] = defaultdict(
        lambda: CoverageMetrics(tested=0, passed=0, failed=0)
    )

    for payload in part_payloads.values():
        probes = filter_probe_map(payload["probes"], probe_filter)
        for probe_kind, probe_info in probes.items():
            if probe_info["attempted"]:
                coverage_by_probe[probe_kind]["tested"] += 1
                if probe_info["status"] == "passed":
                    coverage_by_probe[probe_kind]["passed"] += 1
                else:
                    coverage_by_probe[probe_kind]["failed"] += 1

    # Keep robber rows explicit even when absent, because they are diagnostic anchors.
    if part == "robber":
        for probe_kind in ROBBER_PROBE_TYPES:
            coverage_by_probe.setdefault(
                probe_kind, CoverageMetrics(tested=0, passed=0, failed=0)
            )

    return PartSummary(
        total=total_parts,
        coverage={
            probe_kind: metrics
            for probe_kind, metrics in sorted(coverage_by_probe.items())
        },
    )


def uncovered_for_part(
    part: str, part_payloads: dict[str, PartPayload], tile_mode: str
) -> dict[str, list[str]]:
    uncovered = empty_uncovered()

    if part == "tile":
        for tile_token, payload in part_payloads.items():
            probes = payload["probes"]
            if tile_mode in {"all", "values"} and not probes.get("tile_resource_number"):
                uncovered["tile_resource_number"].append(tile_token)
            if tile_mode in {"all", "identity"} and not probes.get("tile_has_robber"):
                uncovered["tile_has_robber"].append(tile_token)
            if tile_mode in {"all", "identity"} and not probes.get("tile_occupied_nodes"):
                uncovered["tile_occupied_nodes"].append(tile_token)
        # prune tile buckets not requested for this mode
        if tile_mode == "values":
            uncovered.pop("tile_occupied_nodes", None)
            uncovered.pop("tile_has_robber", None)
        elif tile_mode == "identity":
            uncovered.pop("tile_resource_number", None)
        return uncovered

    if part == "node":
        for node_token, payload in part_payloads.items():
            if not payload["probes"].get("node_occupancy"):
                uncovered["node_occupancy"].append(node_token)
        return uncovered

    if part == "edge":
        for edge_token, payload in part_payloads.items():
            if not payload["probes"].get("edge_road_owner"):
                uncovered["edge_road_owner"].append(edge_token)
        uncovered.pop("tile_resource_number", None)
        uncovered.pop("tile_has_robber", None)
        uncovered.pop("tile_occupied_nodes", None)
        return uncovered

    if part == "port":
        for port_token, payload in part_payloads.items():
            probes = payload["probes"]
            if not probes.get("port_trade_type"):
                uncovered["port_trade_type"].append(port_token)
            if not probes.get("port_type_nodes"):
                uncovered["port_type_nodes"].append(port_token)
            if not probes.get("port_occupancy"):
                uncovered["port_occupancy"].append(port_token)
        uncovered.pop("tile_resource_number", None)
        uncovered.pop("tile_has_robber", None)
        uncovered.pop("tile_occupied_nodes", None)
        return uncovered

    if part == "robber":
        robber_token = next(iter(part_payloads.keys()), None)
        if robber_token and not part_payloads[robber_token]["probes"]:
            uncovered["robber_tile"].append(robber_token)
        return uncovered

    return uncovered
