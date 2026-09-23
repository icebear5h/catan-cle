"""Report assembly, part filtering, and the text rendering."""

from __future__ import annotations

from pathlib import Path

from scripts.probes.probe_catan_board_coverage.evaluate import (
    evaluate_part_token,
    summarize_part_payloads,
    summarize_probe,
    uncovered_for_part,
)
from scripts.probes.probe_catan_board_coverage.inventory import build_expected_inventory
from scripts.probes.probe_catan_board_coverage.model import (
    TILE_MODE_PROBES,
    GeneratedFrom,
    JsonDict,
    PartPayload,
    PartSummary,
    Report,
    empty_uncovered,
    filter_probe_map,
)

PART_TO_KIND = {
    "tiles": "tile",
    "nodes": "node",
    "edges": "edge",
    "ports": "port",
    "robber": "robber",
}
PART_KINDS = ("tile", "node", "edge", "port", "robber")

__all__ = ["PART_KINDS", "PART_TO_KIND", "build_report", "filter_parts", "render_text_report"]


def _merge_uncovered(
    base: dict[str, list[str]], additions: dict[str, list[str]]
) -> None:
    for key, values in additions.items():
        if values:
            base[key].extend(values)


def build_report(
    sample_id: str,
    bench_dir: Path,
    contract: JsonDict,
    sample_meta: JsonDict,
    qa_rows: list[JsonDict],
    responses_by_qid: dict[str, JsonDict] | None,
) -> Report:
    expected = build_expected_inventory(contract)
    probe_map = summarize_probe(qa_rows, responses_by_qid)

    report_parts: dict[str, dict[str, PartPayload]] = {}
    for kind, expected_parts in expected.items():
        by_token: dict[str, PartPayload] = {}
        for token, info in sorted(expected_parts.items()):
            probes = dict(probe_map.get(f"{kind}:{token}", {}))
            by_token[token] = PartPayload(
                part=info,
                part_type=kind,
                probes=evaluate_part_token(probes),
            )
        report_parts[kind] = by_token

    summary: dict[str, PartSummary] = {}
    for kind in PART_KINDS:
        summary[kind] = summarize_part_payloads(kind, report_parts.get(kind, {}))

    # explicit top-level list of parts with no direct probe by type
    uncovered = empty_uncovered()
    for kind in PART_KINDS:
        _merge_uncovered(
            uncovered, uncovered_for_part(kind, report_parts.get(kind, {}), "all")
        )

    return Report(
        sample_id=sample_id,
        contract_path=sample_meta.get("contract_path"),
        image_path=sample_meta.get("image_path"),
        source=sample_meta.get("source", {}),
        summary=summary,
        uncovered_parts=uncovered,
        parts=report_parts,
        generated_from=GeneratedFrom(
            bench_dir=str(bench_dir),
            qa_count=len(qa_rows),
            response_count=len(responses_by_qid) if responses_by_qid else 0,
        ),
    )


def _selected_uncovered(
    selected: str, uncovered: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Keep only the selected part's buckets so the display stays unmixed."""
    if selected == "tile":
        wanted = {"tile_resource_number", "tile_has_robber", "tile_occupied_nodes"}
        return {k: v for k, v in uncovered.items() if k in wanted and v}
    if selected == "node":
        return {"node_occupancy": uncovered.get("node_occupancy", [])}
    if selected == "edge":
        return {"edge_road_owner": uncovered.get("edge_road_owner", [])}
    if selected == "port":
        return {
            k: uncovered.get(k, [])
            for k in ["port_trade_type", "port_type_nodes", "port_occupancy"]
            if uncovered.get(k)
        }
    if selected == "robber":
        return {"robber_tile": uncovered.get("robber_tile", [])}
    return uncovered


def filter_parts(report: Report, part: str, tile_mode: str) -> Report:
    if part == "all":
        return report

    selected = PART_TO_KIND[part]

    selected_probe_filter = None
    if selected == "tile":
        selected_probe_filter = TILE_MODE_PROBES.get(tile_mode)

    filtered_payloads = report["parts"].get(selected, {})
    filtered_parts: dict[str, PartPayload] = {}
    for token, payload in filtered_payloads.items():
        filtered_parts[token] = PartPayload(
            part=payload["part"],
            part_type=payload["part_type"],
            probes=filter_probe_map(payload["probes"], selected_probe_filter),
        )

    # override uncovered for the selected part using filtered probe context
    filtered_uncovered = uncovered_for_part(
        selected, filtered_parts, tile_mode if selected == "tile" else "all"
    )

    return Report(
        sample_id=report["sample_id"],
        contract_path=report["contract_path"],
        image_path=report["image_path"],
        source=report["source"],
        summary={selected: summarize_part_payloads(selected, filtered_parts)},
        uncovered_parts=_selected_uncovered(selected, filtered_uncovered),
        parts={selected: filtered_parts},
        generated_from=report["generated_from"],
    )


def render_text_report(report: Report, top_n: int) -> str:
    lines: list[str] = []
    lines.append(f"sample_id={report['sample_id']}")
    lines.append(f"contract={report['contract_path']}")
    lines.append(f"image={report['image_path']}")
    lines.append(f"qa_count={report['generated_from']['qa_count']}")
    lines.append(f"responses={report['generated_from']['response_count']}")

    for kind, payload in report["summary"].items():
        if not payload:
            continue
        total = payload["total"]
        label = "robber" if kind == "robber" else f"{kind}s"
        lines.append(f"{label} total={total}")
        for probe_kind, stats in payload["coverage"].items():
            lines.append(
                f"  {probe_kind}: tested={stats['tested']} passed={stats['passed']} failed={stats['failed']} / {total}"
            )

    lines.append("uncovered:")
    for probe_kind, tokens in report["uncovered_parts"].items():
        if not tokens:
            continue
        preview = ", ".join(tokens[:top_n])
        remaining = len(tokens) - top_n
        suffix = f" (+{remaining} more)" if remaining > 0 else ""
        lines.append(f"  {probe_kind}: {len(tokens)} -> {preview}{suffix}")

    return "\n".join(lines)
