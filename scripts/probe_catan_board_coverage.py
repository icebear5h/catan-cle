#!/usr/bin/env python
"""Probe a single board sample for atlas-part coverage and eval outcomes.

This script reports whether each board object (tile/node/edge/port) that appears in a
sample contract is covered by QA probes, and optionally whether those probes were
answered correctly in a provided eval responses file.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# local canonical alias mapping avoids importing engine deps (for example networkx)
CANONICAL_CATEGORY_MAP = {
    "isolated_tile_resource_number": "tile_resource_number",
    "local_patch_tile_resource_number": "tile_resource_number",
    "isolated_road_owner": "edge_road_owner",
    "local_patch_edge_road_owner": "edge_road_owner",
    "isolated_node_occupancy": "node_occupancy",
    "local_patch_node_occupancy": "node_occupancy",
    "isolated_port_trade_type": "port_trade_type",
    "local_patch_port_trade_type": "port_trade_type",
    "isolated_robber_presence": "robber_presence",
    "local_patch_robber_presence": "robber_presence",
}

TILE_MODE_PROBES = {
    "identity": {"tile_has_robber", "tile_occupied_nodes"},
    "values": {"tile_resource_number"},
}
TILE_MODES = {"all", "identity", "values"}


def filter_probe_map(probes: dict[str, Any], keep: set[str] | None) -> dict[str, Any]:
    if keep is None:
        return dict(probes)
    return {probe_kind: info for probe_kind, info in probes.items() if probe_kind in keep}


def canonical_category(category: str) -> str:
    return CANONICAL_CATEGORY_MAP.get(category, category)


JsonDict = dict[str, Any]

ROBBER_PROBE_TYPES = ("robber_tile", "robber_resource_number", "robber_presence", "robber_adjacent_buildings")


def read_jsonl(path: Path) -> list[JsonDict]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [json.loads(line) for line in lines if line]


def read_manifest(bench_dir: Path) -> dict[str, JsonDict]:
    manifest_path = bench_dir / "manifest.jsonl"
    return {row["sample_id"]: row for row in read_jsonl(manifest_path)}


def normalize_token(token: str | None) -> str | None:
    if token is None:
        return None
    return str(token).strip().upper()


def question_to_targets(qa: JsonDict) -> list[tuple[str, str, str]]:
    """Return part probes targeted by one QA row.

    Returns tuples of (part_type, part_token, probe_kind).
    """

    category = canonical_category(qa.get("category", ""))
    target = qa.get("target", {}) if isinstance(qa.get("target"), dict) else {}

    out: list[tuple[str, str, str]] = []

    if category in {
        "tile_resource_number",
        "tile_has_robber",
        "tile_occupied_nodes",
    }:
        tile_token = normalize_token(target.get("tile_token"))
        if tile_token:
            out.append(("tile", tile_token, category))

    if category in {"robber_tile", "robber_presence", "robber_resource_number", "robber_adjacent_buildings"}:
        tile_token = normalize_token(target.get("tile_token"))
        if tile_token:
            out.append(("robber", tile_token, category))

    if category == "node_occupancy":
        node_token = normalize_token(target.get("node_token"))
        if node_token:
            out.append(("node", node_token, category))

    if category == "node_adjacent_tiles":
        node_token = normalize_token(target.get("node_token"))
        if node_token:
            out.append(("node", node_token, category))

    if category in {"edge_road_owner", "edge_connects_nodes"}:
        edge_token = normalize_token(target.get("edge_token"))
        if edge_token:
            out.append(("edge", edge_token, category))

    if category in {"port_trade_type", "port_type_nodes", "port_occupancy"}:
        port_token = normalize_token(target.get("port_token"))
        if port_token:
            out.append(("port", port_token, category))

    if category == "nodes_connected":
        for token in target.get("node_tokens", []) or []:
            norm = normalize_token(token)
            if norm:
                out.append(("node", norm, "nodes_connected"))

    return out


def summarize_probe(records: list[JsonDict], responses: dict[str, JsonDict] | None = None) -> dict[str, Any]:
    """Summarize QA coverage keyed by (part type, part token, probe kind)."""

    by_part: dict[tuple[str, str], dict[str, list[JsonDict]]] = defaultdict(lambda: defaultdict(list))

    for qa in records:
        question_id = qa.get("id")
        response_info = responses.get(question_id) if responses is not None and question_id else None

        score_payload = response_info.get("score", {}) if response_info else None
        scored_correct = None if response_info is None else bool(score_payload.get("correct", False))
        had_error = bool(response_info.get("error")) if response_info is not None else False

        entry = {
            "question_id": question_id,
            "category": qa.get("category"),
            "canonical_category": canonical_category(qa.get("category", "")),
            "answer": qa.get("answer"),
            "question": qa.get("question"),
            "sample_id": qa.get("sample_id"),
            "response": response_info.get("response") if response_info else None,
            "correct": scored_correct,
            "error": had_error,
        }

        for part_kind, token, probe_kind in question_to_targets(qa):
            by_part[(part_kind, token)][probe_kind].append(entry)

    # keep dict keys stringified as "kind:token" for compact joining by caller
    return {f"{kind}:{token}": probes for (kind, token), probes in by_part.items() for probes in [dict(probes)]}


def build_expected_inventory(contract: JsonDict) -> dict[str, dict[str, JsonDict]]:
    tiles = {
        normalize_token(tile.get("token")): {
            "id": tile.get("id"),
            "token": normalize_token(tile.get("token")),
            "resource": tile.get("resource"),
            "number": tile.get("number"),
            "has_robber": tile.get("id") == (contract.get("robber", {}) or {}).get("tile_id"),
            "node_tokens": tile.get("node_tokens", []),
            "edge_tokens": tile.get("edge_tokens", []),
        }
        for tile in contract.get("tiles", [])
        if normalize_token(tile.get("token"))
    }

    nodes = {
        normalize_token(node.get("token")): {
            "id": node.get("id"),
            "token": normalize_token(node.get("token")),
            "building": node.get("building"),
            "color": node.get("color"),
            "adjacent_tiles": node.get("adjacent_tiles", []),
            "adjacent_edges": node.get("adjacent_edges", []),
            "port_tokens": node.get("port_tokens", []),
        }
        for node in contract.get("nodes", [])
        if normalize_token(node.get("token"))
    }

    edges = {
        normalize_token(edge.get("token")): {
            "id": edge.get("id"),
            "token": normalize_token(edge.get("token")),
            "nodes": edge.get("nodes", []),
            "road_color": edge.get("road_color"),
            "node_tokens": edge.get("node_tokens", []),
        }
        for edge in contract.get("edges", [])
        if normalize_token(edge.get("token"))
    }

    ports = {
        normalize_token(port.get("token")): {
            "id": port.get("id"),
            "token": normalize_token(port.get("token")),
            "resource": port.get("resource"),
            "ratio": port.get("ratio"),
            "nodes": port.get("attached_nodes", []),
            "node_tokens": port.get("attached_node_tokens", []),
            "coord": port.get("coord"),
        }
        for port in contract.get("ports", [])
        if normalize_token(port.get("token"))
    }

    robber_token = normalize_token((contract.get("robber", {}) or {}).get("tile_token"))

    return {
        "tile": tiles,
        "node": nodes,
        "edge": edges,
        "port": ports,
        "robber": {robber_token: {"token": robber_token}} if robber_token else {},
    }


def evaluate_part_token(probes_by_kind: dict[str, list[JsonDict]]) -> dict[str, Any]:
    result: dict[str, Any] = {}

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

        result[probe_kind] = {
            "attempted": attempted,
            "correct": correct,
            "incorrect": incorrect,
            "errors": errors,
            "question_ids": [row["question_id"] for row in records],
            "status": status,
        }

    return result


def summarize_part_payloads(
    part: str, part_payloads: dict[str, Any], probe_filter: set[str] | None = None
) -> dict[str, Any]:
    total_parts = len(part_payloads)
    summary: dict[str, Any] = {"total": total_parts, "coverage": {}}

    coverage_by_probe: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"tested": 0, "passed": 0, "failed": 0}
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
            coverage_by_probe.setdefault(probe_kind, {"tested": 0, "passed": 0, "failed": 0})

    summary["coverage"] = {
        probe_kind: metrics for probe_kind, metrics in sorted(coverage_by_probe.items())
    }

    return summary


def uncovered_for_part(
    part: str, part_payloads: dict[str, Any], tile_mode: str
) -> dict[str, list[str]]:
    uncovered: dict[str, list[str]] = {
        "tile_resource_number": [],
        "tile_has_robber": [],
        "tile_occupied_nodes": [],
        "node_occupancy": [],
        "edge_road_owner": [],
        "port_trade_type": [],
        "port_type_nodes": [],
        "port_occupancy": [],
        "robber_tile": [],
    }

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


def build_report(
    sample_id: str,
    bench_dir: Path,
    contract: JsonDict,
    sample_meta: JsonDict,
    qa_rows: list[JsonDict],
    responses_by_qid: dict[str, JsonDict] | None,
) -> dict[str, Any]:
    expected = build_expected_inventory(contract)
    probe_map = summarize_probe(qa_rows, responses_by_qid)

    report_parts: dict[str, dict[str, Any]] = {}
    for kind, expected_parts in expected.items():
        by_token: dict[str, Any] = {}
        for token, info in sorted(expected_parts.items()):
            if token is None:
                continue
            probes = dict(probe_map.get(f"{kind}:{token}", {}))
            by_token[token] = {
                "part": info,
                "part_type": kind,
                "probes": evaluate_part_token(probes),
            }
        report_parts[kind] = by_token

    summary: dict[str, Any] = {}
    for kind in ["tile", "node", "edge", "port", "robber"]:
        summary[kind] = summarize_part_payloads(kind, report_parts.get(kind, {}))

    # explicit top-level list of parts with no direct probe by type
    uncovered: dict[str, list[str]] = {
        "tile_resource_number": [],
        "tile_has_robber": [],
        "tile_occupied_nodes": [],
        "node_occupancy": [],
        "edge_road_owner": [],
        "port_trade_type": [],
        "port_type_nodes": [],
        "port_occupancy": [],
        "robber_tile": [],
    }

    def merge_uncovered(base: dict[str, list[str]], additions: dict[str, list[str]]) -> None:
        for key, values in additions.items():
            if values:
                base[key].extend(values)

    merge_uncovered(uncovered, uncovered_for_part("tile", report_parts.get("tile", {}), "all"))
    merge_uncovered(uncovered, uncovered_for_part("node", report_parts.get("node", {}), "all"))
    merge_uncovered(uncovered, uncovered_for_part("edge", report_parts.get("edge", {}), "all"))
    merge_uncovered(uncovered, uncovered_for_part("port", report_parts.get("port", {}), "all"))
    merge_uncovered(uncovered, uncovered_for_part("robber", report_parts.get("robber", {}), "all"))

    return {
        "sample_id": sample_id,
        "contract_path": sample_meta.get("contract_path"),
        "image_path": sample_meta.get("image_path"),
        "source": sample_meta.get("source", {}),
        "summary": summary,
        "uncovered_parts": uncovered,
        "parts": report_parts,
        "generated_from": {
            "bench_dir": str(bench_dir),
            "qa_count": len(qa_rows),
            "response_count": len(responses_by_qid) if responses_by_qid else 0,
        },
    }


def load_response_map(
    path: Path,
    sample_id: str | None = None,
    model: str | None = None,
) -> dict[str, JsonDict]:
    if not path.exists():
        raise FileNotFoundError(f"responses file not found: {path}")

    by_qid: dict[str, JsonDict] = {}
    for row in read_jsonl(path):
        if sample_id is not None and row.get("sample_id") != sample_id:
            continue
        if model is not None and row.get("model_key") != model:
            continue
        qid = row.get("question_id")
        if qid:
            by_qid[str(qid)] = row
    return by_qid


def resolve_sample(bench_dir: Path, sample_id: str | None, image: str | None) -> str:
    manifest = read_manifest(bench_dir)
    if sample_id:
        if sample_id not in manifest:
            raise KeyError(f"sample {sample_id} not found in {bench_dir / 'manifest.jsonl'}")
        return sample_id

    if image is None:
        return sorted(manifest)[0]

    image_norm = image.strip().replace("\\", "/")
    for sid, row in manifest.items():
        row_image = str(row.get("image_path"))
        if image_norm == row_image or image_norm == row_image.split("/")[-1]:
            return sid
    raise KeyError(f"could not find sample with image_path {image!r}")


def render_text_report(report: dict[str, Any], top_n: int) -> str:
    lines: list[str] = []
    lines.append(f"sample_id={report['sample_id']}")
    lines.append(f"contract={report['contract_path']}")
    lines.append(f"image={report['image_path']}")
    lines.append(f"qa_count={report['generated_from']['qa_count']}")
    lines.append(f"responses={report['generated_from']['response_count']}")

    for kind, payload in report["summary"].items():
        payload = report["summary"].get(kind, {})
        if not payload:
            continue
        total = payload.get("total", 0)
        label = "robber" if kind == "robber" else f"{kind}s"
        lines.append(f"{label} total={total}")
        for probe_kind, stats in payload.get("coverage", {}).items():
            if not isinstance(stats, dict):
                continue
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-dir", type=Path, default=Path("evals/catan_board_bench/datasets/catan_board_bench_100")
    )
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Resolve sample by image path or file name",
    )
    parser.add_argument(
        "--qa",
        type=Path,
        default=Path("evals/catan_board_bench/datasets/catan_board_bench_100/questions/qa.jsonl"),
        help="QA rows JSONL (usually questions/qa.jsonl)",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=None,
        help="Optional responses.jsonl path",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Directory containing responses.jsonl",
    )
    parser.add_argument("--model", type=str, default=None, help="Filter response rows by model_key")
    parser.add_argument("--top", type=int, default=20, help="Top N uncovered tokens per probe type to show")
    parser.add_argument(
        "--part",
        type=str,
        default="all",
        choices=["all", "tiles", "nodes", "edges", "ports", "robber"],
        help="Limit report to a single object class",
    )
    parser.add_argument(
        "--tile-mode",
        type=str,
        default="all",
        choices=sorted(TILE_MODES),
        help="For --part tiles: all/identity/values probe families",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output")
    return parser


def filter_parts(report: dict[str, Any], part: str, tile_mode: str) -> dict[str, Any]:
    if part == "all":
        return report

    part_to_kind = {
        "tiles": "tile",
        "nodes": "node",
        "edges": "edge",
        "ports": "port",
        "robber": "robber",
    }
    selected = part_to_kind[part]

    filtered = dict(report)
    selected_probe_filter = None
    if selected == "tile":
        selected_probe_filter = TILE_MODE_PROBES.get(tile_mode)

    filtered_payloads = report["parts"].get(selected, {})
    filtered_parts: dict[str, Any] = {}
    for token, payload in filtered_payloads.items():
        filtered_probes = filter_probe_map(payload["probes"], selected_probe_filter)
        filtered_parts[token] = {**payload, "probes": filtered_probes}

    filtered["parts"] = {selected: filtered_parts}
    filtered["summary"] = {selected: summarize_part_payloads(selected, filtered_parts)}

    # override uncovered for the selected part using filtered probe context
    if selected in {"tile", "node", "edge", "port", "robber"}:
        filtered_uncovered = uncovered_for_part(selected, filtered_parts, tile_mode if selected == "tile" else "all")
        filtered["uncovered_parts"] = filtered_uncovered
    else:
        filtered["uncovered_parts"] = report.get("uncovered_parts", {})

    # keep only selected part's uncovered buckets in final display to avoid mixed output
    if selected == "tile":
        filtered["uncovered_parts"] = {
            k: v
            for k, v in filtered["uncovered_parts"].items()
            if k
            in {
                "tile_resource_number",
                "tile_has_robber",
                "tile_occupied_nodes",
            }
            and v
        }
    elif selected == "node":
        filtered["uncovered_parts"] = {"node_occupancy": filtered["uncovered_parts"].get("node_occupancy", [])}
    elif selected == "edge":
        filtered["uncovered_parts"] = {"edge_road_owner": filtered["uncovered_parts"].get("edge_road_owner", [])}
    elif selected == "port":
        filtered["uncovered_parts"] = {
            k: filtered["uncovered_parts"].get(k, [])
            for k in ["port_trade_type", "port_type_nodes", "port_occupancy"]
            if filtered["uncovered_parts"].get(k)
        }
    elif selected == "robber":
        filtered["uncovered_parts"] = {"robber_tile": filtered["uncovered_parts"].get("robber_tile", [])}

    return filtered


def main() -> int:
    args = build_parser().parse_args()

    bench_dir = args.bench_dir
    sample_id = resolve_sample(bench_dir, args.sample_id, args.image)
    manifest_map = read_manifest(bench_dir)
    sample_meta = manifest_map[sample_id]

    contract_path = bench_dir / sample_meta["contract_path"]
    contract = json.loads(contract_path.read_text())

    all_questions = [row for row in read_jsonl(args.qa) if row.get("sample_id") == sample_id]
    if not all_questions:
        raise RuntimeError(f"no QA rows found for sample {sample_id} in {args.qa}")

    responses_path: Path | None = args.responses
    if responses_path is None and args.eval_dir is not None:
        responses_path = args.eval_dir / "responses.jsonl"

    response_map: dict[str, JsonDict] | None = None
    if responses_path is not None:
        response_map = load_response_map(responses_path, sample_id=sample_id, model=args.model)

    report = build_report(
        sample_id=sample_id,
        bench_dir=bench_dir,
        contract=contract,
        sample_meta=sample_meta,
        qa_rows=all_questions,
        responses_by_qid=response_map,
    )

    if args.part != "all":
        report = filter_parts(report, args.part, args.tile_mode)

    report_text = render_text_report(report, top_n=args.top)
    print(report_text)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
