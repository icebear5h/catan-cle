"""Probe Catan eval outputs for part-level failure diagnostics.

This script ingests an OpenRouter eval JSONL (or any similarly shaped run log)
and reports:
- part-level exact/component accuracy (tile/node/edge/port/robber/player)
- common failure modes (wrong color, wrong occupancy, missed EMPTY/NONE, etc.)
- top failed examples for each part to speed up data / prompt iteration.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def canonical_category(category: str) -> str:
    return CANONICAL_CATEGORY_MAP.get(category, category)


def normalize_text(value: Any) -> str:
    text = str(value).upper().strip()
    replacements = {
        "NO NUMBER": "NO_NUMBER",
        "NO-NUMBER": "NO_NUMBER",
        "NO PLAYER": "NONE",
        "NO ONE": "NONE",
        "NONE.": "NONE",
        "EMPTY.": "EMPTY",
        "DESERT": "<DESERT>",
        "WOOD": "<WOOD>",
        "BRICK": "<BRICK>",
        "SHEEP": "<SHEEP>",
        "WHEAT": "<WHEAT>",
        "ORE": "<ORE>",
        "SETTLEMENT": "<SETTLEMENT>",
        "CITY": "<CITY>",
        "GEN": "GENERIC",
    }
    replacements.update(
        {color: f"<{color}>" for color in ["RED", "BLUE", "WHITE", "BLACK", "GREEN", "ORANGE"]}
    )
    replacements.update(
        {
            color_name.replace("_", separator): f"<{color_name}>"
            for color_name in ["MYSTIC_BLUE", "BRONZE", "SILVER", "GOLD", "PINK"]
            if "_" in color_name
            for separator in (" ", "-")
        }
    )
    for src, dst in replacements.items():
        text = re.sub(rf"(?<![A-Z0-9_<]){re.escape(src)}(?![A-Z0-9_>])", dst, text)
    text = re.sub(r"<T(\d{1,2})(?![0-9_]*>)", lambda match: f"<T{int(match.group(1)):02d}>", text)
    text = re.sub(r"<N(\d{1,2})(?![0-9_]*>)", lambda match: f"<N{int(match.group(1)):02d}>", text)
    text = re.sub(r"\bT(\d{1,2})\b", lambda match: f"<T{int(match.group(1)):02d}>", text)
    text = re.sub(r"\bN(\d{1,2})\b", lambda match: f"<N{int(match.group(1)):02d}>", text)
    text = re.sub(r"<E(\d{1,2})[_-](\d{1,2})(?![0-9_]*>)", lambda match: f"<E{int(match.group(1)):02d}_{int(match.group(2)):02d}>", text)
    text = re.sub(r"\bE(\d{1,2})[_-](\d{1,2})\b", lambda match: f"<E{int(match.group(1)):02d}_{int(match.group(2)):02d}>", text)
    return re.sub(r"\s+", " ", text)

JsonDict = dict[str, Any]

MODEL_PREFIX = "model_key"
CATEGORY_KEYS = {
    "tile": {"tile_resource_number", "tile_has_robber", "robber_tile", "robber_resource_number"},
    "node": {"node_occupancy", "nodes_connected"},
    "edge": {"edge_road_owner", "edge_connects_nodes", "color_road_locations"},
    "port": {"port_trade_type", "port_occupancy", "port_type_nodes"},
    "player": {"color_building_counts", "color_road_count"},
    "robber": {"robber_presence", "robber_tile", "robber_resource_number", "tile_has_robber"},
    "board-logic": {"nodes_connected", "edge_connects_nodes", "node_adjacent_tiles"},
}

TILE_RESOURCE_RE = re.compile(r"<(WOOD|BRICK|SHEEP|WHEAT|ORE|DESERT)>")
COLOR_TOKEN_RE = re.compile(r"<(?:RED|BLUE|ORANGE|WHITE|BLACK|GREEN|BRONZE|SILVER|GOLD|PINK|MYSTIC_BLUE)>")
EDGE_TOKEN_RE = re.compile(r"<E(\d{2})_(\d{2})>")
NODE_TOKEN_RE = re.compile(r"<N\d{2}>")
ROAD_RE = re.compile(r"\bROAD(?:S)?\b")
SETTLEMENT_RE = re.compile(r"<SETTLEMENT>")
CITY_RE = re.compile(r"<CITY>")
NUMBER_RE = re.compile(r"\b(?:2|3|4|5|6|8|9|10|11|12)\b")
RATIO_RE = re.compile(r"\b3:1\b|\b2:1\b")


def normalize_response(value: Any) -> str:
    return normalize_text(value).upper().strip()


def classify_part(category: str) -> str:
    canonical = canonical_category(category)
    for part, cats in CATEGORY_KEYS.items():
        if canonical in cats:
            return part
    return "other"


def parse_int(text: str) -> int | None:
    match = re.search(r"\b\d+\b", text)
    return int(match.group(0)) if match else None


def is_empty_like(text: str) -> bool:
    return text in {"EMPTY", "NONE", ""} or bool(re.fullmatch(r"\s*(EMPTY|NONE)\s*", text))


def color_from_text(text: str) -> str | None:
    match = COLOR_TOKEN_RE.search(text)
    return match.group(0) if match else None


def extract_edge_tokens(text: str) -> set[str]:
    return {f"<E{a}_{b}>" for a, b in EDGE_TOKEN_RE.findall(text)}


def extract_numbers_for(label: str, text: str) -> int | None:
    if label in {"SETTLEMENTS", "SETTLEMENT"}:
        m = re.search(rf"<(?:SETTLEMENT|CITY)|\bSETTLEMENTS?\b.*?(\d+)|(\d+).*?\bSETTLEMENTS?\b", text)
        if not m:
            return parse_int(text) if text.isdigit() else None
        for value in m.groups():
            if value is not None:
                return int(value)
        return None
    if label in {"ROADS", "ROAD", "CITIES", "CITY"}:
        m = re.search(rf"<?(?:CITY|ROAD)S?\>?.*?(\d+)|(\d+).*(?:CITY|ROAD)S?\>?", text)
        if not m:
            return parse_int(text) if text.isdigit() else None
        for value in m.groups():
            if value is not None:
                return int(value)
        return None
    return parse_int(text)


def infer_failure_tokens(category: str, expected_raw: str, response_raw: str) -> list[str]:
    expected = normalize_response(expected_raw)
    response = normalize_response(response_raw)
    failures: list[str] = []

    if expected == response:
        return failures

    if category == "tile_resource_number":
        exp_res = TILE_RESOURCE_RE.search(expected)
        res_ok = exp_res and exp_res.group(0) in response
        if not res_ok:
            failures.append("tile_resource_wrong")
        exp_no = "NO_NUMBER" in expected
        res_no = "NO_NUMBER" in response
        if exp_no:
            if not res_no:
                failures.append("tile_number_wrong")
        else:
            exp_num = parse_int(expected)
            if exp_num is None or str(exp_num) not in response:
                failures.append("tile_number_wrong")
        return failures

    if category == "robber_resource_number":
        if "<T" in expected and "<T" not in response:
            failures.append("robber_tile_wrong")
        if TILE_RESOURCE_RE.search(expected):
            exp_res = TILE_RESOURCE_RE.search(expected).group(0) if TILE_RESOURCE_RE.search(expected) else None
            if exp_res and exp_res not in response:
                failures.append("robber_resource_wrong")
        exp_num = parse_int(expected)
        if exp_num is not None and str(exp_num) not in response:
            failures.append("robber_number_wrong")
        return failures

    if category == "robber_tile":
        if expected not in response:
            failures.append("robber_tile_wrong")
        return failures

    if category == "tile_has_robber":
        if "YES" in expected and "YES" not in response:
            failures.append("robber_presence_false_negative")
        if "NO" in expected and "NO" not in response:
            failures.append("robber_presence_false_positive")
        return failures

    if category in {"node_occupancy", "port_occupancy"}:
        if expected == "EMPTY" or expected == "NONE":
            if not is_empty_like(response):
                failures.append("false_positive_occupancy")
        elif "EMPTY" in expected:
            if "EMPTY" in response:
                failures.append("missed_occupancy")
            else:
                failures.append("occupancy_blank")
        else:
            exp_color = color_from_text(expected)
            res_color = color_from_text(response)
            if exp_color and exp_color != res_color:
                failures.append("occupancy_wrong_color")
            if category == "node_occupancy":
                exp_building = "<SETTLEMENT>" if "<SETTLEMENT>" in expected else "<CITY>" if "<CITY>" in expected else None
                if exp_building and exp_building not in response:
                    failures.append("occupancy_wrong_piece")
            else:
                if "<N" in expected and "<N" not in response:
                    failures.append("port_node_wrong")
                if NODE_TOKEN_RE.search(response) is None:
                    failures.append("port_node_missing")
        return failures

    if category == "edge_road_owner":
        if expected == "EMPTY":
            if not is_empty_like(response):
                failures.append("road_false_positive")
        else:
            if expected not in response:
                failures.append("road_wrong_color")
        return failures

    if category == "color_road_locations":
        if expected == "NONE":
            if extract_edge_tokens(response):
                failures.append("roads_extra_edges")
            else:
                failures.append("none_wrong")
        else:
            exp_edges = {e for e in EDGE_TOKEN_RE.findall(expected)}
            if not exp_edges:
                failures.append("roads_parse_failed")
                return failures
            exp_edges = {f"<E{a}_{b}>" for a, b in exp_edges}
            resp_edges = extract_edge_tokens(response)
            missing = sorted(exp_edges - resp_edges)
            extra = sorted(resp_edges - exp_edges)
            if missing:
                failures.append("roads_missing")
            if extra:
                failures.append("roads_extra")
            if not missing and not extra:
                failures.append("unknown_edge_set_mismatch")
        return failures

    if category == "port_trade_type":
        if "GENERIC" in expected:
            if "3:1" not in response:
                failures.append("port_ratio_wrong")
            if "GENERIC" not in response and "<GENERIC>" not in response and "GENERIC" not in response:
                failures.append("port_resource_wrong")
        else:
            if not TILE_RESOURCE_RE.search(expected):
                failures.append("port_resource_parse_failed")
            else:
                exp_res = TILE_RESOURCE_RE.search(expected).group(0)
                if exp_res not in response:
                    failures.append("port_resource_wrong")
            exp_ratio = "2:1" if "2:1" in expected else "3:1"
            if exp_ratio not in response:
                failures.append("port_ratio_wrong")
        return failures

    if category == "color_building_counts":
        exp_color = color_from_text(expected)
        if exp_color and exp_color not in response:
            failures.append("count_wrong_player")
        exp_settlements = parse_int(re.search(r"SETTLEMENTS\\s+(\\d+)", expected).group(1)) if re.search(r"SETTLEMENTS\\s+(\\d+)", expected) else None
        exp_cities = parse_int(re.search(r"CITIES\\s+(\\d+)", expected).group(1)) if re.search(r"CITIES\\s+(\\d+)", expected) else None
        if exp_settlements is not None:
            resp_settlements = extract_numbers_for("SETTLEMENT", response)
            if resp_settlements is None or resp_settlements != exp_settlements:
                failures.append("settlement_count_wrong")
        if exp_cities is not None:
            resp_cities = extract_numbers_for("CITY", response)
            if resp_cities is None or resp_cities != exp_cities:
                failures.append("city_count_wrong")
        return failures

    if category == "color_road_count":
        exp_color = color_from_text(expected)
        if exp_color and exp_color not in response:
            failures.append("count_wrong_player")
        exp_roads = parse_int(expected)
        if exp_roads is None:
            if "ROAD" not in expected and "ROADS" not in expected:
                exp_roads = parse_int(response)
        resp_roads = parse_int(response)
        if exp_roads is None and response.isdigit():
            resp_roads = parse_int(response)
        if exp_roads is not None and resp_roads != exp_roads:
            failures.append("road_count_wrong")
        return failures

    if category in {"nodes_connected", "edge_connects_nodes"}:
        expected_yes = "YES" in expected
        resp_yes = "YES" in response
        if expected_yes != resp_yes:
            failures.append("topology_connectivity_wrong")
        return failures

    # fallback
    if response.strip() != expected.strip():
        failures.append("mismatch")
    return failures


def analyze_records(records: list[JsonDict], model_filter: str | None = None) -> dict[str, Any]:
    selected = [r for r in records if model_filter is None or r.get(MODEL_PREFIX) == model_filter]
    by_part: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "attempted": 0,
            "exact": 0,
            "component": 0,
            "component_total": 0,
            "errors": 0,
            "failure_counts": Counter(),
            "samples": [],
            "examples": [],
        }
    )

    totals = {
        "attempted": 0,
        "exact": 0,
        "component": 0,
        "component_total": 0,
        "errors": 0,
    }

    for row in selected:
        if row.get("error"):
            totals["errors"] += 1
            continue

        category = canonical_category(row.get("category", ""))
        part = classify_part(category)
        model_key = row.get(MODEL_PREFIX, "")
        score = row.get("score") or {}
        component_correct = score.get("component_correct", 0)
        component_total = score.get("component_total", 0)
        exact_ok = bool(score.get("correct", False))
        expected = str(row.get("expected", ""))
        response = str(row.get("response", ""))
        failures = infer_failure_tokens(category, expected, response)

        totals["attempted"] += 1
        totals["exact"] += 1 if exact_ok else 0
        totals["component"] += int(component_correct or 0)
        totals["component_total"] += int(component_total or 0)

        agg = by_part[part]
        agg["attempted"] += 1
        agg["exact"] += 1 if exact_ok else 0
        agg["component"] += int(component_correct or 0)
        agg["component_total"] += int(component_total or 0)
        agg["samples"].append((part, category, model_key, row.get("sample_id", ""), row.get("question_id", "")))
        if failures:
            for failure in failures:
                agg["failure_counts"][failure] += 1
            agg["examples"].append(
                {
                    "sample_id": row.get("sample_id"),
                    "question_id": row.get("question_id"),
                    "category": category,
                    "model_key": model_key,
                    "expected": expected,
                    "response": response,
                    "failures": failures,
                }
            )
        if failures:
            totals["errors"] += 1
        agg["errors"] += 1 if failures else 0

    for agg in by_part.values():
        if agg["component_total"]:
            agg["component_accuracy"] = agg["component"] / agg["component_total"]
        else:
            agg["component_accuracy"] = 0.0
        agg["exact_accuracy"] = agg["exact"] / agg["attempted"] if agg["attempted"] else 0.0
        agg["failure_counts"] = dict(agg["failure_counts"])

    totals["component_accuracy"] = totals["component"] / totals["component_total"] if totals["component_total"] else 0.0
    totals["exact_accuracy"] = totals["exact"] / totals["attempted"] if totals["attempted"] else 0.0
    return {"totals": totals, "parts": by_part}


def top_failures_by_part(part_stats: dict[str, Any], top_n: int) -> dict[str, list[JsonDict]]:
    out: dict[str, list[JsonDict]] = {}
    for part, agg in part_stats.items():
        examples = sorted(
            agg["examples"],
            key=lambda item: item.get("sample_id", ""),
        )
        out[part] = examples[:top_n]
    return out


def render_text_report(result: dict[str, Any], top_n: int) -> str:
    lines: list[str] = []
    totals = result["totals"]
    lines.append(f"attempted={totals['attempted']}")
    lines.append(f"exact_accuracy={totals['exact_accuracy']:.4f}")
    lines.append(f"component_accuracy={totals['component_accuracy']:.4f}")
    lines.append(f"examples_with_errors={totals['errors']}")
    lines.append("")
    lines.append("part diagnostics:")

    for part, agg in result["parts"].items():
        lines.append(
            f"- {part}: attempted={agg['attempted']} exact={agg['exact_accuracy']:.4f} "
            f"component={agg['component_accuracy']:.4f} errors={agg['errors']}"
        )
        top = sorted(agg["failure_counts"].items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        if top:
            lines.append("  top failures: " + ", ".join(f"{name}:{count}" for name, count in top))

    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True, help="Eval JSONL from eval runner.")
    parser.add_argument("--model", type=str, default=None, help="Filter to single model_key (optional).")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output report path.")
    parser.add_argument("--top-failures", type=int, default=10, help="Examples per part in report.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    lines = args.responses.read_text().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]

    result = analyze_records(records, model_filter=args.model)
    by_part = result["parts"]
    fail_examples = {
        part: ex[: args.top_failures]
        for part, ex in top_failures_by_part(by_part, args.top_failures).items()
    }
    result["top_failures"] = fail_examples

    print(render_text_report(result, top_n=max(5, min(args.top_failures, 15))))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as handle:
            serializable_parts = {}
            for part, agg in by_part.items():
                serializable_parts[part] = {
                    "attempted": agg["attempted"],
                    "exact": agg["exact"],
                    "exact_accuracy": agg["exact_accuracy"],
                    "component": agg["component"],
                    "component_total": agg["component_total"],
                    "component_accuracy": agg["component_accuracy"],
                    "errors": agg["errors"],
                    "failure_counts": agg["failure_counts"],
                    "examples": agg["examples"][: args.top_failures],
                }
            payload = {
                "totals": result["totals"],
                "parts": serializable_parts,
                "top_failures": fail_examples,
                "generated_from": str(args.responses),
                "model_filter": args.model,
                "top_failures_limit": args.top_failures,
            }
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
