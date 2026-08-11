"""Mechanistic geometry probe for Catan atlas parts (tile/node/edge/port)
using Qwen visual-language residuals."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from catanbench.tokens import (
    atlas_metadata,
    canonical_edge,
    edge_token,
    node_token,
    port_token,
    tile_token,
)  # noqa: E402


CANONICAL_CATEGORY_MAP: dict[str, str] = {
    "isolated_tile_resource_number": "tile",
    "local_patch_tile_resource_number": "tile",
    "isolated_road_owner": "edge",
    "local_patch_edge_road_owner": "edge",
    "isolated_node_occupancy": "node",
    "local_patch_node_occupancy": "node",
    "isolated_port_trade_type": "port",
    "local_patch_port_trade_type": "port",
    "isolated_robber_presence": "tile",
    "local_patch_robber_presence": "tile",
    "tile_resource_number": "tile",
    "tile_has_robber": "tile",
    "tile_occupied_nodes": "tile",
    "robber_tile": "tile",
    "robber_resource_number": "tile",
    "robber_adjacent_buildings": "tile",
    "node_occupancy": "node",
    "node_adjacent_tiles": "node",
    "edge_road_owner": "edge",
    "edge_connects_nodes": "edge",
    "port_trade_type": "port",
    "port_type_nodes": "port",
    "port_occupancy": "port",
}

PARTS = ("tile", "node", "edge", "port")

TOKEN_PATTERNS = {
    "tile": re.compile(r"<T(\d{1,2})>"),
    "node": re.compile(r"<N(\d{1,2})>"),
    "edge": re.compile(r"<E(\d{1,2})_(\d{1,2})>"),
    "port": re.compile(r"<P(\d{1,2})>"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-jsonl", required=True, type=Path, help="Path to QA JSONL rows.")
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        default=None,
        help="Optional manifest JSONL with sample_id -> image_path.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="HF model id or local path, e.g. Qwen/Qwen2.5-VL-3B-Instruct.",
    )
    parser.add_argument("--adapter-dir", default=None, help="Optional PEFT adapter path.")
    parser.add_argument("--bits", type=int, default=16, choices=[4, 8, 16], help="Model precision.")
    parser.add_argument(
        "--disable-flash-attn2",
        action="store_true",
        help="Use SDPA instead of FlashAttention2.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device.")
    parser.add_argument(
        "--rows-limit",
        type=int,
        default=200,
        help="Max QA rows to process.",
    )
    parser.add_argument(
        "--categories",
        default=",".join(sorted(CANONICAL_CATEGORY_MAP)),
        help="Comma-separated categories to include.",
    )
    parser.add_argument(
        "--parts",
        default=",".join(PARTS),
        help="Comma-separated atlas parts to include: tile,node,edge,port. "
        "If set, category filtering is applied on top of these parts.",
    )
    parser.add_argument(
        "--prompt-prefix",
        default="Answer exactly using Catan tokens. Do not explain.",
        help="Prompt prefix appended before each question.",
    )
    parser.add_argument(
        "--topk-adj",
        default="2,3,4,5",
        help="Top-k values for topology precision/recall metrics.",
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=1,
        help="Skip token classes with fewer examples.",
    )
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Write full probe JSON report here.",
    )
    return parser.parse_args()


def _iter_jsonl(path: Path):
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def _canon_token(token: str) -> str:
    token = str(token).strip().upper()
    if not (token.startswith("<") and token.endswith(">")):
        return token
    match = re.fullmatch(r"<T(\d{1,2})>", token)
    if match:
        return f"<T{int(match.group(1)):02d}>"
    match = re.fullmatch(r"<N(\d{1,2})>", token)
    if match:
        return f"<N{int(match.group(1)):02d}>"
    match = re.fullmatch(r"<P(\d{1,2})>", token)
    if match:
        return f"<P{int(match.group(1)):02d}>"
    match = re.fullmatch(r"<E(\d{1,2})_(\d{1,2})>", token)
    if match:
        return edge_token((int(match.group(1)), int(match.group(2))))
    return token


def _extract_tokens_by_kind(text: str, kind: str) -> list[str]:
    text = str(text or "")
    pattern = TOKEN_PATTERNS[kind]
    matches = pattern.findall(text)

    tokens: list[str] = []
    if kind == "edge":
        for match in matches:
            if isinstance(match, tuple) and len(match) == 2:
                tokens.append(edge_token((int(match[0]), int(match[1]))))
            else:
                tokens.append(str(match))
    elif kind == "tile":
        for raw in matches:
            value = raw if not isinstance(raw, tuple) else raw[0]
            tokens.append(tile_token(int(value)))
    elif kind == "node":
        for raw in matches:
            value = raw if not isinstance(raw, tuple) else raw[0]
            tokens.append(node_token(int(value)))
    elif kind == "port":
        for raw in matches:
            value = raw if not isinstance(raw, tuple) else raw[0]
            tokens.append(port_token(int(value)))

    seen: set[str] = set()
    deduped: list[str] = []
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        deduped.append(tok)
    return deduped


def _kind_from_token(token: str) -> tuple[str, str] | None:
    token = _canon_token(token)
    if re.fullmatch(r"<T\d{2}>", token):
        return "tile", token
    if re.fullmatch(r"<N\d{2}>", token):
        return "node", token
    if re.fullmatch(r"<E\d{2}_\d{2}>", token):
        return "edge", token
    if re.fullmatch(r"<P\d{2}>", token):
        return "port", token
    return None


def _infer_target_token(category: str, target: dict[str, Any]) -> tuple[str, str] | None:
    target_kind = CANONICAL_CATEGORY_MAP.get(category)
    if target_kind is None:
        return None

    preference = {
        "tile": ["tile_token", "tile", "tile_id"],
        "node": ["node_token", "node", "node_id"],
        "edge": ["edge_token", "edge_id", "road_edge", "road_edge_tokens", "nodes"],
        "port": ["port_token", "port", "port_id"],
    }[target_kind]

    for key in preference:
        if key not in target:
            continue
        value = target.get(key)
        if value is None:
            continue

        if isinstance(value, (list, tuple)):
            if not value:
                continue
            if target_kind == "edge" and len(value) == 2:
                return "edge", edge_token((int(value[0]), int(value[1])))
            value = value[0]

        if isinstance(value, int):
            if target_kind == "tile":
                return "tile", tile_token(value)
            if target_kind == "node":
                return "node", node_token(value)
            if target_kind == "port":
                return "port", port_token(value)

        if isinstance(value, str):
            parsed = _kind_from_token(value)
            if parsed and parsed[0] == target_kind:
                return parsed

    # Fallback by explicit id fields that may appear with different keys.
    if target_kind == "tile" and isinstance(target.get("tile_token"), str):
        parsed = _kind_from_token(target["tile_token"])
        if parsed:
            return parsed
    if target_kind == "node" and isinstance(target.get("node_token"), str):
        parsed = _kind_from_token(target["node_token"])
        if parsed:
            return parsed
    if target_kind == "edge" and isinstance(target.get("edge_id"), (list, tuple)) and len(target["edge_id"]) == 2:
        return "edge", edge_token((int(target["edge_id"][0]), int(target["edge_id"][1])))
    if target_kind == "port" and isinstance(target.get("port_token"), str):
        parsed = _kind_from_token(target["port_token"])
        if parsed:
            return parsed
    return None


def _infer_anchor_token(row: dict[str, Any], target_kind: str) -> str | None:
    for text in (row.get("question"), row.get("answer")):
        tokens = _extract_tokens_by_kind(str(text), target_kind)
        if tokens:
            return tokens[0]

    if "target" in row and isinstance(row["target"], dict):
        parsed = _infer_target_token(row.get("category", ""), row["target"])
        if parsed and parsed[0] == target_kind:
            return parsed[1]
    return None


def _load_manifest(path: Path | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if path is None or not path.exists():
        return mapping
    for _, row in _iter_jsonl(path):
        sample_id = row.get("sample_id")
        image_path = row.get("image_path")
        if sample_id and image_path:
            mapping[sample_id] = str(image_path)
    return mapping


def _load_rows(qa_jsonl: Path, manifest: dict[str, str], dataset_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    allowed_categories = {cat.strip() for cat in args.categories.split(",") if cat.strip()}
    requested_parts = {part.strip().lower() for part in args.parts.split(",") if part.strip()}
    requested_parts = {part for part in requested_parts if part in PARTS}
    if not requested_parts:
        requested_parts = set(PARTS)

    rows: list[dict[str, Any]] = []
    for _, row in _iter_jsonl(qa_jsonl):
        category = row.get("category")
        if category not in allowed_categories:
            continue
        target_kind = CANONICAL_CATEGORY_MAP.get(category)
        if target_kind is None:
            continue
        if target_kind not in requested_parts:
            continue

        target = row.get("target")
        if not isinstance(target, dict):
            continue

        anchor_token = _infer_anchor_token(row, target_kind)
        if anchor_token is None:
            continue

        inferred = _infer_target_token(row.get("category", ""), target)
        target_token = inferred[1] if inferred else anchor_token

        sample_id = row.get("sample_id")
        image_path = row.get("image_path")
        if not image_path and sample_id:
            image_path = manifest.get(sample_id)
        if not image_path:
            continue

        image_path = str(image_path)
        image_full = Path(image_path)
        if not image_full.is_absolute():
            image_full = (dataset_root / image_full).resolve()

        rows.append(
            {
                "id": row.get("id"),
                "sample_id": sample_id,
                "category": row.get("category"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "target_type": target_kind,
                "target_token": target_token,
                "anchor_token": anchor_token,
                "image_path": image_full,
            }
        )
        if len(rows) >= args.rows_limit:
            break
    return rows


def _find_subsequence_indices(sequence: list[int], needle: list[int]) -> list[list[int]]:
    if not sequence or not needle:
        return []
    out: list[list[int]] = []
    for start in range(0, len(sequence) - len(needle) + 1):
        if sequence[start : start + len(needle)] == needle:
            out.append(list(range(start, start + len(needle))))
    return out


def _build_atlas_graphs() -> dict[str, dict[int, set[int]]]:
    meta = atlas_metadata()
    token_maps = _token_index_maps()

    tile_neighbors = {tile_id: set() for tile_id in range(19)}
    node_neighbors = {node_id: set() for node_id in range(54)}
    port_neighbors = {port_id: set() for port_id in range(9)}

    # Edge id (canonical tuple) -> edge index
    edge_id_to_index = token_maps["edge"]
    edge_token_to_index = {tok: idx for idx, tok in enumerate(token_maps["edge_tokens"])}
    edge_graph = {idx: set() for idx in range(len(edge_token_to_index))}

    # Build tile neighbors from shared edges.
    edge_to_tiles: dict[tuple[int, int], set[int]] = defaultdict(set)
    for tile in meta["tiles"]:
        tile_id = int(tile["id"])
        for raw_edge in tile["edges"].values():
            edge_tuple = canonical_edge((raw_edge[0], raw_edge[1]))
            edge_to_tiles[edge_tuple].add(tile_id)

    for edge_tiles in edge_to_tiles.values():
        if len(edge_tiles) < 2:
            continue
        tiles = sorted(edge_tiles)
        t0 = tiles[0]
        t1 = tiles[1]
        tile_neighbors[t0].add(t1)
        tile_neighbors[t1].add(t0)

    # Node neighbors and edge neighbors from canonical edge list.
    edge_indices_by_node: dict[int, set[int]] = defaultdict(set)
    for tile in meta["edges"]:
        edge_tuple = canonical_edge((tile["id"][0], tile["id"][1]))
        edge_index = edge_token_to_index.get(edge_token(edge_tuple))
        if edge_index is None:
            continue
        n0, n1 = edge_tuple
        node_neighbors[n0].add(n1)
        node_neighbors[n1].add(n0)
        edge_indices_by_node[n0].add(edge_index)
        edge_indices_by_node[n1].add(edge_index)

    for edges_at_node in edge_indices_by_node.values():
        edges_sorted = sorted(edges_at_node)
        for i, e0 in enumerate(edges_sorted):
            for e1 in edges_sorted[i + 1 :]:
                edge_graph[e0].add(e1)
                edge_graph[e1].add(e0)

    # Ports are adjacent if they share any node.
    node_to_ports: dict[int, set[int]] = defaultdict(set)
    for port in meta["ports"]:
        port_id = int(port["id"])
        for node_id in port["attached_nodes"]:
            node_to_ports[node_id].add(port_id)
    for ports in node_to_ports.values():
        for p in ports:
            port_neighbors[p].update(ports - {p})

    # Reindex edge neighbors into token index space.
    edge_neighbors_by_index: dict[int, set[int]] = {}
    for token, idx in edge_token_to_index.items():
        edgeset = edge_graph.get(idx, set())
        if edgeset:
            edge_neighbors_by_index[idx] = edgeset
        else:
            edge_neighbors_by_index[idx] = set()

    return {
        "tile": {tile_id: neighbors for tile_id, neighbors in tile_neighbors.items()},
        "node": {node_id: neighbors for node_id, neighbors in node_neighbors.items()},
        "edge": edge_neighbors_by_index,
        "port": port_neighbors,
    }


def _token_index_maps() -> dict[str, Any]:
    meta = atlas_metadata()
    tile_map = {tile_token(tile["id"]): tile["id"] for tile in meta["tiles"]}
    node_map = {node_token(i): i for i in range(54)}
    edge_tokens = [edge_token(canonical_edge((edge["id"][0], edge["id"][1]))) for edge in meta["edges"]]
    edge_map = {tok: idx for idx, tok in enumerate(edge_tokens)}
    edge_token_list = edge_tokens
    port_map = {port_token(port["id"]): port["id"] for port in meta["ports"]}
    return {
        "tile": tile_map,
        "node": node_map,
        "edge": edge_map,
        "port": port_map,
        "edge_tokens": edge_token_list,
    }


def _choose_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def _torch_dtype(name: str, fallback: str | None = None) -> torch.dtype:
    if name == "auto":
        name = fallback or "fp32"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _load_model_and_processor(args: argparse.Namespace):
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    from catanbench.tokens import add_tokens_to_tokenizer

    if args.bits in (4, 8):
        import bitsandbytes  # noqa: F401

    processor = AutoProcessor.from_pretrained(args.model_id)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"

    quant = None
    dtype = _torch_dtype(args.dtype, fallback="bf16")
    if args.bits == 4:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
    elif args.bits == 8:
        quant = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=dtype)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        device_map="auto" if args.device == "auto" else None,
        torch_dtype=dtype,
        quantization_config=quant,
        attn_implementation="sdpa" if args.disable_flash_attn2 else "flash_attention_2",
    )

    added = add_tokens_to_tokenizer(processor.tokenizer)
    tokenizer_len = len(processor.tokenizer)
    current_rows = model.get_input_embeddings().num_embeddings
    if tokenizer_len > current_rows:
        model.resize_token_embeddings(tokenizer_len, pad_to_multiple_of=64)
    print(
        f"added_catan_tokens={added}; tokenizer_len={tokenizer_len};"
        f" embedding_rows={model.get_input_embeddings().num_embeddings}"
    )

    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        print(f"loaded_adapter={args.adapter_dir}")

    model.eval()
    return processor, model


def _build_model_inputs(processor: Any, row: dict[str, Any], prompt_prefix: str) -> Any:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(row["image_path"])},
                {
                    "type": "text",
                    "text": f"{prompt_prefix}\n\nQuestion: {row['question']}",
                },
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


def _collect_hidden_vectors(
    rows: list[dict[str, Any]],
    processor: Any,
    model: Any,
    device: torch.device,
    prompt_prefix: str,
) -> tuple[dict[str, dict[int, list[tuple[torch.Tensor, int]]]], dict[str, int]]:
    token_maps = _token_index_maps()
    tk_to_index = {k: v for k, v in token_maps.items() if k in {"tile", "node", "edge", "port"}}

    buckets: dict[str, dict[int, list[tuple[torch.Tensor, int]]]] = {
        "tile": defaultdict(list),
        "node": defaultdict(list),
        "edge": defaultdict(list),
        "port": defaultdict(list),
    }
    used_rows = {"tile": 0, "node": 0, "edge": 0, "port": 0}

    for row in rows:
        token = row["anchor_token"]
        target_type = row["target_type"]
        token_indices = tk_to_index[target_type]
        target_index = token_indices.get(token)
        if target_index is None:
            continue

        token_ids = processor.tokenizer.encode(token, add_special_tokens=False)
        if not token_ids:
            continue

        inputs = _build_model_inputs(processor, row, prompt_prefix)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        input_ids = inputs["input_ids"][0].tolist()
        positions = _find_subsequence_indices(input_ids, token_ids)
        if not positions:
            continue
        pos = positions[0]
        used_rows[target_type] += 1

        for layer_idx, layer_state in enumerate(hidden_states):
            vec = layer_state[0, pos, :].mean(dim=0).detach().cpu()
            buckets[target_type][layer_idx].append((vec, target_index))

    return buckets, used_rows


def _evaluate_identity_by_layer(
    layer_data: dict[int, list[tuple[torch.Tensor, int]]],
    min_samples_per_class: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for layer_idx, samples in sorted(layer_data.items()):
        if not samples:
            summary[str(layer_idx)] = {"samples": 0}
            continue

        labels = np.array([int(y) for _, y in samples], dtype=np.int32)
        feats = np.stack([vec.numpy() for vec, _ in samples], axis=0).astype(np.float32)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)

        cls, counts = np.unique(labels, return_counts=True)
        class_counts = {int(c): int(n) for c, n in zip(cls.tolist(), counts.tolist())}
        usable = [c for c, n in class_counts.items() if n >= min_samples_per_class]
        if not usable:
            summary[str(layer_idx)] = {
                "samples": int(len(labels)),
                "classes": int(len(cls)),
                "usable_classes": 0,
                "id_loo_accuracy": 0.0,
                "tested": 0,
                "note": "too_few_per_class",
            }
            continue

        mask = np.isin(labels, usable)
        labels = labels[mask]
        feats = feats[mask]
        if len(labels) == 0:
            summary[str(layer_idx)] = {"samples": 0, "classes": int(len(cls)), "usable_classes": 0}
            continue

        sum_by_class: dict[int, np.ndarray] = {}
        count_by_class: dict[int, int] = {}
        for c in usable:
            idx = np.where(labels == c)[0]
            if len(idx):
                sum_by_class[c] = feats[idx].sum(axis=0)
                count_by_class[c] = int(len(idx))

        means = {c: sum_by_class[c] / count_by_class[c] for c in count_by_class}
        correct = 0
        tested = 0

        for sample_vec, lbl in zip(feats, labels):
            if lbl not in means or count_by_class[lbl] <= 1:
                continue
            test_means = dict(means)
            test_means[lbl] = (sum_by_class[lbl] - sample_vec) / (count_by_class[lbl] - 1)
            c_ids = sorted(test_means)
            centroid_matrix = np.stack([test_means[c] for c in c_ids], axis=0)
            centroid_matrix = centroid_matrix / (np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-9)
            pred_idx = int(np.argmax(sample_vec @ centroid_matrix.T))
            pred = c_ids[pred_idx]
            tested += 1
            if pred == int(lbl):
                correct += 1

        summary[str(layer_idx)] = {
            "samples": int(len(labels)),
            "classes": int(len(set(labels.tolist()))),
            "usable_classes": int(len(usable)),
            "id_loo_accuracy": float(correct / tested) if tested else 0.0,
            "tested": int(tested),
            "class_counts": {str(k): v for k, v in sorted(class_counts.items())},
        }

    return summary


def _evaluate_topology_by_layer(
    layer_data: dict[int, list[tuple[torch.Tensor, int]]],
    graph_neighbors: dict[int, set[int]],
    min_samples_per_class: int,
    topk_values: list[int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for layer_idx, samples in sorted(layer_data.items()):
        if not samples:
            summary[str(layer_idx)] = {"samples": 0}
            continue

        labels = np.array([int(y) for _, y in samples], dtype=np.int32)
        feats = np.stack([vec.numpy() for vec, _ in samples], axis=0).astype(np.float32)
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)

        cls, counts = np.unique(labels, return_counts=True)
        class_counts = {int(c): int(n) for c, n in zip(cls.tolist(), counts.tolist())}
        usable = [c for c, n in class_counts.items() if n >= min_samples_per_class]
        if len(usable) < 2:
            summary[str(layer_idx)] = {
                "samples": int(len(labels)),
                "classes": int(len(set(labels.tolist()))),
                "usable_classes": int(len(usable)),
                "note": "too_few_classes",
            }
            continue

        means: dict[int, np.ndarray] = {}
        for c in usable:
            idx = np.where(labels == c)[0]
            means[c] = feats[idx].mean(axis=0)

        ids = sorted(means)
        mean_matrix = np.stack([means[c] for c in ids], axis=0)
        normed = mean_matrix / (np.linalg.norm(mean_matrix, axis=1, keepdims=True) + 1e-9)
        sim = normed @ normed.T

        metric_out: dict[str, Any] = {}
        for k in topk_values:
            precision_scores: list[float] = []
            recall_scores: list[float] = []
            mrr_scores: list[float] = []
            for i, node_id in enumerate(ids):
                true_neighbors = graph_neighbors.get(node_id, set())
                if not true_neighbors:
                    continue
                row = sim[i].copy()
                row[i] = -2.0
                order = np.argsort(-row)
                pred = [ids[j] for j in order[:k]]
                hit_set = set(pred) & set(true_neighbors)
                precision_scores.append(len(hit_set) / k)
                recall_scores.append(len(hit_set) / len(true_neighbors))
                for rank, j in enumerate(order, start=1):
                    if ids[j] in true_neighbors:
                        mrr_scores.append(1.0 / rank)
                        break

            metric_out[f"top_{k}"] = {
                "precision": float(np.mean(precision_scores)) if precision_scores else 0.0,
                "recall": float(np.mean(recall_scores)) if recall_scores else 0.0,
                "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
                "nodes_with_neighbors": int(len(precision_scores)),
            }

        summary[str(layer_idx)] = {
            "samples": int(len(labels)),
            "classes": int(len(set(labels.tolist()))),
            "usable_classes": int(len(usable)),
            "metrics": metric_out,
        }

    return summary


def main() -> int:
    args = parse_args()
    qa_jsonl = args.qa_jsonl
    dataset_root = qa_jsonl.parent
    manifest = _load_manifest(args.manifest_jsonl)
    rows = _load_rows(qa_jsonl, manifest, dataset_root, args)
    if not rows:
        raise SystemExit("no rows found after filtering")

    processor, model = _load_model_and_processor(args)
    device = _choose_device(args.device)
    model = model.to(device)

    print(
        f"using_rows={len(rows)} categories={sorted({row['category'] for row in rows})}"
        f" parts={sorted({row['target_type'] for row in rows})} on_device={device}"
    )

    vector_buckets, used_rows = _collect_hidden_vectors(
        rows=rows,
        processor=processor,
        model=model,
        device=device,
        prompt_prefix=args.prompt_prefix,
    )

    atlas_graphs = _build_atlas_graphs()
    topk_values = [int(v) for v in args.topk_adj.split(",") if v.strip()]

    report: dict[str, Any] = {
        "model_id": args.model_id,
        "adapter_dir": args.adapter_dir,
        "rows_requested": int(args.rows_limit),
        "rows_used_by_type": used_rows,
        "dataset_root": str(dataset_root),
        "topk": topk_values,
        "parts": {},
    }

    for part in ("tile", "node", "edge", "port"):
        id_report = _evaluate_identity_by_layer(
            vector_buckets[part],
            min_samples_per_class=args.min_samples_per_class,
        )
        topo_report = _evaluate_topology_by_layer(
            vector_buckets[part],
            graph_neighbors=atlas_graphs.get(part, {}),
            min_samples_per_class=args.min_samples_per_class,
            topk_values=topk_values,
        )
        report["parts"][part] = {
            "identity": id_report,
            "topology": topo_report,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
