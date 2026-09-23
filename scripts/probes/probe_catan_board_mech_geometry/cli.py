"""Argument parsing and the probe run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.probes.probe_catan_board_mech_geometry.atlas import build_atlas_graphs
from scripts.probes.probe_catan_board_mech_geometry.constants import (
    CANONICAL_CATEGORY_MAP,
    PARTS,
)
from scripts.probes.probe_catan_board_mech_geometry.metrics import (
    JsonDict,
    evaluate_identity_by_layer,
    evaluate_topology_by_layer,
)
from scripts.probes.probe_catan_board_mech_geometry.model_io import (
    choose_device,
    collect_hidden_vectors,
    load_model_and_processor,
)
from scripts.probes.probe_catan_board_mech_geometry.rows import load_manifest, load_rows

# The pre-split module path stays the advertised program name and description.
PROG = "probe_catan_board_mech_geometry.py"
DESCRIPTION = """Mechanistic geometry probe for Catan atlas parts (tile/node/edge/port)
using Qwen visual-language residuals."""

__all__ = ["DESCRIPTION", "PROG", "main", "parse_args"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=PROG, description=DESCRIPTION)
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


def main() -> int:
    args = parse_args()
    qa_jsonl = args.qa_jsonl
    dataset_root = qa_jsonl.parent
    manifest = load_manifest(args.manifest_jsonl)
    rows = load_rows(qa_jsonl, manifest, dataset_root, args)
    if not rows:
        raise SystemExit("no rows found after filtering")

    processor, model = load_model_and_processor(args)
    device = choose_device(args.device)
    model = model.to(device)

    print(
        f"using_rows={len(rows)} categories={sorted({str(row['category']) for row in rows})}"
        f" parts={sorted({row['target_type'] for row in rows})} on_device={device}"
    )

    vector_buckets, used_rows = collect_hidden_vectors(
        rows=rows,
        processor=processor,
        model=model,
        device=device,
        prompt_prefix=args.prompt_prefix,
    )

    atlas_graphs = build_atlas_graphs()
    topk_values = [int(v) for v in args.topk_adj.split(",") if v.strip()]

    parts_report: JsonDict = {}
    report: JsonDict = {
        "model_id": args.model_id,
        "adapter_dir": args.adapter_dir,
        "rows_requested": int(args.rows_limit),
        "rows_used_by_type": {key: count for key, count in used_rows.items()},
        "dataset_root": str(dataset_root),
        "topk": list(topk_values),
        "parts": parts_report,
    }

    for part in PARTS:
        id_report = evaluate_identity_by_layer(
            vector_buckets[part],
            min_samples_per_class=args.min_samples_per_class,
        )
        topo_report = evaluate_topology_by_layer(
            vector_buckets[part],
            graph_neighbors=atlas_graphs.get(part, {}),
            min_samples_per_class=args.min_samples_per_class,
            topk_values=topk_values,
        )
        parts_report[part] = {
            "identity": id_report,
            "topology": topo_report,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote report: {args.output}")
    return 0
