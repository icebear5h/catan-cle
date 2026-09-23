"""Build once: python -m sft.cartesian_eval.shorthand_dataset [--source PATH] [--output PATH]."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .contracts import compact, object_list, object_map, parse_json, require, sha256, text
from .dataset import validate_rows as validate_cartesian_rows
from .shorthand import PARENT_SHA256, SCHEMA, VERSION, derived_metadata, validate_rows
from .shorthand_geometry import mapping_artifact, mapping_sha256, sparse_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "artifacts/generated/sft/cartesian_eval_v2/eval.jsonl"
DEFAULT_OUTPUT = DEFAULT_SOURCE.parent.parent / VERSION


def build_rows(source: Path = DEFAULT_SOURCE) -> list[dict[str, object]]:
    """Read the real dense artifact; filtering always starts with its actual user message."""
    raw = source.read_bytes()
    require(sha256(raw) == PARENT_SHA256, "immutable parent source hash mismatch")
    parents = [parse_json(line) for line in raw.decode("utf-8").splitlines()]
    validate_cartesian_rows(parents)
    rows = []
    for parent in parents:
        metadata = derived_metadata(text(parent["id"]), object_map(parent["metadata"]))
        messages = object_list(parent["messages"])
        prompt, counts = sparse_prompt(text(object_map(messages[0])["content"]))
        require(counts == metadata["component_counts"], "actual parent component mismatch")
        require(sha256(prompt.encode("utf-8")) == metadata["prompt_sha256"],
                "actual parent prompt projection mismatch")
        rows.append({
            **parent, "schema": SCHEMA, "metadata": metadata,
            "messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": metadata["answer"]}],
        })
    validate_rows(rows)
    return rows


def render_preview(rows: Sequence[Mapping[str, object]]) -> str:
    selected: dict[str, Mapping[str, object]] = {}
    for row in rows:
        operation = text(row["task_type"])
        if operation not in selected or object_map(selected[operation]["metadata"])["answer"] == "NONE":
            selected[operation] = row
    lines = [
        "# Sparse exact-h Cartesian evaluation panel", "",
        "200 original cases (198 test, 2 validation); stock Qwen3.8-27B, inference only.",
        "No finetuning or added tokenizer tokens. Literal h uses ordinary native vocabulary.",
        "This artifact records construction, not model inference. Only messages[0].content is input.",
        "All static inventories retain 154 entities. Dynamic boards omit only empty nodes/edges,",
        "with an explicit empty default; all 19 tile records, nine ports and the robber remain.",
        f"Parent eval.jsonl SHA-256: `{PARENT_SHA256}`.",
        f"Mapping canonical JSON SHA-256: `{mapping_sha256()}`.", "",
    ]
    for operation, row in selected.items():
        metadata = object_map(row["metadata"])
        messages = object_list(row["messages"])
        lines.extend([
            f"## {operation}", "", f"Case: `{row['id']}`.",
            f"Source: `{metadata['source_id']}` ({metadata['split']}).",
            f"Component counts: `{compact(metadata['component_counts'])}`.", "",
            "```text", text(object_map(messages[0])["content"]), "```", "", "Gold:",
            "```text", text(object_map(messages[1])["content"]), "```", "",
        ])
    return "\n".join(lines)


def build_dataset(
    output: Path = DEFAULT_OUTPUT, *, source: Path = DEFAULT_SOURCE,
) -> dict[str, object]:
    """Validate before exclusive new-directory writes; never replace an existing panel."""
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"destination already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"destination parent does not exist: {output.parent}")
    rows = build_rows(source)
    validation = validate_rows(rows)
    payloads = {
        "eval.jsonl": "".join(compact(row) + "\n" for row in rows),
        "mapping.json": json.dumps(mapping_artifact(), indent=2, sort_keys=True) + "\n",
        "preview.md": render_preview(rows),
    }
    manifest: dict[str, object] = {
        "schema": SCHEMA, "version": VERSION, "kind": "manifest", "rows": len(rows),
        "representation": "cartesian_h", "mapping_entities": 154,
        "mapping_sha256": mapping_sha256(), "mapping_hash_scope": "canonical JSON mapping.json payload",
        "parent": {"path": str(source.resolve()), "sha256": PARENT_SHA256, "rows": 200,
                   "version": "cartesian_eval_v2"},
        "parent_hash_scope": "exact dense eval.jsonl bytes; reconstructed and validated in source order",
        "case_identity": "parent id, row_id and source_id preserved in original order",
        "component_counts": validation["component_counts"],
        "intent": {"model": "Qwen3.8-27B", "weights": "stock", "finetuning": False,
                   "inference_only": True, "model_run_performed": False},
        "added_tokenizer_tokens": [], "admitted_for_training": False,
        "model_input": "messages[0].content only; metadata and mapping.json are oracle-only",
        "selection": "All 200 cases from the validated, hash-pinned cartesian_eval_v2/eval.jsonl",
        "projection": "h=sqrt(3)/4; exact decimal y; omit only empty N/E board entries",
        "limitations": ["Two original validation port-incidence cases retain their split.",
                        "Static topology and shared dynamic states are inherited from the parent."],
        "validation": validation,
        "files": {name: {"path": name, "sha256": sha256(payload.encode("utf-8"))}
                  for name, payload in payloads.items()},
    }
    payloads["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    encoded = {name: payload.encode("utf-8") for name, payload in payloads.items()}
    output.mkdir()
    for name, raw in encoded.items():
        with (output / name).open("xb") as handle:
            handle.write(raw)
    return {"output_dir": str(output), "rows": len(rows), "mapping_entities": 154,
            "parent_sha256": PARENT_SHA256, "mapping_sha256": mapping_sha256(),
            "by_split": validation["by_split"], "component_counts": validation["component_counts"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.output, source=args.source), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
