"""Build BOTH panels: python -m sft.cartesian_eval.scaling_dataset [--source PATH] [--output PATH]."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .contracts import compact, digest, object_list, object_map, parse_json, require, sha256, text
from .scaling import PARENT_SHA256, SCHEMA, VARIANTS, derived_metadata, validate_rows
from .scaling_geometry import VERSION, mapping_artifact, mapping_sha256, scaling_prompt
from .shorthand import validate_rows as validate_shorthand_rows

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "artifacts/generated/sft/cartesian_h_eval_v1/eval.jsonl"
DEFAULT_OUTPUT = DEFAULT_SOURCE.parent.parent / VERSION


def build_rows(source: Path = DEFAULT_SOURCE) -> dict[str, list[dict[str, object]]]:
    """Project both variants from the actual validated, byte-pinned sparse-h artifact."""
    raw = source.read_bytes()
    require(sha256(raw) == PARENT_SHA256, "immutable parent source hash mismatch")
    parents = [parse_json(line) for line in raw.decode("utf-8").splitlines()]
    validate_shorthand_rows(parents)
    panels: dict[str, list[dict[str, object]]] = {}
    for variant in VARIANTS:
        rows = []
        for parent in parents:
            metadata = derived_metadata(text(parent["id"]), object_map(parent["metadata"]), variant)
            messages = object_list(parent["messages"])
            prompt = scaling_prompt(text(object_map(messages[0])["content"]), variant)
            require(sha256(prompt.encode("utf-8")) == metadata["prompt_sha256"],
                    "actual parent prompt projection mismatch")
            rows.append({
                **parent, "schema": SCHEMA, "metadata": metadata,
                "messages": [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": metadata["answer"]}],
            })
        validate_rows(rows)
        panels[variant] = rows
    return panels


def render_preview(panels: Mapping[str, Sequence[Mapping[str, object]]]) -> str:
    lines = [
        "# Factor-four coordinate evaluation panels", "",
        "Two variants of the same 200 sparse-h cases (198 test, 2 validation) in original order.",
        "Intended for stock Qwen3.8-27B inference only, with no finetuning or added tokenizer tokens.",
        "Construction only: no model run performed. Only messages[0].content is model input.",
        "Static inventories retain 154 entities. Dynamic boards retain occupied nodes/edges,",
        "all 19 tiles, nine ports, the robber and the unchanged empty default.",
        "scaled_h: ordinary equal-unit Cartesian geometry, side 4, h=sqrt(3), integer y.",
        "integer_xy: integer grid addresses; physical position (sqrt(3)*x,y), side 4.",
        f"Parent eval.jsonl SHA-256: `{PARENT_SHA256}`.", "",
    ]
    for variant, rows in panels.items():
        lines.extend([f"## {variant}", "", f"Mapping SHA-256: `{mapping_sha256(variant)}`.", ""])
        selected: dict[str, Mapping[str, object]] = {}
        for row in rows:
            operation = text(row["task_type"])
            if operation not in selected or object_map(selected[operation]["metadata"])["answer"] == "NONE":
                selected[operation] = row
        for operation, row in selected.items():
            metadata, messages = object_map(row["metadata"]), object_list(row["messages"])
            lines.extend([
                f"### {operation}", "", f"Case: `{row['id']}`.",
                f"Source: `{metadata['source_id']}` ({metadata['split']}).",
                f"Component counts: `{compact(metadata['component_counts'])}`.", "",
                "```text", text(object_map(messages[0])["content"]), "```", "", "Gold:",
                "```text", text(object_map(messages[1])["content"]), "```", "",
            ])
    return "\n".join(lines)


def build_dataset(
    output: Path = DEFAULT_OUTPUT, *, source: Path = DEFAULT_SOURCE,
) -> dict[str, object]:
    """Validate everything before exclusive new-directory writes; never overwrite history."""
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"destination already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"destination parent does not exist: {output.parent}")
    panels = build_rows(source)
    validations = {variant: validate_rows(rows) for variant, rows in panels.items()}
    mapping: dict[str, object] = {
        "schema": SCHEMA, "version": VERSION, "kind": "mapping",
        "variants": {variant: mapping_artifact(variant) for variant in VARIANTS},
    }
    payloads = {f"{variant}.jsonl": "".join(compact(row) + "\n" for row in rows)
                for variant, rows in panels.items()}
    payloads.update({
        "mapping.json": json.dumps(mapping, indent=2, sort_keys=True) + "\n",
        "preview.md": render_preview(panels),
    })
    manifest: dict[str, object] = {
        "schema": SCHEMA, "version": VERSION, "kind": "manifest", "rows": 400,
        "variants": list(VARIANTS), "rows_per_variant": 200, "mapping_entities_per_variant": 154,
        "mapping_sha256": digest(mapping), "mapping_hash_scope": "canonical JSON mapping.json payload",
        "variant_mapping_sha256": {variant: mapping_sha256(variant) for variant in VARIANTS},
        "parent": {"path": str(source.resolve()), "sha256": PARENT_SHA256, "rows": 200,
                   "version": "cartesian_h_eval_v1"},
        "parent_hash_scope": "exact sparse-h eval.jsonl bytes; reconstructed and validated in source order",
        "case_identity": "parent id, row_id and source_id preserved in original order in BOTH panels",
        "intent": {"model": "Qwen3.8-27B", "weights": "stock", "finetuning": False,
                   "inference_only": True, "model_run_performed": False, "run_both_variants": True},
        "added_tokenizer_tokens": [], "admitted_for_training": False,
        "model_input": "messages[0].content only; metadata and mapping.json are oracle-only",
        "selection": "All 200 cases from the validated, hash-pinned cartesian_h_eval_v1/eval.jsonl",
        "validation": validations,
        "files": {name: {"path": name, "sha256": sha256(payload.encode("utf-8"))}
                  for name, payload in payloads.items()},
    }
    payloads["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    encoded = {name: payload.encode("utf-8") for name, payload in payloads.items()}
    output.mkdir()
    for name, raw in encoded.items():
        with (output / name).open("xb") as handle:
            handle.write(raw)
    return {
        "output_dir": str(output), "rows": 400, "rows_per_variant": 200,
        "variants": list(VARIANTS), "mapping_entities_per_variant": 154,
        "parent_sha256": PARENT_SHA256, "mapping_sha256": digest(mapping),
        "by_split_per_variant": {variant: result["by_split"] for variant, result in validations.items()},
        "files": {name: {"path": str(output / name), "sha256": sha256(raw)}
                  for name, raw in encoded.items()},
        "model_run_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.output, source=args.source), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
