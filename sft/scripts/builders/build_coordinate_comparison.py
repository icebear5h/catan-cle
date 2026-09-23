"""Build the immutable 200-case/400-row matched inference panel, entirely offline.

Run: python -m sft.scripts.builders.build_coordinate_comparison [--source-root PATH] [--output PATH]
The destination must not exist. Canonical targets are metadata-only; mapping.json
is an audit artifact, never a model input. No tokenizer or checkpoint is loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from data_pipeline.board_recognition.sources import file_sha256
from sft.board.coordinate_comparison import (
    DEFAULT_SOURCE,
    PAIR_QUOTAS,
    PROJECT_ROOT,
    REPRESENTATIONS,
    SCHEMA,
    build_comparison_rows,
    load_source_rows,
    mapping_artifact,
    mapping_sha256,
    validate_comparison_rows,
)
from sft.json_types import JsonDict, JsonLikeDict, JsonValue, as_dict, as_list, as_str, json_path

DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/generated/sft/coordinate_comparison_v1"


def _content(message: JsonValue) -> str:
    return as_str(as_dict(message)["content"])


def prompt_length_report(rows: list[JsonDict]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for representation in REPRESENTATIONS:
        lengths = [len(_content(as_list(r["messages"])[0])) for r in rows
                   if json_path(r, "metadata", "representation") == representation]
        result[representation] = {"min": min(lengths), "max": max(lengths), "mean": mean(lengths)}
    return result


def render_preview(rows: list[JsonDict]) -> str:
    selected: list[JsonDict] = []
    for task in PAIR_QUOTAS:
        candidates = [r for r in rows if r["task_type"] == task
                      and json_path(r, "metadata", "representation") == "atlas"]
        # Prefer a nonempty real answer for inspection, never alter selected cases.
        selected.append(next((r for r in candidates if json_path(r, "metadata", "answer") != "NONE"), candidates[0]))
    lines = [
        "# Matched atlas IDs and integer coordinates", "",
        "200 canonical cases, 400 adjacent atlas/coordinate rows. Every example below is an actual rendered row.",
        "Only the prompt is model input; displayed gold/source information is for human inspection.", "",
        "Inference-only usability on an atlas-trained checkpoint, not equal-budget retraining. Coordinates expose geometry.",
        "Static topology may have training exposure; original directional holdouts and dynamic source provenance are retained.",
        "Two port-incidence cases use validation (<P03>, <P08>) because test has no port-origin incidence queries.", "",
        "## Prompt lengths (characters, not tokenizer counts)", "", "```json",
        json.dumps(prompt_length_report(rows), indent=2, sort_keys=True), "```", "",
    ]
    by_pair: dict[str, list[JsonDict]] = {as_str(json_path(r, "metadata", "pair_id")): [] for r in rows}
    for row in rows:
        by_pair[as_str(json_path(row, "metadata", "pair_id"))].append(row)
    for chosen in selected:
        metadata = as_dict(chosen["metadata"])
        lines.extend([f"## {metadata['area']} / {metadata['operation']}", "",
                      f"Source: `{metadata['source_id']}` ({metadata['source_split']}).", ""])
        for row in by_pair[as_str(metadata["pair_id"])]:
            prompt, answer = [_content(m) for m in as_list(row["messages"])]
            lines.extend([f"### {json_path(row, 'metadata', 'representation')} ({len(prompt)} prompt characters)", "",
                          "```text", prompt, "```", "", "Gold:", "```text", answer, "```", ""])
    return "\n".join(lines)


def build_dataset(output_dir: Path = DEFAULT_OUTPUT, *, source_root: Path = DEFAULT_SOURCE) -> JsonLikeDict:
    output_dir, source_root = Path(output_dir).resolve(), Path(source_root).resolve()
    if output_dir.exists():
        raise FileExistsError(f"destination already exists: {output_dir}")
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"destination parent does not exist: {output_dir.parent}")
    sources, source_audit = load_source_rows(source_root)
    rows = build_comparison_rows(sources)
    validation = validate_comparison_rows(rows)
    preview = render_preview(rows)
    mapping = mapping_artifact()
    # All validation precedes the first write; mkdir remains an exclusive race-safe gate.
    output_dir.mkdir()
    (output_dir / "paired.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    (output_dir / "mapping.json").write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    (output_dir / "preview.md").write_text(preview)
    manifest: JsonLikeDict = {
        "schema": SCHEMA, "kind": "manifest", "source": source_audit,
        "rows": 400, "pairs": 200, "pair_quotas": dict(PAIR_QUOTAS),
        "mapping_sha256": mapping_sha256(), "mapping_hash_scope": "canonical JSON of mapping.json payload",
        "source_hash_scope": "source file and physical line byte SHA256 (including newline); "
                             "source row/metadata/prompt/answer canonical JSON SHA256",
        "selection": "Complete test direction/direction_choice/neighbors/piece_owner; two per incidence relation, "
                     "validation P03/P08 for P->N; one per owned_nodes cell; two per road/incident-road roster/polarity cell. "
                     "Least-used dynamic state then original ID; source physical order; adjacent atlas, coordinates arms.",
        "limitations": [
            "Inference-only representation usability on an atlas-trained checkpoint, not equal-training-budget evidence.",
            "Coordinates expose geometry; original static-topology training exposure remains.",
            "Two port-origin incidence cases are validation, all other cases retain their test split.",
            "Dynamic rows can share source states; paired rows are not independent observations.",
            "Character lengths are not tokenizer lengths; checkpoint-specific token preflight belongs to the launcher.",
        ],
        "model_input": "messages[0].content only; no mapping.json crosswalk or canonical metadata",
        "added_tokenizer_tokens": [], "prompt_lengths_chars": prompt_length_report(rows),
        "validation": validation,
        "code_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in (
                Path(__file__).resolve(),
                PROJECT_ROOT / "sft/board/coordinate_comparison/__init__.py",
                PROJECT_ROOT / "sft/board/coordinate_comparison/_constants.py",
                PROJECT_ROOT / "sft/board/coordinate_comparison/_mapping.py",
                PROJECT_ROOT / "sft/board/coordinate_comparison/_rows.py",
                PROJECT_ROOT / "sft/board/coordinate_comparison/_scoring.py",
                PROJECT_ROOT / "sft/board/coordinate_comparison/_summaries.py",
                PROJECT_ROOT / "sft/board/symbolic_board_tasks/__init__.py",
                PROJECT_ROOT / "sft/board/symbolic_board_tasks/_constants.py",
                PROJECT_ROOT / "sft/board/symbolic_board_tasks/_geometry.py",
                PROJECT_ROOT / "sft/board/symbolic_board_tasks/_contracts.py",
                PROJECT_ROOT / "sft/board/symbolic_board_tasks/_solve.py",
                PROJECT_ROOT / "sft/board/symbolic_board_tasks/_scoring.py",
                PROJECT_ROOT / "sft/board/board_fluency_scoring.py",
                PROJECT_ROOT / "evals/catan_board_bench/tokens/__init__.py",
                PROJECT_ROOT / "evals/catan_board_bench/tokens/atlas.py",
                PROJECT_ROOT / "evals/catan_board_bench/tokens/manifest.py",
                PROJECT_ROOT / "evals/catan_board_bench/tokens/vocabulary.py",
            )
        },
        "files": {name: {"path": name, "sha256": file_sha256(output_dir / name)}
                  for name in ("paired.jsonl", "mapping.json", "preview.md")},
    }
    # manifest.json deliberately has no self-hash.
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"output_dir": str(output_dir), "rows": len(rows), "pairs": validation["pairs"],
            "mapping_sha256": mapping_sha256(), "unique_dynamic_states": validation["unique_dynamic_states"],
            "prompt_lengths_chars": manifest["prompt_lengths_chars"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.output, source_root=args.source_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
