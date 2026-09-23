"""Generate the counterfactual-paired curriculum dataset."""

from __future__ import annotations

import copy
from pathlib import Path

from evals.catan_board_bench.render import RenderStyle, render_contract_image
from evals.catan_board_bench.tokens import atlas_metadata_json
from scripts.board_recognition.build_catan_board_recognition_curriculum.contracts import (
    make_stage_contract,
    set_sample_identity,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.counterfactual import (
    apply_counterfactual,
    apply_declared_value,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.jsonio import (
    file_sha256,
    load_leakage_ledger,
    prepare_output_dir,
    write_json,
    write_jsonl,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.labels import dense_labels
from scripts.board_recognition.build_catan_board_recognition_curriculum.metadata import (
    build_metadata,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.paths import (
    DEFAULT_LEAKAGE_LEDGER,
    DEFAULT_SPEC_PATH,
    DEFAULT_STYLE_PATH,
    repository_relative,
    resolve_project_path,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.shapes import (
    ENTITY_TYPES,
    SAMPLE_SCHEMA,
    JsonDict,
    integer,
    obj,
    objs,
    strings,
    text,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.spec import (
    load_render_style,
    load_spec,
    split_for_group,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.validate import (
    validate_dataset,
)
from sft.scripts.builders.build_node_factor_dataset import build_indices

__all__ = ["build_dataset"]


def build_dataset(
    *,
    output_dir: Path,
    spec_path: Path = DEFAULT_SPEC_PATH,
    image_size: int | None = None,
    pairs_per_entity_type: int = 1,
    seed: int = 381_427,
    overwrite: bool = False,
) -> JsonDict:
    if pairs_per_entity_type < 1:
        raise ValueError("pairs_per_entity_type must be positive")
    spec = load_spec(spec_path)
    render_config = obj(spec["render"], "spec render")
    resolved_image_size = image_size or integer(render_config["image_size"], "image_size")
    if resolved_image_size < 64 or resolved_image_size % 16:
        raise ValueError("image_size must be at least 64 and divisible by 16")
    prepare_output_dir(output_dir, overwrite=overwrite)
    contracts_dir = output_dir / "contracts"
    labels_dir = output_dir / "dense_labels"
    images_dir = output_dir / "images"
    splits_dir = output_dir / "splits"
    for directory in (contracts_dir, labels_dir, images_dir, splits_dir):
        directory.mkdir(parents=True, exist_ok=True)

    ledger, _excluded_game_ids = load_leakage_ledger(DEFAULT_LEAKAGE_LEDGER)
    style_path = resolve_project_path(
        text(render_config.get("style_config", str(DEFAULT_STYLE_PATH)), "style_config")
    )
    style = load_render_style(style_path)
    atlas = atlas_metadata_json()
    indices = build_indices(atlas)
    vocabularies = obj(spec["class_vocabularies"], "class_vocabularies")
    colors = tuple(strings(vocabularies["colors"], "colors"))
    manifest_rows: list[JsonDict] = []
    group_index = 0
    splits = obj(spec["splits"], "spec splits")

    for stage_index, stage in enumerate(objs(spec["stages"], "spec stages")):
        stage_id = text(stage["id"], "stage id")
        for entity_type_index, entity_type in enumerate(ENTITY_TYPES):
            for pair_index in range(pairs_per_entity_type):
                group_seed = (
                    seed + stage_index * 100_003 + entity_type_index * 10_007 + pair_index * 1_009
                )
                group_id = f"{stage_id}_{entity_type}_p{pair_index:03d}"
                split = split_for_group(group_index, splits)
                target_node_id = (stage_index * 13 + entity_type_index * 7 + pair_index * 11) % 54
                base_contract = make_stage_contract(
                    atlas=atlas,
                    indices=indices,
                    stage=stage,
                    sample_id=f"{group_id}_base",
                    sample_index=group_index * 2,
                    target_node_id=target_node_id,
                    colors=colors,
                    seed=group_seed,
                    image_size=resolved_image_size,
                )
                target, before, after = apply_counterfactual(
                    base_contract,
                    entity_type=entity_type,
                    indices=indices,
                    colors=colors,
                    selector=stage_index + entity_type_index + pair_index,
                )
                counterfactual_contract = copy.deepcopy(base_contract)
                apply_declared_value(
                    counterfactual_contract, target=target, value=after, colors=colors
                )
                set_sample_identity(counterfactual_contract, f"{group_id}_counterfactual")
                set_sample_identity(base_contract, f"{group_id}_base")
                pair_contracts = (
                    ("base", base_contract),
                    ("counterfactual", counterfactual_contract),
                )
                pair_descriptor: JsonDict = {
                    "group_id": group_id,
                    "target": target,
                    "before": before,
                    "after": after,
                }
                for role, contract in pair_contracts:
                    manifest_rows.append(
                        _emit_sample(
                            contract,
                            output_dir=output_dir,
                            role=role,
                            group_id=group_id,
                            split=split,
                            stage_id=stage_id,
                            group_seed=group_seed,
                            image_size=resolved_image_size,
                            style=style,
                            style_path=style_path,
                            pair_descriptor=pair_descriptor,
                        )
                    )
                group_index += 1

    manifest_rows.sort(key=lambda row: text(row["sample_id"], "sample_id"))
    write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    for split_name in ("train", "validation", "test"):
        write_jsonl(
            splits_dir / f"{split_name}.jsonl",
            [row for row in manifest_rows if row["split"] == split_name],
        )
    metadata = build_metadata(
        spec=spec,
        spec_path=spec_path,
        output_dir=output_dir,
        manifest_rows=manifest_rows,
        image_size=resolved_image_size,
        pairs_per_entity_type=pairs_per_entity_type,
        seed=seed,
        style_path=style_path,
        ledger=ledger,
    )
    write_json(output_dir / "metadata.json", metadata)
    report = validate_dataset(output_dir, spec_path=spec_path)
    report["output_dir"] = str(output_dir)
    return report


def _emit_sample(
    contract: JsonDict,
    *,
    output_dir: Path,
    role: str,
    group_id: str,
    split: str,
    stage_id: str,
    group_seed: int,
    image_size: int,
    style: RenderStyle,
    style_path: Path,
    pair_descriptor: JsonDict,
) -> JsonDict:
    sample = obj(contract["sample"], "contract sample")
    sample_id = text(sample["id"], "sample id")
    contract_rel = Path("contracts") / f"{sample_id}.json"
    label_rel = Path("dense_labels") / f"{sample_id}.json"
    image_rel = Path("images") / f"{sample_id}.png"
    sample["contract_path"] = str(contract_rel)
    sample["image_path"] = str(image_rel)
    sample["image_size"] = [image_size, image_size]
    source: JsonDict = {
        "kind": "synthetic_engine_contract",
        "generator": "scripts/build_catan_board_recognition_curriculum.py",
        "seed": group_seed,
        "curriculum_stage": stage_id,
        "counterfactual_group_id": group_id,
        "benchmark_game_id": None,
    }
    contract["source"] = source
    labels = dense_labels(contract, stage_id=stage_id)
    image = render_contract_image(contract, image_size=image_size, style=style)
    write_json(output_dir / contract_rel, contract)
    write_json(output_dir / label_rel, labels)
    image.save(output_dir / image_rel)
    return {
        "schema": SAMPLE_SCHEMA,
        "sample_id": sample_id,
        "counterfactual_group_id": group_id,
        "counterfactual_role": role,
        "split": split,
        "stage": stage_id,
        "view": "raw_full_board",
        "contract_path": str(contract_rel),
        "label_path": str(label_rel),
        "image_path": str(image_rel),
        "image_size": [image_size, image_size],
        "source": source,
        "counterfactual": pair_descriptor,
        "render": {
            "renderer": "evals.catan_board_bench.render",
            "style_config": repository_relative(style_path),
            "image_annotation": None,
        },
        "sha256": {
            "contract": file_sha256(output_dir / contract_rel),
            "labels": file_sha256(output_dir / label_rel),
            "image": file_sha256(output_dir / image_rel),
        },
    }
