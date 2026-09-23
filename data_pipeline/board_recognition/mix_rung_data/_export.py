"""Write the mixed rung export and its metadata."""


from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

from data_pipeline.board_recognition.mix_rung_data._config import (
    EXPORT_SCHEMA,
    JsonDict,
)
from data_pipeline.board_recognition.mix_rung_data._sampling import (
    sample_group,
)
from data_pipeline.board_recognition.mix_rung_data._sources import (
    eval_sample,
    load_recipe,
    source_image_root,
    source_rows,
)
from data_pipeline.board_recognition.sources import file_sha256
from data_pipeline.board_recognition.spatial_localization import (
    SpatialLocalizationError,
    _deterministic_shuffle,
    _write_json,
    _write_jsonl,
)
from data_pipeline.board_recognition.terrain_readout import link_or_copy
from data_pipeline.json_coerce import as_dict, as_int, as_list, as_str


def export_mixed_rung(recipe_path: str | Path, output_dir: str | Path, *, overwrite: bool = False) -> JsonDict:
    recipe = load_recipe(Path(recipe_path))
    output = Path(output_dir).resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        shutil.rmtree(output)
    images_dir = output / "images"
    images_dir.mkdir(parents=True)
    sources = {
        name: Path(as_str(path)).resolve() for name, path in as_dict(recipe["sources"]).items()
    }
    image_roots = {name: source_image_root(path) for name, path in sources.items()}
    train_split = as_str(recipe.get("train_split", "train"))
    train_by_source = {
        name: source_rows(path, train_split) for name, path in sources.items()
    }

    picked: list[JsonDict] = []
    reports: dict[str, JsonDict] = {}
    for raw_group in as_list(recipe["groups"]):
        group = as_dict(raw_group)
        rows, report = sample_group(group, train_by_source[as_str(group["source"])])
        for row in rows:
            row = dict(row)
            row["mix_group"] = group["name"]
            row["mix_source"] = group["source"]
            picked.append(row)
        reports[as_str(group["name"])] = report
    seen = Counter(row["row_id"] for row in picked)
    duplicates = [row_id for row_id, count in seen.items() if count > 1]
    if duplicates:
        raise SpatialLocalizationError(f"row ids picked by more than one group: {duplicates[:5]}")
    train = _deterministic_shuffle(picked, "mixed_rung")
    for row in train:
        image_name = as_str(as_list(row["images"])[0])
        link_or_copy(
            image_roots[as_str(row["mix_source"])] / image_name, images_dir / image_name
        )
    train_path = output / "stage1" / "train.jsonl"
    _write_jsonl(train_path, train)

    evaluation = eval_sample(recipe, sources)
    for raw_spec in as_list(recipe.get("eval_sample", [])):
        spec = as_dict(raw_spec)
        for row in evaluation:
            eval_image = as_str(as_list(row["images"])[0])
            candidate = image_roots[as_str(spec["source"])] / eval_image
            if candidate.is_file():
                link_or_copy(candidate, images_dir / eval_image)
    eval_path = output / "stage1" / "eval_sample.jsonl"
    _write_jsonl(eval_path, evaluation)

    total_tokens = sum(as_int(report["estimated_tokens"]) for report in reports.values()) or 1
    metadata: JsonDict = {
        "schema": EXPORT_SCHEMA,
        "recipe": str(Path(recipe_path).resolve()),
        "recipe_sha256": file_sha256(Path(recipe_path)),
        "sources": {name: str(path) for name, path in sources.items()},
        "source_image_roots": {name: str(path) for name, path in image_roots.items()},
        "image_root": str(images_dir),
        "rows": len(train),
        "unique_images": len({as_str(as_list(row["images"])[0]) for row in train}),
        "groups": {
            name: {
                **report,
                "estimated_token_share": round(
                    as_int(report["estimated_tokens"]) / total_tokens, 4
                ),
            }
            for name, report in reports.items()
        },
        "by_density": dict(sorted(Counter(str(row.get("density_bin")) for row in train).items())),
        "by_category": dict(sorted(Counter(as_str(row["category"]) for row in train).items())),
        "eval_sample_rows": len(evaluation),
        "eval_sample_by_category": dict(sorted(Counter(as_str(row["category"]) for row in evaluation).items())),
        "files": {"stage1/train.jsonl": file_sha256(train_path), "stage1/eval_sample.jsonl": file_sha256(eval_path)},
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


__all__ = ["export_mixed_rung"]
