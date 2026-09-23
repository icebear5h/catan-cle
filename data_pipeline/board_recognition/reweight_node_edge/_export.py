"""Write the reweighted export and its metadata."""


from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

from data_pipeline.board_recognition.node_edge_readout import (
    EXPORT_SCHEMA,
)
from data_pipeline.board_recognition.replay_dataset import read_jsonl
from data_pipeline.board_recognition.reweight_node_edge._audit import audit
from data_pipeline.board_recognition.reweight_node_edge._config import (
    COMPLETION_SUFFIX,
    EVAL_SPLITS,
    SCHEMA,
    JsonDict,
    MixConfig,
)
from data_pipeline.board_recognition.reweight_node_edge._pool import (
    group,
    parse_readout,
    training_pool,
)
from data_pipeline.board_recognition.reweight_node_edge._resample import resample
from data_pipeline.board_recognition.sources import file_sha256
from data_pipeline.board_recognition.spatial_localization import (
    _write_json,
    _write_jsonl,
)
from data_pipeline.board_recognition.terrain_readout import link_or_copy
from data_pipeline.json_coerce import as_dict, as_list, as_str
from data_pipeline.json_types import JsonValue


def export_reweighted(
    source: Path, output: Path, tokenizer_path: Path, *, config: MixConfig = MixConfig(),
    row_count: int | None = None,
) -> JsonDict:
    config.validate()
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("output must be separate from the immutable source export")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata["schema"] != EXPORT_SCHEMA:
        raise ValueError("expected a node_edge_readout_v1 export")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.token_to_id("<|im_end|>") is None:
        raise ValueError("tokenizer is missing the Qwen completion delimiter")
    rows = read_jsonl(source / "stage1/train.jsonl")
    pool = training_pool(rows)
    atlas = {token for row in rows if group(row) == "readout" for token in parse_readout(row)}
    if any(tokenizer.token_to_id(token) is None or len(tokenizer.encode(token, add_special_tokens=False).ids) != 1 for token in atlas):
        raise ValueError("use the checkpoint tokenizer with atomic atlas tokens, not the base tokenizer")

    @lru_cache(maxsize=None)
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text + COMPLETION_SUFFIX, add_special_tokens=False).ids)

    selected, recipe = resample(pool, count_tokens, row_count=len(rows) if row_count is None else row_count, config=config)
    files: dict[str, JsonValue] = {}
    eval_rows = {}
    train_layouts = {row["layout_id"] for row in rows}
    for split in ("train", *EVAL_SPLITS):
        path = source / "stage1" / f"{split}.jsonl"
        digest = file_sha256(path)
        if digest != metadata["files"][f"stage1/{split}.jsonl"]["sha256"]:
            raise ValueError(f"source fingerprint changed: {path}")
        files[split] = {"source_sha256": digest}
        if split != "train":
            eval_rows[split] = read_jsonl(path)
            if {row["layout_id"] for row in eval_rows[split]} & train_layouts:
                raise ValueError(f"train/eval layout overlap: {split}")
    image_names = {
        as_str(as_list(row["images"])[0])
        for row in [*selected, *(r for rows_ in eval_rows.values() for r in rows_)]
    }
    for name in image_names:
        if Path(name).name != name or not (source / "images" / name).is_file():
            raise ValueError(f"invalid or missing image: {name}")
    result: JsonDict = {
        "schema": SCHEMA, "source_export": str(source), "source_metadata_sha256": file_sha256(source / "metadata.json"),
        "image_root": str(output / "images"), "tokenizer_sha256": file_sha256(tokenizer_path),
        "token_count_basis": "tokenizer.encode(answer + completion_suffix, add_special_tokens=False)",
        "completion_suffix": COMPLETION_SUFFIX,
        "caveat": "Corpus token exposure, not exact gradient shares; no within-readout weighting. Common EOT/newline tokens included; no prompt tokens.",
        "recipe": recipe, "before": audit(rows, count_tokens), "after": audit(selected, count_tokens),
        "prefix_16384": audit(selected[:16384], count_tokens),
        "files": files,
    }
    output.mkdir(parents=True)
    (output / "images").mkdir()
    # Match the existing exporter's immutable-image hard-link/copy convention.
    # Symlinks outside image_root would violate the trainer's path guard.
    for name in sorted(image_names):
        link_or_copy(source / "images" / name, output / "images" / name)
    _write_jsonl(output / "stage1/train.jsonl", selected)
    for split in EVAL_SPLITS:
        shutil.copyfile(source / "stage1" / f"{split}.jsonl", output / "stage1" / f"{split}.jsonl")
    for split in files:
        split_files = as_dict(files[split])
        split_files["sha256"] = file_sha256(output / "stage1" / f"{split}.jsonl")
        if split != "train" and split_files["sha256"] != split_files["source_sha256"]:
            raise RuntimeError(f"evaluation changed while copying: {split}")
    _write_json(output / "metadata.json", result)
    return result


__all__ = ["export_reweighted"]
