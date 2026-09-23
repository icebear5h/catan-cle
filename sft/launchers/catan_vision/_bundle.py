"""Upload the training bundle and pin its contract to a Modal Volume."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from sft.json_types import JsonDict
from sft.launchers.catan_vision._config import REMOTE_DATA, sft_data
from sft.scripts.train.train_trl_catan_vision import (
    inspect_jsonl_contract,
    iter_jsonl,
    load_token_inventory,
    resolve_image_path,
)


def upload_training_bundle(
    train_jsonl: Path,
    image_root: Path,
    token_inventory: Path,
    *,
    remote_dir: str,
    require_curriculum: bool,
    eval_jsonl: Path | None = None,
    eval_image_root: Path | None = None,
    extra_files: dict[str, Path] | None = None,
) -> tuple[str, str | None, str, str, JsonDict, JsonDict | None]:
    """Upload one ordered JSONL and its deduplicated images without reordering.

    ``extra_files`` maps a remote file name under the dataset directory to a
    local file that travels with the data (the visual-delta factors for O-LoRA).
    """

    source = train_jsonl.expanduser().resolve()
    root = image_root.expanduser().resolve()
    inventory_path = token_inventory.expanduser().resolve()
    contract = inspect_jsonl_contract(source, root, require_curriculum=require_curriculum)
    eval_source = eval_jsonl.expanduser().resolve() if eval_jsonl is not None else None
    eval_root = eval_image_root.expanduser().resolve() if eval_image_root is not None else None
    if bool(eval_source) != bool(eval_root):
        raise ValueError("eval_jsonl and eval_image_root must be supplied together")
    eval_contract = (
        inspect_jsonl_contract(eval_source, eval_root, require_curriculum=False)
        if eval_source is not None and eval_root is not None
        else None
    )
    load_token_inventory(inventory_path)

    remote_root = "/" + remote_dir.strip("/")
    remote_jsonl = f"{remote_root}/train.jsonl"
    remote_eval_jsonl = f"{remote_root}/eval.jsonl" if eval_source is not None else None
    remote_images = f"{remote_root}/images"
    remote_inventory = f"{remote_root}/trainable_tokens.json"
    image_names: dict[Path, str] = {}

    with tempfile.TemporaryDirectory(prefix="catan-modal-upload-") as temporary_dir:
        rewritten_path = Path(temporary_dir) / "train.jsonl"
        rewritten_eval_path = Path(temporary_dir) / "eval.jsonl"

        def rewrite_jsonl(input_path: Path, input_root: Path, output_path: Path) -> None:
            with output_path.open("w") as output:
                for line_number, row in iter_jsonl(input_path):
                    image_path = resolve_image_path(input_root, row, line_number=line_number)
                    image_name = image_names.get(image_path)
                    if image_name is None:
                        suffix = image_path.suffix.lower() or ".png"
                        image_name = f"{len(image_names):06d}{suffix}"
                        image_names[image_path] = image_name
                    rewritten = dict(row)
                    rewritten.pop("image", None)
                    rewritten["images"] = [image_name]
                    output.write(json.dumps(rewritten, sort_keys=True) + "\n")

        rewrite_jsonl(source, root, rewritten_path)
        if eval_source is not None and eval_root is not None:
            rewrite_jsonl(eval_source, eval_root, rewritten_eval_path)

        with sft_data.batch_upload(force=True) as batch:
            for image_path, image_name in image_names.items():
                batch.put_file(image_path, f"{remote_images}/{image_name}")
            batch.put_file(rewritten_path, remote_jsonl)
            if remote_eval_jsonl is not None:
                batch.put_file(rewritten_eval_path, remote_eval_jsonl)
            batch.put_file(inventory_path, remote_inventory)
            for remote_name, local_path in (extra_files or {}).items():
                batch.put_file(local_path.expanduser().resolve(), f"{remote_root}/{remote_name}")

    return (
        f"{REMOTE_DATA}{remote_jsonl}",
        f"{REMOTE_DATA}{remote_eval_jsonl}" if remote_eval_jsonl is not None else None,
        f"{REMOTE_DATA}{remote_images}",
        f"{REMOTE_DATA}{remote_inventory}",
        contract,
        eval_contract,
    )
