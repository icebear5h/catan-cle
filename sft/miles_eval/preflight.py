"""CPU-only checkpoint/token checks before starting the Miles inference engine."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, cast

from transformers import AutoTokenizer

from evals.catan_board_bench.tokens import atlas_tokens
from sft.cartesian_eval import SCHEMA as CARTESIAN_SCHEMA
from sft.cartesian_eval.scaling import SCHEMA as SCALING_SCHEMA
from sft.cartesian_eval.shorthand import SCHEMA as SHORTHAND_SCHEMA
from sft.miles_eval.contracts import (
    Json,
    JsonObject,
    json_list,
    json_object,
    parse_json,
    require,
    text,
)
from sft.miles_eval.data import read_panel


class TokenizerView(Protocol):
    def apply_chat_template(
        self, conversation: list[Json], *, tokenize: bool, add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str: ...

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]: ...


class TokenizerLoader(Protocol):
    def from_pretrained(
        self, path: Path, *, local_files_only: bool, trust_remote_code: bool,
    ) -> TokenizerView: ...


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def serving_files(checkpoint: Path) -> list[Path]:
    """Require complete local HF weights: stock or explicitly merged, never adapter-only."""
    require(checkpoint.is_dir(), "--hf-checkpoint must be a local, complete HF model directory")
    require(not (checkpoint / "adapter_config.json").exists(),
            "PEFT adapters must first be merged/exported with both atlas row tables and visual state")
    required = [checkpoint / name for name in ("config.json", "tokenizer.json", "tokenizer_config.json")]
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        names = {text(value) for value in json_object(parse_json(index.read_text())["weight_map"]).values()}
        require(bool(names) and all(Path(name).name == name and name.endswith(".safetensors")
                                   for name in names), "invalid HF weight index")
        required.extend([index, *(checkpoint / name for name in sorted(names))])
    else:
        required.append(checkpoint / "model.safetensors")
    required.extend(path for path in checkpoint.iterdir()
                    if path.name not in {p.name for p in required}
                    and (path.suffix in {".json", ".jinja", ".model", ".txt"}))
    require(all(path.is_file() and path.stat().st_size > 0 for path in required),
            "HF checkpoint has missing or empty weights/tokenizer/config files")
    return sorted(required)


def check_token_lengths(
    tokenizer: TokenizerView, panels: dict[str, Path], *, context: int, new_tokens: int,
) -> JsonObject:
    lengths: list[Json] = []
    token_ids: list[Json] = []
    panel_rows = {name: read_panel(path) for name, path in panels.items()}
    stock_cartesian = all(
        json_object(json_object(json_object(row["metadata"])["catan"])["metadata"]).get("schema")
        in (CARTESIAN_SCHEMA, SHORTHAND_SCHEMA, SCALING_SCHEMA)
        for rows in panel_rows.values() for row in rows
    )
    if not stock_cartesian:
        for token in atlas_tokens():
            ids = tokenizer.encode(token, add_special_tokens=False)
            require(len(ids) == 1, f"saved atlas token is no longer atomic: {token}")
            token_ids.append(ids[0])
        require(len(set(cast("list[int]", token_ids))) == 154, "atlas token IDs collide")
    for name, rows in panel_rows.items():
        for row in rows:
            prompt = tokenizer.apply_chat_template(
                json_list(row["prompt"]), tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_count = len(tokenizer.encode(prompt, add_special_tokens=False))
            gold_count = len(tokenizer.encode(text(row["label"]), add_special_tokens=False))
            require(prompt_count + new_tokens <= context and gold_count <= new_tokens,
                    f"{name}: prompt/gold exceeds context; truncation and row filtering are forbidden")
            catan = json_object(json_object(row["metadata"])["catan"])
            lengths.append({"panel": name, "id": catan["id"],
                            "prompt": prompt_count, "gold": gold_count})
    return {"atlas_token_ids": token_ids, "token_lengths": lengths,
            "context": context, "new_tokens": new_tokens, "enable_thinking": False,
            "requires_atlas_tokenizer": not stock_cartesian}


def preflight(checkpoint: Path, panels: dict[str, Path], *, context: int, new_tokens: int) -> JsonObject:
    files = serving_files(checkpoint)
    tokenizer = cast("TokenizerLoader", AutoTokenizer).from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False,
    )
    report = check_token_lengths(tokenizer, panels, context=context, new_tokens=new_tokens)
    report["checkpoint_files"] = {
        str(path.relative_to(checkpoint)): {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        for path in files
    }
    return report
