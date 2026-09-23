"""Train-only static-operation admission and fresh-only Miles JSONL export."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from sft.json_types import JsonDict, JsonValue, as_dict, as_str

from .contracts import (
    SCHEMA,
    STATIC_OPERATIONS,
    ChatTokenizer,
    CorpusInspection,
    PrepareReceipt,
    SourceRow,
)
from .encoding import encode_pair, message_pair, reject_media

DECLARATIONS = (
    "split", "task_role", "task_type", "training_family", "operation", "category",
    "review_only", "admitted_for_training", "target", "schema", "provenance",
)
QUERY_FIELDS = {
    "symbolic_direction": {"a", "b", "direction"},
    "symbolic_direction_choice": {"a", "b", "direction", "choices"},
    "symbolic_neighbors": {"token"},
    "symbolic_incidence": {"token", "family"},
    "symbolic_oriented_step": {"token", "direction"},
}


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _unique(pairs: list[tuple[str, JsonValue]]) -> JsonDict:
    result: JsonDict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def _declarations(row: JsonDict) -> JsonDict:
    metadata = dict(as_dict(row.get("metadata", {})))
    for key in DECLARATIONS:
        if key in row:
            if key in metadata and metadata[key] != row[key]:
                raise ValueError(f"conflicting declaration: {key}")
            metadata[key] = row[key]
    return metadata


def _operation(metadata: JsonDict) -> str:
    labels = [as_str(metadata[k]) for k in ("task_type", "operation", "training_family")
              if k in metadata]
    if len(set(labels)) > 1:
        raise ValueError("conflicting operation/task_type/training_family declarations")
    # Atlas v0 categories are reported, never silently mapped into the allowlist.
    return labels[0] if labels else as_str(metadata.get("category", "<missing>"))


def _exclusion(metadata: JsonDict, operation: str) -> str | None:
    if "split" not in metadata:
        return "missing_train_split"
    if metadata["split"] != "train":
        return "non_train_split"
    if (metadata.get("task_role", "train") != "train"
            or metadata.get("review_only", False) is not False
            or metadata.get("admitted_for_training", True) is not True):
        return "non_training_role"
    if operation not in STATIC_OPERATIONS:
        return "operation_not_allowed"
    provenance = as_dict(metadata.get("provenance", {}))
    if provenance.get("split", "train") != "train":
        return "non_train_provenance"
    target = as_dict(metadata.get("target", {}))
    if target.get("state") is not None:
        return "state_conditioned"
    if target:
        if set(target) != {"state", "query"}:
            raise ValueError("static target must contain exactly state and query")
        if set(as_dict(target["query"])) != QUERY_FIELDS[operation]:
            raise ValueError("static query fields do not match the declared operation")
    return None


def _read_corpus(source: Path) -> tuple[list[SourceRow], CorpusInspection]:
    raw = source.read_bytes()
    lines = raw.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    if not lines:
        raise ValueError("empty source corpus")
    admitted: list[SourceRow] = []
    seen: set[str] = set()
    included: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    for number, raw_line in enumerate(lines, 1):
        line = raw_line.removesuffix(b"\r")
        try:
            decoded: JsonValue = json.loads(
                line.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_invalid_constant,
            )
            row = as_dict(decoded)
            metadata = _declarations(row)
            row_id = as_str(row.get("id", row.get("row_id")))
            if not row_id.strip() or row_id in seen or row.get("row_id", row_id) != row_id:
                raise ValueError("duplicate, empty or conflicting row ID")
            seen.add(row_id)
            operation = _operation(metadata)
            splits[as_str(metadata.get("split", "<missing>"))] += 1
            reason = _exclusion(metadata, operation)
            if reason is not None:
                excluded[operation] += 1
                reasons[reason] += 1
                continue
            reject_media(row)
            messages = message_pair(row.get("messages"))
            admitted.append(SourceRow(row_id, operation, messages, metadata, number, _sha256(line)))
            included[operation] += 1
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{source}:{number}: {exc}") from exc
    return admitted, {
        "source": str(source.resolve()), "source_sha256": _sha256(raw),
        "total_rows": len(lines), "admitted_rows": len(admitted),
        "excluded_rows": sum(reasons.values()), "admitted_by_operation": dict(included),
        "excluded_by_operation": dict(excluded), "excluded_by_reason": dict(reasons),
        "by_split": dict(splits),
    }


def inspect_corpus(source: Path) -> CorpusInspection:
    """Audit all rows and report mutually exclusive exclusion reasons, without a tokenizer."""
    return _read_corpus(source)[1]


def prepare_dataset(
    source: Path, output: Path, tokenizer: ChatTokenizer, max_tokens: int = 4096,
    limit: int | None = None, *, tokenizer_identity: str | None = None,
) -> PrepareReceipt:
    """Write ``input.jsonl`` and ``metadata.json`` in a new directory after validation.

    Source order is retained; limit selects the first N admitted rows, while corpus
    counts cover the entire source. Overlong selected rows fail, never truncate.
    Row hashes cover exact UTF-8 source lines excluding CR/LF; the source hash
    covers the entire original file. Metadata is the JSON-serializable receipt.
    """
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"fresh output directory required: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent must exist: {output.parent}")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer or None")
    template = tokenizer.chat_template
    identity = tokenizer.name_or_path if tokenizer_identity is None else tokenizer_identity
    if not template or not isinstance(identity, str) or not identity.strip():
        raise ValueError("saved native tokenizer identity and chat template are required")
    admitted, corpus = _read_corpus(source)
    selected = admitted if limit is None else admitted[:limit]
    if not selected:
        raise ValueError(f"no admitted train rows: {json.dumps(corpus, sort_keys=True)}")
    serialized: list[str] = []
    total_tokens = supervised_tokens = longest = 0
    for row in selected:
        try:
            encoded = encode_pair(tokenizer, row.messages, max_tokens)
        except ValueError as exc:
            raise ValueError(f"{source}:{row.line_number} ({row.row_id}): {exc}") from exc
        total_tokens += len(encoded["tokens"])
        supervised_tokens += encoded["response_length"]
        longest = max(longest, len(encoded["tokens"]))
        prepared = {
            "id": row.row_id, "row_id": row.row_id, "messages": row.messages,
            "metadata": {
                **encoded, "operation": row.operation, "split": "train",
                "source_file": corpus["source"], "source_sha256": corpus["source_sha256"],
                "source_line": row.line_number, "source_row_sha256": row.row_sha256,
                "source_metadata": row.metadata,
            },
        }
        serialized.append(json.dumps(prepared, ensure_ascii=False, sort_keys=True))
    payload = ("\n".join(serialized) + "\n").encode("utf-8")
    destination = output.resolve()
    receipt: PrepareReceipt = {
        "schema": SCHEMA, "source": corpus["source"], "output": str(destination),
        "input_path": str(destination / "input.jsonl"),
        "metadata_path": str(destination / "metadata.json"),
        "source_sha256": corpus["source_sha256"], "input_sha256": _sha256(payload),
        "rows": len(selected), "limited_rows": len(admitted) - len(selected),
        "selected_by_operation": dict(Counter(row.operation for row in selected)),
        "corpus": corpus, "tokenizer_identity": identity,
        "chat_template_sha256": _sha256(json.dumps(template, sort_keys=True).encode("utf-8")),
        "max_tokens": max_tokens, "limit": limit, "total_tokens": total_tokens,
        "supervised_tokens": supervised_tokens, "longest_sequence": longest,
    }
    # No output exists until all selected encodings pass. mkdir/open are exclusive;
    # an interrupted write is visibly incomplete and cannot be reused by this API.
    output.mkdir()
    with (output / "input.jsonl").open("xb") as handle:
        handle.write(payload)
    with (output / "metadata.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
