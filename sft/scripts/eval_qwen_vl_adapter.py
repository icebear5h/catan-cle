"""Evaluate a Qwen-VL adapter on local Catan SFT JSONL rows."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors import safe_open

from sft.behavior_diagnostics import summarize_behaviors
from sft.board_fluency_scoring import (
    SCHEMA, SCHEMAS as BOARD_FLUENCY_SCHEMAS, score_board_fluency, validate_board_fluency_metadata,
)
from sft.board_state_readout import TASK as FULL_BOARD_TASK, score_board_state, summarize_board_states
from sft.paths import resolve_dataset_asset, resolve_dataset_image
from sft.scripts.train_trl_catan_vision import (
    INPUT_MODES,
    RUN_CONFIG_FILE,
    VISUAL_STATE_FILE,
    _message_pair,
    assert_runtime_versions,
    encode_text_pair,
    freeze_visual_weights,
    load_checkpoint_text_tokenizer,
    load_token_inventory,
    load_visual_state,
    native_tokenizer,
    pad_text_inputs,
    prepare_semantic_tokens,
    resolve_wrapped_module,
    text_chat_ids,
    validate_text_adapter,
    validate_text_budget,
    validate_text_context_budget,
)
from sft.spatial_tasks import TASK_TYPES as SPATIAL_TASK_TYPES, score_spatial_task
from sft.symbolic_board_tasks import SYMBOLIC_TASKS, score_symbolic_task, symbolic_task_role

BOARD_FLUENCY_SCHEMA = SCHEMA


def iter_jsonl(path: Path):
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                yield line_number, json.loads(line)


def user_text(row: dict[str, Any]) -> str:
    if "conversations" in row:
        value = str(row["conversations"][0]["value"])
        return value.replace("<image>", "", 1).strip()
    content = row["messages"][0]["content"]
    if isinstance(content, str):
        return content.replace("<image>", "", 1).strip()
    return "\n".join(
        str(item.get("text", "")) for item in content if item.get("type") == "text"
    ).strip()


def expected_text(row: dict[str, Any]) -> str:
    if "conversations" in row:
        return str(row["conversations"][1]["value"]).strip()
    content = row["messages"][1]["content"]
    if isinstance(content, str):
        return content.strip()
    return "\n".join(
        str(item.get("text", "")) for item in content if item.get("type") == "text"
    ).strip()


LONG_ANSWER_CHARACTERS = 48
LONG_ANSWER_TASK_TYPES = {"terrain_readout", "node_readout", "edge_readout", FULL_BOARD_TASK}


def is_long_answer(row: dict[str, Any]) -> bool:
    """Rows whose expected answer needs the long generation budget.

    Full-board readouts answer with every tile and port keyed by atlas token,
    far past the 16-token budget the one-token and one-phrase heads use.
    """

    if row.get("task_type") in LONG_ANSWER_TASK_TYPES or (row.get("metadata") or {}).get("task_type") in LONG_ANSWER_TASK_TYPES:
        return True
    return len(expected_text(row)) > LONG_ANSWER_CHARACTERS


def normalize_text(text: str) -> str:
    text = text.strip()
    text = text.split("<|im_end|>", 1)[0].strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(?:answer|assistant)\s*:\s*", "", text, flags=re.I)
    return text.strip()


def extract_json_object(text: str) -> Any | None:
    text = normalize_text(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


READOUT_ITEM_RE = re.compile(r"(<[NETP][0-9_]+>)\s*([^;<]*)")


def readout_items(text: str) -> dict[str, str]:
    """Parse ``<T00> wood 11; <P00> 3:1 port`` into token -> value, whitespace-tolerant."""

    return {token: " ".join(value.split()) for token, value in READOUT_ITEM_RE.findall(text)}


def score_readout(expected: str, response: str) -> dict[str, Any]:
    expected_items = readout_items(expected)
    response_items = readout_items(response)
    matched = sum(1 for token, value in expected_items.items() if response_items.get(token) == value)
    occupied = {token: value for token, value in expected_items.items() if value != "empty"}
    occupied_matched = sum(1 for token, value in occupied.items() if response_items.get(token) == value)
    return {
        "correct": matched == len(expected_items) and len(response_items) == len(expected_items),
        "scoring": "readout_items",
        "expected_normalized": "; ".join(f"{token} {value}" for token, value in expected_items.items()),
        "response_normalized": "; ".join(f"{token} {value}" for token, value in response_items.items()),
        "items_correct": matched,
        "items_total": len(expected_items),
        "items_extra": max(0, len(response_items) - len(expected_items)),
        # A full-list readout is mostly ``empty`` on a real board; the occupied
        # items are the ones that carry the board state, so they are scored apart.
        "occupied_items_correct": occupied_matched,
        "occupied_items_total": len(occupied),
        "empty_items_correct": matched - occupied_matched,
        "empty_items_total": len(expected_items) - len(occupied),
    }


def score_response(
    expected: str, response: str, *, metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if metadata is not None:
        board_fluency_score = score_board_fluency(expected, response, metadata)
        if board_fluency_score is not None:
            return board_fluency_score
        symbolic_score = score_symbolic_task(expected, response, metadata)
        if symbolic_score is not None:
            return symbolic_score
    expected_norm = normalize_text(expected)
    response_norm = normalize_text(response)
    if metadata is not None:
        spatial_score = score_spatial_task(expected_norm, response_norm, metadata)
        if spatial_score is not None:
            return spatial_score
    if "; robber " in expected_norm and len(readout_items(expected_norm)) >= 154:
        return score_board_state(expected_norm, response_norm)
    if len(readout_items(expected_norm)) >= 4:
        return score_readout(expected_norm, response_norm)

    expected_json = extract_json_object(expected_norm)
    if expected_json is not None:
        response_json = extract_json_object(response_norm)
        correct = response_json == expected_json
        return {
            "correct": correct,
            "scoring": "json_exact",
            "expected_normalized": json.dumps(expected_json, sort_keys=True),
            "response_normalized": (
                json.dumps(response_json, sort_keys=True)
                if response_json is not None
                else response_norm
            ),
        }

    return {
        "correct": expected_norm == response_norm,
        "scoring": "text_exact",
        "expected_normalized": expected_norm,
        "response_normalized": response_norm,
    }


IMAGE_VARIANTS = ("original", "blank", "shuffle", "target_occlusion", "control_occlusion")
ATLAS_TOKEN_RE = re.compile(r"^<([NETP])[0-9_]+>$")
MARKER_LETTERS = ("A", "B", "C", "D")
ENTITY_PREFIXES = {"node": "N", "edge": "E", "tile": "T", "port": "P"}


def candidate_answers(
    row: dict[str, Any],
    expected: str,
    atlas_tokens: list[str],
) -> list[str] | None:
    """Return the closed answer set a row licenses, or None for open answers.

    Location-only scoring ranks only these candidates at the first answer
    position, so a wrong entity type or a stray word cannot mask whether the
    model localized the queried position. Atlas answers restrict to tokens of
    the requested entity type, marker answers to the four letters, and polarity
    answers to yes/no.
    """

    metadata = row.get("metadata") or {}
    task_type = row.get("task_type") or metadata.get("task_type")
    if task_type == "token_to_marker":
        return list(MARKER_LETTERS)
    expected_norm = normalize_text(expected)
    match = ATLAS_TOKEN_RE.match(expected_norm)
    if match:
        entity = row.get("entity_type") or metadata.get("entity_type")
        prefix = ENTITY_PREFIXES.get(str(entity), match.group(1))
        candidates = [token for token in atlas_tokens if token[1] == prefix]
        if expected_norm not in candidates:
            raise ValueError(f"expected {expected_norm!r} is not among {prefix} atlas candidates")
        return candidates
    if expected_norm in {"yes", "no"}:
        return ["yes", "no"]
    return None


def candidate_token_ids(tokenizer: Any, candidates: list[str]) -> list[int]:
    """First answer-token id for each candidate; every candidate must be non-empty."""

    ids = []
    for candidate in candidates:
        encoded = tokenizer.encode(candidate, add_special_tokens=False)
        if not encoded:
            raise ValueError(f"candidate {candidate!r} produced no tokens")
        ids.append(int(encoded[0]))
    if len(set(ids)) != len(ids):
        raise ValueError("candidates collide on their first token")
    return ids


def score_candidates(
    first_logits: Any,
    candidates: list[str],
    token_ids: list[int],
    expected: str,
) -> dict[str, Any]:
    """Rank the closed answer set by first-token log-probability."""

    torch = importlib.import_module("torch")
    log_probs = torch.log_softmax(first_logits.float(), dim=-1)
    scores = log_probs[torch.tensor(token_ids)]
    order = torch.argsort(scores, descending=True).tolist()
    expected_norm = normalize_text(expected)
    expected_index = candidates.index(expected_norm)
    ranked = [candidates[index] for index in order]
    return {
        "scoring": "first_token_candidates",
        "candidates": len(candidates),
        "predicted": ranked[0],
        "correct": ranked[0] == expected_norm,
        "expected_rank": ranked.index(expected_norm) + 1,
        "expected_logprob": float(scores[expected_index]),
        "top": [
            {"answer": candidates[index], "logprob": float(scores[index])}
            for index in order[:3]
        ],
    }


def image_reference(row: dict[str, Any]) -> str:
    if row.get("image"):
        return str(row["image"])
    images = row.get("images")
    if isinstance(images, list) and len(images) == 1:
        return str(images[0])
    raise ValueError("eval row must reference exactly one image")


def build_qwen_messages(
    row: dict[str, Any], *, input_mode: str = "vision",
) -> list[dict[str, Any]]:
    if input_mode == "text":
        prompt, _ = _message_pair(row, line_number=0, input_mode="text")
        return [{"role": "user", "content": prompt}]
    if input_mode != "vision":
        raise ValueError(f"unsupported input_mode: {input_mode}")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text(row)},
            ],
        }
    ]


def _spatial_target(row: dict[str, Any]) -> dict[str, Any]:
    targets = row.get("spatial_targets") or []
    if len(targets) != 1:
        raise ValueError("image perturbation requires exactly one spatial target")
    return targets[0]


def _shuffled_image_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    paths = list(dict.fromkeys(image_reference(row) for row in rows))
    if len(paths) < 2:
        raise ValueError("shuffled-image evaluation requires at least two unique images")
    return {path: paths[(index + 1) % len(paths)] for index, path in enumerate(paths)}


def evaluation_image(
    row: dict[str, Any],
    *,
    variant: str,
    shuffled_images: dict[str, str] | None,
    occlusion_margin: float,
) -> Any:
    image_module = importlib.import_module("PIL.Image")
    draw_module = importlib.import_module("PIL.ImageDraw")
    source_path = image_reference(row)
    if variant == "shuffle":
        if shuffled_images is None:
            raise ValueError("shuffle variant requires a precomputed image map")
        source_path = shuffled_images[source_path]
    with image_module.open(source_path) as source:
        image = source.convert("RGB").copy()
    if variant == "blank":
        return image_module.new("RGB", image.size, (127, 127, 127))
    if variant in {"target_occlusion", "control_occlusion"}:
        target = _spatial_target(row)
        key = "bbox" if variant == "target_occlusion" else "control_bbox"
        x1, y1, x2, y2 = (float(value) for value in target[key])
        x1, y1 = max(0.0, x1 - occlusion_margin), max(0.0, y1 - occlusion_margin)
        x2, y2 = min(1.0, x2 + occlusion_margin), min(1.0, y2 + occlusion_margin)
        width, height = image.size
        draw_module.Draw(image).rectangle(
            [round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)],
            fill=(42, 42, 42),
        )
    return image


def generate_responses(
    *,
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
    image_variant: str = "original",
    shuffled_images: dict[str, str] | None = None,
    occlusion_margin: float = 0.03,
    return_first_logits: bool = True,
    preserve_visual_fp32: bool = False,
    input_mode: str = "vision",
    max_sequence_length: int | None = None,
) -> tuple[list[str], Any]:
    """Generate answers and return the raw first-step logits per row."""

    torch = importlib.import_module("torch")

    if input_mode == "text":
        validate_text_budget(max_sequence_length)
        validate_text_context_budget(model, max_sequence_length)
        if image_variant != "original" or shuffled_images is not None:
            raise ValueError("text mode does not support image variants")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        tokenizer = native_tokenizer(processor)
        features = []
        for row in rows:
            prompt, answer = _message_pair(row, line_number=0, input_mode="text")
            encode_text_pair(tokenizer, prompt, answer, max_sequence_length=max_sequence_length)
            ids = text_chat_ids(tokenizer, build_qwen_messages(row, input_mode="text"), generation=True)
            if len(ids) + max_new_tokens > max_sequence_length:
                raise ValueError(
                    f"text prompt ({len(ids)}) + generation budget ({max_new_tokens}) exceeds "
                    f"max_sequence_length={max_sequence_length}; truncation is forbidden"
                )
            features.append({"input_ids": ids})
        inputs = {
            key: tensor.to(model.device)
            for key, tensor in pad_text_inputs(tokenizer, features, left=True).items()
        }
        decoder = tokenizer
    else:
        messages = [build_qwen_messages(row, input_mode=input_mode) for row in rows]
        prompts = [
            processor.apply_chat_template(
                item,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for item in messages
        ]
        images = [
            evaluation_image(
                row,
                variant=image_variant,
                shuffled_images=shuffled_images,
                occlusion_margin=occlusion_margin,
            )
            for row in rows
        ]
        inputs = processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        decoder = processor

    # A disabled autocast context would override autocast owned by imported callers.
    autocast = (
        torch.autocast(model.device.type, dtype=torch.bfloat16)
        if preserve_visual_fp32 and input_mode == "vision"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            output_logits=return_first_logits,
            return_dict_in_generate=True,
        )

    input_width = inputs["input_ids"].shape[1]
    trimmed = [output_ids[input_width:] for output_ids in generated.sequences]
    first_logits = generated.logits[0].detach().float().cpu() if return_first_logits else None
    texts = list(decoder.batch_decode(
        trimmed,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ))
    return texts, first_logits


def summarize_dimension(
    attempted: list[dict[str, Any]],
    metadata_key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for record in attempted:
        value = str(record["metadata"].get(metadata_key) or "unknown")
        grouped[value]["total"] += 1
        if record["score"]["correct"]:
            grouped[value]["correct"] += 1
        candidate = record.get("candidate_score")
        if candidate is not None:
            grouped[value]["candidate_total"] += 1
            grouped[value]["candidate_rank_sum"] += candidate["expected_rank"]
            if candidate["correct"]:
                grouped[value]["candidate_correct"] += 1

    result = {}
    for value, counts in sorted(grouped.items()):
        total = counts["total"]
        correct = counts["correct"]
        entry = {
            "total": total,
            "correct": correct,
            "exact_accuracy": correct / total if total else 0.0,
        }
        candidate_total = counts["candidate_total"]
        if candidate_total:
            entry.update(
                {
                    "candidate_total": candidate_total,
                    "candidate_correct": counts["candidate_correct"],
                    "candidate_exact_accuracy": counts["candidate_correct"] / candidate_total,
                    "candidate_expected_rank_mean": counts["candidate_rank_sum"] / candidate_total,
                }
            )
        result[value] = entry
    return result


NEIGHBOR_CONFUSION_CATEGORIES = ("node.occupancy", "edge.owner")
NEIGHBOR_CONFUSION_BUCKETS = ("names_target", "names_partner", "answers_empty", "other")


def piece_answer(color: Any, piece: Any) -> str | None:
    """Render metadata color/piece as the answer string occupancy rows expect."""

    if not color or not piece:
        return None
    return f"{str(color).replace('_', ' ')} {str(piece)}".lower()


def is_neighbor_confusion_record(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata") or {}
    if metadata.get("category") in NEIGHBOR_CONFUSION_CATEGORIES:
        return True
    return str(metadata.get("task_type") or "").startswith("occupancy_")


def summarize_neighbor_confusion(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Bucket wrong occupancy/owner answers by what the model named instead.

    Reads only row metadata (target and partner color/piece plus distances),
    so it needs no board graph at eval time. Rows without partner fields land
    in names_target / answers_empty / other; rows lacking any of the fields
    simply skip the buckets they cannot support.
    """

    groups: dict[str, Counter] = defaultdict(Counter)
    by_distance: dict[str, dict[str, Counter]] = {
        "names_target": defaultdict(Counter),
        "names_partner": defaultdict(Counter),
    }
    for record in records:
        if not is_neighbor_confusion_record(record):
            continue
        metadata = record.get("metadata") or {}
        score = record.get("score") or {}
        group = str(metadata.get("task_type") or metadata.get("category") or "unknown")
        groups[group]["total"] += 1
        if score.get("correct"):
            continue
        groups[group]["total_wrong"] += 1

        response = str(score.get("response_normalized") or "").strip().lower()
        expected = str(score.get("expected_normalized") or "").strip().lower()
        target = piece_answer(metadata.get("color"), metadata.get("piece"))
        partner = piece_answer(metadata.get("partner_color"), metadata.get("partner_piece"))
        if metadata.get("negative_distance") is not None:
            distance = metadata.get("negative_distance")
        else:
            distance = metadata.get("partner_distance")

        if target is not None and response == target:
            bucket = "names_target"
        elif partner is not None and response == partner:
            bucket = "names_partner"
        elif response == "empty" and expected != "empty":
            bucket = "answers_empty"
        else:
            bucket = "other"
        groups[group][bucket] += 1
        if bucket in by_distance:
            by_distance[bucket][group][str(distance)] += 1

    result: dict[str, Any] = {}
    for group, counts in sorted(groups.items()):
        entry = {
            "total": counts["total"],
            "total_wrong": counts["total_wrong"],
        }
        for bucket in NEIGHBOR_CONFUSION_BUCKETS:
            entry[bucket] = counts[bucket]
        entry["by_distance"] = {
            bucket: dict(sorted(by_distance[bucket][group].items()))
            for bucket in by_distance
        }
        result[group] = entry
    return result


OCCUPANCY_CATEGORIES = ("node.occupancy", "edge.owner")


def occupancy_class(answer: str) -> str:
    """``empty``, or the piece word that ends an occupancy answer (``settlement``, ``city``, ``road``)."""

    words = answer.strip().lower().split()
    return "empty" if not words or words[-1] == "empty" else words[-1]


def summarize_occupancy_classes(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-class recall and precision for the occupancy heads, and their balanced accuracy.

    Real boards are mostly empty, so exact accuracy on these heads rewards a
    model that answers ``empty`` everywhere; recall per piece class and the
    precision of ``empty`` are the numbers that mean something.
    """

    expected_counts: dict[str, int] = {}
    predicted_counts: dict[str, int] = {}
    correct_counts: dict[str, int] = {}
    for record in records:
        if record.get("metadata", {}).get("category") not in OCCUPANCY_CATEGORIES:
            continue
        expected = occupancy_class(str(record["score"].get("expected_normalized", "")))
        predicted = occupancy_class(str(record["score"].get("response_normalized", "")))
        expected_counts[expected] = expected_counts.get(expected, 0) + 1
        predicted_counts[predicted] = predicted_counts.get(predicted, 0) + 1
        if record["score"]["correct"]:
            correct_counts[expected] = correct_counts.get(expected, 0) + 1
    classes = {}
    for name in sorted(set(expected_counts) | set(predicted_counts)):
        expected = expected_counts.get(name, 0)
        predicted = predicted_counts.get(name, 0)
        correct = correct_counts.get(name, 0)
        classes[name] = {
            "expected": expected,
            "predicted": predicted,
            "correct": correct,
            "recall": correct / expected if expected else None,
            "precision": correct / predicted if predicted else None,
        }
    recalls = [entry["recall"] for entry in classes.values() if entry["recall"] is not None]
    return {
        "classes": classes,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
        "occupied_recall": (
            sum(classes[name]["correct"] for name in classes if name != "empty")
            / sum(classes[name]["expected"] for name in classes if name != "empty")
            if sum(classes[name]["expected"] for name in classes if name != "empty")
            else None
        ),
        "empty_precision": classes["empty"]["precision"] if "empty" in classes else None,
    }


def summarize_readout_items(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Occupied and empty item rates per readout category."""

    totals: dict[str, dict[str, int]] = {}
    for record in records:
        score = record["score"]
        if "occupied_items_total" not in score:
            continue
        category = str(record.get("metadata", {}).get("category", "readout"))
        entry = totals.setdefault(category, {"readouts": 0, "exact": 0, "occupied_correct": 0, "occupied_total": 0, "empty_correct": 0, "empty_total": 0, "extra": 0})
        entry["readouts"] += 1
        entry["exact"] += int(score["correct"])
        entry["occupied_correct"] += score["occupied_items_correct"]
        entry["occupied_total"] += score["occupied_items_total"]
        entry["empty_correct"] += score["empty_items_correct"]
        entry["empty_total"] += score["empty_items_total"]
        entry["extra"] += score["items_extra"]
    for entry in totals.values():
        entry["occupied_item_recall"] = entry["occupied_correct"] / entry["occupied_total"] if entry["occupied_total"] else None
        entry["empty_item_accuracy"] = entry["empty_correct"] / entry["empty_total"] if entry["empty_total"] else None
    return totals


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [record for record in records if record.get("response") is not None]

    correct_total = sum(record["score"]["correct"] for record in attempted)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(records),
        "attempted": len(attempted),
        "correct": correct_total,
        "exact_accuracy": correct_total / len(attempted) if attempted else 0.0,
    }
    scored = [record["candidate_score"] for record in attempted if record.get("candidate_score")]
    if scored:
        summary.update(
            {
                "candidate_rows": len(scored),
                "candidate_correct": sum(item["correct"] for item in scored),
                "candidate_exact_accuracy": sum(item["correct"] for item in scored) / len(scored),
                "candidate_expected_rank_mean": sum(item["expected_rank"] for item in scored) / len(scored),
                "candidate_expected_rank_le3": sum(item["expected_rank"] <= 3 for item in scored) / len(scored),
            }
        )
    for key in (
        "category",
        "suite",
        "density_bin",
        "row_kind",
        "curriculum_stage",
        "task_family",
        "task_type",
        "family",
        "operation",
        "entity_type",
        "relationship",
        "polarity",
        "marker_style",
        "probe_style",
        "piece",
        "color",
        "color_heldout",
        "grounding_stage",
        "eval_variant",
        "pair_kind",
        "negative_distance",
    ):
        summary[f"by_{key}"] = summarize_dimension(attempted, key)
    summary["categories"] = summary["by_category"]
    summary["occupancy_classes"] = summarize_occupancy_classes(attempted)
    summary["readout_items"] = summarize_readout_items(attempted)
    board_records = [r for r in attempted if r["score"].get("scoring") == FULL_BOARD_TASK]
    if board_records:
        summary["full_board"] = summarize_board_states(board_records)
    summary["neighbor_confusion"] = summarize_neighbor_confusion(attempted)
    summary["by_behavior"] = summarize_behaviors(attempted)
    symbolic = [r for r in attempted if r["score"].get("scoring") in SYMBOLIC_TASKS]
    if symbolic:
        families = defaultdict(list)
        for record in symbolic:
            families[record["score"]["scoring"]].append(record)
        summary["symbolic_families"] = {}
        for family, group in sorted(families.items()):
            weighted = ["macro_weight" in r["metadata"] for r in group]
            if any(weighted) and not all(weighted):
                raise ValueError("incomplete symbolic macro weights")
            weights = [r["metadata"]["macro_weight"] if all(weighted) else 1.0 for r in group]
            if any(type(w) not in (float, int) or not math.isfinite(w) or w <= 0 for w in weights):
                raise ValueError("invalid symbolic macro weights")
            summary["symbolic_families"][family] = {
                "rows": len(group), "correct": sum(r["score"]["correct"] for r in group),
                "accuracy": sum(w * r["score"]["correct"] for w, r in zip(weights, group, strict=True)) / sum(weights),
                "weighting": "state_mode_polarity_macro" if all(weighted) else "row",
                "by_mode": summarize_dimension(group, "evaluation_mode"),
                "by_polarity": summarize_dimension(group, "polarity"),
            }
    return summary


def evaluation_metadata(row: dict[str, Any], *, image_variant: str) -> dict[str, Any]:
    metadata = dict(row.get("metadata", {}))
    schemas = [source["schema"] for source in (row, metadata) if "schema" in source]
    if (any(schema in BOARD_FLUENCY_SCHEMAS for schema in schemas)
            or any(str(schema).startswith("catan_board_fluency") for schema in schemas)
            or row.get("class") == "board_fluency" or metadata.get("class") == "board_fluency"):
        if not schemas or any(schema != schemas[0] for schema in schemas):
            raise ValueError("missing/conflicting board-fluency schema declarations")
        metadata["schema"] = schemas[0]
        for key in ("split", "task_role", "review_only", "admitted_for_training", "class",
                    "family", "operation", "target", "answer", "provenance"):
            if key in row:
                if key in metadata and (type(row[key]) is not type(metadata[key])
                                        or row[key] != metadata[key]):
                    raise ValueError(f"conflicting board-fluency {key}")
                metadata[key] = row[key]
        validate_board_fluency_metadata(metadata)
    declared = [source["task_type"] for source in (row, metadata) if "task_type" in source]
    spatial = [value for value in declared + [row.get("category"), metadata.get("category")]
               if value in SPATIAL_TASK_TYPES or value in SYMBOLIC_TASKS]
    if spatial and any(value != spatial[0] for value in declared + spatial):
        raise ValueError("conflicting spatial task declarations")
    if spatial and spatial[0] in SYMBOLIC_TASKS:
        for key in ("training_family", "task_role", "split"):
            if key in row and key in metadata and row[key] != metadata[key]:
                raise ValueError(f"conflicting symbolic {key}")
            if key in row:
                metadata[key] = row[key]
        if metadata.get("training_family", spatial[0]) != spatial[0]:
            raise ValueError("conflicting symbolic training family")
        if "task_role" in metadata or "split" in metadata:
            if metadata.get("task_role") != symbolic_task_role(spatial[0], metadata.get("split")):
                raise ValueError("conflicting symbolic task role")
    for key in (
        "category",
        "suite",
        "density_bin",
        "row_kind",
        "curriculum_stage",
        "grounding_stage",
        "task_family",
        "task_type",
        "entity_type",
        "relationship",
        "polarity",
        "marker_style",
        "probe_style",
        "piece",
        "color",
        "color_heldout",
        "target_token",
        "queried_token",
        "pair_kind",
        "partner_token",
        "partner_piece",
        "partner_color",
        "partner_distance",
        "same_color",
        "negative_distance",
        "negative_kind",
        "state_id",
        "layout_id",
        "piece_count",
        "item_count",
        "occupied_count",
        "eval_set_id",
        "eval_source_sha256",
    ):
        if key in row:
            metadata[key] = row[key]
    metadata["eval_variant"] = image_variant
    return metadata


def normalize_non_lora_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize the prefixes emitted by the pinned PEFT training path."""

    normalized = {
        (key[11:] if key.startswith("base_model.") else key): value
        for key, value in state_dict.items()
    }
    if any(key.startswith("model.model.") for key in normalized):
        normalized = {
            (key[6:] if key.startswith("model.") else key): value
            for key, value in normalized.items()
        }
    return normalized


def load_non_lora_adapter_weights(
    model: Any,
    adapter_dir: Path,
    torch: Any,
) -> dict[str, Any]:
    """Restore full visual/merger weights saved beside a PEFT adapter."""

    state_path = adapter_dir / "non_lora_state_dict.bin"
    launch_manifest_path = adapter_dir / "launch_manifest.json"
    requires_non_lora = False
    if launch_manifest_path.is_file():
        launch_manifest = json.loads(launch_manifest_path.read_text())
        requires_non_lora = launch_manifest.get("schema") in {
            "catan_qwen_vision_sft_launch_identity/v1",
            "catan_qwen_vision_sft_launch_identity/v2",
        }

    if not state_path.is_file():
        if requires_non_lora:
            raise FileNotFoundError(
                f"Vision-SFT adapter is missing required non-LoRA weights: {state_path}"
            )
        return {"loaded": False, "path": None, "tensors": 0}

    state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
    normalized = normalize_non_lora_state_dict(state_dict)
    incompatible = model.load_state_dict(normalized, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected non-LoRA state keys: " + ", ".join(incompatible.unexpected_keys[:20])
        )
    print(f"loaded_non_lora_state={state_path} tensors={len(normalized)}")
    return {
        "loaded": True,
        "path": str(state_path),
        "tensors": len(normalized),
        "missing_model_keys": len(incompatible.missing_keys),
    }


def load_model(
    *,
    model_id: str,
    adapter_dir: str | None,
    bits: int,
    disable_flash_attn2: bool,
    token_inventory: str | None = None,
    preserve_visual_fp32: bool = False,
    input_mode: str = "vision",
    model_revision: str | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"unsupported input_mode: {input_mode}")
    text_only = input_mode == "text"
    adapter_path = Path(adapter_dir) if adapter_dir else None
    if text_only:
        if adapter_path is None:
            raise ValueError("text mode requires --adapter-dir with a saved tokenizer and atlas rows")
        preserve_visual_fp32 = True
    if preserve_visual_fp32 and (
        adapter_path is None or not (adapter_path / VISUAL_STATE_FILE).is_file()
    ):
        raise ValueError(
            f"--preserve-visual-fp32 requires an --adapter-dir containing {VISUAL_STATE_FILE}"
        )

    torch = importlib.import_module("torch")
    peft = importlib.import_module("peft")
    transformers = importlib.import_module("transformers")

    processor_source = (
        adapter_path
        if adapter_path is not None and (adapter_path / "tokenizer_config.json").is_file()
        else model_id
    )
    inventory = None
    revision_kwargs = {"revision": model_revision} if model_revision is not None else {}
    if text_only:
        if token_inventory is None:
            raise ValueError("--token-inventory is required for semantic-token evaluation")
        inventory = load_token_inventory(token_inventory)
        processor = load_checkpoint_text_tokenizer(adapter_path, inventory["tokens"])
        parent = json.loads((adapter_path / RUN_CONFIG_FILE).read_text())
        if parent.get("model_id") != model_id:
            raise ValueError("text adapter base model differs from --model-id")
    else:
        processor = transformers.AutoProcessor.from_pretrained(
            processor_source, **(revision_kwargs if processor_source == model_id else {}),
        )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    quantization_config = None
    unquantized_modules = ["model.visual", "visual", "lm_head"]
    if bits == 4:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=unquantized_modules,
        )
    elif bits == 8:
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=unquantized_modules,
        )
    elif bits != 16:
        raise ValueError(f"Unsupported bits: {bits}")

    model = transformers.AutoModelForMultimodalLM.from_pretrained(
        model_id,
        **revision_kwargs,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa" if disable_flash_attn2 else "flash_attention_2",
        quantization_config=quantization_config,
    )
    quantizer = getattr(model, "hf_quantizer", None)
    if quantizer is not None:
        print(
            "modules_to_not_quantize="
            + json.dumps(getattr(quantizer, "modules_to_not_convert", []))
        )
    if token_inventory is None:
        raise ValueError("--token-inventory is required for semantic-token evaluation")
    inventory = inventory if inventory is not None else load_token_inventory(token_inventory)
    token_setup, components = prepare_semantic_tokens(processor, model, inventory)
    if text_only:
        validate_text_adapter(model, adapter_path, token_setup, components)
        model.requires_grad_(False)
    print(
        f"semantic_tokens={len(token_setup.tokens)} "
        f"token_ids={min(token_setup.token_ids)}-{max(token_setup.token_ids)} "
        f"model_vocab={token_setup.model_vocab_size}"
    )

    adapter_evidence = {
        "adapter_loaded": False,
        "adapter_dir": adapter_dir,
        "semantic_tokens": token_setup.as_dict(),
        "visual_state": {"loaded": False, "path": None, "tensors": 0},
        "non_lora_state": {"loaded": False, "path": None, "tensors": 0},
    }
    if text_only:
        adapter_evidence["input_mode"] = "text"
    if adapter_dir:
        assert adapter_path is not None
        visual_path = adapter_path / VISUAL_STATE_FILE
        if not visual_path.is_file():
            adapter_evidence["non_lora_state"] = load_non_lora_adapter_weights(
                model,
                adapter_path,
                torch,
            )
        frozen_dir = adapter_path / "frozen_adapter"
        if frozen_dir.is_dir():
            # An O-LoRA bundle only reproduces its model on top of the parent
            # adapter it carries: merge that into the base before loading it.
            frozen = peft.PeftModel.from_pretrained(model, frozen_dir, is_trainable=False)
            model = frozen.merge_and_unload()
            adapter_evidence["frozen_adapter"] = {"loaded": True, "path": str(frozen_dir)}
            print(f"merged_frozen_adapter={frozen_dir}")
        model = peft.PeftModel.from_pretrained(model, adapter_dir)
        if visual_path.is_file():
            if preserve_visual_fp32:
                # PEFT must establish the saved key layout first. Promote before
                # copying the source, not after a lossy FP32 -> BF16 restore.
                visual = resolve_wrapped_module(model, "model.visual")
                visual.float()
            visual_evidence = load_visual_state(model, adapter_path)
            adapter_evidence["visual_state"] = {
                "loaded": True,
                **visual_evidence,
            }
            if preserve_visual_fp32:
                with safe_open(visual_path, framework="pt", device="cpu") as source:
                    source_dtypes = Counter(source.get_slice(key).get_dtype() for key in source.keys())
                loaded_state = visual.state_dict()
                loaded_dtypes = Counter(str(tensor.dtype) for tensor in loaded_state.values())
                if any(
                    tensor.is_floating_point() and tensor.dtype != torch.float32
                    for tensor in loaded_state.values()
                ):
                    raise RuntimeError("visual floating-point state is not fully FP32 after restore")
                adapter_evidence["visual_precision"] = {
                    "base_load_dtype": str(torch.bfloat16),
                    "promoted_before_restore": True,
                    "source_dtypes": dict(source_dtypes),
                    "loaded_dtypes": dict(loaded_dtypes),
                }
                print(
                    "visual_precision="
                    + json.dumps(adapter_evidence["visual_precision"], sort_keys=True)
                )
        adapter_evidence["adapter_loaded"] = True
        print(f"loaded_adapter={adapter_dir}")

    if text_only:
        freeze_visual_weights(model, components)
        model.requires_grad_(False)
    model.eval()
    return model, processor, adapter_evidence


def run_eval_job(
    *,
    model: Any,
    processor: Any,
    adapter_evidence: dict[str, Any],
    args: argparse.Namespace,
    eval_jsonl: str,
    image_variant: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Score one eval set under one image variant with an already-loaded model."""

    eval_path = Path(eval_jsonl)
    input_mode = getattr(args, "input_mode", "vision")
    if input_mode not in INPUT_MODES:
        raise ValueError(f"unsupported input_mode: {input_mode}")
    text_only = input_mode == "text"
    max_sequence_length = getattr(args, "max_sequence_length", None)
    if text_only:
        validate_text_budget(max_sequence_length)
        if image_variant != "original":
            raise ValueError("text mode does not support image variants")
    rows = [row for _, row in iter_jsonl(eval_path)]
    for line_number, row in enumerate(rows, start=1):
        if text_only:
            prompt, answer = _message_pair(row, line_number=line_number, input_mode="text")
            encode_text_pair(
                native_tokenizer(processor), prompt, answer,
                max_sequence_length=max_sequence_length,
            )
            continue
        reference = image_reference(row)
        image_path = (
            resolve_dataset_image(Path(args.image_root), reference)
            if args.image_root
            else resolve_dataset_asset(eval_path, reference)
        )
        row["image"] = str(image_path)
        row.pop("images", None)
    if args.limit is not None:
        rows = rows[: args.limit]
    skipped_without_target = 0
    if image_variant in ("target_occlusion", "control_occlusion"):
        # Occlusion needs a box to cover; rows such as "empty" hard negatives
        # carry no spatial target and are simply not part of that variant.
        eligible = [row for row in rows if len(row.get("spatial_targets") or []) == 1]
        skipped_without_target = len(rows) - len(eligible)
        rows = eligible
        if not rows:
            raise ValueError(f"no rows with a spatial target for {image_variant}")
    shuffled_images = _shuffled_image_map(rows) if image_variant == "shuffle" else None

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"

    atlas_tokens = list(adapter_evidence["semantic_tokens"]["tokens"])
    tokenizer = native_tokenizer(processor)
    candidate_ids_cache: dict[tuple[str, ...], list[int]] = {}
    preserve_visual_fp32 = getattr(args, "preserve_visual_fp32", False)

    records = []
    short_rows = [row for row in rows if not is_long_answer(row)]
    long_rows = [row for row in rows if is_long_answer(row)]
    batches = [short_rows[start : start + args.batch_size] for start in range(0, len(short_rows), args.batch_size)]
    batches += [long_rows[start : start + args.long_batch_size] for start in range(0, len(long_rows), args.long_batch_size)]
    with records_path.open("w") as handle:
        batch_start = 0
        for batch in batches:
            responses, first_logits = generate_responses(
                model=model,
                processor=processor,
                rows=batch,
                max_new_tokens=args.long_max_new_tokens if is_long_answer(batch[0]) else args.max_new_tokens,
                image_variant=image_variant,
                shuffled_images=shuffled_images,
                occlusion_margin=args.occlusion_margin,
                return_first_logits=args.candidate_scoring,
                preserve_visual_fp32=preserve_visual_fp32,
                input_mode=input_mode,
                max_sequence_length=max_sequence_length,
            )
            for offset, (row, response) in enumerate(zip(batch, responses, strict=True)):
                index = batch_start + offset + 1
                if offset == len(batch) - 1:
                    batch_start += len(batch)
                target = expected_text(row)
                metadata = evaluation_metadata(row, image_variant=image_variant)
                if text_only:
                    metadata["input_mode"] = "text"
                category = metadata.get("category") or metadata.get("task_type", "")
                score = score_response(target, response, metadata=metadata)
                candidate_score = None
                if args.candidate_scoring:
                    candidates = candidate_answers(row, target, atlas_tokens)
                    if candidates is not None:
                        key = tuple(candidates)
                        if key not in candidate_ids_cache:
                            candidate_ids_cache[key] = candidate_token_ids(tokenizer, candidates)
                        candidate_score = score_candidates(
                            first_logits[offset],
                            candidates,
                            candidate_ids_cache[key],
                            target,
                        )
                record = {
                    "index": index,
                    "id": row.get("id") or row.get("row_id"),
                    "metadata": metadata,
                    "expected": target,
                    "response": response,
                    "score": score,
                    "candidate_score": candidate_score,
                }
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                candidate_note = (
                    f" candidate={candidate_score['predicted']!r}"
                    f" rank={candidate_score['expected_rank']}"
                    if candidate_score is not None
                    else ""
                )
                print(
                    f"[{index}/{len(rows)}] "
                    f"{category} correct={score['correct']} "
                    f"response={normalize_text(response)[:120]!r}{candidate_note}"
                )
            handle.flush()

    summary = summarize(records)
    eval_set_ids = {
        str(record["metadata"]["eval_set_id"])
        for record in records
        if record.get("metadata", {}).get("eval_set_id")
    }
    eval_source_hashes = {
        str(record["metadata"]["eval_source_sha256"])
        for record in records
        if record.get("metadata", {}).get("eval_source_sha256")
    }
    if len(eval_set_ids) > 1 or len(eval_source_hashes) > 1:
        raise ValueError("one eval job must contain exactly one immutable eval-set identity")
    summary.update(
        {
            "model_id": args.model_id,
            "model_revision": getattr(args, "model_revision", None),
            "adapter_dir": args.adapter_dir,
            "eval_jsonl": eval_jsonl,
            "bits": args.bits,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "long_max_new_tokens": args.long_max_new_tokens,
            "adapter_evidence": adapter_evidence,
            "image_root": None if text_only else args.image_root,
            "token_inventory": args.token_inventory,
            "reasoning_enabled": False,
            "image_variant": image_variant,
            "occlusion_margin": args.occlusion_margin,
            "candidate_scoring": bool(args.candidate_scoring),
            "precision": {
                "preserve_visual_fp32": preserve_visual_fp32 or text_only,
                "generation_autocast": (
                    {"device_type": model.device.type, "dtype": "torch.bfloat16"}
                    if preserve_visual_fp32 and not text_only
                    else {"policy": "caller_context"}
                ),
            },
            "rows_skipped_without_spatial_target": skipped_without_target,
            "eval_set_id": next(iter(eval_set_ids), None),
            "eval_source_sha256": next(iter(eval_source_hashes), None),
        }
    )
    if text_only:
        summary.update({"input_mode": "text", "max_sequence_length": max_sequence_length, "truncation": False})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def eval_jobs(args: argparse.Namespace) -> list[dict[str, str]]:
    """Expand eval sets x image variants into output directories.

    A single set under a single variant keeps the historical layout, writing
    straight into ``--output-dir``. Any batch nests each job under
    ``<output-dir>/<eval-stem>/<variant>`` so summaries never collide.
    """

    eval_sets = list(args.eval_jsonl)
    variants = [item.strip() for item in args.image_variant.split(",") if item.strip()]
    input_mode = getattr(args, "input_mode", "vision")
    if input_mode not in INPUT_MODES:
        raise ValueError(f"unsupported input_mode: {input_mode}")
    if input_mode == "text":
        validate_text_budget(getattr(args, "max_sequence_length", None))
        if variants != ["original"]:
            raise ValueError("text mode does not support image variants")
    for variant in variants:
        if variant not in IMAGE_VARIANTS:
            raise ValueError(f"unsupported image variant: {variant}")
    single = len(eval_sets) == 1 and len(variants) == 1
    jobs = []
    for eval_jsonl in eval_sets:
        parts = Path(eval_jsonl).parts
        # Three trailing path parts keep sets apart when several datasets share
        # a stage1/validation.jsonl layout.
        stem = "-".join(part for part in parts[-3:] if part).removesuffix(".jsonl")
        for variant in variants:
            output_dir = Path(args.output_dir)
            if not single:
                output_dir = output_dir / stem / variant
            jobs.append({"eval_jsonl": eval_jsonl, "image_variant": variant, "output_dir": str(output_dir)})
    return jobs


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Load the adapter once, then score every requested set and variant."""

    jobs = eval_jobs(args)
    if getattr(args, "input_mode", "vision") == "text":
        assert_runtime_versions()
    model, processor, adapter_evidence = load_model(
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        bits=args.bits,
        disable_flash_attn2=args.disable_flash_attn2,
        token_inventory=args.token_inventory,
        preserve_visual_fp32=getattr(args, "preserve_visual_fp32", False),
        input_mode=getattr(args, "input_mode", "vision"),
        model_revision=getattr(args, "model_revision", None),
    )
    results = []
    for job in jobs:
        summary = run_eval_job(
            model=model,
            processor=processor,
            adapter_evidence=adapter_evidence,
            args=args,
            eval_jsonl=job["eval_jsonl"],
            image_variant=job["image_variant"],
            output_dir=Path(job["output_dir"]),
        )
        results.append(
            {
                **job,
                "rows": summary["rows"],
                "exact_accuracy": summary["exact_accuracy"],
                "candidate_exact_accuracy": summary.get("candidate_exact_accuracy"),
                "candidate_expected_rank_mean": summary.get("candidate_expected_rank_mean"),
                "eval_set_id": summary.get("eval_set_id"),
                "eval_source_sha256": summary.get("eval_source_sha256"),
            }
        )
        if getattr(args, "input_mode", "vision") == "text":
            results[-1]["input_mode"] = "text"
    if len(jobs) == 1:
        return results[0]
    batch = {
        "schema": "catan_qwen_eval_batch/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "adapter_dir": args.adapter_dir,
        "model_id": args.model_id,
        "model_revision": getattr(args, "model_revision", None),
        "jobs": results,
    }
    if getattr(args, "input_mode", "vision") == "text":
        batch.update({"input_mode": "text", "max_sequence_length": args.max_sequence_length})
    batch_path = Path(args.output_dir) / "batch_summary.json"
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n")
    print(json.dumps(batch, indent=2, sort_keys=True))
    return batch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-jsonl",
        required=True,
        nargs="+",
        help="One or more eval sets; all are scored with a single model load.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--model-revision", help="Pinned base-model revision (branch, tag, or commit)")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--image-root")
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="vision")
    parser.add_argument("--max-sequence-length", type=int, help="Required text context budget including generation")
    parser.add_argument("--token-inventory")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8, 16])
    parser.add_argument(
        "--preserve-visual-fp32",
        action="store_true",
        help=(
            "Restore visual_model.safetensors into FP32 visual weights and generate under "
            "BF16 autocast. Use --bits 16 to match the full-board checkpoint evaluation."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--long-max-new-tokens", type=int, default=512)
    parser.add_argument("--long-batch-size", type=int, default=8)
    parser.add_argument(
        "--image-variant",
        default="original",
        help="Comma-separated subset of: " + ", ".join(IMAGE_VARIANTS),
    )
    parser.add_argument("--occlusion-margin", type=float, default=0.03)
    parser.add_argument(
        "--candidate-scoring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also rank the row's closed answer set by first-token log-probability.",
    )
    parser.add_argument("--disable-flash-attn2", action="store_true", default=True)
    return parser.parse_args(argv)


def main() -> int:
    run_eval(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
