"""Question selection, contract attachment, and QA record copying."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from evals.catan_board_bench.scoring.categories import (
    JsonDict,
)
from evals.json_types import as_str


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def select_questions(
    bench_dir: Path,
    *,
    question_dir: Optional[Path] = None,
    categories: Sequence[str],
    limit_samples: int,
    questions_per_sample: int,
    max_requests: Optional[int],
    selection_mode: str = "sample",
) -> List[JsonDict]:
    question_root = question_dir or bench_dir
    qas: List[JsonDict] = [
        json.loads(line) for line in (question_root / "qa.jsonl").read_text().splitlines()
    ]
    contracts: Dict[str, JsonDict] = {}
    selected: List[JsonDict] = []
    category_set = set(categories)
    qas_by_sample: Dict[str, List[JsonDict]] = defaultdict(list)
    sample_order: List[str] = []

    for qa in qas:
        sample_id = as_str(qa["sample_id"], "qa sample_id")
        if sample_id not in qas_by_sample:
            sample_order.append(sample_id)
        if qa["category"] in category_set:
            qas_by_sample[sample_id].append(qa)

    if selection_mode == "flat":
        for category in categories:
            category_qas = [
                qa
                for sample_id in sample_order
                for qa in qas_by_sample[sample_id]
                if qa["category"] == category
            ]
            limit = int(limit_samples)
            for qa in category_qas[:limit] if limit > 0 else category_qas:
                qa = qa_copy(qa)
                attach_contract_if_available(qa, bench_dir, contracts)
                selected.append(qa)
                if max_requests is not None and len(selected) >= max_requests:
                    return selected
        return selected

    if selection_mode != "sample":
        raise ValueError(f"unknown question selection_mode {selection_mode!r}")

    for sample_id in sample_order[:limit_samples]:
        sample_qas = qas_by_sample[sample_id]
        by_category: Dict[str, List[JsonDict]] = defaultdict(list)
        for qa in sample_qas:
            by_category[as_str(qa["category"], "qa category")].append(qa)

        per_category_index: Counter[str] = Counter()
        sample_selected = 0
        while sample_selected < questions_per_sample:
            made_progress = False
            for category in categories:
                category_qas = by_category.get(category, [])
                idx = per_category_index[category]
                if idx >= len(category_qas):
                    continue
                qa = qa_copy(category_qas[idx])
                per_category_index[category] += 1

                attach_contract_if_available(qa, bench_dir, contracts)
                selected.append(qa)
                sample_selected += 1
                made_progress = True
                if max_requests is not None and len(selected) >= max_requests:
                    return selected
                if sample_selected >= questions_per_sample:
                    break

            if not made_progress:
                break

    return selected


def attach_contract_if_available(
    qa: JsonDict, bench_dir: Path, contracts: Dict[str, JsonDict]
) -> None:
    contract_ref = qa.get("contract_path")
    if not contract_ref:
        return
    contract_path = resolve_contract_path(bench_dir, as_str(contract_ref, "qa contract_path"))
    if contract_path is None:
        return
    contract_key = str(contract_path)
    if contract_key not in contracts:
        contracts[contract_key] = json.loads(contract_path.read_text())
    qa["contract"] = contracts[contract_key]


def resolve_contract_path(bench_dir: Path, contract_ref: str) -> Optional[Path]:
    contract_path = Path(contract_ref)
    if contract_path.is_absolute() and contract_path.exists():
        return contract_path
    bench_candidate = bench_dir / contract_path
    if bench_candidate.exists():
        return bench_candidate
    if contract_path.exists():
        return contract_path
    return None


def qa_copy(qa: JsonDict) -> JsonDict:
    """Copy a QA item without mutating the cached JSONL record."""

    copied: JsonDict = json.loads(json.dumps(qa))
    return copied

