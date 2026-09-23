"""Shared fixtures for board recognition eval suite build, cli, and validation contracts."""
from pathlib import Path
from typing import Any

import pytest

from scripts.board_recognition.build_catan_board_recognition_eval_suite import (
    write_jsonl,
)

from .support import CORRECTED_SOURCE, JsonRow


@pytest.fixture
def supplement(tmp_path: Path) -> tuple[Path, list[JsonRow], list[JsonRow]]:
    source: Any = tmp_path / CORRECTED_SOURCE
    audits: Any = [
        {
            "query_id": "node-question",
            "task_family": "spatial_grounding",
            "task_type": "node_direction_token",
            "prompt": "Which node is above the other: <N00> or <N01>?",
            "answer": "<N01>",
        },
        {
            "query_id": "robber-question",
            "task_family": "robber",
            "task_type": "robber_presence_positive",
            "prompt": "Is the robber on <T00>?",
            "answer": "yes",
        },
        {
            "query_id": "tile-question",
            "task_family": "spatial_grounding",
            "task_type": "tile_adjacent_no",
            "prompt": "Are <T00> and <T18> adjacent tiles?",
            "answer": "no",
        },
    ]
    rows: Any = []
    for audit in audits:
        audit.update(state_id="board", split="validation", image_name="nested/board.png")
        rows.append(
            {
                "images": [audit["image_name"]],
                "messages": [
                    {"role": "user", "content": f"<image>\n{audit['prompt']}"},
                    {"role": "assistant", "content": audit["answer"]},
                ],
            }
        )
    write_jsonl(source / "validation.jsonl", rows)
    write_jsonl(source / "audit/validation.jsonl", audits)
    return source, rows, audits
