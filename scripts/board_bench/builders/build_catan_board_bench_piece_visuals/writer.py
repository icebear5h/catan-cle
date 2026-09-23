"""Accumulate samples and write the dataset artifacts."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.board_bench.shapes import JsonDict, text, write_json, write_jsonl

__all__ = ["DatasetWriter"]


class DatasetWriter:
    def __init__(self, output_dir: Path, image_size: int, prompt_prefix: str) -> None:
        self.output_dir = output_dir
        self.image_dir = output_dir / "images"
        self.question_dir = output_dir / "questions"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = image_size
        self.prompt_prefix = prompt_prefix
        self.manifest_rows: list[JsonDict] = []
        self.qa_rows: list[JsonDict] = []

    def add_sample(
        self,
        *,
        sample_id: str,
        category: str,
        image: Image.Image,
        question: str,
        answer: str,
        target: JsonDict,
        source: JsonDict | None = None,
        contract_path: str | None = None,
    ) -> None:
        image_rel = Path("images") / f"{sample_id}.png"
        image.save(self.output_dir / image_rel)
        qa_id = f"{sample_id}_q00_{category}"
        self.manifest_rows.append(
            {
                "sample_id": sample_id,
                "category": category,
                "image_path": str(image_rel),
                "source": source or {"kind": "frontend_asset_composition"},
                "target": target,
            }
        )
        self.qa_rows.append(
            {
                "id": qa_id,
                "sample_id": sample_id,
                "category": category,
                "image_path": str(image_rel),
                "contract_path": contract_path,
                "question": question,
                "answer": answer,
                "target": target,
                "scoring": "exact",
            }
        )

    def write(self, metadata: JsonDict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.question_dir.mkdir(parents=True, exist_ok=True)

        write_jsonl(self.output_dir / "manifest.jsonl", self.manifest_rows)
        write_jsonl(self.question_dir / "qa.jsonl", self.qa_rows)
        write_jsonl(
            self.question_dir / "questions.jsonl",
            [
                {
                    "id": row["id"],
                    "sample_id": row["sample_id"],
                    "image_path": row["image_path"],
                    "contract_path": row["contract_path"],
                    "category": row["category"],
                    "question": row["question"],
                }
                for row in self.qa_rows
            ],
        )
        write_jsonl(
            self.question_dir / "answer_key.jsonl",
            [
                {
                    "id": row["id"],
                    "sample_id": row["sample_id"],
                    "category": row["category"],
                    "answer": row["answer"],
                    "target": row["target"],
                    "scoring": row["scoring"],
                }
                for row in self.qa_rows
            ],
        )
        write_jsonl(
            self.output_dir / "messages.jsonl",
            [self._message_row(row) for row in self.qa_rows],
        )

        (self.output_dir / "README.md").write_text(
            "# Isolated Catan Visuals\n\n"
            "Primitive visual-grounding examples composed from frontend assets and "
            "local board crops.\n\n"
            "Pillow is used for composition only. SVG/PNG art is loaded from "
            "`playground/frontend/public/assets`.\n\n"
            f"Samples: {len(self.manifest_rows)}\n"
            f"QA rows: {len(self.qa_rows)}\n",
        )
        write_json(self.output_dir / "metadata.json", metadata)

    def _message_row(self, row: JsonDict) -> JsonDict:
        return {
            "id": row["id"],
            "image": row["image_path"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": f"{self.prompt_prefix}\n\nQuestion: "
                            f"{text(row['question'], 'question')}",
                        },
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": row["answer"]}]},
            ],
            "metadata": {
                "phase": "isolated_visual_grounding",
                "category": row["category"],
                "sample_id": row["sample_id"],
                "target": row["target"],
            },
        }
