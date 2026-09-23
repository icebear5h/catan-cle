"""Build CatanBoardBench records from Colonist replay files."""

from __future__ import annotations

import asyncio
import importlib
import math
from collections import Counter
from collections.abc import Sized
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, TypedDict, Unpack

from evals.catan_board_bench.builder.constants import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPLAY_DIR,
    BenchmarkBuildResult,
)
from evals.catan_board_bench.builder.readme import write_dataset_readme, write_question_readme
from evals.catan_board_bench.builder.records import (
    _answer_view,
    _question_view,
    _write_json,
    _write_jsonl,
    _write_square_png,
)
from evals.catan_board_bench.builder.replay import (
    _project_relative_path,
    load_colonist_replay,
    require_game,
    require_replay_data,
    step_replay,
)
from evals.catan_board_bench.builder.selection import _spaced_steps
from evals.catan_board_bench.builder.suite import CatanObservationSuite
from evals.catan_board_bench.paths import PROJECT_ROOT
from evals.json_types import JsonDict


class CatanBoardBenchBuilder:
    """Build CatanBoardBench records from Colonist replay files."""

    def __init__(
        self,
        *,
        replay_dir: Path = DEFAULT_REPLAY_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        question_dir: Optional[Path] = None,
        samples: int = 100,
        min_step: int = 8,
        render_images: bool = False,
        image_size: int = 512,
        quiet_replay: bool = True,
    ) -> None:
        self.replay_dir = Path(replay_dir)
        self.output_dir = Path(output_dir)
        self.question_dir = Path(question_dir) if question_dir is not None else self.output_dir
        self.samples = samples
        self.min_step = min_step
        self.render_images = render_images
        self.image_size = image_size
        self.quiet_replay = quiet_replay
        self.suite = CatanObservationSuite()

    async def build(self) -> BenchmarkBuildResult:
        self._prepare_output_dirs()
        replay_files = sorted(self.replay_dir.glob("*.json"))
        if not replay_files:
            raise FileNotFoundError(f"no replay JSON files found in {self.replay_dir}")

        per_replay_limit = max(1, math.ceil(self.samples / len(replay_files)))
        counters: Counter[str] = Counter()
        used_sources: set[Tuple[str, int]] = set()

        metadata_path = self.output_dir / "metadata.json"
        manifest_path = self.output_dir / "manifest.jsonl"
        questions_path = self.question_dir / "questions.jsonl"
        answer_key_path = self.question_dir / "answer_key.jsonl"
        qa_path = self.question_dir / "qa.jsonl"

        self._write_readme()
        self._write_question_readme()

        sample_count = 0
        qa_count = 0
        rendered_images = 0

        screenshotter = None
        if self.render_images:
            screenshotter_class = importlib.import_module(
                "playground.screenshot_board"
            ).FrontendScreenshotter

            screenshotter = screenshotter_class(
                headless=True,
                board_width=1200,
                board_height=1200,
                crop_pct=0.08,
                vertical_offset_pct=0.0,
            )
            await screenshotter.start()

        try:
            with (
                manifest_path.open("w") as manifest_f,
                questions_path.open("w") as questions_f,
                answer_key_path.open("w") as answer_key_f,
                qa_path.open("w") as qa_f,
            ):
                for replay_file in replay_files:
                    if sample_count >= self.samples:
                        break

                    state = load_colonist_replay(replay_file, quiet=self.quiet_replay)
                    parsed_actions = require_replay_data(state).get("parsed_actions", [])
                    if not isinstance(parsed_actions, Sized):
                        raise TypeError("replay parsed_actions is not a sequence")
                    total_steps = len(parsed_actions)
                    target_steps = set(_spaced_steps(total_steps, per_replay_limit, self.min_step))
                    replay_sample_count = 0

                    while state.replay_index < total_steps and sample_count < self.samples:
                        result = step_replay(state, quiet=self.quiet_replay)
                        if isinstance(result, tuple) or result.get("error"):
                            counters["replay_step_errors"] += 1
                            break
                        if result.get("finished"):
                            break

                        step_index = state.replay_index
                        source_key = (replay_file.name, step_index)
                        if step_index not in target_steps or source_key in used_sources:
                            continue

                        sample_id = f"sample_{sample_count:03d}"
                        contract_rel = Path("contracts") / f"{sample_id}.json"
                        image_rel: Optional[Path] = None

                        if screenshotter is not None:
                            image_rel = Path("images") / f"{sample_id}.png"
                            png = await screenshotter.screenshot(state.current_game, settle_ms=250)
                            _write_square_png(
                                png,
                                self.output_dir / image_rel,
                                size=self.image_size,
                            )
                            rendered_images += 1

                        game_id = require_replay_data(state).get("game_id")
                        if game_id is not None and not isinstance(game_id, str):
                            raise TypeError("replay game_id is not a string")
                        source: JsonDict = {
                            "kind": "colonist_replay",
                            "replay_file": str(replay_file.relative_to(PROJECT_ROOT)),
                            "game_id": game_id,
                            "replay_step": step_index,
                            "total_replay_steps": total_steps,
                            "engine_action_count": len(require_game(state).state.actions),
                            "replay_semantic_issue_count": len(
                                getattr(state, "replay_semantic_issues", [])
                            ),
                        }
                        sample_meta: JsonDict = {
                            "id": sample_id,
                            "index": sample_count,
                            "contract_path": str(contract_rel),
                            "image_path": str(image_rel) if image_rel else None,
                            "image_size": [self.image_size, self.image_size] if image_rel else None,
                        }

                        contract = self.suite.public_board_contract(
                            require_game(state),
                            sample=sample_meta,
                            source=source,
                        )
                        contract_path = self.output_dir / contract_rel
                        _write_json(contract_path, contract)

                        qas = self.suite.qa_pairs(contract)
                        for qa in qas:
                            _write_jsonl(qa_f, qa)
                            _write_jsonl(questions_f, _question_view(qa))
                            _write_jsonl(answer_key_f, _answer_view(qa))
                        qa_count += len(qas)

                        manifest_record: JsonDict = {
                            "sample_id": sample_id,
                            "contract_path": str(contract_rel),
                            "image_path": str(image_rel) if image_rel else None,
                            "question_count": len(qas),
                            "source": source,
                        }
                        _write_jsonl(manifest_f, manifest_record)

                        used_sources.add(source_key)
                        counters[f"replay:{replay_file.name}"] += 1
                        sample_count += 1
                        replay_sample_count += 1

                        if replay_sample_count >= per_replay_limit:
                            break

            metadata: JsonDict = {
                "name": "CatanBoardBench-100",
                "schema": "catan_board_bench/v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sample_count": sample_count,
                "qa_count": qa_count,
                "render_images": self.render_images,
                "rendered_images": rendered_images,
                "image_size": [self.image_size, self.image_size],
                "replay_dir": _project_relative_path(self.replay_dir),
                "source_replay_count": len(replay_files),
                "sampling": {
                    "target_samples": self.samples,
                    "per_replay_limit": per_replay_limit,
                    "min_step": self.min_step,
                },
                "counts": {key: count for key, count in counters.items()},
                "files": {
                    "manifest": manifest_path.name,
                    "question_dir": _project_relative_path(self.question_dir),
                    "questions": questions_path.name,
                    "answer_key": answer_key_path.name,
                    "qa": qa_path.name,
                    "contracts_dir": "contracts",
                    "images_dir": "images" if self.render_images else None,
                },
            }
            _write_json(metadata_path, metadata)
        finally:
            if screenshotter is not None:
                await screenshotter.stop()

        if sample_count < self.samples:
            raise RuntimeError(
                f"only built {sample_count}/{self.samples} samples from {len(replay_files)} replays"
            )

        return BenchmarkBuildResult(
            output_dir=self.output_dir,
            sample_count=sample_count,
            qa_count=qa_count,
            rendered_images=rendered_images,
            replay_count=len(replay_files),
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            questions_path=questions_path,
            answer_key_path=answer_key_path,
        )

    def _prepare_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.question_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "contracts").mkdir(parents=True, exist_ok=True)
        if self.render_images:
            (self.output_dir / "images").mkdir(parents=True, exist_ok=True)


    def _write_readme(self) -> None:
        write_dataset_readme(self.output_dir)

    def _write_question_readme(self) -> None:
        write_question_readme(self.question_dir)


class BuilderOptions(TypedDict, total=False):
    """Keyword options accepted by ``CatanBoardBenchBuilder``."""

    replay_dir: Path
    output_dir: Path
    question_dir: Optional[Path]
    samples: int
    min_step: int
    render_images: bool
    image_size: int
    quiet_replay: bool


def build_catan_board_bench_sync(**kwargs: Unpack[BuilderOptions]) -> BenchmarkBuildResult:
    """Synchronous wrapper for scripts/tests."""

    return asyncio.run(CatanBoardBenchBuilder(**kwargs).build())
