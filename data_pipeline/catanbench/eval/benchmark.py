"""OpenBench/Inspect task for CatanBench public board QA."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from data_pipeline.catanbench.scoring import (
    DEFAULT_CATEGORIES,
    LOGIC_SYSTEM_PROMPT,
    PROBE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    SUITE_CATEGORIES,
    JsonDict,
    build_prompt,
    score_answer,
    select_questions,
    split_csv,
)

try:
    from inspect_ai import Task, task
except ImportError:  # Allows static checks without installing eval extras.
    Task = Any

    def task(func: Any) -> Any:
        return func


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCH_DIR = PROJECT_ROOT / "data_pipeline" / "catanbench" / "datasets" / "catanbench_100"
DEFAULT_QUESTION_DIR = DEFAULT_BENCH_DIR / "questions"
DEFAULT_PROBE_DIR = PROJECT_ROOT / "sft" / "data" / "isolated_visuals"
DEFAULT_PROBE_QUESTION_DIR = DEFAULT_PROBE_DIR / "questions"


def _as_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _categories(value: str | Sequence[str] | None) -> list[str]:
    if value is None or value == "default":
        return list(DEFAULT_CATEGORIES)
    if isinstance(value, str):
        return split_csv(value)
    return list(value)


def _suite_categories(suite: str, categories: str | Sequence[str] | None) -> list[str]:
    if categories is None or categories == "default":
        try:
            return list(SUITE_CATEGORIES[suite])
        except KeyError as exc:
            valid = ", ".join(sorted(SUITE_CATEGORIES))
            raise ValueError(f"unknown CatanBench suite {suite!r}; expected one of: {valid}") from exc
    return _categories(categories)


def _input_mode(suite: str, input_mode: str) -> str:
    if input_mode != "auto":
        return input_mode
    return "text" if suite == "logic" else "image"


def _resolved_paths(
    suite: str,
    bench_dir: str | Path,
    question_dir: str | Path,
) -> tuple[Path, Path]:
    bench_path = Path(bench_dir)
    question_path = Path(question_dir)
    if suite == "probe":
        if bench_path == DEFAULT_BENCH_DIR:
            bench_path = DEFAULT_PROBE_DIR
        if question_path == DEFAULT_QUESTION_DIR:
            question_path = DEFAULT_PROBE_QUESTION_DIR
    return bench_path, question_path


def selected_catanbench_items(
    bench_dir: str | Path = DEFAULT_BENCH_DIR,
    question_dir: str | Path = DEFAULT_QUESTION_DIR,
    suite: str = "visual",
    categories: str | Sequence[str] | None = "default",
    limit_samples: int = 10,
    questions_per_sample: int = 0,
    max_requests: Optional[int] = None,
) -> list[JsonDict]:
    """Return engine-scored QA items used by both tests and the Inspect task."""

    selected_categories = _suite_categories(suite, categories)
    per_sample = int(questions_per_sample) or len(selected_categories)
    bench_path, question_path = _resolved_paths(suite, bench_dir, question_dir)
    return select_questions(
        bench_path,
        question_dir=question_path,
        categories=selected_categories,
        limit_samples=int(limit_samples),
        questions_per_sample=per_sample,
        max_requests=max_requests,
        selection_mode="flat" if suite == "probe" else "sample",
    )


def _sample_metadata(qa: JsonDict) -> JsonDict:
    return {key: value for key, value in qa.items() if key != "contract"}


def _image_content(image_path: Path, image_detail: str | None) -> Any:
    from inspect_ai.model import ContentImage

    if image_detail:
        return ContentImage(image=str(image_path), detail=image_detail)
    return ContentImage(image=str(image_path))


def build_samples(
    bench_dir: str | Path = DEFAULT_BENCH_DIR,
    question_dir: str | Path = DEFAULT_QUESTION_DIR,
    suite: str = "visual",
    categories: str | Sequence[str] | None = "default",
    limit_samples: int = 10,
    questions_per_sample: int = 0,
    max_requests: Optional[int] = None,
    atlas_prompt: bool | str = True,
    include_system: bool | str = True,
    input_mode: str = "auto",
    image_detail: str | None = "auto",
) -> list[Any]:
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageUser, ContentText

    bench_path, question_path = _resolved_paths(suite, bench_dir, question_dir)
    resolved_input_mode = _input_mode(suite, input_mode)
    items = selected_catanbench_items(
        bench_path,
        question_dir=question_path,
        suite=suite,
        categories=categories,
        limit_samples=limit_samples,
        questions_per_sample=questions_per_sample,
        max_requests=max_requests,
    )
    samples = []
    for qa in items:
        use_atlas = _as_bool(atlas_prompt) and suite != "probe"
        prompt = build_prompt(qa, use_atlas=use_atlas)
        if not _as_bool(include_system):
            prompt = f"{_system_prompt(suite, resolved_input_mode)}\n\n---\n\n{prompt}"
        content = [ContentText(text=prompt)]
        if resolved_input_mode == "image":
            image_path = (bench_path / qa["image_path"]).resolve()
            content.insert(0, _image_content(image_path, image_detail))
        samples.append(
            Sample(
                id=qa["id"],
                input=[ChatMessageUser(content=content)],
                target=qa["answer"],
                metadata={"qa": _sample_metadata(qa)},
            )
        )
    return samples


def _system_prompt(suite: str, input_mode: str) -> str:
    if suite == "probe":
        return PROBE_SYSTEM_PROMPT
    if suite == "logic" or input_mode == "text":
        return LOGIC_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def _score_from_state(state: Any) -> tuple[str, JsonDict]:
    qa = state.metadata["qa"]
    response = state.output.completion
    return response, score_answer(qa, response)


def catanbench_exact_scorer() -> Any:
    from inspect_ai.scorer import CORRECT, INCORRECT, Score, accuracy, scorer, stderr

    @scorer(metrics=[accuracy(), stderr()])
    def _scorer() -> Any:
        async def score(state: Any, target: Any) -> Score:
            response, result = _score_from_state(state)
            return Score(
                value=CORRECT if result["correct"] else INCORRECT,
                answer=response,
                explanation=f"expected={target.text}",
                metadata=result,
            )

        return score

    return _scorer()


def catanbench_component_scorer() -> Any:
    from inspect_ai.scorer import Score, mean, scorer, stderr

    @scorer(metrics=[mean(), stderr()])
    def _scorer() -> Any:
        async def score(state: Any, target: Any) -> Score:
            response, result = _score_from_state(state)
            return Score(
                value=float(result["component_accuracy"]),
                answer=response,
                explanation=f"expected={target.text}",
                metadata=result,
            )

        return score

    return _scorer()


@task
def catanbench(
    bench_dir: str = str(DEFAULT_BENCH_DIR),
    question_dir: str = str(DEFAULT_QUESTION_DIR),
    suite: str = "visual",
    categories: str = "default",
    limit_samples: int = 10,
    questions_per_sample: int = 0,
    max_requests: Optional[int] = None,
    atlas_prompt: bool = True,
    include_system: bool = True,
    input_mode: str = "auto",
    image_detail: str = "auto",
) -> Any:
    """Catan public-board VLM benchmark.

    Task parameters are exposed through OpenBench/Inspect `-T` flags.
    """

    from inspect_ai.dataset import MemoryDataset
    from inspect_ai.solver import generate, system_message

    samples = build_samples(
        bench_dir=bench_dir,
        question_dir=question_dir,
        suite=suite,
        categories=categories,
        limit_samples=limit_samples,
        questions_per_sample=questions_per_sample,
        max_requests=max_requests,
        atlas_prompt=atlas_prompt,
        include_system=include_system,
        input_mode=input_mode,
        image_detail=image_detail,
    )
    if not samples:
        raise ValueError("CatanBench selected zero samples; check bench_dir/categories.")

    solver = [generate()]
    resolved_input_mode = _input_mode(suite, input_mode)
    if _as_bool(include_system):
        solver.insert(0, system_message(_system_prompt(suite, resolved_input_mode)))

    return Task(
        name=f"catanbench_{suite}",
        dataset=MemoryDataset(samples=samples),
        solver=solver,
        scorer=[catanbench_exact_scorer(), catanbench_component_scorer()],
        metadata={
            "suite": suite,
            "input_mode": resolved_input_mode,
            "bench_dir": str(_resolved_paths(suite, bench_dir, question_dir)[0]),
            "question_dir": str(_resolved_paths(suite, bench_dir, question_dir)[1]),
            "categories": _suite_categories(suite, categories),
            "limit_samples": limit_samples,
            "questions_per_sample": questions_per_sample
            or len(_suite_categories(suite, categories)),
            "atlas_prompt": atlas_prompt and suite != "probe",
            "image_detail": image_detail,
        },
    )
