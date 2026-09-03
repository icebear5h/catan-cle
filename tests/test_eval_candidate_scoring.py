import pytest
import torch

from sft.scripts.eval_qwen_vl_adapter import (
    candidate_answers,
    candidate_token_ids,
    score_candidates,
    summarize,
)


ATLAS = [f"<N{i:02d}>" for i in range(54)] + ["<E00_01>", "<E00_05>"] + [f"<T{i:02d}>" for i in range(19)] + [f"<P{i:02d}>" for i in range(9)]


class _Tokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab[text]]


def test_candidate_sets_follow_entity_type_marker_letters_and_polarity():
    probe = {"metadata": {"entity_type": "node", "task_type": "neutral_probe_token_return"}}
    assert candidate_answers(probe, "<N07>", ATLAS) == [f"<N{i:02d}>" for i in range(54)]
    marker = {"task_type": "token_to_marker", "entity_type": "edge"}
    assert candidate_answers(marker, "C", ATLAS) == ["A", "B", "C", "D"]
    untyped = {"metadata": {}}
    assert candidate_answers(untyped, "<T04>", ATLAS) == [f"<T{i:02d}>" for i in range(19)]
    assert candidate_answers({"metadata": {"polarity": "positive"}}, "yes", ATLAS) == ["yes", "no"]
    assert candidate_answers({"metadata": {}}, "red settlement", ATLAS) is None
    with pytest.raises(ValueError):
        candidate_answers({"metadata": {"entity_type": "tile"}}, "<N07>", ATLAS)


def test_candidate_ranking_reports_argmax_and_expected_rank():
    candidates = ["<N00>", "<N01>", "<N02>"]
    tokenizer = _Tokenizer({"<N00>": 5, "<N01>": 6, "<N02>": 7, "<T00>": 9})
    ids = candidate_token_ids(tokenizer, candidates)
    logits = torch.full((12,), -5.0)
    logits[9] = 4.0  # a tile token wins the open argmax; it is not a candidate
    logits[6] = 2.0
    logits[7] = 1.0
    logits[5] = 0.0

    result = score_candidates(logits, candidates, ids, "<N02>")

    assert result["predicted"] == "<N01>"
    assert result["correct"] is False
    assert result["expected_rank"] == 2
    assert [item["answer"] for item in result["top"]] == ["<N01>", "<N02>", "<N00>"]
    assert result["expected_logprob"] < 0

    exact = score_candidates(logits, candidates, ids, "<N01>")
    assert exact["correct"] is True and exact["expected_rank"] == 1

    with pytest.raises(ValueError):
        candidate_token_ids(_Tokenizer({"a": 1, "b": 1}), ["a", "b"])


def test_summary_carries_candidate_accuracy_beside_exact_match():
    def record(correct: bool, candidate: dict | None, style: str) -> dict:
        return {
            "response": "x",
            "metadata": {"probe_style": style},
            "score": {"correct": correct},
            "candidate_score": candidate,
        }

    records = [
        record(False, {"correct": True, "expected_rank": 1}, "small"),
        record(False, {"correct": False, "expected_rank": 3}, "small"),
        record(True, {"correct": True, "expected_rank": 1}, "large"),
        record(True, None, "large"),
    ]

    summary = summarize(records)

    assert summary["exact_accuracy"] == pytest.approx(0.5)
    assert summary["candidate_rows"] == 3
    assert summary["candidate_exact_accuracy"] == pytest.approx(2 / 3)
    assert summary["candidate_expected_rank_le3"] == pytest.approx(1.0)
    small = summary["by_probe_style"]["small"]
    assert small["candidate_exact_accuracy"] == pytest.approx(0.5)
    assert small["candidate_expected_rank_mean"] == pytest.approx(2.0)
    assert "candidate_total" not in summary["by_probe_style"]["large"] or summary["by_probe_style"]["large"]["candidate_total"] == 1


def test_eval_jobs_keep_single_layout_and_nest_batches(tmp_path):
    import argparse

    from sft.scripts.eval_qwen_vl_adapter import eval_jobs

    single = argparse.Namespace(
        eval_jsonl=["probes/validation.jsonl"], image_variant="original", output_dir=str(tmp_path)
    )
    assert eval_jobs(single) == [
        {"eval_jsonl": "probes/validation.jsonl", "image_variant": "original", "output_dir": str(tmp_path)}
    ]

    batch = argparse.Namespace(
        eval_jsonl=["probes/validation.jsonl", "stage1/validation.jsonl"],
        image_variant="original, blank",
        output_dir=str(tmp_path),
    )
    jobs = eval_jobs(batch)
    assert [job["output_dir"] for job in jobs] == [
        str(tmp_path / "probes-validation" / "original"),
        str(tmp_path / "probes-validation" / "blank"),
        str(tmp_path / "stage1-validation" / "original"),
        str(tmp_path / "stage1-validation" / "blank"),
    ]
    clash = argparse.Namespace(
        eval_jsonl=["v2/stage1/validation.jsonl", "v3/stage1/validation.jsonl"],
        image_variant="original",
        output_dir=str(tmp_path),
    )
    assert len({job["output_dir"] for job in eval_jobs(clash)}) == 2
    with pytest.raises(ValueError):
        eval_jobs(argparse.Namespace(eval_jsonl=["a.jsonl"], image_variant="sepia", output_dir="x"))
