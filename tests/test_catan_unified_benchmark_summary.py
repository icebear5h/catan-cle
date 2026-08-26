import pytest

from scripts.summarize_catan_unified_benchmark import (
    paired_modalities,
    validate_current_model_records,
)


def test_paired_modalities_keeps_image_and_text_panels_separate():
    image_records = []
    text_records = []
    for index in range(60):
        question_id = f"q{index:02d}"
        image_records.append(
            {
                "question_id": question_id,
                "category": "category",
                "score": {"correct": index < 11},
                "usage": {"prompt_tokens": 10},
            }
        )
        text_records.append(
            {
                "question_id": question_id,
                "category": "category",
                "score": {"correct": True},
                "usage": {"prompt_tokens": 20},
            }
        )

    paired = paired_modalities(
        image_label="qwen_image",
        image_records=image_records,
        text_label="qwen_text",
        text_records=text_records,
    )

    assert paired["pairs"] == 60
    assert paired["both_correct"] == 11
    assert paired["image_only"] == 0
    assert paired["text_only"] == 49
    assert paired["both_wrong"] == 0
    assert paired["image_exact"] == 11
    assert paired["text_exact"] == 60
    assert paired["prompt_tokens"]["text_over_image"] == 2.0
    assert paired["exact_mcnemar_binomial_p"] == pytest.approx(3.552713678800501e-15)


def test_old_qwen_checkpoint_is_rejected():
    with pytest.raises(ValueError, match="old Qwen checkpoint"):
        validate_current_model_records([{"model_id": "qwen/qwen3.8-27b"}])

    validate_current_model_records([{"model_id": "qwen/qwen3.8-max"}])
