import pytest

from evals.catan_board_bench.tokens import (
    add_tokens_to_tokenizer,
    node_token,
    recognition_answer_token,
    recognition_query_token,
    recognition_trainable_tokens,
)


transformers = pytest.importorskip("transformers")


def test_qwen_native_tokenizer_encodes_recognition_wire_format_atomically():
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.8-27B",
            local_files_only=True,
        )
    except OSError as exc:
        pytest.skip(f"Qwen3.8 tokenizer is not cached: {exc}")

    add_tokens_to_tokenizer(tokenizer)
    for token in recognition_trainable_tokens():
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        assert len(token_ids) == 1, (token, token_ids)
        assert tokenizer.convert_ids_to_tokens(token_ids[0]) == token

    query = node_token(10) + recognition_query_token("node.occupancy")
    answer = recognition_answer_token("node.occupancy", "WHITE_SETTLEMENT")
    assert len(tokenizer.encode(query, add_special_tokens=False)) == 2
    assert len(tokenizer.encode(answer, add_special_tokens=False)) == 1
