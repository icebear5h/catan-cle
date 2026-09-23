"""Semantic row seeding and answer token metrics."""

import pytest
import torch

from sft.scripts.train.train_trl_catan_vision import (
    ModelComponents,
    TokenSetup,
    answer_token_metrics,
    initialize_semantic_token_rows,
)

from .support import _EmbeddingModel


def test_semantic_rows_are_seeded_from_base_vocabulary_mean_and_are_distinct() -> None:
    vocab, hidden = 512, 8
    token_ids = tuple(range(vocab - 154 - 4, vocab - 4))
    setup = TokenSetup(
        tokens=tuple(f"<T{index}>" for index in range(154)),
        token_ids=token_ids,
        tokenizer_size=vocab - 4,
        model_vocab_size=vocab,
        added_tokens=154,
    )
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        vocab_size=vocab,
        hidden_size=hidden,
    )
    torch.manual_seed(0)
    first = _EmbeddingModel(vocab, hidden)
    torch.manual_seed(0)
    second = _EmbeddingModel(vocab, hidden)

    report = initialize_semantic_token_rows(first, components, setup, seed=42)
    initialize_semantic_token_rows(second, components, setup, seed=42)

    ids = torch.tensor(token_ids)
    for module in (first.model.language_model.embed_tokens, first.lm_head):
        base_mean = module.weight[: min(token_ids)].mean(dim=0)
        rows = module.weight[ids]
        assert torch.allclose(rows.mean(dim=0), base_mean, atol=0.2)
        assert rows.unique(dim=0).shape[0] == 154
        assert not torch.any(rows.abs() == 7.0)
    assert torch.equal(first.lm_head.weight, second.lm_head.weight)
    assert report["input_embedding"]["row_norm_after"] < report["input_embedding"]["row_norm_before"]


def test_family_words_init_seeds_each_family_from_its_word_row() -> None:
    vocab, hidden = 512, 8
    token_ids = tuple(range(vocab - 154 - 4, vocab - 4))
    tokens = tuple([f"<N{i:02d}>" for i in range(54)] + [f"<E{i:02d}_{i + 1:02d}>" for i in range(72)] + [f"<T{i:02d}>" for i in range(19)] + [f"<P{i:02d}>" for i in range(9)])
    family_word_ids = {"N": (10,), "E": (20, 21), "T": (30,), "P": (40,)}
    setup = TokenSetup(
        tokens=tokens,
        token_ids=token_ids,
        tokenizer_size=vocab - 4,
        model_vocab_size=vocab,
        added_tokens=154,
        family_word_ids=family_word_ids,
    )
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        vocab_size=vocab,
        hidden_size=hidden,
    )
    torch.manual_seed(0)
    model = _EmbeddingModel(vocab, hidden)
    with torch.no_grad():
        for family, ids in family_word_ids.items():
            for module in (model.model.language_model.embed_tokens, model.lm_head):
                module.weight[list(ids)] = float(ord(family)) / 10.0

    report = initialize_semantic_token_rows(model, components, setup, seed=42, mode="family_words")

    ids = torch.tensor(token_ids)
    for module in (model.model.language_model.embed_tokens, model.lm_head):
        rows = module.weight[ids]
        for family, word_ids in family_word_ids.items():
            members = torch.tensor([i for i, token in enumerate(tokens) if token[1] == family])
            base = module.weight[list(word_ids)].mean(dim=0)
            assert torch.allclose(rows[members].mean(dim=0), base, atol=0.2)
        assert rows.unique(dim=0).shape[0] == 154
        node_mean = rows[:54].mean(dim=0)
        tile_mean = rows[126:145].mean(dim=0)
        assert not torch.allclose(node_mean, tile_mean, atol=0.5)
    assert report["mode"] == "family_words" and set(report["input_embedding"]["family_base_norms"]) == {"N", "E", "T", "P"}
    with pytest.raises(ValueError):
        initialize_semantic_token_rows(model, components, setup, seed=42, mode="sideways")


def test_vocab_gaussian_init_matches_the_base_vocabulary_spread() -> None:
    vocab, hidden = 512, 8
    token_ids = tuple(range(vocab - 154 - 4, vocab - 4))
    setup = TokenSetup(
        tokens=tuple(f"<T{index}>" for index in range(154)),
        token_ids=token_ids,
        tokenizer_size=vocab - 4,
        model_vocab_size=vocab,
        added_tokens=154,
    )
    components = ModelComponents(
        input_embedding="model.language_model.embed_tokens",
        output_head="lm_head",
        language="model.language_model",
        vision="model.visual",
        merger="model.visual.merger",
        vocab_size=vocab,
        hidden_size=hidden,
    )
    torch.manual_seed(0)
    model = _EmbeddingModel(vocab, hidden)
    report = initialize_semantic_token_rows(model, components, setup, seed=42, mode="vocab_gaussian")
    ids = torch.tensor(token_ids)
    for module in (model.model.language_model.embed_tokens, model.lm_head):
        reference = module.weight[: min(token_ids)]
        rows = module.weight[ids]
        assert torch.allclose(rows.mean(dim=0), reference.mean(dim=0), atol=0.3)
        ratio = rows.std(dim=0) / reference.std(dim=0)
        assert ratio.mean() > 0.7 and ratio.mean() < 1.3
        cosine = torch.nn.functional.normalize(rows - rows.mean(0), dim=1)
        offdiag = (cosine @ cosine.T).fill_diagonal_(0).abs().mean()
        assert offdiag < 0.5
    assert report["noise_scale"] == 1.0 and report["mode"] == "vocab_gaussian"


def test_answer_token_metrics_ignore_prompt_and_trivial_completion_tokens() -> None:
    vocab, hidden = 6, 4
    weight = torch.eye(vocab, hidden)
    # Row 0: prompt (-100), answer token 2 (correct), end-of-turn 4, newline 5.
    # Row 1: prompt, answer tokens 1 (correct) and 3 (wrong), end-of-turn 4.
    labels = torch.tensor([[-100, -100, 2, 4, 5], [-100, 1, 3, 4, -100]])
    hidden_states = torch.zeros(2, 5, hidden)
    hidden_states[0, 1, 2] = 1.0  # predicts 2 at the position preceding label 2
    hidden_states[1, 0, 1] = 1.0  # predicts 1
    hidden_states[1, 1, 0] = 1.0  # predicts 0, expected 3

    metrics = answer_token_metrics(hidden_states, labels, weight, None, trivial_token_ids=(4, 5))

    assert metrics["answer_token_count"] == 3.0
    assert metrics["answer_token_accuracy"] == pytest.approx(2 / 3)
    assert metrics["answer_row_exact"] == pytest.approx(0.5)
