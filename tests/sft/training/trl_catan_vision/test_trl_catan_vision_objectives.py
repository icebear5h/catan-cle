"""Chunked NLL heads, collators, and soft patch objectives."""

import pytest
import torch
import torch.nn.functional as F

from sft.scripts.train.train_trl_catan_vision import (
    SpatialTargetCollator,
    TokenSetup,
    _ChunkedNLLTrainableTokensHead,
    soft_patch_target,
    spatial_patch_loss,
)

from .support import _FakeCollator, _FakeTrainableTokensHead, _FakeTrainableTokensWrapper


def test_chunked_nll_head_view_matches_peft_forward_and_gradients() -> None:
    torch.manual_seed(7)
    head = _FakeTrainableTokensHead()
    view = _ChunkedNLLTrainableTokensHead(head)
    labels = torch.tensor([0, 1, 5, 3])

    direct_hidden = torch.randn(4, 3, requires_grad=True)
    direct_loss = F.cross_entropy(head(direct_hidden).float(), labels)
    direct_loss.backward()
    expected_hidden_grad = direct_hidden.grad.detach().clone()
    expected_delta_grad = head.delta.grad.detach().clone()

    head.delta.grad = None
    chunked_hidden = direct_hidden.detach().clone().requires_grad_(True)
    chunked_loss = F.cross_entropy(F.linear(chunked_hidden, view.weight).float(), labels)
    chunked_loss.backward()

    assert torch.allclose(chunked_loss, direct_loss)
    assert torch.allclose(chunked_hidden.grad, expected_hidden_grad)
    assert torch.allclose(head.delta.grad, expected_delta_grad)
    assert head.base_layer.weight.grad is None


def test_chunked_nll_head_view_respects_disabled_adapters() -> None:
    head = _FakeTrainableTokensHead()
    head.disable_adapters = True

    view_weight = _ChunkedNLLTrainableTokensHead(head).weight

    assert view_weight is head.base_layer.weight


def test_chunked_nll_head_view_supports_lora_trainable_tokens_wrapper() -> None:
    wrapper = _FakeTrainableTokensWrapper()
    view = _ChunkedNLLTrainableTokensHead(wrapper)

    assert torch.equal(view.weight, wrapper.weight)
    assert view.bias is None


def test_spatial_target_collator_selects_correct_or_control_bbox() -> None:
    setup = TokenSetup(
        tokens=("<N00>",),
        token_ids=(17,),
        tokenizer_size=18,
        model_vocab_size=128,
        added_tokens=1,
    )
    target = {
        "token": "<N00>",
        "entity_type": "node",
        "bbox": [0.1, 0.2, 0.3, 0.4],
        "center": [0.2, 0.3],
        "control_bbox": [0.5, 0.6, 0.7, 0.8],
    }
    examples = [{"spatial_targets": [target]}, {"spatial_targets": []}]

    correct = SpatialTargetCollator(_FakeCollator(), setup, target_mode="correct")(examples)
    shuffled = SpatialTargetCollator(_FakeCollator(), setup, target_mode="shuffled")(examples)

    assert correct["spatial_target_token_ids"].tolist() == [17, 0]
    assert correct["spatial_target_mask"].tolist() == [True, False]
    assert correct["spatial_target_bboxes"][0].tolist() == pytest.approx(target["bbox"])
    assert shuffled["spatial_target_bboxes"][0].tolist() == pytest.approx(target["control_bbox"])


def test_soft_patch_objective_localizes_matching_post_merger_feature() -> None:
    pooled = torch.eye(4)
    grids = torch.tensor([[1, 4, 4]])
    token_embeddings = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    bboxes = torch.tensor([[0.0, 0.0, 0.49, 0.49]])
    mask = torch.tensor([True])

    target, xs, ys = soft_patch_target(bboxes[0], grid_height=2, grid_width=2)
    loss, accuracy, count = spatial_patch_loss(
        pooled,
        grids,
        token_embeddings,
        bboxes,
        mask,
        merge_size=2,
        temperature=0.07,
    )

    assert target.shape == xs.shape == ys.shape == (4,)
    assert target.sum().item() == pytest.approx(1.0)
    assert torch.isfinite(loss)
    assert accuracy.item() == 1.0
    assert count == 1
