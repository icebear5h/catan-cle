"""Linear targets and semantic-token preparation."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Sequence

import torch

from sft.json_types import as_list, as_str, json_dict
from sft.scripts.train.train_trl_catan_vision._common import (
    ANSWER_METRIC_TRIVIAL_TOKENS,
    FAMILY_WORDS,
    SEMANTIC_ROW_NOISE_SCALE,
    TOKEN_INIT_MODES,
    VISION_LORA_SUFFIXES,
    JsonDict,
)
from sft.scripts.train.train_trl_catan_vision._config import (
    ModelComponents,
    TokenSetup,
    native_tokenizer,
)
from sft.scripts.train.train_trl_catan_vision._structure import (
    _ChunkedNLLTrainableTokensHead,
    discover_components,
    embedding_model,
    module_tensor,
    resolve_wrapped_module,
)

if TYPE_CHECKING:  # Heavy; transformers loads lazily on the runtime path.
    from transformers import PreTrainedTokenizerBase, ProcessorMixin


def prepare_semantic_tokens(
    processor: ProcessorMixin | PreTrainedTokenizerBase,
    model: torch.nn.Module,
    inventory: JsonDict,
) -> tuple[TokenSetup, ModelComponents]:
    tokenizer = native_tokenizer(processor)
    tokens = tuple(as_str(token) for token in as_list(inventory["tokens"]))
    if len(tokens) != 154 or len(set(tokens)) != 154:
        raise ValueError("semantic inventory must contain 154 unique tokens")
    before = tokenizer.get_vocab()
    present = [token for token in tokens if token in before]
    if present and len(present) != len(tokens):
        raise ValueError("tokenizer contains only part of the semantic inventory")
    added = tokenizer.add_tokens(list(tokens), special_tokens=False)
    if added != (0 if present else 154):
        raise ValueError(f"tokenizer added {added} tokens unexpectedly")
    input_embedding = embedding_model(model).get_input_embeddings()
    current_vocab = int(module_tensor(input_embedding, "weight").shape[0])
    requested_vocab = max(len(tokenizer), current_vocab)
    # transformers' `resize_token_embeddings` is not declared on `torch.nn.Module`.
    getattr(model, "resize_token_embeddings")(
        requested_vocab,
        pad_to_multiple_of=128,
        mean_resizing=False,
    )
    components = discover_components(model)
    raw_ids = tokenizer.convert_tokens_to_ids(list(tokens))
    if isinstance(raw_ids, int):
        raise TypeError("convert_tokens_to_ids returned one id for a token list")
    token_ids = tuple(int(value) for value in raw_ids)
    if len(set(token_ids)) != 154 or min(token_ids) < 0 or max(token_ids) >= components.vocab_size:
        raise ValueError("semantic token IDs do not address 154 unique model rows")
    if set(token_ids) & set(getattr(tokenizer, "all_special_ids", [])):
        raise ValueError("semantic atlas tokens must be regular tokens")
    for token, token_id in zip(tokens, token_ids, strict=True):
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f"semantic token is not atomic: {token}")
    family_word_ids = {
        family: tuple(int(value) for value in tokenizer.encode(word, add_special_tokens=False))
        for family, word in FAMILY_WORDS.items()
    }
    if any(not ids or max(ids) >= min(token_ids) for ids in family_word_ids.values()):
        raise ValueError("family words must tokenize to base-vocabulary ids")
    setup = TokenSetup(
        tokens=tokens,
        token_ids=token_ids,
        tokenizer_size=len(tokenizer),
        model_vocab_size=components.vocab_size,
        added_tokens=added,
        family_word_ids=family_word_ids,
    )
    return setup, components


def initialize_semantic_token_rows(
    model: torch.nn.Module,
    components: ModelComponents,
    setup: TokenSetup,
    *,
    seed: int,
    mode: str = "mean_noise",
) -> JsonDict:
    """Seed the 154 atlas rows from the base vocabulary before PEFT copies them.

    Qwen's embedding matrix is already padded past the tokenizer, so
    ``resize_token_embeddings`` never touches the new ids and they would
    otherwise start from the checkpoint's untrained padding rows. Under
    ``mean_noise`` each row becomes the mean of the original vocabulary plus
    small seeded noise so the rows are distinct from the first step. Under
    ``family_words`` each row starts from the base embedding of its family
    word (" node", " edge", " tile", " port") plus the same noise, so nodes,
    edges, tiles, and ports carry a shared per-family direction from the
    first step instead of one undifferentiated atlas direction. Under
    ``vocab_gaussian`` each row is drawn from the base vocabulary's own
    per-dimension mean and standard deviation, so the 154 rows start as far
    apart as random real words rather than as one direction plus a tenth of
    that spread.
    """

    if mode not in TOKEN_INIT_MODES:
        raise ValueError(f"unknown token init mode: {mode}")
    if mode == "keep":
        return {"mode": mode, "note": "rows left as loaded; the frozen bundle already merged its atlas rows"}
    reference_rows = min(setup.token_ids)
    if reference_rows <= 0:
        raise ValueError("semantic token ids must follow the base vocabulary")
    if mode == "family_words" and not setup.family_word_ids:
        raise ValueError("family_words init needs family_word_ids on the token setup")
    ids = torch.tensor(setup.token_ids, dtype=torch.long)
    families = [token[1] for token in setup.tokens]
    generator = torch.Generator().manual_seed(int(seed))
    report: JsonDict = {
        "mode": mode,
        "reference_rows": reference_rows,
        "noise_scale": 1.0 if mode == "vocab_gaussian" else SEMANTIC_ROW_NOISE_SCALE,
    }
    for name, path in (
        ("input_embedding", components.input_embedding),
        ("output_head", components.output_head),
    ):
        weight = module_tensor(resolve_wrapped_module(model, path), "weight")
        if weight.shape[0] <= max(setup.token_ids):
            raise ValueError(f"{path} does not contain the semantic rows")
        ids = ids.to(weight.device)
        with torch.no_grad():
            reference = weight[:reference_rows].float()
            mean = reference.mean(dim=0)
            noise_std = reference.std(dim=0) * (1.0 if mode == "vocab_gaussian" else SEMANTIC_ROW_NOISE_SCALE)
            del reference
            before = weight[ids].float().norm(dim=1)
            noise = torch.randn((len(setup.token_ids), weight.shape[1]), generator=generator)
            if mode == "family_words":
                family_base = {
                    family: weight[torch.tensor(word_ids, device=weight.device)].float().mean(dim=0)
                    for family, word_ids in setup.family_word_ids.items()
                }
                base = torch.stack([family_base[family] for family in families])
            else:
                base = mean.unsqueeze(0).expand(len(setup.token_ids), -1)
            rows = base + noise.to(mean.device) * noise_std.unsqueeze(0)
            weight[ids] = rows.to(weight.dtype)
            after = weight[ids].float().norm(dim=1)
        entry: JsonDict = {
            "mean_row_norm": float(mean.norm()),
            "row_norm_before": float(before.mean()),
            "row_norm_after": float(after.mean()),
            "row_norm_std_after": float(after.std()),
        }
        if mode == "family_words":
            entry["family_base_norms"] = json_dict(
                {family: float(vec.norm()) for family, vec in family_base.items()}
            )
        report[name] = entry
    return report


def promote_visual_master_weights(model: torch.nn.Module, components: ModelComponents) -> JsonDict:
    """Keep fp32 master weights for the full visual path.

    The base model loads in bf16. AdamW steps at the vision and merger rates are
    smaller than half a bf16 ulp for most weights, so leaving those parameters
    in bf16 silently discards nearly every update. Autocast still runs the
    matmuls in bf16; only the stored parameters and optimizer states widen.
    """

    visual = resolve_wrapped_module(model, components.vision)
    visual.float()
    visual.requires_grad_(True)
    dtypes = Counter(str(parameter.dtype) for parameter in visual.parameters())
    if set(dtypes) != {"torch.float32"}:
        raise RuntimeError(f"visual module is not fully fp32 after promotion: {dict(dtypes)}")
    return {"module": components.vision, "dtypes": dict(dtypes), "tensors": sum(dtypes.values())}


def trivial_completion_token_ids(tokenizer: PreTrainedTokenizerBase) -> tuple[int, ...]:
    ids = []
    for text in ANSWER_METRIC_TRIVIAL_TOKENS:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"expected one token id for {text!r}; received {encoded}")
        ids.append(int(encoded[0]))
    return tuple(ids)


def output_head_weight(head: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the functional output weight for a plain or PEFT-wrapped head."""

    if hasattr(head, "token_adapter") or hasattr(head, "get_merged_weights"):
        view = _ChunkedNLLTrainableTokensHead(head)
        return view.weight, view.bias
    bias = getattr(head, "bias", None)
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError("output head bias is not a tensor")
    return module_tensor(head, "weight"), bias


def answer_token_metrics(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    trivial_token_ids: Sequence[int],
) -> dict[str, float]:

    """Teacher-forced accuracy over answer tokens only.

    ``hidden_states`` is the final language hidden state ``[batch, seq, hidden]``
    and ``labels`` the unshifted label tensor with ``-100`` on unsupervised
    positions. Trivial completion tokens are excluded, so the result is the
    accuracy of the content the evaluator will score. ``answer_row_exact`` is
    the fraction of rows whose answer tokens are all argmax-correct.
    """

    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("hidden_states and labels disagree on batch or sequence length")
    shifted = labels[:, 1:]
    hidden = hidden_states[:, :-1]
    mask = shifted != -100
    for token_id in trivial_token_ids:
        mask &= shifted != int(token_id)
    if not bool(mask.any()):
        return {"answer_token_accuracy": 0.0, "answer_row_exact": 0.0, "answer_token_count": 0.0}
    with torch.no_grad():
        selected = hidden[mask].float()
        logits = selected @ weight.float().t()
        if bias is not None:
            logits = logits + bias.float()
        correct = logits.argmax(dim=-1) == shifted[mask]
        rows = torch.arange(labels.shape[0], device=labels.device).unsqueeze(1).expand_as(shifted)[mask]
        row_correct = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
        row_correct[rows[~correct]] = False
        answered = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
        answered[rows] = True
    return {
        "answer_token_accuracy": float(correct.float().mean()),
        "answer_row_exact": float((row_correct & answered).sum() / answered.sum()),
        "answer_token_count": float(mask.sum()),
    }


def language_linear_targets(model: torch.nn.Module, components: ModelComponents) -> list[str]:
    prefix = components.language + ".layers."
    targets = [
        name
        for name, module in model.named_modules()
        if name.startswith(prefix) and isinstance(module, torch.nn.Linear)
    ]
    if not targets:
        raise RuntimeError("no language linear modules were found for LoRA")
    return targets


def vision_linear_targets(model: torch.nn.Module, components: ModelComponents) -> list[str]:
    """Attention, MLP and merger projections of the vision tower, the O-LoRA vision targets."""

    prefix = components.vision + "."
    targets = [
        name
        for name, module in model.named_modules()
        if name.startswith(prefix) and isinstance(module, torch.nn.Linear) and name.endswith(VISION_LORA_SUFFIXES)
    ]
    if not targets:
        raise RuntimeError("no vision linear modules were found for LoRA")
    return targets
