"""Filenames and rank contracts for a standard Catan LoRA bundle."""

from __future__ import annotations

SUPPORTED_TEXT_LORA_RANKS = (8, 16)
EXPANSION_REPORT_FILE = "lora_expansion.json"
ADAPTER_FILE = "adapter_model.safetensors"
VISUAL_FILE = "visual_model.safetensors"
CONFIG_FILE = "training_config.json"
SCOPE_FILE = "trainable_parameters.json"
INFERENCE_ASSETS = frozenset({
    "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
    "tokenizer.model", "special_tokens_map.json", "added_tokens.json", "vocab.json",
    "merges.txt", "chat_template.jinja", "chat_template.json", "processor_config.json",
    "preprocessor_config.json", "video_preprocessor_config.json",
})
