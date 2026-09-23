"""Pure Catan contracts used by the pinned ms-swift training integration."""

from __future__ import annotations

from ._base import _COMPONENT_ATTRIBUTE as _COMPONENT_ATTRIBUTE
from ._base import CATAN_TUNER_TYPE as CATAN_TUNER_TYPE
from ._base import COMPONENT_SCHEMA as COMPONENT_SCHEMA
from ._base import MS_SWIFT_VERSION as MS_SWIFT_VERSION
from ._base import OPTIMIZER_COVERAGE_SCHEMA as OPTIMIZER_COVERAGE_SCHEMA
from ._base import PEFT_VERSION as PEFT_VERSION
from ._base import SCOPE_SCHEMA as SCOPE_SCHEMA
from ._base import Any as Any
from ._base import Iterable as Iterable
from ._base import JsonDict as JsonDict
from ._base import ModelComponentPaths as ModelComponentPaths
from ._base import Path as Path
from ._base import SemanticTokenSetup as SemanticTokenSetup
from ._base import asdict as asdict
from ._base import dataclass as dataclass
from ._base import json as json
from ._base import semantic_recognition_token_inventory as semantic_recognition_token_inventory
from ._base import torch as torch
from ._components import _config_uses_tied_embeddings as _config_uses_tied_embeddings
from ._components import _embedding_weights_are_tied as _embedding_weights_are_tied
from ._components import _named_modules_with_aliases as _named_modules_with_aliases
from ._components import _normalize_arch_paths as _normalize_arch_paths
from ._components import _path_is_within as _path_is_within
from ._components import _validate_token_modules as _validate_token_modules
from ._components import attach_model_components as attach_model_components
from ._components import discover_model_components as discover_model_components
from ._components import get_model_components as get_model_components
from ._components import module_name_for_instance as module_name_for_instance
from ._components import resolve_model_module as resolve_model_module
from ._coverage import audit_optimizer_coverage as audit_optimizer_coverage
from ._coverage import defaultdict_group_manifest as defaultdict_group_manifest
from ._coverage import write_json_atomic as write_json_atomic
from ._scope import _matches_any as _matches_any
from ._scope import _module_path_matches as _module_path_matches
from ._scope import _parameter_category as _parameter_category
from ._scope import audit_trainable_scope as audit_trainable_scope
from ._scope import parameter_matches_paths as parameter_matches_paths
from ._tokens import load_semantic_token_inventory as load_semantic_token_inventory
from ._tokens import prepare_peft_trainable_token_targets as prepare_peft_trainable_token_targets
from ._tokens import prepare_semantic_tokens as prepare_semantic_tokens

__all__ = [
    "Any",
    "CATAN_TUNER_TYPE",
    "COMPONENT_SCHEMA",
    "Iterable",
    "JsonDict",
    "MS_SWIFT_VERSION",
    "ModelComponentPaths",
    "OPTIMIZER_COVERAGE_SCHEMA",
    "PEFT_VERSION",
    "Path",
    "SCOPE_SCHEMA",
    "SemanticTokenSetup",
    "asdict",
    "attach_model_components",
    "audit_optimizer_coverage",
    "audit_trainable_scope",
    "dataclass",
    "defaultdict_group_manifest",
    "discover_model_components",
    "get_model_components",
    "json",
    "load_semantic_token_inventory",
    "module_name_for_instance",
    "parameter_matches_paths",
    "prepare_peft_trainable_token_targets",
    "prepare_semantic_tokens",
    "resolve_model_module",
    "semantic_recognition_token_inventory",
    "torch",
    "write_json_atomic",
]
