"""Small JSON and structural Miles contracts; no Miles/runtime dependency."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from enum import Enum
from typing import Protocol, TypeAlias, TypedDict, cast

from sft.board.board_fluency_scoring import SCHEMAS as FLUENCY_SCHEMAS
from sft.board.board_fluency_scoring import _strip_transport, score_board_fluency
from sft.board.coordinate_comparison import SCHEMA as COORDINATE_SCHEMA
from sft.board.coordinate_comparison import score_coordinate_comparison
from sft.board.symbolic_board_tasks import SYMBOLIC_TASKS, score_symbolic_task
from sft.cartesian_eval import SCHEMA as CARTESIAN_SCHEMA
from sft.cartesian_eval import score_response as score_cartesian
from sft.cartesian_eval.scaling import SCHEMA as SCALING_SCHEMA
from sft.cartesian_eval.scaling import score_response as score_scaling
from sft.cartesian_eval.shorthand import SCHEMA as SHORTHAND_SCHEMA
from sft.cartesian_eval.shorthand import score_response as score_shorthand

Json: TypeAlias = None | bool | int | float | str | list["Json"] | dict[str, "Json"]
JsonObject: TypeAlias = dict[str, Json]
SYMBOLIC_SCHEMAS = frozenset({"catan_symbolic_board_row/v1", "catan_symbolic_board_row/v2"})


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def as_json(value: object) -> Json:
    """Validate and detach actual JSON data, rejecting lossy string fallbacks."""
    if value is None or type(value) in (bool, int, str):
        return cast("None | bool | int | str", value)
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [as_json(item) for item in cast("list[object]", value)]
    if isinstance(value, dict):
        result: JsonObject = {}
        for key, item in cast("dict[object, object]", value).items():
            require(isinstance(key, str), "JSON object keys must be strings")
            result[text(key)] = as_json(item)
        return result
    raise ValueError(f"not finite JSON data: {type(value).__name__}")


def json_object(value: object) -> JsonObject:
    result = as_json(value)
    if not isinstance(result, dict):
        raise ValueError("expected JSON object")
    return result


def json_list(value: object) -> list[Json]:
    result = as_json(value)
    if not isinstance(result, list):
        raise ValueError("expected JSON list")
    return result


def text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("expected text")
    return value


def compact(value: Json) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_object(pairs: list[tuple[str, Json]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw: str) -> JsonObject:
    return json_object(json.loads(raw, object_pairs_hook=_unique_object))


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def score(expected: str, response: str, metadata: JsonObject) -> JsonObject:
    """Use only the existing exact scorers; unknown contracts never fall back."""
    schema = metadata.get("schema")
    require(metadata.get("split") != "train" and metadata.get("task_role") != "train"
            and metadata.get("admitted_for_training") is not True, "training rows are not eval")
    result: object
    if schema == SCALING_SCHEMA:
        result = score_scaling(expected, response, metadata)
    elif schema == SHORTHAND_SCHEMA:
        result = score_shorthand(expected, response, metadata)
    elif schema == CARTESIAN_SCHEMA:
        result = score_cartesian(expected, response, metadata)
    elif schema == COORDINATE_SCHEMA:
        result = score_coordinate_comparison(expected, response, metadata)
    elif isinstance(schema, str) and schema in FLUENCY_SCHEMAS:
        result = score_board_fluency(expected, response, metadata)
    elif (schema is None or isinstance(schema, str) and schema in SYMBOLIC_SCHEMAS):
        task = text(metadata.get("task_type"))
        require(task in SYMBOLIC_TASKS and metadata.get("class") != "board_fluency",
                "unsupported scoring metadata")
        result = score_symbolic_task(expected, _strip_transport(response), metadata)
    else:
        raise ValueError(f"unsupported scoring schema: {schema}")
    report = json_object(result)
    require(type(report.get("correct")) is bool, "scorer did not return boolean correctness")
    return report


def dimensions(metadata: JsonObject) -> dict[str, str]:
    operation = text(metadata.get("operation", metadata.get("task_type")))
    return {"operation": operation,
            "family": text(metadata.get("family", metadata.get("training_family", operation))),
            "representation": text(metadata.get("representation", "atlas"))}


class SampleView(Protocol):
    @property
    def prompt(self) -> str: ...

    @property
    def label(self) -> str: ...

    @property
    def metadata(self) -> JsonObject: ...

    @property
    def response(self) -> str: ...

    @property
    def tokens(self) -> Sequence[int]: ...

    @property
    def response_length(self) -> int: ...

    @property
    def status(self) -> Enum: ...


class DatasetView(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def path(self) -> str: ...


class ArgsView(Protocol):
    @property
    def eval_datasets(self) -> Sequence[DatasetView]: ...

    @property
    def save_debug_rollout_data(self) -> str: ...

    @property
    def hf_checkpoint(self) -> str: ...


class EvalPanel(TypedDict):
    samples: Sequence[SampleView]
    rewards: Sequence[float]
    truncated: Sequence[bool]
