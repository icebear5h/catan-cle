"""Unambiguous node and edge descriptions, including visible pieces."""

from __future__ import annotations

from data_pipeline.board_recognition import inverse_grounding as api
from data_pipeline.board_recognition.replay_dataset import JsonDict
from data_pipeline.json_coerce import as_str


def _node_description(
    node: JsonDict,
    signatures: dict[str, JsonDict],
    natural: dict[str, str],
    *,
    state_index: int,
) -> tuple[str, str, str | None]:
    token = as_str(node["token"])
    signature = signatures[token]
    choices: list[tuple[str, str]] = []
    if signature["full_unique"]:
        choices.extend(
            [
                ("node_compact_resource_number", as_str(signature["full"])),
                ("node_words_resource_number", as_str(signature["full_words"])),
            ]
        )
    if signature["resources_unique"]:
        choices.extend(
            [
                ("node_compact_resources", as_str(signature["resources"])),
                ("node_words_resources", as_str(signature["resource_words"])),
            ]
        )
    if token in natural:
        choices.append(("node_directional_anchor", natural[token]))
    if not choices:
        raise api.InverseGroundingError(f"node has no unambiguous description: {token}")
    style, base = choices[state_index % len(choices)]
    building = node.get("building")
    color = node.get("color")
    qualifier = None
    if building is not None or color is not None:
        if not isinstance(building, str) or not isinstance(color, str):
            raise api.InverseGroundingError(f"node has partial building state: {token}")
        qualifier = f"{api._words(color)} {api._words(building)}"
        base = f"the {qualifier} at {base}"
        style = f"piece_{style}"
    return base, style, qualifier


def _edge_description(
    edge: JsonDict,
    signatures: dict[str, JsonDict],
    natural: dict[str, str],
    *,
    state_index: int,
) -> tuple[str, str, str | None]:
    token = as_str(edge["token"])
    endpoint_tokens = edge.get("node_tokens")
    if not isinstance(endpoint_tokens, list) or len(endpoint_tokens) != 2:
        raise api.InverseGroundingError(f"edge has invalid node tokens: {token}")
    left, right = (signatures[as_str(endpoint)] for endpoint in endpoint_tokens)
    choices: list[tuple[str, str]] = []
    if left["full_unique"] and right["full_unique"]:
        choices.extend(
            [
                ("edge_compact_endpoints", f"the edge between {left['full']} and {right['full']}"),
                (
                    "edge_words_endpoints",
                    f"the edge between {left['full_words']} and {right['full_words']}",
                ),
            ]
        )
    if token in natural:
        choices.append(("edge_directional_anchor", natural[token]))
    if not choices:
        raise api.InverseGroundingError(f"edge has no unambiguous description: {token}")
    style, base = choices[state_index % len(choices)]
    color = edge.get("road_color")
    qualifier = None
    if color is not None:
        if not isinstance(color, str):
            raise api.InverseGroundingError(f"edge has invalid road color: {token}")
        qualifier = f"{api._words(color)} road"
        if " between " in base:
            base = base.replace("the edge between ", f"the {qualifier} between ", 1)
        else:
            base = f"the {qualifier} on {base.removeprefix('the ')}"
        style = f"piece_{style}"
    return base, style, qualifier
