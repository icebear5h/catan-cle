"""Strict YAML parsing shared by every authored prompt suite format."""

from __future__ import annotations

from pathlib import Path

import yaml

BUILTIN_SUITES_DIR = Path(__file__).resolve().parent / "suites"


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe loader that rejects duplicate and non-scalar mapping keys."""

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[object, object]:
        self.flatten_mapping(node)
        result: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in result:
                    raise ValueError(f"Duplicate YAML key: {key}")
                result[key] = self.construct_object(value_node, deep=deep)
            except TypeError as exc:
                raise ValueError("YAML mapping keys must be scalar values") from exc
        return result


def load_yaml_mapping(text: str, *, source: str) -> dict[object, object]:
    """Parse one top-level mapping; ``source`` (lowercase label) names it in errors."""
    try:
        data = yaml.load(text, Loader=UniqueKeyLoader)
    except (yaml.YAMLError, RecursionError) as exc:
        raise ValueError(f"Invalid {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{source[:1].upper()}{source[1:]} must contain a YAML mapping")
    return data
