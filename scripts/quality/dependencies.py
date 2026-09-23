"""Compare protected TOML values; stdin is JSON with string ``old`` and ``new``.

This only authorizes nondependency manifest edits. It does not attest that uv
authored any file. Dependency changes belong in uv add/remove/lock/sync commands.
"""

from __future__ import annotations

import json
import sys
import tomllib
from typing import TypeAlias, cast

JSON: TypeAlias = None | bool | int | float | str | list["JSON"] | dict[str, "JSON"]
PROTECTED = (
    "project.dependencies",
    "project.optional-dependencies",
    "project.dynamic",
    "dependency-groups",
    "build-system.requires",
    "tool.uv.sources",
    "tool.uv.override-dependencies",
    "tool.uv.constraint-dependencies",
    "tool.uv.dev-dependencies",
    "tool.uv.exclude-dependencies",
    "tool.uv.build-constraint-dependencies",
    "tool.uv.extra-build-dependencies",
)


def _json_value(value: object) -> JSON:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in cast(list[object], value)]
    if isinstance(value, dict):
        return {
            key: _json_value(item) for key, item in cast(dict[str, object], value).items()
        }
    raise ValueError("protected dependency fields must contain JSON-compatible TOML values")


def protected_values(source: str) -> dict[str, JSON]:
    """Ignore formatting, comments and unrelated tables, but retain missing keys."""
    document = cast(dict[str, object], tomllib.loads(source.removeprefix("\ufeff")))
    result: dict[str, JSON] = {}
    for name in PROTECTED:
        value: object = document
        for part in name.split("."):
            if not isinstance(value, dict):
                raise ValueError(f"parent of {name} must be a TOML table")
            table = cast(dict[str, object], value)
            if part not in table:
                break
            value = table[part]
        else:
            if name == "project.dynamic":
                if not isinstance(value, list):
                    raise ValueError("project.dynamic must be an array of strings")
                entries = cast(list[object], value)
                if any(not isinstance(entry, str) for entry in entries):
                    raise ValueError("project.dynamic must be an array of strings")
                markers = sorted(
                    entry for entry in cast(list[str], value)
                    if entry in {"dependencies", "optional-dependencies"}
                )
                if markers:
                    result[name] = list(markers)
                continue
            result[name] = _json_value(value)
    return result


def check_change(old: str, new: str) -> tuple[str, ...]:
    """Return changed protected paths, raising ValueError on invalid TOML."""
    before, after = protected_values(old), protected_values(new)
    return tuple(
        name
        for name in PROTECTED
        if (name in before) != (name in after)
        or json.dumps(before.get(name), sort_keys=True)
        != json.dumps(after.get(name), sort_keys=True)
    )


def main() -> int:
    try:
        raw = sys.stdin.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("manifest preview exceeds 2,000,000 characters")
        payload = cast(JSON, json.loads(raw))
        if not isinstance(payload, dict) or set(payload) != {"old", "new"}:
            raise ValueError('expected JSON object with exactly "old" and "new"')
        old, new = payload["old"], payload["new"]
        if not isinstance(old, str) or not isinstance(new, str):
            raise ValueError('"old" and "new" must be strings')
        changed = check_change(old, new)
        if changed:
            print("Dependency edit blocked: " + ", ".join(changed), file=sys.stderr)
            print("Use uv add/remove/lock/sync for dependency changes.", file=sys.stderr)
            return 1
    except (ValueError, OSError, RecursionError) as error:
        print(f"Invalid dependency preview: {error}", file=sys.stderr)
        return 2
    print("Protected dependency fields unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
