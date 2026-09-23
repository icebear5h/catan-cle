"""Offline contracts stay importable without Miles or Megatron installed.

Runtime entry points live in ``rollout`` and ``hooks``; importing those modules
deliberately requires the real pinned training dependencies.
"""

from .contracts import MILES_COMMIT, REQUIRED_TARGETS

__all__ = ["MILES_COMMIT", "REQUIRED_TARGETS"]
