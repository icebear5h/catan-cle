"""Shared helpers for spatial task graph, scoring, and dispatch contracts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


ZERO = {"wood": 0, "brick": 0, "sheep": 0, "wheat": 0, "ore": 0}


PATH = ["<N00>", "<N01>", "<N02>", "<N03>"]


PATH_METADATA = {"task_type": "shortest_node_path", "target": {"start": PATH[0], "end": PATH[-1]}}


TILE_METADATA = {"task_type": "node_tiles", "target": {"node": "<N00>"}}


LOCAL_METADATA = {"task_type": "local_node_tiles", "target": {"node": "<N00>"}}


DICE_METADATA = {"task_type": "dice_production", "target": {"color": "RED", "roll": 8}}
