"""Local tuning UI for the Python Catan board renderer.

Run:

    uv run python -m sft.scripts.render.renderer_tuning_app       --dataset-dir artifacts/generated/sft/node_factors       --port 8765
"""

from __future__ import annotations

from ._app import _get_float as _get_float
from ._app import _get_float_with_alias as _get_float_with_alias
from ._app import _style_from_request as _style_from_request
from ._app import create_app as create_app
from ._app import main as main
from ._base import DEFAULT_RENDER_STYLE as DEFAULT_RENDER_STYLE
from ._base import FIXTURE_CONTRACT_DIR as FIXTURE_CONTRACT_DIR
from ._base import GENERATED_SFT_ROOT as GENERATED_SFT_ROOT
from ._base import PROJECT_ROOT as PROJECT_ROOT
from ._base import RENDER_CONTRACT_FIXTURE_DIR as RENDER_CONTRACT_FIXTURE_DIR
from ._base import RENDERER_STYLE_CONFIG as RENDERER_STYLE_CONFIG
from ._base import STYLE_CONFIG_PATH as STYLE_CONFIG_PATH
from ._base import Flask as Flask
from ._base import Path as Path
from ._base import RenderStyle as RenderStyle
from ._base import Response as Response
from ._base import abort as abort
from ._base import argparse as argparse
from ._base import asdict as asdict
from ._base import io as io
from ._base import json as json
from ._base import lru_cache as lru_cache
from ._base import render_contract_image as render_contract_image
from ._base import request as request
from ._base import send_file as send_file
from ._page import HTML as HTML
from ._sliders import SLIDERS as SLIDERS

__all__ = [
    "DEFAULT_RENDER_STYLE",
    "FIXTURE_CONTRACT_DIR",
    "Flask",
    "GENERATED_SFT_ROOT",
    "HTML",
    "PROJECT_ROOT",
    "Path",
    "RENDERER_STYLE_CONFIG",
    "RENDER_CONTRACT_FIXTURE_DIR",
    "RenderStyle",
    "Response",
    "SLIDERS",
    "STYLE_CONFIG_PATH",
    "abort",
    "argparse",
    "asdict",
    "create_app",
    "io",
    "json",
    "lru_cache",
    "main",
    "render_contract_image",
    "request",
    "send_file",
]
