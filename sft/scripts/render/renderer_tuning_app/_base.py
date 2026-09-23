"""Style-config and fixture locations for the renderer tuner."""

from __future__ import annotations

import argparse as argparse
import io as io
import json as json
from dataclasses import asdict as asdict
from functools import lru_cache as lru_cache
from pathlib import Path as Path

from flask import Flask as Flask
from flask import Response as Response
from flask import abort as abort
from flask import request as request
from flask import send_file as send_file

from evals.catan_board_bench.render import DEFAULT_RENDER_STYLE as DEFAULT_RENDER_STYLE
from evals.catan_board_bench.render import RenderStyle as RenderStyle
from evals.catan_board_bench.render import render_contract_image as render_contract_image
from sft.paths import GENERATED_SFT_ROOT as GENERATED_SFT_ROOT
from sft.paths import PROJECT_ROOT as PROJECT_ROOT
from sft.paths import RENDER_CONTRACT_FIXTURE_DIR as RENDER_CONTRACT_FIXTURE_DIR
from sft.paths import RENDERER_STYLE_CONFIG as RENDERER_STYLE_CONFIG

STYLE_CONFIG_PATH = RENDERER_STYLE_CONFIG

FIXTURE_CONTRACT_DIR = RENDER_CONTRACT_FIXTURE_DIR
