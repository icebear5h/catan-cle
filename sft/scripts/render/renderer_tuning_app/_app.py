"""Local tuning UI for the Python Catan board renderer.

Run:

    uv run python -m sft.scripts.render.renderer_tuning_app       --dataset-dir artifacts/generated/sft/node_factors       --port 8765
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from flask import Flask, Response, abort, request, send_file

from evals.catan_board_bench.render import DEFAULT_RENDER_STYLE, RenderStyle, render_contract_image
from sft.json_types import JsonDict, load_json_dict
from sft.paths import (
    GENERATED_SFT_ROOT,
    PROJECT_ROOT,
)

from ._base import FIXTURE_CONTRACT_DIR, STYLE_CONFIG_PATH
from ._page import HTML
from ._sliders import SLIDERS


def create_app(dataset_dir: Path) -> Flask:
    contracts_dir = dataset_dir / "contracts"
    contract_paths = [
        *sorted(FIXTURE_CONTRACT_DIR.glob("*.json")),
        *sorted(contracts_dir.glob("*.json")),
    ]
    if not contract_paths:
        raise FileNotFoundError(
            f"no contract JSON files found under {FIXTURE_CONTRACT_DIR} or {contracts_dir}"
        )

    sample_paths = {path.stem: path for path in contract_paths}
    samples = sorted(sample_paths)
    default_sample = "colonist_dummy_setup"
    if default_sample not in sample_paths:
        default_sample = "node_factor_n00_red_settlement_v00"
    if default_sample not in sample_paths:
        default_sample = samples[0]

    @lru_cache(maxsize=1024)
    def load_contract(sample_id: str) -> JsonDict:
        return load_json_dict(sample_paths[sample_id])

    app = Flask(__name__)

    @app.get("/")
    def index() -> Response:
        body = (
            HTML.replace("__SAMPLES_JSON__", json.dumps(samples))
            .replace("__DEFAULT_SAMPLE__", default_sample)
            .replace("__SLIDERS_JSON__", json.dumps(SLIDERS))
        )
        return Response(body, mimetype="text/html")

    @app.get("/render.png")
    def render_png() -> Response:
        sample_id = request.args.get("sample", default_sample)
        if sample_id not in sample_paths:
            abort(404, f"unknown sample {sample_id}")

        style = _style_from_request()
        image_size = int(round(_get_float("image_size", 512.0, 256.0, 1024.0)))
        image = render_contract_image(load_contract(sample_id), image_size=image_size, style=style)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png", max_age=0)

    @app.post("/save_style")
    def save_style() -> JsonDict:
        style = _style_from_request()
        payload: JsonDict = {
            "schema": "catan_python_renderer_style/v0",
            "source": "renderer_tuning_app",
            "style": asdict(style),
        }
        STYLE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        STYLE_CONFIG_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return {
            "ok": True,
            "path": str(STYLE_CONFIG_PATH.relative_to(PROJECT_ROOT)),
            "style": payload["style"],
        }

    return app


def _style_from_request() -> RenderStyle:
    return RenderStyle(
        road_width_factor=_get_float(
            "road_width_factor", DEFAULT_RENDER_STYLE.road_width_factor, 0.35, 1.40
        ),
        road_length_factor=_get_float(
            "road_length_factor", DEFAULT_RENDER_STYLE.road_length_factor, 0.60, 1.50
        ),
        dock_width=_get_float("dock_width", DEFAULT_RENDER_STYLE.dock_width, 1.0, 20.0),
        dock_extend_factor=_get_float(
            "dock_extend_factor", DEFAULT_RENDER_STYLE.dock_extend_factor, -0.45, 0.75
        ),
        dock_ship_clearance_factor=_get_float(
            "dock_ship_clearance_factor",
            DEFAULT_RENDER_STYLE.dock_ship_clearance_factor,
            0.0,
            0.50,
        ),
        dock_port_gap=_get_float("dock_port_gap", DEFAULT_RENDER_STYLE.dock_port_gap, 0.0, 40.0),
        dock_sand_gap=_get_float_with_alias(
            "dock_sand_gap",
            "dock_node_gap",
            DEFAULT_RENDER_STYLE.dock_sand_gap,
            0.0,
            40.0,
        ),
        coast_size_factor=_get_float(
            "coast_size_factor",
            DEFAULT_RENDER_STYLE.coast_size_factor,
            0.90,
            1.25,
        ),
        view_padding_factor=_get_float(
            "view_padding_factor",
            DEFAULT_RENDER_STYLE.view_padding_factor,
            0.50,
            3.00,
        ),
        robber_size_factor=_get_float(
            "robber_size_factor",
            DEFAULT_RENDER_STYLE.robber_size_factor,
            0.30,
            1.50,
        ),
        settlement_size_factor=_get_float(
            "settlement_size_factor",
            DEFAULT_RENDER_STYLE.settlement_size_factor,
            0.30,
            1.20,
        ),
        settlement_y_factor=_get_float(
            "settlement_y_factor",
            DEFAULT_RENDER_STYLE.settlement_y_factor,
            0.20,
            1.20,
        ),
        settlement_x_offset=_get_float(
            "settlement_x_offset",
            DEFAULT_RENDER_STYLE.settlement_x_offset,
            -40.0,
            40.0,
        ),
        city_size_factor=_get_float(
            "city_size_factor",
            DEFAULT_RENDER_STYLE.city_size_factor,
            0.30,
            1.30,
        ),
        city_width_factor=_get_float(
            "city_width_factor",
            DEFAULT_RENDER_STYLE.city_width_factor,
            0.50,
            1.80,
        ),
        city_x_offset=_get_float(
            "city_x_offset",
            DEFAULT_RENDER_STYLE.city_x_offset,
            -40.0,
            40.0,
        ),
        city_y_factor=_get_float(
            "city_y_factor",
            DEFAULT_RENDER_STYLE.city_y_factor,
            0.20,
            1.20,
        ),
    )


def _get_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = request.args.get(name)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            abort(400, f"{name} must be a number")
    return max(minimum, min(maximum, value))


def _get_float_with_alias(
    name: str,
    alias: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = request.args.get(name)
    if raw is None:
        raw = request.args.get(alias)
    if raw is None:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            abort(400, f"{name} must be a number")
    return max(minimum, min(maximum, value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Catan renderer tuning UI.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=GENERATED_SFT_ROOT / "node_factors",
        help="Dataset folder containing contracts/*.json.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    app = create_app(args.dataset_dir)
    app.run(host=args.host, port=args.port, debug=False)
