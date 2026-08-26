"""Local tuning UI for the Python Catan board renderer.

Run:

    uv run python -m sft.scripts.renderer_tuning_app \
      --dataset-dir artifacts/generated/sft/node_factors \
      --port 8765
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from catan_board_bench.render import DEFAULT_RENDER_STYLE, RenderStyle, render_contract_image
from flask import Flask, Response, abort, request, send_file

from sft.paths import (
    GENERATED_SFT_ROOT,
    PROJECT_ROOT,
    RENDER_CONTRACT_FIXTURE_DIR,
    RENDERER_STYLE_CONFIG,
)


STYLE_CONFIG_PATH = RENDERER_STYLE_CONFIG
FIXTURE_CONTRACT_DIR = RENDER_CONTRACT_FIXTURE_DIR


SLIDERS = [
    {
        "key": "road_width_factor",
        "label": "Road width",
        "min": 0.45,
        "max": 1.15,
        "step": 0.01,
        "default": DEFAULT_RENDER_STYLE.road_width_factor,
    },
    {
        "key": "road_length_factor",
        "label": "Road length",
        "min": 0.80,
        "max": 1.25,
        "step": 0.01,
        "default": DEFAULT_RENDER_STYLE.road_length_factor,
    },
    {
        "key": "dock_width",
        "label": "Dock width",
        "min": 2.0,
        "max": 14.0,
        "step": 0.25,
        "default": DEFAULT_RENDER_STYLE.dock_width,
    },
    {
        "key": "dock_extend_factor",
        "label": "Dock length extra",
        "min": -0.45,
        "max": 0.40,
        "step": 0.01,
        "default": DEFAULT_RENDER_STYLE.dock_extend_factor,
    },
    {
        "key": "dock_sand_gap",
        "label": "Dock sand gap",
        "min": 0.0,
        "max": 16.0,
        "step": 0.25,
        "default": DEFAULT_RENDER_STYLE.dock_sand_gap,
    },
    {
        "key": "dock_port_gap",
        "label": "Dock ship gap",
        "min": 0.0,
        "max": 24.0,
        "step": 0.5,
        "default": DEFAULT_RENDER_STYLE.dock_port_gap,
    },
    {
        "key": "dock_ship_clearance_factor",
        "label": "Dock ship clear",
        "min": 0.0,
        "max": 0.35,
        "step": 0.01,
        "default": DEFAULT_RENDER_STYLE.dock_ship_clearance_factor,
    },
    {
        "key": "coast_size_factor",
        "label": "Coast overlap",
        "min": 0.96,
        "max": 1.12,
        "step": 0.005,
        "default": DEFAULT_RENDER_STYLE.coast_size_factor,
    },
    {
        "key": "view_padding_factor",
        "label": "View padding",
        "min": 0.70,
        "max": 2.50,
        "step": 0.05,
        "default": DEFAULT_RENDER_STYLE.view_padding_factor,
    },
    {
        "key": "robber_size_factor",
        "label": "Robber size",
        "min": 0.45,
        "max": 1.20,
        "step": 0.01,
        "default": DEFAULT_RENDER_STYLE.robber_size_factor,
    },
    {
        "key": "settlement_size_factor",
        "label": "Settlement size",
        "min": 0.45,
        "max": 0.90,
        "step": 0.01,
        "default": DEFAULT_RENDER_STYLE.settlement_size_factor,
    },
    {
        "key": "settlement_y_factor",
        "label": "Settlement y offset",
        "min": 0.45,
        "max": 0.95,
        "step": 0.01,
        "default": DEFAULT_RENDER_STYLE.settlement_y_factor,
    },
    {
        "key": "settlement_x_offset",
        "label": "Settlement x offset",
        "min": -20.0,
        "max": 20.0,
        "step": 0.5,
        "default": DEFAULT_RENDER_STYLE.settlement_x_offset,
    },
    {
        "key": "city_size_factor",
        "label": "City size",
        "min": 0.55,
        "max": 1.10,
        "step": 0.01,
        "default": DEFAULT_RENDER_STYLE.city_size_factor,
    },
    {
        "key": "city_width_factor",
        "label": "City width",
        "min": 0.80,
        "max": 1.45,
        "step": 0.01,
        "default": DEFAULT_RENDER_STYLE.city_width_factor,
    },
    {
        "key": "city_y_factor",
        "label": "City y offset",
        "min": 0.45,
        "max": 0.95,
        "step": 0.01,
        "default": DEFAULT_RENDER_STYLE.city_y_factor,
    },
    {
        "key": "city_x_offset",
        "label": "City x offset",
        "min": -20.0,
        "max": 20.0,
        "step": 0.5,
        "default": DEFAULT_RENDER_STYLE.city_x_offset,
    },
]

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Catan Renderer Tuner</title>
  <style>
    :root {
      --bg: #05080b;
      --panel: #0a0f14;
      --line: #16303a;
      --line-strong: #19f6cf;
      --text: #dfeaf2;
      --muted: #8092a6;
      --accent: #28ffd5;
      --warn: #f3ca4d;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        linear-gradient(rgba(25, 246, 207, 0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(25, 246, 207, 0.08) 1px, transparent 1px),
        radial-gradient(circle at 65% -10%, rgba(243, 202, 77, 0.11), transparent 32rem),
        var(--bg);
      background-size: 24px 24px, 24px 24px, auto, auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      letter-spacing: 0;
    }

    header {
      height: 48px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid var(--line-strong);
      background: rgba(5, 8, 11, 0.92);
    }

    h1 {
      margin: 0;
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .tag {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }

    main {
      display: grid;
      grid-template-columns: minmax(360px, 0.95fr) minmax(340px, 0.65fr);
      gap: 18px;
      padding: 18px;
      max-width: 1180px;
      margin: 0 auto;
    }

    .preview,
    .controls {
      border: 1px solid var(--line);
      background: rgba(6, 11, 16, 0.90);
    }

    .preview {
      display: grid;
      place-items: center;
      min-height: calc(100vh - 84px);
      padding: 16px;
    }

    #board {
      width: min(100%, 720px);
      aspect-ratio: 1;
      object-fit: contain;
      border: 1px solid #245161;
      background: #0967a5;
    }

    .controls {
      padding: 16px;
      align-self: start;
    }

    label {
      display: block;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
    }

    select,
    button {
      width: 100%;
      min-height: 38px;
      border: 1px solid var(--line);
      color: var(--text);
      background: #070c11;
      font: inherit;
      font-size: 12px;
    }

    button {
      cursor: pointer;
      border-color: #245161;
    }

    button:hover { border-color: var(--accent); color: var(--accent); }

    .group { margin-bottom: 18px; }

    .slider-row {
      display: grid;
      grid-template-columns: 1fr 64px;
      gap: 12px;
      align-items: center;
      margin-bottom: 18px;
    }

    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }

    .value {
      color: var(--warn);
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-size: 12px;
    }

    pre {
      overflow: auto;
      margin: 0;
      padding: 12px;
      border: 1px solid var(--line);
      background: #05080b;
      color: var(--accent);
      font-size: 12px;
      line-height: 1.55;
      white-space: pre-wrap;
    }

    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 10px;
      margin-top: 10px;
    }

    .status {
      min-height: 18px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 11px;
    }

    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .preview { min-height: auto; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Catan Renderer Tuner</h1>
    <div class="tag">Python render path</div>
  </header>

  <main>
    <section class="preview">
      <img id="board" alt="Rendered Catan board preview">
    </section>

    <section class="controls">
      <div class="group">
        <label for="sample">Sample contract</label>
        <select id="sample"></select>
      </div>

      <div id="sliders"></div>

      <div class="group">
        <label for="image_size">Image size</label>
        <div class="slider-row">
          <input id="image_size" type="range" min="384" max="768" step="64" value="512">
          <div class="value" data-value-for="image_size">512</div>
        </div>
      </div>

      <div class="group">
        <label>Copy these values into render.py when they look right</label>
        <pre id="code"></pre>
        <div class="actions">
          <button id="copy" type="button">Copy</button>
          <button id="save" type="button">Save Style</button>
          <button id="reset" type="button">Reset</button>
        </div>
        <div id="save_status" class="status"></div>
      </div>
    </section>
  </main>

  <script>
    const samples = __SAMPLES_JSON__;
    const defaultSample = "__DEFAULT_SAMPLE__";
    const sliders = __SLIDERS_JSON__;
    const sampleSelect = document.getElementById("sample");
    const sliderMount = document.getElementById("sliders");
    const board = document.getElementById("board");
    const code = document.getElementById("code");

    for (const sample of samples) {
      const option = document.createElement("option");
      option.value = sample;
      option.textContent = sample;
      option.selected = sample === defaultSample;
      sampleSelect.appendChild(option);
    }

    for (const slider of sliders) {
      const group = document.createElement("div");
      group.className = "group";
      group.innerHTML = `
        <label for="${slider.key}">${slider.label}</label>
        <div class="slider-row">
          <input id="${slider.key}" type="range"
            min="${slider.min}" max="${slider.max}" step="${slider.step}" value="${slider.default}">
          <div class="value" data-value-for="${slider.key}">${slider.default}</div>
        </div>
      `;
      sliderMount.appendChild(group);
    }

    const controls = [
      sampleSelect,
      document.getElementById("image_size"),
      ...sliders.map((slider) => document.getElementById(slider.key)),
    ];

    function readValues() {
      const values = { sample: sampleSelect.value, image_size: document.getElementById("image_size").value };
      for (const slider of sliders) {
        values[slider.key] = document.getElementById(slider.key).value;
      }
      return values;
    }

    function refreshValueLabels(values) {
      for (const [key, value] of Object.entries(values)) {
        const label = document.querySelector(`[data-value-for="${key}"]`);
        if (label) label.textContent = value;
      }
    }

    function renderStyleSnippet(values) {
      return `RenderStyle(
    road_width_factor=${Number(values.road_width_factor).toFixed(2)},
    road_length_factor=${Number(values.road_length_factor).toFixed(2)},
    dock_width=${Number(values.dock_width).toFixed(2)},
    dock_extend_factor=${Number(values.dock_extend_factor).toFixed(2)},
    dock_ship_clearance_factor=${Number(values.dock_ship_clearance_factor).toFixed(2)},
    dock_port_gap=${Number(values.dock_port_gap).toFixed(2)},
    dock_sand_gap=${Number(values.dock_sand_gap).toFixed(2)},
    coast_size_factor=${Number(values.coast_size_factor).toFixed(3)},
    view_padding_factor=${Number(values.view_padding_factor).toFixed(2)},
    robber_size_factor=${Number(values.robber_size_factor).toFixed(2)},
    settlement_size_factor=${Number(values.settlement_size_factor).toFixed(2)},
    settlement_x_offset=${Number(values.settlement_x_offset).toFixed(2)},
    settlement_y_factor=${Number(values.settlement_y_factor).toFixed(2)},
    city_size_factor=${Number(values.city_size_factor).toFixed(2)},
    city_width_factor=${Number(values.city_width_factor).toFixed(2)},
    city_x_offset=${Number(values.city_x_offset).toFixed(2)},
    city_y_factor=${Number(values.city_y_factor).toFixed(2)},
)`;
    }

    let timer = 0;
    function scheduleUpdate() {
      clearTimeout(timer);
      timer = setTimeout(update, 90);
    }

    function update() {
      const values = readValues();
      refreshValueLabels(values);
      code.textContent = renderStyleSnippet(values);
      const params = new URLSearchParams(values);
      params.set("cache_bust", Date.now().toString());
      board.src = `/render.png?${params.toString()}`;
    }

    for (const control of controls) control.addEventListener("input", scheduleUpdate);

    document.getElementById("reset").addEventListener("click", () => {
      sampleSelect.value = defaultSample;
      document.getElementById("image_size").value = "512";
      for (const slider of sliders) document.getElementById(slider.key).value = slider.default;
      update();
    });

    document.getElementById("copy").addEventListener("click", async () => {
      await navigator.clipboard.writeText(code.textContent);
    });

    document.getElementById("save").addEventListener("click", async () => {
      const status = document.getElementById("save_status");
      const params = new URLSearchParams(readValues());
      const response = await fetch(`/save_style?${params.toString()}`, { method: "POST" });
      const result = await response.json();
      status.textContent = response.ok ? `Saved ${result.path}` : `Save failed: ${result.error || response.status}`;
    });

    update();
  </script>
</body>
</html>
"""


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
    def load_contract(sample_id: str) -> dict:
        with sample_paths[sample_id].open() as file:
            return json.load(file)

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
    def render_png():
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
    def save_style():
        style = _style_from_request()
        payload = {
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


if __name__ == "__main__":
    main()
