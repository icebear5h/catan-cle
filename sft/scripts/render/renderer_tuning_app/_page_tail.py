"""Tail of the tuner page markup, concatenated verbatim in _page."""

HTML_TAIL = """  </header>

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
