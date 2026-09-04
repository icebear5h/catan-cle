# Inspect evaluation logs

Native Inspect `.eval` logs generated from retained Catan run artifacts live
here locally. They are disposable views, not new model runs or authoritative
score stores, and are ignored by Git because embedded vision logs are large.

Generate the current archive bundle without provider calls:

```bash
uv run --extra eval python scripts/import_catan_inspect_archives.py --replace
```

Validate the import plan without writing logs:

```bash
uv run --extra eval python scripts/import_catan_inspect_archives.py --dry-run
```

Open the recursive log directory in Inspect View:

```bash
uv run --extra eval python -m inspect_ai._cli.main view start \
  --recursive \
  --log-dir artifacts/runs/inspect/catan_archive_v1
```

Render the matched-vision summary with Inspect Viz:

```bash
uv run --extra eval python scripts/render_catan_inspect_viz.py
open artifacts/runs/inspect/catan_archive_v1/strict_vision_comparison.html
```

The module-form command is intentional: this checkout's relocated `.venv` may
contain stale generated `inspect`/`bench` shebangs. A fresh environment can use
`inspect view start` directly.

The importer currently creates:

- Four matched strict raw-image logs: DeepSeek V4 Flash Vision Exp, Gemma 4
  31B, GLM-4.6V, and Qwen3.8 Max.
- One original-provider Qwen3.8-27B replay-policy log. Later setup overrides and
  rationale repairs are excluded unless requested in code, so the log retains
  the published 50/111 and 28/89 descriptive human-action agreement results.

Each archive log carries exact requested-model identity, retained runtime fields
(with unavailable provider fields left null), source paths, hashes, original
usage and latency, benchmark contract, stored scorer result, and recomputed
integrity checks. The archive model API raises if called, making an
accidental provider request a hard failure.

A generated ownership marker makes `--replace` delete only logs written by this
importer and reject symlinked, unowned, or extra `.eval` files.

Use `--no-embed-images` for a lightweight, non-portable log that displays
local image paths instead of image pixels. Embedded images make the four strict-vision logs about
110 MB in total but permit visual review directly in Inspect View.
