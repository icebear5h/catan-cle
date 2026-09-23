"""Head of the tuner page markup, concatenated verbatim in _page."""

HTML_HEAD = """<!doctype html>
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
"""
