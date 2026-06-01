#!/usr/bin/env python3
"""Render a static HTML report from simulated online inference jsonl logs."""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import statistics
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-jsonl",
        required=True,
        help="Path produced by tools/simulate_online_inference.py --log-jsonl.",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="Output HTML path. Defaults to <log-jsonl>.html.",
    )
    return parser.parse_args()


def load_records(path: pathlib.Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    actions = [r.get("action") or [] for r in records]
    action_dim = max((len(a) for a in actions), default=0)
    queues = [int(r.get("queue_remaining") or 0) for r in records]
    errors = [r.get("error") for r in records if r.get("error")]
    summary = {
        "steps": len(records),
        "episodes": len({r.get("episode") for r in records}),
        "action_dim": action_dim,
        "queue_mean": statistics.fmean(queues) if queues else None,
        "queue_max": max(queues) if queues else None,
        "mae": None,
        "rmse": None,
        "max_abs": None,
    }
    if errors:
        summary["mae"] = statistics.fmean(e["mae"] for e in errors)
        summary["rmse"] = statistics.fmean(e["rmse"] for e in errors)
        summary["max_abs"] = max(e["max_abs"] for e in errors)
    return summary


def render_html(records: list[dict[str, Any]], source_path: pathlib.Path) -> str:
    summary = summarize(records)
    data_json = json.dumps(records, ensure_ascii=False)
    source = html.escape(str(source_path))
    first_prompt = html.escape(str(records[0].get("prompt", ""))) if records else ""
    cards = [
        ("Steps", str(summary["steps"])),
        ("Episodes", str(summary["episodes"])),
        ("Action Dim", str(summary["action_dim"])),
        ("Queue Mean", fmt_float(summary["queue_mean"], 2)),
        ("Queue Max", str(summary["queue_max"] if summary["queue_max"] is not None else "-")),
        ("MAE", fmt_float(summary["mae"])),
        ("RMSE", fmt_float(summary["rmse"])),
        ("Max Abs", fmt_float(summary["max_abs"])),
    ]
    card_html = "\n".join(
        f'<section class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>'
        for label, value in cards
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dexbotic Online Inference Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #20242c;
      --muted: #667085;
      --line: #d8dde6;
      --accent: #006d77;
      --accent-2: #c2410c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      padding: 24px 32px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; font-weight: 700; letter-spacing: 0; }}
    .source {{ color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }}
    main {{ padding: 24px 32px 36px; display: grid; gap: 20px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      min-height: 72px;
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 8px; font-size: 22px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      overflow: hidden;
    }}
    .panel h2 {{ margin: 0 0 12px; font-size: 16px; }}
    .prompt {{ color: var(--muted); font-size: 13px; line-height: 1.5; }}
    canvas {{ width: 100%; height: 260px; display: block; }}
    .controls {{ display: flex; gap: 10px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }}
    select {{ padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px; background: white; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; }}
    td.action {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    @media (max-width: 720px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      canvas {{ height: 220px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Dexbotic Online Inference Report</h1>
    <div class="source">{source}</div>
  </header>
  <main>
    <section class="metrics">{card_html}</section>
    <section class="panel">
      <h2>Prompt</h2>
      <div class="prompt">{first_prompt}</div>
    </section>
    <section class="panel">
      <h2>Queue</h2>
      <canvas id="queueChart"></canvas>
    </section>
    <section class="panel">
      <div class="controls">
        <h2 style="margin:0">Action Dimension</h2>
        <select id="dimSelect"></select>
      </div>
      <canvas id="actionChart"></canvas>
    </section>
    <section class="panel">
      <h2>Frames</h2>
      <table>
        <thead><tr><th>Step</th><th>Frame</th><th>Queue</th><th>Action Head</th><th>Error</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </section>
  </main>
  <script>
    const records = {data_json};
    const css = getComputedStyle(document.documentElement);
    const accent = css.getPropertyValue('--accent').trim();
    const accent2 = css.getPropertyValue('--accent-2').trim();
    const muted = css.getPropertyValue('--muted').trim();
    const line = css.getPropertyValue('--line').trim();

    function setupCanvas(canvas) {{
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return {{ ctx, w: rect.width, h: rect.height }};
    }}

    function drawLine(canvas, values, color) {{
      const {{ ctx, w, h }} = setupCanvas(canvas);
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = line;
      ctx.lineWidth = 1;
      for (let i = 0; i < 5; i++) {{
        const y = 20 + i * (h - 44) / 4;
        ctx.beginPath();
        ctx.moveTo(42, y);
        ctx.lineTo(w - 12, y);
        ctx.stroke();
      }}
      if (!values.length) return;
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = Math.max(1e-9, max - min);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((value, index) => {{
        const x = 42 + index * (w - 58) / Math.max(1, values.length - 1);
        const y = h - 24 - ((value - min) / span) * (h - 48);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }});
      ctx.stroke();
      ctx.fillStyle = muted;
      ctx.font = '12px ui-sans-serif, system-ui';
      ctx.fillText(max.toFixed(4), 6, 24);
      ctx.fillText(min.toFixed(4), 6, h - 24);
    }}

    function populateRows() {{
      const tbody = document.getElementById('rows');
      tbody.innerHTML = records.slice(0, 500).map((r, i) => {{
        const action = (r.action || []).slice(0, 6).map(v => Number(v).toFixed(4)).join(', ');
        const err = r.error ? `mae=${{Number(r.error.mae).toFixed(6)}}` : '-';
        return `<tr><td>${{i}}</td><td>${{r.frame_index}}</td><td>${{r.queue_remaining}}</td><td class="action">${{action}}</td><td>${{err}}</td></tr>`;
      }}).join('');
    }}

    function initActionSelect() {{
      const select = document.getElementById('dimSelect');
      const dim = Math.max(0, ...records.map(r => (r.action || []).length));
      select.innerHTML = Array.from({{ length: dim }}, (_, i) => `<option value="${{i}}">dim ${{i}}</option>`).join('');
      select.addEventListener('change', redraw);
    }}

    function redraw() {{
      drawLine(document.getElementById('queueChart'), records.map(r => Number(r.queue_remaining || 0)), accent);
      const dim = Number(document.getElementById('dimSelect').value || 0);
      drawLine(document.getElementById('actionChart'), records.map(r => Number((r.action || [])[dim] || 0)), accent2);
    }}

    initActionSelect();
    populateRows();
    redraw();
    window.addEventListener('resize', redraw);
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    log_path = pathlib.Path(args.log_jsonl)
    output_path = pathlib.Path(args.output_html or f"{args.log_jsonl}.html")
    records = load_records(log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(records, log_path), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
