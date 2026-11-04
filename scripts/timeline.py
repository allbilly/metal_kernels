#!/usr/bin/env python3
"""Render an xctrace export as an encoder/shader HTML timeline.

usage: scripts/timeline.py TRACE_DIR [--out timeline.html] [--process NAME]
                             [--window START_MS:END_MS]
"""

from __future__ import annotations

import argparse
import html
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xctrace_export import Table, base_shader_name, pick_process, short

LANE_H = 16
LANE_GAP = 3
WIDTH = 1600
NAME_W = 132
COLORS = {
    "Compute": "#c9a227",
    "Render": "#4a90d9",
    "Blit": "#7f7f7f",
    "Command buffer": "#5c4d16",
    "Shader": "#b08a1e",
}


def load(path: str, name: str) -> Optional[Table]:
    target = os.path.join(path, name)
    return Table(target) if os.path.isfile(target) else None


def kind_of(event: str) -> str:
    for kind in ("Compute", "Render", "Blit"):
        if kind.lower() in (event or "").lower():
            return kind
    return "Shader"


def block(start: int, dur: int, label: str, kind: str, ident: str = "") -> Dict:
    return {
        "start": start,
        "end": start + max(0, dur),
        "label": label,
        "kind": kind,
        "id": ident,
        "lane": 0,
    }


def parse_window(text: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not text:
        return None, None
    try:
        start, end = text.split(":", 1)
        return int(float(start) * 1e6), int(float(end) * 1e6)
    except ValueError:
        raise SystemExit("--window wants START:END in milliseconds, e.g. 100:150")


def pack(blocks: List[Dict], max_lanes: int = 10) -> int:
    lanes: List[int] = []
    for item in sorted(blocks, key=lambda b: (b["start"], b["end"])):
        placed = False
        for lane, end in enumerate(lanes):
            if item["start"] >= end:
                item["lane"] = lane
                lanes[lane] = item["end"]
                placed = True
                break
        if not placed:
            if len(lanes) < max_lanes:
                item["lane"] = len(lanes)
                lanes.append(item["end"])
            else:
                lane = min(range(len(lanes)), key=lambda i: lanes[i])
                item["lane"] = lane
                lanes[lane] = item["end"]
    return max(1, len(lanes))


def fmt_ms(ns: int) -> str:
    ms = ns / 1e6
    if ms >= 1000:
        return "%.2f s" % (ms / 1000)
    if ms >= 10:
        return "%.1f ms" % ms
    return "%.2f ms" % ms


def esc(text: str) -> str:
    return html.escape(short(text, 80), quote=True)


def render_section(
    title: str,
    blocks: List[Dict],
    t0: int,
    px_per_ms: float,
    labels: bool = True,
) -> Tuple[str, int]:
    if not blocks:
        return "", 0
    lanes = pack(blocks)
    top = 20
    height = top + lanes * (LANE_H + LANE_GAP)
    out = [
        '<div class="row"><div class="name">%s</div><div class="lane" '
        'style="height:%dpx">' % (esc(title), height)
    ]
    for item in blocks:
        left = (item["start"] - t0) / 1e6 * px_per_ms
        width = max(1.0, (item["end"] - item["start"]) / 1e6 * px_per_ms)
        color = COLORS.get(item["kind"], COLORS["Shader"])
        text = ""
        if labels and width >= 46:
            text = esc(item["label"])
        tooltip = html.escape(
            "%s — %s" % (item["label"], fmt_ms(item["end"] - item["start"])), quote=True
        )
        out.append(
            '<div class="block" title="%s" style="left:%.2fpx;width:%.2fpx;'
            'top:%dpx;background:%s">%s</div>'
            % (
                tooltip,
                left,
                width,
                top + item["lane"] * (LANE_H + LANE_GAP),
                color,
                text,
            )
        )
    out.append("</div></div>")
    return "".join(out), height


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", help="trace directory from profile.sh")
    ap.add_argument("--out", help="output HTML (default: TRACE_DIR/timeline.html)")
    ap.add_argument("--process", help="process name substring (default: busiest)")
    ap.add_argument("--window", help="START:END in milliseconds (default: whole trace)")
    ap.add_argument("--title", default="Metal timeline")
    ap.add_argument("--max-lanes", type=int, default=10)
    args = ap.parse_args()

    path = args.trace if os.path.isdir(args.trace) else os.path.dirname(args.trace)
    enc_table = load(path, "encoders.xml")
    sh_table = load(path, "shaders.xml")
    cb_table = load(path, "command-buffers.xml")
    if enc_table is None and sh_table is None:
        raise SystemExit("no encoders.xml or shaders.xml in %s; run scripts/profile.sh" % path)

    process = ""
    enc_rows: List = []
    if enc_table is not None and enc_table.rows:
        enc_rows, process = pick_process(enc_table, enc_table.rows, args.process)
    elif sh_table is not None and sh_table.rows:
        _, process = pick_process(sh_table, sh_table.rows, args.process)

    command_buffers: List[Dict] = []
    if cb_table is not None and cb_table.rows:
        rows = cb_table.rows
        if process:
            rows = [r for r in rows if cb_table.process(r) == process]
        for row in rows:
            labels = cb_table.labels(row)
            ident = cb_table.fmt(cb_table.child(row, "metal-command-buffer-id"))
            command_buffers.append(
                block(
                    cb_table.ival(cb_table.child(row, "start-time")),
                    cb_table.ival(cb_table.child(row, "duration")),
                    labels[0] if labels else ident,
                    "Command buffer",
                    ident,
                )
            )

    encoders: List[Dict] = []
    if enc_table is not None:
        for row in enc_rows:
            labels = enc_table.labels(row)
            event = enc_table.fmt(enc_table.child(row, "metal-event-name"))
            ident = enc_table.fmt(enc_table.child(row, "metal-command-buffer-id"))
            encoders.append(
                block(
                    enc_table.ival(enc_table.child(row, "start-time")),
                    enc_table.ival(enc_table.child(row, "duration")),
                    labels[2] if len(labels) > 2 else (labels[-1] if labels else event),
                    kind_of(event),
                    ident,
                )
            )

    shaders: List[Dict] = []
    if sh_table is not None and sh_table.rows:
        rows = sh_table.rows
        if process:
            rows = [r for r in rows if sh_table.process(r) == process]
        for row in rows:
            labels = sh_table.labels(row)
            name = base_shader_name(labels[0] if labels else "<unknown>")
            shaders.append(
                block(
                    sh_table.ival(sh_table.child(row, "start-time")),
                    sh_table.ival(sh_table.child(row, "duration")),
                    name,
                    "Shader",
                )
            )

    all_blocks = command_buffers + encoders + shaders
    if not all_blocks:
        raise SystemExit("no rows matched process %r" % (args.process or ""))

    win_start, win_end = parse_window(args.window)
    starts = [b["start"] for b in all_blocks]
    ends = [b["end"] for b in all_blocks]
    t0 = win_start if win_start is not None else min(starts)
    t1 = win_end if win_end is not None else max(ends)
    if t1 <= t0:
        raise SystemExit("empty time window")

    def clip(items: List[Dict]) -> List[Dict]:
        kept = []
        for item in items:
            if item["end"] < t0 or item["start"] > t1:
                continue
            item = dict(item)
            item["start"] = max(item["start"], t0)
            item["end"] = min(item["end"], t1)
            kept.append(item)
        return kept

    command_buffers = clip(command_buffers)
    encoders = clip(encoders)
    shaders = clip(shaders)

    span_ns = t1 - t0
    px_per_ms = WIDTH / (span_ns / 1e6)

    sections = []
    for title, items, show in (
        ("Command buffers", command_buffers, False),
        ("Encoders", encoders, True),
        ("Shaders", shaders, True),
    ):
        body, height = render_section(title, items, t0, px_per_ms, show)
        if body:
            sections.append(body)

    ticks = []
    for i in range(11):
        left = WIDTH * i / 10.0
        ticks.append(
            '<div class="tick" style="left:%.1fpx">%s</div>'
            % (left, fmt_ms(int(span_ns * i / 10.0)))
        )

    out_path = args.out or os.path.join(path, "timeline.html")
    doc = """<!doctype html>
<html><head><meta charset="utf-8"><title>%(title)s</title>
<style>
  body { margin: 0; padding: 24px; background: #1b1b1d; color: #ddd;
         font: 12px -apple-system, BlinkMacSystemFont, "SF Mono", monospace; }
  h1 { font-size: 15px; font-weight: 600; margin: 0 0 4px; }
  .meta { color: #999; margin-bottom: 16px; }
  .axis { position: relative; margin-left: %(name_w)dpx; width: %(width)dpx; height: 26px;
          border-bottom: 1px solid #444; }
  .tick { position: absolute; top: 4px; color: #888; transform: translateX(-50%%);
          font-size: 10px; white-space: nowrap; }
  .row { display: flex; }
  .name { width: %(name_w)dpx; flex: none; padding-top: 24px; color: #aaa;
          position: sticky; left: 0; background: #1b1b1d; z-index: 2; }
  .lane { position: relative; flex: none; width: %(width)dpx;
          border-bottom: 1px solid #2c2c2f; margin-top: 8px; }
  .block { position: absolute; height: %(lane_h)dpx; line-height: %(lane_h)dpx;
           border-radius: 2px; color: #111; font-size: 9px; overflow: hidden;
           white-space: nowrap; padding: 0 2px; box-sizing: border-box;
           border: 1px solid rgba(0,0,0,.25); }
  .legend { margin: 14px 0 0 %(name_w)dpx; color: #999; }
  .swatch { display: inline-block; width: 9px; height: 9px; margin: 0 4px 0 12px; }
</style></head><body>
<h1>%(title)s</h1>
<div class="meta">%(process)s · %(span)s · %(counts)s</div>
<div class="axis">%(ticks)s</div>
%(sections)s
<div class="legend"><span class="swatch" style="background:#5c4d16"></span>command buffer
<span class="swatch" style="background:#c9a227"></span>compute
<span class="swatch" style="background:#4a90d9"></span>render
<span class="swatch" style="background:#7f7f7f"></span>blit / other</div>
</body></html>
""" % {
        "title": html.escape(args.title),
        "name_w": NAME_W,
        "width": WIDTH,
        "lane_h": LANE_H,
        "process": html.escape(process or "<unknown>"),
        "span": fmt_ms(span_ns),
        "counts": "%d command buffers, %d encoders, %d shaders"
        % (len(command_buffers), len(encoders), len(shaders)),
        "ticks": "".join(ticks),
        "sections": "".join(sections),
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(
        "wrote %s (%d command buffers, %d encoders, %d shaders, %s)"
        % (out_path, len(command_buffers), len(encoders), len(shaders), fmt_ms(span_ns))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
