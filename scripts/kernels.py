#!/usr/bin/env python3
"""Per-kernel GPU time from an xctrace Shader Timeline export.

usage: scripts/kernels.py TRACE_OR_DIR [--process NAME] [--top N] [--all] [--csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xctrace_export import Table, base_shader_name, pick_process, short


def find_shaders(path: str) -> str:
    if os.path.isdir(path):
        path = os.path.join(path, "shaders.xml")
    if not os.path.isfile(path):
        raise SystemExit("no shaders.xml in %s; run scripts/profile.sh first" % path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", help="trace directory from profile.sh, or shaders.xml")
    ap.add_argument("--process", help="process name substring (default: busiest)")
    ap.add_argument("--top", type=int, default=20, help="rows to print (default 20)")
    ap.add_argument("--all", action="store_true", help="print every kernel")
    ap.add_argument("--csv", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    table = Table(find_shaders(args.trace))
    if not table.rows:
        raise SystemExit(
            "no shader rows: record with MST-shader.tracetemplate and "
            "--instrument 'Metal GPU Counters'"
        )
    rows, process = pick_process(table, table.rows, args.process)

    counts: Dict[str, int] = {}
    nanos: Dict[str, int] = {}
    for row in rows:
        labels = table.labels(row)
        name = base_shader_name(labels[0] if labels else "<unknown>")
        dur = table.ival(table.child(row, "duration"))
        counts[name] = counts.get(name, 0) + 1
        nanos[name] = nanos.get(name, 0) + dur

    total = sum(nanos.values())
    ordered: List[Tuple[str, int, int, float]] = [
        (name, counts[name], ns, (100.0 * ns / total) if total else 0.0)
        for name, ns in nanos.items()
    ]
    ordered.sort(key=lambda item: item[2], reverse=True)

    if args.csv:
        writer = csv.writer(sys.stdout)
        writer.writerow(["kernel", "dispatches", "ms", "gpu_pct"])
        show = ordered if args.all else ordered[: args.top]
        for name, dispatches, ns, pct in show:
            writer.writerow([name, dispatches, "%.3f" % (ns / 1e6), "%.1f" % pct])
        return 0

    show = ordered if args.all else ordered[: args.top]
    rest = ordered[len(show) :]
    if rest:
        show = show + [
            ("rest (%d)" % len(rest), sum(x[1] for x in rest), sum(x[2] for x in rest),
             sum(x[3] for x in rest))
        ]

    name_w = max([len("kernel")] + [len(short(x[0], 56)) for x in show])
    print("process: %s, %d shader intervals" % (process or "<unknown>", len(rows)))
    print()
    print("%-*s  %10s  %9s  %6s" % (name_w, "kernel", "dispatches", "ms", "%gpu"))
    print("%s  %s  %s  %s" % ("-" * name_w, "-" * 10, "-" * 9, "-" * 6))
    for name, dispatches, ns, pct in show:
        print(
            "%-*s  %10d  %9.3f  %6.1f"
            % (name_w, short(name, 56), dispatches, ns / 1e6, pct)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
