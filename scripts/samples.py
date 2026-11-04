#!/usr/bin/env python3
"""Per-kernel PC sample share from an xctrace shader profiler export.

usage: scripts/samples.py TRACE_DIR [--process NAME] [--top N] [--all] [--csv]

Joins gpu-shader-profiler-sample PCs to metal-shader-profiler-shader-list
ranges, using the shader intervals only to break ties when ranges overlap.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import heapq
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xctrace_export import Table, base_shader_name, pick_process, short


def find_file(path: str, name: str) -> str:
    if os.path.isdir(path):
        path = os.path.join(path, name)
    if not os.path.isfile(path):
        raise SystemExit(
            "no %s in trace; record with scripts/profile.sh --samples" % name
        )
    return path


def cell(row, table: Table, column: str):
    return list(row)[table.columns[column]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", help="trace directory from profile.sh")
    ap.add_argument("--process", help="process name substring (default: busiest)")
    ap.add_argument("--top", type=int, default=20, help="rows to print (default 20)")
    ap.add_argument("--all", action="store_true", help="print every kernel")
    ap.add_argument("--csv", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    shaders = Table(find_file(args.trace, "shaders.xml"))
    ranges = Table(find_file(args.trace, "shader-list.xml"))
    samples = Table(find_file(args.trace, "samples.xml"))
    if not shaders.rows:
        raise SystemExit("no shader intervals in %s" % args.trace)

    rows, process = pick_process(shaders, shaders.rows, args.process)
    intervals: List[Tuple[int, int, str]] = []
    for row in rows:
        start = shaders.ival(cell(row, shaders, "start"))
        dur = shaders.ival(cell(row, shaders, "duration"))
        intervals.append((start, start + dur, shaders.labels(row)[0]))
    intervals.sort()

    kernels: List[Tuple[int, int, str]] = []
    for row in ranges.rows:
        if process and ranges.process(row) != process:
            continue
        name = ranges.labels(row)[0] if ranges.labels(row) else ""
        if not name:
            continue
        start = ranges.ival(cell(row, ranges, "pc-start"))
        end = ranges.ival(cell(row, ranges, "pc-end"))
        kernels.append((start, end, name))
    kernels.sort()
    if not kernels:
        raise SystemExit(
            "no shader-list ranges for %s; record with --instrument 'Metal GPU Counters'"
            % (process or "<unknown>")
        )
    starts = [item[0] for item in kernels]

    def candidates(pc: int) -> List[Tuple[int, int, str]]:
        i = bisect.bisect_right(starts, pc) - 1
        found = []
        while i >= 0 and kernels[i][0] <= pc:
            start, end, name = kernels[i]
            if pc < end:
                found.append(kernels[i])
            i -= 1
            if len(found) > 8:
                break
        return found

    counts: Counter = Counter()
    hits = 0
    unmatched = 0
    heap: List[Tuple[int, str]] = []
    active: Counter = Counter()
    index = 0
    last_t = -1

    sample_rows = []
    for row in samples.rows:
        t = samples.ival(cell(row, samples, "timestamp"))
        elem = samples.child(row, "uint64-array")
        text = (samples.fmt(elem) or elem.text or "").strip() if elem is not None else ""
        pcs = [int(x, 0) for x in text.split()]
        sample_rows.append((t, pcs))
    sample_rows.sort()

    for t, pcs in sample_rows:
        if t < last_t:
            raise SystemExit("samples are not chronological")
        last_t = t
        while index < len(intervals) and intervals[index][0] <= t:
            start, end, name = intervals[index]
            heapq.heappush(heap, (end, name))
            active[name] += 1
            index += 1
        while heap and heap[0][0] <= t:
            end, name = heapq.heappop(heap)
            active[name] -= 1
            if not active[name]:
                del active[name]
        live = set(active)
        if not live:
            continue

        for pc in pcs:
            found = candidates(pc)
            if not found:
                unmatched += 1
                continue
            if len(found) > 1:
                live_found = [item for item in found if item[2] in live]
                if live_found:
                    found = live_found
            name = min(found, key=lambda item: item[1] - item[0])[2]
            counts[name] += 1
            hits += 1

    if not hits:
        raise SystemExit("no PC hits mapped to %s" % (process or "<unknown>"))

    merged: Dict[str, int] = {}
    for name, count in counts.items():
        base = base_shader_name(name)
        merged[base] = merged.get(base, 0) + count

    ordered: List[Tuple[str, int, float]] = [
        (name, count, 100.0 * count / hits) for name, count in merged.items()
    ]
    ordered.sort(key=lambda item: item[1], reverse=True)

    if args.csv:
        writer = csv.writer(sys.stdout)
        writer.writerow(["kernel", "pc_hits", "pct"])
        show = ordered if args.all else ordered[: args.top]
        for name, count, pct in show:
            writer.writerow([name, count, "%.1f" % pct])
        return 0

    show = ordered if args.all else ordered[: args.top]
    rest = ordered[len(show) :]
    if rest:
        show = show + [
            ("rest (%d)" % len(rest), sum(x[1] for x in rest), sum(x[2] for x in rest))
        ]

    name_w = max([len("kernel")] + [len(short(x[0], 56)) for x in show])
    print(
        "process: %s, %d shader intervals, %d PC hits (%d unmatched)"
        % (process or "<unknown>", len(intervals), hits, unmatched)
    )
    print()
    print("%-*s  %10s  %6s" % (name_w, "kernel", "pc hits", "%pcs"))
    print("%s  %s  %s" % ("-" * name_w, "-" * 10, "-" * 6))
    for name, count, pct in show:
        print("%-*s  %10d  %6.1f" % (name_w, short(name, 56), count, pct))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
