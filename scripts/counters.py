#!/usr/bin/env python3
"""Approximate per-kernel GPU counters from an xctrace export.

usage: scripts/counters.py TRACE_DIR [--process NAME] [--top N]

TRACE_DIR comes from:
  scripts/profile.sh --attach PID --counters --keep --out TRACE_DIR

Counter samples are global; this joins each sample to the Shader Timeline
interval whose [start, start+duration) contains its timestamp. That is valid
for a standalone one-kernel-at-a-time harness, not for a busy model trace.
"""

from __future__ import annotations

import argparse
import bisect
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xctrace_export import Table, base_shader_name, pick_process, short

SELECTED = [
    "ALU Limiter",
    "Buffer Read Limiter",
    "GPU Last Level Cache Limiter",
    "Compute Occupancy",
    "GPU Read Bandwidth",
]
SHORT = {
    "ALU Limiter": "alu%",
    "Buffer Read Limiter": "buf%",
    "GPU Last Level Cache Limiter": "llc%",
    "Compute Occupancy": "occ%",
    "GPU Read Bandwidth": "read GB/s",
}


def export_info(trace: str, workdir: str) -> Optional[Table]:
    """gpu-counter-info is split over two tables; export the one with rows."""
    toc = ET.fromstring(
        subprocess.check_output(["xcrun", "xctrace", "export", "--input", trace, "--toc"])
    )
    data = toc.find("./run[@number='1']/data")
    if data is None:
        return None
    for index, table in enumerate(data.findall("table"), start=1):
        if table.get("schema") != "gpu-counter-info":
            continue
        path = os.path.join(workdir, "counter-info.xml")
        with open(path, "w") as fh:
            subprocess.run(
                [
                    "xcrun",
                    "xctrace",
                    "export",
                    "--input",
                    trace,
                    "--xpath",
                    '//trace-toc/run[@number="1"]/data/table[%d]' % index,
                ],
                stdout=fh,
                check=True,
            )
        parsed = Table(path)
        if parsed.rows:
            return parsed
    return None


def counter_names(info: Optional[Table]) -> Dict[int, str]:
    names: Dict[int, str] = {}
    if info is None:
        return names
    for row in info.rows:
        cid = None
        name = None
        for child in row:
            if child.tag == "uint32" and cid is None:
                cid = info.ival(child)
            if child.tag == "gpu-counter-name":
                name = info.fmt(child)
        if cid is not None and name:
            names[cid] = name
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", help="directory with run.trace, shaders.xml, counters.xml")
    ap.add_argument("--process", help="process name substring (default: busiest)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    trace_dir = args.trace if os.path.isdir(args.trace) else os.path.dirname(args.trace)
    trace = os.path.join(trace_dir, "run.trace")
    shaders = os.path.join(trace_dir, "shaders.xml")
    counters = os.path.join(trace_dir, "counters.xml")
    for path in (trace, shaders, counters):
        if not os.path.exists(path):
            raise SystemExit(
                "missing %s; run scripts/profile.sh --counters --keep first" % path
            )

    sh = Table(shaders)
    rows, process = pick_process(sh, sh.rows, args.process)
    intervals: List[Tuple[int, int, str]] = []
    for row in rows:
        labels = sh.labels(row)
        name = base_shader_name(labels[0] if labels else "<unknown>")
        start = sh.ival(sh.child(row, "start-time"))
        intervals.append((start, start + sh.ival(sh.child(row, "duration")), name))
    intervals.sort()
    if not intervals:
        raise SystemExit("no shader intervals in %s" % shaders)
    starts = [item[0] for item in intervals]

    info = export_info(trace, trace_dir) if os.path.exists(trace) else None
    names = counter_names(info)

    ids: Dict[str, ET.Element] = {}

    def resolve(elem: Optional[ET.Element]) -> Optional[ET.Element]:
        if elem is None:
            return None
        ref = elem.get("ref")
        return ids.get(ref) if ref else elem

    def ival(elem: Optional[ET.Element]) -> Optional[int]:
        e = resolve(elem)
        if e is None:
            return None
        text = (e.text or "").strip().replace(",", "")
        return int(text) if re.fullmatch(r"-?[0-9]+", text) else None

    def fval(elem: Optional[ET.Element]) -> Optional[float]:
        e = resolve(elem)
        if e is None:
            return None
        text = (e.text or "").strip().replace(",", "")
        try:
            return float(text)
        except ValueError:
            return None

    agg: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    samples = 0
    with open(counters) as fh:
        for line in fh:
            match = re.search(r"<row>(.*)</row>", line)
            if not match:
                continue
            try:
                row = ET.fromstring("<row>" + match.group(1) + "</row>")
            except ET.ParseError:
                continue
            children = list(row)
            if len(children) < 3:
                continue
            for child in children:
                eid = child.get("id")
                if eid:
                    ids[eid] = child
            ts = ival(children[0])
            cid = ival(children[1])
            val = fval(children[2])
            if ts is None or cid is None or val is None:
                continue
            index = bisect.bisect_right(starts, ts) - 1
            hit = None
            for candidate in (index, index - 1, index + 1):
                if 0 <= candidate < len(intervals):
                    s, e, name = intervals[candidate]
                    if s <= ts < e:
                        hit = name
                        break
            if hit:
                agg[hit][cid].append(val)
            samples += 1

    if not agg:
        raise SystemExit("no counter samples landed inside shader intervals")

    order = sorted(
        agg, key=lambda k: sum(len(v) for v in agg[k].values()), reverse=True
    )[: args.top]
    result = []
    for name in order:
        item = {"kernel": name, "intervals": sum(1 for s, e, n in intervals if n == name)}
        for cid, values in agg[name].items():
            cname = names.get(cid)
            if cname in SELECTED and values:
                item[cname] = sum(values) / len(values)
        result.append(item)

    joined = sum(len(v) for per_kernel in agg.values() for v in per_kernel.values())
    if args.json:
        import json

        print(
            json.dumps(
                {"process": process, "samples": samples, "joined": joined, "kernels": result},
                indent=2,
            )
        )
        return 0

    print(
        "process: %s, %d shader intervals, %d/%d counter samples joined"
        % (process, len(intervals), joined, samples)
    )
    print("approx: samples joined by timestamp; run one kernel at a time")
    print()
    header = "%-58s %6s %7s %7s %7s %7s %10s" % (
        "kernel",
        "n",
        "alu%",
        "buf%",
        "llc%",
        "occ%",
        "read GB/s",
    )
    print(header)
    print("-" * len(header))
    for item in result:
        print(
            "%-58s %6d %7s %7s %7s %7s %10s"
            % (
                short(item["kernel"], 58),
                item["intervals"],
                *[
                    ("%.1f" % item[name]) if name in item else "-"
                    for name in SELECTED
                ],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
