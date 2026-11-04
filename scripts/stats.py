#!/usr/bin/env python3
"""Average 5-run decode numbers and bandwidth share.

usage: scripts/stats.py [--model-gb 1.61] [--peak-gbps 68.3] [VALUE ...]
       printf '%s\\n' 33.3 33.1 33.4 | scripts/stats.py --model-gb 1.61
"""

from __future__ import annotations

import argparse
import statistics
import sys
from typing import List


def read_values(argv: List[str]) -> List[float]:
    lines = argv if argv else [line for line in sys.stdin.read().splitlines()]
    values = []
    for line in lines:
        line = line.split("#", 1)[0].strip().replace(",", ".")
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            raise SystemExit("not a number: %r" % line)
    if not values:
        raise SystemExit("no values (pass them as arguments or on stdin)")
    return values


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("values", nargs="*", help="decode tok/s samples")
    ap.add_argument("--model-gb", type=float, help="bytes read per token, in GB")
    ap.add_argument("--peak-gbps", type=float, default=68.3, help="DRAM peak (default 68.3)")
    ap.add_argument("--label", default="", help="optional prefix")
    args = ap.parse_args()

    values = read_values(args.values)
    avg = statistics.fmean(values)
    print(
        "%sn=%d  avg=%.3f  min=%.3f  max=%.3f  stdev=%.3f"
        % (
            (args.label + " ") if args.label else "",
            len(values),
            avg,
            min(values),
            max(values),
            statistics.pstdev(values) if len(values) > 1 else 0.0,
        )
    )
    if args.model_gb:
        gbps = args.model_gb * avg
        print(
            "bandwidth=%.1f GB/s (%.1f%% of %.1f GB/s)"
            % (gbps, 100.0 * gbps / args.peak_gbps, args.peak_gbps)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
