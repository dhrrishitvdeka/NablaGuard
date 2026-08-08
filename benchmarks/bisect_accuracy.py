"""Verify first-bad accuracy and logarithmic probe count on controlled runs."""

from __future__ import annotations

import argparse
import math

from nablaguard.bisect import first_bad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    args = parser.parse_args()
    failures = 0
    probe_counts = []
    boundaries = range(1, args.steps + 1, max(args.steps // 100, 1))
    for boundary in boundaries:
        result = first_bad(0, args.steps, lambda step, target=boundary: step >= target)
        failures += result.first_bad_step != boundary
        probe_counts.append(len(result.probes) + 2)
    print(f"incorrect boundaries: {failures}")
    print(f"maximum predicate evaluations: {max(probe_counts)}")
    print(f"binary-search ceiling plus endpoints: {math.ceil(math.log2(args.steps)) + 2}")


if __name__ == "__main__":
    main()
