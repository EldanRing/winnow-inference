#!/usr/bin/env python3
"""Record decision batch timings; replay the identical cases to compare runtimes."""

import argparse
import copy
import json
import statistics
import time
import urllib.request
from pathlib import Path

from http_client import request_headers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", default="1,8,16,32,64")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compare", type=Path, help="Reference directory produced by this script")
    parser.add_argument("--tolerance", type=float, default=0.002)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    counts = [int(n) for n in args.counts.split(",")]
    if args.repeats < 1 or any(n < 1 or n > 256 for n in counts):
        parser.error("counts must be 1–256 and repeats must be positive")
    records = []
    if args.compare:
        prior = [
            json.loads(line) for line in (args.compare / "responses.jsonl").read_text().splitlines()
        ]
        cases = [(row["name"], row["request"]) for row in prior]
    else:
        cases = []
        for n in counts:
            body = {
                "model": "Winnow-12B",
                "state": {
                    "records": [
                        {
                            "id": f"item_{i:03}",
                            "route": "alpha" if i % 3 == 0 else "beta",
                            "active": bool(i % 2),
                            "notes": "A routine routing decision.",
                        }
                        for i in range(max(64, max(counts)))
                    ]
                },
                "questions": {
                    f"q{i:03}": {
                        "type": "choice",
                        "instructions": f"Which route is recorded for item_{i:03}?",
                        "criteria": {"alpha": None, "beta": None},
                    }
                    for i in range(n)
                },
                "winnow": {"diagnostics": True, "reuse_prefix": False},
            }
            cases.append((f"{n}_cold", copy.deepcopy(body)))
            body["winnow"]["reuse_prefix"] = True
            for repetition in range(args.repeats):
                cases.append((f"{n}_warm_{repetition}", copy.deepcopy(body)))
    for name, body in cases:
        request = urllib.request.Request(
            args.url + "/v1/systemone",
            data=json.dumps(body).encode(),
            headers=request_headers(),
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        row = {
            "name": name,
            "request": body,
            "response": result,
            "seconds": time.perf_counter() - start,
        }
        records.append(row)
        with (args.output / "responses.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{name}: {row['seconds']:.3f}s", flush=True)
    report = {"cases": len(records), "timings": {}}
    for n in sorted({len(row["request"]["questions"]) for row in records}):
        samples = [row for row in records if len(row["request"]["questions"]) == n]
        warm = [row["seconds"] for row in samples if "warm" in row["name"]]
        if warm:
            ordered = sorted(warm)
            report["timings"][str(n)] = {
                "cold_seconds": next(row["seconds"] for row in samples if "cold" in row["name"]),
                "warm_p50_seconds": statistics.median(warm),
                "warm_p95_seconds": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
                "warm_questions_per_second": n / statistics.median(warm),
                "warm_samples": len(warm),
            }
    if args.compare:
        maximum = 0.0
        changed = 0
        for left, right in zip(prior, records, strict=True):
            for key, answer in left["response"]["answers"].items():
                other = right["response"]["answers"][key]
                if answer["type"] == "noul":
                    maximum = max(maximum, abs(answer["noul"] - other["noul"]))
                    changed += (answer["noul"] >= 0.5) != (other["noul"] >= 0.5)
                else:
                    maximum = max(
                        maximum,
                        max(
                            abs(v - other["probabilities"][k])
                            for k, v in answer["probabilities"].items()
                        ),
                    )
                    changed += max(answer["probabilities"], key=answer["probabilities"].get) != max(
                        other["probabilities"], key=other["probabilities"].get
                    )
        report["comparison"] = {
            "maximum_probability_delta": maximum,
            "changed_winners": changed,
            "tolerance": args.tolerance,
            "passed": maximum <= args.tolerance and changed == 0,
        }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.compare and not report["comparison"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
