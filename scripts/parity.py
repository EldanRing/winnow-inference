#!/usr/bin/env python3
"""Save candidate distributions, or compare them with a prior run of these probes."""

import argparse
import json
import urllib.request
from pathlib import Path

from http_client import request_headers

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8091")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--compare", type=Path)
    p.add_argument("--tolerance", type=float, default=0.002)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cases = json.loads((ROOT / "tests/parity-requests.json").read_text())
    for count in (8, 32, 64):
        cases.append(
            {
                "state": "No distinguishing information is available about any item.",
                "questions": {
                    "pick": {
                        "type": "choice",
                        "instructions": "Pick the best item.",
                        "criteria": {f"item_{i}": None for i in range(count)},
                    }
                },
            }
        )
    responses = []
    for case in cases:
        case["model"] = "Winnow-12B"
        case["winnow"] = {"diagnostics": True, "reuse_prefix": False}
        req = urllib.request.Request(
            a.url + "/v1/systemone",
            data=json.dumps(case).encode(),
            headers=request_headers(),
        )
        with urllib.request.urlopen(req, timeout=600) as response:
            responses.append(json.load(response))
    (a.output / "requests.json").write_text(json.dumps(cases, indent=2) + "\n")
    (a.output / "responses.json").write_text(json.dumps(responses, indent=2) + "\n")
    report = {"cases": len(cases)}
    if a.compare:
        old = json.loads((a.compare / "responses.json").read_text())
        assert cases == json.loads((a.compare / "requests.json").read_text()), "Inputs differ"
        delta = 0.0
        logit_delta = 0.0
        changed = 0
        questions = 0
        for left, right in zip(old, responses, strict=True):
            for key, answer in left["answers"].items():
                other = right["answers"][key]
                questions += 1
                x = answer["winnow"]["logits"]
                y = other["winnow"]["logits"]
                logit_delta = max(logit_delta, max(abs(u - v) for u, v in zip(x, y, strict=True)))
                changed += x.index(max(x)) != y.index(max(y))
                if answer["type"] == "noul":
                    delta = max(delta, abs(answer["noul"] - other["noul"]))
                else:
                    delta = max(
                        delta,
                        max(
                            abs(v - other["probabilities"][k])
                            for k, v in answer["probabilities"].items()
                        ),
                    )
        report.update(
            questions=questions,
            maximum_probability_delta=delta,
            maximum_logit_delta=logit_delta,
            changed_winners=changed,
            tolerance=a.tolerance,
            passed=delta <= a.tolerance and changed == 0,
        )
    (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if a.compare and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
