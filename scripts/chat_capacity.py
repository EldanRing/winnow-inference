#!/usr/bin/env python3
"""Reuse a populated decision fixture to verify long-context multimodal chat."""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from http_client import request_headers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8091")
    p.add_argument(
        "--fixture",
        type=Path,
        required=True,
        help="Directory from check.py --long-context",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    fixture = next(
        json.loads(line)["request"]
        for line in (a.fixture / "responses.jsonl").read_text().splitlines()
        if json.loads(line)["name"] == "long_cold"
    )
    content = [
        {
            "type": "text",
            "text": fixture["state"]
            + "\nReturn a short JSON object with the beginning marker, middle marker, final enabled flag, and image color (if present).",
        }
    ]
    for image in fixture["winnow"].get("images", []):
        content.append({"type": "image_url", "image_url": {"url": image}})
    body = {
        "model": "Winnow-12B",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 4096,
        "cache_prompt": False,
        "reasoning_effort": "none",
    }
    request = urllib.request.Request(
        a.url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=request_headers(),
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - start
    (a.output / "response.json").write_text(
        json.dumps({"request": body, "response": result, "seconds": elapsed}, indent=2) + "\n"
    )
    choice = result["choices"][0]
    text = choice["message"]["content"].lower()
    assert choice["finish_reason"] == "stop", choice
    assert all(word in text for word in ["amber", "violet", "true"]), text
    if fixture["winnow"].get("images"):
        assert "red" in text, text
    summary = {
        "seconds": elapsed,
        "timings": result.get("timings"),
        "usage": result.get("usage"),
        "content": choice["message"]["content"],
        "passed": True,
    }
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
