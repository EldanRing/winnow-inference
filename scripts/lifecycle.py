#!/usr/bin/env python3
"""Probe cancellation, invalid input recovery, mixed scheduling, and image chat."""

import argparse
import copy
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from check import png_url
from http_client import request_headers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8091")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--vision", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    rows = []

    def post(name, body, path="/v1/systemone", expected=200):
        request = urllib.request.Request(
            a.url + path,
            data=json.dumps(body).encode(),
            headers=request_headers(),
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                status = response.status
                result = json.load(response)
        except urllib.error.HTTPError as e:
            status = e.code
            result = json.loads(e.read())
        row = {
            "name": name,
            "request": body,
            "response": result,
            "status": status,
            "seconds": time.perf_counter() - start,
        }
        rows.append(row)
        with (a.output / "responses.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        assert status == expected, row
        return result

    body = {
        "model": "Winnow-12B",
        "state": {"enabled": True},
        "questions": {"flag": {"type": "noul", "instructions": "Is enabled true?"}},
        "winnow": {"diagnostics": True},
    }
    first = post("baseline", body)
    invalid = copy.deepcopy(body)
    invalid["state"] = "Background. " * 400000
    post("context_overflow", invalid, expected=400)
    invalid = copy.deepcopy(body)
    invalid["winnow"]["images"] = ["file:///etc/passwd"]
    post("invalid_image", invalid, expected=400)
    post("recovery", body)
    # A closed caller must not occupy the inference queue until a whole long prefill finishes.
    long = copy.deepcopy(body)
    long["state"] = "Background. " * 5000 + "\nEnabled: true."
    long["winnow"]["reuse_prefix"] = False
    payload = json.dumps(long).encode()
    target = urllib.parse.urlparse(a.url)
    if target.scheme != "http":
        p.error("Cancellation probe requires a local HTTP URL")
    auth = request_headers().get("Authorization")
    auth_header = f"Authorization: {auth}\r\n" if auth else ""
    with socket.create_connection((target.hostname, target.port or 80), timeout=5) as sock:
        headers = f"POST /v1/systemone HTTP/1.1\r\nHost: {target.netloc}\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\n{auth_header}Connection: close\r\n\r\n".encode()
        sock.sendall(headers + payload)
        time.sleep(0.15)
        sock.shutdown(socket.SHUT_RDWR)
    after = post("after_disconnect", body)
    assert after["winnow"]["cancelled_requests"] > first["winnow"]["cancelled_requests"], after
    assert after["answers"]["flag"]["noul"] > 0.5
    chat = {
        "model": "Winnow-12B",
        "messages": [
            {
                "role": "user",
                "content": "Explain prefix caching in one short paragraph.",
            }
        ],
        "temperature": 0,
        "max_tokens": 4096,
        "cache_prompt": False,
        "reasoning_effort": "none",
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        fchat = pool.submit(post, "mixed_chat", chat, "/v1/chat/completions")
        time.sleep(0.1)
        fdecision = pool.submit(post, "mixed_long_decision", long)
        c = fchat.result()
        d = fdecision.result()
    assert c["choices"][0]["finish_reason"] == "stop" and d["answers"]["flag"]["noul"] > 0.5
    checks = [
        "overflow_recovery",
        "invalid_image_recovery",
        "cancelled_prefill_recovery",
        "mixed_long_request",
    ]
    if a.vision:
        vision = copy.deepcopy(chat)
        vision["messages"][0]["content"] = [
            {
                "type": "text",
                "text": "What color fills this image? Answer with one word.",
            },
            {"type": "image_url", "image_url": {"url": png_url((255, 0, 0))}},
        ]
        v = post("vision_chat", vision, "/v1/chat/completions")
        assert (
            v["choices"][0]["finish_reason"] == "stop"
            and "red" in v["choices"][0]["message"]["content"].lower()
        )
        checks.append("vision_chat")
    post("final_recovery", body)
    report = {
        "checks": checks,
        "passed": len(checks),
        "mixed_policy": d["winnow"]["memory_policy"],
        "mixed_cooperative_yield_ms": d["winnow"]["cooperative_yield_ms"],
        "mixed_queue_ms": d["winnow"]["queue_ms"],
    }
    (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
