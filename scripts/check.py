#!/usr/bin/env python3
"""Focused, fully recorded API/cache/mixed-load checks against a running server."""

import argparse
import base64
import copy
import json
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from http_client import request_headers


def png_url(rgb, size=512):
    def chunk(kind, data):
        return (
            struct.pack("!I", len(data))
            + kind
            + data
            + struct.pack("!I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    pixels = (b"\0" + bytes(rgb) * size) * size
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!2I5B", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(data).decode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8091")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--vision", action="store_true")
    p.add_argument(
        "--long-context",
        type=int,
        default=0,
        help="Populate approximately this many input positions; reserve 512 for questions",
    )
    p.add_argument("--image-size", type=int, default=512)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    mutex = threading.Lock()
    observations = []

    def post(name, body, path="/v1/systemone", expected=200):
        req = urllib.request.Request(
            a.url + path,
            data=json.dumps(body).encode(),
            headers=request_headers(),
        )
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                status = response.status
                result = json.load(response)
        except urllib.error.HTTPError as e:
            status = e.code
            result = json.loads(e.read())
        row = {
            "name": name,
            "request": body,
            "path": path,
            "response": result,
            "status": status,
            "seconds": time.perf_counter() - t,
        }
        with mutex:
            observations.append(row)
            with (a.output / "responses.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"{name}: HTTP {status}, {row['seconds']:.3f}s", flush=True)
        assert status == expected, (name, status, result)
        return result

    q = {
        "enabled": {"type": "noul", "instructions": "Is the enabled flag true?"},
        "queue": {
            "type": "choice",
            "instructions": "What is the assigned queue?",
            "criteria": {"alpha": None, "beta": None},
        },
        "level": {
            "type": "score",
            "instructions": "Which verification level is recorded?",
            "criteria": ["unverified", "partly verified", "fully verified"],
        },
    }
    body = {
        "model": "Winnow-12B",
        "state": {"enabled": True, "queue": "beta", "verification": "fully verified"},
        "questions": q,
        "winnow": {"diagnostics": True},
    }
    checks = []

    def same(left, right, ids=None, tolerance=1e-6):
        for k in ids or left["answers"]:
            x = left["answers"][k]
            y = right["answers"][k]
            if x["type"] == "noul":
                assert abs(x["noul"] - y["noul"]) <= tolerance, (k, x, y)
            else:
                assert (
                    max(
                        abs(x["probabilities"][s] - y["probabilities"][s])
                        for s in x["probabilities"]
                    )
                    <= tolerance
                ), (k, x, y)
                if x["type"] == "choice":
                    assert x["choice"] == y["choice"]

    first = post("cold", body)
    warm = post("warm", body)
    same(first, warm)
    assert warm["winnow"]["prefix_processed_tokens"] == 0
    assert (
        first["answers"]["enabled"]["noul"] > 0.5 and first["answers"]["queue"]["choice"] == "beta"
    )
    checks += ["cold_warm_parity", "warm_prefix_reuse", "basic_typed_values"]
    reverse = copy.deepcopy(body)
    reverse["questions"] = dict(reversed(list(q.items())))
    same(first, post("reverse", reverse), tolerance=0.005)
    checks.append("question_order")
    dup = copy.deepcopy(body)
    dup["questions"] = {"a": q["enabled"], "b": q["enabled"]}
    d = post("duplicates", dup)
    assert d["answers"]["a"]["noul"] == d["answers"]["b"]["noul"]
    assert d["winnow"]["unique_questions"] == 1
    checks.append("deduplication")
    changed = copy.deepcopy(body)
    changed["state"]["enabled"] = False
    r = post("changed_state", changed)
    assert r["winnow"]["prefix_processed_tokens"] > 0 and r["answers"]["enabled"]["noul"] < 0.5
    checks.append("state_invalidation")
    post("bad_schema", {"questions": q}, expected=400)
    post("recover", body)
    checks.append("invalid_request_recovery")
    with ThreadPoolExecutor(max_workers=4) as pool:
        requests = [copy.deepcopy(body) for _ in range(4)]
        for i, r in enumerate(requests):
            r["questions"] = {f"caller_{i}": q["enabled"]}
        responses = list(pool.map(lambda x: post("concurrent_decision", x), requests))
    assert all(set(r["answers"]) == {f"caller_{i}"} for i, r in enumerate(responses))
    checks.append("response_isolation")
    chat = {
        "model": "Winnow-12B",
        "messages": [
            {
                "role": "user",
                "content": "Explain why the sky looks blue in one short paragraph.",
            }
        ],
        "temperature": 0,
        "max_tokens": 4096,
        "cache_prompt": False,
        "reasoning_effort": "none",
    }
    before = post("chat_before", chat, "/v1/chat/completions")
    assert (
        before["choices"][0]["finish_reason"] == "stop"
        and len(before["choices"][0]["message"]["content"]) > 40
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        fchat = pool.submit(post, "mixed_chat", chat, "/v1/chat/completions")
        fdecision = pool.submit(post, "mixed_decision", body)
        mixed_chat = fchat.result()
        mixed_decision = fdecision.result()
    same(first, mixed_decision, tolerance=0.005)
    assert (
        before["choices"][0]["message"]["content"] == mixed_chat["choices"][0]["message"]["content"]
    )
    checks += ["ordinary_chat", "mixed_chat_parity", "mixed_decision_parity"]
    if a.vision:
        vision = copy.deepcopy(body)
        vision["state"] = {"task": "Use the attached image."}
        vision["questions"] = {
            "color": {
                "type": "choice",
                "instructions": "What color fills the image?",
                "criteria": {"red": None, "blue": None},
            }
        }
        vision["winnow"]["images"] = [png_url((255, 0, 0), a.image_size)]
        red = post("red_image", vision)
        same(red, post("red_image_warm", vision))
        assert red["answers"]["color"]["choice"] == "red"
        vision["winnow"]["images"] = [png_url((0, 0, 255), a.image_size)]
        blue = post("blue_image", vision)
        assert (
            blue["answers"]["color"]["choice"] == "blue"
            and blue["winnow"]["prefix_processed_tokens"] > 0
        )
        checks += ["vision_counterfactual", "vision_warm_parity", "image_invalidation"]
    if a.long_context:
        target = a.long_context - 512
        long = copy.deepcopy(body)
        long["questions"] = {
            "flag": {"type": "noul", "instructions": "Is the final enabled flag true?"},
            "beginning": {
                "type": "choice",
                "instructions": "What is the beginning marker?",
                "criteria": {"amber": None, "violet": None},
            },
            "middle": {
                "type": "choice",
                "instructions": "What is the middle marker?",
                "criteria": {"amber": None, "violet": None},
            },
        }

        def long_state(n):
            return (
                "Beginning marker: amber.\n"
                + "Background record. " * (n // 2)
                + "\nMiddle marker: violet.\n"
                + "Background record. " * (n - n // 2)
                + "\nFinal enabled flag: true."
            )

        if a.vision:
            long["questions"]["color"] = {
                "type": "choice",
                "instructions": "What color fills the attached image?",
                "criteria": {"red": None, "blue": None},
            }
        if a.vision:
            long["winnow"]["images"] = [png_url((255, 0, 0), a.image_size)]
        low, high = 1, target
        while low < high:
            n = (low + high + 1) // 2
            long["state"] = long_state(n)
            r = post("size_probe", long, "/v1/winnow/inspect")
            if r["prefix_tokens"] <= target:
                low = n
            else:
                high = n - 1
        long["state"] = long_state(low)
        sized = post("long_inspect", long, "/v1/winnow/inspect")
        assert target - 10 <= sized["prefix_tokens"] <= target
        cold = post("long_cold", long)
        same(cold, post("long_warm", long))
        assert cold["answers"]["flag"]["noul"] > 0.5
        assert (
            cold["answers"]["beginning"]["choice"] == "amber"
            and cold["answers"]["middle"]["choice"] == "violet"
        )
        if a.vision:
            assert cold["answers"]["color"]["choice"] == "red"
        checks += ["populated_long_context", "long_context_warm_parity"]
    # Streaming must use the original llama-server SSE shape.
    streaming = copy.deepcopy(chat)
    streaming["stream"] = True
    req = urllib.request.Request(
        a.url + "/v1/chat/completions",
        data=json.dumps(streaming).encode(),
        headers=request_headers(),
    )
    chunks = []
    content = []
    done = False
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            text = line.decode().strip()
            if not text.startswith("data: "):
                continue
            payload = text[6:]
            if payload == "[DONE]":
                done = True
                break
            chunk = json.loads(payload)
            chunks.append(chunk)
            for choice in chunk.get("choices", []):
                content.append(choice.get("delta", {}).get("content") or "")
    (a.output / "stream.json").write_text(json.dumps(chunks, indent=2) + "\n")
    assert done and "".join(content) == before["choices"][0]["message"]["content"]
    checks.append("streaming_chat_parity")
    (a.output / "summary.json").write_text(
        json.dumps(
            {
                "checks": checks,
                "count": len(checks),
                "seconds": sum(x["seconds"] for x in observations),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"passed": len(checks), "output": str(a.output)}))


if __name__ == "__main__":
    main()
