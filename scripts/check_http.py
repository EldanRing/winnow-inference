#!/usr/bin/env python3
"""Check authentication and request boundaries on an owned, authenticated server."""

import argparse
import copy
import json
import urllib.error
import urllib.request
from pathlib import Path

from http_client import api_key, request_headers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not api_key():
        parser.error("Set WINNOW_API_KEY_FILE or WINNOW_API_KEY for the test server")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []

    def probe(name, path, body, expected, auth="valid"):
        headers = request_headers() if auth == "valid" else {"Content-Type": "application/json"}
        if auth == "wrong":
            headers["Authorization"] = "Bearer invalid-" + api_key()
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        request = urllib.request.Request(args.url + path, data=payload, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = response.status
                response.read()
        except urllib.error.HTTPError as error:
            status = error.code
            error.read()
        row = {
            "name": name,
            "path": path,
            "bytes": len(payload),
            "status": status,
            "expected": expected,
            "passed": status == expected,
        }
        rows.append(row)
        with (args.output / "checks.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if status != expected:
            raise AssertionError(row)

    for path in ("/v1/systemone", "/v1/winnow/inspect", "/v1/chat/completions"):
        probe("missing_key", path, {}, 401, "none")
        probe("wrong_key", path, {}, 401, "wrong")
    body = {
        "state": {"enabled": True},
        "questions": {"flag": {"type": "noul", "instructions": "Is enabled true?"}},
    }
    probe("authorized_decision", "/v1/systemone", body, 200)
    probe("authorized_inspection", "/v1/winnow/inspect", body, 200)
    for path in ("/v1/systemone", "/v1/winnow/inspect"):
        probe("bad_json", path, b"{", 400)
        probe("missing_state", path, {"questions": body["questions"]}, 400)
        invalid = copy.deepcopy(body)
        invalid["questions"] = {str(i): body["questions"]["flag"] for i in range(257)}
        probe("too_many_questions", path, invalid, 400)
        invalid = copy.deepcopy(body)
        invalid["questions"] = {
            "pick": {"type": "choice", "criteria": {str(i): None for i in range(65)}}
        }
        probe("too_many_options", path, invalid, 400)
        invalid = copy.deepcopy(body)
        invalid["winnow"] = {"temperature": 0}
        probe("invalid_temperature", path, invalid, 400)
        invalid["winnow"] = {"images": ["data:image/png;base64,AA=="] * 17}
        probe("too_many_images", path, invalid, 400)
        probe("request_size_limit", path, b" " * (32 * 1024 * 1024 + 1), 413)
    probe("recovery", "/v1/systemone", body, 200)
    summary = {"passed": len(rows), "checks": rows}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"passed": len(rows)}))


if __name__ == "__main__":
    main()
