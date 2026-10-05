#!/usr/bin/env python3
"""Call native decisions, or explicitly opt into a pinned experimental adaptive policy."""

import argparse
import http.client
import json
import socket
import sys
import threading
from pathlib import Path
from urllib.parse import urlsplit

from adaptive_policy import DecisionPipeline
from assets import MODELS, MODEL_ALIASES, selection
from http_client import request_headers


class HTTPTransport:
    """Synchronous CLI use, hard per-request wall limit, close socket on timeout."""

    def __init__(self, base_url, headers=None):
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
            raise ValueError("Use an http(s) backend URL without embedded credentials")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("Backend URL must be an origin")
        self.scheme, self.host, self.port = parsed.scheme, parsed.hostname, parsed.port
        self.headers = headers if headers is not None else request_headers()
        self.lock = threading.Lock()

    def post(self, endpoint, body, seconds):
        with self.lock:
            payload = json.dumps(body, allow_nan=False).encode()
            if len(payload) > 32 * 1024 * 1024:
                raise ValueError("Request exceeds32MiB")
            cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
            connection = cls(self.host, self.port, timeout=seconds)
            done, result = threading.Event(), []

            def worker():
                try:
                    connection.request("POST", endpoint, payload, self.headers)
                    response = connection.getresponse()
                    data = response.read(32 * 1024 * 1024 + 1)
                    if len(data) > 32 * 1024 * 1024:
                        raise ValueError("Backend response exceeds32MiB")
                    if response.status != 200:
                        # Error details may contain request text; keep logs/CLI failures bounded.
                        raise RuntimeError("Backend HTTP " + str(response.status))
                    decoded = json.loads(data)
                    if not isinstance(decoded, dict):
                        raise ValueError("Backend response must be an object")
                    result.append(decoded)
                except (OSError, ValueError, http.client.HTTPException, RuntimeError) as error:
                    result.append(error)
                finally:
                    connection.close()
                    done.set()

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            if not done.wait(seconds):
                if connection.sock:
                    try:
                        connection.sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                connection.close()
                thread.join(timeout=1)
                raise TimeoutError("Backend request exceeded wall deadline; connection closed")
            thread.join(timeout=1)
            if isinstance(result[0], BaseException):
                raise result[0]
            return result[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", help="Explicitly override the input request's model alias")
    parser.add_argument("--target", choices=[*MODEL_ALIASES, *MODELS], help="Require the selected artifact identity")
    parser.add_argument("--mode", choices=["direct", "experimental-adaptive"], default="direct")
    parser.add_argument("--policy", choices=["nvfp4-entropy-v1", "e4b-calibrated50-v1", "e4b-calibrated75-g95-v1", "q8-fixed50-v1"])
    parser.add_argument("--mtp", choices=["on", "off"], default="on", help="Must match the adaptive server's MTP setting")
    args = parser.parse_args()
    try:
        body = json.loads(args.input.read_text())
        if args.model:
            if not isinstance(body, dict):
                raise ValueError("Decision input must be an object")
            body["model"] = args.model
        target = selection(args.target)[0] if args.target else None
        response = DecisionPipeline(HTTPTransport(args.base_url), args.mode, args.policy, args.mtp, target).decide(body)
        note = response.get("winnow", {}).get("adaptive", {}).get("configuration_note")
        if note:
            print(note, file=sys.stderr)
        text = json.dumps(response, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(text)
        else:
            print(text, end="")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, http.client.HTTPException) as error:
        parser.exit(1, "Decision failed: " + str(error) + "\n")


if __name__ == "__main__":
    main()
