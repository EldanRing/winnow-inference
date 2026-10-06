#!/usr/bin/env python3
"""Call native decisions, or explicitly opt into a pinned experimental adaptive policy."""

import argparse
import http.client
import json
import math
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from adaptive_policy import DecisionPipeline, REASONING_CHOICES
from assets import MODELS, MODEL_ALIASES, selection
from http_client import request_headers
from reasoning_contract import BackendUnavailable, RequestCancelled


class _BackendState:
    def __init__(self):
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.draining = None

    def check_available(self):
        with self.state_lock:
            if self.draining is not None:
                raise BackendUnavailable("Backend unavailable until the aborted request worker stops")


_BACKENDS, _BACKENDS_LOCK = {}, threading.Lock()


def _release_after_exit(worker, lock, backend=None):
    """A daemon waiter owns the lock until join proves the I/O worker has exited."""
    def release():
        worker.join()
        if backend is not None:
            with backend.state_lock:
                backend.draining = None
        lock.release()
    threading.Thread(target=release, daemon=True).start()


class HTTPTransport:
    """Bounded caller wait; aborted I/O retains backend ownership until it exits."""

    def __init__(self, base_url, headers=None, cancelled=None):
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
            raise ValueError("Use an http(s) backend URL without embedded credentials")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("Backend URL must be an origin")
        self.scheme, self.host, self.port = parsed.scheme, parsed.hostname, parsed.port
        self.headers = dict(headers if headers is not None else request_headers())
        self.cancelled = cancelled
        origin = (self.scheme, self.host.lower(), self.port or (443 if self.scheme == "https" else 80))
        with _BACKENDS_LOCK:
            self.backend = _BACKENDS.setdefault(origin, _BackendState())
        self.lock = self.backend.lock
        self._worker = None

    def check_available(self):
        self.backend.check_available()

    def release_when_idle(self, lock):
        """Transfer a pipeline's held request lock to a waiter if cancellation left I/O."""
        worker = self._worker
        if worker is not None and worker.is_alive():
            _release_after_exit(worker, lock)
        else:
            lock.release()

    def post(self, endpoint, body, seconds):
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Backend phase deadline must be positive and finite")
        deadline = time.monotonic() + seconds

        def check_cancelled():
            if self.cancelled and self.cancelled():
                raise RequestCancelled("Decision request cancelled")
            if time.monotonic() >= deadline:
                raise TimeoutError("Backend request exceeded wall deadline")

        check_cancelled()
        self.check_available()
        while not self.lock.acquire(timeout=0.1):
            check_cancelled()
            self.check_available()
        thread = None
        try:
            check_cancelled()
            self.check_available()
            payload = json.dumps(body, allow_nan=False).encode()
            if len(payload) > 32 * 1024 * 1024:
                raise ValueError("Request exceeds32MiB")
            cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
            connection = cls(self.host, self.port, timeout=max(0.001, deadline - time.monotonic()))
            # request()/send() must never silently reconnect after the caller closes this socket.
            connection.auto_open = 0
            done, aborted, result = threading.Event(), threading.Event(), []
            abort_error = [None]

            def check_active():
                if aborted.is_set():
                    raise abort_error[0]
                check_cancelled()

            original_send = connection.send

            def guarded_send(data):
                check_active()
                original_send(data)
                check_active()

            connection.send = guarded_send

            def close_connection():
                sock = connection.sock
                shutdown = getattr(sock, "shutdown", None)
                if shutdown is not None:
                    try:
                        shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                try:
                    connection.close()
                except OSError:
                    pass  # Cleanup must not mask cancellation or its non-fallback behavior.

            def abort(error):
                abort_error[0] = error
                aborted.set()  # Set before close: connect may still be blocked without a socket.
                close_connection()

            def worker():
                try:
                    check_active()
                    connection.connect()
                    check_active()  # A delayed connect must not send after cancellation/deadline.
                    connection.request("POST", endpoint, payload, self.headers)
                    check_active()
                    response = connection.getresponse()
                    data = response.read(32 * 1024 * 1024 + 1)
                    check_active()
                    if len(data) > 32 * 1024 * 1024:
                        raise ValueError("Backend response exceeds32MiB")
                    if response.status != 200:
                        # Error details may contain request text; keep logs/CLI failures bounded.
                        raise RuntimeError("Backend HTTP " + str(response.status))
                    decoded = json.loads(data)
                    if not isinstance(decoded, dict):
                        raise ValueError("Backend response must be an object")
                    result.append(decoded)
                except (OSError, ValueError, http.client.HTTPException, RuntimeError, RequestCancelled) as error:
                    result.append(error)
                finally:
                    try:
                        close_connection()
                    finally:
                        done.set()

            thread = threading.Thread(target=worker, daemon=True)
            self._worker = thread
            thread.start()
            try:
                while not done.wait(min(0.1, max(0, deadline - time.monotonic()))):
                    check_cancelled()
                thread.join(timeout=0.05)
                check_cancelled()
                if not result:
                    raise RuntimeError("Backend transport ended without a result")
                if isinstance(result[0], BaseException):
                    raise result[0]
                return result[0]
            except BaseException as error:
                abort(error)
                raise
        finally:
            if thread is not None and thread.is_alive():
                with self.backend.state_lock:
                    self.backend.draining = thread
                _release_after_exit(thread, self.lock, self.backend)
            else:
                self.lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", help="Explicitly override the input request's model alias")
    parser.add_argument("--target", choices=[*MODEL_ALIASES, *MODELS], help="Require the selected artifact identity")
    parser.add_argument("--mode", choices=["direct", "experimental-adaptive"],
                        help="Compatibility mode: direct (default) or selective experimental-adaptive")
    parser.add_argument("--reasoning", choices=REASONING_CHOICES,
                        help="off, selective, or always; on aliases selective; reasoning requires --policy")
    parser.add_argument("--policy", help="Versioned policy id; local policies require --policy-manifest and --runtime-profile")
    parser.add_argument("--policy-manifest", type=Path, help="Trusted local policy manifest (never request-controlled)")
    parser.add_argument("--runtime-profile", help="Supported vision preset name or trusted runtime-contract JSON path")
    parser.add_argument("--mtp", choices=["on", "off"], default="on", help="Must match the adaptive server's MTP setting")
    args = parser.parse_args()
    try:
        body = json.loads(args.input.read_text())
        if args.model:
            if not isinstance(body, dict):
                raise ValueError("Decision input must be an object")
            body["model"] = args.model
        target = selection(args.target)[0] if args.target else None
        response = DecisionPipeline(HTTPTransport(args.base_url), args.mode, args.policy, args.mtp,
                                    target, reasoning=args.reasoning, policy_manifest=args.policy_manifest,
                                    runtime_profile=args.runtime_profile).decide(body)
        note = response.get("winnow", {}).get("adaptive", {}).get("configuration_note")
        if note:
            print(note, file=sys.stderr)
        text = json.dumps(response, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(text)
        else:
            print(text, end="")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, http.client.HTTPException, RequestCancelled) as error:
        parser.exit(1, "Decision failed: " + str(error) + "\n")


if __name__ == "__main__":
    main()
