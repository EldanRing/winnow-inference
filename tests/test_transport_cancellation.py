"""Delayed-connect/send cancellation without sockets, including cross-request ownership."""
import http.client
import io
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from adaptive_policy import DecisionPipeline
from decision_client import HTTPTransport
from reasoning_contract import BackendUnavailable, RequestCancelled


class RecordingSocket:
    def __init__(self, wire):
        self.wire = wire
        self.closed = False

    def sendall(self, data):
        if self.closed:
            raise OSError("Socket closed")
        self.wire.append(data)

    def shutdown(self, how):
        self.closed = True

    def close(self):
        self.closed = True

    def makefile(self, *args, **kwargs):
        return io.BytesIO(b'HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n{"ok":true}')


class CancellationOwnership(unittest.TestCase):
    def test_delayed_connect_cannot_send_after_cancel_or_deadline_and_keeps_backend_owned(self):
        for timeout in (False, True):
            entered, resume, cancelled = threading.Event(), threading.Event(), threading.Event()
            wire, errors = [], []
            shared_lock = threading.Lock()
            origin = "http://127.0.0.1:" + str(19101 + timeout)

            class ShortTransport(HTTPTransport):
                def post(self, endpoint, body, seconds):
                    return super().post(endpoint, body, 0.05 if timeout else seconds)

            first = ShortTransport(origin, {"Authorization": "first"}, cancelled.is_set)
            second = HTTPTransport(origin, {"Authorization": "second"})

            def delayed_connect(connection):
                entered.set()
                if not resume.wait(3):
                    raise TimeoutError("Test did not release connect")
                connection.sock = RecordingSocket(wire)

            def run():
                try:
                    DecisionPipeline(first, request_lock=shared_lock, cancelled=cancelled.is_set).decide({"first": True})
                except BaseException as error:
                    errors.append(error)

            with self.subTest(timeout=timeout), patch.object(http.client.HTTPConnection, "connect", delayed_connect):
                caller = threading.Thread(target=run)
                caller.start()
                try:
                    self.assertTrue(entered.wait(1))
                    if not timeout:
                        cancelled.set()
                    caller.join(0.5)
                    self.assertFalse(caller.is_alive(), "Caller must return while connect is still stuck")
                    self.assertEqual(len(errors), 1)
                    self.assertIsInstance(errors[0], TimeoutError if timeout else RequestCancelled)
                    self.assertEqual(wire, [])
                    self.assertTrue(first._worker.is_alive())
                    self.assertTrue(shared_lock.locked(), "Pipeline must retain ownership until worker exit")
                    self.assertTrue(first.lock.locked(), "Transport must retain ownership until worker exit")
                    # A new request-scoped instance must not overlap, even with a different pipeline lock.
                    with self.assertRaises(BackendUnavailable):
                        DecisionPipeline(second).decide({"second": True})
                    with self.assertRaises(BackendUnavailable):
                        DecisionPipeline(second, request_lock=shared_lock).decide({"second": True})
                finally:
                    resume.set()
                    caller.join(1)
                    if first._worker:
                        first._worker.join(1)
                self.assertFalse(first._worker.is_alive())
                self.assertTrue(shared_lock.acquire(timeout=1))
                shared_lock.release()
                self.assertEqual(wire, [], "Delayed connect must not submit the cancelled request")
                self.assertTrue(first.lock.acquire(timeout=1))
                first.lock.release()
                self.assertEqual(DecisionPipeline(second, request_lock=shared_lock).decide({"second": True}), {"ok": True})
                self.assertIn(b"second", b"".join(wire))
                self.assertNotIn(b"first", b"".join(wire))

    def test_cancel_between_connect_and_send_never_reconnects_or_writes(self):
        # Keep the real request/send path, but pause request() immediately before it uses send().
        for timeout in (False, True):
            entered, resume, cancelled = threading.Event(), threading.Event(), threading.Event()
            wire, connects, errors = [], [], []
            origin = "http://127.0.0.1:" + str(19111 + timeout)
            transport = HTTPTransport(origin, {}, cancelled.is_set)
            original_request = http.client.HTTPConnection.request

            def connect(connection):
                connects.append(True)
                connection.sock = RecordingSocket(wire)

            def delayed_request(connection, *args, **kwargs):
                entered.set()
                if not resume.wait(3):
                    raise TimeoutError("Test did not release request")
                return original_request(connection, *args, **kwargs)

            def run():
                try:
                    transport.post("/v1/systemone", {}, 0.05 if timeout else 2)
                except BaseException as error:
                    errors.append(error)

            with self.subTest(timeout=timeout), patch.object(http.client.HTTPConnection, "connect", connect), patch.object(http.client.HTTPConnection, "request", delayed_request):
                caller = threading.Thread(target=run)
                caller.start()
                try:
                    self.assertTrue(entered.wait(1))
                    if not timeout:
                        cancelled.set()
                    caller.join(0.5)
                    self.assertFalse(caller.is_alive())
                    self.assertIsInstance(errors[0], TimeoutError if timeout else RequestCancelled)
                    self.assertTrue(transport.lock.locked())
                finally:
                    resume.set()
                    caller.join(1)
                    if transport._worker:
                        transport._worker.join(1)
                self.assertEqual(wire, [])
                self.assertEqual(len(connects), 1)
                self.assertTrue(transport.lock.acquire(timeout=1))
                transport.lock.release()

    def test_same_origin_uses_shared_guard_but_different_origins_do_not(self):
        first = HTTPTransport("http://localhost", {})
        same = HTTPTransport("http://LOCALHOST:80", {})
        other = HTTPTransport("https://localhost", {})
        self.assertIs(first.backend, same.backend)
        self.assertIsNot(first.backend, other.backend)


if __name__ == "__main__":
    unittest.main()
