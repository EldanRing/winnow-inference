"""Exercise interrupted downloads and launcher defaults without GPU/model downloads."""

import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from download import fetch  # noqa: E402
from profiles import resolve_profile  # noqa: E402


class Onboarding(unittest.TestCase):
    def test_interrupted_download_resume_and_range_ignored(self):
        payload = b"download fixture" * 100
        for mode in ["resume", "ignore", "bad-range"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                ranges = []

                class Handler(BaseHTTPRequestHandler):
                    def do_GET(self, mode=mode, ranges=ranges):
                        ranges.append(self.headers.get("Range"))
                        offset = 13 if mode != "ignore" else 0
                        self.send_response(206 if offset else 200)
                        if offset:
                            start = offset + (1 if mode == "bad-range" else 0)
                            self.send_header(
                                "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
                            )
                        self.send_header("Content-Length", str(len(payload) - offset))
                        self.end_headers()
                        self.wfile.write(payload[offset:])

                    def log_message(self, *_args):
                        pass

                server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                try:
                    path = Path(temporary) / "model.gguf"
                    partial = path.with_name(path.name + ".part")
                    partial.write_bytes(payload[:13])
                    artifact = {
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "url": f"http://127.0.0.1:{server.server_port}/model",
                    }
                    if mode == "bad-range":
                        with self.assertRaisesRegex(ValueError, "byte range"):
                            fetch(path, artifact)
                        self.assertFalse(path.exists())
                        self.assertEqual(partial.read_bytes(), payload[:13])
                    else:
                        fetch(path, artifact)
                        self.assertEqual(path.read_bytes(), payload)
                        self.assertFalse(partial.exists())
                        fetch(path, artifact)  # Existing verified file makes no network call.
                    self.assertEqual(ranges, ["bytes=13-"])
                finally:
                    server.shutdown()
                    server.server_close()
                    worker.join()

    def test_invalid_existing_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.gguf"
            path.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                fetch(path, {"bytes": 3, "sha256": hashlib.sha256(b"yes").hexdigest()})
            self.assertEqual(path.read_bytes(), b"bad")

    def test_profile_platform_boundaries(self):
        with (
            patch("profiles.platform.system", return_value="Darwin"),
            patch("profiles.platform.machine", return_value="arm64"),
        ):
            self.assertEqual(resolve_profile()[0], "apple-silicon")
            with self.assertRaises(ValueError):
                resolve_profile("5070ti-64k")
        with patch("profiles.platform.system", return_value="Windows"):
            with self.assertRaisesRegex(ValueError, "Windows"):
                resolve_profile()

    def test_launch_default_files_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "gguf").mkdir()
            model = directory / "gguf/Winnow-12B-Q8_0.gguf"
            projector = directory / "gguf/mmproj-F16.gguf"
            server = directory / "server"
            for path in (model, projector, server):
                path.touch()
            base = [
                sys.executable,
                str(ROOT / "scripts/serve.py"),
                "--model-dir",
                str(directory),
                "--server",
                str(server),
                "--dry-run",
            ]
            result = json.loads(subprocess.check_output(base, text=True))
            self.assertIn(str(projector.resolve()), result["command"])
            self.assertEqual(result["environment"]["WINNOW_MEMORY"], "exclusive")
            override = json.loads(
                subprocess.check_output(
                    base
                    + [
                        "--context",
                        "32768",
                        "--decision-parallel",
                        "2",
                        "--memory",
                        "auto",
                        "--text-only",
                    ],
                    text=True,
                )
            )
            self.assertNotIn("--mmproj", override["command"])
            self.assertEqual(override["environment"]["WINNOW_CONTEXT"], "32768")
            self.assertEqual(override["environment"]["WINNOW_PARALLEL"], "2")
            self.assertEqual(override["environment"]["WINNOW_MEMORY"], "auto")
            custom = json.loads(subprocess.check_output(base + ["--model", str(model)], text=True))
            self.assertNotIn("--mmproj", custom["command"])


if __name__ == "__main__":
    unittest.main()
