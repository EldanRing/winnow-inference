"""Regression checks for private-file exclusion and artifact identity."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from http_client import request_headers
from package_source import allowed, package
from verify_model import verify


class ReleaseTools(unittest.TestCase):
    def test_model_corruption_and_truncation_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.gguf"
            data = b"same-size artifact"
            expected = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            path.write_bytes(data)
            self.assertEqual(verify(path, expected)["sha256"], expected["sha256"])
            path.write_bytes(b"x" * len(data))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify(path, expected)
            path.write_bytes(data[:-1])
            with self.assertRaisesRegex(ValueError, "Size"):
                verify(path, expected)

    def test_package_boundaries(self):
        for path in [
            "private-release/data.jsonl",
            ".release-work/key",
            "scripts/.env",
            "native/model.gguf",
            "../README.md",
            "/README.md",
            "docs/../../secret",
            "docs/__pycache__/module.pyc",
            "models/model.pt",
        ]:
            with self.subTest(path=path):
                self.assertFalse(allowed(path))
        self.assertTrue(allowed("scripts/check.py"))
        self.assertTrue(allowed(".github/workflows/checks.yml"))

    def test_snapshot_omits_history_and_ignored_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repository"
            root.mkdir()

            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL)

            git("init", "-b", "main")
            git("config", "user.name", "Fixture")
            git("config", "user.email", "fixture@example.invalid")
            (root / "private-plan.txt").write_text("historical private fixture")
            git("add", ".")
            git("commit", "-m", "Historical fixture")
            git("rm", "private-plan.txt")
            (root / "README.md").write_text("Public source")
            (root / ".gitignore").write_text("private-release/\n")
            git("add", ".")
            git("commit", "-m", "Public tree")
            (root / "private-release").mkdir()
            (root / "private-release/data.json").write_text("private fixture")
            output = Path(temporary) / "public"
            receipt = package(root, output, init_git=True)
            self.assertEqual(set(receipt["files"]), {"README.md", ".gitignore"})
            count = subprocess.check_output(["git", "rev-list", "--count", "HEAD"], cwd=output)
            self.assertEqual(count.strip(), b"1")
            self.assertFalse((output / "private-release").exists())
            self.assertFalse((output / "private-plan.txt").exists())
            self.assertEqual(
                json.loads(output.with_name("public.receipt.json").read_text()), receipt
            )
            (root / "README.md").write_text("Uncommitted change")
            with self.assertRaisesRegex(ValueError, "Commit"):
                package(root, Path(temporary) / "dirty")

    def test_key_file_is_used_without_changing_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            key = Path(temporary) / "key"
            key.write_text("test-fixture-value\n")
            with patch.dict(os.environ, {"WINNOW_API_KEY_FILE": str(key)}, clear=True):
                self.assertEqual(request_headers()["Authorization"], "Bearer test-fixture-value")
                self.assertNotIn("WINNOW_API_KEY", os.environ)
            key.write_text("first\nsecond")
            with patch.dict(os.environ, {"WINNOW_API_KEY_FILE": str(key)}, clear=True):
                with self.assertRaises(ValueError):
                    request_headers()

    def test_snapshot_rejects_tracked_private_files_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repository"
            root.mkdir()

            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL)

            git("init", "-b", "main")
            git("config", "user.name", "Fixture")
            git("config", "user.email", "fixture@example.invalid")
            (root / "private-release").mkdir()
            (root / "private-release/data.json").write_text("private fixture")
            git("add", ".")
            git("commit", "-m", "Accidentally tracked data fixture")
            with self.assertRaisesRegex(ValueError, "Not approved"):
                package(root, Path(temporary) / "bad-data")
            self.assertFalse((Path(temporary) / "bad-data").exists())
            git("rm", "-r", "private-release")
            (root / "README.md").symlink_to("../outside.txt")
            git("add", ".")
            git("commit", "-m", "Symlink fixture")
            with self.assertRaisesRegex(ValueError, "Not approved"):
                package(root, Path(temporary) / "bad-link")
            self.assertFalse((Path(temporary) / "bad-link").exists())


if __name__ == "__main__":
    unittest.main()
