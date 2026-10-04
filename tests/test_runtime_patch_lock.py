"""Overlapping pinned patches must be reproducible and reject dirty/mismatched source."""

import difflib
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build import prepare_runtime  # noqa: E402


class RuntimePatchLock(unittest.TestCase):
    def test_overlapping_series_is_idempotent_and_keeps_lock_enforcement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "runtime"
            source.mkdir()
            subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
            original, middle, final = "one\ntwo\nthree\n", "one\nsecond\nthree\n", "one\nfinal\nthree\n"
            path = source / "sample.txt"
            path.write_text(original)
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run(["git", "-C", str(source), "-c", "user.name=Fixture",
                            "-c", "user.email=fixture@example.invalid", "commit", "-m", "Fixture"],
                           check=True, capture_output=True)
            patches = []
            for i, before, after in [(1, original, middle), (2, middle, final)]:
                patch = root / ("patch" + str(i) + ".patch")
                patch.write_text("".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                            fromfile="a/sample.txt", tofile="b/sample.txt")))
                patches.append({"file": patch.name, "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()})
            lock = {"patches": patches, "source_sha256": {"sample.txt": hashlib.sha256(final.encode()).hexdigest()}}
            prepare_runtime(source, lock, root)
            self.assertEqual(path.read_text(), final)
            prepare_runtime(source, lock, root)
            self.assertEqual(path.read_text(), final)
            path.write_text("one\nunauthorized\nthree\n")
            with self.assertRaises(SystemExit):
                prepare_runtime(source, lock, root)
            path.write_text(final)
            (root / "patch1.patch").write_text("wrong patch")
            with self.assertRaisesRegex(SystemExit, "Patch does not match"):
                prepare_runtime(source, lock, root)


if __name__ == "__main__":
    unittest.main()
