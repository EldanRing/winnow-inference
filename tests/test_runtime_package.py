"""Check private runtime package boundaries and standalone client entrypoints."""
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from package_local_candidate import ROOT, package, sha


class RuntimePackage(unittest.TestCase):
    def test_runtime_is_complete_and_has_no_development_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "binary"
            binary.write_bytes(b"fixture")
            output = Path(temporary) / "candidate"
            with patch("package_local_candidate.subprocess.check_output", side_effect=[sha(ROOT / "runtime.lock.json"), ""]):
                receipt = package(binary, output)
            self.assertEqual(receipt["binary_sha256"], hashlib.sha256(b"fixture").hexdigest())
            for name, expected in receipt["files"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected["sha256"])
                self.assertFalse(any(part in {"models", "results", ".runtime", ".git"} for part in Path(name).parts))
            for filename, copyright in [
                ("cpp-httplib-LICENSE.txt", "2017 yhirose"),
                ("nlohmann-json-LICENSE.txt", "2013-2025 Niels Lohmann"),
                ("rotate-bits-LICENSE.txt", "2021 William Casarin"),
            ]:
                notice = (output / "third_party" / filename).read_text()
                self.assertIn(copyright, notice)
                self.assertIn("Permission is hereby granted", notice)
            for entrypoint in ["winnow", "winnow-serve", "winnow-decide"]:
                result = subprocess.run([str(output / "bin" / entrypoint), "--help"], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            with self.assertRaises(FileExistsError):
                package(binary, output)

    def test_missing_dependencies_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "binary"
            binary.write_bytes(b"fixture")
            output = Path(temporary) / "candidate"
            with patch("package_local_candidate.subprocess.check_output", side_effect=[sha(ROOT / "runtime.lock.json"), "libbad => not found"]):
                with self.assertRaisesRegex(ValueError, "dependency is missing"):
                    package(binary, output)
            self.assertFalse(output.exists())

    def test_previous_runtime_is_rejected_before_package_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "binary"
            binary.write_bytes(b"old fixture")
            output = Path(temporary) / "candidate"
            with patch("package_local_candidate.subprocess.check_output", return_value="7a2ab005c30b8c4cc8d7e2f5e9512581e851c7d072e55959d7638f21f1866354"):
                with self.assertRaisesRegex(ValueError, "runtime lock"):
                    package(binary, output)
            self.assertFalse(output.exists())

    def test_embedded_private_build_path_rejects_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "binary"
            binary.write_bytes(b"fixture")
            output = Path(temporary) / "candidate"
            with patch("package_local_candidate.subprocess.check_output", return_value="/home/example/private-build/source.cpp"):
                with self.assertRaisesRegex(ValueError, "private build paths"):
                    package(binary, output)
            self.assertFalse(output.exists())
