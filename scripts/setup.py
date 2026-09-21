#!/usr/bin/env python3
"""Check prerequisites, download verified release weights and build Winnow."""

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from download import download_models
from profiles import PROFILES, resolve_profile
from verify_model import ROOT


def output(command):
    try:
        return subprocess.check_output(
            command, stderr=subprocess.STDOUT, text=True, timeout=15
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def prerequisites(profile, skip_build=False):
    errors = []
    if sys.version_info < (3, 10):  # noqa: UP036 -- show an actionable error on old Python
        errors.append("Python 3.10 or newer is required.")
    if not skip_build:
        for tool in ["git", "cmake", "c++"]:
            if not shutil.which(tool):
                errors.append(f"Missing command: {tool}")
        cmake = re.search(r"cmake version (\d+)\.(\d+)", output(["cmake", "--version"]))
        if cmake and tuple(map(int, cmake.groups())) < (3, 24):
            errors.append("CMake 3.24 or newer is required.")
        if profile == "apple-silicon":
            if not output(["xcode-select", "-p"]):
                errors.append("Install Xcode command-line tools: xcode-select --install")
            if not os.environ.get("OPENSSL_ROOT_DIR"):
                prefix = output(["brew", "--prefix", "openssl@3"])
                if prefix and Path(prefix, "include/openssl/ssl.h").is_file():
                    os.environ["OPENSSL_ROOT_DIR"] = prefix
        else:
            nvcc = os.environ.get("CUDACXX") or shutil.which("nvcc")
            if not nvcc or not output([nvcc, "--version"]):
                errors.append(
                    "CUDA toolkit compiler nvcc is missing. Install the NVIDIA CUDA toolkit; a driver alone is not enough. Set CUDACXX if nvcc is outside PATH."
                )
        include = []
        if os.environ.get("OPENSSL_ROOT_DIR"):
            include = ["-I", str(Path(os.environ["OPENSSL_ROOT_DIR"]) / "include")]
        if shutil.which("c++"):
            check = subprocess.run(
                ["c++", *include, "-x", "c++", "-E", "-"],
                input="#include <openssl/ssl.h>\n",
                text=True,
                capture_output=True,
            )
            if check.returncode:
                errors.append(
                    "OpenSSL development headers are missing (libssl-dev on Ubuntu; openssl@3 on Homebrew)."
                )
    if profile != "apple-silicon":
        gpu = output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
                "--id=0",
            ]
        )
        if not gpu:
            errors.append(
                "No accessible NVIDIA GPU/driver. Check nvidia-smi; this native setup requires a GPU."
            )
        else:
            print("GPU:", gpu, flush=True)
            fields = gpu.splitlines()[0].split(",")
            if len(fields) == 3:
                try:
                    if float(fields[1]) < 16000:
                        errors.append(
                            "The 64K CUDA profile needs a 16 GB class GPU. This card needs a separately tuned profile; see docs/INSTALL.md."
                        )
                except ValueError:
                    pass
                if not skip_build and (nvcc := (os.environ.get("CUDACXX") or shutil.which("nvcc"))):
                    arch = "compute_" + fields[2].strip().replace(".", "")
                    if arch not in output([nvcc, "--list-gpu-arch"]).split():
                        errors.append(
                            f"Your CUDA toolkit cannot target {arch}. Install a toolkit supporting this GPU (Blackwell requires CUDA 12.8+)."
                        )
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=["auto", *PROFILES], default="auto")
    p.add_argument("--model-dir", type=Path, default=ROOT / "models")
    p.add_argument("--text-only", action="store_true")
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Only check prerequisites; do not download or build",
    )
    p.add_argument(
        "--download-only",
        action="store_true",
        help="Download weights without requiring CUDA/build tools (for Docker)",
    )
    p.add_argument("--skip-build", action="store_true", help="Use an already built native server")
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    if a.jobs < 1:
        p.error("--jobs must be positive")
    if a.check_only and a.download_only:
        p.error("Choose --check-only or --download-only")
    try:
        selected = None
        if not a.download_only:
            selected, _ = resolve_profile(a.profile)
            print(f"Profile: {selected}", flush=True)
            errors = prerequisites(selected, a.skip_build)
            if a.skip_build and not (ROOT / ".build/bin/winnow-server").is_file():
                errors.append("No server found. Omit --skip-build to build it.")
            if errors:
                print("\nBefore setup can continue:", file=sys.stderr)
                for error in errors:
                    print(" - " + error, file=sys.stderr)
                print(
                    "\nUbuntu: sudo apt-get install build-essential cmake git python3 libssl-dev\nmacOS: xcode-select --install; brew install cmake openssl@3\nCUDA installation: https://developer.nvidia.com/cuda-downloads\nFull instructions: docs/INSTALL.md\nNo system packages were changed.",
                    file=sys.stderr,
                )
                return 1
            print("Prerequisites passed.", flush=True)
            if a.check_only:
                return 0
        download_models(a.model_dir, a.text_only)
        if a.download_only:
            print("Weights are ready. Follow the container instructions in docs/INSTALL.md.")
            return 0
        if not a.skip_build:
            print("Building Winnow (first build can take several minutes)...", flush=True)
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/build.py"), "--jobs", str(a.jobs)], check=True
            )
        command = ["python3", "scripts/serve.py", "--profile", selected]
        if a.model_dir.resolve() != ROOT / "models":
            command.extend(["--model-dir", str(a.model_dir.resolve())])
        if a.text_only:
            command.append("--text-only")
        print("\nReady. Start the server from the repository directory:\n  " + shlex.join(command))
        print(
            "In a second terminal, try:\n  python3 examples/client.py\nThe example prints a typed decision and a regular chat response."
        )
        return 0
    except KeyboardInterrupt:
        print(
            "\nSetup interrupted. Rerun the same command; completed downloads are reused and partial downloads resume.",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(
            f"Setup could not finish: {error}\nSee docs/INSTALL.md. Rerun after correcting the issue; downloads are verified and reused.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
