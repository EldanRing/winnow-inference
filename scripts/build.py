#!/usr/bin/env python3
"""Build the pinned Winnow server on CUDA/Linux or Metal/macOS."""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--cuda-arch", default="native")
    p.add_argument(
        "--backend",
        choices=["auto", "cuda", "metal", "cpu"],
        default="auto",
        help="CPU is for compilation/CI checks, not a supported serving profile",
    )
    p.add_argument("--build-dir", type=Path, default=ROOT / ".build")
    a = p.parse_args()
    if a.jobs < 1:
        p.error("jobs must be positive")
    backend = a.backend
    if backend == "auto":
        backend = "metal" if platform.system() == "Darwin" else "cuda"
    if backend == "metal" and platform.system() != "Darwin":
        p.error("Metal requires macOS")
    build_dir = a.build_dir.resolve()
    lock = json.loads((ROOT / "runtime.lock.json").read_text())
    source = ROOT / ".runtime/llama.cpp"
    if not source.exists():
        source.parent.mkdir(exist_ok=True)
        run(
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            lock["repository"],
            str(source),
        )
        run("git", "-C", str(source), "checkout", "--detach", lock["commit"])
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != lock["commit"]:
        raise SystemExit("Unexpected runtime revision; use a separate clean source checkout")
    for item in lock["patches"]:
        patch = ROOT / item["file"]
        if hashlib.sha256(patch.read_bytes()).hexdigest() != item["sha256"]:
            raise SystemExit(f"Patch does not match runtime lock: {patch.name}")
        check = subprocess.run(
            ["git", "-C", str(source), "apply", "--check", str(patch)],
            capture_output=True,
        )
        if check.returncode == 0:
            run("git", "-C", str(source), "apply", str(patch))
        elif subprocess.run(
            ["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)],
            capture_output=True,
        ).returncode:
            raise SystemExit(f"Cannot apply patch: {patch.name}")
    changed = subprocess.check_output(
        ["git", "-C", str(source), "diff", "HEAD", "--name-only"], text=True
    ).splitlines()
    if set(changed) != set(lock["source_sha256"]) or any(
        hashlib.sha256((source / k).read_bytes()).hexdigest() != v
        for k, v in lock["source_sha256"].items()
    ):
        raise SystemExit("Runtime contains changes outside its verified lock")
    args = [
        "cmake",
        "-S",
        str(ROOT),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DGGML_CUDA={'ON' if backend == 'cuda' else 'OFF'}",
        f"-DGGML_METAL={'ON' if backend == 'metal' else 'OFF'}",
    ]
    if backend == "cuda":
        args.append(f"-DCMAKE_CUDA_ARCHITECTURES={a.cuda_arch}")
    run(*args)
    run(
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "llama-server",
        "winnow-unit",
        "-j",
        str(a.jobs),
    )
    run("ctest", "--test-dir", str(build_dir), "--output-on-failure")
    print("Ready:", build_dir / "bin/winnow-server")


if __name__ == "__main__":
    main()
