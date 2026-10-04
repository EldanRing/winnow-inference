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


def runtime_matches(source, lock):
    changed = subprocess.check_output(
        ["git", "-C", str(source), "diff", "HEAD", "--name-only"], text=True
    ).splitlines()
    return set(changed) == set(lock["source_sha256"]) and all(
        (source / name).is_file()
        and hashlib.sha256((source / name).read_bytes()).hexdigest() == value
        for name, value in lock["source_sha256"].items()
    )


def runtime_mismatches(source, lock):
    changed = set(subprocess.check_output(
        ["git", "-C", str(source), "diff", "HEAD", "--name-only"], text=True
    ).splitlines())
    expected_files = set(lock["source_sha256"])
    details = []
    for name in sorted(changed - expected_files):
        details.append(f"Unexpected changed file: {name}")
    for name in sorted(expected_files - changed):
        details.append(f"Expected patched file is unchanged: {name}")
    for name, expected in sorted(lock["source_sha256"].items()):
        path = source / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        if actual != expected:
            details.append(f"{name}: expected SHA256 {expected}; actual {actual}")
    return details


def prepare_runtime(source, lock, root=ROOT):
    for item in lock["patches"]:
        patch = root / item["file"]
        if hashlib.sha256(patch.read_bytes()).hexdigest() != item["sha256"]:
            raise SystemExit(f"Patch does not match runtime lock: {patch.name}")
    # Later patches may overlap earlier ones. Exact final hashes are the source of
    # truth; individually reverse-checking earlier patches is not sufficient.
    if normalize_verified_crlf(source, lock) or runtime_matches(source, lock):
        return
    deferred = []
    for item in lock["patches"]:
        patch = root / item["file"]
        check = subprocess.run(
            ["git", "-C", str(source), "apply", "--check", str(patch)], capture_output=True
        )
        if check.returncode == 0:
            run("git", "-C", str(source), "apply", str(patch))
        elif subprocess.run(
            ["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)],
            capture_output=True,
        ).returncode:
            # An already-applied patch may be obscured by a later overlapping patch.
            # Continue to new suffix patches, then require the exact complete locked tree.
            deferred.append(patch.name)
    if not normalize_verified_crlf(source, lock) and not runtime_matches(source, lock):
        raise SystemExit("Runtime contains changes outside its verified lock\n"
                         + ("Unresolved patches: " + ", ".join(deferred) + "\n" if deferred else "")
                         + "\n".join(runtime_mismatches(source, lock))
                         + "\nKeep this runtime for inspection; use a separate clean source checkout."
                         + " Do not bypass the lock or force-apply patches.")


def normalize_verified_crlf(source, lock):
    """Repair only an entire locked tree whose sole byte difference is CRLF."""
    changed = set(subprocess.check_output(
        ["git", "-C", str(source), "diff", "HEAD", "--name-only"], text=True
    ).splitlines())
    if changed != set(lock["source_sha256"]):
        return False
    updates = {}
    for name, expected in lock["source_sha256"].items():
        path = source / name
        if not path.is_file() or path.is_symlink():
            return False
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() == expected:
            continue
        canonical = raw.replace(b"\r\n", b"\n")
        if hashlib.sha256(canonical).hexdigest() != expected:
            return False
        updates[path] = canonical
    for path, canonical in updates.items():
        path.write_bytes(canonical)
        print("Restored locked LF bytes:", path.relative_to(source))
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--cuda-arch", default="native")
    p.add_argument("--cuda-compiler", type=Path, help="Existing nvcc executable; useful with multiple toolkits")
    p.add_argument(
        "--backend",
        choices=["auto", "cuda", "metal", "cpu"],
        default="auto",
        help="CPU is for compilation/CI checks, not a supported serving profile",
    )
    p.add_argument("--build-dir", type=Path, default=ROOT / ".build")
    p.add_argument("--sanitize-build-paths", action="store_true",
                   help="Map this checkout's compiler file/debug paths to relative paths in release binaries")
    p.add_argument(
        "--static-openssl",
        action="store_true",
        help="Link OpenSSL statically for portable native archives",
    )
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
            "-c", "core.autocrlf=false",
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
    prepare_runtime(source, lock)
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
        if a.cuda_compiler:
            if not a.cuda_compiler.is_file():
                p.error("CUDA compiler does not exist")
            args.append(f"-DCMAKE_CUDA_COMPILER={a.cuda_compiler.resolve()}")
            # ggml finds CUDAToolkit before enabling CUDA; pin headers/libraries too.
            args.append(f"-DCUDAToolkit_ROOT={a.cuda_compiler.resolve().parents[1]}")
    if a.sanitize_build_paths:
        mapping = f"-ffile-prefix-map={ROOT.resolve()}=."
        args += [f"-DCMAKE_C_FLAGS={mapping}", f"-DCMAKE_CXX_FLAGS={mapping}"]
        if backend == "cuda":
            args.append(f"-DCMAKE_CUDA_FLAGS=-Xcompiler={mapping}")
    if a.static_openssl:
        args.append("-DOPENSSL_USE_STATIC_LIBS=TRUE")
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
