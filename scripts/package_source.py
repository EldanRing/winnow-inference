#!/usr/bin/env python3
"""Export the committed public source tree without development Git history."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT_FILES = {
    ".clang-format",
    ".dockerignore",
    ".gitattributes",
    ".gitignore",
    "CMakeLists.txt",
    "Dockerfile",
    "LICENSE",
    "README.md",
    "ruff.toml",
    "runtime.lock.json",
}
PUBLIC_DIRECTORIES = {
    ".github",
    "docs",
    "examples",
    "manifests",
    "native",
    "patches",
    "scripts",
    "tests",
    "third_party",
}


def allowed(name):
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and (
            name in PUBLIC_ROOT_FILES
            or (len(path.parts) > 1 and path.parts[0] in PUBLIC_DIRECTORIES)
        )
        and not any(
            part.startswith(".env")
            or part
            in {"__pycache__", ".git", "private-release", ".release-work", "models", "results"}
            for part in path.parts
        )
        and path.suffix not in {".gguf", ".safetensors", ".pt", ".pyc"}
    )


def package(root, output, init_git=False):
    if output.exists():
        raise FileExistsError(
            "Choose a new output directory; existing snapshots are never overwritten"
        )
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root
    ):
        raise ValueError("Commit the intended source changes before packaging")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    data = subprocess.check_output(["git", "archive", "--format=tar", "HEAD"], cwd=root)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        members = [member for member in archive if not member.isdir()]
        for member in members:
            if not member.isfile() or not allowed(member.name):
                raise ValueError(f"Not approved for the public source package: {member.name}")
        output.mkdir(parents=True)
        for member in members:
            payload = archive.extractfile(member).read()
            path = output / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            path.chmod(0o755 if member.mode & 0o111 else 0o644)
            files[member.name] = hashlib.sha256(payload).hexdigest()
    if init_git:
        subprocess.run(["git", "init", "-b", "main", str(output)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(output), "add", "."], check=True)
        # Keep the local source repository's author identity; do not alter global settings.
        config = []
        for key in ("user.name", "user.email"):
            value = subprocess.check_output(
                ["git", "config", "--get", key], cwd=root, text=True
            ).strip()
            config += ["-c", f"{key}={value}"]
        subprocess.run(
            ["git", *config, "-C", str(output), "commit", "-m", "Prepare Winnow inference release"],
            check=True,
            capture_output=True,
        )
    receipt = {"source_commit": revision, "files": files, "new_git_history": init_git}
    output.with_name(output.name + ".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--init-git", action="store_true", help="Create one fresh local commit, no remote"
    )
    args = parser.parse_args()
    receipt = package(ROOT, args.output.resolve(), args.init_git)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "source_commit": receipt["source_commit"],
                "files": len(receipt["files"]),
                "new_git_history": receipt["new_git_history"],
            }
        )
    )


if __name__ == "__main__":
    main()
