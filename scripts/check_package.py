#!/usr/bin/env python3
"""Reject non-public files in an exported source tree or Docker source stage."""

import argparse
import json
from pathlib import Path

from package_source import allowed


def check(directory):
    if not directory.is_dir():
        raise ValueError("Package directory does not exist")
    files = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink: {path}")
        if path.is_file():
            name = path.relative_to(directory).as_posix()
            if not allowed(name):
                raise ValueError(f"Non-public packaged file: {name}")
            files.append(name)
    if not {"CMakeLists.txt", "runtime.lock.json", "native/engine.h", "scripts/serve.py"} <= set(
        files
    ):
        raise ValueError("Incomplete inference source package")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"public_files": len(check(args.directory)), "passed": True}))


if __name__ == "__main__":
    main()
