#!/usr/bin/env python3
"""Package an arm64 Metal executable with public source after checking portability."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path

from package_source import ROOT, package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/winnow-macos-arm64")
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Build and package on an Apple Silicon Mac")
    server = ROOT / ".build/bin/winnow-server"
    dependencies = subprocess.check_output(["otool", "-L", str(server)], text=True)
    for line in dependencies.splitlines()[1:]:
        dependency = line.strip().split(" (", 1)[0]
        if not dependency.startswith(("/usr/lib/", "/System/Library/")):
            raise ValueError(
                f"Non-portable dependency: {dependency}. Use a fresh build with --static-openssl."
            )
    receipt = package(ROOT, args.output.resolve())
    binary = args.output / ".build/bin/winnow-server"
    binary.parent.mkdir(parents=True)
    shutil.copy2(server, binary)
    # OpenSSL is linked statically in this distribution; preserve its license.
    prefix = subprocess.check_output(["brew", "--prefix", "openssl@3"], text=True).strip()
    license_path = Path(prefix) / "share/doc/openssl@3/LICENSE.txt"
    if not license_path.exists():
        candidates = list(Path(prefix).glob("**/LICENSE*"))
        if not candidates:
            raise FileNotFoundError("Cannot locate the static OpenSSL license")
        license_path = candidates[0]
    shutil.copy2(license_path, args.output / "third_party/OpenSSL-LICENSE.txt")
    for path in (ROOT / ".runtime/llama.cpp/vendor").rglob("*"):
        if path.is_file() and any(word in path.name.upper() for word in ("LICENSE", "COPYING")):
            target = (
                args.output
                / "third_party/vendor"
                / path.relative_to(ROOT / ".runtime/llama.cpp/vendor")
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    (args.output / "manifests/binary.json").write_text(
        json.dumps(
            {
                "source_commit": receipt["source_commit"],
                "platform": "macOS arm64; Metal + Accelerate",
                "built_on": platform.mac_ver()[0],
                "server_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "dependencies": dependencies,
                "validation": "native unit checks and executable startup; no model inference in CI",
            },
            indent=2,
        )
        + "\n"
    )
    (args.output / "START-HERE.txt").write_text(
        "Winnow for Apple Silicon\n\n"
        "Requires macOS 15 or newer and Python 3.10+.\n"
        "From this folder in Terminal:\n"
        "  python3 scripts/setup.py --skip-build --profile apple-silicon\n"
        "  python3 scripts/serve.py --profile apple-silicon\n\n"
        "In a second terminal in this folder:\n"
        "  python3 examples/client.py\n\n"
        "Model weights download separately (about 12.85 GB).\n"
        "This is an unsigned command-line build, not a notarized Mac application.\n"
    )
    archive = args.output.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(args.output, arcname=args.output.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_name(archive.name + ".sha256").write_text(f"{digest}  {archive.name}\n")
    print(archive)


if __name__ == "__main__":
    main()
