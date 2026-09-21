#!/usr/bin/env python3
"""Verify GGUF files against a public model manifest without loading the model."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(path, expected):
    if path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Size mismatch: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path.name}")
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/models.json")
    parser.add_argument("--artifact", choices=["release", "tested_reference"], default="release")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path)
    args = parser.parse_args()
    artifact = json.loads(args.manifest.read_text()).get(args.artifact)
    if not artifact:
        parser.error("The selected artifact is not published in this manifest yet")
    results = [verify(args.model, artifact["model"])]
    if args.mmproj:
        results.append(verify(args.mmproj, artifact["projector"]))
    print(json.dumps({"artifact": args.artifact, "verified": results}, indent=2))


if __name__ == "__main__":
    main()
