#!/usr/bin/env python3
"""Download the pinned public GGUF files; resume interrupted transfers and verify SHA256."""

import argparse
import hashlib
import os
import stat
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from verify_model import ROOT, verify


def fetch(path, artifact):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Download destination symlinks are not allowed")
    if path.exists():
        print(f"Verifying existing {path.name}...", flush=True)
        verify(path, artifact)
        return
    partial = path.with_name(path.name + ".part")
    if partial.is_symlink():
        raise ValueError("Partial download symlinks are not allowed")
    # Hold the no-follow descriptor through resume, hash verification and install.
    flags = os.O_RDWR | os.O_NOFOLLOW
    try:
        fd = os.open(partial, flags | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        fd = os.open(partial, flags)
    with os.fdopen(fd, "r+b") as stream:
        inode = os.fstat(stream.fileno())
        if not stat.S_ISREG(inode.st_mode) or inode.st_nlink != 1:
            raise ValueError("Partial download must be a regular unlinked file")
        size = artifact["bytes"]
        offset = inode.st_size
        if offset > size:
            raise ValueError(f"Partial download is too large. Remove {partial} and retry.")
        if offset < size:
            if shutil.disk_usage(path.parent).free < size - offset + 64 * 1024**2:
                raise ValueError(f"Not enough disk space for {path.name}")
            headers = {"User-Agent": "winnow-inference-setup", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            print(f"Downloading {path.name} ({size / 1024**3:.2f} GiB), starting at {offset / 1024**3:.2f} GiB...", flush=True)
            request = urllib.request.Request(artifact["url"], headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or tuple(map(int, match.groups())) != (offset, size - 1, size):
                        raise ValueError("Download server returned an unexpected byte range; retry later.")
                elif response.status == 200:
                    offset = 0
                else:
                    raise ValueError(f"Unexpected download status: {response.status}")
                if not offset:
                    stream.truncate(0)
                stream.seek(offset)
                start = last = time.monotonic()
                received = offset
                while block := response.read(4 * 1024**2):
                    received += len(block)
                    if received > size:
                        raise ValueError("Download exceeds the pinned artifact size.")
                    stream.write(block)
                    now = time.monotonic()
                    if now - last >= 5:
                        rate = (received - offset) / max(now - start, 0.001) / 1024**2
                        print(f"  {100 * received / size:.1f}% — {rate:.1f} MiB/s", flush=True)
                        last = now
        stream.flush()
        if os.fstat(stream.fileno()).st_size != size:
            raise ValueError("Downloaded size does not match pinned artifact")
        stream.seek(0)
        digest = hashlib.sha256()
        while block := stream.read(8 * 1024**2):
            digest.update(block)
        if digest.hexdigest() != artifact["sha256"]:
            raise ValueError("SHA256 mismatch for downloaded artifact")
        current = partial.lstat()
        if (current.st_dev, current.st_ino) != (inode.st_dev, inode.st_ino):
            raise ValueError("Partial download path changed during transfer")
        # Never replace a concurrent final file, and never follow a final symlink.
        os.link(partial, path, follow_symlinks=False)
        installed = path.lstat()
        if (installed.st_dev, installed.st_ino) != (inode.st_dev, inode.st_ino):
            path.unlink()
            raise ValueError("Partial download path changed during installation")
        if partial.lstat().st_ino == inode.st_ino:
            partial.unlink()
    print(f"  Ready: {path}", flush=True)


def download_models(model_dir, text_only=False, manifest=ROOT / "manifests/models.json"):
    release = json.loads(Path(manifest).read_text())["release"]
    for key in ["model"] if text_only else ["model", "projector"]:
        item = release[key]
        relative = Path(item["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid artifact path in model manifest")
        fetch(Path(model_dir) / relative, item)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=ROOT / "models")
    p.add_argument("--text-only", action="store_true", help="Skip the vision projector")
    a = p.parse_args()
    try:
        download_models(a.model_dir, a.text_only)
    except urllib.error.HTTPError as error:
        print(
            f"Download failed (HTTP {error.code}). Check your connection and the model page at https://huggingface.co/EldanRing/Winnow-12B. Rerun to resume.",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError) as error:
        print(
            f"Download failed: {error}\nRerun to resume an interrupted transfer. For a checksum mismatch, remove the named corrupt file before retrying.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\nDownload paused. Rerun the same command to resume.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
