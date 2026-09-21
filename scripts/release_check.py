#!/usr/bin/env python3
"""Run a short release smoke check with owned local servers; no downloads or paid APIs."""

import argparse
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", type=Path, default=ROOT / ".build/bin/winnow-server")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--decision-parallel", type=int)
    parser.add_argument("--memory", choices=["auto", "exclusive"], default="auto")
    parser.add_argument(
        "--long-context",
        action="store_true",
        help="Also fill context; longer than the default smoke",
    )
    parser.add_argument(
        "--lifecycle",
        action="store_true",
        help="Add cancellation/long mixed-load checks; needs >=32K",
    )
    args = parser.parse_args()
    if args.lifecycle and args.context < 32768:
        parser.error("Lifecycle fixture requires at least 32768 context positions")
    for path in (args.model, args.server, args.mmproj):
        if path and not path.is_file():
            parser.error(f"File not found: {path}")
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {
        "model": {"name": args.model.name, "sha256": file_hash(args.model)},
        "server_sha256": file_hash(args.server),
        "runtime": json.loads((ROOT / "runtime.lock.json").read_text()),
        "context": args.context,
        "decision_parallel": args.decision_parallel,
        "memory": args.memory,
        "purpose": "Functional runtime smoke, not model-quality benchmark",
        "output_allowance": 4096,
    }
    if args.mmproj:
        identity["projector"] = {"name": args.mmproj.name, "sha256": file_hash(args.mmproj)}
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    steps = []
    start = time.monotonic()
    process = None
    log = None
    with tempfile.TemporaryDirectory(prefix="winnow-check-") as temporary:
        key = Path(temporary) / "api-key"
        key.write_text(secrets.token_urlsafe(32))
        key.chmod(0o600)
        env = os.environ.copy()
        env["WINNOW_API_KEY_FILE"] = str(key)

        def run(script, name, *options):
            command = [
                sys.executable,
                str(ROOT / "scripts" / script),
                "--url",
                url,
                "--output",
                str(args.output / name),
                *map(str, options),
            ]
            then = time.monotonic()
            subprocess.run(command, check=True, env=env)
            steps.append({"name": name, "seconds": time.monotonic() - then})

        try:
            for head in ("selected", "full"):
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                url = f"http://127.0.0.1:{port}"
                command = [
                    sys.executable,
                    str(ROOT / "scripts/serve.py"),
                    "--model",
                    str(args.model.resolve()),
                    "--server",
                    str(args.server.resolve()),
                    "--context",
                    str(args.context),
                    "--memory",
                    args.memory,
                    "--head",
                    head,
                    "--port",
                    str(port),
                    "--gpu",
                    args.gpu,
                    "--api-key-file",
                    str(key),
                ]
                if args.mmproj:
                    command += ["--mmproj", str(args.mmproj.resolve())]
                if args.decision_parallel is not None:
                    command += ["--decision-parallel", str(args.decision_parallel)]
                log = (args.output / f"{head}-server.log").open("w")
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"Owned {head} server exited; see its log")
                    try:
                        with urllib.request.urlopen(url + "/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (OSError, urllib.error.URLError):
                        pass
                    time.sleep(0.5)
                else:
                    raise TimeoutError("Server readiness exceeded 180 seconds")
                if head == "selected":
                    options = ["--vision"] if args.mmproj else []
                    if args.long_context:
                        options += ["--long-context", str(args.context), "--image-size", "1536"]
                    run("check.py", "functional", *options)
                    run("check_http.py", "http")
                    if args.lifecycle:
                        run("lifecycle.py", "lifecycle", *(["--vision"] if args.mmproj else []))
                    run("parity.py", "selected")
                else:
                    run("parity.py", "full", "--compare", args.output / "selected")
                process.terminate()
                process.wait(timeout=30)
                process = None
                log.close()
                log = None
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if log is not None:
                log.close()
    summary = {"passed": True, "steps": steps, "seconds": time.monotonic() - start}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
