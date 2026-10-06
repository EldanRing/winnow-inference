#!/usr/bin/env python3
"""Prepare a private Linux/CUDA runtime candidate, with dependency and file receipts."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from package_source import PUBLIC_DOC_FILES

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = ["scripts/serve.py", "scripts/profiles.py", "scripts/verify_model.py",
                 "scripts/winnow.py", "scripts/assets.py", "scripts/launch_options.py", "scripts/download.py",
                 "manifests/release-assets-v1.json", "docs/QUICKSTART.md", "docs/INSTALL.md", "docs/API.md", "docs/VALIDATION.md",
                 "examples/adaptive-decision.json", "examples/decisions.json", "examples/client.py",
                 "third_party/gemma-assistants-LICENSE-APACHE-2.0.txt", "third_party/gemma-assistants-NOTICE.txt",
                 "scripts/adaptive_policy.py", "scripts/reasoning_contract.py", "scripts/decision_client.py", "scripts/http_client.py",
                 "scripts/install_assistants.py",
                 "manifests/models.json", "manifests/adaptive-v1.json", "runtime.lock.json",
                 "manifests/assistants-v1.json", "manifests/runtime-presets-v1.json",
                 "LICENSE", "third_party/llama.cpp-LICENSE",
                 "third_party/cpp-httplib-LICENSE.txt", "third_party/nlohmann-json-LICENSE.txt",
                 "third_party/rotate-bits-LICENSE.txt", "docs/ADAPTIVE.md", "docs/PRESETS-AND-ASSETS.md"]

RUNTIME_FILES = list(dict.fromkeys([
    *RUNTIME_FILES, *sorted(PUBLIC_DOC_FILES),
    *(str(p.relative_to(ROOT)) for p in sorted((ROOT / "docs/assets").rglob("*")) if p.is_file()),
]))


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def package(binary, output):
    if output.exists():
        raise FileExistsError("Choose a new output; candidate packages are never overwritten")
    if not binary.is_file():
        raise ValueError("Candidate binary is missing")
    binary_strings = subprocess.check_output(["strings", str(binary)], text=True)
    if "/home/" in binary_strings or "/Users/" in binary_strings:
        raise ValueError("Binary contains private build paths; rebuild with --sanitize-build-paths")
    runtime_hash = sha(ROOT / "runtime.lock.json")
    if runtime_hash not in binary_strings.splitlines():
        raise ValueError("Binary does not embed this package's runtime lock; rebuild from the matching source")
    linked = subprocess.check_output(["ldd", str(binary)], text=True)
    if "not found" in linked:
        raise ValueError("A runtime dependency is missing")
    dependencies = []
    for line in linked.splitlines():
        parts = line.split()
        path = next((Path(part) for part in parts if part.startswith("/")), None)
        if path:
            resolved = path.resolve()
            if ".build" in str(resolved) or "/.runtime/" in str(resolved):
                raise ValueError("Binary depends on development build/runtime directories")
            dependencies.append({"soname": parts[0], "path_on_test_host": str(resolved),
                                 "bytes": resolved.stat().st_size, "sha256": sha(resolved)})
    output.mkdir(parents=True)
    for name in RUNTIME_FILES:
        source = ROOT / name
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (output / "bin").mkdir()
    (output / "README.md").write_text(
        "# Winnow Linux/CUDA runtime\n\n"
        "Start with [the quickstart and supported mode matrix](docs/QUICKSTART.md).\n\n"
        "Use `bin/winnow download`, `bin/winnow serve` and `bin/winnow decide`.\n"
        "Select `--model q8`, `nv4`, `e4b` or `e2b`; choose `--reasoning off|selective|always`,\n"
        "with `on` retained as an alias for selective. Routing happens in the client.\n"
        "`--mtp on|off` and `--vision on|off` explicitly. Defaults are off, context 8K.\n"
        "Use `bin/winnow presets` for context and memory guidance. E2B has explicit 8K/64K profiles.\n"
        "Ordinary chat thinking uses `--native-chat-reasoning on|off` independently.\n\n"
        "Model weights are separate. This archive requires compatible existing Linux/CUDA\n"
        "libraries listed in candidate-manifest.json. E2B payloads are private; use verified\n"
        "local assets as described in [the E2B guide](docs/E2B.md). Source setup.py is a separate\n"
        "source-build workflow.\n"
    )
    shutil.copy2(binary, output / "bin/winnow-server")
    for name, script, extra in [("winnow", "winnow.py", []), ("winnow-serve", "serve.py", ['--server', 'SERVER']),
                                ("winnow-decide", "decision_client.py", [])]:
        text = (
            "#!/usr/bin/env python3\nimport os,sys\nfrom pathlib import Path\n"
            "root=Path(__file__).resolve().parents[1]\n"
            "command=[sys.executable,str(root/'scripts'/" + repr(script) + ")]\n"
        )
        if extra:
            text += "command+=['--server',str(root/'bin/winnow-server')]\n"
        text += "os.execv(sys.executable,command+sys.argv[1:])\n"
        path = output / "bin" / name
        path.write_text(text)
        path.chmod(0o755)
    receipt = {"status": "Private local review candidate; publication held",
               "platform": "Linux x86_64/CUDA, tested RTX5070Ti SM120; not a portable/Mac claim",
               "binary_sha256": sha(binary), "runtime_lock_sha256": runtime_hash,
               "dependencies": dependencies,
               "private_build_path_scan": "Passed; no tested-host private build prefix in binary strings",
               "dependency_mode": "Requires the recorded runtime libraries already installed on the target host. Libraries/weights/build tools are not silently bundled or installed.",
               "files": {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha(path)}
                         for path in sorted(output.rglob("*")) if path.is_file()}}
    (output / "candidate-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = package(args.binary.resolve(), args.output.resolve())
    print(json.dumps({"output": str(args.output), "files": len(result["files"]),
                      "binary_sha256": result["binary_sha256"]}))


if __name__ == "__main__":
    main()
