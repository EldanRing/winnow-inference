# Maintainer checks

The public package contains the inference server, pinned llama.cpp patches,
examples, tests, documentation and artifact checksums. Model weights are hosted
on [Hugging Face](https://huggingface.co/EldanRing/Winnow-12B). No training dataset
or credentials belong in the source repository.

## Source checks

```sh
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/build.py
```

The build verifies runtime/patch hashes and runs native tests. GitHub CI compiles
on Linux without CUDA or weights and on macOS with Metal. It never downloads
models or invokes a paid API. `--backend cpu` is a compilation check, not a
validated serving profile.

## Model checks

Verify the model/projector against `manifests/models.json`, then run the short
owned-server smoke described in [INSTALL.md](INSTALL.md#quick-verification).
The smoke checks authentication, API behavior, chat, streaming, vision and
selected/full-head consistency. `--long-context` and `--lifecycle` are explicit
additional checks. Functional checks do not substitute for quality benchmarks.

When changing model artifacts, record new checksums and repeat the relevant
runtime/capacity checks before updating public measurements. Preserve benchmark
hardware, precision, sample counts and cold/warm cache definitions.

## Source packaging

From a clean, committed source tree:

```sh
python3 scripts/package_source.py --output dist/winnow-inference --init-git
```

The exporter copies allowlisted committed source, creates fresh history with no
remote, and writes a file-hash receipt beside the output. It refuses dirty trees,
symlinks, unapproved files and existing output directories. Nothing is uploaded.

The Docker context is independently allowlisted. Verify its actual source stage:

```sh
docker build --target source --output type=local,dest=dist/docker-context .
python3 scripts/check_package.py --directory dist/docker-context/source
```

## Prebuilt distributions

`Build prebuilt CUDA container` is a manually dispatched workflow. It builds
Ampere/Ada/consumer Blackwell kernels (`86;89;120`), runs source/native checks,
and checks the packaged executable for unresolved libraries. Publishing is opt-in
and restricted to main; it creates a commit-specific GHCR tag and `cuda` alias.
CI has no GPU or weights, so perform the existing short GPU smoke against the
actual image before recommending a new image. Never include model downloads or
credentials in a container layer.

`Build Apple Silicon native archive` produces a macOS 15+ arm64 tarball with Metal
shaders embedded, Accelerate enabled, and static OpenSSL. Packaging rejects any
remaining non-system dynamic library, includes checksums and third-party licenses,
and retains the matching public source. CI verifies build/unit checks and binary
startup; it does not establish GPU inference quality. The archive is unsigned
and not notarized. Download the workflow artifact, review its checksums and
validation status, then attach the tarball/checksum to a runtime release. Users
run `python3 scripts/setup.py --skip-build` and `python3 scripts/serve.py`.
