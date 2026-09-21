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

The build verifies runtime/patch hashes and runs native tests locally.
GitHub Actions and hosted CI are disabled by project-owner instruction; do not
add or run hosted workflows. Run validation on the local machine.
`--backend cpu` is a compilation check, not a validated serving profile.

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

Build and validate containers locally with Docker before publishing a runtime image.
Run the short GPU smoke against that exact image. Keep model files outside image
layers and mount them read-only when serving.

On an Apple Silicon Mac, a portable archive can be built locally:

```sh
OPENSSL_ROOT_DIR="$(brew --prefix openssl@3)" python3 scripts/build.py --backend metal --static-openssl --jobs 3
python3 scripts/package_macos.py
```

Packaging checks non-system dynamic dependencies and includes the public source,
checksums, runtime binary and third-party licenses. Validate the archive locally
before publishing. This command-line archive is unsigned and not notarized.
It needs no CUDA or Docker; Metal and Accelerate remain enabled.
