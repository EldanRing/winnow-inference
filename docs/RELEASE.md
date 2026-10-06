# E2B support and upgrades

E2B adds exact artifact bindings, 8K/64K F16-KV runtime profiles, optional MTP4
and text/image decisions. Its native compatibility patch admits width 1536
while retaining target/assistant width and vocabulary checks. See [E2B setup
and evidence limits](E2B.md). The target, projector and matching assistant are pinned downloads from the model repository.

The new runtime lock requires a matching rebuilt server. Older adaptive clients
reject its changed identity; upgrade the client, manifests and binary together.
12B/E4B model files and policy definitions are preserved. Target K/V cache
defaults change to F16 as described below.

Choose `--reasoning off`, `selective`, or `always` in the CLI and Python client.
Off remains the default; existing `on` commands retain selective routing.
Always attempts reasoning after each valid direct decision, with the same model
identity checks, calibrated blend, natural stopping, context/deadline limits,
and direct fallback. It does not inherit selective-policy quality or latency
claims. See [mode examples](QUICKSTART.md#reasoning-modes) and [Python/API usage](API.md#client-reasoning-modes).

The E4B study's aggregate evidence and selection/calibration provenance are now
linked from [its results](REASONING-RESULTS.md#e4b-policy-update-measured-quality-and-cost).
That earlier client update preserved policy constants, model files and the server.

## Release v2026.10.06

The E2B default selective policy is `e2b-raw99-blend50-v3`; the previous numbered
policy remains available. The verified inference source is
`ee6bd37d34ae35d2e69ebb0c4b0957a10c727357`. The documentation and evidence
update records the completed 3,277-case frozen comparison in [E2B](E2B.md).
Numerical policy settings and native code are unchanged. The runtime reuses the
sealed E2B-compatible server with SHA256
`d531df29615f2ed906c038d27ffb54d965d6f9c94e47c7ff574839a1047200c3`
and its dependency receipt. No server rebuild is needed for this update.
The release assets are `winnow-source-v2026.10.06.tar.gz`,
`winnow-linux-x86_64-cuda-v2026.10.06.tar.gz`, `winnow-update.py`, and `SHA256SUMS`.
Archive hashes and the review diff are supplied beside the release package. For a first install, extract once; later releases use `bin/winnow update` with existing assets.

## Target cache default update

F16 target K/V cache is now the default and recommendation for all supported
models and platform profiles. Quantized target cache remains an explicit
`--cache q8_0` option. Assistant draft-cache flags remain Q8_0 (Gemma 4 assistant attention shares
the target K/V tensors); model/assistant/projector
weight precision is unchanged. Existing historical Q8-cache measurements and
reproduction commands retain their recorded settings. Native code and the sealed
server binary are unchanged. See [cache precision](PRESETS-AND-ASSETS.md#cache-precision).

## Updating

From the installation directory, run:

```sh
bin/winnow update
```

It checks the latest official release, downloads and verifies only the matching
code/runtime asset, and switches the client, manifests and native server together.
No repeated Git pulls, cloning, model downloads or manually managed install
folders are needed. If already current, it makes no code download. Keep using
your existing launch command, model paths and explicit settings. Running services
are not restarted; the new version applies on your next start.

### Existing installations: one-time bootstrap

For installations older than v2026.10.06, run this once after the release is
published, replacing the path with your current Winnow installation:

```sh
curl -fsSL https://raw.githubusercontent.com/EldanRing/winnow-inference/v2026.10.06/scripts/bootstrap_update.py | python3 - /absolute/path/to/winnow
```

The bootstrap obtains the updater from the official release and verifies its
SHA256 before running it. It creates the actual `bin/winnow` command inside that
installation. Thereafter, use `bin/winnow update`.

For a new source checkout or freshly unpacked source archive, complete the
[initial setup and build](../README.md#2-choose-a-preset-and-set-up) first:

```sh
python3 scripts/winnow.py setup --model q8
python3 scripts/winnow.py update
```

Choose another model with `--model nv4`, `e4b` or `e2b`. The first `update` enables
`bin/winnow update` for later releases. If the source archive is already current,
`update` reports `already_current` and does not perform the initial build. New
runtime installs already provide `bin/winnow update`.

Model files and user configuration outside managed code remain in place. The
updater stores active code and one automatic recovery version under `.winnow/`;
users do not manage those directories. Interrupted or invalid downloads leave
the current version selected. Rerun the same command after an interruption.
Local modifications to source/package files are refused rather than overwritten;
commit/stash source edits or keep that customized install outside automatic updates.

### Platforms and settings

Linux x86-64 runtime installs use a matching CUDA archive after runtime-library
and GPU compatibility checks. Source installs build the new source automatically
with their existing prerequisites; cached pinned llama.cpp sources are reused.
Existing CMake backend/compiler/architecture settings are retained. macOS uses a
source build, never the Linux/CUDA binary. To choose a host-specific source build
explicitly, run `bin/winnow update --source`.

**F16 target K/V is the new default.** Existing weight files are unchanged. Keep
`--cache q8_0` explicitly on both `serve` and `decide` to retain Q8 cache. Context,
vision, MTP, reasoning, ports, GPU and authentication settings remain your saved
launch options; no configuration file is rewritten or service automatically edited.
Reasoning and MTP still default to off. Server/client settings must match.

If a release causes a problem, the optional `bin/winnow rollback` selects the
automatic recovery version. No folders or backups need to be managed manually.

## Earlier E4B policy update (2026.10.05)

The opt-in E4B policy now uses raw max probability <0.95 and a 25:75
direct/reasoned blend. The existing E4B direct and augmented temperatures are
unchanged. The former `e4b-calibrated50-v1` remains selectable; Q8 and NVFP4
policies, model weights, assistant, projector, and server binary are unchanged.
Direct decisions remain the default. See [measured results and limits](REASONING-RESULTS.md#e4b-policy-update-measured-quality-and-cost).

Download the new [versioned release](https://github.com/EldanRing/winnow-inference/releases/tag/v2026.10.05-e4b-policy)
and verify `SHA256SUMS-e4b-policy`. Extract the Linux runtime into a new
directory, then reuse verified assets. Source users can update to tag
`v2026.10.05-e4b-policy`. Existing tags and archives remain available.

## Earlier client compatibility patch

The optional Linux/CUDA adaptive client now accepts one named question with a
text, object, or array state. It keeps the original structured state for direct
native scoring, JSON-serializes it only in the generation prompt, and passes a
completed structured-state analysis to augmented native scoring as
`{"original_state": state, "model_reasoning": text}`. String behavior, image
rejection, invalid-state rejection, model/runtime checks, the frozen gates and
50/50 blends are unchanged. Native `/v1/systemone` already supported objects and
arrays; direct decisions remain the default.

The private bounded check used the released Winnow-12B Q8 artifact and official
MTP assistant on the public Jev231 subset. The old adaptive client scored 196
text-state cases and rejected 35 object-state cases; fresh direct scored all 231.
Six string controls matched exactly after the patch. All 35 object states then
received adaptive responses, with 26/35 correct direct and 30/35 adaptive.
A **two-run composite**, combining the original 196 text paths with the 35 later
patched object paths, was 198/231 direct and 205/231 adaptive; it routed 28
cases, completed 27 blends, and had one natural-EOS/context fallback. This is
public, previously observed evidence, not a continuous 231-case patched run or
independent deployment validation. It does not compare always-on reasoning with
the selective blend. See the [adaptive contract](ADAPTIVE.md) for limits.

Existing users should download the new versioned runtime and checksum file from
[v2026.10.05](https://github.com/EldanRing/winnow-inference/releases/tag/v2026.10.05),
verify `SHA256SUMS-client1`, and extract into a new directory. Reuse verified
model and assistant files via `--model-dir` or `--asset-dir ... --offline`.
Source users should update to the `v2026.10.05` tag and continue with their
existing build; the change is in the Python client. No model file, assistant,
server binary, policy, existing release asset, or old tag was replaced.

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
symlinks, unapproved files and existing output directories. Private review
worksheets and draft model cards are excluded by explicit path, while the public
adaptive contract and technical limits remain included. Nothing is uploaded.

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
