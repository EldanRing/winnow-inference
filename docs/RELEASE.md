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
`winnow-linux-x86_64-cuda-v2026.10.06.tar.gz`, and `SHA256SUMS`.
Archive hashes and the review diff are supplied beside the release package. Extract into a separate directory and reuse verified assets.

## Target cache default update

F16 target K/V cache is now the default and recommendation for all supported
models and platform profiles. Quantized target cache remains an explicit
`--cache q8_0` option. Assistant draft-cache flags remain Q8_0 (Gemma 4 assistant attention shares
the target K/V tensors); model/assistant/projector
weight precision is unchanged. Existing historical Q8-cache measurements and
reproduction commands retain their recorded settings. Native code and the sealed
server binary are unchanged. See [cache precision](PRESETS-AND-ASSETS.md#cache-precision).

## Upgrading

There is no automatic updater. The easiest Linux/CUDA upgrade is the new runtime
archive: it contains the matching client, manifests and native binary, so no build
is needed. Keep the previous install and launch command for rollback. The new
runtime requires compatible CUDA 13, NCCL 2 and OpenSSL 3 libraries already installed;
their exact dependency receipt is in `release-manifest.json`.

### Linux/CUDA runtime: download, verify, extract

Run these commands in a fresh directory after this version appears on the release
page. `mkdir` intentionally fails if the directory already exists; do not extract
over an existing installation.

```sh
mkdir winnow-upgrade-v2026.10.06
cd winnow-upgrade-v2026.10.06
curl -fLO https://github.com/EldanRing/winnow-inference/releases/download/v2026.10.06/winnow-linux-x86_64-cuda-v2026.10.06.tar.gz
curl -fLO https://github.com/EldanRing/winnow-inference/releases/download/v2026.10.06/SHA256SUMS
sha256sum --check --ignore-missing SHA256SUMS
tar -xzf winnow-linux-x86_64-cuda-v2026.10.06.tar.gz
cd winnow-linux-x86_64-cuda-v2026.10.06
```

Use an absolute path to your existing model directory. An optional offline check
verifies those files without downloading or replacing them. This example is for
E4B text with MTP and decision reasoning off; use your own model and mode flags:

```sh
WINNOW_OLD_MODELS='/absolute/path/to/existing/models'
bin/winnow download --model e4b --model-dir "$WINNOW_OLD_MODELS" \
  --vision off --mtp off --offline
bin/winnow serve --model e4b --model-dir "$WINNOW_OLD_MODELS" \
  --context 8k --vision off --mtp off --reasoning off --cache f16 --port 8091
```

Before starting, stop only your own old server on that port. Launch never downloads
models. Keep the model, context, vision, MTP, port, GPU, authentication and other
explicit options from your saved command; replace its executable with the new
`bin/winnow`, remove an old `--server` override, and supply the existing model path.
Use the new directory's client too: replace `python3 scripts/winnow.py decide`
with `bin/winnow decide` and retain its request, URL and mode options. Existing
`--reasoning on` and `--mode experimental-adaptive` still mean selective;
`--mode direct` still means direct. Server/client vision, MTP and context settings
must match. Ordinary chat thinking remains a separate option.

**Cache change:** target K/V now defaults to F16 for every model. To retain the
previous Q8 cache, explicitly pass `--cache q8_0` to both `serve` and `decide`.
This changes no weight files. F16 can use more memory, so keep an explicit Q8
override when preserving the previous memory configuration. Decision reasoning,
MTP and ordinary chat thinking still default to off; changing executables does
not enable them. Existing custom settings are retained through your saved launch
options and authentication files/environment; no configuration file is migrated
or rewritten. Automatic llama.cpp configuration files remain ignored by the
managed launcher, as in the prior release.

### Source installs

Source users need the new client/manifests **and a rebuilt server** because the
runtime lock changed. Keep local edits by creating a separate checkout:

```sh
git fetch origin --tags
git worktree add --detach ../winnow-v2026.10.06 v2026.10.06
cd ../winnow-v2026.10.06
python3 scripts/build.py
python3 scripts/winnow.py serve --model e4b \
  --model-dir /absolute/path/to/existing/models --cache f16
```

Alternatively extract `winnow-source-v2026.10.06.tar.gz` into a new directory and
build there. Retain your own launch options in the final command. Do not reuse
the earlier public 12B/E4B binary or overlay old scripts/manifests onto the new
package. Apple Silicon users use a source build; the Linux runtime is not a Mac
binary. Source build prerequisites are in [the quickstart](QUICKSTART.md#source-setup).

### Rollback

Stop your new server, then restart the saved command from the old install with its
old client and original port/settings. The old directory, model files and custom
configuration remain in place. No models need to be deleted, copied or downloaded
again. No user service is edited automatically.

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
