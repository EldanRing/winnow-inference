# 2026.10.05 E4B adaptive policy update

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
