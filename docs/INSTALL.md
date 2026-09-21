# Install and run

The supported serving paths are Linux/NVIDIA CUDA and macOS/Apple Silicon Metal.
The decision engine accepts merged Gemma 4 GGUF models. CUDA and Metal builds are
source builds; no Python inference framework or Python packages are required.
Windows, CPU-only serving, and multi-GPU sharding are not validated release profiles.

Clone the inference source before following the commands below:

```sh
git clone https://github.com/EldanRing/winnow-inference.git
cd winnow-inference
```

## Build prerequisites

Linux: Python 3.10+, Git, CMake 3.24+, a C++17 compiler, OpenSSL headers, and the
NVIDIA CUDA toolkit. The container pins CUDA 13.0.2. The RTX 5070 Ti build target
is `120`; use a toolkit with that target and a compatible NVIDIA driver.

```sh
sudo apt-get install build-essential cmake git python3 libssl-dev
python3 scripts/build.py --cuda-arch 120 --jobs 8
```

The command above assumes the CUDA toolkit is already installed and discoverable.
If necessary, set `CUDACXX` to your toolkit's `nvcc` path before building.

Apple Silicon: install Xcode command-line tools, Python 3.10+, Git, CMake and
OpenSSL. For a Homebrew installation:

```sh
brew install cmake openssl@3
OPENSSL_ROOT_DIR="$(brew --prefix openssl@3)" python3 scripts/build.py --jobs 4
```

The build fetches the exact runtime commit, verifies patch hashes and patched
source hashes, compiles the server and runs native unit tests. Use a separate
`--build-dir` when checking a different backend. `--backend cpu` exists for CI
compilation without CUDA or weights; the serving launcher still targets GPU use.

## Obtain and verify weights

[Winnow-12B on Hugging Face](https://huggingface.co/EldanRing/Winnow-12B)
contains merged BF16 safetensors, Q8 GGUF and the matching F16 vision projector.
This server loads the Q8 GGUF; the BF16 safetensors are available for other
compatible runtimes. The projector is required only for vision.

Download the two GGUF files with the Hugging Face CLI (a download-only dependency):

```sh
hf download EldanRing/Winnow-12B gguf/Winnow-12B-Q8_0.gguf \
  gguf/mmproj-F16.gguf \
  --revision b2b14213dfa252e6d6b543c8b334762e51772d29 --local-dir models
python3 scripts/verify_model.py \
  --model models/gguf/Winnow-12B-Q8_0.gguf --mmproj models/gguf/mmproj-F16.gguf
```

You can also download the files through the model page. Exact sizes and SHA256
values are in [the manifest](../manifests/models.json). Do not apply a training
adapter on top of an already merged Winnow model.

## Launch profiles

The following command reproduces the measured release profile: RTX 5070 Ti 16 GB,
65,536 context positions, vision, four decision branches, one chat slot, Q8 KV,
and exclusive context scheduling. Model weights and projector stay on the GPU.

```sh
python3 scripts/serve.py --model models/gguf/Winnow-12B-Q8_0.gguf \
  --mmproj models/gguf/mmproj-F16.gguf --context 65536 --cache q8_0 \
  --decision-parallel 4 --chat-parallel 1 --memory exclusive
```

Context includes formatting, image positions, questions and generated replies.
At this capacity, requests queue while switching between chat and decisions;
model weights remain loaded but switching APIs discards the idle context's KV
cache. See [the measurements](VALIDATION.md) and
[concurrency behavior](API.md#chat-and-concurrency).

Apple Silicon uses Metal and F16 KV by default:

```sh
python3 scripts/serve.py --model models/gguf/Winnow-12B-Q8_0.gguf \
  --mmproj models/gguf/mmproj-F16.gguf --context 65536 --cache f16 \
  --decision-parallel 1 --chat-parallel 1
```

An earlier Winnow checkpoint passed this 64K + vision profile on an M4 Pro with
24 GB unified memory. The same earlier checkpoint also passed a 128K CUDA profile
with one decision branch. Those checks demonstrate runtime capacity; the final
release's measured local profile is the 5070 Ti 64K configuration above.
Longer context is configurable, subject to available memory and the model's
262,144-position trained limit. No fixed chat/decision memory split is imposed.

The server listens on `127.0.0.1:8091` by default.

## Authenticated serving and checks

For an explicitly network-accessible service, supply `--host` and llama-server's
`--api-key-file /path/to/key-file`. Clients send `Authorization: Bearer <key>`.
Use a file outside the source repository. The included checks accept
`WINNOW_API_KEY_FILE` or `WINNOW_API_KEY`; they do not save credentials in results.
The `/health` endpoint follows llama-server's public health-check behavior.

## Quick verification

This starts and stops its own authenticated servers on temporary localhost ports.
It runs API, vision (when a projector is supplied), chat/streaming, input-boundary,
authentication and selected/full-head parity checks. It does not change another
server, download weights, train, or call paid APIs. Output allowance is 4,096;
ordinary replies stop naturally. Expect a few minutes, depending on hardware.

```sh
python3 scripts/release_check.py \
  --model models/gguf/Winnow-12B-Q8_0.gguf --mmproj models/gguf/mmproj-F16.gguf \
  --context 65536 --decision-parallel 4 --memory exclusive --output results/release-smoke
```

Add `--long-context` to populate the configured capacity, or `--lifecycle` for
cancellation and longer mixed-load probes. These are explicit additional checks;
the default smoke does not run a quality benchmark or a context/concurrency sweep.
Use a fresh output directory for each artifact or configuration.

## Common setup errors

- `nvcc` or CUDA architecture not found: check the toolkit path and target support.
- Patch/revision mismatch: retain your changed runtime separately and use a fresh
  `.runtime/llama.cpp` checkout for the pinned build. Do not force-apply patches.
- Projector/model mismatch: verify both files against the same model manifest.
- A context allocation fails: choose a measured profile for the card. Increasing
  parallel chat slots divides their context; it does not increase per-chat capacity.
- HTTP 401 during a check: give the check the server's key through the variables above.
