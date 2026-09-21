# Winnow-12B inference

A native llama.cpp server for local typed decisions and regular chat, sharing one
loaded model. `/v1/systemone` evaluates `noul`, `choice`, and `score` questions
against a shared state. `/v1/chat/completions` retains llama-server's chat, vision,
and streaming interfaces.

**64K context and vision on a 16 GB RTX 5070 Ti**, with Q8 weights fully on the
GPU. Winnow-12B is a fine-tune of Gemma 4 12B; the same loaded model serves typed
decisions and regular chat.

[Model weights and model card](https://huggingface.co/EldanRing/Winnow-12B) ·
[Benchmarks](docs/BENCHMARKS.md) · [API](docs/API.md) ·
[Installation](docs/INSTALL.md)

On the 231-item public JevBench subset, **Winnow-12B Q8 matched hosted Jev at
85.7% (198/231)**; Winnow-12B BF16 scored 85.3%. Local-model comparisons use the
same RTX PRO 5000 Blackwell. See the [full benchmark report](docs/BENCHMARKS.md).

![Decision-quality comparison of Winnow-12B BF16 and Q8, hosted Jev, Kev and Laya](docs/assets/02-decision-quality.png)

Merged BF16 weights, Q8 GGUF, and the matching vision projector are separate
downloads. The training dataset is private. Exact release checksums are in
[the model manifest](manifests/models.json).

## Build

Linux needs a C++17 compiler, CMake 3.24+, Git, Python 3.10+, OpenSSL development
headers, and a CUDA toolkit supporting your GPU (Blackwell requires a suitable
recent toolkit). Apple Silicon needs Xcode command-line tools, CMake, Git and
Python 3.10+; Metal and Accelerate are used automatically. No Python packages
are needed for serving or the included API checks.

```sh
git clone https://github.com/EldanRing/winnow-inference.git
cd winnow-inference
python3 scripts/build.py --jobs 8
```

The script fetches the pinned llama.cpp commit, verifies and applies the patches,
builds `.build/bin/winnow-server`, and runs the native unit checks. It refuses an
unexpected runtime revision or modified patched source. To build a Linux image
for the RTX 5070 Ti without a GPU present, use `--cuda-arch 120`.

## Run

```sh
python3 scripts/serve.py \
  --model models/gguf/Winnow-12B-Q8_0.gguf \
  --mmproj models/gguf/mmproj-F16.gguf \
  --context 65536 --cache q8_0 --decision-parallel 4 \
  --chat-parallel 1 --memory exclusive
```

Omit `--mmproj` for text only. The launcher uses full GPU offload, disables context
shifting and automatic weight fitting, and listens on `127.0.0.1:8091`.
This is the measured 5070 Ti profile. The default KV format is Q8 on CUDA and
F16 on Metal; see installation instructions for the Apple Silicon profile.

64K means the context capacity, including prompt formatting, image positions,
questions, and any generated reply. It is not a 64K text allowance plus unlimited
images or output. Longer contexts are configurable up to the model's trained
limit, subject to hardware capacity; see the measured profiles in the validation
report. No fixed decision/chat memory split is imposed.

Chat and decisions have independent KV contexts and share the model and projector.
When both contexts fit, chat advances between decision prefill chunks. Otherwise,
requests queue and idle contexts are released and recreated without reloading the
weights. This evicts that context's KV cache and is reported in diagnostics.
`--memory exclusive` selects this behavior immediately. `--decision-parallel`
controls question branches per wave; the API accepts up to 256 questions and
runs as many waves as needed. `--chat-parallel` divides llama-server's total chat
context among slots, so increasing it reduces each chat slot's capacity.

```sh
curl http://127.0.0.1:8091/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @examples/decisions.json
python3 examples/client.py
node examples/client.mjs
```

See [the API contract](docs/API.md) for probabilities, image inputs, diagnostics,
and compatibility boundaries. Native llama-server flags can follow the wrapper
arguments; `python3 scripts/serve.py --help` shows wrapper settings. Explicit
native overrides can change the tested profile.

## Verify your setup

A short, repeatable release check starts and stops its own authenticated local
servers, including selected/full-head comparison:

```sh
python3 scripts/release_check.py --model models/gguf/Winnow-12B-Q8_0.gguf \
  --mmproj models/gguf/mmproj-F16.gguf --context 65536 \
  --decision-parallel 4 --memory exclusive --output results/release-smoke
```

It uses a 4,096-token output allowance and does not run a quality benchmark or
context sweep. Add `--long-context` explicitly for capacity confirmation.
For an already running server, use the individual checks below. Authenticated
servers are supported through `WINNOW_API_KEY_FILE` or `WINNOW_API_KEY`.

```sh
python3 scripts/check.py --output results/check --vision
python3 scripts/check.py --output results/64k --vision --long-context 65536 --image-size 1536
python3 scripts/bench.py --output results/timings --repeats 5
python3 scripts/parity.py --output results/selected
```

Restart with `--head full`, then use `scripts/parity.py --output results/full
--compare results/selected` to verify selected-head probabilities. The full-head
mode is a reference implementation; normal chat always retains its full head.
All scripts save requests and responses locally. These focused probes validate
runtime behavior and timings, not general model quality or calibrated confidence.

## Container

GPU execution requires a Linux Docker engine configured with the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Building the image alone does not require GPU access.

```sh
docker build --build-arg CUDA_ARCH=120 -t winnow-inference .
docker run --rm --gpus 'device=0' -p 127.0.0.1:8091:8091 \
  -v /path/to/models:/models:ro winnow-inference \
  --model /models/gguf/Winnow-12B-Q8_0.gguf --mmproj /models/gguf/mmproj-F16.gguf \
  --context 65536 --cache q8_0 --decision-parallel 4 \
  --chat-parallel 1 --memory exclusive
```

The Docker build context includes public source files only. The container uses a
pinned CUDA image and runs as an unprivileged user. It does not download weights
or embed credentials. Local native builds are the validated serving path; the
container's separate build status is recorded in the report.

## Source layout

- `native/`: protocol, prefix/branch planner, selected-answer engine and queue bridge.
- `patches/`: small, auditable changes to the pinned llama.cpp runtime/server.
- `scripts/`: dependency-free build, launch and validation tools.
- `tests/`: protocol/planner tests and public synthetic runtime probes.
- `manifests/`: artifact identity and release status.

[Maintainer checks](docs/RELEASE.md) cover automated checks and clean source
packaging.

The code is MIT licensed. Model weights retain their own model terms. Nothing is
published automatically by these scripts.
