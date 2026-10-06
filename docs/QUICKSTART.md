# Download and run Winnow

Choose a short preset with `--model q8`, `nv4`, `e4b` or `e2b` (`--preset` is an alias).
Run `python3 scripts/winnow.py presets` to see defaults and memory guidance.
The old model names remain accepted. Set context with `--context 4k`, `16k` or
`65536`. Choose `--reasoning off|selective|always`; `on` remains an alias for
`selective`. Vision and MTP have independent on/off flags. Reasoning adds a short
model-generated analysis before rescoring a decision with the model's
frozen temperatures and blend. MTP uses the matching assistant to draft ordinary chat tokens. Native direct decisions
generate no tokens, so MTP does not itself add decision reasoning.

12B/E4B model and assistant assets are published in the existing Winnow repositories.
E2B uses pinned target, projector and assistant downloads; see [the E2B guide](E2B.md).
Exact sizes and SHA256 are checked; no other weights are substituted. Nothing in
setup publishes files or uses a paid service.

## Reasoning modes

| Option | Decision behavior |
|---|---|
| `--reasoning off` | Native direct scoring; the default. |
| `--reasoning selective` | Reason only when the model's frozen confidence/entropy gate routes the decision. |
| `--reasoning always` | Attempt reasoning on every valid decision after direct scoring; bypass the gate. |

`--reasoning on` keeps its previous selective behavior. Both reasoning modes
retain the model-specific temperatures and blend, natural-EOS requirement,
context limit, request deadline, and calibrated direct fallback. Always mode
can cost more and does not inherit the selective policy's measured quality or
latency. It does not reproduce historical all-cases experiments with other blends
or token limits. See [results and limits](REASONING-RESULTS.md).

Routing runs in the decision client. `serve --reasoning selective` and
`serve --reasoning always` prepare the same compatible server; choose routing on
each `decide` call. Native `/v1/systemone` requests remain direct.
Ordinary chat thinking uses `serve --native-chat-reasoning on|off` separately.
It defaults to off; adaptive internal generation explicitly disables it.

## Supported choices

| Model | Direct, MTP off | Direct, MTP on | Reasoning, MTP off | Reasoning, MTP on |
|---|---|---|---|---|
| 12B Q8 | Text or vision | Text; vision needs more memory* | Text | Text |
| 12B NVFP4 | Text or vision | Text or vision | Text | Text or vision (8K profile) |
| E4B Q8 | Text or vision | Text or vision | Text | Text or vision (8K profile) |
| E2B Q8, experimental | Text or vision | Text or vision | Text or vision* | Text or vision (8K/64K profiles) |

Optional reasoning and MTP require Linux/CUDA and a compatible GPU. Direct serving
also supports Apple Silicon Metal through a source build; optional modes have not
been validated on Mac. CPU-only, native Windows and multi-GPU serving are outside
this release. Reasoning accepts one question and a text, object, or array state.
Images require a supported vision profile. See [image reasoning](IMAGE-REASONING.md)
for commands, explicit context contracts, and quality limits.
*Q8 vision plus MTP exceeded the measured 16 GB profile; larger or custom
configurations are unvalidated. E2B has a recorded bounded MTP on/off check;
Dani confirms his own 64K testing. Reported full-panel scores use 8K.

All four model selectors default to **8K text, reasoning off, MTP off**.
All models are configured and recommended with **F16 target KV cache**; assistant
draft-cache flags remain Q8_0 when MTP is enabled. Gemma 4 assistants share the
target K/V tensors. Use `--cache q8_0` explicitly for
quantized target cache. See [cache precision](PRESETS-AND-ASSETS.md#cache-precision).
E2B is
Linux/CUDA only and pins its named profile settings. For 12B/E4B, context, native
branches, batch, microbatch and cache can be overridden. MTP requires one chat
slot and auto memory; reasoning requires one question with a text, object, or array state. Presets
supply defaults. Image reasoning pins its runtime settings; custom image/context
configurations require an explicit operator contract. The legacy direct platform profiles remain
available through `scripts/serve.py`.

| Preset | Default modes/context | Estimated GPU memory |
|---|---|---|
| `q8` — 12B Q8 | 8K, text, MTP/reasoning off | 12.8–14.8 GiB |
| `nv4` — 12B NVFP4 | 8K, text, MTP/reasoning off | 8.6–10.6 GiB |
| `e4b` — E4B Q8 | 8K, text, MTP/reasoning off | 8.4–10.4 GiB |
| `e2b` — E2B Q8 | 8K, text, MTP/reasoning off | 5.5–7.5 GiB |

All models now default to F16 target KV. These advisory memory ranges retain
the earlier measured recipes; F16 memory use can differ. 12B/E4B use four native
branches; E2B uses one. All use one chat slot. Context,
images, cache precision, batches, concurrency, hardware and other GPU workloads
change usage. Setup warns about estimated capacity; it does not impose a blanket
16 GB gate. Historical measured baselines on RTX 5070 Ti: E4B direct 8K text
8.56 GiB; E4B direct 64K vision 10.72 GiB; Q8 8K text + MTP 14.75 GiB sampled;
NVFP4 8K vision + MTP 11.50 GiB sampled. These are different configurations.

Custom reasoning configurations retain the frozen policy but do not inherit its
published calibration, quality or latency evidence. The client reports a brief
configuration note when settings differ. MTP-off latency is also distinct.

## Source setup

From the source directory, install the [system prerequisites](INSTALL.md)
once. Linux needs Python 3.10+, Git, CMake 3.24+, C++17, OpenSSL development headers,
a compatible NVIDIA driver and CUDA toolkit. Apple Silicon needs Xcode command-line
tools, arm64 Python, CMake and OpenSSL. No pip or Hugging Face CLI is needed.

Select the desired combination during setup; for example:

```sh
python3 scripts/winnow.py setup --model e4b --vision off --reasoning off --mtp off
```

Setup checks prerequisites, acquires verified files, builds, and prints the launch
command. Change the same switches to prepare another mode. For multiple installed
CUDA toolkits, add `--cuda-compiler /path/to/cuda/bin/nvcc --cuda-arch 120`, using
your GPU's architecture. Setup does not install system packages.

## Download and launch

These examples use E4B. Replace `e4b` with either 12B model selector. Download is
explicit; launching never starts a hidden download. Run one server at a time.
For example, append `--context 16k` to a serve command; add `--batch 1024
--ubatch 512` if needed. Numerical overrides affect memory use.

| Choice | Download | Start server |
|---|---|---|
| Direct decisions and ordinary chat | `python3 scripts/winnow.py download --model e4b` | `python3 scripts/winnow.py serve --model e4b` |
| Direct decisions and MTP chat | `python3 scripts/winnow.py download --model e4b --mtp on` | `python3 scripts/winnow.py serve --model e4b --mtp on` |
| Adaptive decisions without MTP | `python3 scripts/winnow.py download --model e4b --reasoning selective` | `python3 scripts/winnow.py serve --model e4b --reasoning selective` |
| Adaptive decisions with MTP | `python3 scripts/winnow.py download --model e4b --reasoning selective --mtp on` | `python3 scripts/winnow.py serve --model e4b --reasoning selective --mtp on` |

For vision, add `--vision on` to download, serve and reasoning `decide` commands,
using a supported matrix cell. Image reasoning uses the 8K vision+MTP profile;
changing context for E4B/NVFP4 requires a [custom runtime contract](IMAGE-REASONING.md).
E2B also has a named 64K profile; pass the same `--context`, `--vision` and `--mtp`
on its server and client commands.
The projector is downloaded only for vision, and the assistant only for MTP.
Targets/projectors live under `models/gguf/`; assistants under `models/assistants/`.
Use `--model-dir PATH` on download and serve to store them elsewhere.

In a second terminal, query a direct server:

```sh
python3 scripts/winnow.py decide --model e4b --input examples/decisions.json
```

For a reasoning-ready server, choose the client routing mode and match its MTP switch:

```sh
python3 scripts/winnow.py decide --model e4b --reasoning selective --mtp on \
  --input examples/adaptive-decision.json
```

For every-decision reasoning, replace `--reasoning selective` with
`--reasoning always` in that command. Omit `--mtp on` when the server has MTP off. A
mismatch is an error, not a silent mode change. Direct clients also verify the
selected model and quantization; a leftover server for another model is rejected. Ordinary API calls to
`/v1/systemone` always remain direct; adaptive routing is performed by this client.
Selective mode can bypass reasoning. Both reasoning modes fall back to the
policy-calibrated direct result if generation fails; neither guarantees improved accuracy.

By default the server uses `127.0.0.1:8091`. `--port`, `--host`, `--threads`,
`--gpu`, `--server` and `--api-key-file` are passed to the underlying launcher.
Use `--dry-run` to inspect a launch. Clients accept `--base-url`; authentication
uses `WINNOW_API_KEY_FILE` or `WINNOW_API_KEY`. Stop your server with Ctrl+C.

## Reuse verified local assets

To avoid redownloading, give one or more explicit directories containing the
manifest filenames, either flat or in their canonical subdirectories:

```sh
python3 scripts/winnow.py download --model e4b --reasoning selective --mtp on \
  --asset-dir /path/to/targets --asset-dir /path/to/assistants --offline
```

Files are verified before copying and again before installation. Existing correct
files are reused; corrupt files are not overwritten. `--offline` fails clearly
when an asset is missing. Without it, published files use the recorded release URL;
interrupted HTTP downloads resume. A missing release file reports its
URL/status and stops. Pending/private assets never trigger a speculative download.
The same asset options work with `scripts/setup.py`.

## Thin Linux runtime archive

Download a Linux/CUDA runtime archive and its checksum file from the
[versioned releases](https://github.com/EldanRing/winnow-inference/releases).
Use the filename and verification command shown for that version. Extract into
a new directory and run its `bin/winnow` commands. For later releases, run `bin/winnow update`; it updates the matching client,
manifests and native binary together while retaining existing models and settings.
See [the one-time bootstrap and update details](RELEASE.md#updating) for older installs.
The archive requires CUDA 13, NCCL 2 and OpenSSL 3 runtime libraries, with exact
dependencies listed in `release-manifest.json`; they must already be installed.
This is not a portable Mac binary
or a system dependency installer. Model weights are separate.

Use `bin/winnow` in place of `python3 scripts/winnow.py` in every table command.
The archive includes the downloader, manifests, license/attribution files and
examples, and automatically selects its bundled `bin/winnow-server`. No source
build is needed. Source setup.py is intentionally a source-build workflow.

## Explicit launch settings

The launcher ignores automatic llama.cpp system/user configuration and clears
inherited backend asset/speculation settings that would override the selected
modes. Other environment settings are retained. Launch is offline; use the
separate download command to acquire verified weights.

## Validation scope

The release readiness receipt distinguishes focused first-response checks in a
fresh source/export directory from installation on a new machine. Reusing local
verified assets and installed toolkits avoids another large download; it does not
establish a clean-machine install. Downloader resume/error/integrity behavior is
covered separately. Existing numerical quality/latency evidence retains its
original model, binary and profile provenance.
