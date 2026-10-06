# Presets and assets

Use `python3 scripts/winnow.py presets` (or `bin/winnow presets` in a runtime)
for short preset names, mode defaults, memory estimates and measured baselines.
`q8`, `nv4`, `e4b` and `e2b` select the exact model and matching assistant/projector.
The older model names and low-level preset names remain available as aliases.

## Cache precision

Use **F16 target KV cache for E2B**. Every named E2B text/vision, 8K/64K,
MTP-on/off profile already selects F16. Keep the assistant's draft KV cache
at **Q8_0**, as tested; its BF16 weights are a separate setting.

| Launch path | Target KV default | Assistant draft KV with MTP |
|---|---|---|
| E2B Linux/CUDA profiles | F16, recommended | Q8_0 |
| 12B Q8, 12B NVFP4 and E4B Linux/CUDA profiles | Q8_0, recorded release recipe | Q8_0 |
| Apple Silicon direct profile | F16 | MTP is not part of this profile |

The matched E2B cache confirmation changed only target KV: F16 scored 37/48
versus 35/48 with Q8_0, with NLL 0.776 versus 0.853 and Brier 0.398 versus
0.445. Mean client time was 0.891 versus 0.911 seconds; matched routed decode
was 395.3 versus 355.5 tokens/s. Outputs and lengths changed. The owner chose
F16 as the preferred E2B configuration, and the completed full-panel E2B run
uses F16. See [E2B evidence](E2B.md).

The recorded 12B/NVFP4/E4B release measurements use Q8_0 target KV; this E2B
comparison does not establish a cache preference for those models. Their
existing defaults and the Apple Silicon profile are retained. `--cache f16`
is an explicit target-cache override for the configurable 12B/E4B launchers;
it does not change weight quantization or the assistant draft cache.

All unified presets default to 8K text, MTP off and reasoning off. Enable MTP
explicitly with `--mtp on` on download, serve and matching client commands.
For 12B/E4B, override
`--context`, `--decision-context`, `--batch`, `--ubatch`, `--decision-parallel`
and `--cache` as needed. Use `--vision on|off`, `--mtp on|off` and
`--reasoning off|selective|always` explicitly (`on` aliases selective). Microbatch must not exceed batch. MTP requires
Linux/CUDA, one chat slot, auto memory and the matching verified assistant.
Reasoning requires one named question and a text, object, or array state.
Image reasoning supports the E4B and NVFP4 8K vision+MTP presets; other contexts
need an [explicit runtime contract](IMAGE-REASONING.md). Q8 vision plus MTP
exceeded the measured 16 GB profile; custom/larger
configurations are unvalidated. E2B adds experimental 8K/64K text/vision profiles
with F16 KV, backend temperature sampling and MTP4 or explicit MTP off. Its named
profiles pin numerical settings, and its assets currently require verified local
reuse. See [E2B setup and limits](E2B.md). No silent MTP disable or context reduction occurs.

Memory estimates are advisory, not admission guarantees. Context, images, cache,
batches and concurrency change usage. Published measurements remain specific to
their recorded settings. Custom reasoning settings retain the frozen policy but
do not inherit calibration/quality/latency evidence. See the [adaptive contract](ADAPTIVE.md).

Targets and projectors live in `models/gguf/`; assistants in `models/assistants/`.
Use `--model-dir` for another location. Verified local assets can be reused with
`download --asset-dir PATH --offline`. Every selected target, projector and
assistant is checked against the release manifest. Other models are not silently
substituted. The thin Linux runtime includes no weights and requires its recorded
CUDA 13, NCCL 2 and OpenSSL 3 libraries. It is not a portable Mac binary.

Existing separate assistant archives can still be installed with
`scripts/install_assistants.py --assets PATH --model-dir models/assistants`.
Apache-2.0 license, attribution and exact upstream/conversion provenance accompany
the assistant files. Corrupt or incompatible files are rejected.

`--host`, `--port`, `--gpu`, `--threads`, `--server`, `--api-key-file`,
`--http-threads`, `--metrics`, `--slots` and `--dry-run` remain available.
Loopback is the default. Clients read `WINNOW_API_KEY_FILE` or `WINNOW_API_KEY`;
keys are not packaged. Native direct requests support structured states and multiple
questions; clients using an explicit runtime profile accept one question.
Native direct decisions generate no tokens, regardless of MTP.

See [Quickstart](QUICKSTART.md) for complete commands and [API](API.md) for details.
