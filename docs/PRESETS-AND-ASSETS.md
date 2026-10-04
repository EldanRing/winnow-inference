# Presets and assets

Use `python3 scripts/winnow.py presets` (or `bin/winnow presets` in a runtime)
for short preset names, mode defaults, memory estimates and measured baselines.
`q8`, `nv4` and `e4b` select the exact release model and matching assistant/projector.
The older model names and low-level preset names remain available as aliases.

All unified presets default to 8K text with MTP and reasoning off. Override
`--context`, `--decision-context`, `--batch`, `--ubatch`, `--decision-parallel`
and `--cache` as needed. Use `--vision on|off`, `--mtp on|off` and
`--reasoning on|off` explicitly. Microbatch must not exceed batch. MTP requires
Linux/CUDA, one chat slot, auto memory and the matching verified assistant.
Reasoning requires one named question and a text state; image reasoning is not
supported. Q8 vision plus MTP exceeded the measured 16 GB profile; custom/larger
configurations are unvalidated. No silent MTP disable or context reduction occurs.

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
keys are not packaged. Direct requests support structured states and multiple
questions. Native direct decisions generate no tokens, regardless of MTP.

See [Quickstart](QUICKSTART.md) for complete commands and [API](API.md) for details.
