# Image-aware decisions

Images can participate in direct scoring, generated reasoning, and augmented
scoring. `selective` uses the selected model's raw confidence/entropy gate;
`always` bypasses that gate. Both retain its temperatures, blend and calibrated
direct fallback. Image reasoning has no accuracy or calibration validation claim.
The published quality comparisons concern text.

## Use a supported preset

E4B and NVFP4 support the 8K vision+MTP client contract. For E4B:

```sh
python3 scripts/winnow.py download --model e4b --vision on --mtp on --reasoning selective
python3 scripts/winnow.py serve --model e4b --vision on --mtp on --reasoning selective
python3 scripts/winnow.py decide --model e4b --vision on --mtp on --reasoning selective --input request.json
```

Use `nv4` for NVFP4 or `always` for every-request reasoning. Supply one question
and `winnow.images`, an ordered list of up to 16 base64 image data URLs. Remote
URLs, filesystem paths and malformed base64 are rejected. The native decoder
accepts still images; its image/microbatch and 32 MiB request limits also apply.
Q8 vision+MTP and MTP-off image reasoning have no built-in client preset.

Every phase receives the same image bytes in the same order, before the state.
Images are not removed when reasoning is added. Native inspection counts image
positions when checking both direct and augmented input budgets. No image is
silently converted into a text-only request.

## Python and custom runtimes

```python
client = DecisionPipeline(
    transport, reasoning="selective", policy_id="e4b-calibrated75-g95-v1",
    mtp="on", runtime_profile="e4b-q8-vision8k-mtp",
)
result = client.decide(request)
```

`runtime_profile` also accepts a trusted JSON path or dictionary. This is operator
configuration, never a field from an incoming decision request. The schema in
[`reasoning_contract.py`](../scripts/reasoning_contract.py) requires:

| Field | Contract |
|---|---|
| `schema_version`, `id`, `alias` | Version 1 and explicit profile/model names. |
| `runtime` | Exact target, assistant and runtime hashes; vision/MTP booleans; native/chat contexts; cache, memory, slots, branches, batch, microbatch, head and pipeline. |
| `max_images` | 0 for text-only; 1–16 for vision. |
| `native_seconds`, `generation_seconds` | Per-phase wall deadlines, each 1–300 seconds. |
| `launcher_requirements` | No context shifting; full projector verification for vision; draft length 4 for MTP. |

The launcher must verify the projector and enforce the draft length and disabled
context shifting: the current inspect endpoint does not attest these three
settings. The client verifies all exposed runtime fields on direct and augmented
inspection. A mismatch fails before scoring that phase.

For a separately managed model, `policy_manifest` accepts a trusted local version
1 policy manifest alongside `policy_id` and the runtime contract. This does not
add the model to download/setup or replace any shipped policy. The lower-level
CLI offers `--policy-manifest PATH --runtime-profile PATH`. Use the model's own
verified launcher. A 64K contract declares capacity and settings; it does not
validate policy quality at 64K.

The local `e2b-canonical-v2` prompt format preserves native candidate semantics:
choice and Boolean keys name the option, non-null values describe it, and null
score values use their numeric index. Options retain native candidate order.
This format has distinct prompt serialization from historical E2B experiments;
it does not inherit their quality measurements.

For direct scoring with the same checks, use `reasoning="off"` and the runtime
profile, without a policy. The legacy direct API without a profile retains its
existing request/response behavior.

## Completion, cancellation and services

Only nonempty natural-EOS generation can be blended. The response must explicitly
report no truncation, and prompt plus completion usage must fit the configured
chat context. Overflow, deadlines, failed generation or failed augmented scoring
return the saved calibrated direct distribution with a fallback reason. A failed
direct call is an error. Partial generation is never rescored.

`winnow.adaptive` reports route, fallback, image hashes, inspected native position
counts and runtime limits. `measured_profile` is false for explicit capability
contracts; published text measurements do not validate these configurations.

Service adapters should create a request-scoped transport and forward that
request's authorization to every backend call. `HTTPTransport` accepts explicit
`headers` and a `cancelled` callback, and closes the active connection on a wall
deadline or cancellation. It checks cancellation before/after connect and before
HTTP writes, with automatic reconnection disabled. If connection setup or another
I/O operation has not stopped, the caller returns but that backend origin remains
unavailable until the worker exits. The pipeline retains its shared request lock
during cleanup; a new request cannot overlap the abandoned worker.
Pass the same callback to `DecisionPipeline`, plus one
shared `request_lock` for pipelines using the same backend. Do not hold that
non-reentrant lock outside `decide`. Custom transports implement
`post(endpoint, body, seconds)` with a hard wall deadline and raise
`RequestCancelled` from `reasoning_contract` when the caller disconnects.
Custom transports must stop all I/O before returning. An asynchronous wrapper
around `HTTPTransport` must also forward `check_available()` and
`release_when_idle(lock)` so the pipeline can retain ownership during cleanup.
`BackendUnavailable` reports an aborted worker that is still stopping.
Cancellation propagates without fallback. Keep service authentication and
network exposure under the service's existing access controls.
