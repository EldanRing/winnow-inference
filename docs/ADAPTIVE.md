# Experimental adaptive decision contract

For measured gains, regressions, latency and calibration tradeoffs, see
[reasoning results](REASONING-RESULTS.md).

Use the [quickstart](QUICKSTART.md) for download, launch and client commands.
Native `/v1/systemone` calls remain direct. Client reasoning has `off`,
`selective` and `always` modes; `on` aliases selective. Selective applies the
frozen gate below; always bypasses only that gate. Both reasoning modes are
Linux/CUDA only and accept one question, with an 8K text default. Images require
an [explicit vision runtime contract](IMAGE-REASONING.md).
Matching verified model/runtime identity is required. MTP is independently
optional; with MTP off no assistant is loaded and speculation is disabled.
The same frozen policy applies. Earlier MTP latency evidence does not describe
MTP-off latency. See [presets and assets](PRESETS-AND-ASSETS.md) for default
numerical settings, accepted overrides and artifact verification.

## Frozen version 1 policy contract

`manifests/adaptive-v1.json` is the product configuration. Policy IDs bind exact model/assistant files.

| ID | Gate on raw native T=1 | Direct/augmented temperatures | Completed blend |
|---|---|---|---|
| `nvfp4-entropy-v1` | normalized entropy >0.48619198949270803 | 1 /1 | 50/50 |
| `e4b-calibrated75-g95-v1` (E4B default when opted in) | max probability <0.95 | 1.2041180007310734 /3.4209273427377678 | 25/75 direct/reasoned |
| `e4b-calibrated50-v1` (legacy) | max probability <0.8 | 1.2041180007310734 /3.4209273427377678 | 50/50 |
| `q8-fixed50-v1` | max probability <0.8 | 1 /1 | 50/50 |
| `e2b-raw99-blend50-v3` (E2B default when opted in) | max probability <0.99 | 1 /1 | 50/50 |
| `e2b-raw99-blend50-v2` (previous numbered prompt) | max probability <0.99 | 1 /1 | 50/50 |

E2B requires an explicit runtime profile. Its default reasoning prompt uses
the native scorer labels; the previous numbered serialization remains available
through its policy identifier. The completed frozen comparison is recorded in
the E2B evidence.
See [E2B settings and evidence](E2B.md).

Adaptive inputs must have **one named question and a text, object, or array state**.
Null/scalar states and multiple questions are explicitly rejected. The
client preserves a structured state in the direct native request and JSON-serializes
it only in the generation prompt. For completed reasoning, text states receive the
existing `"\n\nModel reasoning:\n"` suffix; objects and arrays are passed to
augmented native scoring as `{"original_state": state, "model_reasoning": text}`.
This retains the original structure rather than replacing it with JSON text.
Choice insertion order is preserved; noul uses false/true; score uses
numeric index order and retains expected score/legend. Ties choose the first
candidate. Only state, that question, and any supplied images enter the reasoning prompt, never labels,
benchmark metadata or caller result fields. The supported ordinary chat template
is used with `reasoning_effort:"none"` and explicit
`chat_template_kwargs.enable_thinking=false`. Ordinary native chat thinking is
controlled separately by `--native-chat-reasoning on|off`.

The default generation profile is temperature 0/seed 314159, max_tokens −1 within
the selected context, EOS respected, one slot, matching official MTP4, F16 target
KV and retained Q8_0 assistant draft-cache flags. Gemma 4 assistant attention
shares the target K/V tensors. `--cache q8_0` is an explicit target-cache option.
A 100-word instruction for the released 12B/E4B recipes is a soft instruction,
not an output bound; E2B retains its distinct uncapped reasoning instruction. The default
75-second hard client wall deadline closes the HTTP connection. Explicit runtime
contracts pin their own context/cache settings and bounded phase deadlines.
E2B profiles default to F16 target KV and use a 180-second generation deadline.
Historical Q8-cache receipts and the `measured_profile` flag retain their original
cache definition; selecting the new F16 default does not relabel those results.
HTTP requests serialize by backend origin within the client process. Cancelled
workers retain backend ownership until they exit, even when the caller has
already returned; there is no process-wide SIGALRM. Native calls serialize with chat in
the resident bridge, which refuses context eviction to preserve speculative state.

After a successful direct decision, missing/empty/malformed/non-EOS generation,
context exhaustion, watchdog/network/backend failures or failed augmented scoring
fall back to the saved **policy-calibrated direct** distribution. Partial text is
never scored. Client cancellation aborts the request without returning fallback.
A failed or malformed direct decision remains an error. Successful
completed reasoning is added as context, then the two separately calibrated
distributions are blended with the fixed weight. No model selection, threshold
refitting or tuning occurs in the client.

Responses retain the typed native answer shape. `winnow.adaptive` records version,
policy, reasoning mode, whether the gate was applied, raw gate, route, calibration,
blend/fallback and observed generation/native
usage. Native `usage.output_tokens` remains0 because native decisions generate no
tokens; generated prompt/output tokens are reported separately. If a failed HTTP
request supplied no generation usage, those fields arenull with
`generation_usage_available:false`; unobserved GPU work is not called zero.

Custom context/cache/batch/branch settings keep the same frozen policy constants.
The client reports whether native and chat settings match the measured profile;
changed configurations do not inherit its calibration, quality or latency claims.
Always mode retains the calibrated scoring recipe but does not inherit selective
routing quality or latency results. It still falls back when reasoning cannot complete.

## Validation limits

The supporting adaptive comparisons combine calibration and generated context;
they do not isolate reasoning causality or guarantee improved accuracy.
The new E4B policy has held-out text coverage for choice, Boolean, and rating decisions.
Image routing is supported under explicit contracts; image reasoning quality is
not established. Other policies retain their documented limits.
The 2026.10.05 compatibility patch adds structured-state input coverage; its
35-object Jev follow-up is public-panel evidence, not independent validation of
structured-state quality or a new calibration for any model policy.
Base-pretraining/semantic overlap and production cold/concurrency latency remain
unknown. Mac adaptive, image reasoning quality, multi-GPU and lossless speculation are
not established. Focused integration checks are not new quality benchmarks.
