# Experimental adaptive decision contract

For measured gains, regressions, latency and calibration tradeoffs, see
[reasoning results](REASONING-RESULTS.md).

Use the [quickstart](QUICKSTART.md) for download, launch and client commands.
Native `/v1/systemone` calls remain direct. Adaptive reasoning is explicit,
Linux/CUDA only, one question and no images, with an 8K default.
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
| `e4b-calibrated50-v1` | max probability <0.8 | 1.2041180007310734 /3.4209273427377678 | 50/50 |
| `q8-fixed50-v1` | max probability <0.8 | 1 /1 | 50/50 |

Adaptive inputs must have **one named question and a text, object, or array state**.
Null/scalar states, multiple questions and vision are explicitly rejected. The
client preserves a structured state in the direct native request and JSON-serializes
it only in the generation prompt. For completed reasoning, text states receive the
existing `"\n\nModel reasoning:\n"` suffix; objects and arrays are passed to
augmented native scoring as `{"original_state": state, "model_reasoning": text}`.
This retains the original structure rather than replacing it with JSON text.
Choice insertion order is preserved; noul uses false/true; score uses
numeric index order and retains expected score/legend. Ties choose the first
candidate. Only state and that question enter the reasoning prompt, never labels,
benchmark metadata or caller result fields. The supported ordinary chat template
is used, with reasoning effortnone; no separate thinking mode is claimed.

Generation is temperature0/seed314159, max_tokens−1 within the selected context (default 8K), EOS respected,
one slot, matching official MTP4, q8_0 target/draft KV, n_min0/p_min0. A100-word
instruction is a soft instruction, not an output bound. A75-second hard client
wall deadline closes the HTTP connection. Requests serialize within each client
transport; there is no process-wide SIGALRM. Native calls serialize with chat in
the resident bridge, which refuses context eviction to preserve speculative state.

After a successful direct decision, missing/empty/malformed/non-EOS generation,
context exhaustion, watchdog/network/backend failures or failed augmented scoring
fall back to the saved **policy-calibrated direct** distribution. Partial text is
never scored. A failed or malformed direct decision remains an error. Successful
completed reasoning is added as context, then the two separately calibrated
distributions are blended with the fixed weight. No model selection, threshold
refitting or tuning occurs in the client.

Responses retain the typed native answer shape. `winnow.adaptive` records version,
policy, raw gate, route, calibration, blend/fallback and observed generation/native
usage. Native `usage.output_tokens` remains0 because native decisions generate no
tokens; generated prompt/output tokens are reported separately. If a failed HTTP
request supplied no generation usage, those fields arenull with
`generation_usage_available:false`; unobserved GPU work is not called zero.

Custom context/cache/batch/branch settings keep the same frozen policy constants.
The client reports whether native and chat settings match the measured profile;
changed configurations do not inherit its calibration, quality or latency claims.

## Validation limits

The supporting adaptive comparisons combine calibration and generated context;
they do not isolate reasoning causality or guarantee improved accuracy.
Independent final coverage is choice/rating; Boolean transfer is not established.
The 2026.10.05 compatibility patch adds structured-state input coverage; its
35-object Jev follow-up is public-panel evidence, not independent validation of
structured-state quality or a new calibration for any model policy.
Base-pretraining/semantic overlap and production cold/concurrency latency remain
unknown. Mac adaptive, image reasoning, multi-GPU and lossless speculation are
not established. Focused integration checks are not new quality benchmarks.
