# API and runtime behavior

## Typed decisions

`POST /v1/systemone` accepts:

```json
{
  "model": "Winnow-12B",
  "state": {"paid": true, "priority": "high"},
  "questions": {
    "paid": {"type": "noul", "instructions": "Has the customer paid?"},
    "route": {
      "type": "choice", "instructions": "Which priority is recorded?",
      "criteria": {"low": null, "high": null}
    },
    "rating": {
      "type": "score", "instructions": "How urgent is this priority?",
      "criteria": ["not urgent", "moderately urgent", "very urgent"]
    }
  },
  "winnow": {"diagnostics": true}
}
```

State and instructions can be text, an object, or an array. State cannot be null.
Questions are a named map; IDs identify outputs and are not model instructions.
There must be 1–256 questions and 2–64 alternatives per question.

- `noul`: returns `{"type":"noul","noul":p_true}`. Optional criteria provide
  descriptions under `false` and `true`.
- `choice`: criteria are a map from result keys to descriptions. Null descriptions
  use the key as the option text. Non-null descriptions retain both key and text.
  Returns the selected key, a probability map, and confidence.
- `score`: criteria are an ordered array. Returns a probability map keyed by
  zero-based indices, an expected value on `0..K-1`, confidence, and a legend.

The response envelope is `{"model":...,"answers":{...},"usage":{...}}`.
`input_tokens` counts the logical shared prefix once plus all submitted question
suffixes, including duplicates. Physical work after prefix reuse/deduplication is
reported separately in diagnostics. Decisions generate no answer tokens, so
`output_tokens` is zero.

## Probabilities

The model sees a letter-labeled question. Winnow selects verified single-token
labels, reads their logits, and normalizes over those candidate labels:
`p_i = softmax(logit_i / temperature)`. These are conditional answer probabilities,
not full-vocabulary probabilities or measured frequencies of correctness.

Choice/score confidence is `1 - entropy(p)/log(K)`. It measures concentration of
the answer distribution. It is not a calibrated correctness guarantee. Temperature
defaults to 1; a different temperature requires validation/calibration on held-out
examples for the intended workload.

The selected head reads exactly the model's post-normalization hidden states and
projects only the candidate rows of the output tensor, including Gemma's final
logit softcap. The full-head reference uses normal vocabulary projection and then
selects those same rows. Quantized backend arithmetic can produce small numeric
differences; the supplied parity probes measure them rather than assume equality.

## Vision and diagnostics

Optional Winnow extensions are contained in one `winnow` object:

```json
{
  "diagnostics": true,
  "temperature": 1.0,
  "reuse_prefix": true,
  "images": ["data:image/png;base64,..."]
}
```

Images are ordered, decoded, and identified by content hashes. Changing an image
invalidates its cached state. Decision images must be still-image data URLs,
not local paths or remote URLs. A matching `--mmproj` is required. Maximum: 16
images, 24 MiB encoded per image, 32 MiB for the whole HTTP request. Actual model
context and preprocessing limits still apply. Noncausal image token blocks must
fit `--ubatch`; they are never silently split into a different attention pattern.

Diagnostics add candidate logits to answers and timing/cache metrics under the
response's `winnow` field: actual input/image positions, prefix reuse, unique
questions, waves, decode time, cooperative chat time, queue time, context evictions,
capacity, selected/full head and memory policy. `backend_allocated_bytes` is a
backend/device-wide memory observation, not a precise per-request allocation.

`POST /v1/winnow/inspect` accepts the same body and reports token/position counts,
label IDs, and runtime information without evaluating model answers. It can
allocate a decision context and preprocess images; it is not a zero-cost endpoint. `winnow.include_token_ids: true` additionally returns
prompt token IDs for rendering/export parity checks; it is opt-in.

Both custom endpoints use llama-server's API-key middleware. Configure
`--api-key-file` and send `Authorization: Bearer <key>` when authentication is
needed. Included checks read the key through `WINNOW_API_KEY_FILE` or
`WINNOW_API_KEY`; credentials are excluded from their saved request records.

## Chat and concurrency

Normal `/v1/chat/completions`, streaming SSE, `/v1/models`, `/health`, and other
llama-server endpoints remain available. Chat uses the original chat template and
full vocabulary head. Vision chat uses standard OpenAI-style image content parts.
See the pinned llama-server documentation for its complete chat interface.

One inference thread owns both contexts and the projector. HTTP callers enqueue
work; decision requests are processed one at a time, with their unique questions
batched into configurable parallel branches. They are not independent worker
processes loading their own weights. Chat can make progress between decision
chunks when both contexts fit. Long GPU operations are not preempted midway.

With `--memory auto`, Winnow first attempts to admit the decision context alongside
chat. If allocation fails, it waits for active chat to finish and releases its
idle context before retrying. `--memory exclusive` skips that first allocation
attempt. The latter trades mixed execution for predictable memory residency.
A subsequent chat request releases the idle decision context and restores chat.
The model and projector stay loaded. Reported context evictions explain cache loss.

Disconnected decision callers are cancelled between evaluation chunks. Cancellation
clears potentially partial decision state; the next request starts safely. Pending
queue capacity is 128 decision requests; overflow returns HTTP 429. The HTTP layer
also has its own worker limits. Invalid schemas/capacity requests return 400;
backend failures return 500; an unavailable decision service returns 503.

## Compatibility boundaries

This is a local implementation of Jev's typed decision request/response shape.
It does not reproduce Jev's trained model, calibration, image protocol, hosted
service, quotas, tools, or every future SDK feature. `jev-latest` is accepted as
an input alias; the response identifies the loaded model. Winnow extensions should
be ignored by clients that only need the core contract.

The decision engine currently supports merged Gemma 4 GGUF weights, including
Winnow-12B. Startup LoRA adapters and speculative decoding disable the decision
service because the selected projection/shared-context path does not support
those variants. Use merged weights. The tested launcher runs a single loaded
model, not llama-server router mode. CPU-only and multi-GPU model sharding are
not release profiles. No benchmark or runtime command calls a paid API.
