# Inference architecture

Winnow adds typed decisions to a pinned llama.cpp server. One inference thread
owns a single loaded model and optional vision projector, with separate chat and
decision KV contexts. The implementation currently targets merged Gemma 4 GGUF.

`native/protocol.h` validates and renders the typed request. It selects from
verified single-token answer labels and maps logits back to noul, choice and
ordered score results. Probabilities are normalized over the supplied options;
confidence measures distribution concentration, not calibrated correctness.

`native/planner.h` shares the state prefix, deduplicates identical questions,
extracts common question prefixes and schedules suffixes in capacity-aware waves.
Cache reuse requires token-prefix and image-content identity. Changed state or
images invalidate the corresponding cached work.

`native/engine.h` evaluates the shared prefix and question branches. The selected
head projects only answer rows from post-normalization hidden states and applies
Gemma's final logit softcap. A full-vocabulary reference is retained for parity
checks. Ordinary chat always uses its normal vocabulary head and chat template.

`native/bridge.cpp` integrates decisions with llama-server's inference queue.
Chat progresses between decision chunks when both KV contexts fit. Exclusive
admission releases an idle context when necessary while retaining model weights.
Cancellation clears partially evaluated state before the next request. The API
reports queue time, cache work and context evictions when diagnostics are enabled.

The runtime patches provide borrowed-model contexts, bounded sliding-window KV
forks, tied-embedding handling, selected-head evaluation and server routing.
`runtime.lock.json` pins the upstream commit and verifies every patch and modified
upstream file. Native Winnow files are versioned with the public source commit.

See [API.md](API.md) for the contract and supported boundaries,
[VALIDATION.md](VALIDATION.md) for measured configurations, and
[RELEASE.md](RELEASE.md) for artifact verification and publication.
