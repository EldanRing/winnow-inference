# Runtime validation

These release measurements use **Winnow-12B Q8**, identified by the SHA256 in
[the model manifest](../manifests/models.json). Model-quality and cross-model
results are in the [benchmark report](BENCHMARKS.md).

## RTX 5070 Ti: 64K context and vision

The validated local profile uses full GPU model/projector offload, Q8 KV,
65,536 context positions, four decision branches, one chat slot and exclusive
context scheduling. The host has 96 GB system RAM. The RTX 3060 was not used.

| Measurement | Result |
|---|---:|
| Populated decision prefix | 65,022 positions, including 1,024 image positions |
| Retrieval questions correct | 4 / 4, beginning/middle/end/image |
| Cold four-question response | 25.00 s |
| Cached four-question response | 143.0 ms median of three repeats |
| Short chat generation | 55.5 tokens/s, three 512-token outputs |
| Long multimodal prompt | 62,435 positions |
| Long prompt processing | 2,893.9 tokens/s |
| Long-context generation | 46.9 tokens/s, 512-token output |
| Long-context first token | 21.75 s, cold |
| Long-context total response | 32.64 s |
| Peak total GPU use during capacity smoke | 15.01 GiB |
| Peak process-tree system RAM during capacity smoke/loading | 12.25 GiB PSS |

The GPU peak includes a 450 MiB desktop baseline. Memory sampling was every
500 ms. These are observed peaks, not minimum installed RAM requirements.
The long-generation result is one cold sample. Repeated-state latency assumes
the decision prefix remains cached; switching between chat and decisions under
exclusive scheduling evicts the idle KV context.

Full-context streaming chat retrieval also passed at 64,998 prompt positions.
The release artifact passed API/cache/chat/streaming/vision, lifecycle/mixed API,
and selected/full-head checks. The populated capacity fixtures are synthetic;
they establish usable allocation and retrieval behavior, not broad long-context
or general chat-quality equivalence to the base model.

## Other runtime profiles

An earlier Winnow checkpoint passed the same runtime's 128K + vision profile on
the RTX 5070 Ti with one decision branch and Q8 KV. The matching earlier model
also passed 64K + vision on an M4 Pro with 24 GB unified memory, one decision
branch and F16 KV. These are runtime compatibility observations, separate from
this release artifact's 64K CUDA measurements above.

The 32K automatic-memory configuration encountered a vision-allocation failure;
it is not a validated release profile. Use the explicit exclusive setting in
[the installation command](INSTALL.md#launch-profiles) for the measured CUDA setup.

## Repeatable checks

The final release-package smoke passed on September 21, 2026 with the exact
model/projector hashes in the manifest and the same server binary used for the
64K capacity measurements:

- 15 functional API/cache/chat/streaming/vision checks.
- 23 HTTP/authentication/input-boundary checks.
- Eight selected/full-head cases covering 15 questions: no changed winners;
  maximum absolute probability difference 0.000916.

The owned-server checks took 12.55 seconds after file hashing. This short smoke
did not repeat the full-context run or general model-quality benchmarks.

The included release check starts only its own authenticated localhost servers
and tests API boundaries, chat, streaming, vision, authentication, and selected
versus full vocabulary-head consistency. A 4,096-token output allowance lets
ordinary replies end naturally. Capacity and cancellation probes are explicit
additional options. See [maintainer checks](RELEASE.md).

Source-package tests verify private-file/history exclusions, refusal of dirty
or unapproved trees, and model-corruption detection. The pinned runtime has
built and passed native unit checks on CUDA, Linux CPU and Apple Silicon Metal.
The Blackwell CUDA runtime image is approximately 1.71 GB uncompressed and
excludes weights and build tools. Its exact server binary also passed the local
64K-configured short smoke: 15 functional checks, 23 HTTP checks and 15-question
selected/full-head parity. This does not add a new populated-context benchmark.
The image contents, unprivileged user, launcher and downloader were checked.
GPU inference inside Docker was not exercised because the local Docker Desktop
engine has no NVIDIA container runtime. The binary smoke ran directly on the
host GPU; it is not represented as a Docker GPU pass.
