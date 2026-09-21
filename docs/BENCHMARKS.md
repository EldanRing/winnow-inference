# Winnow-12B benchmark report

Winnow-12B is evaluated in **BF16 and Q8** against Kev, Laya, and hosted Jev. The local-model comparison uses one RTX PRO 5000 Blackwell, one model at a time, with localhost clients. RTX 5070 Ti deployment results are reported separately. Measurements: September 21, 2026.

The downloadable model is a Gemma 4 12B IT LoRA fine-tune with the adapter merged into the weights. **Training data is private and is not distributed in this repository.**

## Decision quality

| Model | JevBench public (231) | Kev-v9 clean (1,046) | v9 additions (390) | Typed teacher agreement (2,000) |
|---|---:|---:|---:|---:|
| Winnow-12B Q8 | 85.71% | 81.55% | 69.23% | 70.00% |
| Winnow-12B BF16 | 85.28% | 81.45% | 69.23% | 70.20% |
| Jev 1.13 (OpenRouter) | 85.71% | 87.00% | 85.64% | 73.80% |
| Kev-9B BF16 | 76.19% | 78.20% | 68.97% | 71.60% |
| Kev-4B BF16 | 71.43% | 75.62% | 63.59% | 65.75% |
| Laya Typed Decisions | 53.68% | 55.64% | 36.67% | 76.75% |
| Laya English | 58.44% | 52.96% | 32.05% | 36.10% |

JevBench is the 231-record public subset, not the official composite leaderboard score. Kev-v9 contains 1,264 test records; its predefined clean/knowable accuracy denominator is 1,046, including 390 new-v9 decisions. “Clean” here is the benchmark scoring category, not a claim about training-data overlap. Typed results measure agreement with synthetic teacher labels across 400 test cases / 2,000 decisions.

Laya Typed Decisions was trained on that dataset’s separate training partition. Laya’s configured context limits truncate some public inputs. Earlier Kev-v4 measurements informed Winnow’s later data refinement; the inherited and new-v9 results must not be described as wholly untouched task families. Winnow Q8 and Kev-9B differ by only one correct answer on the 390 new-v9 decisions.

## Calibration

| Model | JevBench Brier | JevBench ECE | Kev-v9 Brier | Kev-v9 ECE | Typed ECE |
|---|---:|---:|---:|---:|---:|
| Winnow-12B Q8 | 0.2057 | 0.0742 | 0.2855 | 0.0980 | 0.1571 |
| Winnow-12B BF16 | 0.2042 | 0.0655 | 0.2857 | 0.0989 | 0.1551 |
| Jev 1.13 (OpenRouter) | 0.1795 | 0.0474 | 0.1914 | 0.0308 | 0.0370 |
| Kev-9B BF16 | 0.3541 | 0.1623 | 0.3182 | 0.1008 | 0.0591 |
| Kev-4B BF16 | 0.4543 | 0.1925 | 0.3578 | 0.1168 | 0.1254 |
| Laya Typed Decisions | 0.5088 | 0.0712 | 0.5406 | 0.0518 | 0.2147 |
| Laya English | 0.5321 | 0.0904 | 0.5825 | 0.1301 | 0.1752 |

Lower is better for the listed Brier/ECE metrics. The shared evaluator normalizes candidate distributions consistently; no temperature was fitted on these evaluation outputs. Typed distribution agreement is retained in the machine-readable report. Candidate probabilities and entropy-based confidence are not a guarantee of correctness.

## Measured memory

| Model | VRAM loading | VRAM idle | VRAM inference | RAM loading (PSS) | RAM inference (PSS) |
|---|---:|---:|---:|---:|---:|
| Winnow-12B Q8 | 12.82 | 14.33 | 14.36 | 12.18 | 0.59 |
| Winnow-12B BF16 | 23.22 | 24.77 | 24.80 | 22.58 | 0.72 |
| Kev-9B BF16 | 32.29 | 15.24 | 21.87 | 47.33 | 2.42 |
| Kev-4B BF16 | 17.57 | 8.28 | 14.89 | 23.11 | 2.49 |
| Laya Typed Decisions | 1.82 | 1.92 | 4.94 | 3.25 | 2.05 |
| Laya English | 1.84 | 1.92 | 4.00 | 3.25 | 2.05 |

All memory values are GiB. GPU figures are total device observations with a 2 MiB idle baseline in the pod. System RAM is process-tree proportional set size (PSS). A 500 ms sampler records phase-specific peaks; these are observed usages, not minimum installed-RAM guarantees. Loading includes each runtime’s initialization/merge, while inference can retain allocator reservations. Disk size and GPU memory are separate quantities.

Winnow Q8 uses 14.36 GiB peak inference VRAM versus 21.87 GiB for the tested Kev-9B BF16 configuration. That comparison includes different precision and runtime choices; quantized Kev was not measured. Both Winnow BF16 and Q8 use Q8 KV. Laya stores FP32 weights and uses BF16 autocast. Jev’s hosted GPU/RAM usage is not exposed.

## Pod latency and throughput

B1/B8/B16/B32/B64 mean that many questions in one shared-state HTTP request, not separate concurrent clients. Each fixture uses ten consecutive warm samples after priming and then ten distinct-state cold samples. Values below are medians. Hosted Jev response time includes networking and is not pooled into these localhost tables.

### Short common state — warm latency (ms)

| Model | B1 | B8 | B16 | B32 | B64 |
|---|---:|---:|---:|---:|---:|
| Winnow-12B Q8 | 22.1 | 93.0 | 187.8 | 367.6 | 736.4 |
| Winnow-12B BF16 | 33.5 | 108.0 | 216.6 | 427.3 | 856.0 |
| Kev-9B BF16 | 66.1 | 72.5 | 117.6 | 229.2 | 484.2 |
| Kev-4B BF16 | 65.6 | 72.7 | 77.5 | 148.9 | 312.9 |
| Laya Typed Decisions | 22.8 | 27.3 | 29.8 | 35.8 | 57.6 |
| Laya English | 23.8 | 28.1 | 30.7 | 36.2 | 57.4 |

### Short common state — cold latency (ms)

| Model | B1 | B8 | B16 | B32 | B64 |
|---|---:|---:|---:|---:|---:|
| Winnow-12B Q8 | 52.7 | 121.4 | 220.4 | 399.4 | 772.4 |
| Winnow-12B BF16 | 74.2 | 147.6 | 258.4 | 467.9 | 897.4 |
| Kev-9B BF16 | 65.6 | 72.8 | 119.1 | 231.8 | 494.2 |
| Kev-4B BF16 | 65.8 | 72.5 | 77.6 | 150.6 | 319.2 |
| Laya Typed Decisions | 23.3 | 27.5 | 30.0 | 35.5 | 58.0 |
| Laya English | 24.1 | 28.1 | 30.9 | 36.5 | 57.9 |

### Short common state — warm decisions per second

| Model | B1 | B8 | B16 | B32 | B64 |
|---|---:|---:|---:|---:|---:|
| Winnow-12B Q8 | 45.2 | 86.0 | 85.2 | 87.0 | 86.9 |
| Winnow-12B BF16 | 29.8 | 74.1 | 73.9 | 74.9 | 74.8 |
| Kev-9B BF16 | 15.1 | 110.3 | 136.0 | 139.6 | 132.2 |
| Kev-4B BF16 | 15.2 | 110.0 | 206.5 | 214.9 | 204.5 |
| Laya Typed Decisions | 43.8 | 292.9 | 536.2 | 893.7 | 1111.7 |
| Laya English | 41.9 | 284.9 | 521.8 | 883.6 | 1114.6 |

### Longer shared state — warm latency (ms)

| Model | B1 | B8 | B16 | B32 | B64 |
|---|---:|---:|---:|---:|---:|
| Winnow-12B Q8 | 23.3 | 100.7 | 198.0 | 391.9 | 788.9 |
| Winnow-12B BF16 | 34.9 | 112.9 | 226.2 | 446.2 | 893.4 |
| Kev-9B BF16 | 71.0 | 77.3 | 82.7 | 135.1 | 231.4 |
| Kev-4B BF16 | 71.2 | 78.1 | 83.6 | 102.6 | 179.3 |
| Laya Typed Decisions | 24.9 | 67.0 | 142.2 | 314.6 | 647.4 |
| Laya English | 25.9 | 45.3 | 89.2 | 191.6 | 412.5 |

### Longer shared state — cold latency (ms)

| Model | B1 | B8 | B16 | B32 | B64 |
|---|---:|---:|---:|---:|---:|
| Winnow-12B Q8 | 181.3 | 256.4 | 357.4 | 549.7 | 947.8 |
| Winnow-12B BF16 | 195.4 | 273.4 | 388.0 | 607.1 | 1054.3 |
| Kev-9B BF16 | 151.8 | 158.6 | 164.7 | 219.0 | 316.5 |
| Kev-4B BF16 | 133.1 | 139.4 | 145.2 | 164.3 | 240.8 |
| Laya Typed Decisions | 24.9 | 67.8 | 143.7 | 317.7 | 655.2 |
| Laya English | 25.5 | 45.3 | 88.6 | 191.8 | 415.1 |

### Longer shared state — warm decisions per second

| Model | B1 | B8 | B16 | B32 | B64 |
|---|---:|---:|---:|---:|---:|
| Winnow-12B Q8 | 42.9 | 79.5 | 80.8 | 81.7 | 81.1 |
| Winnow-12B BF16 | 28.7 | 70.9 | 70.7 | 71.7 | 71.6 |
| Kev-9B BF16 | 14.1 | 103.6 | 193.4 | 236.9 | 276.6 |
| Kev-4B BF16 | 14.0 | 102.4 | 191.3 | 311.8 | 356.9 |
| Laya Typed Decisions | 40.2 | 119.3 | 112.6 | 101.7 | 98.9 |
| Laya English | 38.6 | 176.7 | 179.5 | 167.0 | 155.1 |

Laya English truncates the longer speed fixture; its timing does not represent processing the full input. Kev uses the pinned author runtime, Flash Linear Attention 0.5.2 and Triton 3.7.1. Optional causal-conv1d is absent, so convolution uses the reference PyTorch path. These are results of the tested configurations, not maximum possible competitor speeds.

## RTX 5070 Ti: 64K plus vision

Q8 weights; Q8 KV; four decision branches; one chat slot; exclusive memory scheduling; full GPU offload. Capacity is 65,536 positions including formatting, images, questions, and replies. The populated decision fixture has **65,022 shared-prefix positions, including 1,024 image positions**.

| Measurement | Observed result |
|---|---:|
| Peak GPU memory, 64K smoke | 15.01 GiB total device, including 450 MiB desktop baseline |
| Peak CPU memory, 64K smoke | 12.25 GiB process-tree PSS including loading; host has 96 GB RAM |
| Four decisions, cold | 25.00 seconds |
| Four decisions, cached | 143.0 ms median, 3 samples |
| Short-prompt generation | 55.5 output tokens/s; median of 3 fixed 512-token outputs |
| Long multimodal prompt | 62,435 prompt positions; 512 output tokens |
| Long prompt processing | 2,893.9 tokens/s, native timing |
| Long generation | 46.9 output tokens/s |
| Long first-content latency | 21.75 seconds |
| Long total response | 32.64 seconds |

The long-generation figures are one cold sample; the decision and generation fixtures are separate. Capacity/retrieval and chat/vision functionality passed. This does not establish unchanged general chat quality after fine-tuning. Chat and decisions share loaded weights; exclusive mode switches their KV contexts rather than keeping independent full-capacity contexts resident simultaneously. Switching can evict a cached prefix.

The 32K automatic-memory vision profile encountered an allocation failure. The advertised 5070 Ti recipe explicitly uses the successful 64K exclusive profile. These results should not be extrapolated to every automatic profile or lower installed system RAM.

## Fine-tuning and quantization analysis

Stock Gemma 4 12B IT and Winnow use the same inference engine and settings within each precision. BF16 below means BF16 weights with Q8 KV, not an all-BF16 cache.

| Model | JevBench public | Kev-v9 clean | v9 additions | Typed teacher agreement |
|---|---:|---:|---:|---:|
| Winnow-12B Q8 | 85.71% | 81.55% | 69.23% | 70.00% |
| Winnow-12B BF16 | 85.28% | 81.45% | 69.23% | 70.20% |
| Stock Gemma 4 12B IT Q8 | 83.55% | 78.68% | 66.67% | 71.30% |
| Stock Gemma 4 12B IT BF16 | 83.55% | 78.30% | 65.90% | 71.40% |

Q8 minus BF16 for Winnow: +0.43 percentage points on public JevBench, +0.10 on Kev-v9 clean, 0.00 on the additions, and −0.20 on typed teacher agreement. Small observed differences do not imply identical answers on arbitrary inputs. Fine-tuning improved the first three comparisons against stock; typed teacher-label accuracy decreased.

A 32-item development adapter/merged-model probe retained 31 answers, with correct count changing 24→23 and maximum probability drift 0.229293. A matched-placement recheck reproduced it. All release scores above evaluate the saved exported artifacts directly; none are transferred from the adapter.

## Artifacts and reproduction

Model files: [EldanRing/Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B). The BF16 pod evaluation served a BF16 GGUF converted from the released merged BF16 safetensors. The Q8 evaluation used the downloadable Q8 GGUF. The corresponding file hashes and serving settings appear in [benchmarks.json](benchmarks.json), with download hashes in [the model manifest](../manifests/models.json).

Both pod Winnow configurations: 8,192 context, four decision branches, one chat slot, Q8 KV, selected head, optimized pipeline, automatic memory policy, full GPU offload, no projector. The separate 5070 Ti capability test enables vision at 65,536 context with exclusive memory scheduling.

Pinned datasets/runtimes: [JevBench](https://github.com/fstandhartinger/jevbench), [Kev](https://github.com/jaredpalmer/kev), [Laya](https://github.com/NandhaKishorM/laya), and [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions). Revisions, source-manifest hashes, serving settings and latency distributions are retained in the machine-readable report. The private orchestration used to run all competitor environments is not distributed with this inference server.

See [evaluation and reproduction](EVALUATION.md) for pinned upstream commands, metric definitions and the typed-dataset method. To reproduce runtime behavior on another checkpoint, use the [release check and launch profiles](INSTALL.md#quick-verification). For timing, run `python3 scripts/bench.py --output results/timings --repeats 10` against your running server. Keep native answer ordering and case grouping. Score all 231 JevBench public records, the predefined Kev-v9 clean/knowable subset, and all 2,000 typed decisions; never fit calibration on those test outputs.

## Jev collection

Hosted model revisions: `["typesafe/jev-1.13-20260917"]`. Total recorded cost: **$0.046054**. Gold labels and local predictions were not sent. A format pilot was followed by durable collection with duplicate prevention. Coverage and probability handling are in [benchmarks.json](benchmarks.json).

## Fine-tuning data overlap check

The checksum-verified initial 16,000-example training mixture and 26,000-example
refinement mixture were scanned against all 231 JevBench records, 1,264 Kev-v9
records, and 400 typed-decisions cases. The combined training inputs contained
25,999 unique normalized inputs because refinement replays earlier data.

The audit found **zero exact normalized input or state matches** and **zero
high-containment state candidates**. Normalization casefolds alphanumeric text;
the containment check uses five-word state shingles at an 85% threshold, with
at least 20 shingles. Question IDs and label fields are excluded from matching.

This audit covers the fine-tuning inputs, not the base model's pretraining. It
does not establish absence of conceptual similarity, source-family overlap, or
development exposure. Earlier Kev-v4 results informed refinement choices.

