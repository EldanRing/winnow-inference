# Evaluation method and reproduction

The [results](BENCHMARKS.md) and [machine-readable measurements](benchmarks.json)
identify the artifacts, runtime revisions, dataset pins and hardware. This guide
uses the public upstream evaluators; the internal campaign orchestration is not
part of the inference package. No training data is needed for these evaluations.

## Serve the measured decision profile

From this repository, build and verify the model as described in
[INSTALL.md](INSTALL.md). The pod quality/speed comparison used an 8K decision
profile with no vision projector:

```sh
python3 scripts/serve.py --model models/gguf/Winnow-12B-Q8_0.gguf \
  --context 8192 --decision-context 8192 --cache q8_0 \
  --decision-parallel 4 --chat-parallel 1 --memory auto
```

BF16 used a BF16 GGUF converted from the released merged safetensors with the
pinned llama.cpp converter, retaining the same serving settings and Q8 KV.
After building the pinned runtime, the optional conversion uses an isolated
Python environment (conversion dependencies are not needed for GGUF serving):

```sh
hf download EldanRing/Winnow-12B \
  --revision b2b14213dfa252e6d6b543c8b334762e51772d29 \
  --exclude 'gguf/*' 'docs/*' --local-dir models/Winnow-12B-BF16
python3 -m venv .venv/convert-bf16
.venv/convert-bf16/bin/python -m pip install \
  -r .runtime/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
.venv/convert-bf16/bin/python .runtime/llama.cpp/convert_hf_to_gguf.py \
  models/Winnow-12B-BF16 --outtype bf16 --outfile models/Winnow-12B-BF16.gguf
```

Supply that GGUF to the same 8K serving command above. The exact BF16 GGUF hash
used in the campaign is recorded in `benchmarks.json`; conversion metadata or
library changes can alter a file hash, so record your resulting artifact as well.
Use the separate 64K + vision recipe in the installation guide for that capacity
check; do not mix its performance numbers with the pod comparison.

## JevBench public subset

Use the frozen upstream revision and its native `/v1/systemone` adapter. Run the
following from this repository in a second terminal while the server is active:

```sh
git clone https://github.com/fstandhartinger/jevbench.git .reference/jevbench-public
git -C .reference/jevbench-public checkout c6004e008ffba24aec091261ca1a5c02f7324702
mkdir -p results/jevbench-public
PYTHONPATH=.reference/jevbench-public python3 -m jevbench.cli run \
  --tasks .reference/jevbench-public/datasets/public/easy.jsonl,.reference/jevbench-public/datasets/public/original.jsonl,.reference/jevbench-public/datasets/public/hard.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8091 --model Winnow-12B \
  --key-env '' --price-in-per-m 0 --price-out-per-m 0 --reserve-usd 0 \
  --results results/jevbench-public/predictions.jsonl \
  --ledger results/jevbench-public/ledger.jsonl \
  --raw-dir results/jevbench-public/raw \
  --manifest results/jevbench-public/run.json
PYTHONPATH=.reference/jevbench-public python3 -m jevbench.cli summarize \
  --tasks .reference/jevbench-public/datasets/public/easy.jsonl,.reference/jevbench-public/datasets/public/original.jsonl,.reference/jevbench-public/datasets/public/hard.jsonl \
  --results results/jevbench-public/predictions.jsonl \
  --public-export results/jevbench-public/summary.json
```

This command calls only the local server. Zero prices identify unbilled local
calls, not a claim of zero hardware cost. For an authenticated server, use an
environment variable via `--key-env` instead of the empty value. Reuse an existing
source clone only when it is at the stated revision; choose a fresh results
directory for each model/run.

All 231 public items are evaluated once in easy/original/hard file order. Only
state and question `type`, `instructions`, and `criteria` enter model requests;
labels and other scoring metadata do not. Noul outputs map to `yes`/`no`, choices
to their exact option keys, and score outputs to stringified level indices.

This reproduces the public accuracy/calibration evaluation, not the official
JevBench composite or its held-out evaluation. The upstream CLI's timing and
cost bookkeeping are separate from our shared-state throughput fixtures.

## Kev transfer-v9

Use upstream Kev at `bd058057ad0aa9df3dd6d14e3542c95a5ce367b2`. Follow that
revision's installation instructions in a separate environment; it requires
Python 3.12+ and evaluation dependencies even for a remote endpoint. No Kev
weights are needed to score Winnow through its server.

From the Kev checkout with that environment active:

```sh
python -m kev.benchmark --remote http://127.0.0.1:8091 \
  --remote-model Winnow-12B --suite evals/v9/transfer-v9 --allow-test \
  --out /path/to/new/kev-v9-results
```

This evaluates all 1,264 test records. The headline `clean` metrics include only
records with `variant == "clean"` and `source != "unknowable"`: 1,046 decisions.
For the separate 390-decision additions metric, remove IDs found in the pinned
`evals/v4/transfer-v4` test split before applying the same eligibility rule.
The `--allow-test` flag selects the test partition; no calibrator is fitted.

## Typed-decisions dataset

The comparison uses `LocalLLaMA/typed-decisions` at revision
`ea9306458d6e9563628369a3d1e72e362fb381d2`, file
`all/test-00000-of-00001.parquet`: 400 cases, 2,000 questions. For each case, parse
its JSON `state` and `questions`, send one `/v1/systemone` request, and retain the
five returned answers. Read `gold` only in scoring, never in the request.

Accuracy means agreement with the synthetic teacher label. Choice uses the
returned choice; noul uses `p(true) >= 0.5`; score uses the most probable level,
not the rounded expected score. The teacher distribution is based on three
teacher samples. Laya Typed Decisions was trained on the separate training split;
this metric is teacher agreement rather than independent ground-truth accuracy.

## Metric definitions

- **Accuracy:** fraction of eligible decisions whose selected label equals the
  reference label. JevBench uses lexicographic label order to break equal maxima;
  Kev uses the requested label order. Both use the maximum-probability score
  level for discrete score accuracy.
- **Brier:** mean of `sum((p[k] - target[k])**2)` over decisions, without dividing
  by the number of classes. JevBench/Kev targets are one-hot labels. The typed
  teacher-Brier compares distributions and includes only choice/noul decisions.
- **ECE:** ten equal-width confidence bins across `[0,1]`, with 1 in the final
  bin; sum each bin's fraction of decisions times its absolute gap between mean
  maximum candidate probability and accuracy. This uses top-label probability,
  not the API's entropy-based `confidence` field.
- **Probability validation:** use the pinned suite's native validator for every
  model. JevBench permits rounding renormalization within 0.02 of total mass 1
  and separately records strict validity within 0.001; malformed outputs count
  as incorrect. Kev validates exact option keys and normalizes within its
  option-count-based rounding tolerance. Do not drop failed items or invent
  missing probabilities.
- **Latency/throughput:** one shared-state request contains B1/B8/B16/B32/B64
  questions. Prime once, take ten consecutive warm samples, then ten distinct
  state-prefix samples. Cold means a prefix miss, not model loading. Decisions/s
  is batch size divided by median seconds. Ten-sample p95 is descriptive, not a
  production tail-latency estimate. Hosted Jev includes network/provider time.
- **Memory:** 500 ms sampling of total device VRAM and process-tree system RAM
  PSS, separated into load, idle and inference phases. This measures observed
  peaks, not a proven minimum installed RAM requirement.

The public runtime probes (`scripts/check.py`, `scripts/bench.py`, and
`scripts/release_check.py`) exercise API behavior, capacity and timing. They do
not substitute for these external decision-quality datasets.
