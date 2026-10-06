# Winnow-E2B

E2B supports native decisions, ordinary chat and image-aware adaptive decisions
on Linux/CUDA. It uses its own Q8 target, F16 projector and optional BF16 MTP
assistant. Exact files are pinned in the [asset manifest](../manifests/release-assets-v1.json).
The target, projector and assistant are pinned to model repository revision
`f0f9931c4d44e97091d14c0b8dba118eef32ff1b`. The downloader verifies their sizes
and SHA256 hashes. Existing matching local assets can be reused with `--asset-dir`.

## Start and choose a mode

Download the manifest-pinned files, then run:

```sh
python3 scripts/winnow.py download --model e2b --vision on --mtp on
python3 scripts/winnow.py serve --model e2b --vision on --mtp on --context 64k
# In another terminal, matching the server's vision, MTP and context:
python3 scripts/winnow.py decide --model e2b --vision on --mtp on --context 64k \
  --reasoning selective --input examples/adaptive-decision.json
```

Source users must build the matching server first; see [installation](INSTALL.md).
Use `bin/winnow` instead of `python3 scripts/winnow.py` in a matching runtime archive.
The exact converted assistant is included in the model repository at the same
pinned revision; it does not require a local conversion.
Its pinned upstream and conversion hashes are in [assistant provenance](../manifests/assistants-v1.json).

| Decision mode | Behavior |
|---|---|
| `--reasoning off` | Native direct scoring. |
| `--reasoning selective` | Generate analysis when raw maximum probability is strictly below 0.99, then blend direct and augmented probabilities equally. |
| `--reasoning always` | Attempt the same analysis and blend for every valid decision. |

Both reasoning modes use identity temperatures and retain the saved direct
distribution when generation or augmented scoring fails. Only a nonempty,
untruncated natural-EOS completion within the context and deadline is rescored.
`on` remains an alias for `selective`. Routing runs in the decision client;
the native `/v1/systemone` endpoint remains direct.

The client accepts one question with a text, object or array state. Images use
an ordered `winnow.images` list of base64 image data URLs; the same images reach
direct scoring, generation and augmented scoring. See the [image contract](IMAGE-REASONING.md).

## Chat thinking and runtime settings

Ordinary chat thinking is separate: add `--native-chat-reasoning on` to `serve`
to enable it by default. It defaults to off, and ordinary chat clients can
override the template setting per request. For an explicit per-request opt-out,
send `"chat_template_kwargs":{"enable_thinking":false}`. On this pinned runtime,
`reasoning_effort:"none"` alone did not disable thinking. Internal adaptive
generation always sends the explicit false template setting.

E2B profiles pin 8K or 64K context, default F16 target KV, backend temperature sampling,
one native branch, one chat slot, batch/microbatch 1024, auto memory and disabled
context shifting. MTP uses the exact assistant, retained Q8 draft-cache flags and draft length 4.
The pinned Gemma 4 assistant shares the target K/V tensors, so its effective
attention cache follows the selected target-cache precision.
`--mtp off` omits the assistant; CPU contract checks and the recorded bounded
MTP on/off check cover this setting. `--vision off` omits the projector.
The default E2B command uses 8K text, MTP off and decision reasoning off.
The examples above explicitly enable MTP with `--mtp on` on every command.

The server and client must use matching context, vision and MTP settings.
Named E2B profiles reject numerical overrides that break their contract;
`--cache q8_0` is an explicit target-cache alternative and must also be selected
on the profile-checked client. It retains Q8_0 assistant draft-cache flags and leaves
F16-only backend sampling off.
Other capacities require an explicit operator profile. Backend sampling can be
disabled explicitly with `--backend-sampling off`; historical measurements then
do not describe that configuration. The launcher verifies projector bytes and
pins draft length and sampling; runtime inspection does not attest all of these
settings. This remains an operator-controlled backend contract.

On the checked 16 GB RTX 5070 Ti, 64K vision with MTP4 served text/image
requests at about 8.1 GiB sampled peak GPU memory. Dani also confirms his own
64K testing. The full-panel scores below use the recorded 8K configuration.

## Evidence and limits

The current `e2b-raw99-blend50-v3` policy preserves canonical option names,
descriptions, null fallbacks and native order, and labels them using the scorer's
actual inspected letter labels. The previous numbered `e2b-raw99-blend50-v2`
remains available for reproducibility. Gate, blend, calibration, safe state
augmentation and explicit thinking-off are unchanged.

A private frozen 24-case controlled check reproduced the numbered baseline
exactly and isolated option serialization as a cause of changed reasoning.
Native labels reproduced historical explanations on all 23 inputs without
literal `<` escaping; escaping remains enabled. Template-setting pairs and
same-text native rescoring were identical. The subsequent frozen full-panel
run completed all 3,277 cases using native scorer labels. No policy fitting was
performed on these panels.

| Panel | Direct | Previous numbered | Native labels |
|---|---:|---:|---:|
| Jev verified label accuracy | 175/231 | 195/231 | 202/231 |
| Kev verified label accuracy | 728/1046 | 845/1046 | 851/1046 |
| Typed hard synthetic teacher agreement | 1237/2000 | 1342/2000 | 1366/2000 |

The run used 8,192 context positions, F16 target KV, the official MTP4 assistant,
uncapped generation and natural EOS. All saved direct probabilities matched
the reference exactly. Of 2,584 routed cases, 2,582 completed a blend; two Kev
cases reached the context ceiling and retained their direct result. There were
no request errors. Mean clean serial latency was 1.356, 1.212 and 1.951 seconds
for Jev, Kev and Typed; 16 GPU-overlap cases are excluded from timing only.
All cases remain in the quality denominator. The full-panel comparison is
previously exposed regression evidence; it is separate from the earlier image
generation, final-scoring and projector-memory checks. The previous numbered policy remains
selectable through the lower-level client with `--policy e2b-raw99-blend50-v2`.

The historical 8K F16/backend study used 288 selection cases and 288 disjoint
holdout cases from LogiQA2, PAWS and HelpSteer2. Against the earlier 0.80/50% rule,
the 0.99/50% rule matched 158 versus 155 of 288 labels/ratings. Its paired 95%
top-one interval was −1.74 to +3.82 percentage points; NLL and Brier intervals
also included zero. Estimated serial API mean rose from 1.066 to 1.677 seconds.
Per-source counts changed 48→52, 76→75 and 31→31. HelpSteer2 is subjective rating
agreement; 74 of its 96 holdout cases had prior cross-model scoring exposure.
Public-source pretraining exposure is unknown.

On previously exposed full panels, the same historical rule scored 203/231 Jev,
851/1046 Kev and 1365/2000 Typed, versus 193, 815 and 1340 for 0.80/50%.
Typed measures synthetic teacher agreement. These are regression/transfer
comparisons, and the extra routing increased cost. The earlier synthetic image
panel used a different Q8-KV recipe and does not validate this profile.

[Detailed metrics, per-type results, intervals and source hashes](e2b-evidence.json)
keep historical results, the completed regression run and image checks separate.
Always mode has no independent quality claim.

## Upgrade safely

This version adds the native E2B width-1536 MTP compatibility patch. Rebuild from
the matching source or use its matching runtime package. Older client-only
instructions to reuse the previous 12B/E4B binary do not apply. The runtime lock
is now `9b20caf5c6032f6274350c8d20b1d90e5cbc5b7dfa468ca85c77e198e24845df`.
Older adaptive clients reject this new runtime identity; keep client, manifests
and server together. Existing 12B/E4B weights, policies and default settings are
preserved, but the new binary still requires a bounded compatibility check.

Extract into a new directory and reuse verified models with `--model-dir`.
Preserve your previous directory, launch command, ports and authentication for
rollback. No service configuration is migrated automatically. See [upgrades](RELEASE.md#upgrading).
