# Winnow-E2B

E2B supports native decisions, ordinary chat and image-aware adaptive decisions
on Linux/CUDA. It uses its own Q8 target, F16 projector and optional BF16 MTP
assistant. Exact files are pinned in the [asset manifest](../manifests/release-assets-v1.json).
The model and projector are verified in private repository revision
`7439a194a1a948262115b02fb380daf4ecc77369`. Access requires authorization; the public
downloader uses verified local copies for private assets.

## Start and choose a mode

Place the manifest-named files in a local asset directory, then run:

```sh
python3 scripts/winnow.py download --model e2b --vision on --mtp on \
  --asset-dir /path/to/verified-assets --offline
python3 scripts/winnow.py serve --model e2b --vision on --mtp on --context 64k
# In another terminal, matching the server's vision, MTP and context:
python3 scripts/winnow.py decide --model e2b --vision on --mtp on --context 64k \
  --reasoning selective --input examples/adaptive-decision.json
```

Source users must build the matching server first; see [installation](INSTALL.md).
Use `bin/winnow` instead of `python3 scripts/winnow.py` in a matching runtime archive.
The assistant is a separately verified local conversion, not a bundled download.
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

E2B profiles pin 8K or 64K context, F16 target KV, backend temperature sampling,
one native branch, one chat slot, batch/microbatch 1024, auto memory and disabled
context shifting. MTP uses the exact assistant, Q8 draft KV and draft length 4.
`--mtp off` omits the assistant; these profiles have CPU contract coverage and
still need separate live validation. `--vision off` omits the projector.
The default E2B command uses 8K text, MTP off and decision reasoning off.
The examples above explicitly enable MTP with `--mtp on` on every command.

The server and client must use matching context, vision and MTP settings.
Named E2B profiles reject numerical overrides that break their contract.
Other capacities require an explicit operator profile. Backend sampling can be
disabled explicitly with `--backend-sampling off`; historical measurements then
do not describe that configuration. The launcher verifies projector bytes and
pins draft length and sampling; runtime inspection does not attest all of these
settings. This remains an operator-controlled backend contract.

On the previously checked 16 GB RTX 5070 Ti, 64K vision with MTP4 served small
text/image requests at about 8.1 GiB sampled peak GPU memory. This demonstrates
startup, fit and small-request behavior. Full-length 64K generation, image quality
and policy quality at 64K have not been validated.

## Evidence and limits

The current `e2b-raw99-blend50-v2` policy uses canonical option names,
descriptions, null fallbacks and native order. Its option serialization differs
from the original benchmark prompt. Historical measurements below therefore
do not validate this integrated prompt; a new quality comparison remains open.

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
keep these historical results separate from current integration checks.
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
