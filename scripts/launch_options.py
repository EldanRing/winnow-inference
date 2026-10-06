"""Small public defaults and advisory memory estimates; never a fit guarantee."""
from assets import MODELS, MODEL_ALIASES, canonical_model, selection

DEFAULT_CONTEXT = 8192
# Historical measured baselines, not measurements of arbitrary overrides.
BASELINES = {
    'q8': '8K text + MTP: 14.75 GiB sampled',
    'nv4': '8K vision + MTP: 11.50 GiB sampled',
    'e4b': '8K direct text: 8.56 GiB; 64K direct vision: 10.72 GiB',
    'e2b': '64K vision + MTP4, F16 KV: 8.12 GiB sampled peak on small requests; no full-context generation test',
}


def context_size(value):
    text = str(value).lower()
    size = int(text[:-1]) * 1024 if text.endswith('k') else int(text)
    if size < 512:
        raise ValueError('Context must be at least 512 tokens')
    return size


def memory_estimate(model, context=DEFAULT_CONTEXT, vision='off', mtp='off', reasoning='off'):
    spec, kinds = selection(model, reasoning, mtp, vision)
    weights = sum(spec[k]['bytes'] for k in kinds) / 1024**3
    # Coarse overhead allowance for default q8 KV / four native branches / one chat slot.
    # Deliberately a range: tensor layout, image tokens and allocator overhead vary.
    context_overhead = context / 8192 * (0.10 if canonical_model(model) in {'e4b-q8', 'e2b-q8'} else 0.18)
    overhead = 0.8 + context_overhead + (0.8 if mtp == 'on' else 0) + (0.4 if vision == 'on' else 0)
    low, high = weights + overhead, weights + overhead + (3.0 if mtp == 'on' else 2.0)
    return round(low, 1), round(high, 1)


def memory_note(model, context, vision, mtp, reasoning):
    lo, hi = memory_estimate(model, context, vision, mtp, reasoning)
    return (f'{model}: context {context}, vision {vision}, MTP {mtp}, reasoning {reasoning}; '
            f'estimated GPU memory {lo:.1f}–{hi:.1f} GiB. Estimate only; image size, cache, '
            'batch, slots and hardware change usage; no fit guarantee.')


def preset_list():
    lines = ['Preset  Context  Vision  MTP  Reasoning  Estimated GPU memory',
             '------  -------  ------  ---  ---------  --------------------']
    for name in MODEL_ALIASES:
        lo, hi = memory_estimate(name)
        lines.append(f'{name:6}  8K       off     off  off        {lo:.1f}–{hi:.1f} GiB')
    lines += ['', 'All presets are defaults; override --context (e.g. 4k, 16k, 65536),',
              '--vision on|off, --mtp on|off and --reasoning off|selective|always (on aliases selective).',
              '12B/E4B estimates assume q8 KV and four native branches; E2B uses F16 KV and one native branch.',
              'All use one chat slot; estimates are not fit guarantees.',
              'Measured historical baselines on RTX 5070 Ti (different modes):']
    lines += [f'  {name}: {value}' for name, value in BASELINES.items()]
    lines += ['Image reasoning supports E4B/NVFP4 8K vision+MTP and experimental E2B 8K/64K profiles.',
              'E2B assets are private; use exact verified local files. Other E2B capacities require a custom contract.',
              'MTP uses one chat slot and auto memory on Linux/CUDA.',
              'Q8 vision + MTP did not fit the measured 16 GB profile; larger/custom configurations are unvalidated.',
              'Existing long model and preset names remain compatibility aliases.']
    return '\n'.join(lines)
