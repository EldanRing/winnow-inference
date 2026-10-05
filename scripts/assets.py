"""Pinned release assets and verified local reuse; no inference or hidden installs."""
import json
import os
import shutil
import tempfile
import urllib.error
from pathlib import Path
from download import fetch
from verify_model import verify
from adaptive_policy import normalize_reasoning

ROOT = Path(__file__).resolve().parents[1]
MODELS = json.loads((ROOT / 'manifests/release-assets-v1.json').read_text())['models']

MODEL_ALIASES = {'q8': '12b-q8', 'nv4': '12b-nvfp4', 'e4b': 'e4b-q8'}

def canonical_model(name):
    return MODEL_ALIASES.get(name, name)


def selection(model, reasoning='off', mtp='off', vision='on'):
    model = canonical_model(model)
    reasoning = normalize_reasoning(reasoning)
    if model not in MODELS or mtp not in {'on', 'off'} or vision not in {'on', 'off'}:
        raise ValueError('Unknown model or on/off selection')
    if reasoning != 'off' and vision == 'on':
        raise ValueError('Adaptive reasoning supports text only; choose --vision off')
    spec = MODELS[model]
    return spec, ['model'] + (['projector'] if vision == 'on' else []) + (['assistant'] if mtp == 'on' else [])


def destination(model_dir, relative):
    root = Path(model_dir).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Invalid asset destination')
    target = root / relative
    if target.is_symlink() or not target.resolve().is_relative_to(root):
        raise ValueError('Asset destination escapes the model directory')
    return target


def acquire(model, model_dir, reasoning='off', mtp='off', vision='on', asset_dirs=(), offline=False):
    spec, kinds = selection(model, reasoning, mtp, vision)
    resolved = {}
    for kind in kinds:
        artifact = spec[kind]
        target = destination(model_dir, artifact['file'])
        if target.exists():
            verify(target, artifact)
        else:
            source = None
            # Explicit cache directories only; no recursive machine-wide search.
            names = [artifact['file'], Path(artifact['file']).name, Path(artifact['url']).name]
            for directory in asset_dirs:
                for name in dict.fromkeys(names):
                    candidate = Path(directory) / name
                    if candidate.is_file():
                        verify(candidate, artifact)
                        source = candidate
                        break
                if source is not None:
                    break
            if source is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.asset-', delete=False) as stream:
                        temporary = Path(stream.name)
                        with source.open('rb') as original:
                            shutil.copyfileobj(original, stream, 8 * 1024 * 1024)
                    verify(temporary, artifact)
                    temporary.chmod(0o644)
                    os.link(temporary, target)  # Never overwrite a concurrent owner's file.
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            elif offline:
                raise ValueError(f'Missing verified local asset: {artifact["file"]}. Supply --asset-dir; offline mode never downloads.')
            else:
                # Resume verified HTTP downloads. Planned URLs may not be live yet.
                partial = target.with_name(target.name + '.part')
                if partial.is_symlink():
                    raise ValueError('Asset partial-file symlinks are not allowed')
                try:
                    fetch(target, artifact)
                except urllib.error.HTTPError as error:
                    raise ValueError(f'Asset unavailable (HTTP {error.code}): {artifact["url"]}. '
                                     f'Status: {artifact["availability"]}. No alternative weights were used; '
                                     'supply a verified --asset-dir or wait for the release asset.') from error
        resolved[kind] = str(target)
    if mtp == 'on':
        for name in ['LICENSE-APACHE-2.0.txt', 'NOTICE.txt']:
            target = destination(model_dir, 'assistants/' + name)
            data = (ROOT / 'third_party' / ('gemma-assistants-' + name)).read_bytes()
            if target.exists() and target.read_bytes() != data:
                raise ValueError('Existing assistant attribution differs: ' + str(target))
            if not target.exists():
                with target.open('xb') as stream:
                    stream.write(data)
    return resolved
