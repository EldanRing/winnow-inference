#!/usr/bin/env python3
"""Download, launch or query Winnow with explicit reasoning and MTP choices."""
import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from assets import MODELS, MODEL_ALIASES, ROOT, acquire, selection, canonical_model, decision_preset
from launch_options import DEFAULT_CONTEXT, context_size, preset_list, memory_note
from adaptive_policy import REASONING_CHOICES, normalize_reasoning


def serve_command(a):
    spec, _ = selection(a.model, a.reasoning, a.mtp, a.vision)
    reasoning = normalize_reasoning(a.reasoning)
    e2b = canonical_model(a.model) == 'e2b-q8'
    if (reasoning != 'off' or a.mtp == 'on' or e2b) and platform.system() != 'Linux':
        raise ValueError('Optional reasoning/MTP profiles are supported only on Linux/CUDA')
    args = [sys.executable, str(ROOT / 'scripts/serve.py'), '--model-dir', str(a.model_dir),
            '--model', str(a.model_dir / spec['model']['file']), '--alias', spec['alias'],
            '--target', a.model, '--context', str(a.context)]
    if e2b:
        args += ['--preset', decision_preset(a.model, a.context, a.vision, a.mtp), '--mtp', a.mtp]
    elif a.vision == 'on' and a.mtp == 'on':
        if reasoning != 'off' and a.context != 8192:
            raise ValueError('Image reasoning presets pin 8K; custom contexts require an explicit operator runtime contract')
        args += ['--preset', spec['mtp_vision_preset']]
    elif reasoning != 'off' or (a.mtp == 'on' and a.vision == 'off'):
        args += ['--experimental-adaptive', spec['policy'], '--text-only', '--mtp', a.mtp]
    elif a.vision == 'off':
        args += ['--text-only']
    if getattr(a, 'cache', None) is not None:
        args += ['--cache', a.cache]
    if a.mtp == 'on':
        args += ['--assistant', str(a.model_dir / spec['assistant']['file'])]
    if a.vision == 'on':
        args += ['--mmproj', str(a.model_dir / spec['projector']['file'])]
    if a.server:
        args += ['--server', str(a.server)]
    elif (ROOT / 'bin/winnow-server').is_file():
        args += ['--server', str(ROOT / 'bin/winnow-server')]
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['download', 'serve', 'decide', 'presets', 'setup', 'update', 'rollback'])
    p.add_argument('--model', '--preset', choices=[*MODEL_ALIASES, *MODELS], default='q8', help='Short model preset; explicit mode/context flags override defaults')
    p.add_argument('--context', type=context_size, default=DEFAULT_CONTEXT, help='Tokens, e.g. 4k, 16k or 65536 (default: 8k)')
    p.add_argument('--cache', choices=['f16', 'q8_0'], default='f16', help='Target K/V cache (default: f16); draft cache flags stay q8_0')
    p.add_argument('--model-dir', type=Path, default=ROOT / 'models')
    p.add_argument('--reasoning', choices=REASONING_CHOICES, default='off',
                   help='Client routing: off (default), selective, always; on aliases selective')
    p.add_argument('--mtp', choices=['off', 'on'], default='off')
    p.add_argument('--vision', choices=['off', 'on'], default='off')
    p.add_argument('--asset-dir', type=Path, action='append', default=[], help='Explicit verified local asset cache; repeatable')
    p.add_argument('--offline', action='store_true', help='Require local assets; never contact a download server')
    p.add_argument('--server', type=Path)
    a, extra = p.parse_known_args()
    if a.action in {'update', 'rollback'}:
        from update import main as update_main
        update_main([a.action, *sys.argv[2:]])
        return
    a.reasoning = normalize_reasoning(a.reasoning)
    try:
        if a.action == 'presets':
            if extra:
                p.error('Unknown preset-list arguments')
            print(preset_list())
            return
        if a.action == 'setup':
            script = ROOT / 'scripts/setup.py'
            if not script.is_file():
                p.error('Setup builds from source; this runtime already contains a binary. Use download then serve.')
            os.execv(sys.executable, [sys.executable, str(script), *sys.argv[2:]])
        spec, _ = selection(a.model, a.reasoning, a.mtp, a.vision)
        if a.action == 'download':
            if extra:
                p.error('Unknown download arguments: ' + ' '.join(extra))
            print(json.dumps(acquire(a.model, a.model_dir, a.reasoning, a.mtp, a.vision, a.asset_dir, a.offline), indent=2))
        elif a.action == 'serve':
            if a.asset_dir or a.offline:
                p.error('--asset-dir/--offline apply to download; launch never downloads implicitly')
            print(memory_note(a.model, a.context, a.vision, a.mtp, a.reasoning), file=sys.stderr)
            command = serve_command(a) + extra
            os.execv(command[0], command)
        else:
            if a.asset_dir or a.offline or a.server:
                p.error('Asset/server options do not apply to decide')
            command = [sys.executable, str(ROOT / 'scripts/decision_client.py'), '--model', spec['alias'], '--target', a.model]
            if a.reasoning != 'off':
                command += ['--reasoning', a.reasoning, '--policy', spec['policy'], '--mtp', a.mtp]
                if a.vision == 'on' and canonical_model(a.model) != 'e2b-q8':
                    if a.context != 8192:
                        raise ValueError('Use decision_client.py with an explicit runtime profile for custom image reasoning contexts')
                    command += ['--runtime-profile', spec['mtp_vision_preset']]
            if canonical_model(a.model) == 'e2b-q8':
                command += ['--runtime-profile', decision_preset(a.model, a.context, a.vision, a.mtp)]
                if a.reasoning == 'off':
                    command += ['--mtp', a.mtp]
            if a.cache and '--runtime-profile' in command:
                command += ['--cache', a.cache]
            subprocess.run(command + extra, check=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        p.exit(1, 'Winnow: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
