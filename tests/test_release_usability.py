"""Asset integrity, independent mode selection and CRLF lock recovery."""
import argparse
import difflib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import assets
import build
import decision_client
import serve
import setup
import winnow
from adaptive_policy import DecisionPipeline
from test_adaptive_pipeline import FakeTransport, body, generation, identity, native


class ReleaseUsability(unittest.TestCase):
    def test_explicit_model_selection_overrides_example_without_changing_decisions(self):
        original = json.loads((ROOT / 'examples/decisions.json').read_text())
        transport = FakeTransport([{'model': 'Winnow-E4B', 'answers': {}}])
        argv = ['decide', '--input', str(ROOT / 'examples/decisions.json'), '--model', 'Winnow-E4B']
        with patch.object(sys, 'argv', argv), patch.object(decision_client, 'HTTPTransport', return_value=transport), patch.object(sys, 'stdout', io.StringIO()):
            decision_client.main()
        expected = dict(original, model='Winnow-E4B')
        self.assertEqual(transport.calls[0][1], expected)

    def test_crlf_exact_final_guard_reproduction_and_safe_repair(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / 'runtime'; source.mkdir()
            subprocess.run(['git', 'init', str(source)], check=True, capture_output=True)
            subprocess.run(['git', '-C', str(source), 'config', 'core.autocrlf', 'true'], check=True)
            original = 'one\ntwo\nthree\n'; final = 'one\nsecond\nthree\n'
            path = source / 'sample.txt'; path.write_bytes(original.encode())
            subprocess.run(['git', '-C', str(source), 'add', '.'], check=True, capture_output=True)
            subprocess.run(['git', '-C', str(source), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-m', 'fixture'], check=True, capture_output=True)
            path.unlink()
            subprocess.run(['git', '-C', str(source), 'checkout', '--', 'sample.txt'], check=True)
            self.assertIn(b'\r\n', path.read_bytes())
            patchfile = root / 'change.patch'
            patchfile.write_text(''.join(difflib.unified_diff(original.splitlines(True), final.splitlines(True), fromfile='a/sample.txt', tofile='b/sample.txt')))
            lock = {'patches': [{'file': patchfile.name, 'sha256': hashlib.sha256(patchfile.read_bytes()).hexdigest()}], 'source_sha256': {'sample.txt': hashlib.sha256(final.encode()).hexdigest()}}
            subprocess.run(['git', '-C', str(source), 'apply', str(patchfile)], check=True)
            self.assertFalse(build.runtime_matches(source, lock))
            self.assertIn('expected SHA256', '\n'.join(build.runtime_mismatches(source, lock)))
            build.prepare_runtime(source, lock, root)
            self.assertEqual(path.read_bytes(), final.encode())
            self.assertTrue(build.runtime_matches(source, lock))
            build.prepare_runtime(source, lock, root)
            # Never normalize or overwrite a real source edit.
            changed = b'one\r\nowner edit\r\nthree\r\n'; path.write_bytes(changed)
            self.assertFalse(build.normalize_verified_crlf(source, lock))
            with self.assertRaises(SystemExit):
                build.prepare_runtime(source, lock, root)
            self.assertEqual(path.read_bytes(), changed)
            # Directly exercise the final mismatch guard, independent of apply errors.
            lock['patches'] = []
            with self.assertRaisesRegex(SystemExit, 'sample.txt: expected SHA256 .*; actual'):
                build.prepare_runtime(source, lock, root)

    def test_asset_local_reuse_missing_remote_and_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); cache = root/'cache'; cache.mkdir(); models = root/'models'
            payload = b'exact artifact'; (cache/'model.gguf').write_bytes(payload)
            item = {'file': 'gguf/model.gguf', 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(), 'url': 'https://example.invalid/planned/model.gguf', 'availability': 'planned; not yet published'}
            registry = {'fixture': {'model': item}}
            with patch.object(assets, 'MODELS', registry), patch.object(assets, 'fetch') as fetch:
                result = assets.acquire('fixture', models, vision='off', asset_dirs=[cache], offline=True)
                self.assertEqual(Path(result['model']).read_bytes(), payload)
                self.assertEqual(assets.acquire('fixture', models, vision='off', offline=True), result)
                fetch.assert_not_called()
                target = models/item['file']; target.write_bytes(b'corrupt')
                with self.assertRaises(ValueError):
                    assets.acquire('fixture', models, vision='off', asset_dirs=[cache])
                self.assertEqual(target.read_bytes(), b'corrupt')
                target.unlink()
                with self.assertRaisesRegex(ValueError, 'Missing verified local asset'):
                    assets.acquire('fixture', models, vision='off', offline=True)
                fetch.side_effect = urllib.error.HTTPError(item['url'], 404, 'Not Found', {}, None)
                with self.assertRaisesRegex(ValueError, 'not published.*planned'):
                    assets.acquire('fixture', models, vision='off')
                fetch.assert_not_called()
                item['availability'] = 'published'
                with self.assertRaisesRegex(ValueError, 'HTTP 404.*published.*No alternative'):
                    assets.acquire('fixture', models, vision='off')
                target.symlink_to(cache/'model.gguf')
                with self.assertRaisesRegex(ValueError, 'escapes'):
                    assets.acquire('fixture', models, vision='off')

    def test_all_supported_modes_select_only_needed_assets(self):
        for model in assets.MODELS:
            for reasoning in ['on', 'off']:
                for mtp in ['on', 'off']:
                    spec, kinds = assets.selection(model, reasoning, mtp, 'off')
                    self.assertEqual('assistant' in kinds, mtp == 'on')
                    self.assertNotIn('projector', kinds)
                    a = argparse.Namespace(model=model, reasoning=reasoning, mtp=mtp, vision='off', context=8192, model_dir=Path('/models'), server=None)
                    with patch('winnow.platform.system', return_value='Linux'):
                        command = winnow.serve_command(a)
                    self.assertEqual('--assistant' in command, mtp == 'on')
                    self.assertIn('/models/' + spec['model']['file'], command)
                    self.assertEqual('--experimental-adaptive' in command,
                                     model != 'e2b-q8' and (reasoning == 'on' or mtp == 'on'))
                    if model == 'e2b-q8':
                        self.assertIn(assets.decision_preset(model, 8192, 'off', mtp), command)
        self.assertIn('projector', assets.selection('q8', 'off', 'on', 'on')[1])
        with self.assertRaises(ValueError): assets.selection('e4b-q8', 'on', 'off', 'on')

    def test_mtp_off_adaptive_preserves_policy_math_and_rejects_wrong_backend(self):
        policy = 'e4b-calibrated50-v1'; request = body()
        ident = identity(policy); ident['runtime']['resident_mtp'] = False; ident['runtime'].pop('assistant_sha256')
        direct = native(request, [0.1, 0.0], policy); augmented = native(request, [0.0, 1.0], policy)
        for r in [direct, augmented]: r['winnow']['resident_speculative_chat'] = False
        off = DecisionPipeline(FakeTransport([ident, direct, generation(), augmented]), 'experimental-adaptive', policy, 'off').decide(request)
        on = DecisionPipeline(FakeTransport([identity(policy), native(request, [0.1, 0.0], policy), generation(), native(request, [0.0, 1.0], policy)]), 'experimental-adaptive', policy).decide(request)
        self.assertEqual(off['answers'], on['answers'])
        self.assertTrue(off['winnow']['adaptive']['completed_blend'])
        for wrong in [identity(policy), dict(runtime={**ident['runtime'], 'target_sha256': 'wrong'})]:
            with self.assertRaisesRegex(ValueError, 'Backend does not match'):
                DecisionPipeline(FakeTransport([wrong]), 'experimental-adaptive', policy, 'off').decide(request)

    def test_mtp_off_launcher_never_loads_assistant_or_drafts(self):
        with tempfile.TemporaryDirectory() as td:
            model=Path(td)/'model'; binary=Path(td)/'server';model.touch();binary.touch()
            argv=['serve','--experimental-adaptive','e4b-calibrated50-v1','--text-only','--mtp','off','--model',str(model),'--server',str(binary),'--dry-run']
            out=io.StringIO()
            with patch.object(sys,'argv',argv),patch('serve.platform.system',return_value='Linux'),patch.object(serve,'verify') as verify,patch.object(sys,'stdout',out),patch.dict(serve.os.environ, {'LLAMA_ARG_SPEC_TYPE':'draft-mtp','LLAMA_ARG_SPEC_DRAFT_MODEL':'wrong','LLAMA_ARG_MMPROJ_URL':'https://invalid.test/file','KEEP_USER_SETTING':'yes'}):serve.main()
            result=json.loads(out.getvalue());command=result['command']
            self.assertEqual(verify.call_count,1)
            self.assertEqual(command[command.index('--spec-type')+1],'none')
            self.assertNotIn('--spec-draft-model',command)
            self.assertIn('--no-mmproj',command)
            self.assertIn('--offline',command)
            self.assertEqual(result['environment']['WINNOW_MANAGED_LAUNCH'],'1')
            self.assertEqual(result['environment']['WINNOW_RESIDENT_MTP'],'0')
            self.assertNotIn('WINNOW_ASSISTANT_SHA256',result['environment'])

    def test_setup_forwards_explicit_cuda_compiler_and_arch(self):
        argv=['setup','--model','e4b-q8','--vision','off','--cuda-compiler','/opt/cuda/bin/nvcc','--cuda-arch','120']
        with patch.object(sys,'argv',argv),patch.object(setup,'resolve_profile',return_value=('5070ti-64k',{})),patch.object(setup,'prerequisites',return_value=[]) as prereq,patch.object(setup,'acquire'),patch.object(setup.subprocess,'run') as run,patch.object(sys,'stdout',io.StringIO()):
            self.assertEqual(setup.main(),0)
        self.assertEqual(prereq.call_args.args[2],Path('/opt/cuda/bin/nvcc'))
        command=run.call_args.args[0]
        self.assertEqual(command[command.index('--cuda-compiler')+1],'/opt/cuda/bin/nvcc')
        self.assertEqual(command[command.index('--cuda-arch')+1],'120')

    def test_setup_accepts_tested_16gb_driver_report(self):
        with patch.object(setup, 'output', return_value='RTX 5070 Ti, 15882, 12.0'), patch.object(sys, 'stdout', io.StringIO()):
            self.assertEqual(setup.prerequisites('5070ti-64k', skip_build=True), [])


    def test_selected_direct_identity_rejects_wrong_model_or_quantization(self):
        request = {'model': 'Winnow-12B', 'state': 'x', 'questions': {'q': {'type': 'noul', 'instructions': 'x'}}}
        q8 = assets.selection('q8')[0]; nv4 = assets.selection('nv4')[0]
        self.assertNotEqual(q8['model']['sha256'], nv4['model']['sha256'])
        right = {'runtime': {'target_sha256': q8['model']['sha256']}}
        response = {'model': 'Winnow-12B', 'answers': {'q': {'type': 'noul', 'noul': 0.7}}}
        self.assertEqual(DecisionPipeline(FakeTransport([right, response]), target=q8).decide(request), response)
        for replies in [[{'runtime': {'target_sha256': nv4['model']['sha256']}}],
                        [right, {**response, 'model': 'Winnow-E4B'}]]:
            with self.assertRaisesRegex(ValueError, 'Selected model'):
                DecisionPipeline(FakeTransport(replies), target=q8).decide(request)

    def test_e4b_12gb_8k_admission_uses_selected_configuration(self):
        out = io.StringIO()
        with patch.object(setup, 'output', return_value='Test GPU, 12288, 8.6'), patch.object(sys, 'stdout', out):
            self.assertEqual(setup.prerequisites('5070ti-64k', skip_build=True, model='e4b', context=8192), [])
        self.assertIn('e4b: context 8192', out.getvalue())
        self.assertNotIn('64K CUDA profile', out.getvalue())

if __name__ == '__main__': unittest.main()
