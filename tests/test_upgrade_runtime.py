"""Exercise runtime upgrade commands with a CPU recorder replacing native inference."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from package_local_candidate import package, sha

class UpgradeRuntime(unittest.TestCase):
    def test_new_runtime_reuses_old_models_and_preserves_launch_options_and_rollback(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();old=root/'old-install';old.mkdir()
            models=old/'models';(models/'gguf').mkdir(parents=True)
            payload=b'CPU-only weight fixture'
            model=models/'gguf/Winnow-E4B-Q8_0.gguf';model.write_bytes(payload)
            saved=old/'launch.txt';saved.write_text('original custom command --cache q8_0\n')
            config=old/'settings.json';config.write_text('{"custom":"retained"}\n')
            auth=old/'auth-fixture';auth.write_text('test fixture, not a credential\n')
            snapshots={str(p.relative_to(old)):p.read_bytes() for p in old.rglob('*') if p.is_file()}
            recorder=root/'recorder'
            recorder.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                'print(json.dumps({"args":sys.argv[1:],"custom":os.environ.get("UPGRADE_CUSTOM"),'
                '"cache":os.environ.get("WINNOW_CACHE")}))\n')
            recorder.chmod(0o755)
            new=root/'winnow-linux-x86_64-cuda-v2026.10.06'
            with patch('package_local_candidate.subprocess.check_output',
                       side_effect=[sha(ROOT/'runtime.lock.json'),'']):
                package(recorder,new)
            # Small exact-hash fixture avoids real model reads; no native GPU process can start.
            manifest=new/'manifests/release-assets-v1.json'
            registry=json.loads(manifest.read_text())
            registry['models']['e4b-q8']['model'].update(bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest())
            manifest.write_text(json.dumps(registry))
            base=[str(new/'bin/winnow')]
            result=subprocess.run(base+['download','--model','e4b','--model-dir',str(models),
                '--vision','off','--mtp','off','--offline'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['model'],str(model))
            for cache in (None,'q8_0'):
                command=base+['serve','--model','e4b','--model-dir',str(models),'--context','16k',
                    '--vision','off','--mtp','off','--reasoning','off','--port','18399',
                    '--threads','6','--api-key-file',str(auth)]
                if cache:command+=['--cache',cache]
                env=dict(os.environ,UPGRADE_CUSTOM='retained',PYTHONDONTWRITEBYTECODE='1')
                result=subprocess.run(command,capture_output=True,text=True,env=env)
                self.assertEqual(result.returncode,0,result.stderr)
                answer=json.loads(result.stdout.splitlines()[-1]);args=answer['args']
                self.assertEqual(args[args.index('--model')+1],str(model))
                self.assertEqual(args[args.index('--port')+1],'18399')
                self.assertEqual(args[args.index('--threads')+1],'6')
                self.assertEqual(args[args.index('--ctx-size')+1],'16384')
                self.assertEqual(args[args.index('--api-key-file')+1],str(auth))
                self.assertEqual(args[args.index('--cache-type-k')+1],cache or 'f16')
                self.assertEqual(answer['cache'],cache or 'f16')
                self.assertEqual(answer['custom'],'retained')
                self.assertNotIn('--spec-draft-model',args)
            self.assertEqual(snapshots,{str(p.relative_to(old)):p.read_bytes()
                for p in old.rglob('*') if p.is_file()})
            self.assertTrue((new/'release-manifest.json').is_file())
            self.assertFalse((new/'candidate-manifest.json').exists())

if __name__=='__main__':
    unittest.main()
