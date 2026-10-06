"""Managed upgrades using real package contents and CPU-only native recorders."""
import copy
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import update
import bootstrap_update
import winnow
from package_local_candidate import package

PAYLOAD=b'preserved model fixture'
RECORDER=b'#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n'

class ManagedUpdate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage=tempfile.TemporaryDirectory()
        cls.base=Path(cls.storage.name).resolve()
        binary=cls.base/'cpu-recorder'
        binary.write_bytes(RECORDER+b'# '+update.sha(ROOT/'runtime.lock.json').encode()+b'\n');binary.chmod(0o755)
        cls.template=cls.base/'template'
        with patch('package_local_candidate.subprocess.check_output',
                   side_effect=[update.sha(ROOT/'runtime.lock.json'),'']):
            package(binary,cls.template)
        registry=cls.template/'manifests/release-assets-v1.json'
        data=json.loads(registry.read_text())
        data['models']['e4b-q8']['model'].update(bytes=len(PAYLOAD),sha256=hashlib.sha256(PAYLOAD).hexdigest())
        registry.write_text(json.dumps(data))
        cls.artifacts={}
        for version in ('v2026.10.06','v2026.10.07','v2026.10.08'):
            cls.artifacts[version]={kind:cls.artifact(version,kind) for kind in ('runtime','source')}

    @classmethod
    def tearDownClass(cls):cls.storage.cleanup()

    @classmethod
    def artifact(cls,tag,kind):
        root_name=f'winnow-{"source" if kind=="source" else "linux-x86_64-cuda"}-{tag}'
        path=cls.base/(root_name+'-fixture');shutil.copytree(cls.template,path)
        manifest=path/'release-manifest.json'
        data=json.loads(manifest.read_text());data.update(release_version=tag,package_kind=kind)
        if kind=='source':
            shutil.rmtree(path/'bin')
            shutil.copy2(ROOT/'CMakeLists.txt',path/'CMakeLists.txt')
            shutil.copy2(ROOT/'scripts/build.py',path/'scripts/build.py')
        data['files']={str(p.relative_to(path)):{'bytes':p.stat().st_size,'sha256':update.sha(p)}
                       for p in path.rglob('*') if p.is_file() and p!=manifest}
        manifest.write_text(json.dumps(data))
        out=io.BytesIO()
        with tarfile.open(fileobj=out,mode='w:gz') as t:t.add(path,arcname=root_name)
        return out.getvalue()

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name).resolve()/'install'
        shutil.copytree(self.template,self.root)
        data=json.loads((self.root/'release-manifest.json').read_text());data['release_version']='v2026.10.05'
        data['files']={str(p.relative_to(self.root)):{'bytes':p.stat().st_size,'sha256':update.sha(p)}
                       for p in self.root.rglob('*') if p.is_file() and p.name!='release-manifest.json'}
        (self.root/'release-manifest.json').write_text(json.dumps(data))
        (self.root/'models/gguf').mkdir(parents=True)
        (self.root/'models/gguf/Winnow-E4B-Q8_0.gguf').write_bytes(PAYLOAD)
        (self.root/'settings.ini').write_text('custom=retained\ncache=q8_0\n')
        (self.root/'auth-fixture').write_text('test only, not a credential\n')
        self.calls=[]

    def tearDown(self):self.temporary.cleanup()

    def fetcher(self,tag='v2026.10.06',corrupt=False,archive=None):
        blobs=self.artifacts[tag] if archive is None else {kind:archive for kind in ('runtime','source')}
        names={kind:f'winnow-{"source" if kind=="source" else "linux-x86_64-cuda"}-{tag}.tar.gz' for kind in blobs}
        urls={kind:update.BASE+tag+'/'+name for kind,name in names.items()}
        sums=''.join(('0'*64 if corrupt else hashlib.sha256(blobs[k]).hexdigest())+'  '+names[k]+'\n' for k in blobs).encode()
        api={'tag_name':tag,'draft':False,'prerelease':False,'assets':
             [{'name':names[k],'browser_download_url':urls[k],'size':len(blobs[k])} for k in blobs]
             +[{'name':'SHA256SUMS','browser_download_url':update.BASE+tag+'/SHA256SUMS','size':len(sums)}]}
        def fetch(url,limit):
            update.official(url);self.calls.append(url)
            if url.startswith(update.API):return json.dumps(api).encode()
            if url.endswith('/SHA256SUMS'):return sums
            for kind in blobs:
                if url==urls[kind]:return blobs[kind]
            raise AssertionError(url)
        return fetch

    def run_update(self,tag='v2026.10.06',**kwargs):
        with patch('update.platform.system',return_value='Linux'),patch('update.platform.machine',return_value='x86_64'):
            return update.run(self.root,fetch=self.fetcher(tag),runtime_check=lambda _:None,**kwargs)

    def preserved(self):
        self.assertEqual((self.root/'models/gguf/Winnow-E4B-Q8_0.gguf').read_bytes(),PAYLOAD)
        self.assertEqual((self.root/'settings.ini').read_text(),'custom=retained\ncache=q8_0\n')

    def test_success_noop_real_launcher_explicit_cache_and_rollback(self):
        self.assertEqual(self.run_update()['status'],'updated');self.preserved()
        before=update.active(self.root/'.winnow');self.calls.clear()
        self.assertEqual(self.run_update()['status'],'already_current')
        self.assertEqual(update.active(self.root/'.winnow'),before)
        self.assertEqual(len(self.calls),1)
        for cache in ('f16','q8_0'):
            result=subprocess.run([str(self.root/'bin/winnow'),'serve','--model','e4b','--vision','off',
                '--mtp','off','--cache',cache,'--port','18399','--api-key-file',str(self.root/'auth-fixture')],
                capture_output=True,text=True,env={k:v for k,v in os.environ.items() if k!='PYTHONDONTWRITEBYTECODE'})
            self.assertEqual(result.returncode,0,result.stderr)
            args=json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(args[args.index('--cache-type-k')+1],cache)
            self.assertEqual(args[args.index('--port')+1],'18399')
            self.assertEqual(Path(args[args.index('--model')+1]).resolve(),self.root/'models/gguf/Winnow-E4B-Q8_0.gguf')
        self.assertTrue(list((before/'scripts/__pycache__').glob('*.pyc')))
        self.assertEqual(self.run_update()['status'],'already_current')
        with patch('update.platform.system',return_value='Linux'):
            result=update.run(self.root,rollback=True)
        self.assertEqual(result['version'],'v2026.10.05');self.preserved()

    def test_checksum_failure_and_interrupted_download_leave_current_usable(self):
        with patch('update.platform.system',return_value='Linux'):
            with self.assertRaisesRegex(ValueError,'checksum'):
                update.run(self.root,fetch=self.fetcher(corrupt=True),runtime_check=lambda _:None)
            self.assertEqual(update.metadata(update.active(self.root/'.winnow'))['version'],'v2026.10.05')
            def interrupt(url,limit):
                if url.endswith('.tar.gz'):raise OSError('download interrupted')
                return self.fetcher()(url,limit)
            with self.assertRaisesRegex(OSError,'interrupted'):
                update.run(self.root,fetch=interrupt,runtime_check=lambda _:None)
        self.preserved();self.assertEqual(self.run_update()['status'],'updated')

    def test_interruptions_recover_by_rerunning_without_partial_activation(self):
        for event in ('extract','build','before_swap','after_swap'):
            with self.subTest(event=event):
                # Start each fault from the old version using the previous successful version as recovery.
                if (self.root/'.winnow/current').is_symlink() and update.metadata(update.active(self.root/'.winnow'))['version']=='v2026.10.06':
                    update.run(self.root,rollback=True)
                def fault(point):
                    if point==event:raise KeyboardInterrupt('simulated interruption')
                with self.assertRaises(KeyboardInterrupt):self.run_update(fault=fault)
                current=update.metadata(update.active(self.root/'.winnow'))['version']
                self.assertEqual(current,'v2026.10.06' if event=='after_swap' else 'v2026.10.05')
                self.run_update();self.preserved()
                self.assertLessEqual(len(list((self.root/'.winnow/releases').iterdir())),2)

    def test_repeat_updates_keep_only_one_automatic_recovery_version(self):
        for tag in ('v2026.10.06','v2026.10.07','v2026.10.08'):
            self.run_update(tag);self.preserved()
            self.assertEqual(len(list((self.root/'.winnow/releases').iterdir())),2)
        update.run(self.root,rollback=True)
        self.assertEqual(update.metadata(update.active(self.root/'.winnow'))['version'],'v2026.10.07')

    def test_traversal_links_incomplete_packages_and_mismatched_versions_are_rejected(self):
        for name,kind in [('../escape','file'),('/absolute','file'),('winnow-linux-x86_64-cuda-v2026.10.06/link','symlink')]:
            out=io.BytesIO()
            with tarfile.open(fileobj=out,mode='w:gz') as t:
                item=tarfile.TarInfo(name)
                if kind=='symlink':item.type=tarfile.SYMTYPE;item.linkname='../../models'
                else:item.size=1
                t.addfile(item,io.BytesIO(b'x') if kind=='file' else None)
            with patch('update.platform.system',return_value='Linux'),self.assertRaises(ValueError):
                update.run(self.root,fetch=self.fetcher(archive=out.getvalue()),runtime_check=lambda _:None)
        with patch('update.platform.system',return_value='Linux'),self.assertRaises(ValueError):
            update.run(self.root,fetch=self.fetcher('v2026.10.07',archive=self.artifacts['v2026.10.06']['runtime']),runtime_check=lambda _:None)
        self.preserved();self.assertEqual(self.run_update()['status'],'updated')

    def test_dirty_source_and_modified_managed_code_are_not_overwritten(self):
        subprocess.run(['git','init','-q',str(self.root)],check=True)
        subprocess.run(['git','-C',str(self.root),'add','.'],check=True)
        subprocess.run(['git','-C',str(self.root),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid',
                        'commit','-qm','fixture'],check=True)
        (self.root/'README.md').write_text('user edit')
        with self.assertRaisesRegex(ValueError,'Dirty source'):self.run_update()
        self.assertFalse((self.root/'.winnow/current').exists())
        subprocess.run(['git','-C',str(self.root),'restore','README.md'],check=True)
        self.run_update()
        (self.root/'scripts/winnow.py').write_text('user code edit')
        with self.assertRaisesRegex(ValueError,'Local code'):self.run_update('v2026.10.07')
        self.assertEqual((self.root/'scripts/winnow.py').read_text(),'user code edit');self.preserved()

    def test_bootstrap_interruption_recovers_and_symlink_storage_is_refused(self):
        def fault(event):
            if event=='adopt':raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):self.run_update(fault=fault)
        self.assertTrue((self.root/'.winnow/adoption.json').exists())
        self.assertEqual(self.run_update()['status'],'updated');self.preserved()
        other=self.root.parent/'unsafe';other.mkdir();(other/'.winnow').symlink_to(self.root/'.winnow')
        with self.assertRaisesRegex(ValueError,'symlink'):update.run(other,fetch=self.fetcher())

    def test_interrupted_initial_snapshot_copy_recovers_without_losing_original_files(self):
        def fault(event):
            if event=='adopt_copy':raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):self.run_update(fault=fault)
        self.assertTrue((self.root/'scripts/winnow.py').is_file())
        self.assertFalse((self.root/'.winnow/current').exists())
        self.preserved();self.assertEqual(self.run_update()['status'],'updated');self.preserved()

    def test_macos_selects_source_and_builds_at_final_path_before_activation(self):
        built=[]
        def build(path):
            self.assertIn('releases',path.parts)
            p=path/'.build/bin/winnow-server';p.parent.mkdir(parents=True);p.write_bytes(RECORDER);p.chmod(0o755)
            built.append(path)
        with patch('update.platform.system',return_value='Darwin'),patch('update.platform.machine',return_value='arm64'):
            result=update.run(self.root,fetch=self.fetcher(),build=build,
                runtime_check=lambda _:self.fail('CUDA runtime selected on Mac'))
        self.assertEqual(result['status'],'updated');self.assertEqual(len(built),1)
        self.assertEqual(update.metadata(update.active(self.root/'.winnow'))['kind'],'source');self.preserved()

    def test_release_availability_and_official_url_checks(self):
        with self.assertRaisesRegex(ValueError,'official'):
            update.official('https://example.invalid/release.tar.gz')
        with self.assertRaises(ValueError):update.release_info('arbitrary/path',self.fetcher())
        with self.assertRaises(ValueError):
            update.release_info(fetch=lambda *_:json.dumps({'tag_name':'v2026.10.06','draft':True,'prerelease':False}).encode())

    def test_incomplete_member_corruption_and_native_lock_mismatch_are_rejected(self):
        blob=self.artifacts['v2026.10.06']['runtime']
        for change in ('missing','member','binary'):
            out=io.BytesIO()
            with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as source,tarfile.open(fileobj=out,mode='w:gz') as target:
                for item in source:
                    if change=='missing' and item.name.endswith('/scripts/update.py'):continue
                    data=source.extractfile(item).read() if item.isfile() else None
                    if change=='member' and item.name.endswith('/scripts/winnow.py'):data+=b'\n# corruption\n'
                    if change=='binary' and item.name.endswith('/bin/winnow-server'):data=RECORDER
                    if change=='binary' and item.name.endswith('/release-manifest.json'):
                        manifest=json.loads(data)
                        manifest['binary_sha256']=hashlib.sha256(RECORDER).hexdigest()
                        manifest['files']['bin/winnow-server']={'bytes':len(RECORDER),'sha256':hashlib.sha256(RECORDER).hexdigest()}
                        data=json.dumps(manifest).encode()
                    if data is not None:item.size=len(data)
                    target.addfile(item,io.BytesIO(data) if data is not None else None)
            with patch('update.platform.system',return_value='Linux'),self.assertRaises(ValueError):
                update.run(self.root,fetch=self.fetcher(archive=out.getvalue()),runtime_check=lambda _:None)
            self.preserved()
        self.assertEqual(self.run_update()['status'],'updated')

    def test_bootstrap_verifies_helper_checksum_and_cli_forwards_exact_update_command(self):
        helper=(ROOT/'scripts/update.py').read_bytes()
        checksum=(hashlib.sha256(helper).hexdigest()+'  winnow-update.py\n').encode()
        with patch.object(sys,'argv',['bootstrap_update.py',str(self.root)]), \
                patch('bootstrap_update.download',side_effect=[checksum,helper]), \
                patch('bootstrap_update.subprocess.run') as run:
            bootstrap_update.main()
            command=run.call_args.args[0]
            self.assertEqual(command[2:],['update','--install-dir',str(self.root)])
        with patch.object(sys,'argv',['bootstrap_update.py',str(self.root)]), \
                patch('bootstrap_update.download',side_effect=[b'0'*64+b'  winnow-update.py\n',helper]), \
                patch('bootstrap_update.subprocess.run') as run:
            with self.assertRaisesRegex(ValueError,'checksum'):bootstrap_update.main()
            run.assert_not_called()
        with patch.object(sys,'argv',['winnow','update','--source','--version','v2026.10.06']), \
                patch('update.main') as main:
            winnow.main();main.assert_called_once_with(['update','--source','--version','v2026.10.06'])

    def test_source_build_retains_compiler_flags_and_reuses_pinned_runtime_cache(self):
        control=self.root/'.winnow';(control/'releases').mkdir(parents=True)
        previous=self.root/'previous';previous.mkdir()
        shutil.copy2(ROOT/'runtime.lock.json',previous/'runtime.lock.json')
        (previous/'.runtime/llama.cpp').mkdir(parents=True)
        (previous/'.runtime/llama.cpp/fixture').write_text('pinned runtime fixture')
        (previous/'.build').mkdir()
        (previous/'.build/CMakeCache.txt').write_text('GGML_CUDA:BOOL=ON\nCMAKE_CUDA_COMPILER:FILEPATH=/custom/nvcc\nCMAKE_CUDA_ARCHITECTURES:STRING=120\n')
        for name in ('first','second'):
            destination=control/'releases'/name;destination.mkdir()
            shutil.copy2(ROOT/'runtime.lock.json',destination/'runtime.lock.json')
            def compile(command,cwd,check):
                self.assertEqual(command[command.index('--cuda-compiler')+1],'/custom/nvcc')
                self.assertEqual(command[command.index('--cuda-arch')+1],'120')
                p=Path(cwd)/'.build/bin/winnow-server';p.parent.mkdir(parents=True)
                p.write_bytes(update.sha(ROOT/'runtime.lock.json').encode())
            with patch('update.subprocess.run',side_effect=compile):update.build_source(destination,previous)
            self.assertTrue((destination/'.runtime').is_symlink())
            self.assertEqual((destination/'.runtime/llama.cpp/fixture').read_text(),'pinned runtime fixture')
        self.assertEqual(len(list((control/'runtime-cache').iterdir())),1)

if __name__=='__main__':unittest.main()
