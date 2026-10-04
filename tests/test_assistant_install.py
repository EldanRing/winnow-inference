"""Assistant installation integrity, containment and atomicity regressions."""
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import install_assistants as installer

class AssistantInstallTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.repo=self.root/'repo';(self.repo/'manifests').mkdir(parents=True);self.assets=self.root/'assets';self.assets.mkdir();self.models=self.root/'models'
  self.data={'12b':b'GGUF fixture twelve','e4b':b'GGUF fixture four'};self.registry={key:{'file':key+'.gguf','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for key,data in self.data.items()};self.write_registry()
  for key,data in self.data.items():(self.assets/self.registry[key]['file']).write_bytes(data)
  for name in ('LICENSE-APACHE-2.0.txt','NOTICE.txt'):(self.assets/name).write_text('Fixture '+name)
  (self.assets/'manifest.json').write_text(json.dumps({'assets':self.registry}));self.patch=patch.object(installer,'ROOT',self.repo);self.patch.start();self.addCleanup(self.patch.stop)
 def write_registry(self):(self.repo/'manifests/assistants-v1.json').write_text(json.dumps({'assets':self.registry}))
 def test_local_install_is_exact_and_idempotent(self):
  result=installer.install(self.assets,self.models,['12b','e4b']);before={key:(self.models/(key+'.gguf')).stat().st_ino for key in self.data};self.assertEqual(installer.install(self.assets,self.models,['12b','e4b']),result)
  for key,data in self.data.items():
   target=self.models/(key+'.gguf');self.assertEqual(target.read_bytes(),data);self.assertEqual(target.stat().st_ino,before[key]);self.assertEqual(target.stat().st_mode&0o777,0o644);self.assertEqual(result[key],str(target.resolve()))
  self.assertFalse(list(self.models.glob('.assistant-*')))
 def test_corrupt_and_truncated_sources_fail_before_any_install(self):
  for bad in (b'x'*len(self.data['e4b']),self.data['e4b'][:-1]):
   (self.assets/'e4b.gguf').write_bytes(bad)
   with self.assertRaises(ValueError):installer.install(self.assets,self.models,['12b','e4b'])
   self.assertFalse(self.models.exists())
 def test_existing_wrong_target_is_never_replaced(self):
  self.models.mkdir();target=self.models/'12b.gguf';target.write_bytes(b'wrong target')
  with self.assertRaises(ValueError):installer.install(self.assets,self.models,['12b'])
  self.assertEqual(target.read_bytes(),b'wrong target');self.assertFalse(list(self.models.glob('.assistant-*')))
 def test_existing_valid_target_does_not_excuse_wrong_source(self):
  self.models.mkdir();(self.models/'12b.gguf').write_bytes(self.data['12b']);(self.assets/'12b.gguf').write_bytes(b'x'*len(self.data['12b']))
  with self.assertRaisesRegex(ValueError,'SHA256'):installer.install(self.assets,self.models,['12b'])
  self.assertEqual((self.models/'12b.gguf').read_bytes(),self.data['12b'])
 def test_parent_absolute_and_symlink_source_escape(self):
  outside=self.root/'outside.gguf';outside.write_bytes(self.data['12b'])
  for bad in ('../outside.gguf',str(outside)):
   self.registry['12b']['file']=bad;self.write_registry()
   with self.assertRaisesRegex(ValueError,'asset directory'):installer.install(self.assets,self.models,['12b'])
   self.assertFalse(self.models.exists())
  self.registry['12b']['file']='12b.gguf';self.write_registry();(self.assets/'12b.gguf').unlink();(self.assets/'12b.gguf').symlink_to(outside)
  with self.assertRaisesRegex(ValueError,'asset directory'):installer.install(self.assets,self.models,['12b'])
  self.assertFalse(self.models.exists())
 def test_missing_licensing_rejects_without_destination_mutation(self):
  (self.assets/'NOTICE.txt').unlink()
  with self.assertRaisesRegex(ValueError,'license'):installer.install(self.assets,self.models,['12b'])
  self.assertFalse(self.models.exists())
 def test_unknown_or_empty_selection_is_rejected(self):
  for identifiers in ([],['unknown'],['12b','unknown']):
   with self.assertRaises(ValueError):installer.install(self.assets,self.models,identifiers)
  self.assertFalse(self.models.exists())
 def test_concurrent_target_is_not_overwritten_and_temp_is_cleaned(self):
  def create_racing_target(source,target):
   Path(target).write_bytes(b'racing owner');raise FileExistsError('Concurrent target')
  with patch.object(installer.os,'link',side_effect=create_racing_target):
   with self.assertRaises(FileExistsError):installer.install(self.assets,self.models,['12b'])
  self.assertEqual((self.models/'12b.gguf').read_bytes(),b'racing owner');self.assertFalse(list(self.models.glob('.assistant-*')))
 def test_failed_copy_validation_cleans_temp_without_final_artifact(self):
  actual=installer.verify;calls=0
  def verify_after_copy(path,expected):
   nonlocal calls
   calls+=1
   if calls==2:Path(path).write_bytes(b'truncated copy')
   return actual(path,expected)
  with patch.object(installer,'verify',side_effect=verify_after_copy):
   with self.assertRaises(ValueError):installer.install(self.assets,self.models,['12b'])
  self.assertFalse((self.models/'12b.gguf').exists());self.assertFalse(list(self.models.glob('.assistant-*')))
 def test_licensing_source_symlinks_are_rejected_before_install(self):
  for name in ('LICENSE-APACHE-2.0.txt','NOTICE.txt','manifest.json'):
   with self.subTest(name=name):
    original=(self.assets/name).read_bytes();outside=self.root/('external-'+name);outside.write_bytes(original);(self.assets/name).unlink();(self.assets/name).symlink_to(outside)
    try:
     with self.assertRaises((ValueError,OSError)):installer.install(self.assets,self.models,['12b'])
     self.assertFalse(self.models.exists())
    finally:
     (self.assets/name).unlink();(self.assets/name).write_bytes(original)
 def test_destination_license_symlink_cannot_overwrite_outside(self):
  self.models.mkdir();outside=self.root/'unrelated.txt';outside.write_text('Untouched unrelated data');(self.models/'NOTICE.txt').symlink_to(outside)
  with self.assertRaises((ValueError,OSError)):installer.install(self.assets,self.models,['12b'])
  self.assertEqual(outside.read_text(),'Untouched unrelated data')
 def test_existing_target_symlink_cannot_escape_models(self):
  self.models.mkdir();outside=self.root/'outside.gguf';outside.write_bytes(self.data['12b']);(self.models/'12b.gguf').symlink_to(outside)
  with self.assertRaises((ValueError,OSError)):installer.install(self.assets,self.models,['12b'])
if __name__=='__main__':unittest.main()
