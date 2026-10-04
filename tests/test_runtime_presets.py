"""Artifact/profile override guards; no GPU required."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import serve

class RuntimePresets(unittest.TestCase):
 def test_named_profiles_bind_exact_target_assistant_projector_and_settings(self):
  for name in ['12b-nvfp4-vision8k-mtp','e4b-q8-vision8k-mtp','12b-q8-text8k-mtp','12b-q8-vision8k-mtp']:
   with tempfile.TemporaryDirectory() as td:
    files=[Path(td)/n for n in ['target','assistant','projector','server']]
    for p in files:p.touch()
    argv=['serve','--preset',name,'--model',str(files[0]),'--assistant',str(files[1]),'--server',str(files[3])]
    if 'vision' in name:argv+=['--mmproj',str(files[2])]
    argv+=['--context','4k','--decision-parallel','2','--batch','512','--ubatch','256']
    with patch.object(sys,'argv',argv),patch('platform.system',return_value='Linux'),patch.object(serve,'verify') as verify,patch.object(serve.os,'execve') as launch,patch.dict(serve.os.environ,{'LLAMA_ARG_SPEC_TYPE':'draft-mtp','LLAMA_ARG_MMPROJ_URL':'https://invalid.test/a','KEEP_USER_SETTING':'yes'}):
     serve.main()
    command=launch.call_args.args[1];env=launch.call_args.args[2];spec=serve.RUNTIME_PRESETS[name]
    self.assertEqual(command[command.index('--ctx-size')+1],'4096')
    self.assertEqual(command[command.index('--batch-size')+1],'512')
    self.assertEqual(env['WINNOW_RESIDENT_MTP'],'1')
    self.assertNotIn('LLAMA_ARG_SPEC_TYPE',env)
    self.assertNotIn('LLAMA_ARG_MMPROJ_URL',env)
    self.assertEqual(env['KEEP_USER_SETTING'],'yes')
    self.assertEqual(env['WINNOW_MANAGED_LAUNCH'],'1')
    self.assertEqual(env['WINNOW_RESIDENT_VISION_MTP'],'1' if 'vision' in name else '0')
    self.assertEqual(verify.call_args_list[0].args[1],spec['target'])
    self.assertEqual(verify.call_args_list[1].args[1],spec['assistant'])
    self.assertEqual(len(verify.call_args_list),3 if 'vision' in name else 2)
 def test_incompatible_overrides_are_rejected_before_loading(self):
  for override in [['--memory','exclusive'],['--chat-parallel','2'],['--text-only'],['--','--n-gpu-layers','0']]:
   with patch.object(sys,'argv',['serve','--preset','12b-nvfp4-vision8k-mtp',*override]),patch('platform.system',return_value='Linux'),patch.object(serve.os,'execve') as launch:
    with self.assertRaises(SystemExit):serve.main()
    launch.assert_not_called()

if __name__=='__main__':unittest.main()
