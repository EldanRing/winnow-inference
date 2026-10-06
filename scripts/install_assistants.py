#!/usr/bin/env python3
"""Install exact MTP GGUFs from an unpacked release asset; never download implicitly."""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from verify_model import verify
ROOT=Path(__file__).resolve().parents[1]

def install(asset_dir, model_dir, identifiers):
    registry=json.loads((ROOT/'manifests/assistants-v1.json').read_text())['assets']
    if not identifiers or any(i not in registry for i in identifiers):
        raise ValueError('Choose a known assistant ID')
    asset_dir=asset_dir.resolve()
    expected={i:registry[i] for i in identifiers}
    sources={i:asset_dir/d['file'] for i,d in expected.items()}
    for i,p in sources.items():
        if p.resolve().parent!=asset_dir:
            raise ValueError('Assistant source must remain within the asset directory')
        verify(p,expected[i])
    license_file=asset_dir/'LICENSE-APACHE-2.0.txt'
    notice=asset_dir/'NOTICE.txt'
    if not license_file.is_file() or not notice.is_file():
        raise ValueError('Release asset license and conversion notice required')
    supporting=[license_file,notice,asset_dir/'manifest.json']
    for source in supporting:
        if source.exists() and source.resolve().parent!=asset_dir:
            raise ValueError('Supporting source must remain within the asset directory')
    # Validate every destination before copying anything, including notice files.
    for filename in [d['file'] for d in expected.values()]+[p.name for p in supporting]:
        target=model_dir/filename
        if target.is_symlink():
            raise ValueError('Destination symlinks are not allowed')
    model_dir.mkdir(parents=True,exist_ok=True)
    result={}
    for i,source in sources.items():
        target=model_dir/expected[i]['file']
        if target.exists():
            verify(target,expected[i])
        else:
            name=None
            try:
                with tempfile.NamedTemporaryFile(dir=model_dir,prefix='.assistant-',delete=False) as temporary:
                    name=Path(temporary.name)
                    with source.open('rb') as stream:shutil.copyfileobj(stream,temporary,8*1024*1024)
                verify(name,expected[i]);name.chmod(0o644)
                # Refuse to overwrite a concurrently created file.
                os.link(name,target)
            finally:
                if name is not None:name.unlink(missing_ok=True)
        result[i]=str(target.resolve())
    for source in supporting:
        if source.is_file():shutil.copy2(source,model_dir/source.name)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True,help='Unpacked assistant-assets-v1 directory')
    p.add_argument('--model-dir',type=Path,default=ROOT/'models/assistants')
    p.add_argument('--assistant',choices=['12b','e4b','e2b'],action='append',help='Default: 12B and E4B; select E2B explicitly')
    a=p.parse_args()
    try:print(json.dumps(install(a.assets,a.model_dir,a.assistant or ['12b','e4b']),indent=2))
    except (ValueError,OSError) as error:p.exit(1,'Assistant install failed: '+str(error)+'\n')
if __name__=='__main__':main()
