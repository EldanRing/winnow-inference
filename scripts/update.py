#!/usr/bin/env python3
"""Verified official updates with an atomic active version and one recovery copy."""
import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path, PurePosixPath

REPOSITORY = 'EldanRing/winnow-inference'
API = f'https://api.github.com/repos/{REPOSITORY}/releases/'
BASE = f'https://github.com/{REPOSITORY}/releases/download/'
MAX_DOWNLOAD = 256 * 1024**2
MAX_UNPACKED = 1024**3
OWNED = {'scripts','manifests','docs','examples','third_party','native','patches','tests',
         '.github','bin','README.md','LICENSE','runtime.lock.json','release-manifest.json',
         'candidate-manifest.json','CMakeLists.txt','Dockerfile','.clang-format','.dockerignore',
         '.gitattributes','.gitignore','ruff.toml'}

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):digest.update(block)
    return digest.hexdigest()

def atomic_bytes(path, data, mode=0o600):
    temporary=path.with_name(path.name+'.new')
    if temporary.is_symlink() or path.is_symlink():
        raise ValueError('Unsafe metadata symlink')
    if temporary.exists():
        info=temporary.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:raise ValueError('Unsafe metadata temporary file')
        temporary.unlink()
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,mode)
    with os.fdopen(fd,'wb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

def write_json(path, value):
    atomic_bytes(path,(json.dumps(value,indent=2)+'\n').encode())

def official(url):
    parsed=urllib.parse.urlsplit(url)
    if parsed.scheme!='https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Release URLs must be official HTTPS endpoints')
    if not (url.startswith(API) or url.startswith(BASE)):
        raise ValueError('Only the official Winnow release repository is allowed')

def get_bytes(url, limit=MAX_DOWNLOAD):
    official(url)
    request=urllib.request.Request(url,headers={'User-Agent':'winnow-update','Accept-Encoding':'identity'})
    with urllib.request.urlopen(request,timeout=60) as response:
        final=urllib.parse.urlsplit(response.geturl())
        if final.scheme!='https' or final.hostname not in {
                'github.com','api.github.com','release-assets.githubusercontent.com',
                'objects.githubusercontent.com'}:
            raise ValueError('Unexpected release redirect')
        data=response.read(limit+1)
    if len(data)>limit:raise ValueError('Release download exceeds its size limit')
    return data

def release_info(version=None, fetch=get_bytes):
    if version is not None and not re.fullmatch(r'v\d{4}\.\d{2}\.\d{2}(?:-[a-z0-9-]+)?',version):
        raise ValueError('Use a published version such as v2026.10.06')
    url=API+('tags/'+version if version else 'latest')
    value=json.loads(fetch(url,1024**2))
    tag=value.get('tag_name')
    if (not isinstance(tag,str) or not re.fullmatch(r'v\d{4}\.\d{2}\.\d{2}(?:-[a-z0-9-]+)?',tag)
            or value.get('draft') is not False or value.get('prerelease') is not False
            or (version and version!=tag)):
        raise ValueError('No matching stable published release is available')
    assets={}
    for item in value.get('assets',[]):
        name=item.get('name','');url=item.get('browser_download_url','')
        if not isinstance(name,str) or '/' in name or name in assets:
            raise ValueError('Invalid or duplicate release asset')
        if url!=BASE+tag+'/'+name:raise ValueError('Unexpected release asset URL')
        size=item.get('size')
        if type(size) is not int or not 0<size<=MAX_DOWNLOAD:raise ValueError('Invalid asset size')
        assets[name]={'url':url,'bytes':size}
    return tag,assets

def checksum_map(data):
    result={}
    for line in data.decode('utf-8').splitlines():
        if not line.strip():continue
        match=re.fullmatch(r'([0-9a-f]{64})  ([A-Za-z0-9._-]+)',line)
        if not match or match[2] in result:raise ValueError('Invalid release checksum list')
        result[match[2]]=match[1]
    return result

def extract(archive, staging, expected_root, fault=lambda _:None):
    staging.mkdir()
    seen=set();total=0
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            p=PurePosixPath(member.name)
            if (p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!=expected_root
                    or '\\' in member.name or member.name in seen or len(seen)>=4096):
                raise ValueError('Unsafe or duplicate archive path')
            seen.add(member.name)
            if not (member.isfile() or member.isdir()):raise ValueError('Archive links/devices are not allowed')
            total+=member.size
            if member.size<0 or member.size>MAX_DOWNLOAD or total>MAX_UNPACKED:
                raise ValueError('Archive exceeds unpacking limits')
            path=staging.joinpath(*p.parts)
            if member.isdir():path.mkdir(parents=True,exist_ok=True);continue
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as output,tar.extractfile(member) as source:
                shutil.copyfileobj(source,output,1024**2)
            if path.stat().st_size!=member.size:raise ValueError('Incomplete archive member')
            path.chmod(0o755 if member.mode & 0o111 else 0o644)
            fault('extract')
    return staging/expected_root

def verified_payload(payload, tag, kind):
    manifest_path=payload/'release-manifest.json'
    value=json.loads(manifest_path.read_text())
    if value.get('release_version')!=tag or value.get('package_kind')!=kind:
        raise ValueError('Release version/platform does not match the package')
    files=value.get('files')
    if not isinstance(files,dict) or not files:raise ValueError('Missing complete package inventory')
    actual={str(p.relative_to(payload)) for p in payload.rglob('*') if p.is_file()}
    if actual!=set(files)|{'release-manifest.json'}:raise ValueError('Incomplete or unexpected package contents')
    for name,entry in files.items():
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or p.parts[0] not in OWNED:
            raise ValueError('Unsafe package inventory path')
        if (not isinstance(entry,dict) or type(entry.get('bytes')) is not int
                or entry.get('bytes')!=(payload/name).stat().st_size
                or entry.get('sha256')!=sha(payload/name)):
            raise ValueError('Package member checksum mismatch: '+name)
    required={'scripts/winnow.py','scripts/update.py','scripts/serve.py','scripts/assets.py',
              'manifests/release-assets-v1.json','runtime.lock.json'}
    if not required<=set(files):raise ValueError('Incomplete Winnow installation')
    if kind=='runtime':
        if value.get('platform_system')!='Linux' or value.get('platform_machine')!='x86_64':
            raise ValueError('Unsupported runtime platform')
        if value.get('runtime_lock_sha256')!=sha(payload/'runtime.lock.json'):
            raise ValueError('Runtime lock mismatch')
        if value.get('binary_sha256')!=sha(payload/'bin/winnow-server'):
            raise ValueError('Native binary mismatch')
        if value['runtime_lock_sha256'].encode() not in (payload/'bin/winnow-server').read_bytes():
            raise ValueError('Native binary does not embed the matching runtime lock')
    return value

def build_source(payload, previous=None):
    command=[sys.executable,str(payload/'scripts/build.py')]
    control=payload.parents[1]
    cache_root=control/'runtime-cache'
    if cache_root.is_symlink():raise ValueError('Unsafe source runtime cache')
    cache_root.mkdir(exist_ok=True)
    runtime_cache=cache_root/sha(payload/'runtime.lock.json')
    if runtime_cache.is_symlink():raise ValueError('Unsafe source runtime cache entry')
    if runtime_cache.exists() and not (runtime_cache/'.ready').is_file():
        shutil.rmtree(runtime_cache)
    if not runtime_cache.exists():
        if (previous and (previous/'.runtime/llama.cpp').is_dir()
                and sha(previous/'runtime.lock.json')==runtime_cache.name):
            shutil.copytree(previous/'.runtime',runtime_cache)
        else:runtime_cache.mkdir()
    (payload/'.runtime').symlink_to(runtime_cache,target_is_directory=True)
    cache=previous/'.build/CMakeCache.txt' if previous else None
    if cache and cache.is_file():
        settings={}
        for line in cache.read_text(errors='replace').splitlines():
            if '=' in line and ':' in line.split('=',1)[0]:
                key,value=line.split('=',1);settings[key.split(':',1)[0]]=value
        if settings.get('GGML_METAL')=='ON':command+=['--backend','metal']
        elif settings.get('GGML_CUDA')=='ON':command+=['--backend','cuda']
        elif settings.get('GGML_CUDA')=='OFF' and settings.get('GGML_METAL')=='OFF':command+=['--backend','cpu']
        if settings.get('CMAKE_CUDA_COMPILER'):command+=['--cuda-compiler',settings['CMAKE_CUDA_COMPILER']]
        if settings.get('CMAKE_CUDA_ARCHITECTURES'):command+=['--cuda-arch',settings['CMAKE_CUDA_ARCHITECTURES']]
        if settings.get('OPENSSL_USE_STATIC_LIBS')=='TRUE':command+=['--static-openssl']
    subprocess.run(command,cwd=payload,check=True)
    if not (payload/'.build/bin/winnow-server').is_file():raise ValueError('Source build did not produce a server')
    if sha(payload/'runtime.lock.json').encode() not in (payload/'.build/bin/winnow-server').read_bytes():
        raise ValueError('Source server does not embed the matching runtime lock')
    atomic_bytes(runtime_cache/'.ready',runtime_cache.name.encode())

def check_runtime(payload):
    result=subprocess.run(['ldd',str(payload/'bin/winnow-server')],capture_output=True,text=True)
    if result.returncode or 'not found' in result.stdout:raise ValueError('Missing native runtime libraries; use update --source with build prerequisites')
    manifest=json.loads((payload/'release-manifest.json').read_text())
    supported=manifest.get('cuda_architectures')
    if not isinstance(supported,list) or not supported:raise ValueError('Runtime GPU compatibility metadata is missing')
    try:
        response=subprocess.check_output(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],text=True)
        available={int(float(line.strip())*10) for line in response.splitlines() if line.strip()}
    except (OSError,ValueError,subprocess.SubprocessError):
        raise ValueError('Cannot verify CUDA GPU compatibility; use update --source with build prerequisites')
    if not available or not available<=set(supported):
        raise ValueError('This CUDA archive does not cover all detected GPUs; use update --source to build for this host')

def no_symlinks(path):
    if path.is_symlink() or any(p.is_symlink() for p in path.rglob('*')):
        raise ValueError('Unexpected installation symlink: '+str(path))

def active(control):
    pointer=control/'current'
    if not pointer.is_symlink():raise ValueError('Managed active version is missing')
    target=Path(os.readlink(pointer))
    if (target.is_absolute() or len(target.parts)!=2 or target.parts[0]!='releases'
            or not re.fullmatch(r'[A-Za-z0-9._-]+',target.parts[1])):
        raise ValueError('Unsafe active version pointer')
    result=control/target
    if result.is_symlink() or not result.is_dir():raise ValueError('Active version is missing')
    return result

def select(control, payload):
    temp=control/'current.new'
    if temp.exists() or temp.is_symlink():temp.unlink()
    temp.symlink_to('releases/'+payload.name)
    os.replace(temp,control/'current')
    fd=os.open(control,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def metadata(payload):
    path=payload/'.install.json'
    if path.is_symlink():raise ValueError('Unsafe managed metadata')
    value=json.loads(path.read_text())
    previous=value.get('previous')
    if previous is not None and (not isinstance(previous,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',previous)):
        raise ValueError('Unsafe recovery version identifier')
    if value.get('kind') not in {'source','runtime'} or not isinstance(value.get('owned_files'),dict):
        raise ValueError('Invalid managed version metadata')
    for name,digest in value['owned_files'].items():
        p=PurePosixPath(name)
        if (p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] not in OWNED
                or not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest)):
            raise ValueError('Invalid managed file inventory')
    return value

def unchanged(payload):
    value=metadata(payload)
    for name,digest in value['owned_files'].items():
        p=payload/name
        if p.is_symlink() or not p.is_file() or sha(p)!=digest:
            raise ValueError('Local code/configuration edits detected; update will not overwrite '+name)
    allowed=set(value['owned_files'])|{'.install.json'}
    for p in payload.rglob('*'):
        rel=p.relative_to(payload)
        if rel.parts[0] in {'models','.build','.runtime'}:continue
        if '__pycache__' in rel.parts and p.suffix=='.pyc' and not p.is_symlink():continue
        if p.is_symlink() or (p.is_file() and str(rel) not in allowed):
            raise ValueError('Unmanaged file in update payload; move user data outside managed code: '+str(rel))

def prune(control):
    current=active(control);previous=metadata(current).get('previous')
    keep={current.name,previous}
    for p in (control/'releases').iterdir():
        if p.name in keep:continue
        if p.is_symlink() or not p.is_dir():raise ValueError('Unsafe managed release directory')
        # Only remove updater-created versions, never arbitrary user directories.
        if not (p/'.install.json').is_file():raise ValueError('Unknown directory in managed releases')
        unchanged(p)
        shutil.rmtree(p)
    cache_root=control/'runtime-cache'
    if cache_root.is_symlink():raise ValueError('Unsafe source runtime cache')
    if cache_root.exists():
        keys={sha((control/'releases'/name)/'runtime.lock.json') for name in keep if name}
        for p in cache_root.iterdir():
            if p.name in keys:continue
            if p.is_symlink() or not p.is_dir() or not re.fullmatch(r'[0-9a-f]{64}',p.name):
                raise ValueError('Unexpected source runtime cache entry')
            shutil.rmtree(p)

def shim(root, control):
    bin_dir=root/'bin'
    if bin_dir.is_symlink():raise ValueError('Unsafe bootstrap bin directory')
    bin_dir.mkdir(exist_ok=True)
    text='''#!/usr/bin/env python3
import os,sys
from pathlib import Path
root=Path(__file__).absolute().parents[1]
control=root/'.winnow'
action=sys.argv[1] if len(sys.argv)>1 else ''
if action in {'update','rollback'}:
    script=control/'updater.py'
    command=[sys.executable,str(script),action,'--install-dir',str(root),*sys.argv[2:]]
else:
    payload=(control/'current').resolve()
    command=[sys.executable,str(payload/'scripts/winnow.py'),*sys.argv[1:]]
os.execv(sys.executable,command)
'''
    p=bin_dir/'winnow'
    atomic_bytes(p,text.encode(),0o755)
    for name,script in [('winnow-serve','serve.py'),('winnow-decide','decision_client.py'),('winnow-server',None)]:
        p=bin_dir/name
        text='#!/usr/bin/env python3\nimport os,sys\nfrom pathlib import Path\n'
        text+='root=Path(__file__).absolute().parents[1]\npayload=(root/".winnow/current").resolve()\n'
        text+='server=payload/"bin/winnow-server"\nif not server.is_file():server=payload/".build/bin/winnow-server"\n'
        if script:
            text+='command=[sys.executable,str(payload/"scripts"/'+repr(script)+')]\n'
            if name=='winnow-serve':text+='command += ["--server",str(server)]\n'
            text+='os.execv(sys.executable,command+sys.argv[1:])\n'
        else:text+='os.execv(str(server),[str(server),*sys.argv[1:]])\n'
        atomic_bytes(p,text.encode(),0o755)

def adopt(root, control, fault):
    """Bootstrap once; a durable journal makes incomplete adoption resumable."""
    journal=control/'adoption.json'
    if journal.is_symlink():raise ValueError('Unsafe bootstrap journal')
    if journal.exists():
        state=json.loads(journal.read_text());payload=control/'releases'/state['payload']
        if (not re.fullmatch(r'bootstrap-[0-9a-f]{32}',state['payload'])
                or not isinstance(state.get('names'),list) or not set(state['names'])<=OWNED
                or payload.is_symlink() or (not state.get('copying') and not payload.is_dir())):
            raise ValueError('Unsafe bootstrap journal contents')
        if state.get('copying'):
            if (control/'current').is_symlink():raise ValueError('Unexpected bootstrap copy state')
            if payload.exists():shutil.rmtree(payload)
            journal.unlink()
            return adopt(root,control,fault)
        if not (control/'current').is_symlink():select(control,payload)
    else:
        if (root/'.git').exists():
            changed=subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True)
            if changed:raise ValueError('Dirty source checkout: commit/stash local source edits before using update')
            untracked=subprocess.check_output(['git','-C',str(root),'ls-files','--others','--exclude-standard'],text=True).splitlines()
            if any(PurePosixPath(p).parts[0] in OWNED for p in untracked):
                raise ValueError('Untracked files in source directories; move them outside managed code before update')
        if not (root/'scripts/winnow.py').is_file() or not (root/'runtime.lock.json').is_file():
            raise ValueError('Choose the existing Winnow installation directory')
        if not (root/'.git').exists():
            for name in ('release-manifest.json','candidate-manifest.json'):
                p=root/name
                if p.is_file():
                    inventory=json.loads(p.read_text()).get('files',{})
                    for file,entry in inventory.items():
                        expected=entry if isinstance(entry,str) else entry['sha256']
                        if not (root/file).is_file() or sha(root/file)!=expected:
                            raise ValueError('Local runtime edits detected; update will not overwrite '+file)
                    unexpected=[str(p.relative_to(root)) for n in OWNED for p in (root/n).rglob('*')
                                if p.is_file() and str(p.relative_to(root)) not in inventory
                                and p.name not in {'release-manifest.json','candidate-manifest.json'}
                                and '__pycache__' not in p.parts and p.suffix!='.pyc']
                    if unexpected:raise ValueError('Unmanaged files inside code directories; preserve them outside managed code: '+unexpected[0])
                    break
        names=sorted(n for n in OWNED if (root/n).exists())
        for name in names:no_symlinks(root/name)
        identifier='bootstrap-'+uuid.uuid4().hex
        payload=control/'releases'/identifier
        state={'payload':identifier,'names':names,'copying':True};write_json(journal,state)
        payload.mkdir()
        for name in names:
            p=root/name;out=payload/name
            if p.is_dir():shutil.copytree(p,out,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            else:shutil.copy2(p,out)
            fault('adopt_copy')
        (payload/'models').symlink_to(root/'models',target_is_directory=True)
        for name in ('.build','.runtime'):
            if (root/name).exists():(payload/name).symlink_to(root/name,target_is_directory=True)
        version='bootstrap'
        for name in ('release-manifest.json','candidate-manifest.json'):
            if (payload/name).exists():version=json.loads((payload/name).read_text()).get('release_version','bootstrap')
        owned={str(p.relative_to(payload)):sha(p) for p in payload.rglob('*')
               if p.is_file() and p.relative_to(payload).parts[0] not in {'models','.build','.runtime'}}
        write_json(payload/'.install.json',{'version':version,'kind':'source' if (root/'CMakeLists.txt').exists() else 'runtime',
                   'previous':None,'owned_files':owned})
        state['copying']=False;write_json(journal,state)
        select(control,payload)
    remnants=control/'adoption-remnants'
    if remnants.is_symlink():raise ValueError('Unsafe bootstrap remnants')
    remnants.mkdir(exist_ok=True)
    for name in state['names']:
        p=root/name
        if name=='bin':
            if p.exists() and not (remnants/name).exists():os.rename(p,remnants/name)
            shim(root,control)
        elif p.is_symlink():
            if os.readlink(p)!=f'.winnow/current/{name}':raise ValueError('Unexpected installation view')
        else:
            if p.exists() and not (remnants/name).exists():os.rename(p,remnants/name)
            p.symlink_to(f'.winnow/current/{name}',target_is_directory=(payload/name).is_dir())
        fault('adopt')
    shim(root,control)
    # These are duplicated owned code files copied into the bootstrap payload, not models/config roots.
    shutil.rmtree(remnants);journal.unlink()
    return payload

def run(root, rollback=False, version=None, fetch=get_bytes, build=build_source,
        runtime_check=check_runtime, fault=lambda _:None, from_source=False):
    system,machine=platform.system(),platform.machine().lower()
    if system not in {'Linux','Darwin'}:raise ValueError('Updates support Linux and macOS source installations')
    root=Path(root).absolute()
    if root.is_symlink():raise ValueError('Choose the real installation directory, not a symlink')
    root=root.resolve()
    control=root/'.winnow'
    if control.is_symlink():raise ValueError('Unsafe updater storage symlink')
    control.mkdir(mode=0o700,exist_ok=True)
    releases=control/'releases'
    if releases.is_symlink():raise ValueError('Unsafe release storage symlink')
    releases.mkdir(exist_ok=True)
    lock=control/'lock'
    fd=os.open(lock,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'w') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):raise ValueError('Unsafe updater lock')
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('Another update is running')
        # The rescue updater is independent of the version selected for user commands.
        rescue=control/'updater.py'
        if rescue.is_symlink():raise ValueError('Unsafe rescue updater')
        source=Path(__file__).read_bytes()
        if not rescue.exists():atomic_bytes(rescue,source)
        if (control/'current').exists() and not (control/'current').is_symlink():
            raise ValueError('Unsafe active-version storage')
        if (control/'adoption.json').exists() or not (control/'current').is_symlink():
            current=adopt(root,control,fault)
        else:current=active(control)
        unchanged(current)
        # Recover the helper refresh if activation was interrupted after the switch.
        if (current/'scripts/update.py').is_file():
            atomic_bytes(rescue,(current/'scripts/update.py').read_bytes())
        old=metadata(current)
        prune(control)
        if rollback:
            previous=old.get('previous')
            if not previous:raise ValueError('No automatic recovery version is available')
            candidate=releases/previous
            if candidate.is_symlink() or not candidate.is_dir():raise ValueError('Recovery version is missing')
            unchanged(candidate);info=metadata(candidate);info['previous']=current.name
            write_json(candidate/'.install.json',info);select(control,candidate);prune(control)
            return {'status':'rolled_back','version':info['version'],'command':str(root/'bin/winnow')}
        tag,assets=release_info(version,fetch)
        if old['version']==tag and not (from_source and old['kind']!='source'):
            prune(control)
            return {'status':'already_current','version':tag,'command':str(root/'bin/winnow')}
        kind='source' if from_source or old['kind']=='source' or system=='Darwin' or machine not in {'x86_64','amd64'} else 'runtime'
        if kind=='runtime' and system!='Linux':raise ValueError('CUDA archives do not run on macOS')
        filename=f'winnow-{"source" if kind=="source" else "linux-x86_64-cuda"}-{tag}.tar.gz'
        print(f'Updating to {tag} ({kind})...',file=sys.stderr,flush=True)
        if filename not in assets or 'SHA256SUMS' not in assets:raise ValueError('Release does not provide matching verified update assets')
        sums=checksum_map(fetch(assets['SHA256SUMS']['url'],1024**2))
        if filename not in sums:raise ValueError('Release checksum is missing for the selected asset')
        stage=control/'staging'
        if stage.is_symlink():raise ValueError('Unsafe staging symlink')
        if stage.exists():shutil.rmtree(stage)
        stage.mkdir()
        data=fetch(assets[filename]['url'],MAX_DOWNLOAD)
        if len(data)!=assets[filename]['bytes'] or hashlib.sha256(data).hexdigest()!=sums[filename]:
            raise ValueError('Release download checksum/size mismatch; current installation unchanged')
        archive=stage/'download.tar.gz';archive.write_bytes(data);fault('download')
        payload=extract(archive,stage/'unpacked',filename[:-7],fault)
        verified_payload(payload,tag,kind)
        if (payload/'models').exists():raise ValueError('Release must not contain user models')
        (payload/'models').symlink_to(root/'models',target_is_directory=True)
        identifier=tag+'-'+uuid.uuid4().hex
        owned={str(p.relative_to(payload)):sha(p) for p in payload.rglob('*')
               if p.is_file() and p.relative_to(payload).parts[0] not in {'models','.build','.runtime'}}
        write_json(payload/'.install.json',{'version':tag,'kind':kind,'previous':current.name,'owned_files':owned})
        destination=releases/identifier;os.rename(payload,destination)
        # Build at the final stable path, so source-build library paths survive activation.
        if kind=='source':
            if build is build_source:build(destination,current)
            else:build(destination)
        else:runtime_check(destination)
        fault('build');fault('before_swap')
        select(control,destination);fault('after_swap')
        prune(control)
        shutil.rmtree(stage)
        # Atomic replace keeps the rescue updater current without changing the launcher.
        atomic_bytes(rescue,(destination/'scripts/update.py').read_bytes())
        return {'status':'updated','version':tag,'command':str(root/'bin/winnow'),
                'cache_note':'F16 is the default; retain --cache q8_0 explicitly on serve/decide to keep Q8.',
                'service_note':'No running server was restarted. Use this launcher on your next start.'}

def install_root():
    here=Path(__file__).resolve()
    for parent in here.parents:
        if parent.name=='.winnow':return parent.parent
    return here.parents[1]

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['update','rollback'])
    parser.add_argument('--install-dir',type=Path,default=install_root())
    parser.add_argument('--version',help='Optional exact stable published release; default latest')
    parser.add_argument('--source',action='store_true',help='Build the verified source release for this host')
    args=parser.parse_args(argv)
    try:
        if args.action=='rollback' and args.version:raise ValueError('Rollback uses the automatic recovery version')
        print(json.dumps(run(args.install_dir,args.action=='rollback',args.version,from_source=args.source),indent=2))
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError,tarfile.TarError) as error:
        parser.exit(1,'Update stopped: '+str(error)+'\nYour models/settings and running service are untouched. Rerun the same command after fixing the reported issue.\n')

if __name__=='__main__':main()
