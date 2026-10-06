#!/usr/bin/env python3
"""One-time bootstrap for installations predating the update command."""
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

REPO='EldanRing/winnow-inference'
TAG='v2026.10.06'
BASE=f'https://github.com/{REPO}/releases/download/{TAG}/'

def download(url,limit):
    with urllib.request.urlopen(url,timeout=60) as response:
        final=urllib.parse.urlsplit(response.geturl())
        if final.scheme!='https' or final.hostname not in {'github.com','release-assets.githubusercontent.com','objects.githubusercontent.com'}:
            raise ValueError('Unexpected bootstrap redirect')
        data=response.read(limit+1)
    if len(data)>limit:raise ValueError('Bootstrap download exceeds its limit')
    return data

def main():
    if len(sys.argv)!=2:raise SystemExit('Usage: bootstrap_update.py /path/to/existing/winnow')
    root=Path(sys.argv[1]).absolute()
    sums=download(BASE+'SHA256SUMS',1024**2).decode()
    rows=[re.fullmatch(r'([0-9a-f]{64})  winnow-update.py',line) for line in sums.splitlines()]
    matches=[row[1] for row in rows if row]
    if len(matches)!=1:raise ValueError('Missing or duplicate updater checksum')
    payload=download(BASE+'winnow-update.py',1024**2)
    if hashlib.sha256(payload).hexdigest()!=matches[0]:raise ValueError('Updater checksum mismatch')
    with tempfile.TemporaryDirectory(prefix='winnow-bootstrap-') as td:
        script=Path(td)/'update.py';script.write_bytes(payload)
        subprocess.run([sys.executable,str(script),'update','--install-dir',str(root)],check=True)

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,subprocess.SubprocessError) as error:raise SystemExit('Bootstrap stopped: '+str(error))
