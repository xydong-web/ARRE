#!/usr/bin/env python3
"""Paper-named ARRE release verification / document reconstruction interface."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def verify():
    manifest=json.loads((ROOT/'SHA256SUMS.json').read_text())
    failed=[]
    for rel,dig in manifest.items():
        p=ROOT/rel
        if not p.is_file() or checksum(p)!=dig:failed.append(rel)
    print(json.dumps({'files':len(manifest),'valid':not failed,'bad':failed[:30]},indent=2))
    if failed:raise SystemExit(2)

def run(script,addition=None):
    f=ROOT/'manuscript'/'scripts'/script
    if not f.exists():raise SystemExit(f'Missing archived script: {f}')
    if script in ('figures.py','rebuild_all_quantitative.py') and not (ROOT/'manuscript/source/data').is_dir():
        raise SystemExit('Frozen figure inputs missing')
    subprocess.run([sys.executable,str(f)]+(addition or []),cwd=f.parent.parent,check=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('task',choices=['verify','rebuild-figures','rebuild-figures-narrative'])
    a=ap.parse_args()
    if a.task=='verify':verify()
    elif a.task=='rebuild-figures-narrative':run('figures.py',['--narrative-revision'])
    else:run('rebuild_all_quantitative.py')
