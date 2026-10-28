import json, collections, pathlib, hashlib, os
ROOT = pathlib.Path(os.environ['KURAMOTO_REPO'])
RUN = pathlib.Path(os.environ['KURAMOTO_SOURCE_RUN'])
def shape(v):
    if isinstance(v, dict): return {k: shape(a) for k,a in v.items()}
    if isinstance(v, list): return {'length':len(v),'first':shape(v[0]) if v else None}
    return v
for name in ['sweep/checkpoint.json','sweep/result.json','sweep/top-limit.json','finalist-refinement.json']:
    path=pathlib.Path(os.environ['KURAMOTO_FINITE_WITNESS']) if name=='finalist-refinement.json' else RUN/name
    d=json.loads(path.read_text())
    print('\nFILE',path, path.stat().st_size)
    if 'checkpoint' in name:
        for k,v in d.items():
            print(k, shape(v[:2]) if isinstance(v,list) else (list(v)[:12] if isinstance(v,dict) else v))
    elif name.endswith('top-limit.json'):
        print(json.dumps(d,indent=1)[:15000])
    else:
        print(json.dumps(shape(d),indent=1)[:8500])
print('snapshots',len(list((RUN/'sweep/joint-results').glob('*.json'))))
print('TRACE FIRST', (RUN/'sweep/trace.jsonl').read_text().splitlines()[:4])
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ('src','docs','examples') for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
hashes['README.md']=hashlib.sha256((ROOT/'README.md').read_bytes()).hexdigest()
pathlib.Path('work/source_hashes_before.json').write_text(json.dumps(hashes,indent=2))
