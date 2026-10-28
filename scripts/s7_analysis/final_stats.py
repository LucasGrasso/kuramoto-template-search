import os,sys,pathlib,json,itertools,hashlib,shutil,collections,subprocess
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=pathlib.Path(os.environ['KURAMOTO_REPO']);sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.cluster.hierarchy import linkage,fcluster,leaves_list
from scipy.spatial.distance import squareform
from kuramoto_template.asymptotic_search.joint_limit import result_from_data
from kuramoto_template.asymptotic_search.expansions import U4,internal_basis,coefficients,effective_matrix,internal_solve
from kuramoto_template.asymptotic_search.persistence import _code_hash
OUT=pathlib.Path('outputs');d=json.loads((OUT/'analisis-datos.json').read_text(encoding='utf-8'));e=json.loads((OUT/'experimentos.json').read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
rg=np.array([0,0,1,1,2,2,3]);rs=np.array(d['reference']['s']);byid={r['id']:r for r in d['all_rows']}
orig=OUT/'evidencia'/'originales';orig.mkdir(exist_ok=True)
for i in d['selected']:
 source=pathlib.Path(os.environ['KURAMOTO_SOURCE_RUN'])/f'sweep/joint-results/{i:06d}.json';shutil.copyfile(source,orig/source.name)
 r=result_from_data(json.loads(source.read_text()));c=r.candidate;M=np.array([c.x[c.groups==g].sum() for g in range(4)])
 cap=min(M[g]+2*min(M[(g-1)%4],M[(g+1)%4]) for g in range(4))
 byid[i]['capacity_bound']=float(cap);byid[i]['capacity_gap']=float(cap-r.mu0)
 a=d['details'][str(i)]['comparison'];sn=a['orientation']*(c.s-c.x@c.s);sn/=np.sqrt(c.x@(sn*sn));gg=(a['orientation']*c.groups+a['shift'])%4
 assign=np.array([min(np.flatnonzero(rg==gg[k]),key=lambda j:abs(sn[k]-rs[j])) for k in range(c.m)])
 sets=[np.flatnonzero(assign==k) for k in range(7)];assert all(len(z) for z in sets)
 masses=np.array([c.x[z].sum() for z in sets]);offset=np.array([c.x[z]@sn[z]/masses[j] for j,z in enumerate(sets)])
 A=np.zeros((c.m,7))
 for j,z in enumerate(sets):A[z,j]=c.x[z]/masses[j]
 P=A.T@c.p0@A;Q=P[np.ix_(assign,assign)];scatter=float(np.sqrt(np.sum(np.outer(c.x,c.x)*(c.p0-Q)**2)))
 d['details'][str(i)]['projection7']=dict(assignment=assign.tolist(),masses=masses.tolist(),offsets=offset.tolist(),p0=P.tolist(),row_scatter=scatter,
    offset_scatter=float(np.sqrt(np.sum(c.x*(sn-offset[assign])**2))))
 ee=coefficients(c,r.r);soft=U4.T@effective_matrix(c,r.r)@U4;vals,V=np.linalg.eigh(soft);eta=U4@V[:,0]
 z,_,_=internal_solve(c,ee.lin,ee.g@eta)
 sensitivities=[dict(edge=[a,b],physical_weight_partial=float((z[a]-z[b])**2),density_partial=float(c.x[a]*c.x[b]*(z[a]-z[b])**2)) for a,b in itertools.combinations(range(c.m),2) if c.groups[a]==c.groups[b]]
 d['details'][str(i)]['internal_sensitivities']=sorted(sensitivities,key=lambda v:-v['density_partial'])
 d['details'][str(i)]['internal_sensitivity_normalization']='unit Euclidean weakest effective-soft parent eigenvector'
 byid[i]['projection7_row_scatter']=scatter
 if i==600:
  vals,V=np.linalg.eigh(r.hlim);eta=U4@V[:3,0];eta/=np.linalg.norm(eta)
  d['best_weak_parent_mode']=eta.tolist()

L=linkage(squareform(np.array(d['distance_matrix'])),method='average')
d['clustering_threshold_sensitivity']={str(t):int(max(fcluster(L,t,criterion='distance'))) for t in (.05,.1,.15,.2,.3)}
d['cluster_leaf_order']=[d['selected'][i] for i in leaves_list(L)]
d['source_integrity']={}
before=json.loads(pathlib.Path('work/source_hashes_before.json').read_text())
changed=[p for p,h in before.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
d['source_integrity']=dict(files_checked=len(before),changed=changed,hash_current=_code_hash(),matches_saved_hash=_code_hash()==d['source_hash_saved'])
d['experiment_status_counts']=dict(collections.Counter(r['status'] for r in e['calls']))
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pathlib.Path(os.environ['KURAMOTO_SOURCE_RUN'])/'sweep/checkpoint.json',pathlib.Path(os.environ['KURAMOTO_SOURCE_RUN'])/'sweep/trace.jsonl',pathlib.Path(os.environ['KURAMOTO_SOURCE_RUN'])/'sweep/top-limit.json',pathlib.Path(os.environ['KURAMOTO_FINITE_WITNESS'])]}
write(OUT/'manifiesto.json',dict(repo=str(ROOT),branch='asymptotic_search',git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                              source=d['source_integrity'],input_sha256=manifest,original_selected=d['selected'],new_costs=e['counts']))
write(OUT/'analisis-datos.json',d)
import csv
for name,ids in [('top20.csv',d['top20']),('muestra-diversa.csv',[i for i in d['selected'] if i not in d['top20']]),('todos-los-resultados.csv',list(byid))]:
 rows=[byid[i] for i in ids];keys=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
with (OUT/'experimentos.csv').open('w',encoding='utf-8-sig',newline='') as f:
 rows=[dict(name=c['name'],status=c['status'],mu=c['mu'],**{k:v for k,v in c['meta'].items() if not isinstance(v,list)}) for c in e['calls']]
 keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
print('INTEGRITY',d['source_integrity'],'THRESHOLD',d['clustering_threshold_sensitivity'],'COUNTS',d['experiment_status_counts'])
print('SCATTER991',d['details']['991']['projection7']['row_scatter'],'WEAK',d['best_weak_parent_mode'])
