"""Finite-h probes of C4-derived geometries, with exact rational certification."""

from dataclasses import asdict
import argparse,gzip,hashlib,json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from kuramoto_template.asymptotic_search.c4_discovery import generic_geometry,generic_parameters
from kuramoto_template.asymptotic_search.c4_multisplit import MirrorFamily
from kuramoto_template.asymptotic_search.c4_finite_probe import FiniteC4Probe
from kuramoto_template.asymptotic_search.c4_session import DurableLedger,DeadlineReached
from kuramoto_template.asymptotic_search.interval_certification import rational_finite_repair,verify_finite
from kuramoto_template.asymptotic_search.joint_limit import JointLimitConfig,result_from_data
from kuramoto_template.asymptotic_search.joint_outer import _write_with_retry as write
from kuramoto_template.asymptotic_search.persistence import _code_hash,json_safe
from kuramoto_template.asymptotic_search.solvers import CallLimit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=900.)
    parser.add_argument('--max-calls',type=int,default=20000)
    parser.add_argument('--local-calls',type=int,default=1500)
    parser.add_argument('--allow-opposite',action='store_true')
    parser.add_argument('--free',action='store_true')
    parser.add_argument('--mass-by-occupancy',action='store_true')
    parser.add_argument('--optimize-h',action='store_true')
    parser.add_argument('--h',type=float,nargs='+',default=[.0003,.001,.003,.01,.03,.1,.3])
    parser.add_argument('--gamma',type=float,default=1e-8)
    args=parser.parse_args()
    raw=json.loads(args.input.read_text());source=result_from_data(raw.get('result',raw));c=source.candidate
    if c is None or not c.family_id.startswith('c4/'):
        parser.error('Need a C4-derived source')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'winners').mkdir();(args.output/'certificates').mkdir()
    journal=gzip.open(args.output/'events.jsonl.gz','wt',encoding='utf-8',compresslevel=4)
    def emit(data):journal.write(json.dumps(json_safe(data),allow_nan=False)+'\n');journal.flush()
    ledger=DurableLedger(emit,args.max_calls);ledger.limits['c4_finite_sdp']=args.max_calls
    cfg=JointLimitConfig(gamma0=args.gamma,secondary=False,box_buffer=1e-10,
        solver_options=dict(tol_gap_abs=1e-12,tol_gap_rel=1e-12,tol_feas=1e-12,max_iter=1000))
    family=MirrorFamily(tuple(int(k) for k in np.bincount(c.groups,minlength=4)),mass_by_occupancy=args.mass_by_occupancy)
    v0=generic_parameters(c) if args.free else family.encode(c)
    geometry=(lambda v:generic_geometry(c,v)) if args.free else family.geometry
    engine=FiniteC4Probe(c,cfg,allow_opposite=args.allow_opposite,ledger=ledger)
    started=perf_counter();deadline=started+args.seconds
    state=dict(schema_version=1,settings=vars(args),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_hash=_code_hash(),
        counts=ledger.counts,s7_data_used=False,best_numeric_mu=None,best_certified_mu=None,
        stages=[],stop_reason='active',proposals=0)
    best=None;certified=-1.

    def certify(result,index):
        nonlocal certified
        if result['P'] is None or result['mu']<=certified:return
        data=None
        try:
            data,changes=rational_finite_repair(json_safe(result))
            proof=verify_finite(data,precision_bits=256)
            emit(dict(event='certificate_success',source=index,exact_data=data,changes=changes,result=proof))
            mu=float(proof['density']['lower'])
            if mu>certified:
                certified=mu
                path=args.output/'winners'/f'{index:07d}.json';write(path,result)
                root=Path(__file__).resolve().parents[1]
                certificate=dict(schema_version=1,name='C4 finite probe',kind='finite',precision_bits=256,
                    arithmetic='python-flint / FLINT-Arb',source_file=path.resolve().relative_to(root).as_posix(),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_reported_mu=result['mu'],
                    exact_data=data,changes=changes,result=proof,
                    original_snapshot_attempt=dict(certified=False,reason='Explicit rational repair only'),
                    graph_realization_certified=False)
                write(args.output/'certificates'/f'{index:07d}.json',certificate)
                write(args.output/'best-certificate.json',certificate)
                state['best_certified_mu']=proof['density']['lower']
                print('CERTIFIED FINITE',proof['density']['lower'],'h',result['h'],flush=True)
        except Exception as exc:
            emit(dict(event='certificate_failure',source=index,failure=str(exc),exact_data=data))

    reason='completed'
    try:
        for start_h in args.h:
            cache={};first=ledger.counts.get('c4_finite_sdp',0);local_best=None
            start=np.r_[v0,np.log(start_h)] if args.optimize_h else v0.copy()
            bounds=[(-6.,6.)]*len(v0)+([(-9.21,-.69)] if args.optimize_h else [])
            def objective(v):
                nonlocal best,local_best
                if perf_counter()>=deadline:raise DeadlineReached('wall_time')
                key=np.asarray(v).tobytes()
                if key in cache:return cache[key]
                if ledger.counts.get('c4_finite_sdp',0)-first>=args.local_calls:raise CallLimit('local_budget')
                g=geometry(v[:-1] if args.optimize_h else v)
                h=float(np.exp(v[-1])) if args.optimize_h else start_h
                if g.x.min()<1e-5:return 1.
                index=state['proposals'];state['proposals']+=1
                emit(dict(event='proposal',id=index,parameters=v,candidate=asdict(g),h=h,config=asdict(cfg)))
                result=engine.run(g,h)
                emit(dict(event='result',id=index,result=result))
                valid=result['mu'] is not None and result.get('diagnostics',{}).get('independent_primal_check_inf',1)<1e-6
                value=(-result['mu']+10*max(0.,result['diagnostics']['independent_primal_check_inf']-1e-9)) if valid else 1.
                cache[key]=value
                if valid and (local_best is None or result['mu']>local_best[0]['mu']):local_best=result,index,v.copy()
                if valid and (best is None or result['mu']>best[0]['mu']):
                    best=result,index,v.copy();state['best_numeric_mu']=result['mu'];write(args.output/'top-search.json',result)
                if index%25==0:
                    state['elapsed_seconds']=perf_counter()-started;write(args.output/'summary.json',state)
                return value
            try:
                if not len(start):objective(start)
                else:
                    simplex=np.tile(start,(len(start)+1,1))
                    for i in range(len(start)):simplex[i+1,i]+=.005
                    opt=minimize(objective,start,method='Nelder-Mead',bounds=bounds,
                        options=dict(initial_simplex=simplex,maxfev=3*args.local_calls,
                            maxiter=3*args.local_calls,xatol=1e-8,fatol=1e-11,adaptive=True))
            except CallLimit as exc:
                if str(exc)!='local_budget':raise
            if local_best:
                certify(local_best[0],local_best[1])
                state['stages'].append(dict(start_h=start_h,best_mu=local_best[0]['mu'],
                    best_h=local_best[0]['h'],calls=ledger.counts['c4_finite_sdp']-first))
                print(state['stages'][-1],flush=True)
    except (DeadlineReached,CallLimit) as exc:reason=str(exc)
    except KeyboardInterrupt:reason='interrupted'
    finally:
        if best:certify(best[0],best[1])
        state.update(stop_reason=reason,elapsed_seconds=perf_counter()-started)
        write(args.output/'summary.json',state);journal.close()


if __name__=='__main__':main()
