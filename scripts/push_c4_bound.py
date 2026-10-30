"""Bounded C4-only multisplit, symmetry-release and birth experiments.

Complete proposals, failures and actual invocations are retained in a compressed
journal. Numeric exploration merits never replace the original validator or
the independent rational certificate.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from kuramoto_template.asymptotic_search.c4_discovery import generic_geometry,generic_parameters
from kuramoto_template.asymptotic_search.c4_multisplit import MirrorFamily,c4_partitions
from kuramoto_template.asymptotic_search.c4_session import C4Session,DeadlineReached
from kuramoto_template.asymptotic_search.families import split_candidate
from kuramoto_template.asymptotic_search.families import normalize_offsets
from kuramoto_template.asymptotic_search.joint_limit import JointLimitConfig,result_from_data
from kuramoto_template.asymptotic_search.solvers import CallLimit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('polish','scan','birth','free','refine','random','needle'),default='scan')
    parser.add_argument('--engine',choices=('joint','recovered'),default='joint')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--seconds',type=float,default=600.)
    parser.add_argument('--max-calls',type=int,default=50000)
    parser.add_argument('--local-calls',type=int,default=400)
    parser.add_argument('--seed',type=int,default=401)
    parser.add_argument('--restarts',type=int,default=2)
    parser.add_argument('--max-per-parent',type=int,default=4)
    parser.add_argument('--max-size',type=int,default=16)
    parser.add_argument('--top',type=int,default=20)
    parser.add_argument('--initial-balance',type=float,default=.05)
    parser.add_argument('--mass-floor',type=float,default=1e-5)
    parser.add_argument('--birth-depth',type=int,default=1)
    parser.add_argument('--gradient-step',type=float,default=1e-5)
    parser.add_argument('--target-penalty',type=float,default=0.)
    parser.add_argument('--method',choices=('envelope','SLSQP','Nelder-Mead','Powell'),default='Nelder-Mead')
    parser.add_argument('--mass-by-occupancy',action='store_true')
    parser.add_argument('--no-parent-symmetry',action='store_true')
    parser.add_argument('--baseline',type=float,default=0.6929076590503417)
    args=parser.parse_args()
    if args.seconds<=0 or args.max_calls<1 or args.local_calls<1 or args.restarts<1:
        parser.error('Require positive budgets and restart count')
    config_data=json.loads(args.config.read_text()) if args.config else dict(
        gamma0=1e-8,secondary=False,box_buffer=1e-9,h_guard=.0003)
    cfg=JointLimitConfig(**config_data.get('inner',config_data))
    if cfg.secondary:
        parser.error('Use primary solves for explicit accounting')
    def target_start(c):
        if cfg.mode!='margin':return c
        blend=args.initial_balance
        x=c.x.copy();s=c.s.copy()
        for j in range(4):
            ids=c.groups==j
            mass=x[ids].sum()
            x[ids]*=(1-blend)+blend*.25/mass
            s[ids]-=x[ids]@s[ids]/x[ids].sum()
        x/=x.sum()
        return replace(c,x=x,s=normalize_offsets(x,s))
    source=None
    if args.input and args.mode!='refine':
        raw=json.loads(args.input.read_text())
        source=result_from_data(raw.get('result',raw))
        if source.candidate is None or not source.candidate.family_id.startswith('c4/'):
            parser.error('Input must have explicit C4 provenance')
        source=replace(source,candidate=target_start(source.candidate))
    if args.mode not in ('scan','refine','random') and source is None:
        parser.error('This mode requires a C4-derived --input')
    settings=vars(args).copy()
    settings['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.input and args.input.is_file():
        settings['input_sha256']=hashlib.sha256(args.input.read_bytes()).hexdigest()
    rng=np.random.default_rng(args.seed)
    session=C4Session(args.output,seconds=args.seconds,max_calls=args.max_calls,settings=settings)
    last_certificate=0.

    def local(geometry_function,v0,context,*,calls=None,bound=5.):
        nonlocal last_certificate
        v0=np.asarray(v0,dtype=float)
        budget=args.local_calls if calls is None else calls
        start_count=session.ledger.counts.get('joint_sdp',0)
        cache={}
        local_best=None
        info=dict(context=context,first_call=start_count,budget=budget,dimension=len(v0),method=args.method)
        session.state['stages'].append(info)

        def evaluate(v):
            nonlocal local_best,last_certificate
            session.check()
            key=np.asarray(v).tobytes()
            if key in cache:
                session.cache_hit(cache[key][2],context,v)
                return cache[key]
            if session.ledger.counts.get('joint_sdp',0)-start_count>=budget:
                raise CallLimit('local_budget')
            c=geometry_function(v)
            if c.x.min()<args.mass_floor:
                session.journal(dict(event='mass_floor_rejection',parameters=v,context=context))
                return 1.,np.zeros_like(v),None,None
            result,gradient,ident,valid=session.evaluate(c,cfg,context=context,parameters=v,
                geometry_function=geometry_function,gradient=args.method=='envelope')
            violation=result.diagnostics.get('independent_primal_check_inf',1.)
            merit=result.diagnostics.get('search_merit',result.reported_objective) if cfg.mode=='margin' else result.mu0
            value=(-merit+10*max(0.,violation-cfg.residual_tolerance)) if valid else 1.
            derivative=-gradient if gradient is not None else np.zeros_like(v)
            cache[key]=(value,derivative,ident,result)
            prior=(local_best[0].diagnostics.get('search_merit',local_best[0].reported_objective) if cfg.mode=='margin' else local_best[0].mu0) if local_best else -np.inf
            if valid and merit>prior:
                local_best=result,v.copy(),ident
            floor=float(session.state['best_certified_mu'] or args.baseline)
            if valid and result.mu0>floor+1e-10 and result.diagnostics.get('full_gap',-1.)>1e-13 and perf_counter()-last_certificate>.25:
                session.certify(result,ident,minimum=floor)
                last_certificate=perf_counter()
            return cache[key]

        try:
            if not len(v0):
                evaluate(v0)
            else:
                if args.method=='envelope':
                    optimum=minimize(lambda v:evaluate(v)[0],v0,jac=lambda v:evaluate(v)[1],
                        method='SLSQP',bounds=[(-bound,bound)]*len(v0),options=dict(maxiter=300,ftol=1e-11))
                elif args.method=='SLSQP':
                    optimum=minimize(lambda v:evaluate(v)[0],v0,method='SLSQP',
                        bounds=[(-bound,bound)]*len(v0),options=dict(maxiter=300,eps=1e-5,ftol=1e-11))
                elif args.method=='Powell':
                    optimum=minimize(lambda v:evaluate(v)[0],v0,method='Powell',
                        bounds=[(-bound,bound)]*len(v0),options=dict(maxiter=300,xtol=1e-8,ftol=1e-11))
                else:
                    step=np.full(len(v0),.015)
                    if context.get('local_scale'):
                        step[:]=context['local_scale']
                    simplex=np.tile(v0,(len(v0)+1,1))
                    for i in range(len(v0)):
                        simplex[i+1,i]+=step[i] if v0[i]+step[i]<=bound else -step[i]
                    optimum=minimize(lambda v:evaluate(v)[0],v0,method='Nelder-Mead',
                        bounds=[(-bound,bound)]*len(v0),options=dict(initial_simplex=simplex,
                            maxfev=3*budget,maxiter=3*budget,xatol=1e-9,fatol=1e-12,adaptive=True))
                info.update(optimizer_message=str(optimum.message),optimizer_success=bool(optimum.success))
        except CallLimit as exc:
            if str(exc)!='local_budget':
                raise
            info['stop_reason']='local_budget'
        if local_best:
            info.update(best_mu=local_best[0].mu0,best_id=local_best[2],
                        best_margin=local_best[0].diagnostics.get('full_gap'))
            floor=float(session.state['best_certified_mu'] or args.baseline)
            if local_best[0].mu0>floor:
                session.certify(local_best[0],local_best[2],minimum=floor)
        info['actual_calls']=session.ledger.counts.get('joint_sdp',0)-start_count
        session.persist()
        print(json.dumps(dict(stage=context,calls=info['actual_calls'],mu=info.get('best_mu'),
                              total=session.ledger.counts.get('joint_sdp',0))),flush=True)
        return local_best

    reason='completed'
    try:
        if args.mode=='scan':
            partitions=list(c4_partitions(args.max_per_parent,args.max_size))
            for repeat in range(args.restarts):
                rng.shuffle(partitions)
                for counts in partitions:
                    family=MirrorFamily(counts,parent_symmetry=not args.no_parent_symmetry,
                                        mass_by_occupancy=args.mass_by_occupancy)
                    v=family.initial(rng,.35)
                    v[:family.mass_dim]=0. # Actual C4 root masses, before optimization.
                    local(family.geometry,v,dict(operation='fresh_C4_multisplit',counts=counts,repeat=repeat))
        elif args.mode=='refine':
            sources=[]
            for path in (args.input/'strata').glob('*.json'):
                raw=json.loads(path.read_text())
                if raw.get('candidate') and raw.get('mu0') is not None:
                    sources.append((raw['mu0'],path,result_from_data(raw)))
            sources.sort(key=lambda x:x[0],reverse=True)
            if not sources:
                parser.error('Need a C4 scan directory containing strata/*.json')
            for repeat in range(args.restarts):
                for _,path,origin in sources[:args.top]:
                    counts=tuple(int(k) for k in np.bincount(origin.candidate.groups,minlength=4))
                    if not origin.candidate.family_id.startswith('c4/'):
                        raise ValueError('Non-C4 scan input rejected')
                    family=MirrorFamily(counts,parent_symmetry=not args.no_parent_symmetry,
                                        mass_by_occupancy=args.mass_by_occupancy)
                    v=family.encode(target_start(origin.candidate))
                    if repeat:
                        v+=rng.normal(0,.02,len(v))
                    local(family.geometry,v,dict(operation='C4_stratum_refinement',source=str(path),
                        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),repeat=repeat),bound=7.)
        elif args.mode=='polish':
            counts=tuple(int(k) for k in np.bincount(source.candidate.groups,minlength=4))
            family=MirrorFamily(counts,parent_symmetry=not args.no_parent_symmetry,
                                mass_by_occupancy=args.mass_by_occupancy)
            v=family.encode(source.candidate)
            for repeat in range(args.restarts):
                start=v.copy() if repeat==0 else v+rng.normal(0,.002,len(v))
                winner=local(family.geometry,start,dict(operation='C4_continuation',source=str(args.input),
                    repeat=repeat,local_scale=.001 if repeat==0 else .005))
                if winner is not None:
                    v=winner[1]
        elif args.mode=='random':
            from kuramoto_template.asymptotic_search.c4_discovery import generic_c4_split
            sizes=[m for m in (7,8,9,10,12,14,16) if m<=args.max_size]
            for repeat in range(args.restarts):
                c=generic_c4_split(sizes[repeat%len(sizes)],rng)
                v=generic_parameters(c)
                local(lambda values,base=c:generic_geometry(base,values),v,
                    dict(operation='fresh_unequal_C4_split',size=c.m,repeat=repeat),bound=7.)
        else:
            for repeat in range(args.restarts):
                c=source.candidate
                if args.mode in ('birth','needle'):
                    births=[]
                    for depth in range(args.birth_depth):
                        index=int(rng.integers(c.m))
                        if args.mode=='needle':
                            epsilon=float(rng.choice([1e-3,1e-5,1e-7,1e-9]))
                            ratio=1-epsilon
                            separation=float(rng.choice([.1,1.,3.]))/np.sqrt(epsilon)
                        else:
                            ratio=float(rng.uniform(.2,.8))
                            separation=float(rng.choice([0.,.005,.03,.15,.5]))
                        c=split_candidate(c,index,ratio,separation,min_mass=args.mass_floor)
                        births.append(dict(index=index,ratio=ratio,separation=separation))
                    context=dict(operation='C4_birth',source=str(args.input),repeat=repeat,
                                 births=births,needle=args.mode=='needle')
                else:
                    context=dict(operation='C4_release_reflection',source=str(args.input),repeat=repeat)
                v=generic_parameters(c)
                if args.mode=='needle':
                    scales=np.maximum(1.,np.abs(c.s))
                    v[c.m-1:]=0.
                    local(lambda values,base=c,scale=scales:generic_geometry(base,
                        np.r_[values[:base.m-1],base.s+scale*values[base.m-1:]]),v,context,bound=30.)
                    continue
                if repeat and args.mode=='free':
                    v+=rng.normal(0,.01,len(v))
                local(lambda values,base=c:generic_geometry(base,values),v,context,bound=7.)
    except (DeadlineReached,CallLimit) as exc:
        reason=str(exc)
    except KeyboardInterrupt:
        reason='interrupted'
    except Exception as exc:
        reason=f'failed:{type(exc).__name__}:{exc}'
        raise
    finally:
        session.finish(reason)


if __name__=='__main__':
    main()
