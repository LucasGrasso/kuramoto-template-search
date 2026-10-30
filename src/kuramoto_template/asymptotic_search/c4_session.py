"""Durable, compressed ledgers for bounded C4-only research runs."""

from dataclasses import asdict
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from .c4_parametric import ParametricJoint
from .c4_recovered import RecoveredJoint
from .interval_certification import rational_limiting_repair, verify_limiting
from .joint_outer import _write_with_retry as write
from .joint_outer import repair_tree
from .persistence import _code_hash, json_safe
from .solvers import SolverLedger, CallLimit


class DeadlineReached(RuntimeError):
    pass


class DurableLedger(SolverLedger):
    def __init__(self, journal, max_calls):
        super().__init__(limits=dict(joint_sdp=max_calls))
        self.journal=journal

    def invoke(self,kind,callback):
        if self.remaining(kind)<=0:
            raise CallLimit(kind)
        self.journal(dict(event="call_started",kind=kind,number=self.counts.get(kind,0)+1))
        try:
            return super().invoke(kind,callback)
        finally:
            self.journal(dict(event="call_finished",data=self.events[-1]))


class C4Session:
    def __init__(self, output, *, seconds, max_calls, settings):
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        (self.output/"winners").mkdir()
        (self.output/"certificates").mkdir()
        self.handle=gzip.open(self.output/"events.jsonl.gz","wt",encoding="utf-8",compresslevel=4)
        self.started=perf_counter()
        self.deadline=self.started+seconds
        self.ledger=DurableLedger(self.journal,max_calls)
        self.engines={}
        self.engine_class=RecoveredJoint if settings.get('engine')=='recovered' else ParametricJoint
        self.state=dict(schema_version=1,source_hash=_code_hash(),settings=settings,
            seconds=seconds,max_calls=max_calls,s7_seed_used=False,counts=self.ledger.counts,
            versions={n:importlib.metadata.version(n) for n in ("numpy","scipy","cvxpy","clarabel","python-flint")},
            proposals=0,cache_hits=0,certificate_attempts=0,best_numeric_mu=None,
            best_certified_mu=None,best_certificate=None,status_counts={},stop_reason="active",stages=[])
        self.state['thread_environment']={name:os.environ.get(name) for name in
            ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
        root=Path(__file__).resolve().parents[3]
        snapshot=self.output/'source-snapshot'
        sources=list(Path(__file__).resolve().parents[1].rglob('*.py'))+[root/'scripts/push_c4_bound.py']
        for source in sources:
            target=snapshot/source.relative_to(root)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
        self.best=None
        self.best_margin_by_size={}
        self.best_by_partition={}
        self.best_certified=None
        self.journal(dict(event="identity",data=self.state))
        self.persist()

    def journal(self,data):
        self.handle.write(json.dumps(json_safe(data),allow_nan=False)+"\n")
        self.handle.flush()

    def persist(self):
        self.state["elapsed_seconds"]=perf_counter()-self.started
        write(self.output/"summary.json",self.state)

    def check(self):
        if perf_counter()>=self.deadline:
            raise DeadlineReached("wall_time")
        if self.ledger.remaining("joint_sdp")<=0:
            raise CallLimit("joint_sdp")

    def evaluate(self,c,config,*,context,parameters=None,geometry_function=None,gradient=False):
        self.check()
        if not c.family_id.startswith("c4/"):
            raise ValueError("This experiment accepts only explicitly C4-derived candidates")
        ident=self.state["proposals"]
        self.state["proposals"]+=1
        self.journal(dict(event="proposal",id=ident,context=context,parameters=parameters,
                          candidate=asdict(c),config=asdict(config)))
        key=(c.groups.tobytes(),c.allowed_mask.tobytes(),c.backbone_cycle,
             json.dumps(json_safe(asdict(config)),sort_keys=True))
        if key not in self.engines:
            if len(self.engines)>=3:
                self.engines.pop(next(iter(self.engines)))
            kwargs=dict(ledger=self.ledger)
            if self.engine_class is RecoveredJoint:kwargs['elastic_penalty']=self.state['settings'].get('target_penalty',0.)
            self.engines[key]=self.engine_class(c,config,**kwargs)
        engine=self.engines[key]
        result=engine.run(c)
        derivative=None
        if gradient and engine.last is not None and result.mu0 is not None:
            derivative=engine.gradient(geometry_function,parameters,step=self.state['settings'].get('gradient_step',1e-5))
        self.journal(dict(event="result",id=ident,result=result.data(),gradient=derivative))
        self.state["status_counts"][result.status]=self.state["status_counts"].get(result.status,0)+1
        # This is an exploration merit gate, not a strictness/proof status.
        valid=(result.candidate is not None and result.mu0 is not None and
               result.diagnostics.get("independent_primal_check_inf",np.inf)<1e-7 and
               (config.mode=='margin' or result.diagnostics.get("full_gap",-np.inf)>-1e-7))
        if valid and config.mode=='margin':
            selected='physical_gap' if config.margin_metric=='mass' else 'full_gap'
            valid=abs(result.reported_objective-result.diagnostics[selected])<1e-7
        if valid:
            partition=tuple(np.bincount(c.groups,minlength=4))
            entry=(result,ident,np.array(parameters) if parameters is not None else None,context)
            quality=result.diagnostics.get('search_merit',result.reported_objective) if config.mode=='margin' else result.mu0
            previous=self.best_by_partition.get(partition)
            prior=(previous[0].diagnostics.get('search_merit',previous[0].reported_objective) if config.mode=='margin' else previous[0].mu0) if previous else -np.inf
            if quality>prior:
                self.best_by_partition[partition]=entry
            if config.mode=='margin':
                if c.m not in self.best_margin_by_size or quality>self.best_margin_by_size[c.m]:
                    self.best_margin_by_size[c.m]=quality
                    self.state['best_margin_by_size']=self.best_margin_by_size.copy()
                    self.state.setdefault('target_shortfall_by_size',{})[c.m]=result.diagnostics.get('target_shortfall',0.)
                    write(self.output/f'top-margin-m{c.m}.json',result.data())
            # Global density reporting uses only positive candidates in margin
            # mode. Raw margins are never ranked across different dimensions.
            positive=(config.mode!='margin' or result.diagnostics.get('full_gap',-1.)>0)
            prior=self.best[0].mu0 if self.best else -np.inf
            if positive and result.mu0>prior:
                self.best=entry
                self.state["best_numeric_mu"]=result.mu0
                self.state["best_numeric_status"]=result.status
                write(self.output/"top-search.json",result.data())
        if ident%25==0:
            self.persist()
        return result,derivative,ident,valid

    def cache_hit(self,source,context,parameters):
        self.state["cache_hits"]+=1
        self.journal(dict(event="exact_cache_hit",source_id=source,context=context,parameters=parameters))

    def certify(self,result,ident,*,minimum=0.):
        if result.candidate is None or result.mu0 is None or result.mu0<minimum:
            return False
        self.state["certificate_attempts"]+=1
        saved=json_safe(result.data())
        data=None
        try:
            data,changes=rational_limiting_repair(saved)
            if not data['repair_tree']:
                data['repair_tree']=repair_tree(result.candidate)
                changes['lifting_tree_selected_after_optimization']=True
            proof=verify_limiting(data,precision_bits=256)
            lower=proof["density"]["lower"]
            self.journal(dict(event="certificate_success",source_id=ident,lower=lower,
                              changes=changes,exact_data=data,result=proof))
            if self.best_certified is None or float(lower)>float(self.best_certified["result"]["density"]["lower"]):
                source=self.output/"winners"/f"{ident:07d}.json"
                write(source,saved)
                root=Path(__file__).resolve().parents[3]
                certificate=dict(schema_version=1,name="c4-research",kind="limiting",precision_bits=256,
                    arithmetic="python-flint / FLINT-Arb",source_file=source.resolve().relative_to(root).as_posix(),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_reported_mu=result.mu0,
                    exact_data=data,changes=changes,result=proof,graph_realization_certified=False,
                    original_snapshot_attempt=dict(certified=False,reason="Explicit rational variant only"))
                write(self.output/"certificates"/f"{ident:07d}.json",certificate)
                write(self.output/"best-certificate.json",certificate)
                write(self.output/"best-certified-source.json",saved)
                self.best_certified=certificate
                self.state["best_certified_mu"]=lower
                self.state["best_certificate"]=f"certificates/{ident:07d}.json"
                print(json.dumps(dict(certified_lower=lower,source_id=ident,
                                      calls=self.ledger.counts.get("joint_sdp",0))),flush=True)
            self.persist()
            return True
        except Exception as exc:
            self.journal(dict(event="certificate_failure",source_id=ident,failure=f"{type(exc).__name__}: {exc}",
                              exact_data=data))
            self.persist()
            return False

    def finish(self,reason="completed"):
        self.state["stop_reason"]=reason
        self.state["strata"]=[dict(partition=list(p),mu0=e[0].mu0,id=e[1],status=e[0].status,
                                  margin=e[0].diagnostics.get('full_gap'))
                               for p,e in sorted(self.best_by_partition.items())]
        (self.output/'strata').mkdir(exist_ok=True)
        for partition,entry in self.best_by_partition.items():
            write(self.output/'strata'/('-'.join(map(str,partition))+'.json'),entry[0].data())
        self.persist()
        self.handle.close()
