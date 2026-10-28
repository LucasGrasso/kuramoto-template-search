"""Bounded aggregate-screened research orchestration using the existing solvers.

The pilot allocates proposals, not fictitious equal costs: primary joint SDPs,
compensation QPs, analytic projections, tree solves and audits remain separate.
"""

import argparse
from collections import Counter
from dataclasses import asdict,dataclass,field,replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from kuramoto_template.experiments.reporting import json_safe
from .aggregate_flow import screen_aggregate,uniform_block_rule,CONCRETE_PAPER_EXAMPLE
from .audit import high_precision_audit
from .compensated import compensated_move,dipole_slacks
from .expansions import full_rescaled_matrix
from .families import s7,appendix_a_correction,normalize_offsets
from .finite_lift import lift_candidate
from .flow_families import flow_balanced_seed,winding_two_seed,merge_to_seven,split_physical,s7_distance
from .joint_limit import JointLimitConfig,result_from_data,solve_joint_limit,validate_joint
from .joint_outer import _write_with_retry,repair_tree
from .models import LimitingCandidate,MU_RECORD,candidate_data
from .physical import physical_spectrum
from .search import _code_hash,_shortlist
from .solvers import SolverLedger,CallLimit
from .stage2 import validate_correction
from .topology import shortest_nonzero_cycle
from .weak_modes import high_precision_limit_audit


@dataclass(frozen=True)
class FlowSearchConfig:
    independent: int=300
    balanced_updates: int=300
    split_merge: int=150
    compensated: int=150
    seed: int=0
    primary_calls: int=900
    secondary_calls: int=60
    tree_calls: int=90
    lift_attempts: int=90
    audit_calls: int=12
    saved_audit_count: int=30
    shortlist_size: int=15
    target: float=MU_RECORD
    parent_root: str="runs/joint-research-final/sweep"
    saved_inputs: str="runs/joint-research-final/paired"
    h_values: tuple=(.003,.001,.0003,.0001,.00003,.00001)
    max_wall_seconds: float|None=None
    inner: JointLimitConfig=field(default_factory=lambda:JointLimitConfig(
        secondary=False,margin_metric="mass",gamma0=1e-5,hard_floor=1e-4))

    def __post_init__(self):
        for name in ("independent","balanced_updates","split_merge","compensated","seed","primary_calls",
                     "secondary_calls","tree_calls","lift_attempts","audit_calls","saved_audit_count","shortlist_size"):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int) or v<0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.inner.secondary:
            raise ValueError("This bounded pilot uses primary-only joint solves; secondary budget belongs to paired compensation QPs")
        if self.inner.margin_metric!="mass":
            raise ValueError("Split/merge comparisons require the explicit mass metric")
        if not 0<self.target<=1 or not self.h_values or any(not np.isfinite(h) or h<=0 for h in self.h_values):
            raise ValueError("Invalid target or h grid")
        if self.max_wall_seconds is not None and (not np.isfinite(self.max_wall_seconds) or self.max_wall_seconds<=0):
            raise ValueError("Invalid wall limit")
        object.__setattr__(self,"h_values",tuple(self.h_values))


def _fallback_parent():
    c=s7(.003)
    return validate_joint(c,appendix_a_correction(c),fixed_p0=c.p0)


def load_parents(root):
    path=Path(root)
    selected=[]
    if (path/"result.json").exists():
        state=json.loads((path/"result.json").read_text(encoding="utf-8"))
        strict=[r for r in state["records"] if r["status"]=="joint_asymptotic_strict"]
        strata={}
        for record in sorted(strict,key=lambda r:r["mu0"],reverse=True):
            key=(record["size"],record["family"])
            if key not in strata:
                strata[key]=record["id"]
        ids=list(dict.fromkeys(([state["best_limit_id"]] if state.get("best_limit_id") is not None else [])+
                               list(strata.values())+[991]))
        for index in ids:
            file=path/"joint-results"/f"{index:06d}.json"
            if file.exists():
                result=result_from_data(json.loads(file.read_text(encoding="utf-8")))
                if result.candidate is not None and result.r is not None:
                    selected.append((f"historical-{index}",result))
    return selected or [("analytic-S7",_fallback_parent())]


def make_plan(cfg):
    rng=np.random.default_rng(cfg.seed)
    plan=[]
    for i in range(cfg.independent):
        family=("independent","q_mass_window","winding_two")[i%3]
        size=(7,8,9,10,12)[(i//3)%5] if family!="winding_two" else (8,9,10,12)[(i//3)%4]
        plan.append(dict(kind="independent",family=family,size=size,seed=int(rng.integers(2**63)),case=i))
    for kind,count in (("balanced",cfg.balanced_updates),("split_merge",cfg.split_merge),("compensated",cfg.compensated)):
        for i in range((count+1)//2):
            seed=int(rng.integers(2**63))
            for role in range(2):
                if 2*i+role<count:
                    plan.append(dict(kind=kind,pair=f"{kind}-{i}",pair_index=i,role=role,seed=seed))
    return plan


def _balanced_candidate(parent,rng,policy):
    c=parent.candidate
    epsilon=float(np.exp(rng.uniform(np.log(1e-4),np.log(1e-3))))
    before,_=dipole_slacks(c)
    candidates=[]
    for index in rng.choice(c.m,size=min(c.m,6),replace=False):
        for sign in (-1,1):
            for kind in ("mass","offset"):
                x,s=c.x.copy(),c.s.copy()
                if kind=="mass":
                    logits=np.log(x); logits[index]+=sign*epsilon
                    x=np.exp(logits-logits.max()); x/=x.sum()
                else:
                    s[index]+=sign*epsilon
                s=normalize_offsets(x,s)
                candidate=replace(c,x=x,s=s)
                slacks,residual=dipole_slacks(candidate)
                screen=screen_aggregate(x,c.groups,MU_RECORD,allowed_mask=c.allowed_mask)
                score=(screen.status!="aggregate_rejected",float(np.min(slacks-before)),screen.cap)
                candidates.append((score,candidate,dict(coordinate=int(index),sign=sign,coordinate_kind=kind,
                    step=epsilon,proxy_dipoles_before=before,proxy_dipoles_after=slacks,
                    proxy_range_residual=residual,proxy_is_feasibility_check=False)))
    choice=max(candidates,key=lambda value:value[0]) if policy=="balanced" else candidates[int(rng.integers(len(candidates)))]
    return choice[1],choice[2]


def prepare_proposal(descriptor,parents,cfg,ledger):
    rng=np.random.default_rng(descriptor["seed"])
    kind=descriptor["kind"]
    metadata=dict(q_mass_ratio=None,q_cross_density=None,flow_target=None,operation=kind,
                  pair=descriptor.get("pair"),role=descriptor.get("role"),requires_joint_reoptimization=True)
    if kind=="independent":
        family,size=descriptor["family"],descriptor["size"]
        case=descriptor["case"]//3
        if family=="winding_two":
            c,extra=winding_two_seed(size,rng,relaxed_chords=bool((case//4)%2))
        else:
            from .aggregate_flow import q_mass_critical
            q=1.005+(q_mass_critical()-1.01)*((case%20)+.5)/20 if family=="q_mass_window" else None
            fraction=(.85,.92,.97,.995,.9995)[(case//5)%5]
            c,extra=flow_balanced_seed(size,rng,q_mass_ratio=q,flow_fraction=fraction)
            if family=="independent":
                extra["flow_target"]=None  # General restarts let the joint SDP select flow.
        metadata.update(extra,family=family)
    else:
        pool=parents
        if kind=="balanced":
            near=[p for p in parents if screen_aggregate(p[1].candidate.x,p[1].candidate.groups,cfg.target).cap>=cfg.target-.001]
            pool=near or parents
        elif kind=="compensated":
            eligible=[p for p in parents if screen_aggregate(p[1].candidate.x,p[1].candidate.groups,cfg.target).status!="aggregate_rejected"]
            pool=eligible or parents
        index=descriptor["pair_index"]%len(pool)
        name,parent=pool[index]
        c=parent.candidate
        metadata.update(parent=name,parent_mu0=parent.mu0,parent_status=parent.status,
                        parent_physical_gap=float(physical_spectrum(c,parent.hlim).min()),
                        parent_reoptimized=asdict(parent.config)==asdict(cfg.inner),fresh_comparison=True)
        if kind=="balanced":
            policy="balanced" if descriptor["role"]==0 else "random"
            c,extra=_balanced_candidate(parent,rng,policy)
            metadata.update(extra,policy=policy,family="local_asymmetric")
        elif kind=="split_merge":
            target=(8,9,10,12)[descriptor["pair_index"]%4]
            if c.m>7:
                c,_=merge_to_seven(c)
            while c.m<target:
                available=[a for a in range(c.m) if np.sum(c.groups==c.groups[a])<4]
                a=int(rng.choice(available))
                c=split_physical(c,a,float(rng.uniform(.25,.75)),float(rng.uniform(1e-4,1e-3)))
            metadata.update(split_size=c.m,physical_offset_units="inherited; no rescaling of h",
                            source_winding=parent.candidate.backbone_winding)
            if descriptor["role"]==1:
                c,merges=merge_to_seven(c)
                metadata["merge_steps"]=merges
            metadata.update(policy="split" if descriptor["role"]==0 else "fresh_merge_7",family="split_merge")
            metadata["comparison_only"]=True
        else:
            policy="balanced" if descriptor["role"]==0 else "random"
            method="qp" if descriptor["pair_index"]<min(30,cfg.secondary_calls//2) else "projection"
            move=compensated_move(parent,rng,policy=policy,method=method,ledger=ledger)
            metadata.update(compensation=asdict(move),policy=policy,method=method,family="compensated_topology")
            # Save the raw direction and diagnostics without duplicating candidate arrays.
            metadata["compensation"].pop("candidate",None)
            if move.status!="proposal":
                return None,dict(metadata,proposal_status=move.status)
            c=move.candidate
    c=replace(c,family_id=metadata["family"])
    metadata.update(initial_distance_s7=s7_distance(c),backbone_winding=c.backbone_winding,
                    minimum_cycle_allowed=shortest_nonzero_cycle(c.groups,c.allowed_mask),
                    source_offset_variance=float(c.x@(c.s-c.x@c.s)**2))
    return c,metadata


def _input_key(c,metadata,inner):
    h=hashlib.sha256()
    for value in (c.x,c.s,c.groups,c.allowed_mask):
        h.update(value.tobytes())
    h.update(json.dumps(json_safe(dict(config=asdict(inner),cycle=c.backbone_cycle,
                flow_target=metadata.get("flow_target"))),sort_keys=True).encode())
    return h.hexdigest()


@dataclass
class FlowSearchResult:
    state: dict
    output: Path|None


def run_flow_search(config=None,*,output=None,resume=False,cancelled=None,parents=None):
    cfg=config or FlowSearchConfig()
    destination=Path(output) if output is not None else None
    identity=dict(config=json_safe(asdict(cfg)),source_hash=_code_hash(),versions={
        name:importlib.metadata.version(name) for name in ("numpy","scipy","cvxpy","clarabel")})
    if resume:
        if destination is None or parents is not None:
            raise ValueError("Resume uses a saved directory and saved parents")
        state=json.loads((destination/"checkpoint.json").read_text(encoding="utf-8"))
        if state["identity"]!=identity:
            raise ValueError("Incompatible checkpoint settings/source/dependencies")
        parents=[(p["name"],result_from_data(json.loads((destination/p["result_file"]).read_text(encoding="utf-8"))
                     if "result_file" in p else p["result"])) for p in state["parents"]]
    else:
        if destination is not None:
            destination.mkdir(parents=True,exist_ok=False)
            (destination/"solutions").mkdir()
        parents=parents or load_parents(cfg.parent_root)
        files=sorted(Path(cfg.saved_inputs).glob("*/input.json"))[:cfg.saved_audit_count]
        saved=[dict(path=str(f),data=json.loads(f.read_text(encoding="utf-8"))) for f in files]
        state=dict(schema_version=1,identity=identity,phase="saved_audit",cursor=0,pending=None,
            parents=[dict(name=n,result=r.data()) for n,r in parents],saved=saved,plan=make_plan(cfg),
            records=[],trace=[],cache={},cache_hits=0,screened=[],filter_comparison=[],outcomes=[],
            counts={k:0 for k in ("joint_sdp","compensation_qp","compensation_projection","finite_tree_linear","high_precision_audit")},
            solver_events=[],best_limit_id=None,best_limit_mu=None,best_finite=None,
            archives=dict(strict=[],uncertain=[],failed=[],screened=[],rigorously_certified=[]),parent_baselines=[],
            shortlist=[],final_cursor=0,finite=[],audits=[],elapsed_seconds=0.,stop_reason="active",proof_available=False)
        if destination is not None:
            (destination/"parent-inputs").mkdir()
            for i,(name,parent) in enumerate(parents):
                file=f"parent-inputs/{i:03d}.json"
                _write_with_retry(destination/file,parent.data())
                state["parents"][i]=dict(name=name,result_file=file)
    limits=dict(joint_sdp=cfg.primary_calls,compensation_qp=cfg.secondary_calls,
                compensation_projection=cfg.compensated,finite_tree_linear=cfg.tree_calls,high_precision_audit=cfg.audit_calls)
    ledger=SolverLedger(limits,state["counts"],state["solver_events"])
    memory={}
    started,previous=perf_counter(),state["elapsed_seconds"]

    def persist():
        state["elapsed_seconds"]=previous+perf_counter()-started
        if destination is not None:
            _write_with_retry(destination/"checkpoint.json",state)

    def event(kind,**data):
        entry=json_safe(dict(id=len(state["trace"]),kind=kind,**data))
        entry.setdefault("q_mass_ratio",entry.get("metadata",{}).get("q_mass_ratio"))
        entry.setdefault("q_cross_density",entry.get("metadata",{}).get("q_cross_density"))
        if isinstance(entry.get("metadata"),dict):
            entry["metadata"].pop("proposal_elapsed_seconds",None)
        state["trace"].append(entry)
        return entry["id"]

    def stop():
        if cancelled is not None and cancelled():
            state["stop_reason"]="cancelled"; return True
        if cfg.max_wall_seconds is not None and previous+perf_counter()-started>=cfg.max_wall_seconds:
            state["stop_reason"]="wall_time"; return True
        return False

    def solution(index):
        if index not in memory:
            memory[index]=result_from_data(json.loads((destination/state["records"][index]["file"]).read_text(encoding="utf-8")))
        return memory[index]

    def register(result,metadata,source_id):
        index=len(state["records"])
        q=metadata.get("q_mass_ratio"); crossed=metadata.get("q_cross_density")
        tag=lambda v:"na" if v is None else f"{v:.8g}".replace(".","p")
        name=f"solutions/{index:06d}-qmass-{tag(q)}-qcross-{tag(crossed)}-wind-{metadata.get('backbone_winding',1)}.json"
        if result.candidate is not None:
            try:
                result.diagnostics["dipole_slacks"],result.diagnostics["dipole_range_residual"]=dipole_slacks(result.candidate)
            except ValueError as exc:
                result.diagnostics["dipole_slacks"]=None
                result.diagnostics["dipole_audit_error"]=str(exc)
            if result.status=="joint_asymptotic_strict" and (result.diagnostics["dipole_slacks"] is None or min(result.diagnostics["dipole_slacks"]) < -cfg.inner.residual_tolerance):
                result.status,result.message="failed","Independent dipole test contradicts strict classification"
            result.diagnostics["distance_s7"]=s7_distance(result.candidate)
            result.diagnostics["minimum_cycle_actual"]=shortest_nonzero_cycle(result.candidate.groups,result.candidate.p0>0)
            result.diagnostics["r_mass_norm"]=float(np.sqrt(np.sum(np.outer(result.candidate.x,result.candidate.x)*result.r**2)))
            result.diagnostics["degree_first_order"]=result.r@result.candidate.x
        memory[index]=result
        if destination is not None:
            _write_with_retry(destination/name,result.data())
        row=dict(id=index,file=name,status=result.status,mu0=result.mu0,metadata=metadata,source_trace_id=source_id,
                 size=result.candidate.m if result.candidate else metadata.get("size"),
                 physical_gap=result.diagnostics.get("physical_gap"),r_norm=result.diagnostics.get("r_norm"),
                 r_mass_norm=result.diagnostics.get("r_mass_norm"),distance_s7=result.diagnostics.get("distance_s7"))
        state["records"].append(json_safe(row))
        bucket="strict" if result.status=="joint_asymptotic_strict" else "uncertain" if result.status=="uncertain" else "failed"
        state["archives"][bucket].append(index)
        if bucket=="strict" and (state["best_limit_mu"] is None or result.mu0>state["best_limit_mu"]):
            state["best_limit_mu"],state["best_limit_id"]=result.mu0,index
        event("joint_result",source_trace_id=source_id,status=result.status,mu0=result.mu0,
              q_mass_ratio=q,q_cross_density=crossed,best_limit_mu=state["best_limit_mu"])
        return index

    persist()
    try:
        while state["phase"]=="saved_audit" and state["cursor"]<len(state["saved"]):
            if stop(): break
            if ledger.remaining("joint_sdp")<=0:
                state["stop_reason"]="primary_budget"; break
            item=state["saved"][state["cursor"]]
            c=LimitingCandidate(**item["data"]["candidate"])
            begin=perf_counter(); screen=screen_aggregate(c.x,c.groups,cfg.target,allowed_mask=c.allowed_mask,tolerance=cfg.inner.residual_tolerance)
            screening_seconds=perf_counter()-begin
            inner=replace(cfg.inner,mode="margin",mu_target=cfg.target)
            result=solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,inner,backbone_cycle=c.backbone_cycle,
                fixed_p0=item["data"].get("fixed_p0"),ledger=ledger)
            comparison=dict(path=item["path"],screen=screen.data(),
                screening_seconds=screening_seconds,unfiltered_status=result.status,
                unfiltered_seconds=result.elapsed_seconds,actual_unfiltered_primary_calls=1,
                counterfactual_filtered_primary_calls=0 if screen.status=="aggregate_rejected" else 1,
                full_input=item["data"],result=result.data())
            if destination is not None:
                (destination/"saved-filter-audit").mkdir(exist_ok=True)
                file=f"saved-filter-audit/{state['cursor']:03d}.json"
                _write_with_retry(destination/file,comparison)
                comparison={k:v for k,v in comparison.items() if k not in ("full_input","result")}
                comparison["file"]=file
            state["filter_comparison"].append(comparison)
            if screen.status=="aggregate_rejected" and result.status=="joint_asymptotic_strict":
                raise RuntimeError("Aggregate screen contradicted a strict target-density solution")
            state["cursor"]+=1; persist()
        if state["phase"]=="saved_audit" and state["cursor"]==len(state["saved"]):
            state["phase"],state["cursor"]="parent_calibration",0; persist()
        while state["phase"]=="parent_calibration" and state["cursor"]<len(parents):
            if stop(): break
            if ledger.remaining("joint_sdp")<=0:
                state["stop_reason"]="primary_budget"; break
            i=state["cursor"]; name,old=parents[i]; c=old.candidate
            result=solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,cfg.inner,backbone_cycle=c.backbone_cycle,
                repair_edges=repair_tree(c),family_id="parent_baseline",ledger=ledger)
            metadata=dict(family="parent_baseline",size=c.m,parent=name,historical_mu0=old.mu0,
                q_mass_ratio=None,q_cross_density=None,flow_target=None,backbone_winding=c.backbone_winding,
                initial_distance_s7=s7_distance(c),operation="fresh_parent_baseline")
            ident=event("parent_baseline",metadata=metadata)
            index=register(result,metadata,ident)
            state["parent_baselines"].append(index)
            if result.status=="joint_asymptotic_strict":
                parents[i]=(name,result)
                state["parents"][i]=(dict(name=name,result_file=state["records"][index]["file"]) if destination is not None
                                        else dict(name=name,result=result.data()))
            state["cursor"]+=1; persist()
        if state["phase"]=="parent_calibration" and state["cursor"]==len(parents):
            state["phase"],state["cursor"]="main",0; persist()
        while state["phase"]=="main" and state["cursor"]<len(state["plan"]):
            if stop(): break
            if state["pending"] is None:
                descriptor=state["plan"][state["cursor"]]
                before_qp=ledger.counts.get("compensation_qp",0)
                proposal_started=perf_counter()
                c,metadata=prepare_proposal(descriptor,parents,cfg,ledger)
                metadata.update(descriptor=descriptor,size=c.m if c else None)
                metadata["secondary_conic_calls"]=ledger.counts.get("compensation_qp",0)-before_qp
                metadata["proposal_elapsed_seconds"]=perf_counter()-proposal_started
                trace_id=event("proposal",metadata=metadata,q_mass_ratio=metadata.get("q_mass_ratio"),q_cross_density=metadata.get("q_cross_density"))
                if c is None:
                    state["screened"].append(dict(status="proposal_rejected",metadata=metadata,source_trace_id=trace_id))
                    state["outcomes"].append(dict(metadata=metadata,status="proposal_rejected",primary_calls=0,mu0=None))
                    state["cursor"]+=1; persist(); continue
                state["pending"]=dict(candidate=candidate_data(c),metadata=metadata,trace_id=trace_id)
                if destination is not None:
                    (destination/"proposals").mkdir(exist_ok=True)
                    qtag="na" if metadata.get("q_mass_ratio") is None else str(round(metadata["q_mass_ratio"],8)).replace(".","p")
                    _write_with_retry(destination/"proposals"/f"{state['cursor']:06d}-qmass-{qtag}-qcross-na-wind-{c.backbone_winding}.json",dict(
                        candidate=candidate_data(c),metadata=metadata,config=asdict(cfg.inner),flow_target=metadata.get("flow_target")))
                persist(); continue
            pending=state["pending"]; c=LimitingCandidate(**pending["candidate"]); metadata=pending["metadata"]
            screen=screen_aggregate(c.x,c.groups,cfg.target,allowed_mask=c.allowed_mask,
                                    flow_target=metadata.get("flow_target"),tolerance=cfg.inner.residual_tolerance)
            metadata["aggregate_screen"]=screen.data()
            metadata["uniform_analytic_rule"]=uniform_block_rule(c,metadata=metadata)
            if screen.status=="aggregate_rejected" and not metadata.get("comparison_only",False):
                state["screened"].append(dict(screen=screen.data(),metadata=metadata,source_trace_id=pending["trace_id"],
                                              candidate=candidate_data(c),status=screen.status))
                state["archives"]["screened"].append(len(state["screened"])-1)
                state["outcomes"].append(dict(metadata=metadata,status=screen.status,screen=screen.data(),primary_calls=0,mu0=None))
            else:
                key=_input_key(c,metadata,cfg.inner)
                if key in state["cache"] and not metadata.get("fresh_comparison",False):
                    state["cache_hits"]+=1
                    cached=solution(state["cache"][key])
                    event("exact_cache_hit",source_trace_id=pending["trace_id"],record=state["cache"][key],
                          q_mass_ratio=metadata.get("q_mass_ratio"),q_cross_density=metadata.get("q_cross_density"))
                    state["outcomes"].append(dict(metadata=metadata,status=cached.status,cache_hit=True,
                        record=state["cache"][key],primary_calls=0,mu0=cached.mu0,
                        realized_degree_gain=cached.mu0-metadata["parent_mu0"] if cached.mu0 is not None and metadata.get("parent_mu0") is not None else None))
                else:
                    if ledger.remaining("joint_sdp")<=0:
                        state["stop_reason"],state["phase"]="primary_budget","select"; state["pending"]=None; break
                    result=solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,cfg.inner,backbone_cycle=c.backbone_cycle,
                        repair_edges=repair_tree(c),flow_target=metadata.get("flow_target"),family_id=metadata["family"],ledger=ledger)
                    metadata["realized_degree_gain"]=result.mu0-metadata["parent_mu0"] if result.mu0 is not None and metadata.get("parent_mu0") is not None else None
                    index=register(result,metadata,pending["trace_id"])
                    state["cache"][key]=index
                    state["outcomes"].append(dict(metadata=metadata,status=result.status,cache_hit=False,
                        record=index,primary_calls=1,mu0=result.mu0,realized_degree_gain=metadata.get("realized_degree_gain")))
            state["pending"]=None; state["cursor"]+=1; persist()
        if state["phase"]=="main" and state["cursor"]==len(state["plan"]):
            state["phase"]="select"
        if state["phase"]=="select":
            strict=state["archives"]["strict"]
            proxies=[dict(candidate=candidate_data(solution(i).candidate),metrics=dict(status="stage1_screened",
                mu0=solution(i).mu0,dipole_slacks=solution(i).diagnostics["dipole_slacks"])) for i in strict]
            selected=[strict[i] for i in _shortlist(proxies,cfg.shortlist_size)]
            for family in ("q_mass_window","winding_two"):
                members=[i for i in strict if state["records"][i]["metadata"]["family"]==family]
                if members:
                    best=max(members,key=lambda i:state["records"][i]["mu0"])
                    if best not in selected:
                        selected=([best]+selected)[:cfg.shortlist_size]
            state["shortlist"],state["phase"]=selected,"finalize"
            state["pending"]=None; persist()
        while state["phase"]=="finalize" and state["final_cursor"]<len(state["shortlist"]):
            if stop(): break
            index=state["shortlist"][state["final_cursor"]]; result=solution(index)
            if state["pending"] is None:
                if ledger.remaining("high_precision_audit")>1:
                    data=ledger.invoke("high_precision_audit",lambda:high_precision_limit_audit(result))
                    state["audits"].append(dict(source=index,kind="limiting",data=data))
                state["pending"]=dict(source=index,h_cursor=0); persist(); continue
            hs=[h for h in cfg.h_values if h<=result.config.h_guard]
            cursor=state["pending"]["h_cursor"]
            if cursor>=len(hs) or len(state["finite"])>=cfg.lift_attempts:
                state["pending"]=None; state["final_cursor"]+=1; persist(); continue
            if ledger.remaining("finite_tree_linear")<=0:
                state["stop_reason"]="tree_budget"; break
            correction=validate_correction(result.candidate,result.r,numerics=result.config.lift_numerics(),solver_status=result.solver_status)
            finite=lift_candidate(correction,hs[cursor],numerics=result.config.lift_numerics(),ledger=ledger)
            data=asdict(finite); data["source"]=index
            state["finite"].append(data)
            event("finite_lift",source=index,h=hs[cursor],status=finite.status,
                  q_mass_ratio=state["records"][index]["metadata"].get("q_mass_ratio"),q_cross_density=None)
            if finite.status=="finite_h_numeric_strict" and (state["best_finite"] is None or finite.mu>state["best_finite"]["mu"]):
                state["best_finite"]=data
            state["pending"]["h_cursor"]+=1; persist()
        if state["phase"]=="finalize" and state["final_cursor"]==len(state["shortlist"]):
            state["phase"]="done"
            if state["stop_reason"] in ("active","cancelled","interrupted","wall_time","tree_budget"):
                state["stop_reason"]="completed"
        if state["phase"]=="done" and state["best_finite"] and ledger.remaining("high_precision_audit")>0:
            if not any(a["kind"]=="finite" for a in state["audits"]):
                best=state["best_finite"]
                data=ledger.invoke("high_precision_audit",lambda:high_precision_audit(best["candidate"]["x"],best["phi"],best["P"]))
                state["audits"].append(dict(source=best["source"],kind="finite",h=best["h"],data=data))
    except KeyboardInterrupt:
        state["stop_reason"]="interrupted"
    except CallLimit as exc:
        state["stop_reason"]=str(exc)+"_budget"
    finally:
        persist()
    # Keep evidence-rich Pareto rows separate from the protected density winner.
    pareto_rows=[]
    for index in state["archives"]["strict"]:
        result=solution(index); finite=[v for v in state["finite"] if v["source"]==index]
        stable=[v for v in finite if v["status"]=="finite_h_numeric_strict"]
        pareto_rows.append(dict(source=index,size=result.candidate.m,family=state["records"][index]["metadata"]["family"],
            mu_limit=result.mu0,positive_limiting_margin=result.diagnostics["physical_gap"],
            r_norm=result.diagnostics["r_norm"],r_mass_norm=result.diagnostics["r_mass_norm"],
            first_degree_slope_min=float(np.min(result.diagnostics["degree_first_order"])),
            viable_h=max((v["h"] for v in stable),default=None),mu_finite=max((v["mu"] for v in stable),default=None),
            finite_attempted=bool(finite)))
    nondominated=[]
    for row in pareto_rows:
        def score(r):
            base=(r["mu_limit"],r["positive_limiting_margin"],-r["r_mass_norm"],r["first_degree_slope_min"])
            return base+((r["viable_h"] or 0.,r["mu_finite"] or -1.) if r["finite_attempted"] else ())
        rivals=[r for r in pareto_rows if (r["size"],r["family"],r["finite_attempted"])==(row["size"],row["family"],row["finite_attempted"])]
        if not any(all(a>=b for a,b in zip(score(r),score(row))) and any(a>b for a,b in zip(score(r),score(row))) for r in rivals):
            nondominated.append(row)
    state["pareto"]=nondominated
    persist()
    if destination is not None:
        _write_with_retry(destination/"result.json",state)
        if state["best_limit_id"] is not None:
            _write_with_retry(destination/"top-limit.json",solution(state["best_limit_id"]).data())
        if state["best_finite"]:
            _write_with_retry(destination/"top-finite.json",state["best_finite"])
        with (destination/"trace.jsonl").open("w",encoding="utf-8") as handle:
            for entry in state["trace"]:
                handle.write(json.dumps(json_safe(entry),allow_nan=False)+"\n")
    completed=FlowSearchResult(state,destination)
    from .flow_report import write_flow_report
    write_flow_report(completed)
    return completed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True); parser.add_argument("--config")
    parser.add_argument("--resume",action="store_true"); parser.add_argument("--seed",type=int,default=0)
    args=parser.parse_args()
    data=json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else (
        json.loads((Path(args.output)/"checkpoint.json").read_text(encoding="utf-8"))["identity"]["config"] if args.resume else dict(seed=args.seed))
    if "inner" in data: data["inner"]=JointLimitConfig(**data["inner"])
    result=run_flow_search(FlowSearchConfig(**data),output=args.output,resume=args.resume)
    print(json.dumps(dict(best_limit_mu=result.state["best_limit_mu"],best_finite_mu=result.state["best_finite"]["mu"] if result.state["best_finite"] else None,
                          counts=result.state["counts"],stop_reason=result.state["stop_reason"])))


if __name__=="__main__":
    main()
