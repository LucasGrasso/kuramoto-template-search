"""Balanced outer search with weak-mode proposals and atomic replay boundaries.

The earlier two-stage driver is unchanged. Full evaluations use its existing
SearchSession adapter. New conic and audit costs have disjoint ledger kinds.
"""

import argparse
from dataclasses import asdict,dataclass,field,replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter,sleep

import numpy as np

from kuramoto_template.experiments.reporting import json_safe
from .audit import high_precision_audit
from .families import normalize_offsets,s7,split_candidate
from .finite_lift import lift_candidate
from .joint_limit import JointLimitConfig,result_from_data,solve_joint_limit
from .models import LimitingCandidate,candidate_data
from .search import _code_hash,_full_evaluation,_write
from .solvers import CallLimit,SolverLedger
from .stage2 import validate_correction
from .weak_modes import (apply_topology_move,continuous_move,high_precision_limit_audit,
                         rank_topology_moves)

FAMILIES = ("complete","anchor_zero","cut_zero","mixed_zero","internal_sparse","nonuniform_anchor")
JOINT_KINDS = ("joint_sdp","finite_tree_linear","full_density_sdp","full_margin_sdp","high_precision_audit")


def _write_with_retry(path,data):
    """Windows readers may briefly deny delete-sharing during atomic replace.

    Retry only the same file write, never the completed solver invocation. An
    enduring permission error still propagates rather than dropping a checkpoint.
    """
    for attempt in range(6):
        try:
            _write(path,data)
            return
        except PermissionError:
            if attempt==5:
                raise
            sleep(.01*(attempt+1))


@dataclass(frozen=True)
class JointOuterConfig:
    proposals: int = 100
    seed: int = 0
    sizes: tuple = (7,8,9,10,12)
    families: tuple = FAMILIES
    guard_scales: tuple = (.01,.003,.001)
    targets: tuple = (.6930,.6935,.6950)
    joint_sdp_calls: int = 200
    finite_tree_calls: int = 60
    full_sdp_calls: int = 0
    audit_calls: int = 10
    shortlist_size: int = 10
    archive_per_stratum: int = 4
    mass_floor: float = 1e-5
    mass_radius: float = .008
    offset_radius: float = .015
    guided_probability: float = .6
    h_fractions: tuple = (.3,.1,.03)
    max_wall_seconds: float | None = None
    inner: JointLimitConfig = field(default_factory=lambda:JointLimitConfig(secondary=False))

    def __post_init__(self):
        for name in ("proposals","seed","joint_sdp_calls","finite_tree_calls","full_sdp_calls",
                     "audit_calls","shortlist_size","archive_per_stratum"):
            v = getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int) or v<0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.archive_per_stratum<1:
            raise ValueError("archive_per_stratum must be positive")
        if not self.sizes or any(m not in (7,8,9,10,12) for m in self.sizes):
            raise ValueError("Choose sizes from 7/8/9/10/12")
        if not self.families or any(f not in FAMILIES for f in self.families):
            raise ValueError("Unknown structural family")
        for name in ("sizes","families","guard_scales","targets","h_fractions"):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        for name in ("guard_scales","targets","h_fractions"):
            if not getattr(self,name) or any(not np.isfinite(v) or not 0<v<1 for v in getattr(self,name)):
                raise ValueError(f"{name} must contain finite values in (0,1)")
        if not 0<=self.guided_probability<=1 or not 0<self.mass_floor<.05:
            raise ValueError("Invalid guided probability or mass floor")
        if self.mass_radius<=0 or self.offset_radius<=0:
            raise ValueError("Move radii must be positive")
        if self.max_wall_seconds is not None and (not np.isfinite(self.max_wall_seconds) or self.max_wall_seconds<=0):
            raise ValueError("Wall limit must be positive")


def repair_tree(candidate):
    """A deterministic neighbor-only spanning tree, without relying on P values."""
    c = candidate
    pairs = [(a,b) for a in range(c.m) for b in range(a+1,c.m)
             if c.allowed_mask[a,b] and (c.groups[a]-c.groups[b])%2==1]
    protected = {tuple(sorted((a,b))) for a,b in zip(c.backbone_cycle,c.backbone_cycle[1:]+c.backbone_cycle[:1])}
    pairs.sort(key=lambda pair:(pair not in protected,pair))
    parents = list(range(c.m))

    def root(a):
        while parents[a]!=a:
            a=parents[a]
        return a

    chosen=[]
    for a,b in pairs:
        ra,rb=root(a),root(b)
        if ra!=rb:
            parents[ra]=rb
            chosen.append((a,b))
    return tuple(chosen) if len(chosen)==c.m-1 else ()


def make_outer_seed(size,family,rng,*,min_mass=1e-5):
    if family not in FAMILIES or size not in (7,8,9,10,12):
        raise ValueError("Unknown size/family")
    c=s7(float(rng.choice([.03,.01,.003,0.,-.003])))
    while c.m<size:
        eligible=[a for a in range(c.m) if np.sum(c.groups==c.groups[a])<4]
        a=int(rng.choice(eligible))
        c=split_candidate(c,a,float(rng.uniform(.2,.8)),float(rng.uniform(.02,.3)),min_mass=min_mass)
    # Larger moves are reserved for fresh restarts; guided moves have their own
    # recorded trust radii. All crossed and anchor densities remain independent.
    if family!="complete":
        c=continuous_move(c,rng,mass_radius=.01,offset_radius=.02,min_mass=min_mass)
    else:
        c=replace(c,s=normalize_offsets(c.x,c.s))
    mask,p=c.allowed_mask.copy(),c.p0.copy()
    protected=set(repair_tree(c)) | {tuple(sorted((a,b))) for a,b in zip(
        c.backbone_cycle,c.backbone_cycle[1:]+c.backbone_cycle[:1])}
    eligible=[(a,b) for a in range(c.m) for b in range(a+1,c.m)
              if mask[a,b] and c.groups[a]!=c.groups[b] and (a,b) not in protected]
    if family=="anchor_zero":
        eligible=[e for e in eligible if 3 in c.groups[list(e)]]
    elif family=="cut_zero":
        eligible=[e for e in eligible if set(c.groups[list(e)])=={0,1}]
    elif family=="internal_sparse":
        for j in range(4):
            vertices=np.flatnonzero(c.groups==j)
            for i,a in enumerate(vertices):
                for b in vertices[i+2:]:
                    mask[a,b]=mask[b,a]=False
    if family in ("anchor_zero","cut_zero","mixed_zero","internal_sparse") and eligible:
        count=min(len(eligible),1+int(rng.integers(1 if family!="mixed_zero" else 3)))
        for index in rng.choice(len(eligible),size=count,replace=False):
            a,b=eligible[int(index)]
            mask[a,b]=mask[b,a]=False
    p[~mask]=0
    c=replace(c,p0=p,allowed_mask=mask,family_id=family)
    fixed=np.full_like(p,np.nan)
    fixed[~mask]=0
    np.fill_diagonal(fixed,1)
    if family=="nonuniform_anchor":
        for a,b in [(a,b) for a,b in eligible if 3 in c.groups[[a,b]]]:
            fixed[a,b]=fixed[b,a]=float(rng.uniform(.88,.99))
    return c,fixed


def canonical_outer(candidate,fixed):
    """Exact sorting within parent labels, after the explicit outer scale gauge.

    No quantization is used. Tied labels may miss a symmetry cache hit; they
    cannot merge unequal matrices. The solver always receives this stored order.
    """
    c=candidate
    order=np.array(sorted(range(c.m),key=lambda a:(c.groups[a],c.s[a],c.x[a])))
    inverse=np.argsort(order)
    offsets=normalize_offsets(c.x[order],c.s[order])
    out=LimitingCandidate(c.x[order],offsets,c.groups[order],c.p0[np.ix_(order,order)],
                         c.allowed_mask[np.ix_(order,order)],tuple(inverse[list(c.backbone_cycle)]),c.family_id)
    return out,fixed[np.ix_(order,order)],order


def exact_outer_key(c,fixed,tree,inner):
    digest=hashlib.sha256()
    for value in (c.x,c.s,c.groups,c.allowed_mask,fixed):
        digest.update(np.asarray(value).tobytes())
    digest.update(json.dumps(json_safe(dict(backbone=c.backbone_cycle,tree=tree,config=asdict(inner))),sort_keys=True).encode())
    return digest.hexdigest()


@dataclass
class JointOuterResult:
    state: dict
    solutions: dict
    output: Path | None

    @property
    def call_counts(self):
        return self.state["counts"]

    @property
    def best_limit_mu(self):
        return self.state["best_limit_mu"]

    @property
    def best_finite_mu(self):
        return self.state["best_finite"]["mu"] if self.state["best_finite"] else None


def run_joint_outer(config=None,*,output=None,resume=False,cancelled=None,candidates=None):
    cfg=config or JointOuterConfig()
    destination=Path(output) if output is not None else None
    identity=dict(config=json_safe(asdict(cfg)),source_hash=_code_hash(),versions={
        name:importlib.metadata.version(name) for name in ("numpy","scipy","cvxpy","clarabel")})
    if resume:
        if destination is None or candidates is not None:
            raise ValueError("Resume requires a directory and saved proposals")
        state=json.loads((destination/"checkpoint.json").read_text(encoding="utf-8"))
        if identity!=state["identity"]:
            raise ValueError("Incompatible checkpoint configuration, source, or versions")
    else:
        if destination is not None:
            destination.mkdir(parents=True,exist_ok=False)
            (destination/"joint-results").mkdir()
        state=dict(schema_version=1,identity=identity,phase="outer",cursor=0,pending=None,
                   records=[],trace=[],counts={kind:0 for kind in JOINT_KINDS},solver_events=[],
                   archives=dict(strict=[],neutral=[],unstable=[],uncertain=[],failed=[],rigorously_certified=[]),
                   parents={},cache={},cache_hits=0,rng=np.random.default_rng(cfg.seed).bit_generator.state,
                   best_limit_id=None,best_limit_mu=None,best_finite=None,shortlist=[],final_cursor=0,
                   finite=[],audits=[],elapsed_seconds=0.,stop_reason="active",proof_available=False,
                   custom=[candidate_data(c) for c in candidates] if candidates is not None else None)
    solutions={}
    rng=np.random.default_rng()
    rng.bit_generator.state=state["rng"]
    limits=dict(joint_sdp=cfg.joint_sdp_calls,finite_tree_linear=cfg.finite_tree_calls,
                full_density_sdp=cfg.full_sdp_calls,high_precision_audit=cfg.audit_calls)
    ledger=SolverLedger(limits,state["counts"],state["solver_events"])
    started,previous=perf_counter(),state["elapsed_seconds"]

    def persist():
        state["elapsed_seconds"]=previous+perf_counter()-started
        state["rng"]=rng.bit_generator.state
        if destination is not None:
            _write_with_retry(destination/"checkpoint.json",state)

    def event(kind,parent=None,**data):
        record=json_safe(dict(id=len(state["trace"]),kind=kind,parent_id=parent,**data))
        state["trace"].append(record)
        return record["id"]

    def get_solution(index):
        if index not in solutions:
            data=json.loads((destination/"joint-results"/f"{index:06d}.json").read_text(encoding="utf-8"))
            solutions[index]=result_from_data(data)
        return solutions[index]

    def stop():
        if cancelled is not None and cancelled():
            state["stop_reason"]="cancelled"
            return True
        if cfg.max_wall_seconds is not None and previous+perf_counter()-started>=cfg.max_wall_seconds:
            state["stop_reason"]="wall_time"
            return True
        return False

    def register(result,pending):
        index=len(state["records"])
        solutions[index]=result
        if destination is not None:
            _write_with_retry(destination/"joint-results"/f"{index:06d}.json",result.data())
        record=dict(id=index,size=pending["size"],family=pending["family"],mode=result.config.mode,
                    mu_target=result.config.mu_target,h_guard=result.config.h_guard,status=result.status,
                    solver_status=result.solver_status,mu0=result.mu0,full_gap=result.diagnostics.get("full_gap"),
                    mask_hash=result.candidate.mask_hash if result.candidate else None,
                    trace_id=event("joint_solve",pending["trace_id"],status=result.status,mu0=result.mu0),
                    file=f"joint-results/{index:06d}.json")
        state["records"].append(record)
        kind=("strict" if result.status=="joint_asymptotic_strict" else "neutral" if result.status=="joint_neutral"
              else "unstable" if result.status=="joint_unstable"
              else "uncertain" if result.status=="uncertain" else "failed")
        state["archives"][kind].append(index)
        if kind=="strict" and (state["best_limit_mu"] is None or result.mu0>state["best_limit_mu"]):
            state["best_limit_mu"],state["best_limit_id"]=result.mu0,index
        if kind in ("strict","neutral","unstable") and result.candidate is not None:
            key=f"{pending['size']}:{pending['family']}"
            pool=state["parents"].setdefault(key,[])
            pool.append(index)
            # Same size, normalization, and family only. Never rank gaps across m.
            pool.sort(key=lambda i:(state["records"][i]["status"]=="joint_asymptotic_strict",
                                    state["records"][i]["mu0"] or 0,
                                    state["records"][i]["full_gap"] or -1e30),reverse=True)
            del pool[cfg.archive_per_stratum:]
        return index

    persist()
    try:
        while state["phase"]=="outer" and state["cursor"]<cfg.proposals:
            if stop():
                break
            i=state["cursor"]
            if state["pending"] is None:
                size=cfg.sizes[i%len(cfg.sizes)]
                family=cfg.families[(i//len(cfg.sizes))%len(cfg.families)]
                round_index=i//(len(cfg.sizes)*len(cfg.families))
                inner=replace(cfg.inner,h_guard=cfg.guard_scales[round_index%len(cfg.guard_scales)],
                              mode="margin" if round_index%4==3 else "density",
                              mu_target=cfg.targets[(round_index//4)%len(cfg.targets)])
                parent,move=None,dict(operation="restart")
                if state["custom"] is not None:
                    if i>=len(state["custom"]):
                        raise ValueError("Insufficient custom proposals")
                    c=LimitingCandidate(**state["custom"][i])
                    size,family=c.m,"custom"
                    fixed=np.full_like(c.p0,np.nan)
                else:
                    pool=state["parents"].get(f"{size}:{family}",[])
                    if pool and rng.random()<cfg.guided_probability:
                        parent=int(rng.choice(pool))
                        previous_result=get_solution(parent)
                        c,fixed=previous_result.candidate,previous_result.fixed_p0.copy()
                        if rng.random()<.5:
                            moves=rank_topology_moves(c,previous_result.r,previous_result.hlim)
                            chosen=moves[int(rng.integers(min(4,len(moves))))]
                            c=apply_topology_move(c,chosen)
                            a,b=chosen.edge
                            fixed[a,b]=fixed[b,a]=0. if chosen.operation=="close" else np.nan
                            move=asdict(chosen)
                            move["topology_operation"]=move.pop("operation")
                            move["operation"]="weak_topology"
                        else:
                            c=continuous_move(c,rng,mass_radius=cfg.mass_radius,
                                              offset_radius=cfg.offset_radius,min_mass=cfg.mass_floor)
                            move=dict(operation="mass_logits_offsets",mass_radius=cfg.mass_radius,offset_radius=cfg.offset_radius)
                    else:
                        c,fixed=make_outer_seed(size,family,rng,min_mass=cfg.mass_floor)
                c,fixed,order=canonical_outer(c,fixed)
                tree=repair_tree(c)
                key=exact_outer_key(c,fixed,tree,inner)
                ident=event("outer_proposal",state["records"][parent]["trace_id"] if parent is not None else None,
                            proposal=i,size=size,family=family,move=move,input_order=order)
                state["pending"]=dict(candidate=candidate_data(c),fixed_p0=fixed,tree=tree,
                                      config=asdict(inner),key=key,trace_id=ident,size=size,family=family)
                persist()
                continue  # A cancellable, durable OUTER-PROPOSAL boundary.
            pending=state["pending"]
            if pending["key"] in state["cache"]:
                state["cache_hits"]+=1
                event("exact_cache_hit",pending["trace_id"],source=state["cache"][pending["key"]])
            else:
                if ledger.remaining("joint_sdp")<=0:
                    event("joint_budget_skip",pending["trace_id"])
                    state["stop_reason"],state["phase"],state["pending"]="joint_budget","select",None
                    break
                c=LimitingCandidate(**pending["candidate"])
                result=solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,JointLimitConfig(**pending["config"]),
                    backbone_cycle=c.backbone_cycle,repair_edges=pending["tree"],fixed_p0=pending["fixed_p0"],
                    family_id=pending["family"],ledger=ledger)
                index=register(result,pending)
                state["cache"][pending["key"]]=index
            state["pending"]=None
            state["cursor"]+=1
            persist()  # JOINT-SOLVE boundary; completed solves are never repaid.
        if state["phase"]=="outer" and state["cursor"]==cfg.proposals:
            state["phase"]="select"
        if state["phase"]=="select":
            # Reuse the existing cluster-first/mask-second shortlist policy.
            from .search import _shortlist
            proxies=[]
            strict_ids=state["archives"]["strict"]
            for index in strict_ids:
                result=get_solution(index)
                proxies.append(dict(candidate=candidate_data(result.candidate),metrics=dict(
                    status="stage1_screened",mu0=result.mu0,dipole_slacks=(result.diagnostics["full_gap"],)*2)))
            state["shortlist"]=[strict_ids[i] for i in _shortlist(proxies,cfg.shortlist_size)]
            state["phase"]="finalize"
            persist()
        while state["phase"]=="finalize" and state["final_cursor"]<len(state["shortlist"]):
            if stop():
                break
            index=state["shortlist"][state["final_cursor"]]
            result=get_solution(index)
            if state["pending"] is None:
                if ledger.remaining("high_precision_audit")>(1 if cfg.finite_tree_calls else 0):
                    audit=ledger.invoke("high_precision_audit",lambda:high_precision_limit_audit(result))
                    state["audits"].append(dict(source=index,kind="limiting",data=audit))
                    event("high_precision_limit_audit",state["records"][index]["trace_id"],positive_pivots=audit["positive_pivots"])
                state["pending"]=dict(source=index,h_cursor=0)
                persist()
                continue
            h_cursor=state["pending"]["h_cursor"]
            if h_cursor==len(cfg.h_fractions):
                state["pending"]=None
                state["final_cursor"]+=1
                persist()
                continue
            if ledger.remaining("finite_tree_linear")<=0:
                state["stop_reason"]="finite_tree_budget"
                break
            h=result.config.h_guard*cfg.h_fractions[h_cursor]
            numerics=result.config.lift_numerics()
            correction=validate_correction(result.candidate,result.r,numerics=numerics,solver_status=result.solver_status)
            finite=lift_candidate(correction,h,numerics=numerics,ledger=ledger)
            data=asdict(finite)
            data.update(source=index,trace_id=event("finite_lift",state["records"][index]["trace_id"],h=h,status=finite.status))
            if finite.status=="finite_h_numeric_strict":
                if state["best_finite"] is None or finite.mu>state["best_finite"]["mu"]:
                    state["best_finite"]=data
            elif ledger.remaining("full_density_sdp")>0:
                evaluation,session,settings=_full_evaluation(result.candidate,h,ledger.remaining("full_density_sdp"),
                                                              numerics,cfg.seed,ledger)
                data["support_only_fallback"]=dict(family_lift=False,sdp_calls=session.sdp_calls,
                    status=evaluation.status,mu=evaluation.stable_mu,P=evaluation.P,evaluation_config=asdict(settings))
            state["finite"].append(data)
            state["pending"]["h_cursor"]+=1
            persist()  # FINITE-LIFT boundary.
        if state["phase"]=="finalize" and state["final_cursor"]==len(state["shortlist"]):
            state["phase"]="done"
            if state["stop_reason"] in ("active","cancelled","interrupted","wall_time","finite_tree_budget"):
                state["stop_reason"]="completed"
        if state["phase"]=="done" and state["best_finite"] is not None and ledger.remaining("high_precision_audit")>0:
            best=state["best_finite"]
            if not any(a["kind"]=="finite" and a["source"]==best["source"] and a.get("h")==best["h"] for a in state["audits"]):
                audit=ledger.invoke("high_precision_audit",lambda:high_precision_audit(
                    best["candidate"]["x"],best["phi"],best["P"]))
                state["audits"].append(dict(source=best["source"],kind="finite",h=best["h"],data=audit))
    except CallLimit as exc:
        state["stop_reason"]=f"{exc}_budget"
    except KeyboardInterrupt:
        state["stop_reason"]="interrupted"
    finally:
        persist()
    if destination is not None:
        _write_with_retry(destination/"result.json",state)
        if state["best_limit_id"] is not None:
            _write_with_retry(destination/"top-limit.json",get_solution(state["best_limit_id"]).data())
        if state["best_finite"] is not None:
            _write_with_retry(destination/"top-finite.json",state["best_finite"])
        with (destination/"trace.jsonl").open("w",encoding="utf-8") as handle:
            for item in state["trace"]:
                handle.write(json.dumps(json_safe(item),allow_nan=False)+"\n")
    return JointOuterResult(state,solutions,destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--config")
    parser.add_argument("--proposals",type=int,default=100)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.config:
        data=json.loads(Path(args.config).read_text(encoding="utf-8"))
    elif args.resume:
        data=json.loads((Path(args.output)/"checkpoint.json").read_text(encoding="utf-8"))["identity"]["config"]
    else:
        data=dict(proposals=args.proposals,seed=args.seed,joint_sdp_calls=2*args.proposals)
    if "inner" in data:
        data["inner"]=JointLimitConfig(**data["inner"])
    result=run_joint_outer(JointOuterConfig(**data),output=args.output,resume=args.resume)
    print(json.dumps(dict(best_limit_mu=result.best_limit_mu,best_finite_mu=result.best_finite_mu,
                          calls=result.call_counts,stop_reason=result.state["stop_reason"])))


if __name__=="__main__":
    main()
