"""Reconstruct branch actions for imitation; reconcile serial episodes for TD.

Historical context synthesized for BC is explicitly not a temporal transition.
No SDP calls are made by the importer. Cached legacy matrices are not trusted.
"""
from collections import Counter
from dataclasses import dataclass,field,fields,asdict
import hashlib
import json
from pathlib import Path
import numpy as np

from kuramoto_template.template import PhaseTemplate
from kuramoto_template.environment.env import build_observation,STATUSES
from kuramoto_template.environment.config import EnvironmentConfig
from kuramoto_template.environment.actions import decode_action,action_masks
from kuramoto_template.experiments.cache import canonical_key
from kuramoto_template.search.evaluator import EvaluationConfig,TemplateEvaluation,SolverDiagnostics
from kuramoto_template.sdp_optimizer.template_metrics import template_metrics
from kuramoto_template.sdp_optimizer.validate_input import validate_input,in_open_semicircle
from .replay import Transition


def key(template):
    return repr(canonical_key(template,decimals=6))


def template(data):
    return PhaseTemplate(data["x"],data["phi"])


def same_template(a,b):
    return (a.n_clusters==b.n_clusters and np.allclose(a.x,b.x,atol=2e-12,rtol=0)
        and np.max(np.abs(np.angle(np.exp(1j*(a.phi-b.phi)))))<=2e-7)


def rescreen(t,data,cfg):
    e=TemplateEvaluation(t,"uncertain")
    try:validate_input(t.x,t.phi)
    except ValueError:return TemplateEvaluation(t,"invalid")
    if not data:return e
    if data.get("cached"):return e
    for source in ("density","margin"):
        d=data.get(source)
        if not d or d.get("P") is None:continue
        P=np.asarray(d["P"],dtype=float)
        if P.shape!=(t.n_clusters,t.n_clusters) or not np.isfinite(P).all():continue
        metrics=template_metrics(t.x,t.phi,P)
        diag=SolverDiagnostics(source,d.get("status","unknown"),mu=metrics.mu,
            max_force_error=metrics.max_force_error,min_restoring=metrics.min_restoring,
            min_cluster_eigenvalue=metrics.min_cluster_eigenvalue)
        setattr(e,source,diag)
        tol=cfg.residual_tolerance
        common=(diag.status in ("optimal","optimal_inaccurate") and d.get("solver_agrees") is True
            and np.max(abs(P-P.T))<=tol and P.min()>=-tol and P.max()<=1+tol
            and (not cfg.force_cliques or np.max(abs(np.diag(P)-1))<=tol)
            and metrics.max_force_error<=tol)
        if source=="density":
            valid=(common and d.get("numerically_valid") is True
                and metrics.min_restoring>=cfg.internal_margin-min(tol,.05*cfg.internal_margin)
                and metrics.min_cluster_eigenvalue>=cfg.cluster_margin-min(tol,.05*cfg.cluster_margin))
            stable=(valid and data.get("status")=="stable" and d.get("stable") is True
                and not in_open_semicircle(t.phi) and metrics.min_restoring>cfg.stability_tolerance
                and metrics.min_cluster_eigenvalue>cfg.stability_tolerance)
            if stable:e.status="stable";e.stable_mu=metrics.mu
        else:
            gamma=d.get("gamma")
            valid=(common and d.get("numerically_valid") is True and gamma is not None and np.isfinite(gamma)
                and metrics.mu>=cfg.mu_target-tol and metrics.min_restoring>=gamma-tol
                and metrics.min_cluster_eigenvalue>=gamma-tol)
            if valid:
                diag.gamma=float(gamma);e.margin_gamma=float(gamma)
                if e.status!="stable":e.status="near_feasible"
        diag.numerically_valid=bool(valid);diag.stable=e.status=="stable" and source=="density"
        diag.solver_agrees=bool(common)
    if e.status=="uncertain" and data.get("status") in ("infeasible","solver_error","invalid"):
        if not any(d and d.get("P") is not None for d in (data.get("density"),data.get("margin"))):
            e.status=data["status"]
    return e


def reconstruct(parent,record,cfg):
    """Map a physical mutation into the exact existing normalized Gym action."""
    op=record["operation"];p=record.get("parameters",{})
    action=dict(operation=0,i=0,j=0,parameters=np.zeros(2,dtype=np.float32))
    if op=="split":
        ratio,sep=float(p["ratio"]),float(p["separation"])
        if abs(sep)<cfg.separation_min or abs(sep)>cfg.separation_max*(1+1e-10):raise ValueError("separation outside decoder support")
        if cfg.separation_min==cfg.separation_max:q=1 if sep>=0 else -1
        else:q=np.sign(sep)*np.log(abs(sep)/cfg.separation_min)/np.log(cfg.separation_max/cfg.separation_min)
        action.update(operation=0,i=int(p["index"]),parameters=np.array([2*(ratio-.2)/.6-1,q],dtype=np.float32))
    elif op in ("merge","prune_merge"):
        i,j=sorted((int(p["i"]),int(p["j"])))
        action.update(operation=1,i=i,j=j)
    elif op=="perturb":
        action.update(operation=2,i=int(p["index"]),parameters=np.array([p["delta"]/cfg.phase_scale,0],dtype=np.float32))
    elif op=="resize":
        i,j=int(p["source"]),int(p["target"])
        fraction=float(p["amount"])/(parent.x[i]*cfg.resize_fraction)
        action.update(operation=3,i=i,j=j,parameters=np.array([2*(fraction-.01)/.99-1,0],dtype=np.float32))
    else:raise ValueError("operation not represented by one Gym action")
    if (np.abs(action["parameters"])>1+1e-6).any():raise ValueError("parameters outside decoder support")
    # Roundoff at an old physical-domain endpoint is encoded at that endpoint.
    action["parameters"]=np.clip(action["parameters"],-1,1)
    child=decode_action(action,parent,cfg).template
    if child is None or not same_template(child,template(record["template"])):
        raise ValueError("mutation replay mismatch")
    return action


@dataclass
class ImitationSample:
    observation: dict
    action: dict
    weight: float
    run_id: str
    family: str
    parent_key: str
    child_key: str


@dataclass
class HistoricalDataset:
    actions: list=field(default_factory=list)
    validation: list=field(default_factory=list)
    transitions: list=field(default_factory=list)
    states: list=field(default_factory=list)
    manifest: dict=field(default_factory=dict)


def _config(cls,data):
    return cls(**{f.name:data[f.name] for f in fields(cls) if f.name in data and data[f.name] is not None})


def serial_transitions(root,rows,meta,cfg,ecfg):
    """Reject the entire episode unless full state, cost and boundaries agree."""
    path=root/'transitions.jsonl';events=root/'events.jsonl'
    if meta.get("search_config",{}).get("algorithm")!="gymnasium_serial" or not path.exists() or not events.exists():return []
    if meta.get('evaluator_schema_version')!=2:return []
    tx=[json.loads(line) for line in path.read_text().splitlines()]
    ev=[json.loads(line) for line in events.read_text().splitlines()]
    if not ev or ev[0].get("event")!="reset" or ev[-1].get("event")!="episode_end":return []
    if not tx:return []
    seed=rows[0];remaining=cfg.max_sdp_calls-int(seed["evaluation"]["sdp_calls"])
    if any(x.get('evaluation',{}).get('cached') for x in rows.values() if x.get('evaluation')):return []
    initial=rescreen(template(seed["template"]),seed["evaluation"],ecfg)
    best=initial.stable_mu or 0.;current=seed["id"];zero=0;result=[]
    for step,r in enumerate(tx):
        required=("source_id","record_id","next_state_id","best_mu_before","best_mu_after",
                  "remaining_calls_before","remaining_calls_after","sdp_calls","reward","terminated","truncated","action")
        if any(k not in r for k in required):return []
        if r["source_id"]!=current or r["remaining_calls_before"]!=remaining or abs(r["best_mu_before"]-best)>1e-7:return []
        source=rows[current];child=rows[r["next_state_id"]];attempt=rows[r["record_id"]]
        if attempt.get("parent_id")!=current:return []
        actual_cost=(attempt.get('evaluation') or {}).get('sdp_calls',0)
        if r['sdp_calls']!=actual_cost:return []
        decoded=decode_action(r["action"],template(source["template"]),cfg).template
        if decoded is None:
            if r["sdp_calls"]!=0 or r["next_state_id"]!=current or r['remaining_calls_after']!=remaining or r['best_mu_after']!=best:return []
            zero+=1
            continue  # actor masks exclude structural rejects; never impute an action
        if not same_template(decoded,template(child["template"])):return []
        before=rescreen(template(source["template"]),source.get("evaluation"),ecfg)
        after=rescreen(template(child["template"]),child.get("evaluation"),ecfg)
        next_best=max(best,after.stable_mu or 0.);cost=int(r["sdp_calls"])
        if cost not in (0,1,2) or r["remaining_calls_after"]!=remaining-cost:return []
        if abs(r["best_mu_after"]-next_best)>1e-7 or abs(r["reward"]-cfg.reward_scale*(next_best-best))>1e-5:return []
        o=build_observation(before.template,before,cfg,best_mu=best,remaining_sdp_calls=remaining,steps=step,zero_cost_steps=zero)
        zero=zero+1 if cost==0 else 0;remaining-=cost
        no=build_observation(after.template,after,cfg,best_mu=next_best,remaining_sdp_calls=remaining,steps=step+1,zero_cost_steps=zero)
        if r["terminated"]!=(remaining==0):return []
        action={**r["action"],"parameters":np.array(r["action"]["parameters"],dtype=np.float32)}
        result.append(Transition(o,action,float(r["reward"]),no,r["terminated"],r["truncated"],cost,meta["run_id"]))
        current=child["id"];best=next_best
    if result and not(result[-1].terminated or result[-1].truncated):result[-1].truncated=True
    if ev[-1].get("total_sdp_calls")!=cfg.max_sdp_calls-remaining:return []
    return result


def load_history(paths,*,train_families,held_out_families=(),seed=0,validation_fraction=.2,max_records=None,
                 forbidden_keys=(),margin_target=.693,action_config=None):
    result=HistoricalDataset();counts=Counter();candidates=[];td_by_run={};validation_keys=set();sources=[]
    validation_keys.update(forbidden_keys)
    trace_paths=set()
    for name in paths:
        p=Path(name)
        trace_paths.update([p] if p.name=='trace.jsonl' else p.rglob('trace.jsonl'))
    for path in sorted(trace_paths):
        if max_records is not None and counts['records']>=max_records:counts['record_cap_reached']+=1;break
        meta_path=path.parent/'metadata.json'
        if not meta_path.exists():counts['missing_metadata']+=1;continue
        meta=json.loads(meta_path.read_text());rows={}
        digest=hashlib.sha256()
        for line in path.open():
            digest.update(line.encode());r=json.loads(line);rows[r['id']]=r;counts['records']+=1
            if max_records is not None and counts['records']>=max_records:break
        result_path=path.parent/'result.json'
        source_cost=json.loads(result_path.read_text()).get('sdp_calls') if result_path.exists() else None
        sources.append(dict(path=str(path),sha256_prefix=digest.hexdigest(),records=len(rows),source_total_sdp_calls=source_cost))
        benchmark=path.parent/'benchmark.json'
        family=(json.loads(benchmark.read_text()).get('spec',{}).get('initial','unknown') if benchmark.exists()
                else meta.get('search_config',{}).get('reset_family','unknown'))
        provenance=path.parent/'policy_episode.json'
        if family=='unknown' and provenance.exists():family=json.loads(provenance.read_text()).get('family','unknown')
        run_id=meta['run_id']
        validation=(family in held_out_families or int(hashlib.sha256(f'{seed}:{run_id}'.encode()).hexdigest()[:8],16)/2**32<validation_fraction)
        allowed=family in train_families and family not in held_out_families
        if not allowed:counts['excluded_family_runs']+=1
        search_config=meta.get('search_config',{})
        # Legacy random-search phase_scale is a Gaussian sampling standard
        # deviation, NOT the Gym decoder's bounded displacement. Never copy
        # sampler settings into the current physical action decoder.
        cfg=action_config or EnvironmentConfig()
        serial_cfg=(_config(EnvironmentConfig,search_config) if search_config.get('algorithm')=='gymnasium_serial' else None)
        stored_ecfg=_config(EvaluationConfig,meta.get('evaluation_config',{}))
        ecfg=EvaluationConfig(max_clusters=stored_ecfg.max_clusters,mu_target=margin_target)
        if abs(stored_ecfg.mu_target-margin_target)>1e-12:
            counts['different_margin_target_runs']+=1;continue
        # Historical budget may be <3: it is not a supported live reset.
        for r in rows.values():
            if not r.get('template'):continue
            try:t=template(r['template']);e=rescreen(t,r.get('evaluation'),ecfg)
            except (ValueError,TypeError):counts['invalid_state']+=1;continue
            counts['states_retained']+=1
            result.states.append(dict(run_id=run_id,family=family,key=key(t),status=e.status,
                                      stable_mu=e.stable_mu,margin_gamma=e.margin_gamma,template=r['template']))
            if validation or not allowed:validation_keys.add(key(t))
            parent_id=r.get('parameters',{}).get('source_id',r.get('parameters',{}).get('split_id',r.get('parent_id')))
            parent=rows.get(parent_id)
            if parent is None or not parent.get('template'):continue
            try:
                pt=template(parent['template']);a=reconstruct(pt,r,cfg)
                pe=rescreen(pt,parent.get('evaluation'),ecfg)
            except (ValueError,KeyError,IndexError,TypeError):counts['action_excluded']+=1;continue
            if pe.status=='invalid':counts['invalid_parent']+=1;continue
            obs=build_observation(pt,pe,cfg,best_mu=pe.stable_mu or 0.,remaining_sdp_calls=cfg.max_sdp_calls)
            gain=max(0.,(e.stable_mu or 0.)-(pe.stable_mu or 0.))
            weight=1+min(4,1000*gain) if e.status=='stable' else .25
            sample=ImitationSample(obs,a,weight,run_id,family,key(pt),key(t))
            candidates.append((sample,validation,allowed))
        same_decoder=(serial_cfg is not None and all(getattr(serial_cfg,k)==getattr(cfg,k) for k in
            ('phase_scale','separation_min','separation_max','resize_fraction','min_clusters','reward_scale',
             'max_zero_cost_steps','max_wall_seconds')) and serial_cfg.max_clusters<=cfg.max_clusters
             and abs(serial_cfg.max_steps/serial_cfg.max_sdp_calls-cfg.max_steps/cfg.max_sdp_calls)<1e-12)
        try:td_by_run[run_id]=(serial_transitions(path.parent,rows,meta,serial_cfg,ecfg)
                              if same_decoder and asdict(stored_ecfg)==asdict(ecfg) else [])
        except (ValueError,KeyError,IndexError,TypeError):counts['serial_episode_rejected']+=1;td_by_run[run_id]=[]
    retained_runs=set()
    for sample,validation,allowed in candidates:
        if validation or not allowed:result.validation.append(sample)
        elif sample.parent_key in validation_keys or sample.child_key in validation_keys:counts['canonical_overlap_removed']+=1
        else:result.actions.append(sample);retained_runs.add(sample.run_id)
    train_keys={k for a in result.actions for k in (a.parent_key,a.child_key)}
    for run_id,transitions in td_by_run.items():
        if run_id in retained_runs:
            # TD contexts must belong to train-only episodes, not partial
            # branch records. Every state key is also screened for overlap.
            bad=any(s['key'] in validation_keys for s in result.states if s['run_id']==run_id)
            if not bad:result.transitions.extend(transitions)
    result.manifest=dict(counts=dict(counts),sources=sources,train_actions=len(result.actions),
        validation_actions=len(result.validation),clean_td_transitions=len(result.transitions),
        train_families=list(train_families),held_out_families=list(held_out_families),
        validation_fraction=validation_fraction,seed=seed,canonical_decimals=6,
        canonical_overlap=len(train_keys&validation_keys),train_keys=sorted(train_keys),
        heldout_keys=sorted(validation_keys),
        branch_context='synthetic imitation context, never TD',
        historical_source_sdp_calls=(sum(s['source_total_sdp_calls'] for s in sources) if all(s['source_total_sdp_calls'] is not None for s in sources) else None),
        historical_cost_scope='Whole referenced source runs, including excluded/capped records; not an amortized sample cost')
    return result
