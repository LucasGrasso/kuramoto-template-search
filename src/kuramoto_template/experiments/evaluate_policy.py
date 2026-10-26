"""Frozen held-out evaluation; reports training, inference and audit costs apart."""
import argparse
import hashlib
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

from kuramoto_template.environment import KuramotoEnv,EnvironmentConfig
from kuramoto_template.policy.train import initial,write_json
from kuramoto_template.policy.dataset import key
from kuramoto_template.search.policy_search import load_policy,run_policy_search
from kuramoto_template.search.evaluator import TemplateEvaluator,EvaluationConfig
from kuramoto_template.search.random_search import run_random_search,RandomSearchConfig,twisted_seed
from kuramoto_template.search.infrastructure import SearchSession
from .benchmark import derive_seed
from .metrics import compute_run_metrics,summarize_runs,paired_comparisons
from .reporting import write_benchmark_report
from .refinement import audit_candidate,RefinementConfig


def run_serial_random(t,*,seed,config,output,evaluator=None):
    with KuramotoEnv(config,initial_template=t,output=output,evaluator=evaluator) as env:
        observation,info=env.reset(seed=seed);done=False
        while not done:
            observation,_,terminated,truncated,info=env.step(env.sample_action());done=terminated or truncated
        return env.last_result


def revalidate(t,max_clusters,margin_target=.693):
    # Fresh, uncached same-margin solve with explicit tighter conic tolerances.
    evaluation=TemplateEvaluator(EvaluationConfig(max_clusters=max_clusters,mu_target=margin_target,
        solver_options=dict(tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=300))).evaluate(t)
    checks,passed,message=audit_candidate(t,evaluation,RefinementConfig(max_clusters=max_clusters),"refined")
    return evaluation,passed,dict(checks=[asdict(c) for c in checks],gate_passed=passed,message=message,
        proof_available=False,template=dict(x=t.x.tolist(),phi=t.phi.tolist()),
        P=evaluation.P.tolist() if evaluation.P is not None else None,
        stable_mu=evaluation.stable_mu,sdp_calls=evaluation.sdp_calls,status=evaluation.status,
        density_status=evaluation.density.status if evaluation.density else None,
        residuals=evaluation.density.residuals if evaluation.density else {})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--bc-checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--families',default='eight_twisted')
    parser.add_argument('--methods',default='random,awac')
    parser.add_argument('--budgets',default='32')
    parser.add_argument('--repetitions',type=int,default=2)
    parser.add_argument('--max-clusters',type=int,default=24)
    parser.add_argument('--seed',type=int,default=20261026)
    parser.add_argument('--allow-training-families',action='store_true',help='Explicit in-distribution evaluation; not held-out evidence')
    args=parser.parse_args(argv)
    families=args.families.split(',');methods=args.methods.split(',');budgets=[int(x) for x in args.budgets.split(',')]
    if min(budgets)<3 or args.repetitions<1:parser.error('Invalid evaluation budget/repetitions')
    if set(methods)-{'random','awac','bc','archive_random','twisted'}:parser.error('Unknown method')
    if 'bc' in methods and not args.bc_checkpoint:parser.error('bc requires --bc-checkpoint')
    if 'twisted' in methods and min(budgets)<args.max_clusters-2:parser.error('Twisted enumeration needs at least one call per seed')
    for family in families:initial(family,args.seed,args.max_clusters)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    model,checkpoint=load_policy(args.checkpoint)
    train_keys=set(checkpoint['state'].get('train_keys',[]));training_families=set(checkpoint['state'].get('families',[]))
    bc=None
    if args.bc_checkpoint:
        bc,bc_state=load_policy(args.bc_checkpoint)
        if bc.config.margin_target!=model.config.margin_target:raise ValueError('Checkpoints have different SDP margin targets')
        training_families|=set(bc_state['state'].get('families',[]))
        train_keys|=set(bc_state['state'].get('train_keys',[]))
    if training_families&set(families) and not args.allow_training_families:
        parser.error('Evaluation family occurs in checkpoint training provenance; use a held-out family or explicit --allow-training-families')
    args.output.mkdir(parents=True,exist_ok=False)
    runs=[];details=[];refinement_calls=0;inference_seconds=0.;test_calls=0
    for family in families:
        for budget in budgets:
            for repetition in range(args.repetitions):
                seed=derive_seed(args.seed,family,budget,repetition)
                t=initial(family,seed,args.max_clusters)
                for method in methods:
                    output=args.output/'runs'/method/family/f'budget-{budget}'/f'rep-{repetition:03d}'
                    cfg=EnvironmentConfig(max_clusters=args.max_clusters,max_sdp_calls=budget,max_steps=10*budget)
                    started=perf_counter();latency=0.
                    if method in ('awac','bc'):
                        result=run_policy_search(t,model if method=='awac' else bc,seed=seed,config=cfg,output=output,
                            evaluator=TemplateEvaluator(EvaluationConfig(max_clusters=args.max_clusters,mu_target=model.config.margin_target)))
                        latency=result.inference_seconds;search=result.search
                    elif method=='random':search=run_serial_random(t,seed=seed,config=cfg,output=output,
                        evaluator=TemplateEvaluator(EvaluationConfig(max_clusters=args.max_clusters,mu_target=model.config.margin_target)))
                    elif method=='archive_random':
                        search=run_random_search([t],TemplateEvaluator(EvaluationConfig(max_clusters=args.max_clusters,mu_target=model.config.margin_target)),
                            RandomSearchConfig(seed=seed,iterations=10*budget,max_clusters=args.max_clusters,max_sdp_calls=budget),output=output)
                    else:
                        with SearchSession([twisted_seed(m) for m in range(3,args.max_clusters+1)],
                            TemplateEvaluator(EvaluationConfig(max_clusters=args.max_clusters,mu_target=model.config.margin_target)),seed=seed,
                            max_clusters=args.max_clusters,max_sdp_calls=budget,output=output) as session:
                            session.evaluate_seeds();search=session.finish('enumeration_complete')
                    wall=perf_counter()-started;test_calls+=search.sdp_calls;inference_seconds+=latency
                    metric=compute_run_metrics(search,algorithm=method,initial=family,budget=budget,repetition=repetition,run_seed=seed,wall_seconds=wall)
                    screened_mu=metric.best_stable_mu;fresh=None
                    if search.best is not None:
                        evaluation,passed,fresh=revalidate(search.best.evaluation.template,args.max_clusters,model.config.margin_target)
                        refinement_calls+=evaluation.sdp_calls
                        metric.best_stable_mu=(min(screened_mu,evaluation.stable_mu) if passed and evaluation.stable_mu is not None else None)
                    visited={key(r.template) for r in search.trace if r.template is not None}
                    non_twisted=[];distances=[]
                    for record in search.trace:
                        e=record.evaluation
                        if e is None or e.stable_mu is None:continue
                        candidate=e.template;m=candidate.n_clusters;angles=np.sort(candidate.phi)
                        gaps=np.diff(np.r_[angles,angles[0]+2*np.pi])
                        distance=float(np.sqrt(np.mean((candidate.x-1/m)**2))+np.std(gaps)/(2*np.pi))
                        distances.append(distance)
                        if distance>1e-5:non_twisted.append(e.stable_mu)
                    detail=dict(method=method,family=family,budget=budget,repetition=repetition,run_seed=seed,
                        screened_mu=screened_mu,revalidated_mu=metric.best_stable_mu,revalidation=fresh,
                        actual_sdp_calls=search.sdp_calls,attempts=search.proposals,inference_seconds=latency,
                        training_family=family in training_families,canonical_training_overlap=len(visited&train_keys),
                        distinct_canonical=len(visited),best_non_twisted_screened_mu=max(non_twisted,default=None),
                        non_twisted_metric='mass RMS deviation plus circular gap SD; threshold 1e-5; not proof of new structure')
                    write_json(output/'evaluation.json',detail);details.append(detail);runs.append(metric)
    comparisons=[]
    baseline='random' if 'random' in methods else methods[0]
    for method in methods:
        if method!=baseline:comparisons.extend(paired_comparisons(runs,baseline=baseline,candidate=method))
    provenance={name:dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                for name,p in (('awac',args.checkpoint),('bc',args.bc_checkpoint)) if p is not None}
    report=write_benchmark_report(args.output,config_data=dict(max_clusters=args.max_clusters,budgets=budgets,
        checkpoints=provenance,margin_target=model.config.margin_target,
        methods=methods,families=families,repetitions=args.repetitions,seed=args.seed,
        headline='Fresh tighter same-margin numerical audit; conservative minimum of original and fresh mu',
        progress='Screening curve only; intermediate points are not all freshly revalidated',
        archive_random='Uncached evaluator; existing exact deduplication remains',
        twisted='Direct enumeration can finish below its allocation; actual costs reported'),
        runs=runs,summaries=summarize_runs(runs,target_mu=.6929077),comparisons=comparisons)
    practical=[]
    for method in methods:
        if method==baseline:continue
        base={(r.initial,r.budget,r.repetition):r.best_stable_mu for r in runs if r.algorithm==baseline}
        w=l=tie=0
        for r in runs:
            if r.algorithm!=method:continue
            a,b=r.best_stable_mu,base[(r.initial,r.budget,r.repetition)]
            if a is None and b is None:tie+=1
            elif a is None:l+=1
            elif b is None:w+=1
            elif a>b+1e-6:w+=1
            elif b>a+1e-6:l+=1
            else:tie+=1
        practical.append(dict(candidate=method,baseline=baseline,wins=w,losses=l,ties=tie,tolerance=1e-6))
    values=[r.best_stable_mu for r in runs if r.best_stable_mu is not None]
    summary=dict(runs=len(runs),online_training_sdp_calls=checkpoint['state']['training_calls'],
        checkpoints=provenance,historical_source_sdp_calls=checkpoint['state'].get('historical_source_sdp_calls'),
        frozen_inference_sdp_calls=test_calls,fresh_revalidation_sdp_calls=refinement_calls,
        policy_inference_seconds=inference_seconds,practical_paired=practical,
        threshold_counts={str(t):sum(v>t+1e-6 for v in values) for t in (.68196575,.683875483,.6929077)},
        confidence_intervals='Not estimated in tiny M4 run; use adequately repeated M5 matrix',
        performance_claim=False,proof_available=False,artifacts={k:str(v) for k,v in report.items()})
    write_json(args.output/'evaluation_summary.json',summary);print(json.dumps(summary));return summary


if __name__=='__main__':main()
