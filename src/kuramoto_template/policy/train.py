"""Bounded BC + online AWAC training. No expensive experiment runs by default."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
from time import perf_counter
from types import SimpleNamespace
import numpy as np
import torch

from kuramoto_template.environment import EnvironmentConfig,KuramotoEnv
from kuramoto_template.environment.env import STATUSES
from kuramoto_template.experiments.seeds import build_initial,twisted,perturbed_twisted
from kuramoto_template.experiments.cache import canonical_key
from kuramoto_template.experiments.reporting import json_safe
from kuramoto_template.search.evaluator import EvaluationConfig
from .config import PolicyConfig
from .learner import Learner
from .replay import Replay,Transition,tensorize
from .dataset import load_history
from .archive import ResetArchive


def initial(family,seed,max_clusters):
    if family.startswith('twisted_'):t=twisted(int(family.split('_')[-1])).build(np.random.default_rng(seed))
    elif family.startswith('perturbed_'):t=perturbed_twisted(int(family.split('_')[-1])).build(np.random.default_rng(seed))
    else:t=build_initial(family,seed)
    if not 3<=t.n_clusters<=max_clusters:raise ValueError(f'{family} outside cluster bounds')
    return t


def write_json(path,data):
    Path(path).write_text(json.dumps(json_safe(data),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def save_checkpoint(path,learner,replay,state,archive):
    payload=dict(schema_version=1,learner=learner.state_dict(),replay=replay.state_dict(),
        state=state,archive=archive.state_dict(),python_rng=random.getstate())
    temp=Path(str(path)+'.tmp');torch.save(payload,temp);temp.replace(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--training-calls',type=int,default=256,help='Total SDP-call cap, including earlier resumed training')
    parser.add_argument('--episode-calls',type=int,default=32)
    parser.add_argument('--max-clusters',type=int,default=24)
    parser.add_argument('--families',default='five_twisted,four_perturbed,random_phases')
    parser.add_argument('--held-out-families',default='eight_twisted,twisted_16,twisted_19')
    parser.add_argument('--warmup-calls',type=int,default=32)
    parser.add_argument('--bc-steps',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--max-updates',type=int,default=256)
    parser.add_argument('--updates-per-step',type=int,default=1)
    parser.add_argument('--random-fraction',type=float,default=.2)
    parser.add_argument('--archive-reset-fraction',type=float,default=.25)
    parser.add_argument('--hidden',type=int,default=128)
    parser.add_argument('--layers',type=int,default=2)
    parser.add_argument('--auxiliary-weight',type=float,default=0.)
    parser.add_argument('--reflection',action='store_true')
    parser.add_argument('--history',type=Path,action='append',default=[])
    parser.add_argument('--history-limit',type=int,default=2000,help='Maximum imported rows; 0 imports all rows')
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--temperature',type=float,default=1.)
    parser.add_argument('--max-weight',type=float,default=20.)
    parser.add_argument('--discount',type=float,default=1.)
    parser.add_argument('--tau',type=float,default=.01)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args(argv)
    if min(args.training_calls,args.episode_calls)<3 or args.batch_size<1 or min(args.bc_steps,args.max_updates,args.updates_per_step)<0:
        parser.error('Invalid budget, batch size or update limit')
    if args.history_limit<0:parser.error('history-limit must be nonnegative')
    if not 0<=args.random_fraction<=1 or not 0<=args.archive_reset_fraction<=1:parser.error('Fractions must lie in [0,1]')
    families=tuple(x for x in args.families.split(',') if x);heldout=tuple(x for x in args.held_out_families.split(',') if x)
    if not families or set(families)&set(heldout):parser.error('Training and held-out families must be nonempty/disjoint')
    for family in families:initial(family,args.seed,args.max_clusters)
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);random.seed(args.seed)
    rng=np.random.default_rng(args.seed)
    cfg=PolicyConfig(hidden=args.hidden,layers=args.layers,auxiliary_weight=args.auxiliary_weight,
        learning_rate=args.learning_rate,temperature=args.temperature,max_weight=args.max_weight,
        discount=args.discount,tau=args.tau)
    learner=Learner(cfg,args.seed);replay=Replay(seed=args.seed);archive=ResetArchive()
    state=dict(training_calls=0,action_calls=0,random_calls=0,episodes=0,bc_done=False,train_keys=[],
               max_clusters=args.max_clusters,families=list(families),held_out_families=list(heldout),episode_calls=args.episode_calls)
    settings={k:(str(v) if isinstance(v,Path) else [str(x) for x in v] if k=='history' else v)
              for k,v in vars(args).items() if k not in ('output','resume','training_calls')}
    state['training_settings']=settings
    defaults=EnvironmentConfig()
    state['decoder_config']={k:getattr(defaults,k) for k in ('phase_scale','separation_min','separation_max','resize_fraction','min_clusters')}
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=True)
        if not checkpoint['state'].get('resume_boundary'):raise ValueError('Only complete episode-boundary checkpoints can resume training')
        learner=Learner(PolicyConfig(**checkpoint['learner']['policy_config']),args.seed)
        learner.load_state_dict(checkpoint['learner']);replay.load_state_dict(checkpoint['replay'])
        state=checkpoint['state'];archive.load_state_dict(checkpoint['archive']);random.setstate(checkpoint['python_rng'])
        rng.bit_generator.state=state['numpy_rng']
        if args.training_calls<state['training_calls']:raise ValueError('Requested cap is below already-spent training calls')
        if state.get('training_settings')!=settings:raise ValueError('Resume settings differ; only the total training-call cap may increase')
        if state['max_clusters']!=args.max_clusters or state['families']!=list(families) or state['episode_calls']!=args.episode_calls:
            raise ValueError('Resume environment/reset configuration differs')
    forbidden=set()
    for family in heldout:
        for holdout_seed in range(20):
            try:forbidden.add(repr(canonical_key(initial(family,holdout_seed,args.max_clusters),decimals=6)))
            except ValueError:pass  # e.g. m=19 is outside the m<=12 condition
    historical=load_history(args.history,train_families=families,held_out_families=heldout,
        seed=args.seed,max_records=args.history_limit or None,forbidden_keys=forbidden,margin_target=learner.config.margin_target,
        action_config=EnvironmentConfig(max_clusters=args.max_clusters,max_sdp_calls=args.episode_calls,max_steps=10*args.episode_calls)) if args.history else None
    if historical:
        forbidden.update(historical.manifest['heldout_keys'])
        fingerprint=hashlib.sha256(json.dumps(historical.manifest['sources'],sort_keys=True).encode()).hexdigest()
        if args.resume and state.get('history_fingerprint')!=fingerprint:raise ValueError('Historical source data changed since checkpoint')
        state['history_fingerprint']=fingerprint
        write_json(args.output/'dataset_manifest.json',historical.manifest)
        torch.save(tensorize(dict(actions=[vars(a) for a in historical.actions],states=historical.states)),args.output/'historical_dataset.pt')
        if not args.resume:
            for transition in historical.transitions:replay.add(transition)
        state['train_keys']=sorted(set(state['train_keys'])|set(historical.manifest['train_keys']))
    else:write_json(args.output/'dataset_manifest.json',dict(train_actions=0,clean_td_transitions=0,reason='No history supplied; clean online replay only'))
    state['historical_source_sdp_calls']=(historical.manifest['historical_source_sdp_calls'] if historical else state.get('historical_source_sdp_calls',0))
    source_root=Path(__file__).resolve().parents[1]
    digest=hashlib.sha256()
    for p in sorted(source_root.rglob('*.py')):digest.update(p.read_bytes())
    state['source_sha256']=digest.hexdigest()
    write_json(args.output/'config.json',dict(arguments={k:str(v) if isinstance(v,Path) else [str(x) for x in v] if k=='history' else v for k,v in vars(args).items()},
        policy=asdict(learner.config),source_sha256=digest.hexdigest(),python=platform.python_version(),
        versions={p:importlib.metadata.version(p) for p in ('torch','gymnasium','numpy','cvxpy','clarabel')},
        discount_convention='Both terminal and truncation end continuation; discount=1 telescopes within each episode',
        exploration='Pace random actions by actual paid action calls; seeds excluded from allocation but included in training cost'))
    keys=set(state['train_keys']);started=perf_counter()
    with (args.output/'metrics.jsonl').open('x',encoding='utf-8') as log:
        def metric(data):
            log.write(json.dumps(dict(training_calls=state['training_calls'],updates=learner.updates,**data),allow_nan=False)+'\n');log.flush()
        while args.training_calls-state['training_calls']>=3:
            capacity=min(args.episode_calls,args.training_calls-state['training_calls'])
            family=families[int(rng.integers(len(families)))];episode_seed=int(rng.integers(2**31))
            if archive.entries and rng.random()<args.archive_reset_fraction:t,family=archive.sample(rng);reset_source='archive'
            else:t=initial(family,episode_seed,args.max_clusters);reset_source='family'
            ecfg=EnvironmentConfig(max_clusters=args.max_clusters,max_sdp_calls=capacity,max_steps=10*capacity)
            root=args.output/'episodes'/f'collection-{state["episodes"]:06d}'
            env=KuramotoEnv(ecfg,initial_template=t,output=root,
                evaluation_config=EvaluationConfig(max_clusters=args.max_clusters,mu_target=learner.config.margin_target))
            episode_id=f'{args.output.name}:{state["episodes"]}'
            try:
                observation,info=env.reset(seed=episode_seed);state['training_calls']+=info['sdp_calls']
                write_json(root/'episode-000000'/'policy_episode.json',dict(family=family,reset_source=reset_source,seed=episode_seed))
                initial_best=info['best_mu'];total_reward=0.;done=False
                if repr(canonical_key(t,decimals=6)) not in forbidden:keys.add(repr(canonical_key(t,decimals=6)))
                while not done:
                    random_action=(state['training_calls']<args.warmup_calls or not state['bc_done'] or
                        (args.random_fraction>0 and state['random_calls']<=args.random_fraction*state['action_calls']))
                    action=env.sample_action() if random_action else learner.model.sample_action(observation,learner.generator)[0]
                    before=info['best_mu'];current_template=env._record.template
                    next_observation,reward,terminated,truncated,info=env.step(action)
                    done=terminated or truncated;cost=info['sdp_calls']
                    state['training_calls']+=cost;state['action_calls']+=cost
                    if random_action:state['random_calls']+=cost
                    if info['steps']>0 and 'record_id' in info:
                        e=env._record.evaluation
                        status=e.status;stable=None if status in ('uncertain','solver_error') else float(status=='stable')
                        transition=Transition(observation,action,reward,next_observation,terminated,truncated,cost,episode_id,
                            stable,e.stable_mu,float(np.tanh(e.margin_gamma*e.template.n_clusters)) if e.margin_gamma is not None else None,
                            STATUSES.index(status) if status in STATUSES else None)
                        if (repr(canonical_key(env._record.template,decimals=6)) not in forbidden
                            and repr(canonical_key(current_template,decimals=6)) not in forbidden):
                            replay.add(transition)
                    observation=next_observation;total_reward+=reward
                    current=env._record
                    candidate_key=repr(canonical_key(current.template,decimals=6))
                    if candidate_key not in forbidden:
                        keys.add(candidate_key);archive.consider(current.template,current.evaluation,family)
                    if len(replay)>=4 and state['training_calls']>=args.warmup_calls:
                        if not state['bc_done']:
                            pool=(historical.actions if historical and historical.actions else
                                  [SimpleNamespace(observation=t.observation,action=t.action,
                                      weight=1+min(4,t.reward) if t.reward>0 else .25) for t in replay.items])
                            for _ in range(args.bc_steps):
                                batch=[pool[i] for i in rng.integers(len(pool),size=args.batch_size)]
                                if args.reflection:
                                    from .encoding import reflect
                                    from copy import deepcopy
                                    batch=deepcopy(batch)
                                    for item in batch:
                                        if rng.random()<.5 and not (item.action['operation']==0 and item.action['parameters'][1]==0):
                                            item.observation,item.action=reflect(item.observation,item.action)
                                metric(learner.bc_update(batch))
                            state['bc_done']=True
                            learner.target.load_state_dict(learner.model.state_dict())
                            state['numpy_rng']=rng.bit_generator.state;state['train_keys']=sorted(keys)
                            # BC is a frozen ablation; this checkpoint is for
                            # inference, not mid-episode training resumption.
                            torch.save(dict(schema_version=1,learner=learner.state_dict(),state={**state,'resume_boundary':False}),args.output/'bc.pt')
                        for _ in range(args.updates_per_step):
                            if learner.updates>=args.max_updates:break
                            metric(learner.update(replay.sample(args.batch_size)))
                if abs(total_reward-ecfg.reward_scale*(info['best_mu']-initial_best))>1e-5:raise RuntimeError('Reward did not telescope')
                state['episodes']+=1
                metric(dict(event='episode_end',family=family,reset_source=reset_source,best_mu=info['best_mu'],
                    episode_calls=capacity- info['remaining_sdp_calls'],reward=total_reward,stop_reason=info['stop_reason']))
            finally:env.close()
            state['numpy_rng']=rng.bit_generator.state;state['train_keys']=sorted(keys);state['resume_boundary']=True
            save_checkpoint(args.output/'checkpoint.pt',learner,replay,state,archive)
        state['numpy_rng']=rng.bit_generator.state;state['train_keys']=sorted(keys);state['resume_boundary']=True
        save_checkpoint(args.output/'checkpoint.pt',learner,replay,state,archive)
    summary=dict(online_training_sdp_calls=state['training_calls'],requested_cap=args.training_calls,
        unused_calls=args.training_calls-state['training_calls'],episodes=state['episodes'],updates=learner.updates,
        replay_transitions=len(replay),historical_bc_actions=len(historical.actions) if historical else 0,
        random_action_paid_calls=state['random_calls'],all_action_paid_calls=state['action_calls'],
        random_paid_fraction=state['random_calls']/max(1,state['action_calls']),wall_seconds=perf_counter()-started,
        historical_source_sdp_calls=historical.manifest['historical_source_sdp_calls'] if historical else 0,
        performance_claim=False,numerical_certificate=False)
    write_json(args.output/'summary.json',summary);write_json(args.output/'archive.json',archive.state_dict())
    print(json.dumps(summary));return summary


if __name__=='__main__':main()
