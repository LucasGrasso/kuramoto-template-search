"""Frozen actor adapter. It returns the existing SearchResult and timings."""
from dataclasses import dataclass
from time import perf_counter
import torch
from kuramoto_template.environment import KuramotoEnv,EnvironmentConfig
from kuramoto_template.policy.config import PolicyConfig
from kuramoto_template.policy.network import ActorCritic


def load_policy(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=ActorCritic(PolicyConfig(**checkpoint['learner']['policy_config']))
    model.load_state_dict(checkpoint['learner']['model']);model.eval()
    model.decoder_config=checkpoint['state'].get('decoder_config')
    return model,checkpoint


@dataclass
class PolicySearchResult:
    search: object
    inference_seconds: float
    policy_decisions: int


def run_policy_search(initial_template,model,*,seed,config=None,output=None,evaluator=None):
    config=config or EnvironmentConfig()
    if getattr(model,'decoder_config',None) is not None and any(
        getattr(config,k)!=v for k,v in model.decoder_config.items()):
        raise ValueError('Environment action decoder differs from checkpoint')
    elapsed=0.;decisions=0
    with KuramotoEnv(config,initial_template=initial_template,output=output,evaluator=evaluator) as env:
        observation,info=env.reset(seed=seed);done=False
        while not done:
            start=perf_counter();action=model.mode_action(observation);elapsed+=perf_counter()-start;decisions+=1
            observation,_,terminated,truncated,info=env.step(action);done=terminated or truncated
        return PolicySearchResult(env.last_result,elapsed,decisions)
