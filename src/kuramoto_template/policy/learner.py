"""AWAC updates with detached advantages, twin targets and explicit masks."""
import copy
import itertools
import math
from dataclasses import asdict
import torch
from torch.nn import functional as F
from .network import ActorCritic
from .config import PolicyConfig
from .encoding import batch_observations,batch_actions


class Learner:
    def __init__(self,config=None,seed=0):
        self.config=config or PolicyConfig()
        torch.manual_seed(seed)
        self.model=ActorCritic(self.config)
        self.target=copy.deepcopy(self.model).requires_grad_(False)
        self.actor_optimizer=torch.optim.Adam(self.model.actor.parameters(),lr=self.config.learning_rate)
        self.bc_parameters=list(itertools.chain(self.model.encoder.parameters(),self.model.actor.parameters()))
        self.bc_optimizer=torch.optim.Adam(self.bc_parameters,lr=self.config.learning_rate)
        self.critic_parameters=list(itertools.chain(self.model.encoder.parameters(),self.model.critics.parameters()))
        self.critic_optimizer=torch.optim.Adam(self.critic_parameters,lr=self.config.learning_rate)
        self.generator=torch.Generator().manual_seed(seed+1)
        self.updates=0

    def bc_update(self,samples):
        obs=batch_observations([s.observation for s in samples]);actions=batch_actions([s.action for s in samples])
        h,c=self.model.encoder(obs)
        logp=self.model.actor.log_prob(h,c,obs,actions)
        weights=torch.tensor([getattr(s,"weight",1.) for s in samples],dtype=logp.dtype)
        loss=-(weights*logp).sum()/weights.sum().clamp_min(1e-8)
        self._step(loss,self.bc_optimizer,self.bc_parameters,self.config.gradient_clip)
        return {"bc_loss":float(loss.detach()),"bc_log_prob":float(logp.detach().mean())}

    @staticmethod
    def _step(loss,optimizer,parameters,clip=5.):
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters,clip,error_if_nonfinite=True)
        optimizer.step()
        return float(norm)

    def update(self,batch):
        cfg=self.config
        obs=batch_observations([t.observation for t in batch]);nxt=batch_observations([t.next_observation for t in batch])
        actions=batch_actions([t.action for t in batch])
        reward=torch.tensor([t.reward for t in batch],dtype=torch.float32)
        continuation=torch.tensor([t.continuation for t in batch],dtype=torch.bool)
        # A corrupt all-empty next mask cannot produce a NaN target.
        continuation &= nxt["operation_mask"].bool().any(-1)
        with torch.no_grad():
            target=reward.clone()
            if continuation.any():
                sub={k:v[continuation] for k,v in nxt.items()}
                ah,ac=self.model.encoder(sub)
                next_action,_=self.model.actor.sample(ah,ac,sub,self.generator)
                th,tc=self.target.encoder(sub)
                q1,q2=self.target.critics(th,tc,sub,next_action)
                target[continuation]+=cfg.discount*torch.minimum(q1,q2)
        h,c=self.model.encoder(obs);q1,q2=self.model.critics(h,c,obs,actions)
        critic_loss=F.smooth_l1_loss(q1,target)+F.smooth_l1_loss(q2,target)
        auxiliary_loss=critic_loss.new_zeros(())
        if cfg.auxiliary_weight:
            prediction=self.model.critics.auxiliary(self.model.critics.features(h,c,obs,actions))
            for column,attribute in ((6,"stable_label"),(7,"mu_label"),(8,"gamma_label")):
                mask=torch.tensor([getattr(t,attribute) is not None for t in batch])
                if mask.any():
                    label=torch.tensor([getattr(t,attribute) for t in batch if getattr(t,attribute) is not None])
                    auxiliary_loss+=F.binary_cross_entropy_with_logits(prediction[mask,column],label) if column==6 else F.smooth_l1_loss(prediction[mask,column],label)
            mask=torch.tensor([t.outcome_label is not None for t in batch])
            if mask.any():
                label=torch.tensor([t.outcome_label for t in batch if t.outcome_label is not None])
                auxiliary_loss+=F.cross_entropy(prediction[mask,:6],label)
        critic_loss=critic_loss+cfg.auxiliary_weight*auxiliary_loss
        norm=self._step(critic_loss,self.critic_optimizer,self.critic_parameters,cfg.gradient_clip)
        with torch.no_grad():
            h,c=self.model.encoder(obs)
            qa,qb=self.model.critics(h,c,obs,actions)
            sampled,_=self.model.actor.sample(h,c,obs,self.generator)
            va,vb=self.model.critics(h,c,obs,sampled)
            advantage=torch.minimum(qa,qb)-torch.minimum(va,vb)
            weights=torch.exp((advantage/cfg.temperature).clamp(-20,math.log(cfg.max_weight)))
            weights=weights/weights.mean().clamp_min(1e-8)
            weights=weights.clamp(max=cfg.max_weight)
        logp=self.model.actor.log_prob(h.detach(),c.detach(),obs,actions)
        actor_loss=-(weights.detach()*logp).mean()
        self._step(actor_loss,self.actor_optimizer,list(self.model.actor.parameters()),cfg.gradient_clip)
        with torch.no_grad():
            for slow,live in zip(self.target.parameters(),self.model.parameters()):slow.lerp_(live,cfg.tau)
        self.updates+=1
        return dict(actor_loss=float(actor_loss.detach()),critic_loss=float(critic_loss.detach()),
            auxiliary_loss=float(auxiliary_loss.detach()),weight_mean=float(weights.mean()),weight_max=float(weights.max()),
            q_mean=float(torch.minimum(qa,qb).mean()),q_min=float(torch.minimum(qa,qb).min()),q_max=float(torch.minimum(qa,qb).max()),
            td_error=float((q1.detach()-target).abs().mean()),target_max=float(target.abs().max()),gradient_norm=norm,
            replay_nll=float(-logp.detach().mean()))

    def state_dict(self):
        return dict(policy_config=asdict(self.config),model=self.model.state_dict(),target=self.target.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),critic_optimizer=self.critic_optimizer.state_dict(),
            bc_optimizer=self.bc_optimizer.state_dict(),
            generator=self.generator.get_state(),torch_rng=torch.get_rng_state(),updates=self.updates)

    def load_state_dict(self,state):
        if asdict(self.config)!=state["policy_config"]:raise ValueError("Checkpoint config differs")
        self.model.load_state_dict(state["model"]);self.target.load_state_dict(state["target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"]);self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.bc_optimizer.load_state_dict(state["bc_optimizer"])
        self.generator.set_state(state["generator"]);torch.set_rng_state(state["torch_rng"]);self.updates=state["updates"]
