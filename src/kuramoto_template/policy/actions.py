"""Masked categorical and properly normalized truncated Gaussian actions.

Continuous likelihoods refer to Gym's normalized coordinates. They are not
physical-angle densities. Sampling uses inverse CDF, not post-hoc clipping.
"""
import math
import torch
from torch import nn


class NoValidAction(ValueError):
    pass


def masked_log_probs(logits, mask):
    mask=mask.bool()
    if not mask.any(-1).all():raise NoValidAction("No action is valid in at least one state")
    return torch.log_softmax(logits.masked_fill(~mask,-torch.inf),dim=-1)


def categorical(logits,mask,generator=None,deterministic=False):
    lp=masked_log_probs(logits,mask)
    index=lp.argmax(-1) if deterministic else torch.multinomial(lp.exp(),1,generator=generator).squeeze(-1)
    return index,lp.gather(-1,index[...,None]).squeeze(-1)


def normal_parameters(raw):
    return .8*torch.tanh(raw[...,:2]), .05+.45*torch.sigmoid(raw[...,2:])


def truncated_log_prob(value,loc,scale):
    # Double-precision CDF/log normalizer; loc and scale keep Z away from zero.
    x,mu,s=value.double(),loc.double(),scale.double()
    normal=torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(s))
    z=normal.cdf((1-mu)/s)-normal.cdf((-1-mu)/s)
    lp=-.5*((x-mu)/s).square()-torch.log(s)-.5*math.log(2*math.pi)-torch.log(z)
    return lp.masked_fill((x < -1)|(x > 1),-torch.inf).to(loc.dtype)


def truncated_sample(loc,scale,generator=None,deterministic=False):
    if deterministic:return loc.detach()
    mu,s=loc.double(),scale.double()
    normal=torch.distributions.Normal(torch.zeros_like(mu),torch.ones_like(s))
    left,right=normal.cdf((-1-mu)/s),normal.cdf((1-mu)/s)
    u=torch.rand(mu.shape,device=mu.device,dtype=torch.float64,generator=generator)
    # Protect inverse CDF only against finite-RNG endpoints, not action clipping.
    eps=torch.finfo(torch.float64).eps
    probability=(left+(right-left)*u).clamp(eps,1-eps)
    return (mu+s*normal.icdf(probability)).to(loc.dtype).detach()


def mlp(inputs,hidden,outputs):
    return nn.Sequential(nn.Linear(inputs,hidden),nn.SiLU(),nn.Linear(hidden,outputs))


class HybridActor(nn.Module):
    def __init__(self,hidden):
        super().__init__();self.embedding=nn.Embedding(4,hidden)
        self.operation=mlp(hidden,hidden,4)
        self.node=mlp(3*hidden,hidden,1)
        self.pair=mlp(4*hidden+2,hidden,1)
        self.parameters_head=mlp(4*hidden,hidden,4)

    def logits(self,nodes,context,obs):
        B,N,H=nodes.shape
        node_scores=[];pair_scores=[]
        for op in range(4):
            embedding=self.embedding.weight[op]
            node_scores.append(self.node(torch.cat((nodes,context[:,None,:].expand(-1,N,-1),
                embedding.expand(B,N,-1)),dim=-1)).squeeze(-1))
            if op in (1,3):
                a=nodes[:,:,None,:].expand(-1,-1,N,-1)
                b=nodes[:,None,:,:].expand(-1,N,-1,-1)
                c=context[:,None,None,:].expand(-1,N,N,-1)
                raw=self.pair(torch.cat((a,b,c,embedding.expand(B,N,N,-1),obs["edges"]),-1)).squeeze(-1)
                if op==1:raw=(raw+raw.transpose(1,2))/2
            else:raw=nodes.new_zeros((B,N,N))
            pair_scores.append(raw)
        return self.operation(context),torch.stack(node_scores,1),torch.stack(pair_scores,1)

    def continuous_parameters(self,nodes,context,action):
        B=nodes.shape[0]; rows=torch.arange(B,device=nodes.device)
        op=action["operation"];first=nodes[rows,action["i"]]
        second=nodes[rows,action["j"]]*((op==3).float()[:,None])
        return normal_parameters(self.parameters_head(torch.cat((context,first,second,self.embedding(op)),-1)))

    def log_prob(self,nodes,context,obs,action):
        ops,ns,ps=self.logits(nodes,context,obs)
        op=action["operation"];B,N=ns.shape[0],ns.shape[2];rows=torch.arange(B,device=nodes.device)
        lp=masked_log_probs(ops,obs["operation_mask"]).gather(1,op[:,None]).squeeze(1)
        for kind in (0,1,2,3):
            selected=op==kind
            if not selected.any():continue
            if kind in (0,2):
                conditional=masked_log_probs(ns[selected,kind],obs["node_mask"][selected])
                lp=lp.index_add(0,rows[selected],conditional.gather(1,action["i"][selected,None]).squeeze(1))
            else:
                conditional=masked_log_probs(ps[selected,kind].flatten(1),obs["pair_mask"][selected,kind].flatten(1))
                index=action["i"][selected]*N+action["j"][selected]
                lp=lp.index_add(0,rows[selected],conditional.gather(1,index[:,None]).squeeze(1))
        loc,scale=self.continuous_parameters(nodes,context,action)
        continuous=truncated_log_prob(action["parameters"],loc,scale)
        active=torch.stack((op!=1,op==0),-1)
        return lp+torch.where(active,continuous,torch.zeros_like(continuous)).sum(-1)

    def sample(self,nodes,context,obs,generator=None,deterministic=False):
        ops,ns,ps=self.logits(nodes,context,obs);B,N=ns.shape[0],ns.shape[2]
        op,_=categorical(ops,obs["operation_mask"],generator,deterministic)
        i=torch.zeros_like(op);j=torch.zeros_like(op)
        for kind in range(4):
            selected=op==kind
            if not selected.any():continue
            if kind in (0,2):
                index,_=categorical(ns[selected,kind],obs["node_mask"][selected],generator,deterministic)
                i[selected]=index
            else:
                index,_=categorical(ps[selected,kind].flatten(1),obs["pair_mask"][selected,kind].flatten(1),generator,deterministic)
                i[selected],j[selected]=index//N,index%N
        action=dict(operation=op,i=i,j=j)
        loc,scale=self.continuous_parameters(nodes,context,action)
        parameters=truncated_sample(loc,scale,generator,deterministic)
        active=torch.stack((op!=1,op==0),-1)
        action["parameters"]=torch.where(active,parameters,torch.zeros_like(parameters))
        return action,self.log_prob(nodes,context,obs,action)

    def entropy(self,nodes,context,obs,generator=None):
        """Monte Carlo joint entropy, including conditional continuous density."""
        return -self.sample(nodes,context,obs,generator)[1]
