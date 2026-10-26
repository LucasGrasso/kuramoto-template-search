import torch
from torch import nn
from .actions import HybridActor,mlp
from .config import PolicyConfig
from .encoding import batch_observations,numpy_action


class GraphEncoder(nn.Module):
    def __init__(self,config):
        super().__init__();H=config.hidden;self.config=config
        self.nodes=mlp(2,H,H)
        self.messages=nn.ModuleList(mlp(2*H+2,H,H) for _ in range(config.layers))
        self.updates=nn.ModuleList(mlp(2*H,H,H) for _ in range(config.layers))
        self.norms=nn.ModuleList(nn.LayerNorm(H) for _ in range(config.layers))
        # 8 global + 6 statuses + 8 diagnostics + 8 availability + m + two targets.
        self.context=mlp(H+33,H,H)

    def forward(self,obs):
        mask=obs["node_mask"].bool();B,N=mask.shape
        count=mask.sum(1).clamp_min(1)
        h=self.nodes(obs["nodes"])*mask[:,:,None]
        pair_mask=mask[:,:,None]&mask[:,None,:]
        for message,update,norm in zip(self.messages,self.updates,self.norms):
            pair=torch.cat((h[:,:,None,:].expand(-1,-1,N,-1),h[:,None,:,:].expand(-1,N,-1,-1),obs["edges"]),-1)
            aggregate=(message(pair)*pair_mask[:,:,:,None]).sum(2)/count[:,None,None]
            h=norm(h+update(torch.cat((h,aggregate),-1)))*mask[:,:,None]
        pooled=h.sum(1)/count[:,None]
        global_features=obs["global_features"].clone()
        global_features[:,4:6]/=1000
        diagnostics=torch.asinh(obs["diagnostics"])*obs["diagnostic_mask"]
        features=torch.cat((global_features,obs["status"],diagnostics.flatten(1),obs["diagnostic_mask"].flatten(1),
            count[:,None]/24,torch.full((B,1),self.config.target_mu,device=h.device),
            torch.full((B,1),self.config.margin_target,device=h.device)),-1)
        return h,self.context(torch.cat((pooled,features),-1))


class TwinCritics(nn.Module):
    def __init__(self,H):
        super().__init__();self.embedding=nn.Embedding(4,H)
        self.q1=mlp(4*H+4,H,1);self.q2=mlp(4*H+4,H,1)
        self.auxiliary=mlp(4*H+4,H,9)

    def features(self,nodes,context,obs,action):
        B=nodes.shape[0];r=torch.arange(B,device=nodes.device);op=action["operation"]
        a,b=nodes[r,action["i"]],nodes[r,action["j"]]
        merge=(op==1)[:,None];pair=((op==1)|(op==3))[:,None]
        first=torch.where(merge,(a+b)/2,a)
        second=torch.where(merge,(a-b).abs(),b)*pair
        edge=obs["edges"][r,action["i"],action["j"]].clone()
        edge[:,0]=torch.where(op==1,edge[:,0].abs(),edge[:,0]);edge*=pair
        active=torch.stack((op!=1,op==0),-1)
        return torch.cat((context,first,second,self.embedding(op),action["parameters"]*active,edge),-1)

    def forward(self,nodes,context,obs,action):
        f=self.features(nodes,context,obs,action)
        return self.q1(f).squeeze(-1),self.q2(f).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self,config=None):
        super().__init__();self.config=config or PolicyConfig()
        self.encoder=GraphEncoder(self.config);self.actor=HybridActor(self.config.hidden)
        self.critics=TwinCritics(self.config.hidden)

    def sample_action(self,observation,generator=None,deterministic=False):
        device=next(self.parameters()).device
        with torch.no_grad():
            obs=batch_observations([observation],device);h,c=self.encoder(obs)
            a,lp=self.actor.sample(h,c,obs,generator,deterministic)
        return numpy_action(a),float(lp[0])

    def mode_action(self,observation):
        """Argmax discrete hierarchy and truncated-Normal central location."""
        return self.sample_action(observation,deterministic=True)[0]
