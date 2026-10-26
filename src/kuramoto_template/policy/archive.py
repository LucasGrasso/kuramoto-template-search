"""Bounded reset diversity, never an alternative numerical evaluator."""
import numpy as np
from kuramoto_template.experiments.cache import canonical_key
from kuramoto_template.template import PhaseTemplate
from kuramoto_template.sdp_optimizer.validate_input import validate_input


class ResetArchive:
    def __init__(self,capacity=128):self.capacity=capacity;self.entries=[];self.seen=set()

    def consider(self,t,e,family):
        try:validate_input(t.x,t.phi)
        except ValueError:return
        k=repr(canonical_key(t))
        if k in self.seen:return
        phi=np.sort(t.phi);gaps=np.diff(np.r_[phi,phi[0]+2*np.pi])
        cell=[t.n_clusters,int(np.std(t.x)*100),int(np.std(gaps)*10)]
        entry=dict(x=t.x.tolist(),phi=t.phi.tolist(),mu=e.stable_mu,cell=cell,family=family,key=k)
        same=[x for x in self.entries if x['cell']==cell]
        if len(same)>=2:
            victim=min(same,key=lambda x:x['mu'] if x['mu'] is not None else -1)
            if (victim['mu'] or -1)>(entry['mu'] or -1):return
            self.entries.remove(victim)
        self.entries.append(entry);self.seen.add(k)
        if len(self.entries)>self.capacity:
            best=max(self.entries,key=lambda x:x['mu'] if x['mu'] is not None else -1)
            victim=next(x for x in self.entries if x is not best)
            self.entries.remove(victim)

    def sample(self,rng):
        item=self.entries[int(rng.integers(len(self.entries)))]
        return PhaseTemplate(item['x'],item['phi']),item['family']

    def state_dict(self):return dict(entries=self.entries,seen=sorted(self.seen),capacity=self.capacity)
    def load_state_dict(self,state):self.entries=state['entries'];self.seen=set(state['seen']);self.capacity=state['capacity']
