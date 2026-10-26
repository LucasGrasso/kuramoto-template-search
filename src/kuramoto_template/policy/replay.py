"""Only complete serial transitions enter TD replay; branch actions are BC data."""
from dataclasses import dataclass
import copy
import numpy as np
import torch


@dataclass
class Transition:
    observation: dict
    action: dict
    reward: float
    next_observation: dict
    terminated: bool
    truncated: bool
    sdp_calls: int
    episode_id: str
    stable_label: float | None = None
    mu_label: float | None = None
    gamma_label: float | None = None
    outcome_label: int | None = None

    @property
    def continuation(self):
        # V1 treats every recorded collection boundary as a hard episode end.
        return not (self.terminated or self.truncated)


def tensorize(value):
    if isinstance(value,np.ndarray):return torch.from_numpy(value.copy())
    if isinstance(value,dict):return {k:tensorize(v) for k,v in value.items()}
    if isinstance(value,list):return [tensorize(v) for v in value]
    if isinstance(value,tuple):return tuple(tensorize(v) for v in value)
    return value


def untensorize(value):
    if isinstance(value,torch.Tensor):return value.cpu().numpy().copy()
    if isinstance(value,dict):return {k:untensorize(v) for k,v in value.items()}
    if isinstance(value,list):return [untensorize(v) for v in value]
    if isinstance(value,tuple):return tuple(untensorize(v) for v in value)
    return value


class Replay:
    def __init__(self,capacity=20000,seed=0):
        if capacity < 1:raise ValueError("Replay capacity must be positive")
        self.capacity=capacity;self.items=[];self.position=0;self.rng=np.random.default_rng(seed)

    def __len__(self):return len(self.items)

    def add(self,transition):
        if not isinstance(transition,Transition):raise TypeError("TD replay requires a Transition")
        if transition.sdp_calls not in (0,1,2) or not np.isfinite(transition.reward):
            raise ValueError("Invalid cost or reward")
        if not transition.episode_id:raise ValueError("Missing episode provenance")
        # Serialized observations include the incumbent and exact call context.
        before=transition.observation["global_features"]
        after=transition.next_observation["global_features"]
        if abs(float(before[4]-after[4])-transition.sdp_calls)>1e-4:
            raise ValueError("Transition call balance does not reconcile")
        if transition.terminated and after[4]!=0:
            raise ValueError("SDP terminal state must have zero remaining calls")
        if len(self.items)<self.capacity:self.items.append(copy.deepcopy(transition))
        else:self.items[self.position]=copy.deepcopy(transition)
        self.position=(self.position+1)%self.capacity

    def sample(self,size):
        if not self.items:raise ValueError("Empty replay")
        return [self.items[i] for i in self.rng.integers(len(self.items),size=size)]

    def state_dict(self):
        return dict(capacity=self.capacity,position=self.position,rng=self.rng.bit_generator.state,
                    items=[tensorize(vars(t)) for t in self.items])

    def load_state_dict(self,state):
        self.capacity=state["capacity"];self.position=state["position"]
        self.rng.bit_generator.state=state["rng"]
        self.items=[Transition(**untensorize(t)) for t in state["items"]]
