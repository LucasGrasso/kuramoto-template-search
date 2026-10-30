"""Reflection-symmetric multiple splits derived solely from a C4 partition."""

from itertools import product

import numpy as np

from .families import normalize_offsets, parent_mask
from .models import LimitingCandidate


def partition_orbit(counts):
    counts = tuple(counts)
    return min(tuple(counts[(sign*j+turn)%4] for j in range(4))
               for sign in (-1,1) for turn in range(4))


def c4_partitions(max_per_parent=4, max_size=16):
    return tuple(sorted({partition_orbit(c) for c in product(range(1,max_per_parent+1),repeat=4)
                         if 5<=sum(c)<=max_size},key=lambda c:(sum(c),c)))


class MirrorFamily:
    """Equal mirrored masses, arbitrary positive offset levels and parent masses.

    Parent ties are imposed only by the stabilizer of the occupancy partition
    in C4's dihedral group. They are an explicit family restriction, not a
    claim that arbitrary templates may be symmetrized without loss.
    """

    def __init__(self, counts, *, parent_symmetry=True, mass_by_occupancy=False):
        self.counts = tuple(counts)
        if len(self.counts)!=4 or any(not isinstance(k,int) or k<1 for k in self.counts) or sum(self.counts)<5:
            raise ValueError("Need four positive occupancies and at least one split")
        maps = [(sign,turn) for sign in (-1,1) for turn in range(4)
                if all(self.counts[j]==self.counts[(sign*j+turn)%4] for j in range(4))]
        remaining,orbits = set(range(4)),[]
        while remaining:
            j = min(remaining)
            orbit = tuple(sorted({(sign*j+turn)%4 for sign,turn in maps})) if parent_symmetry else (j,)
            orbits.append(orbit)
            remaining.difference_update(orbit)
        self.orbits = tuple(orbits)
        self.mass_orbits = (tuple(tuple(j for j,k in enumerate(self.counts) if k==size)
                                  for size in sorted(set(self.counts)))
                            if mass_by_occupancy else self.orbits)
        self.mass_dim = len(self.mass_orbits)-1
        self.local_mass_dims = tuple((self.counts[o[0]]+1)//2-1 for o in orbits)
        self.level_dims = tuple(self.counts[o[0]]//2 for o in orbits)
        self.local_mass_dim = sum(self.local_mass_dims)
        self.offset_dim = sum(self.level_dims)-1
        self.dimension = self.mass_dim+self.local_mass_dim+self.offset_dim
        self.groups = np.repeat(np.arange(4),self.counts)
        self.mask = parent_mask(self.groups)
        self.cycle = tuple(int(np.flatnonzero(self.groups==j)[0]) for j in range(4))
        self.parent_symmetry = bool(parent_symmetry)
        self.mass_by_occupancy = bool(mass_by_occupancy)

    def geometry(self, parameters):
        v = np.asarray(parameters,dtype=float)
        if v.shape!=(self.dimension,) or not np.all(np.isfinite(v)):
            raise ValueError("Invalid multisplit geometry vector")
        logits = np.r_[v[:self.mass_dim],0.]
        parent = np.array([np.exp(logits[next(i for i,o in enumerate(self.mass_orbits) if j in o)]-logits.max())
                           for j in range(4)])
        parent /= parent.sum()
        fractions,levels = {},{}
        cursor = self.mass_dim
        for orbit,n in zip(self.orbits,self.local_mass_dims):
            k = self.counts[orbit[0]]
            logs = np.r_[v[cursor:cursor+n],0.]
            cursor += n
            weights = np.exp(logs-logs.max())
            multiplicities = np.r_[np.full(k//2,2.),[1.] if k%2 else []]
            weights /= multiplicities@weights
            per_vertex = np.r_[weights[:k//2][::-1],weights[-1:] if k%2 else [],weights[:k//2]]
            for j in orbit:
                fractions[j] = per_vertex
        gaps = np.exp(np.r_[v[cursor:],0.])
        cursor = 0
        for orbit,n in zip(self.orbits,self.level_dims):
            positive = np.cumsum(gaps[cursor:cursor+n])
            cursor += n
            k = self.counts[orbit[0]]
            offsets = np.r_[-positive[::-1],[0.] if k%2 else [],positive]
            for j in orbit:
                levels[j] = offsets
        x = np.concatenate([parent[j]*fractions[j] for j in range(4)])
        s = normalize_offsets(x,np.concatenate([levels[j] for j in range(4)]))
        return LimitingCandidate(x,s,self.groups,self.mask.astype(float),self.mask,self.cycle,
                                 "c4/multisplit:"+",".join(map(str,self.counts)))

    def initial(self, rng=None, radius=.3):
        return np.zeros(self.dimension) if rng is None else rng.normal(0,radius,self.dimension)

    def encode(self, candidate):
        """Project mirrored C4 data to this family's explicitly imposed ties."""
        c=candidate
        if tuple(np.bincount(c.groups,minlength=4))!=self.counts:
            raise ValueError("Occupancies do not match")
        parent=np.array([c.x[c.groups==j].sum() for j in range(4)])
        parent_orbits=np.array([np.mean(parent[list(o)]) for o in self.mass_orbits])
        values=list(np.log(parent_orbits[:-1]/parent_orbits[-1]))
        gaps=[]
        for orbit in self.orbits:
            j=orbit[0]
            ids=np.flatnonzero(c.groups==j)
            ids=ids[np.argsort(c.s[ids])]
            k=len(ids)
            half=k//2
            positive_ids=ids[half+(k%2):]
            weights=np.r_[c.x[positive_ids],c.x[ids[half:half+1]] if k%2 else []]
            values.extend(np.log(weights[:-1]/weights[-1]))
            if half:
                levels=(c.s[positive_ids]-c.s[ids[:half]][::-1])/2
                gaps.extend(np.diff(np.r_[0.,levels]))
        gaps=np.array(gaps)
        if np.any(gaps<=0):
            raise ValueError("Need distinct positive mirrored offset levels")
        values.extend(np.log(gaps[:-1]/gaps[-1]))
        return np.array(values)
