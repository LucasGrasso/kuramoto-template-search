from dataclasses import replace

import numpy as np
import pytest

from kuramoto_template.asymptotic_search.families import parent_mask, s7, split_candidate, winding
from kuramoto_template.asymptotic_search.models import LimitingCandidate
from kuramoto_template.asymptotic_search.stage1 import screen_candidate


def test_winding_uses_actual_edges_and_rejects_branch_boundaries():
    c = s7()
    assert winding(c,c.p0,.01)
    missing = c.p0.copy()
    missing[0,2] = missing[2,0] = 0
    assert not winding(c,missing,.01)
    offsets = c.s.copy()
    offsets[2] = offsets[0]+np.pi/(2*.01)
    assert not winding(replace(c,s=offsets),c.p0,.01)


def test_arrays_and_invalid_inputs():
    c = s7()
    with pytest.raises(ValueError):
        c.x[0] = 1
    with pytest.raises(ValueError):
        replace(c,x=c.x*2)
    with pytest.raises(ValueError):
        replace(c,groups=np.zeros(7))
    with pytest.raises(ValueError):
        replace(c,allowed_mask=np.ones((7,7),bool))
    with pytest.raises(ValueError):
        split_candidate(c,0,.5,.1,min_mass=.2)


def test_degeneracies_bad_dipole_and_singleton_hard_block():
    c = s7()
    assert screen_candidate(replace(c,s=np.zeros(7))).status=="asymptotic_degenerate"
    p = c.p0.copy()
    p[0,1] = p[1,0] = 0
    assert screen_candidate(replace(c,p0=p)).status=="asymptotic_degenerate"
    # q above the published critical point is a robust negative control.
    assert screen_candidate(s7(-.01)).status=="screen_rejected"
    assert screen_candidate(s7(0)).status=="uncertain"
    groups = np.arange(4)
    c4 = LimitingCandidate(np.ones(4)/4,[-1,1,-1,1],groups,parent_mask(groups).astype(float),
                          parent_mask(groups),(0,1,2,3))
    assert screen_candidate(c4).status=="asymptotic_degenerate"
