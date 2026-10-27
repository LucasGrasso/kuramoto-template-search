from dataclasses import replace
import json

import numpy as np
import pytest

from kuramoto_template.asymptotic_search.families import s7
from kuramoto_template.asymptotic_search.joint_limit import JointLimitConfig
from kuramoto_template.asymptotic_search.joint_outer import (
    FAMILIES,JointOuterConfig,canonical_outer,exact_outer_key,make_outer_seed,
    repair_tree,run_joint_outer,
)


def small_config(**kwargs):
    return JointOuterConfig(proposals=4,sizes=(7,),families=("complete",),guard_scales=(.003,),
        targets=(.693,),joint_sdp_calls=4,finite_tree_calls=4,audit_calls=3,shortlist_size=2,
        h_fractions=(.1,),inner=JointLimitConfig(secondary=False),**kwargs)


@pytest.mark.integration
@pytest.mark.parametrize("cancel_poll",[2,3,10,12])
def test_proposal_solve_and_lift_boundaries_resume_without_repayment(tmp_path,cancel_poll):
    config=small_config()
    candidates=[s7(t) for t in (.03,.01,.003,.01)]
    polls=0

    def cancelled():
        nonlocal polls
        polls+=1
        return polls>=cancel_poll

    partial=run_joint_outer(config,output=tmp_path/"resumed",candidates=candidates,cancelled=cancelled)
    assert partial.state["stop_reason"]=="cancelled"
    result=run_joint_outer(config,output=tmp_path/"resumed",resume=True)
    fresh=run_joint_outer(config,candidates=candidates)
    assert result.call_counts==fresh.call_counts
    assert result.state["trace"]==fresh.state["trace"]
    assert result.best_limit_mu==fresh.best_limit_mu
    assert result.best_finite_mu==fresh.best_finite_mu
    assert result.state["phase"]=="done"
    again=run_joint_outer(config,output=tmp_path/"resumed",resume=True)
    assert again.call_counts==result.call_counts
    with pytest.raises(ValueError,match="Incompatible"):
        run_joint_outer(replace(config,seed=5),output=tmp_path/"resumed",resume=True)


@pytest.mark.integration
def test_exact_cache_and_disjoint_budgets(tmp_path):
    cfg=JointOuterConfig(proposals=2,sizes=(7,),families=("complete",),guard_scales=(.003,),
        joint_sdp_calls=1,finite_tree_calls=2,audit_calls=2,shortlist_size=1,h_fractions=(.1,))
    result=run_joint_outer(cfg,output=tmp_path/"cached",candidates=[s7(),s7()])
    assert result.state["cache_hits"]==1
    assert result.call_counts["joint_sdp"]==1
    assert result.call_counts["finite_tree_linear"]<=2
    assert result.call_counts["high_precision_audit"]<=2
    assert not result.call_counts["full_density_sdp"]
    assert result.state["archives"]["rigorously_certified"]==[]
    for index in result.state["archives"]["strict"]:
        assert (tmp_path/"cached"/"joint-results"/f"{index:06d}.json").exists()
    assert result.best_limit_mu==max(r["mu0"] for r in result.state["records"] if r["status"]=="joint_asymptotic_strict")


def test_family_generation_and_exact_label_cache_keep_tiny_changes():
    masks=set()
    for size in (7,8,9,10,12):
        for i,family in enumerate(FAMILIES):
            c,fixed=make_outer_seed(size,family,np.random.default_rng(30+i))
            assert c.m==size and c.x.min()>=1e-5
            assert len(repair_tree(c))==size-1
            assert abs(c.x@c.s)<1e-14
            assert c.x@(c.s*c.s)==pytest.approx(1)
            masks.add((size,c.mask_hash))
    assert len(masks)>10
    c=s7()
    fixed=np.full_like(c.p0,np.nan)
    a,fa,_=canonical_outer(c,fixed)
    order=np.array([1,0,3,2,5,4,6])
    inv=np.argsort(order)
    changed=replace(c,x=c.x[order],s=c.s[order],groups=c.groups[order],p0=c.p0[np.ix_(order,order)],
                    allowed_mask=c.allowed_mask[np.ix_(order,order)],backbone_cycle=tuple(inv[list(c.backbone_cycle)]))
    b,fb,_=canonical_outer(changed,fixed)
    cfg=JointLimitConfig()
    assert exact_outer_key(a,fa,repair_tree(a),cfg)==exact_outer_key(b,fb,repair_tree(b),cfg)
    tiny=a.s.copy()
    tiny[0]=np.nextafter(tiny[0],np.inf)
    moved=replace(a,s=tiny)
    assert exact_outer_key(a,fa,repair_tree(a),cfg)!=exact_outer_key(moved,fa,repair_tree(moved),cfg)


def test_transient_checkpoint_replace_retries_only_io(tmp_path,monkeypatch):
    from kuramoto_template.asymptotic_search import joint_outer

    original=joint_outer._write
    attempts=[]

    def transient(path,data):
        attempts.append(path)
        if len(attempts)==1:
            raise PermissionError("sharing violation")
        return original(path,data)

    monkeypatch.setattr(joint_outer,"_write",transient)
    monkeypatch.setattr(joint_outer,"sleep",lambda _:None)
    joint_outer._write_with_retry(tmp_path/"checkpoint.json",dict(counts=dict(joint_sdp=7)))
    assert len(attempts)==2
    assert json.loads((tmp_path/"checkpoint.json").read_text())["counts"]["joint_sdp"]==7


@pytest.mark.integration
def test_balanced_counts_and_weak_proposals_are_resolved_again():
    cfg=JointOuterConfig(proposals=65,joint_sdp_calls=65,finite_tree_calls=0,audit_calls=0,
                         shortlist_size=0,guided_probability=1.)
    result=run_joint_outer(cfg)
    proposals=[e for e in result.state["trace"] if e["kind"]=="outer_proposal"]
    counts={m:sum(e["size"]==m for e in proposals) for m in cfg.sizes}
    assert max(counts.values())-min(counts.values())<=1
    guided=[e for e in proposals if e["move"]["operation"]!="restart"]
    assert guided
    assert all(e["parent_id"] is not None for e in guided)
    assert result.call_counts["joint_sdp"]==len(result.state["records"])
