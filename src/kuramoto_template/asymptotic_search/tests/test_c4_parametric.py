import numpy as np
import pytest

from kuramoto_template.asymptotic_search.c4_discovery import reflection_geometry
from kuramoto_template.asymptotic_search.c4_multisplit import MirrorFamily, c4_partitions, partition_orbit
from kuramoto_template.asymptotic_search.c4_parametric import ParametricJoint
from kuramoto_template.asymptotic_search.c4_recovered import RecoveredJoint
from kuramoto_template.asymptotic_search.joint_limit import JointLimitConfig,solve_joint_limit
from kuramoto_template.asymptotic_search.joint_outer import repair_tree
from kuramoto_template.asymptotic_search.solvers import SolverLedger


def test_multisplit_masses_offsets_and_c4_orbits():
    rng=np.random.default_rng(52)
    partitions=c4_partitions()
    assert len(partitions)==len(set(partitions))
    assert (1,2,2,2) in partitions and (4,4,4,4) in partitions
    for counts in partitions:
        assert partition_orbit(counts)==counts
        family=MirrorFamily(counts)
        c=family.geometry(family.initial(rng))
        assert c.x.min()>0 and c.x.sum()==pytest.approx(1)
        assert c.x@c.s==pytest.approx(0,abs=1e-14)
        assert c.x@(c.s*c.s)==pytest.approx(1)
        for j in range(4):
            ids=c.groups==j
            assert c.s[ids]==pytest.approx(-c.s[ids][::-1])
            assert c.x[ids]==pytest.approx(c.x[ids][::-1])
        assert np.all(c.p0[~c.allowed_mask]==0)
        back=family.geometry(family.encode(c))
        assert back.x==pytest.approx(c.x)
        assert back.s==pytest.approx(c.s)


def test_equal_occupancy_mass_ablation_is_explicit():
    family=MirrorFamily((1,2,2,2),mass_by_occupancy=True)
    assert family.dimension==2
    c=family.geometry([-.2,-.3])
    parent=[c.x[c.groups==j].sum() for j in range(4)]
    assert parent[1:]==pytest.approx([parent[1]]*3)


@pytest.mark.integration
def test_parametric_solver_matches_original_and_counts_each_call():
    cfg=JointLimitConfig(secondary=False,gamma0=1e-6)
    family=MirrorFamily((1,2,2,2))
    v=np.array([.1,.2,-.2])
    c=family.geometry(v)
    ledger=SolverLedger()
    reusable=ParametricJoint(c,cfg,ledger=ledger)
    for shift in (0.,.005):
        c=family.geometry(v+shift)
        fast=reusable.run(c)
        reference=solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,cfg,
            backbone_cycle=c.backbone_cycle,repair_edges=repair_tree(c))
        assert fast.mu0==pytest.approx(reference.mu0,abs=2e-9)
        assert fast.diagnostics['independent_primal_check_inf']<=cfg.residual_tolerance
        assert fast.diagnostics['affine_error']<1e-12
    assert ledger.counts['joint_sdp']==2
    assert len(ledger.events)==2


@pytest.mark.integration
def test_dual_envelope_direction_agrees_with_reoptimized_local_difference():
    cfg=JointLimitConfig(secondary=False,gamma0=1e-5)
    family=MirrorFamily((1,2,2,2))
    v=np.array([.12,.23,-.3])
    reusable=ParametricJoint(family.geometry(v),cfg)
    center=reusable.run(family.geometry(v))
    assert center.mu0 is not None
    gradient=reusable.gradient(family.geometry,v)
    slow=[]
    for i in range(len(v)):
        a,b=v.copy(),v.copy()
        a[i]-=1e-5;b[i]+=1e-5
        slow.append((reusable.envelope(family.geometry(b))-reusable.envelope(family.geometry(a)))/2e-5)
    assert gradient==pytest.approx(slow,abs=1e-7)
    direction=np.array([.2,-.3,.1])
    eps=1e-5
    left=reusable.run(family.geometry(v-eps*direction))
    right=reusable.run(family.geometry(v+eps*direction))
    actual=(right.mu0-left.mu0)/(2*eps)
    assert gradient@direction==pytest.approx(actual,abs=2e-4)


@pytest.mark.integration
def test_free_cut_model_requires_and_checks_actual_correction_recovery():
    cfg=JointLimitConfig(secondary=False,gamma0=1e-5,box_buffer=1e-7)
    family=MirrorFamily((1,2,2,2))
    v=np.array([.12,.23,-.3])
    ledger=SolverLedger()
    engine=RecoveredJoint(family.geometry(v),cfg,ledger=ledger)
    result=engine.run(family.geometry(v))
    assert result.mu0 is not None and result.candidate is not None
    assert result.diagnostics['independent_primal_check_inf']<1e-9
    assert result.solve_records[0]['recovery']['residual']<1e-10
    assert ledger.counts=={'joint_sdp':1,'correction_linear':1}
    assert result.repair_edges==() # No inward tree was imposed by this relaxation.
    gradient=engine.gradient(family.geometry,v)
    slow=[]
    for i in range(len(v)):
        a,b=v.copy(),v.copy();a[i]-=1e-5;b[i]+=1e-5
        slow.append((engine.envelope(family.geometry(b))-engine.envelope(family.geometry(a)))/2e-5)
    assert gradient==pytest.approx(slow,abs=1e-7)


@pytest.mark.integration
@pytest.mark.parametrize('engine_type',[ParametricJoint,RecoveredJoint])
def test_margin_mode_retains_unstable_search_directions_and_target(engine_type):
    cfg=JointLimitConfig(mode='margin',mu_target=.70,secondary=False)
    family=MirrorFamily((1,2,2,2))
    v=np.array([.12,.23,-.3]) # A regular point, away from equal-parent kinks.
    engine=engine_type(family.geometry(v),cfg)
    result=engine.run(family.geometry(v))
    assert result.mu0>=cfg.mu_target-1e-9
    assert result.reported_objective<0
    assert result.status=='joint_unstable'
    assert result.reported_objective==pytest.approx(result.diagnostics['full_gap'],abs=1e-9)
    g=engine.gradient(family.geometry,v)
    direction=np.array([.1,-.2,.3]);eps=1e-5
    left=engine.run(family.geometry(v-eps*direction))
    right=engine.run(family.geometry(v+eps*direction))
    actual=(right.reported_objective-left.reported_objective)/(2*eps)
    assert g@direction==pytest.approx(actual,abs=3e-4)


@pytest.mark.integration
@pytest.mark.parametrize('engine_type',[ParametricJoint,RecoveredJoint])
def test_mass_whitening_preserves_generalized_margin_and_envelope(engine_type):
    cfg=JointLimitConfig(mode='margin',mu_target=.70,secondary=False,margin_metric='mass')
    family=MirrorFamily((1,2,2,2));v=np.array([.12,.23,-.3])
    engine=engine_type(family.geometry(v),cfg)
    result=engine.run(family.geometry(v))
    assert result.reported_objective==pytest.approx(result.diagnostics['physical_gap'],abs=1e-8)
    g=engine.gradient(family.geometry,v)
    direction=np.array([.1,-.2,.3]);eps=1e-5
    left=engine.run(family.geometry(v-eps*direction));right=engine.run(family.geometry(v+eps*direction))
    actual=(right.reported_objective-left.reported_objective)/(2*eps)
    assert g@direction==pytest.approx(actual,abs=5e-4)


@pytest.mark.integration
def test_mass_whitened_density_matches_original_conic_model():
    cfg=JointLimitConfig(secondary=False,margin_metric='mass',gamma0=1e-6)
    family=MirrorFamily((1,2,2,2));c=family.geometry([.12,.23,-.3])
    result=ParametricJoint(c,cfg).run(c)
    original=solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,cfg,
        backbone_cycle=c.backbone_cycle,repair_edges=repair_tree(c))
    assert result.mu0==pytest.approx(original.mu0,abs=2e-9)
    assert result.diagnostics['physical_gap']>=.95*cfg.gamma0


@pytest.mark.integration
def test_elastic_target_records_shortfall_and_has_correct_merit_gradient():
    cfg=JointLimitConfig(mode='margin',mu_target=.72,secondary=False)
    family=MirrorFamily((1,2,2,2));v=np.array([.12,.23,-.3])
    engine=RecoveredJoint(family.geometry(v),cfg,elastic_penalty=3.)
    result=engine.run(family.geometry(v))
    assert result.candidate is not None
    assert result.diagnostics['target_shortfall']>=0
    assert result.diagnostics['requested_target']==cfg.mu_target
    assert result.diagnostics['search_merit']==pytest.approx(
        result.reported_objective-3*result.diagnostics['target_shortfall'],abs=1e-9)
    assert result.mu0>=result.config.mu_target-1e-9
    g=engine.gradient(family.geometry,v)
    direction=np.array([.1,-.2,.3]);eps=1e-5
    a=engine.run(family.geometry(v-eps*direction));b=engine.run(family.geometry(v+eps*direction))
    actual=(b.diagnostics['search_merit']-a.diagnostics['search_merit'])/(2*eps)
    assert g@direction==pytest.approx(actual,abs=5e-4)
