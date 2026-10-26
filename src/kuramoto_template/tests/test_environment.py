"""Gym API and SDP-budget task properties. Synthetic evaluators test plumbing,
while real-solver cases independently check the full integration boundary.
"""
import json
import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env
from kuramoto_template.environment import KuramotoEnv, EnvironmentConfig, ENV_ID
from kuramoto_template.environment.actions import decode_action, action_masks
from kuramoto_template.template import PhaseTemplate
from kuramoto_template.search.evaluator import EvaluationConfig, TemplateEvaluation
from kuramoto_template.search.random_search import twisted_seed
from kuramoto_template.experiments.cache import CachedEvaluator


class SyntheticEvaluator:
    """Known cost and landscape for reward/accounting, not physical stability."""
    config = EvaluationConfig(max_clusters=24)
    def __init__(self, calls=1):
        self.calls = calls
        self.caps = []
    def evaluate(self, template, *, max_solver_calls=None):
        self.caps.append(max_solver_calls)
        cost = min(self.calls, max_solver_calls) if max_solver_calls is not None else self.calls
        if cost < self.calls:
            return TemplateEvaluation(template,"uncertain",sdp_calls=cost)
        score = .5 + .1*np.sin(template.phi[0])
        return TemplateEvaluation(template,"stable",stable_mu=float(score),sdp_calls=cost)


def action(op=2, i=0, j=0, p=.5, q=0):
    return dict(operation=op,i=i,j=j,parameters=np.array([p,q],dtype=np.float32))


def env(config=None, **kwargs):
    return KuramotoEnv(config or EnvironmentConfig(max_sdp_calls=8),
                       evaluator=SyntheticEvaluator(), **kwargs)


def test_gymnasium_checker_and_registration():
    e=env(); check_env(e,skip_render_check=True); e.close()
    with gym.make(ENV_ID,evaluator=SyntheticEvaluator(),config=EnvironmentConfig(max_sdp_calls=4)) as e:
        observation, info=e.reset(seed=2)
        assert e.observation_space.contains(observation)
        assert info["total_sdp_calls"] == 1


def test_reward_telescopes_to_best_improvement_through_worse_states():
    e=env(); _,initial=e.reset(seed=3)
    rewards=[]
    for p in (.8,-.8,1):
        observation,reward,terminated,truncated,info=e.step(action(p=p))
        rewards.append(reward)
    assert rewards[0]>0 and rewards[1]==0 and rewards[2]>0
    assert sum(rewards)==pytest.approx(1000*(info["best_mu"]-initial["best_mu"]))
    assert e._record.template.phi[0]==pytest.approx(.25)
    e.close()


def test_reset_seed_and_one_call_fallback_count_toward_same_budget():
    evaluator=SyntheticEvaluator(calls=2)
    e=KuramotoEnv(EnvironmentConfig(max_sdp_calls=5),evaluator=evaluator)
    _,info=e.reset(seed=0)
    assert info["sdp_calls"]==2 and info["remaining_sdp_calls"]==3
    _,_,terminated,truncated,info=e.step(action())
    assert not terminated and not truncated and info["sdp_calls"]==2
    _,_,terminated,truncated,info=e.step(action())
    assert terminated and not truncated and info["sdp_calls"]==1
    assert evaluator.caps==[5,3,1]
    assert info["total_sdp_calls"]==5
    assert e.last_result.sdp_calls==sum(r.evaluation.sdp_calls for r in e.last_result.trace)
    with pytest.raises(ResetNeeded):e.step(action())


def test_repeat_states_pay_calls_without_hidden_exact_deduplication():
    e=env(EnvironmentConfig(max_sdp_calls=3)); e.reset(seed=0)
    first=e._record.template.phi.copy()
    _,_,done,_,info=e.step(action(p=0))
    assert info["sdp_calls"]==1 and info["disposition"]=="evaluated" and not done
    _,_,done,_,info=e.step(action(p=0))
    assert done and info["total_sdp_calls"]==3
    np.testing.assert_array_equal(first,e._record.template.phi)


def test_masked_actions_do_not_solve_and_zero_cost_guard_truncates():
    e=env(EnvironmentConfig(max_sdp_calls=8,max_zero_cost_steps=2));e.reset(seed=0)
    for expected in (False,True):
        observation,reward,done,truncated,info=e.step(action(i=23))
        assert reward==0 and not done and truncated==expected
        assert info["sdp_calls"]==0 and info["total_sdp_calls"]==1
        assert info["disposition"]=="rejected"
    assert info["stop_reason"]=="zero_cost_limit"


def test_step_limit_is_truncation_not_budget_terminal():
    e=env(EnvironmentConfig(max_sdp_calls=8,max_steps=1));e.reset(seed=0)
    _,_,done,truncated,info=e.step(action())
    assert not done and truncated and info["stop_reason"]=="step_limit"
    assert info["remaining_sdp_calls"]==6


def test_wall_time_exit_does_not_execute_or_charge_an_action():
    e=env(EnvironmentConfig(max_sdp_calls=8,max_wall_seconds=100));e.reset(seed=0)
    e._session.budget.clock=lambda:e._session.budget.started+101
    observation,reward,done,truncated,info=e.step(action())
    assert reward==0 and not done and truncated
    assert info["stop_reason"]=="wall_time" and info["sdp_calls"]==0 and info["steps"]==0


def test_step_contract_rejects_out_of_space_action_without_cost():
    e=env()
    with pytest.raises(ResetNeeded):e.step(action())
    e.reset(seed=0)
    with pytest.raises(ValueError):e.step(action(p=2))
    assert e._session.budget.sdp_calls==1 and e._steps==0
    e.close()
    with pytest.raises(ResetNeeded):e.sample_action()


def test_list_parameters_persist_and_lifecycle_keeps_episodes_separate(tmp_path):
    e=env(output=tmp_path/'episodes')
    e.reset(seed=0)
    a=action();a["parameters"]=[.5,0]
    with pytest.warns(UserWarning,match="Casting input"):
        e.step(a)
    e.reset(seed=1)
    e.step(action(op=0))
    e.close();e.close()
    roots=sorted((tmp_path/'episodes').iterdir())
    assert len(roots)==2
    for root in roots:
        result=json.loads((root/'result.json').read_text())
        rows=[json.loads(line) for line in (root/'transitions.jsonl').read_text().splitlines()]
        trace=[json.loads(line) for line in (root/'trace.jsonl').read_text().splitlines()]
        assert result["sdp_calls"]==trace[0]["evaluation"]["sdp_calls"]+sum(r["sdp_calls"] for r in rows)
        assert rows[0]["source_id"]==trace[1]["parent_id"]==0
        assert rows[0]["next_state_id"]==trace[1]["id"]
        assert isinstance(rows[0]["action"]["parameters"],list)
        assert result["stop_reason"] in ("reset","closed")
        events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
        assert events[0]["event"]=="reset"
        assert events[-1]["event"]=="episode_end" and events[-1]["truncated"]
        assert events[-1]["stop_reason"]==result["stop_reason"]


def test_rejected_action_successor_and_pre_action_wall_stop_are_logged(tmp_path):
    e=env(EnvironmentConfig(max_sdp_calls=8,max_wall_seconds=100),output=tmp_path/'run')
    e.reset(seed=0)
    e.step(action(i=23))
    e._session.budget.clock=lambda:e._session.budget.started+101
    e.step(action())
    root=tmp_path/'run'/'episode-000000'
    rows=[json.loads(line) for line in (root/'transitions.jsonl').read_text().splitlines()]
    assert len(rows)==1 and rows[0]["record_id"]==1 and rows[0]["next_state_id"]==0
    events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    assert events[-1]["stop_reason"]=="wall_time" and events[-1]["state_id"]==0
    assert events[-1]["total_sdp_calls"]==1


def test_seeded_action_sampler_and_observations_are_reproducible():
    e=env();f=env()
    oa,ia=e.reset(seed=123);ob,ib=f.reset(seed=123)
    for key in oa:np.testing.assert_array_equal(oa[key],ob[key])
    assert ia==ib
    for _ in range(4):
        aa=e.sample_action();ab=f.sample_action()
        for key in aa:np.testing.assert_array_equal(aa[key],ab[key])
        xa=e.step(aa);xb=f.step(ab)
        for key in xa[0]:np.testing.assert_array_equal(xa[0][key],xb[0][key])
        assert xa[1:]==xb[1:]
    e.close();f.close()


def test_node_edge_features_permute_and_relative_phases_ignore_rotation():
    t=PhaseTemplate([.1,.2,.3,.4],[0,.7,2.3,4.5]);order=[2,0,3,1]
    e=env(initial_template=t);f=env(initial_template=PhaseTemplate(t.x[order],t.phi[order]+.4))
    a,_=e.reset(seed=0);b,_=f.reset(seed=0)
    np.testing.assert_allclose(b["nodes"][:4],a["nodes"][order])
    np.testing.assert_allclose(b["edges"][:4,:4],a["edges"][np.ix_(order,order)],atol=2e-7)
    assert not a["node_mask"][4:].any() and not a["nodes"][4:].any()
    a["nodes"][0]=99
    assert e._record.template.x[0]==.1
    e.close();f.close()


@pytest.mark.parametrize("m",[3,5,24])
def test_masks_and_sampled_actions_respect_cluster_bounds(m):
    e=env(initial_template=twisted_seed(m));observation,_=e.reset(seed=2)
    assert bool(observation["operation_mask"][0])==(m<24)
    assert bool(observation["operation_mask"][1])==(m>3)
    for _ in range(30):
        a=e.sample_action();proposal=decode_action(a,e._record.template,e.config)
        assert e.action_space.contains(a) and proposal.template is not None
        assert 3<=proposal.template.n_clusters<=24
        assert proposal.template.x.sum()==pytest.approx(1)
        assert np.all(proposal.template.x>0)
    e.close()


def test_decode_split_and_resize_match_mass_conservation_equations():
    t=twisted_seed(5);c=EnvironmentConfig()
    split=decode_action(action(op=0,p=-1,q=1),t,c)
    np.testing.assert_allclose(split.template.x[:2],[.04,.16])
    assert split.parameters["ratio"]==pytest.approx(.2)
    assert split.parameters["separation"]==pytest.approx(.1)
    resize=decode_action(action(op=3,i=0,j=1,p=1),t,c)
    np.testing.assert_allclose(resize.template.x[:2],[.18,.22])


def test_antipodal_merge_is_masked():
    t=PhaseTemplate([.25,.25,.5],[0,np.pi,np.pi/2]);c=EnvironmentConfig(min_clusters=2,max_clusters=3)
    operations,nodes,pairs=action_masks(t,c)
    assert not pairs[1,0,1]
    assert decode_action(action(op=1,i=0,j=1),t,c).template is None


def test_reset_options_invalid_bounds_and_cached_evaluator():
    e=env()
    with pytest.raises(ValueError):e.reset(options={"unknown":True})
    with pytest.raises(TypeError):e.reset(options={"template":None})
    with pytest.raises(ValueError):e.reset(options={"template":twisted_seed(2)})
    with pytest.raises(ValueError):KuramotoEnv(evaluator=CachedEvaluator(SyntheticEvaluator()))
    with pytest.raises(ValueError):KuramotoEnv(evaluation_config=EvaluationConfig(max_clusters=12))
    e.close()


@pytest.mark.integration
def test_real_solver_gym_episode_includes_seed_and_stops_at_exact_budget():
    e=KuramotoEnv(EnvironmentConfig(max_sdp_calls=4),render_mode="ansi")
    observation,info=e.reset(seed=1)
    assert info["best_mu"]==pytest.approx(.65835918,abs=1e-6)
    assert e.observation_space.contains(observation)
    terminated=truncated=False
    while not(terminated or truncated):
        observation,reward,terminated,truncated,info=e.step(action(p=0))
    assert terminated and not truncated and info["total_sdp_calls"]==4
    assert "remaining_calls=0" in e.render()
