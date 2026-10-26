"""Policy properties derived from distributions, symmetries and task dynamics."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('gymnasium')
torch.set_num_threads(1)
from kuramoto_template.environment import KuramotoEnv,EnvironmentConfig
from kuramoto_template.environment.env import build_observation
from kuramoto_template.environment.actions import action_masks,decode_action
from kuramoto_template.search.evaluator import EvaluationConfig,TemplateEvaluation
from kuramoto_template.search.random_search import twisted_seed
from kuramoto_template.template import PhaseTemplate
from kuramoto_template.policy.config import PolicyConfig
from kuramoto_template.policy.network import ActorCritic
from kuramoto_template.policy.actions import masked_log_probs,NoValidAction,truncated_log_prob,truncated_sample
from kuramoto_template.policy.encoding import batch_observations,batch_actions,numpy_action,reflect
from kuramoto_template.policy.learner import Learner
from kuramoto_template.policy.replay import Replay,Transition
from kuramoto_template.policy.dataset import load_history,reconstruct,rescreen,key


def observation(t=None,remaining=8):
    t=t or twisted_seed(5)
    return build_observation(t,TemplateEvaluation(t,'uncertain'),EnvironmentConfig(),
                             best_mu=0,remaining_sdp_calls=remaining)


def action(i=0,op=2,p=0.,q=0.,j=0):
    return dict(operation=op,i=i,j=j,parameters=np.array([p,q],dtype=np.float32))


def model():
    torch.manual_seed(4)
    return ActorCritic(PolicyConfig(hidden=16,layers=2))


def test_masked_categorical_normalization_empty_and_single_action():
    logits=torch.tensor([[1.,2.,3.],[0.,5.,8.]])
    mask=torch.tensor([[1,0,1],[0,1,0]])
    lp=masked_log_probs(logits,mask)
    torch.testing.assert_close(lp.exp().sum(-1),torch.ones(2))
    assert lp[1,1]==0 and torch.isneginf(lp[0,1])
    with pytest.raises(NoValidAction):masked_log_probs(logits,torch.zeros_like(mask))


def test_truncated_gaussian_integrates_to_one_and_samples_need_no_clipping():
    x=torch.linspace(-1,1,20001)[:,None]
    loc=torch.tensor([-.7,.6]);scale=torch.tensor([.1,.45])
    density=truncated_log_prob(x,loc,scale).exp()
    torch.testing.assert_close(torch.trapezoid(density,x[:,0],dim=0),torch.ones(2),atol=2e-5,rtol=0)
    a=truncated_sample(loc.expand(1000,-1),scale.expand(1000,-1),torch.Generator().manual_seed(4))
    assert (a.abs()<=1).all() and torch.isfinite(truncated_log_prob(a,loc,scale)).all()


@pytest.mark.parametrize('m',[3,5,24])
def test_sampled_joint_likelihood_matches_recomputed_exact_action(m):
    network=model();o=observation(twisted_seed(m));obs=batch_observations([o]*12)
    h,c=network.encoder(obs);a,lp=network.actor.sample(h,c,obs,torch.Generator().manual_seed(m))
    torch.testing.assert_close(lp,network.actor.log_prob(h,c,obs,a),atol=2e-6,rtol=0)
    assert torch.isfinite(lp).all()
    for row in range(12):
        decoded=decode_action(numpy_action(a,row),twisted_seed(m),EnvironmentConfig())
        assert decoded.template is not None
        assert np.all(decoded.template.x>0) and decoded.template.x.sum()==pytest.approx(1)


def test_rotation_and_permutation_policy_equivariance_including_unordered_merge():
    t=PhaseTemplate([.1,.2,.3,.4],[0,.7,2.3,4.5]);order=np.array([2,0,3,1]);inverse=np.argsort(order)
    network=model();base=observation(t);other=observation(PhaseTemplate(t.x[order],t.phi[order]+.31))
    # Crop inactive padding for an exact arbitrary permutation of active nodes.
    for o in (base,other):
        o['nodes']=o['nodes'][:4];o['node_mask']=o['node_mask'][:4]
        o['edges']=o['edges'][:4,:4];o['pair_mask']=o['pair_mask'][:,:4,:4]
    a=batch_observations([base]);b=batch_observations([other]);ha,ca=network.encoder(a);hb,cb=network.encoder(b)
    torch.testing.assert_close(hb,ha[:,order],atol=3e-6,rtol=0)
    torch.testing.assert_close(cb,ca,atol=3e-6,rtol=0)
    oa,na,pa=network.actor.logits(ha,ca,a);ob,nb,pb=network.actor.logits(hb,cb,b)
    torch.testing.assert_close(oa,ob,atol=3e-6,rtol=0)
    torch.testing.assert_close(nb,na[:,:,order],atol=3e-6,rtol=0)
    for op in (1,3):torch.testing.assert_close(pb[:,op],pa[:,op][:,order][:,:,order],atol=3e-6,rtol=0)
    for op in (0,1,2,3):
        original=action(i=0,op=op,j=1,p=.2,q=.3)
        transformed=action(i=int(inverse[0]),op=op,j=int(inverse[1]) if op in (1,3) else 0,p=.2,q=.3)
        if op==1:transformed['i'],transformed['j']=sorted((transformed['i'],transformed['j']))
        lpa=network.actor.log_prob(ha,ca,a,batch_actions([original]))
        lpb=network.actor.log_prob(hb,cb,b,batch_actions([transformed]))
        torch.testing.assert_close(lpa,lpb,atol=3e-6,rtol=0)


def test_padding_does_not_change_unpadded_outputs():
    network=model();a=observation();b=deepcopy(a)
    for k in ('nodes','node_mask'):b[k]=np.pad(b[k],[(0,3)]+[(0,0)]*(b[k].ndim-1))
    b['edges']=np.pad(b['edges'],[(0,3),(0,3),(0,0)])
    b['pair_mask']=np.pad(b['pair_mask'],[(0,0),(0,3),(0,3)])
    ha,ca=network.encoder(batch_observations([a]));hb,cb=network.encoder(batch_observations([b]))
    torch.testing.assert_close(ca,cb,atol=3e-6,rtol=0)
    torch.testing.assert_close(ha,hb[:,:ha.shape[1]],atol=3e-6,rtol=0)
    assert not hb[:,ha.shape[1]:].any()


@pytest.mark.parametrize('op',[0,1,2,3])
def test_reflection_augmentation_transforms_actual_mutation(op):
    t=twisted_seed(5);a=action(op=op,j=1,p=.2,q=.4)
    _,ra=reflect(observation(t),a)
    rt=PhaseTemplate(t.x,-t.phi)
    child=decode_action(a,t,EnvironmentConfig()).template
    reflected=decode_action(ra,rt,EnvironmentConfig()).template
    np.testing.assert_allclose(child.x,reflected.x,atol=1e-14)
    assert np.max(abs(np.angle(np.exp(1j*(child.phi+reflected.phi)))))<1e-12
    if op==0:
        with pytest.raises(ValueError):reflect(observation(t),action(op=0,q=0))


def transition(remaining=2,reward=0,terminal=False):
    o=observation(remaining=remaining);n=observation(remaining=remaining-1)
    return Transition(o,action(),reward,n,terminal,False,1,'test')


def test_replay_validation_boundaries_and_checkpoint_rng(tmp_path):
    replay=Replay(seed=9)
    for i in range(4):replay.add(transition(reward=i))
    p=tmp_path/'replay.pt';torch.save(replay.state_dict(),p)
    expected=[t.reward for t in replay.sample(20)]
    restored=Replay();restored.load_state_dict(torch.load(p,weights_only=True))
    assert [t.reward for t in restored.sample(20)]==expected
    with pytest.raises(TypeError):replay.add(SimpleNamespace(observation=observation()))
    bad=transition();bad.next_observation['global_features'][4]=0
    with pytest.raises(ValueError):replay.add(bad)
    assert not Transition({}, {},0,{},False,True,0,'end').continuation


def test_finite_awac_unknown_auxiliary_labels_and_exact_resume(tmp_path):
    learner=Learner(PolicyConfig(hidden=16,auxiliary_weight=1),seed=3)
    replay=Replay(seed=4)
    for _ in range(5):replay.add(transition())
    metrics=learner.update(replay.sample(4))
    assert all(np.isfinite(x) for x in metrics.values()) and metrics['auxiliary_loss']==0
    assert metrics['weight_max']<=learner.config.max_weight
    path=tmp_path/'learner.pt';torch.save(dict(learner=learner.state_dict(),replay=replay.state_dict()),path)
    expected=learner.update(replay.sample(4));parameters=[p.detach().clone() for p in learner.model.parameters()]
    saved=torch.load(path,weights_only=True);other=Learner(learner.config,seed=99);other.load_state_dict(saved['learner'])
    restored=Replay();restored.load_state_dict(saved['replay']);actual=other.update(restored.sample(4))
    assert actual==expected
    for x,y in zip(parameters,other.model.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)


def test_neither_terminal_nor_truncation_bootstraps_through_boundary():
    learner=Learner(PolicyConfig(hidden=16),seed=3)
    for critic in (learner.target.critics.q1,learner.target.critics.q2):
        critic[-1].bias.data.fill_(10000)
    a=transition(remaining=1,reward=2,terminal=True)
    b=transition(reward=3);b.truncated=True
    for t in (a,b):t.next_observation['operation_mask'][:]=0
    metrics=learner.update([a,b])
    assert metrics['target_max']==3


def test_bc_overfits_tiny_data_and_trains_encoder():
    learner=Learner(PolicyConfig(hidden=16,learning_rate=.005),seed=2)
    o=observation(PhaseTemplate([.3,.7],[0,2.]));a=action(i=0,p=.3)
    samples=[SimpleNamespace(observation=o,action=a,weight=1.)]*8
    before=learner.bc_update(samples)['bc_log_prob'];encoder=[p.detach().clone() for p in learner.model.encoder.parameters()]
    for _ in range(70):after=learner.bc_update(samples)['bc_log_prob']
    assert after>before+1
    assert any(not torch.equal(x,y) for x,y in zip(encoder,learner.model.encoder.parameters()))


def test_two_step_mdp_learns_initially_worse_action():
    # Unequal node masses make alternatives distinguishable equivariantly.
    # Node0: zero immediate gain then reward3. Node1: greedy reward1 and end.
    learner=Learner(PolicyConfig(hidden=16,layers=1,learning_rate=.003,tau=.05,temperature=.3),seed=12)
    t=PhaseTemplate([.3,.7],[0,2.]);start=observation(t,2);middle=observation(t,1);end=observation(t,0)
    for o in (start,middle,end):o['operation_mask'][:]=[0,0,1,0]
    middle['global_features'][1]=-.5
    replay=Replay(seed=1)
    replay.add(Transition(start,action(i=0),0,middle,False,False,1,'good'))
    replay.add(Transition(start,action(i=1),1,end,True,False,2,'greedy'))
    replay.add(Transition(middle,action(i=0),3,end,True,False,1,'good'))
    replay.add(Transition(middle,action(i=1),3,end,True,False,1,'good'))
    for _ in range(450):learner.update(replay.sample(16))
    obs=batch_observations([start]);h,c=learner.model.encoder(obs)
    _,scores,_=learner.model.actor.logits(h,c,obs)
    probs=masked_log_probs(scores[:,2],obs['node_mask']).exp()[0,:2]
    assert probs[0]>probs[1]+.2
    qgood=min(q.item() for q in learner.model.critics(h,c,obs,batch_actions([action(i=0)])))
    qgreedy=min(q.item() for q in learner.model.critics(h,c,obs,batch_actions([action(i=1)])))
    assert qgood>qgreedy+.3


def write_branch(root,family='five_twisted'):
    root.mkdir(parents=True)
    t=twisted_seed(5);cfg=EnvironmentConfig(max_sdp_calls=8)
    parent=dict(id=0,parent_id=None,operation='seed',parameters={},template=dict(x=t.x.tolist(),phi=t.phi.tolist()),evaluation=None)
    rows=[parent]
    for i in (0,1):
        a=action(i=i,p=.2)
        proposal=decode_action(a,t,cfg)
        rows.append(dict(id=i+1,parent_id=0,operation=proposal.operation,parameters=proposal.parameters,
            template=dict(x=proposal.template.x.tolist(),phi=proposal.template.phi.tolist()),evaluation=None))
    (root/'trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    (root/'metadata.json').write_text(json.dumps(dict(run_id=root.name,search_config=asdict(cfg),evaluation_config=asdict(EvaluationConfig(max_clusters=24)))))
    (root/'benchmark.json').write_text(json.dumps(dict(spec=dict(initial=family))))
    return root


def test_historical_siblings_are_bc_only_and_canonical_leakage_is_removed(tmp_path):
    train=write_branch(tmp_path/'train')
    dataset=load_history([train],train_families=['five_twisted'],validation_fraction=0)
    assert len(dataset.actions)==2 and not dataset.transitions
    assert dataset.actions[0].observation['node_mask'].sum()==5
    assert dataset.manifest['branch_context'].endswith('never TD')
    heldout=write_branch(tmp_path/'test','eight_twisted')
    filtered=load_history([train,heldout],train_families=['five_twisted'],held_out_families=['eight_twisted'],validation_fraction=0)
    assert not filtered.actions and filtered.manifest['canonical_overlap']==0
    assert filtered.manifest['counts']['canonical_overlap_removed']==2


def test_historical_gaussian_scale_is_not_mistaken_for_gym_displacement_bound(tmp_path):
    root=write_branch(tmp_path/'legacy')
    path=root/'metadata.json';meta=json.loads(path.read_text())
    meta['search_config']['phase_scale']=.05  # old sampler's std, not decoder bound
    path.write_text(json.dumps(meta))
    dataset=load_history([root],train_families=['five_twisted'],validation_fraction=0)
    assert dataset.actions[0].action['parameters'][0]==pytest.approx(.2)


def test_historical_stable_label_with_missing_or_cached_matrix_is_not_accepted():
    t=twisted_seed(5);cfg=EvaluationConfig(max_clusters=24)
    assert rescreen(t,dict(status='stable',stable_mu=.99),cfg).stable_mu is None
    assert rescreen(t,dict(status='stable',cached=True,stable_mu=.99),cfg).stable_mu is None


def test_uncertain_numeric_exports_preserve_null_diagnostics_without_nan_json(tmp_path):
    from kuramoto_template.policy.train import write_json
    path=tmp_path/'uncertain.json'
    write_json(path,dict(status='uncertain',P=np.array([[np.nan]]),residual=float('inf')))
    assert json.loads(path.read_text())==dict(status='uncertain',P=[[None]],residual=None)


def test_reconstructed_historical_prune_uses_explicit_source_and_unsupported_undo_is_excluded():
    t=twisted_seed(5);cfg=EnvironmentConfig();proposal=decode_action(action(op=1,i=0,j=1),t,cfg)
    record=dict(operation='prune_merge',parameters=dict(i=0,j=1,source_id=0),template=dict(x=proposal.template.x,phi=proposal.template.phi))
    assert reconstruct(t,record,cfg)['operation']==1
    with pytest.raises(ValueError):reconstruct(t,{**record,'operation':'prune_undo_split'},cfg)


@pytest.mark.integration
def test_clean_environment_episode_reconstructs_td_without_new_solves(tmp_path):
    root=tmp_path/'serial'
    with KuramotoEnv(EnvironmentConfig(max_sdp_calls=5,max_steps=50),output=root) as e:
        o,info=e.reset(seed=3);done=False
        while not done:
            _,_,term,trunc,_=e.step(action());done=term or trunc
    episode=root/'episode-000000'
    (episode/'policy_episode.json').write_text(json.dumps(dict(family='five_twisted')))
    dataset=load_history([root],train_families=['five_twisted'],validation_fraction=0)
    assert dataset.transitions and dataset.transitions[-1].terminated
    assert sum(t.sdp_calls for t in dataset.transitions)+info['sdp_calls']==5
    assert dataset.manifest['historical_source_sdp_calls']==5
    metadata=episode/'metadata.json';saved=json.loads(metadata.read_text())
    changed=deepcopy(saved);changed['search_config']['reward_scale']=2000
    metadata.write_text(json.dumps(changed))
    assert not load_history([root],train_families=['five_twisted'],validation_fraction=0).transitions
    metadata.write_text(json.dumps(saved))
    # Missing full sequential context never silently falls back to adjacent rows.
    (episode/'events.jsonl').unlink()
    assert not load_history([root],train_families=['five_twisted'],validation_fraction=0).transitions


@pytest.mark.integration
def test_frozen_search_is_deterministic_and_does_not_consume_policy_rng():
    from kuramoto_template.search.policy_search import run_policy_search
    network=model().eval();state=torch.get_rng_state().clone()
    cfg=EnvironmentConfig(max_sdp_calls=5)
    a=run_policy_search(twisted_seed(5),network,seed=17,config=cfg)
    b=run_policy_search(twisted_seed(5),network,seed=17,config=cfg)
    assert torch.equal(state,torch.get_rng_state())
    assert a.search.sdp_calls==b.search.sdp_calls==5
    assert [(r.operation,r.parameters) for r in a.search.trace]==[(r.operation,r.parameters) for r in b.search.trace]
    for x,y in zip(a.search.trace,b.search.trace):
        if x.template is not None:np.testing.assert_array_equal(x.template.phi,y.template.phi)


@pytest.mark.integration
def test_training_checkpoint_resumes_at_episode_boundary_with_same_updates(tmp_path):
    from kuramoto_template.policy.train import main
    common=['--episode-calls','4','--warmup-calls','3','--bc-steps','1','--batch-size','2',
            '--max-updates','2','--hidden','16','--layers','1','--families','five_twisted']
    first=tmp_path/'first';resumed=tmp_path/'resumed';whole=tmp_path/'whole'
    main(['--output',str(first),'--training-calls','8',*common])
    main(['--output',str(resumed),'--training-calls','12','--resume',str(first/'checkpoint.pt'),*common])
    main(['--output',str(whole),'--training-calls','12',*common])
    a=torch.load(resumed/'checkpoint.pt',weights_only=True);b=torch.load(whole/'checkpoint.pt',weights_only=True)
    assert a['state']['training_calls']==b['state']['training_calls']==12
    assert a['learner']['updates']==b['learner']['updates']
    for name,value in a['learner']['model'].items():torch.testing.assert_close(value,b['learner']['model'][name],atol=0,rtol=0)
    with pytest.raises(ValueError,match='already-spent'):
        main(['--output',str(tmp_path/'bad-cap'),'--training-calls','3','--resume',str(first/'checkpoint.pt'),*common])
    with pytest.raises(ValueError,match='settings differ'):
        main(['--output',str(tmp_path/'bad-config'),'--training-calls','12','--resume',str(first/'checkpoint.pt'),*common,'--random-fraction','.3'])
