"""Regressions derived from mathematical obstructions and audited data flow.

Synthetic evaluators below test search plumbing only, not physical stability.
"""
from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest

from kuramoto_template.template import PhaseTemplate
from kuramoto_template.mutations import merge_clusters, perturb_phase, split_cluster
from kuramoto_template.search.evaluator import EvaluationConfig, SolverDiagnostics, TemplateEvaluation, TemplateEvaluator
from kuramoto_template.search.infrastructure import MutationProposal, SearchRecord, SearchResult
from kuramoto_template.search.random_search import twisted_seed
from kuramoto_template.search import adaptive, amcs
from kuramoto_template.sdp_optimizer.validate_input import circular_span, in_open_semicircle, validate_input
from kuramoto_template.sdp_optimizer.sdp import optimize_template
from kuramoto_template.sdp_optimizer.stability_sdp import maximize_stability_margin
from kuramoto_template.experiments.cache import CachedEvaluator, canonical_key
from kuramoto_template.experiments.metrics import compute_run_metrics, summarize_runs, RunMetrics
from kuramoto_template.experiments.refinement import RefinementConfig, refine_candidate


@pytest.mark.parametrize("offset", [0, .3, 2*np.pi-1e-7])
def test_circular_spread_and_validity_do_not_depend_on_first_cluster(offset):
    phi = np.array([0, .75e-8, 1.5e-8]) + offset
    for order in permutations(range(3)):
        phases = phi[list(order)]
        assert circular_span(phases) == pytest.approx(1.5e-8, abs=3e-15)
        validate_input(np.ones(3)/3, phases)
        assert in_open_semicircle(phases)
    for order in permutations(range(3)):
        with pytest.raises(ValueError, match="synchronized"):
            validate_input(np.ones(3)/3, (np.array([0, .25e-8, .5e-8])+offset)[list(order)])


@pytest.mark.parametrize("phi", [[0,1e-7,2e-7], [0,2*np.pi-1e-7,1e-7]])
def test_open_semicircle_cannot_receive_stable_density(phi):
    # Extreme-phase forces have one sign: exact equilibrium requires zero
    # interphase edges and hence extra Hessian zero modes. No metric helper
    # or solver status is an oracle for this analytic impossibility.
    x = np.ones(3)/3
    density = optimize_template(x, phi)
    assert not density.numerically_feasible
    margin = maximize_stability_margin(x, phi, .3)
    assert not margin.numerically_feasible
    evaluation = TemplateEvaluator().evaluate(PhaseTemplate(x, phi))
    assert evaluation.status != "stable"
    assert evaluation.stable_mu is None
    assert evaluation.sdp_calls == 2
    assert "semicircle" in evaluation.density.message


def test_closed_semicircle_boundary_not_rejected_as_open():
    for phi in ([0,np.pi], [.3,.3+np.pi], [0,np.pi/2,np.pi]):
        assert not in_open_semicircle(phi)


def test_direct_density_api_has_absolute_mass_tolerance():
    with pytest.raises(ValueError, match="sum to one"):
        optimize_template([.5,.500001], [0,np.pi])


class MatrixEvaluator:
    """Distinct matrix entries expose an incorrect permutation immediately."""
    config = EvaluationConfig()
    def __init__(self):
        self.calls = 0
    def evaluate(self, template, **kwargs):
        self.calls += 1
        m = template.n_clusters
        P = np.arange(m*m, dtype=float).reshape(m,m)
        P = (P+P.T)/100
        diagnostic = SolverDiagnostics("density", "optimal", P=P)
        return TemplateEvaluation(template, "uncertain", density=diagnostic, sdp_calls=1)


def test_cache_remaps_matrix_to_requested_order_and_isolates_returned_arrays():
    template = PhaseTemplate([.1,.2,.3,.4], [0,.7,2.3,4.5])
    permutation = [2,0,3,1]
    other = PhaseTemplate(template.x[permutation], template.phi[permutation]+.4)
    inner = MatrixEvaluator(); cache = CachedEvaluator(inner)
    first = cache.evaluate(template)
    hit = cache.evaluate(other)
    assert hit.cached and hit.sdp_calls == 0 and inner.calls == 1
    expected = first.P[np.ix_(permutation,permutation)]
    np.testing.assert_array_equal(hit.P, expected)
    hit.P[:] = -99
    np.testing.assert_array_equal(cache.evaluate(other).P, expected)


def test_cache_recovers_rotation_at_decimal_boundary_without_solver_call():
    first = PhaseTemplate([.3,.7], [0,.1234567890125])
    other = PhaseTemplate(first.x, first.phi+.009999999999935838)
    inner = MatrixEvaluator(); cache = CachedEvaluator(inner)
    cache.evaluate(first)
    assert cache.evaluate(other).cached
    assert inner.calls == 1


def test_cache_does_not_replace_invalid_input_after_quantization_collision():
    cache = CachedEvaluator(TemplateEvaluator(), decimals=0)
    valid = PhaseTemplate(np.ones(3)/3, [0,1e-7,2e-7])
    invalid = PhaseTemplate(np.ones(3)/3, [0,1e-10,2e-10])
    assert canonical_key(valid,decimals=0) == canonical_key(invalid,decimals=0)
    cache.evaluate(valid)
    answer = cache.evaluate(invalid)
    assert answer.status == "invalid" and answer.sdp_calls == 0 and not answer.cached


@pytest.mark.parametrize("reverse", [False,True])
def test_cache_permutation_order_cannot_create_stable_near_synchrony(reverse):
    a = PhaseTemplate(np.ones(3)/3, [0,.75e-8,1.5e-8])
    b = PhaseTemplate(a.x[[1,0,2]],a.phi[[1,0,2]])
    cache = CachedEvaluator(TemplateEvaluator())
    results = [cache.evaluate(t) for t in ((b,a) if reverse else (a,b))]
    assert all(e.stable_mu is None for e in results)
    assert results[0].status == results[1].status


class CountEvaluator:
    """One paid invocation with a cluster-count-dependent synthetic score."""
    config = EvaluationConfig()
    def evaluate(self, template, **kwargs):
        return TemplateEvaluation(template,"stable",stable_mu=template.n_clusters/10,sdp_calls=1)


@pytest.mark.parametrize("pruning", [0,1])
def test_adaptive_keeps_last_call_restart_and_counts_root_improvement(monkeypatch, pruning):
    monkeypatch.setattr(adaptive,"restart_proposal",lambda rng,config: MutationProposal("restart",{},twisted_seed(6)))
    result = adaptive.run_adaptive_search([twisted_seed(5)],CountEvaluator(),adaptive.AdaptiveSearchConfig(
        iterations=1,level=0,depth=1,restart_probability=1,prune_probability=pruning,
        max_sdp_calls=2))
    assert result.best.id == 1
    assert any(r.id == 1 for r in result.archives.stable)
    assert len(result.episodes) == 1
    episode = result.episodes[0]
    assert episode.sdp_calls == 1 and episode.evaluations == 1
    assert episode.improved and episode.incumbent_mu_before == .5
    assert episode.incumbent_mu_after == .6 and not episode.complete
    assert sum(e.sdp_calls for e in result.episodes)+1 == result.sdp_calls


def test_pruning_records_actual_source_and_keeps_episode_selected_root(monkeypatch):
    monkeypatch.setattr(adaptive,"_choose_adaptive_root",lambda ctx,config,archives: (ctx.session.trace[0],"elite"))
    def proposal(ctx,config,archives):
        actual = ctx.session.trace[1]
        return MutationProposal("prune_merge",{"i":0,"j":1,"source_id":actual.id},merge_clusters(actual.template,0,1))
    monkeypatch.setattr(adaptive,"_pruning_proposal",proposal)
    result = adaptive.run_adaptive_search([twisted_seed(5),twisted_seed(6)],CountEvaluator(),
        adaptive.AdaptiveSearchConfig(iterations=1,level=0,depth=0,restart_probability=0,prune_probability=1,max_sdp_calls=3))
    record = result.trace[2]
    assert record.parent_id == record.parameters["source_id"] == 1
    assert result.episodes[0].source_root_id == 0
    assert any(r.id == record.id for r in result.archives.stable)
    assert result.episodes[0].sdp_calls == 1


def test_undo_split_logs_all_parameters_needed_to_replay():
    template = split_cluster(twisted_seed(5),0,.5,.02)
    record = SearchRecord(0,None,"split",{"index":0},"test",template,None,"evaluated",None)
    ctx = SimpleNamespace(session=SimpleNamespace(rng=np.random.default_rng(7),trace=[record],archive=SimpleNamespace(entries=[])))
    proposal = adaptive._pruning_proposal(ctx,adaptive.AdaptiveSearchConfig(),adaptive.AdaptiveArchives())
    assert proposal.operation == "prune_undo_split"
    p = proposal.parameters
    replay = perturb_phase(merge_clusters(template,p["index"],p["index"]+1),p["perturb_index"],p["delta"])
    np.testing.assert_array_equal(replay.x,proposal.template.x)
    np.testing.assert_array_equal(replay.phi,proposal.template.phi)


def test_gamma_metrics_retain_each_size_without_ranking_across_sizes():
    trace = []
    for i,(m,gamma) in enumerate([(4,-.1),(5,-.001),(4,-.05)]):
        t=twisted_seed(m)
        e=TemplateEvaluation(t,"near_feasible",margin_gamma=gamma,sdp_calls=2)
        trace.append(SearchRecord(i,None,"test",{},"test",t,e,"evaluated",None))
    result = SearchResult(None,[],trace,3,6,3,1,"sdp_budget","audit")
    metrics = compute_run_metrics(result,algorithm="random",initial="test",budget=6,repetition=0,run_seed=1)
    assert metrics.best_margin_by_cluster == {4:-.05,5:-.001}
    assert metrics.best_margin_gamma is None and metrics.best_margin_n_clusters is None
    summary = summarize_runs([metrics],target_mu=.693)[0]
    assert summary.margin_by_cluster[4].maximum == -.05
    assert summary.margin_by_cluster[5].maximum == -.001
    assert summary.metrics["best_margin_gamma"].maximum is None
    assert RunMetrics.from_dict(metrics.to_dict()).best_margin_by_cluster == metrics.best_margin_by_cluster


def test_refinement_explicit_accuracy_and_both_margin_requests():
    config=RefinementConfig(solver_options={"max_iter":500})
    refined=config.evaluation_config("refined"); certified=config.evaluation_config("certified")
    assert refined.solver_options["tol_feas"] == 1e-10
    assert refined.solver_options["max_iter"] == 500
    assert certified.internal_margin >= 2*config.certification_margin
    assert certified.cluster_margin >= 2*config.certification_margin
    assert config.evaluation_config("fast").solver_options == {"max_iter":500}


def test_density_records_optimized_objective_and_fresh_independent_audit():
    t=twisted_seed(5); result=optimize_template(t.x,t.phi)
    assert result.optimized_mu is not None and result.objective_gap < 1e-7
    refined=refine_candidate(t,RefinementConfig(),level="refined")
    checks={c.name:c for c in refined.checks}
    assert checks["independent_positive_restoring"].passed
    assert checks["independent_positive_cluster_eigenvalue"].passed
    assert checks["independent_equilibrium_threshold"].passed
    assert refined.to_dict()["arithmetic_precision"] == "float64"
    assert refined.proof_available is False
