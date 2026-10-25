import json
from pathlib import Path

import numpy as np
import pytest

from kuramoto_template.search import random_search as module
from kuramoto_template.search.evaluator import EvaluationConfig, SolverDiagnostics, TemplateEvaluation, TemplateEvaluator
from kuramoto_template.search.random_search import (
    ArchiveEntry, DiversityArchive, MutationProposal, RandomSearchConfig,
    SearchResult, run_random_search, sample_mutation, twisted_seed,
)
from kuramoto_template.template import PhaseTemplate


class FakeEvaluator:
    config = EvaluationConfig()

    def __init__(self, mode: str = "mixed") -> None:
        self.mode = mode
        self.limits: list[int | None] = []

    def evaluate(
        self, template: PhaseTemplate, *, max_solver_calls: int | None = None,
    ) -> TemplateEvaluation:
        self.limits.append(max_solver_calls)
        calls = min(2, max_solver_calls) if max_solver_calls is not None else 2
        if self.mode == "errors":
            return TemplateEvaluation(template, "solver_error", sdp_calls=calls, message="fake solver failed")
        if self.mode == "near" or template.n_clusters % 2 == 0:
            return TemplateEvaluation(template, "near_feasible", margin_gamma=float(template.n_clusters * 100), sdp_calls=calls)
        mu = float(0.5 + 0.1 * np.dot(template.x, np.cos(template.phi)**2))
        diagnostic = SolverDiagnostics("density", "optimal", P=np.eye(template.n_clusters), mu=mu)
        return TemplateEvaluation(template, "stable", stable_mu=mu, density=diagnostic, sdp_calls=calls)


def actions(result: SearchResult):
    return [(r.id, r.parent_id, r.operation, r.parameters, r.disposition, r.best_mu,
             r.evaluation.status if r.evaluation else None) for r in result.trace]


def test_reproducibility_incumbent_and_budget_bookkeeping():
    config = RandomSearchConfig(iterations=60, seed=42, archive_size=8)
    seeds = [twisted_seed(5), twisted_seed(4)]
    first = run_random_search(seeds, FakeEvaluator(), config)
    second = run_random_search(seeds, FakeEvaluator(), config)
    assert actions(first) == actions(second)
    assert first.evaluations == sum(r.evaluation is not None for r in first.trace)
    assert first.sdp_calls == 2 * first.evaluations
    assert first.proposals == 60 and len(first.trace) == 62
    assert first.stop_reason == "iterations"
    assert len(first.archive) <= 8
    best_history = [r.best_mu for r in first.trace if r.best_mu is not None]
    assert best_history == sorted(best_history)
    assert first.best is not None
    assert second.best is not None
    assert first.best.evaluation.stable_mu == second.best.evaluation.stable_mu
    assert any(r.selection == "restart" for r in first.trace)
    assert any(r.evaluation is not None and r.evaluation.margin_gamma is not None for r in first.trace)
    for a, b in zip(first.trace, second.trace):
        if a.template is not None:
            assert b.template is not None
            np.testing.assert_array_equal(a.template.x, b.template.x)
            np.testing.assert_array_equal(a.template.phi, b.template.phi)


def test_high_gamma_does_not_replace_stable_incumbent():
    archive = DiversityArchive(3)
    stable = ArchiveEntry(0, TemplateEvaluation(twisted_seed(), "stable", stable_mu=0.6))
    archive.consider(stable)
    for i in range(1, 8):
        archive.consider(ArchiveEntry(i, TemplateEvaluation(twisted_seed(4), "near_feasible", margin_gamma=1e10 + i)))
    assert archive.best is stable and stable in archive.entries
    assert any(e.evaluation.status == "near_feasible" for e in archive.entries)


def test_archive_retains_cluster_diversity():
    archive = DiversityArchive(3)
    archive.consider(ArchiveEntry(0, TemplateEvaluation(twisted_seed(4), "near_feasible", margin_gamma=0)))
    archive.consider(ArchiveEntry(1, TemplateEvaluation(twisted_seed(6), "near_feasible", margin_gamma=-100)))
    for i in range(2, 10):
        archive.consider(ArchiveEntry(i, TemplateEvaluation(twisted_seed(5), "stable", stable_mu=0.5 + i / 100)))
    assert {e.evaluation.template.n_clusters for e in archive.entries} == {4, 5, 6}
    assert archive.best is not None
    assert archive.best.evaluation.stable_mu == pytest.approx(0.59)


@pytest.mark.parametrize("m", [3, 6])
def test_sampler_respects_bounds_and_invariants(m):
    rng = np.random.default_rng(10)
    config = RandomSearchConfig(min_clusters=3, max_clusters=6)
    for _ in range(100):
        proposal = sample_mutation(twisted_seed(m), rng, config)
        assert proposal.operation != ("merge" if m == 3 else "split")
        if proposal.template is None:
            assert proposal.operation == "merge" and "antipodal" in proposal.message
            continue
        assert 3 <= proposal.template.n_clusters <= 6
        assert proposal.template.x.sum() == pytest.approx(1)
        assert np.all(proposal.template.x > 0)
        assert np.all((proposal.template.phi >= 0) & (proposal.template.phi < 2 * np.pi))


def test_tiny_resize_scale_is_supported():
    proposal = sample_mutation(twisted_seed(), np.random.default_rng(0),
                              RandomSearchConfig(mutation_weights=(0, 0, 0, 1), resize_fraction=1e-4))
    assert proposal.template is not None and proposal.operation == "resize"


def test_sdp_budget_includes_seeds_and_reserves_calls():
    evaluator = FakeEvaluator()
    result = run_random_search([twisted_seed(5), twisted_seed(4)], evaluator,
                               RandomSearchConfig(iterations=100, max_sdp_calls=3))
    assert result.evaluations == 2 and result.sdp_calls == 3
    assert evaluator.limits == [2, 1]
    assert result.proposals == 0 and result.stop_reason == "sdp_budget"


def test_proposals_stop_at_exact_sdp_budget():
    result = run_random_search([twisted_seed()], FakeEvaluator(),
                               RandomSearchConfig(iterations=100, max_sdp_calls=7))
    assert result.sdp_calls == 7 and result.stop_reason == "sdp_budget"


@pytest.mark.parametrize("mode", ["errors", "near"])
def test_no_stable_candidates_is_graceful(mode):
    result = run_random_search([twisted_seed()], FakeEvaluator(mode), RandomSearchConfig(iterations=12))
    assert result.best is None
    assert result.proposals == 12 and result.archive
    assert all(r.best_mu is None for r in result.trace)
    assert all(r.evaluation is None or r.evaluation.status != "stable" for r in result.trace)


def test_rejected_and_duplicate_proposals_do_not_spend_calls(monkeypatch):
    seed = twisted_seed()
    sequence = iter([MutationProposal("perturb", {"delta": 0}, seed),
                     MutationProposal("merge", {}, message="antipodal cancellation")])
    monkeypatch.setattr(module, "sample_mutation", lambda *args: next(sequence))
    result = run_random_search([seed], FakeEvaluator(), RandomSearchConfig(iterations=2, restart_probability=0))
    assert [r.disposition for r in result.trace] == ["evaluated", "duplicate", "rejected"]
    assert result.evaluations == 1 and result.proposals == 2 and result.sdp_calls == 2


def test_exact_dedup_does_not_round_nearby_phases(monkeypatch):
    seed = twisted_seed()
    phi = seed.phi.copy()
    phi[0] += 1e-14
    near = PhaseTemplate(seed.x, phi)
    monkeypatch.setattr(module, "sample_mutation", lambda *args: MutationProposal("perturb", {"delta": 1e-14}, near))
    result = run_random_search([seed], FakeEvaluator(), RandomSearchConfig(iterations=1, restart_probability=0))
    assert result.evaluations == 2


def test_wall_limit_is_checked_after_mandatory_seeds(monkeypatch):
    ticks = iter([0, 1, 2])
    monkeypatch.setattr(module, "perf_counter", lambda: next(ticks))
    result = run_random_search([twisted_seed()], FakeEvaluator(), RandomSearchConfig(iterations=10, max_wall_seconds=0.5))
    assert result.evaluations == 1 and result.proposals == 0 and result.stop_reason == "wall_time"


def test_jsonl_and_result_preserve_diagnostics_and_refuse_overwrite(tmp_path: Path):
    destination = tmp_path / "run"
    result = run_random_search([twisted_seed()], FakeEvaluator(), RandomSearchConfig(iterations=4), output=destination)
    metadata = json.loads((destination / "metadata.json").read_text())
    lines = [json.loads(line) for line in (destination / "trace.jsonl").read_text().splitlines()]
    saved = json.loads((destination / "result.json").read_text())
    assert len(lines) == len(result.trace)
    assert metadata["search_config"]["seed"] == 42
    assert metadata["evaluation_config"]["mu_target"] == 0.693
    assert metadata["code_sha256"] and metadata["versions"]["cvxpy"]
    assert result.best is not None
    assert result.best.evaluation.P is not None
    assert saved["best"]["evaluation"]["stable_mu"] == result.best.evaluation.stable_mu
    np.testing.assert_array_equal(saved["best"]["evaluation"]["density"]["P"], result.best.evaluation.P)
    np.testing.assert_array_equal(saved["best"]["template"]["phi"], result.best.evaluation.template.phi)
    before = (destination / "trace.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        run_random_search([twisted_seed()], FakeEvaluator(), output=destination)
    assert (destination / "trace.jsonl").read_bytes() == before


def test_nonfinite_diagnostics_are_standard_json(tmp_path: Path):
    class BadDiagnostics(FakeEvaluator):
        def evaluate(
            self, template: PhaseTemplate, *, max_solver_calls: int | None = None,
        ) -> TemplateEvaluation:
            return TemplateEvaluation(template, "uncertain", density=SolverDiagnostics(
                "density", "optimal_inaccurate", residuals={"reported_mu": np.inf}, P=np.full((5, 5), np.nan)), sdp_calls=1)
    run_random_search([twisted_seed()], BadDiagnostics(), RandomSearchConfig(iterations=0), output=tmp_path / "run")
    text = (tmp_path / "run" / "trace.jsonl").read_text()
    assert "NaN" not in text and "Infinity" not in text
    data = json.loads(text)
    assert data["evaluation"]["status"] == "uncertain"


@pytest.mark.parametrize("kwargs", [
    {"iterations": -1}, {"iterations": 1.5}, {"seed": -1}, {"archive_size": 0},
    {"min_clusters": 1}, {"min_clusters": 6, "max_clusters": 4},
    {"max_sdp_calls": 0}, {"max_wall_seconds": np.inf}, {"restart_probability": 2},
    {"separation_min": 0}, {"phase_scale": -1}, {"resize_fraction": 1},
    {"mutation_weights": (0, 0, 0, 0)}, {"mutation_weights": (1, -1, 1, 1)},
    {"parent_selection": "greedy"},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        RandomSearchConfig(**kwargs)


@pytest.mark.parametrize("seeds,config", [
    ([], RandomSearchConfig()),
    ([twisted_seed(2)], RandomSearchConfig()),
    ([PhaseTemplate(np.full(5, 0.2), np.zeros(5))], RandomSearchConfig()),
    ([twisted_seed(), twisted_seed()], RandomSearchConfig(max_sdp_calls=1)),
    ([twisted_seed()], RandomSearchConfig(max_clusters=13)),
])
def test_invalid_run_inputs(seeds, config):
    with pytest.raises(ValueError):
        run_random_search(seeds, FakeEvaluator(), config)


@pytest.mark.integration
def test_real_seed_smoke():
    result = run_random_search([twisted_seed()], TemplateEvaluator(),
                               RandomSearchConfig(iterations=2, max_sdp_calls=5))
    assert result.best is not None
    assert result.best.evaluation.stable_mu is not None
    assert result.best.evaluation.stable_mu > 0.59
    assert result.sdp_calls <= 5
    assert result.best.evaluation.density is not None
    assert result.best.evaluation.density.min_cluster_eigenvalue is not None
    assert result.best.evaluation.density.min_cluster_eigenvalue > 0
