"""Metric computation, aggregation, and paired comparison."""

import math
from typing import Any

import numpy as np
import pytest

from kuramoto_template.search.evaluator import SolverDiagnostics, TemplateEvaluation
from kuramoto_template.search.infrastructure import (
    ArchiveEntry,
    SearchRecord,
    SearchResult,
)
from kuramoto_template.search.random_search import (
    RandomSearchConfig,
    run_random_search,
    twisted_seed,
)
from kuramoto_template.template import PhaseTemplate

from ..cache import CachedEvaluator
from ..metrics import (
    RunMetrics,
    best_mu_at,
    compute_run_metrics,
    paired_comparisons,
    summarize_runs,
    summarize_values,
)
from .support import FakeEvaluator


def density(mu: float, status: str = "optimal") -> SolverDiagnostics:
    return SolverDiagnostics(
        "density",
        status,
        mu=mu,
        numerically_valid=True,
        stable=True,
        solver_agrees=True,
    )


def invalid_density(status: str) -> SolverDiagnostics:
    return SolverDiagnostics(
        "density",
        status,
        numerically_valid=False,
        stable=False,
        solver_agrees=False,
    )


def margin(gamma: float) -> SolverDiagnostics:
    return SolverDiagnostics(
        "margin",
        "optimal",
        mu=0.693,
        gamma=gamma,
        numerically_valid=True,
        stable=False,
        solver_agrees=True,
    )


def record(
    identifier: int,
    template: PhaseTemplate,
    evaluation: TemplateEvaluation,
    *,
    disposition: str = "evaluated",
) -> SearchRecord:
    return SearchRecord(
        id=identifier,
        parent_id=None if identifier == 0 else identifier - 1,
        operation="test",
        parameters={},
        selection="test",
        template=template,
        evaluation=evaluation,
        disposition=disposition,
        best_mu=None,
    )


def manual_result() -> SearchResult:
    stable_template = twisted_seed(5)
    near_template = twisted_seed(4)
    inaccurate_template = twisted_seed(6)
    error_template = twisted_seed(3)
    invalid_template = twisted_seed(7)
    cached_template = twisted_seed(5)

    stable = TemplateEvaluation(
        stable_template,
        "stable",
        stable_mu=0.70,
        density=density(0.70),
        sdp_calls=1,
    )
    near = TemplateEvaluation(
        near_template,
        "near_feasible",
        margin_gamma=-0.02,
        density=invalid_density("infeasible"),
        margin=margin(-0.02),
        sdp_calls=2,
    )
    inaccurate = TemplateEvaluation(
        inaccurate_template,
        "stable",
        stable_mu=0.68,
        density=density(0.68, "optimal_inaccurate"),
        sdp_calls=1,
    )
    failure = TemplateEvaluation(
        error_template,
        "solver_error",
        density=invalid_density("solver_error"),
        sdp_calls=1,
    )
    invalid = TemplateEvaluation(invalid_template, "invalid", sdp_calls=0)
    cached = TemplateEvaluation(
        cached_template,
        "stable",
        stable_mu=0.70,
        density=density(0.70),
        sdp_calls=0,
        cached=True,
    )

    trace = [
        record(0, stable_template, stable),
        record(1, near_template, near),
        record(2, inaccurate_template, inaccurate),
        record(3, error_template, failure),
        record(4, invalid_template, invalid, disposition="rejected"),
        record(5, cached_template, cached, disposition="duplicate"),
    ]
    best = ArchiveEntry(0, stable)
    return SearchResult(
        best=best,
        archive=[best],
        trace=trace,
        evaluations=6,
        sdp_calls=5,
        proposals=3,
        elapsed_seconds=1.25,
        stop_reason="iterations",
        run_id="manual",
    )


def run_metrics(**overrides: Any) -> RunMetrics:
    data: dict[str, Any] = {
        "algorithm": "random",
        "initial": "five_twisted",
        "budget": 100,
        "repetition": 0,
        "run_seed": 1,
        "run_id": "run",
    }
    data.update(overrides)
    return RunMetrics(**data)


def test_manual_run_metrics_split_signals_and_costs() -> None:
    metrics = compute_run_metrics(
        manual_result(),
        algorithm="random",
        initial="five_twisted",
        budget=100,
        repetition=0,
        run_seed=7,
        cache_stats=None,
        wall_seconds=2.5,
        cache_decimals=12,
    )
    assert metrics.best_stable_mu == 0.70
    assert metrics.best_margin_gamma == -0.02
    assert metrics.best_margin_n_clusters == 4
    assert metrics.best_density_mu == 0.70
    assert metrics.stable_templates == 2
    assert metrics.near_feasible_templates == 1
    assert metrics.unique_templates == 5
    assert metrics.evaluations == 6
    assert metrics.unique_evaluations == 5
    assert metrics.cache_hits == 1
    assert metrics.cache_misses == 5
    assert metrics.duplicate_proposals == 1
    assert metrics.rejected_proposals == 1
    assert metrics.proposals == 3
    assert metrics.sdp_calls == 5
    assert metrics.density_sdp_calls == 4
    assert metrics.margin_sdp_calls == 1
    assert metrics.density_sdp_calls + metrics.margin_sdp_calls == metrics.sdp_calls
    assert metrics.solver_errors == 1
    assert metrics.inaccurate_results == 1
    assert metrics.invalid_evaluations == 1
    assert metrics.numerically_invalid == 2
    assert metrics.wall_seconds == 2.5
    assert metrics.elapsed_seconds == 1.25
    assert metrics.status_counts == {
        "invalid": 1,
        "near_feasible": 1,
        "solver_error": 1,
        "stable": 3,
    }
    assert metrics.unique_status_counts == {
        "invalid": 1,
        "near_feasible": 1,
        "solver_error": 1,
        "stable": 2,
    }
    assert metrics.density_status_counts == {
        "infeasible": 1,
        "optimal": 1,
        "optimal_inaccurate": 1,
        "solver_error": 1,
    }
    assert metrics.margin_status_counts == {"optimal": 1}
    assert metrics.best_mu_curve == [{"sdp_calls": 1, "best_mu": 0.70, "record_id": 0}]
    assert metrics.first_stable_call == 1
    assert metrics.best_mu_call == 1


def test_cached_evaluations_are_excluded_from_unique_counts() -> None:
    result = manual_result()
    metrics = compute_run_metrics(
        result,
        algorithm="random",
        initial="five_twisted",
        budget=100,
        repetition=0,
        run_seed=7,
    )
    assert metrics.cache_hits == 1
    assert metrics.unique_evaluations == 5


def test_live_fake_run_accounts_for_every_solver_call() -> None:
    inner = FakeEvaluator()
    cache = CachedEvaluator(inner)
    result = run_random_search(
        [twisted_seed(5)],
        cache,
        RandomSearchConfig(iterations=10, seed=3, max_sdp_calls=200),
    )
    metrics = compute_run_metrics(
        result,
        algorithm="random",
        initial="five_twisted",
        budget=200,
        repetition=0,
        run_seed=3,
        cache_stats=cache.stats,
    )
    assert metrics.sdp_calls == metrics.density_sdp_calls + metrics.margin_sdp_calls
    assert metrics.unique_evaluations == metrics.cache_misses
    assert metrics.unique_templates == metrics.cache_misses
    assert metrics.cache_hits == cache.stats.hits
    assert metrics.evaluations == result.evaluations


def test_summarize_values_ignores_missing_and_nonfinite() -> None:
    stats = summarize_values([1.0, None, 3.0, math.inf])
    assert stats.count == 2 and stats.missing == 2
    assert stats.mean == 2.0
    assert stats.median == 2.0
    assert stats.std == 1.0
    assert stats.minimum == 1.0 and stats.maximum == 3.0
    assert stats.q25 == 1.5 and stats.q75 == 2.5


def test_summarize_runs_rates_and_progress() -> None:
    runs = [
        run_metrics(
            repetition=0,
            best_stable_mu=0.70,
            best_mu_curve=[
                {"sdp_calls": 25, "best_mu": 0.68, "record_id": 0},
                {"sdp_calls": 50, "best_mu": 0.69, "record_id": 1},
                {"sdp_calls": 100, "best_mu": 0.70, "record_id": 2},
            ],
        ),
        run_metrics(repetition=1, best_stable_mu=None),
        run_metrics(
            repetition=2,
            best_stable_mu=0.692,
            best_mu_curve=[{"sdp_calls": 100, "best_mu": 0.692, "record_id": 5}],
        ),
    ]
    summaries = summarize_runs(runs, target_mu=0.693)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.runs == 3
    assert summary.stable_run_rate == 2 / 3
    assert summary.target_run_rate == 1 / 3
    assert summary.metrics["best_stable_mu"].mean == np.mean([0.70, 0.692])
    assert summary.metrics["best_stable_mu"].missing == 1
    assert summary.progress["25pct"].mean == 0.68
    assert summary.progress["50pct"].mean == 0.69
    assert summary.progress["100pct"].mean == np.mean([0.70, 0.692])


def test_best_mu_at_uses_step_function() -> None:
    curve = [
        {"sdp_calls": 10, "best_mu": 0.60, "record_id": 0},
        {"sdp_calls": 20, "best_mu": 0.65, "record_id": 1},
    ]
    assert best_mu_at(curve, 0) is None
    assert best_mu_at(curve, 10) == 0.60
    assert best_mu_at(curve, 19) == 0.60
    assert best_mu_at(curve, 20) == 0.65
    assert best_mu_at(curve, 1000) == 0.65


def test_paired_comparisons_use_matched_repetitions() -> None:
    values = {
        ("random", 0): 0.70,
        ("random", 1): None,
        ("random", 2): 0.68,
        ("random", 3): 0.69,
        ("nested", 0): 0.71,
        ("nested", 1): 0.60,
        ("nested", 2): 0.68,
        ("nested", 3): None,
    }
    runs = [
        run_metrics(algorithm=algorithm, repetition=repetition, best_stable_mu=mu)
        for (algorithm, repetition), mu in values.items()
    ]
    comparisons = paired_comparisons(runs, baseline="random", candidate="nested")
    overall = comparisons[-1]
    assert overall.initial == "all" and overall.budget is None
    assert overall.pairs == 4
    assert overall.candidate_wins == 2
    assert overall.baseline_wins == 1
    assert overall.ties == 1
    assert overall.candidate_only_stable == 1
    assert overall.baseline_only_stable == 1
    assert overall.mean_delta == pytest.approx(0.005)
    assert overall.median_delta == pytest.approx(0.005)
