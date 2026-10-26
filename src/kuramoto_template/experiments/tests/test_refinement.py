"""Refinement triggers, independent audits, and candidate loading."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kuramoto_template.search.evaluator import (
    EvaluationConfig,
    SolverDiagnostics,
    TemplateEvaluation,
)
from kuramoto_template.search.random_search import twisted_seed
from kuramoto_template.template import PhaseTemplate

from ..refinement import (
    RefinementConfig,
    RefinementResult,
    audit_candidate,
    load_templates,
    refine_candidate,
    refine_candidates,
    trigger_reasons,
)


def diagnostic(
    source: str,
    status: str,
    *,
    mu: float | None = None,
    gamma: float | None = None,
    eigenvalue: float | None = None,
    force_error: float = 1e-12,
    valid: bool = True,
    stable: bool = True,
    P: np.ndarray | None = None,
) -> SolverDiagnostics:
    return SolverDiagnostics(
        source,  # type: ignore[arg-type]
        status,
        P=P,
        mu=mu,
        gamma=gamma,
        max_force_error=force_error,
        min_restoring=0.1,
        min_cluster_eigenvalue=eigenvalue,
        residuals={"reported_mu": 0.0},
        numerically_valid=valid,
        stable=stable,
        solver_agrees=valid,
    )


def evaluation(
    *,
    status: str = "stable",
    mu: float | None = 0.65,
    gamma: float | None = None,
    density_status: str = "optimal",
    eigenvalue: float | None = 1e-3,
    force_error: float = 1e-12,
    n_clusters: int = 5,
    sdp_calls: int = 1,
    cached: bool = False,
) -> TemplateEvaluation:
    template = twisted_seed(n_clusters)
    density = diagnostic(
        "density",
        density_status,
        mu=mu,
        eigenvalue=eigenvalue,
        force_error=force_error,
        P=np.eye(n_clusters),
    )
    margin = (
        diagnostic("margin", "optimal", gamma=gamma, stable=False)
        if gamma is not None
        else None
    )
    return TemplateEvaluation(
        template,
        status,
        stable_mu=mu if status == "stable" else None,
        margin_gamma=gamma,
        density=density,
        margin=margin,
        sdp_calls=sdp_calls,
        cached=cached,
    )


def test_trigger_is_silent_for_ordinary_stable_candidate() -> None:
    config = RefinementConfig()
    assert trigger_reasons(evaluation(mu=0.65), config) == []


def test_trigger_flags_high_mu_inaccuracy_and_small_eigenvalues() -> None:
    config = RefinementConfig()
    reasons = trigger_reasons(
        evaluation(mu=0.695, density_status="optimal_inaccurate", eigenvalue=1e-9),
        config,
    )
    text = " | ".join(reasons)
    assert "stable_mu" in text
    assert "optimal_inaccurate" in text
    assert "eigenvalue" in text


def test_trigger_flags_near_zero_target_margin() -> None:
    config = RefinementConfig()
    reasons = trigger_reasons(
        evaluation(
            status="near_feasible",
            mu=None,
            gamma=1e-9,
            density_status="infeasible",
            eigenvalue=None,
        ),
        config,
    )
    assert any("margin" in reason for reason in reasons)


def test_refined_audit_rejects_cached_and_inaccurate_evaluations() -> None:
    config = RefinementConfig(target_mu=0.5, mu_threshold=0.4)
    cached = evaluation(mu=0.65, cached=True)
    checks, passed, _ = audit_candidate(
        cached.template, cached, config, "refined"
    )
    assert not passed
    assert next(check for check in checks if check.name == "evaluated_by_solver").passed is False

    inaccurate = evaluation(mu=0.695, density_status="optimal_inaccurate")
    checks, passed, _ = audit_candidate(
        inaccurate.template, inaccurate, config, "refined"
    )
    assert not passed
    assert (
        next(
            check for check in checks if check.name == "signal_status_accurate"
        ).passed
        is False
    )


def test_entry_bound_audit_uses_the_refinement_tolerance() -> None:
    """Independent bound checks must use the solve tolerance, not a fixed 1e-10."""

    config = RefinementConfig(refined_residual_tolerance=1e-8)

    within = evaluation(mu=0.65)
    P = within.density.P.copy()  # type: ignore[union-attr]
    P[0, 0] = 1.0 + 1e-9
    within.density.P = P  # type: ignore[union-attr]
    checks, _, _ = audit_candidate(within.template, within, config, "refined")
    entry = next(
        check for check in checks if check.name == "density_entries_in_unit_interval"
    )
    assert entry.passed is True
    assert entry.threshold == pytest.approx(1e-8)

    outside = evaluation(mu=0.65)
    P = outside.density.P.copy()  # type: ignore[union-attr]
    P[0, 0] = 1.0 + 1e-6
    outside.density.P = P  # type: ignore[union-attr]
    checks, _, _ = audit_candidate(outside.template, outside, config, "refined")
    entry = next(
        check for check in checks if check.name == "density_entries_in_unit_interval"
    )
    assert entry.passed is False


@pytest.mark.integration
def test_real_sdp_refined_and_certified_gates() -> None:
    config = RefinementConfig(
        target_mu=0.6,
        mu_threshold=0.65,
        certification_margin=1e-12,
    )
    template = twisted_seed(5)

    refined = refine_candidate(template, config, level="refined")
    assert refined.evaluation.status == "stable"
    assert refined.proof_available is False
    names = {check.name for check in refined.checks}
    assert {
        "independent_mu",
        "independent_cluster_eigenvalue",
        "stable_mu_matches_independent_metrics",
        "evaluated_by_solver",
    } <= names
    assert refined.gate_passed

    certified = refine_candidate(template, config, level="certified")
    assert certified.proof_available is False
    assert certified.status == "certified_numerical_gate_passed"
    assert certified.gate_passed


@pytest.mark.integration
def test_certified_gate_rejects_candidate_below_target() -> None:
    """A strict margin alone is not enough: the target density must also hold."""

    config = RefinementConfig(
        target_mu=0.7,
        mu_threshold=0.65,
        certification_margin=1e-6,
    )
    certified = refine_candidate(twisted_seed(5), config, level="certified")

    assert certified.status == "certified_numerical_gate_failed"
    assert certified.proof_available is False
    checks = {check.name: check for check in certified.checks}
    assert checks["positive_cluster_eigenvalue"].passed is True
    assert checks["target_density"].passed is False


def test_certified_solve_requests_the_strict_margin() -> None:
    """The certified solve must enforce the margin, not just test for it."""

    config = RefinementConfig(
        certification_margin=1e-6,
        refined_cluster_margin=1e-9,
    )
    refined = config.evaluation_config("refined")
    certified = config.evaluation_config("certified")
    assert refined.cluster_margin == pytest.approx(1e-9)
    assert certified.cluster_margin == pytest.approx(2e-6)
    assert certified.stability_tolerance == pytest.approx(1e-12)


class StubEvaluator:
    config = EvaluationConfig()

    def __init__(self, *, mu: float, status: str = "optimal") -> None:
        self.mu = mu
        self.status = status
        self.calls: list[PhaseTemplate] = []
        self.evaluations: list[TemplateEvaluation] = []

    def evaluate(
        self, template: PhaseTemplate, *, max_solver_calls: int | None = None
    ) -> TemplateEvaluation:
        self.calls.append(template)
        density = SolverDiagnostics(
            "density",
            self.status,
            P=np.eye(template.n_clusters),
            mu=self.mu,
            max_force_error=1e-12,
            min_restoring=0.1,
            min_cluster_eigenvalue=1e-3,
            residuals={"reported_mu": 0.0},
            numerically_valid=True,
            stable=True,
            solver_agrees=True,
        )
        result = TemplateEvaluation(
            template,
            "stable",
            stable_mu=self.mu,
            density=density,
            sdp_calls=1,
        )
        self.evaluations.append(result)
        return result


def test_refine_candidates_screens_then_refines_only_triggered() -> None:
    config = RefinementConfig(target_mu=0.6, mu_threshold=0.65)
    made: list[StubEvaluator] = []

    def factory(
        cfg: RefinementConfig, level: str
    ) -> StubEvaluator:
        del cfg, level
        evaluator = StubEvaluator(mu=0.695)
        made.append(evaluator)
        return evaluator

    results, warnings = refine_candidates(
        [
            ("high", twisted_seed(5)),
            ("ordinary", twisted_seed(5)),
        ],
        config,
        evaluator_factory=factory,
    )
    # One evaluator is created for screening and one for refinement. Every
    # candidate triggers, so both are used for both candidates.
    assert len(made) == 2
    assert [len(evaluator.calls) for evaluator in made] == [2, 2]
    assert all(result.refined is not None for result in results)
    assert all(result.trigger_reasons for result in results)
    assert all(
        result.refined is not None
        and result.refined.evaluation is made[1].evaluations[index]
        for index, result in enumerate(results)
    )
    assert warnings == []


def test_untriggered_candidate_is_not_refined() -> None:
    config = RefinementConfig(target_mu=0.6, mu_threshold=0.65)
    made: list[StubEvaluator] = []

    def factory(
        cfg: RefinementConfig, level: str
    ) -> StubEvaluator:
        del cfg, level
        evaluator = StubEvaluator(mu=0.60)
        made.append(evaluator)
        return evaluator

    results, _ = refine_candidates(
        [("ordinary", twisted_seed(5))],
        config,
        evaluator_factory=factory,
    )
    assert len(made) == 2  # one fast evaluator and one unused refined evaluator
    assert results[0].trigger_reasons == []
    assert results[0].refined is None


def test_fast_level_reuses_the_screening_evaluation() -> None:
    config = RefinementConfig(target_mu=0.6, mu_threshold=0.65)
    made: list[StubEvaluator] = []

    def factory(
        cfg: RefinementConfig, level: str
    ) -> StubEvaluator:
        del cfg, level
        evaluator = StubEvaluator(mu=0.695)
        made.append(evaluator)
        return evaluator

    results, _ = refine_candidates(
        [("high", twisted_seed(5))],
        config,
        level="fast",
        evaluator_factory=factory,
    )
    assert len(made) == 2
    assert results[0].refined is not None
    # The fast level must reuse the screening evaluation, not pay for a second
    # identical solve through the refinement evaluator.
    assert results[0].refined.evaluation is made[0].evaluations[0]
    assert made[1].calls == []
    assert results[0].refined.level == "fast"
    assert any(
        check.name == "screening_only" for check in results[0].refined.checks
    )


def test_refinement_status_labels_distinguish_levels() -> None:
    template = twisted_seed(5)
    evaluation = TemplateEvaluation(template, "stable", stable_mu=0.65)

    def result(level: str, passed: bool) -> RefinementResult:
        return RefinementResult(
            template=template,
            level=level,  # type: ignore[arg-type]
            evaluation=evaluation,
            trigger_reasons=[],
            checks=[],
            gate_passed=passed,
        )

    assert result("fast", True).status == "screened"
    assert result("fast", False).status == "screening_incomplete"
    assert result("refined", True).status == "refined"
    assert result("refined", False).status == "refinement_failed"
    assert result("certified", True).status == "certified_numerical_gate_passed"
    assert result("certified", False).status == "certified_numerical_gate_failed"


def test_load_templates_from_result_and_trace(tmp_path: Path) -> None:
    template = twisted_seed(5)
    payload = {
        "best": {
            "template": {
                "x": template.x.tolist(),
                "phi": template.phi.tolist(),
                "n_clusters": 5,
            }
        },
        "archive": [],
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_templates(result_path)
    assert len(loaded) == 1
    assert np.allclose(loaded[0][1].x, template.x)
    assert np.allclose(loaded[0][1].phi, template.phi)

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "template": {
                            "x": template.x.tolist(),
                            "phi": template.phi.tolist(),
                            "n_clusters": 5,
                        }
                    }
                ),
                json.dumps({"template": None}),
            ]
        ),
        encoding="utf-8",
    )
    assert len(load_templates(trace_path)) == 1

    with pytest.raises(ValueError, match="result.json"):
        load_templates(tmp_path / "missing")
    with pytest.raises(ValueError):
        load_templates(tmp_path / "unknown.txt")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_mu": 0.0},
        {"mu_threshold": 1.0},
        {"eigenvalue_threshold": -1},
        {"margin_threshold": -1},
        {"force_tolerance": -1},
        {"certification_margin": -1},
        {"refined_internal_margin": 0},
        {"refined_cluster_margin": 0},
        {"refined_residual_tolerance": 0},
        {"refined_stability_tolerance": -1},
        {"max_clusters": 1},
        {"solver": ""},
    ],
)
def test_invalid_refinement_config(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RefinementConfig(**kwargs)
