from dataclasses import replace
from unittest.mock import Mock

import numpy as np
from numpy.typing import NDArray
import pytest

from kuramoto_template.search import evaluator as module
from kuramoto_template.search.evaluator import EvaluationConfig, TemplateEvaluator
from kuramoto_template.sdp_optimizer.sdp_result import SDPResult
from kuramoto_template.sdp_optimizer.stability_sdp_result import StabilitySDPResult
from kuramoto_template.sdp_optimizer.template_metrics import template_metrics
from kuramoto_template.template import PhaseTemplate


@pytest.fixture
def ring() -> tuple[PhaseTemplate, NDArray[np.float64]]:
    template = PhaseTemplate(np.full(5, 0.2), np.arange(5) * 2 * np.pi / 5)
    P = np.eye(5)
    for i in range(5):
        P[i, (i + 1) % 5] = P[(i + 1) % 5, i] = 1
    return template, P


def density_result(template: PhaseTemplate, P: NDArray[np.float64], **kwargs) -> SDPResult:
    result = SDPResult(
        status="optimal",
        P=P,
        mu=template_metrics(template.x, template.phi, P).mu,
        numerically_feasible=True,
    )
    return replace(result, **kwargs)


def test_density_success_skips_fallback(ring, monkeypatch):
    template, P = ring
    density = Mock(return_value=density_result(template, P))
    margin = Mock()
    monkeypatch.setattr(module, "optimize_template", density)
    monkeypatch.setattr(module, "maximize_stability_margin", margin)
    result = TemplateEvaluator().evaluate(template, allowed=P.astype(bool))
    assert result.status == "stable" and result.stable_mu == pytest.approx(0.6)
    assert result.margin_gamma is None and result.sdp_calls == 1
    assert result.P_source == "density"
    assert result.density is not None
    assert result.density.min_cluster_eigenvalue is not None
    assert result.density.min_cluster_eigenvalue > 0
    assert result.density.residuals["forbidden"] == 0
    assert result.P is not None
    assert not np.shares_memory(result.P, P)
    margin.assert_not_called()
    assert density.call_args is not None
    assert density.call_args.kwargs["cluster_margin"] == 1e-8


def test_negative_margin_retained_without_density_score(monkeypatch):
    template = PhaseTemplate(np.array([0.5, 0.5]), np.array([0, np.pi]))
    density = Mock(return_value=SDPResult("infeasible"))
    margin = Mock(
        return_value=StabilitySDPResult(
            "optimal",
            mu=1.0,
            P=np.ones((2, 2)),
            gamma=-0.5,
            target_mu=1.0,
            numerically_valid=True,
            numerically_feasible=False,
        )
    )
    monkeypatch.setattr(module, "optimize_template", density)
    monkeypatch.setattr(module, "maximize_stability_margin", margin)
    result = TemplateEvaluator(EvaluationConfig(mu_target=1.0)).evaluate(template)
    assert result.status == "near_feasible" and result.stable_mu is None
    assert result.margin_gamma == -0.5 and result.sdp_calls == 2
    assert result.P_source == "margin"
    assert result.margin is not None
    assert result.margin.min_cluster_eigenvalue == pytest.approx(-0.5)
    density.assert_called_once()
    margin.assert_called_once()


def test_synchronized_template_is_invalid_without_solver_calls(monkeypatch):
    solver = Mock()
    monkeypatch.setattr(module, "optimize_template", solver)
    result = TemplateEvaluator().evaluate(
        PhaseTemplate(np.array([0.5, 0.5]), np.array([0, 2 * np.pi]))
    )
    assert result.status == "invalid" and result.sdp_calls == 0
    assert "synchronized" in result.message
    solver.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [SDPResult("infeasible"), SDPResult("solver_error"), SDPResult("optimal")],
)
def test_failure_never_sets_stable_mu(ring, failure, monkeypatch):
    monkeypatch.setattr(module, "optimize_template", Mock(return_value=failure))
    result = TemplateEvaluator(EvaluationConfig(use_margin_fallback=False)).evaluate(
        ring[0]
    )
    assert result.stable_mu is None and result.status != "stable"
    assert result.sdp_calls == 1 and result.margin is None


@pytest.mark.parametrize(
    "corruption,key",
    [
        ("asymmetry", "symmetry"),
        ("bounds", "upper_bound"),
        ("negative", "lower_bound"),
        ("force", "equilibrium"),
        ("clique", "cliques"),
        ("forbidden", "forbidden"),
        ("curvature", "cluster_eigenvalue"),
        ("reported_mu", "reported_mu"),
    ],
)
def test_independent_constraint_checks(ring, corruption, key, monkeypatch):
    template, original = ring
    P = original.copy()
    if corruption == "asymmetry":
        P[0, 1] = 0.5
    elif corruption == "bounds":
        P[0, 0] = 1.1
    elif corruption == "negative":
        P[0, 2] = P[2, 0] = -0.1
    elif corruption == "force":
        P[0, 1] = P[1, 0] = 0.5
    elif corruption == "clique":
        P[0, 0] = 0.9
    elif corruption == "forbidden":
        P[0, 2] = P[2, 0] = 0.1
    elif corruption == "curvature":
        P = np.ones((5, 5))
    result = density_result(template, P)
    if corruption == "reported_mu":
        result.mu = 0.9
    monkeypatch.setattr(module, "optimize_template", Mock(return_value=result))
    evaluation = TemplateEvaluator(
        EvaluationConfig(use_margin_fallback=False)
    ).evaluate(template, allowed=original.astype(bool))
    assert evaluation.status == "uncertain" and evaluation.stable_mu is None
    assert evaluation.density is not None
    assert evaluation.density.residuals[key] > 1e-7
    np.testing.assert_array_equal(evaluation.density.P, P)


def test_solver_flag_disagreement_is_uncertain(ring, monkeypatch):
    template, P = ring
    monkeypatch.setattr(
        module,
        "optimize_template",
        Mock(return_value=density_result(template, P, numerically_feasible=False)),
    )
    result = TemplateEvaluator(EvaluationConfig(use_margin_fallback=False)).evaluate(
        template
    )
    assert result.status == "uncertain" and result.stable_mu is None
    assert result.density is not None
    assert not result.density.solver_agrees


def test_exceptions_and_budget_limit(ring, monkeypatch):
    monkeypatch.setattr(
        module, "optimize_template", Mock(side_effect=RuntimeError("test failure"))
    )
    fallback = Mock()
    monkeypatch.setattr(module, "maximize_stability_margin", fallback)
    result = TemplateEvaluator().evaluate(ring[0], max_solver_calls=1)
    assert result.status == "solver_error" and result.sdp_calls == 1
    assert "RuntimeError: test failure" in result.message
    assert result.density is not None
    assert result.density.elapsed_seconds >= 0
    fallback.assert_not_called()


class _Panic(BaseException):
    """Stand-in for pyo3_runtime.PanicException, which is not an Exception."""


def test_solver_panic_is_recorded_not_raised(ring, monkeypatch):
    panic = _Panic("Eigval error: Eigen(1)")
    monkeypatch.setattr(module, "optimize_template", Mock(side_effect=panic))
    monkeypatch.setattr(module, "maximize_stability_margin", Mock())
    result = TemplateEvaluator().evaluate(ring[0], max_solver_calls=1)

    assert result.status == "solver_error"
    assert result.sdp_calls == 1
    assert result.density is not None
    assert "_Panic: Eigval error: Eigen(1)" in result.density.message


def test_interrupts_still_propagate(ring, monkeypatch):
    monkeypatch.setattr(
        module, "optimize_template", Mock(side_effect=KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        TemplateEvaluator().evaluate(ring[0])


def test_bad_margin_residuals_are_not_ranked(monkeypatch):
    template = PhaseTemplate(np.array([0.5, 0.5]), np.array([0, np.pi]))
    monkeypatch.setattr(
        module, "optimize_template", Mock(return_value=SDPResult("infeasible"))
    )
    monkeypatch.setattr(
        module,
        "maximize_stability_margin",
        Mock(
            return_value=StabilitySDPResult(
                "optimal_inaccurate",
                mu=1.0,
                P=np.ones((2, 2)),
                gamma=0.1,
                target_mu=1.0,
                numerically_valid=True,
            )
        ),
    )
    result = TemplateEvaluator(EvaluationConfig(mu_target=1.0)).evaluate(template)
    assert result.status == "uncertain" and result.margin_gamma is None
    assert result.margin is not None
    assert result.margin.gamma == 0.1
    assert result.margin.residuals["cluster_eigenvalue"] > 0.5


def test_tiny_eigenvalue_is_uncertain(monkeypatch):
    template = PhaseTemplate(np.full(4, 0.25), np.arange(4) * np.pi / 2)
    P = np.eye(4) + np.roll(np.eye(4), 1, axis=0) + np.roll(np.eye(4), -1, axis=0)
    monkeypatch.setattr(
        module, "optimize_template", Mock(return_value=density_result(template, P))
    )
    result = TemplateEvaluator(EvaluationConfig(use_margin_fallback=False)).evaluate(
        template
    )
    assert result.status == "uncertain" and result.stable_mu is None


@pytest.mark.parametrize("P", [np.ones((2, 2)), np.full((5, 5), np.nan)])
def test_bad_matrix_shape_or_nonfinite(ring, monkeypatch, P):
    monkeypatch.setattr(
        module,
        "optimize_template",
        Mock(return_value=SDPResult("optimal", P=P, mu=0.6)),
    )
    result = TemplateEvaluator(EvaluationConfig(use_margin_fallback=False)).evaluate(
        ring[0]
    )
    assert result.status == "uncertain" and result.stable_mu is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mu_target": np.nan},
        {"internal_margin": 0},
        {"cluster_margin": -1},
        {"residual_tolerance": np.inf},
        {"stability_tolerance": -1},
        {"max_clusters": 1},
    ],
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        EvaluationConfig(**kwargs)


@pytest.mark.integration
def test_real_ring_and_rotation(ring):
    template, P = ring
    evaluator = TemplateEvaluator()
    result = evaluator.evaluate(template, allowed=P.astype(bool))
    rotated = evaluator.evaluate(
        PhaseTemplate(template.x, template.phi + 0.123), allowed=P.astype(bool)
    )
    assert result.status == rotated.status == "stable"
    assert result.stable_mu == pytest.approx(0.6, abs=1e-6)
    assert rotated.stable_mu == pytest.approx(result.stable_mu, abs=1e-7)
    assert result.density is not None
    assert result.density.min_cluster_eigenvalue is not None
    assert result.density.min_cluster_eigenvalue > 0
