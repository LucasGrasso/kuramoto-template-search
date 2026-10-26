"""Test support: a deterministic, solver-free evaluator."""

import numpy as np
from numpy.typing import NDArray

from kuramoto_template.search.evaluator import (
    EvaluationConfig,
    SolverDiagnostics,
    TemplateEvaluation,
)
from kuramoto_template.template import PhaseTemplate


class FakeEvaluator:
    """Deterministic landscape used to test orchestration, not physics.

    Cluster count selects the status, so canonical rotation/permutation
    equivalence produces the same answer. Solver calls follow the real
    evaluator's 1-or-2-call pattern, including budget truncation.
    """

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig(max_clusters=12)
        self.calls = 0
        self.templates: list[PhaseTemplate] = []

    def evaluate(
        self,
        template: PhaseTemplate,
        *,
        allowed: NDArray[np.bool_] | None = None,
        max_solver_calls: int | None = None,
    ) -> TemplateEvaluation:
        del allowed
        self.calls += 1
        self.templates.append(template)
        m = template.n_clusters

        if m % 3 == 2:
            density = SolverDiagnostics(
                "density",
                "optimal",
                mu=0.6 + 0.01 * m,
                numerically_valid=True,
                stable=True,
                solver_agrees=True,
            )
            return TemplateEvaluation(
                template,
                "stable",
                stable_mu=0.6 + 0.01 * m,
                density=density,
                sdp_calls=1,
            )

        if m % 3 == 1:
            density = SolverDiagnostics(
                "density",
                "infeasible",
                numerically_valid=False,
                stable=False,
                solver_agrees=False,
            )
            if max_solver_calls is not None and max_solver_calls < 2:
                return TemplateEvaluation(
                    template,
                    "infeasible",
                    density=density,
                    sdp_calls=1,
                    message="truncated before margin fallback",
                )
            gamma = -0.01 * m
            margin = SolverDiagnostics(
                "margin",
                "optimal",
                mu=0.693,
                gamma=gamma,
                numerically_valid=True,
                stable=False,
                solver_agrees=True,
            )
            return TemplateEvaluation(
                template,
                "near_feasible",
                margin_gamma=gamma,
                density=density,
                margin=margin,
                sdp_calls=2,
            )

        density = SolverDiagnostics(
            "density",
            "solver_error",
            numerically_valid=False,
            stable=False,
            solver_agrees=False,
            message="fake solver failure",
        )
        return TemplateEvaluation(
            template,
            "solver_error",
            density=density,
            sdp_calls=1,
            message="fake solver failure",
        )
