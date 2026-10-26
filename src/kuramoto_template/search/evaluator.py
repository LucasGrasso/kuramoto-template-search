"""Independent numerical screening around the existing SDP implementations.

No result here certifies an exact equilibrium or finite graph construction.
"""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from kuramoto_template.template import PhaseTemplate
from kuramoto_template.sdp_optimizer.sdp import optimize_template
from kuramoto_template.sdp_optimizer.sdp_result import SDPResult
from kuramoto_template.sdp_optimizer.stability_sdp import maximize_stability_margin
from kuramoto_template.sdp_optimizer.stability_sdp_result import StabilitySDPResult
from kuramoto_template.sdp_optimizer.template_metrics import template_metrics
from kuramoto_template.sdp_optimizer.validate_input import in_open_semicircle, validate_input


class _SolverArguments(TypedDict):
    allowed: NDArray[np.bool_] | None
    force_cliques: bool
    solver: str
    solver_options: dict[str, Any]
    residual_tolerance: float


@dataclass(frozen=True)
class EvaluationConfig:
    mu_target: float = 0.693
    internal_margin: float = 1e-7
    cluster_margin: float = 1e-8
    residual_tolerance: float = 1e-7
    stability_tolerance: float = 1e-10
    force_cliques: bool = True
    use_margin_fallback: bool = True
    solver: str = "CLARABEL"
    solver_options: dict[str, Any] = field(default_factory=dict)
    max_clusters: int = 12

    def __post_init__(self) -> None:
        if not np.isfinite(self.mu_target) or not 0 <= self.mu_target <= 1:
            raise ValueError("mu_target must be finite and in [0, 1]")
        for name in ("internal_margin", "cluster_margin", "residual_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.stability_tolerance) or self.stability_tolerance < 0:
            raise ValueError("stability_tolerance must be finite and nonnegative")
        if (
            isinstance(self.max_clusters, bool)
            or not isinstance(self.max_clusters, int)
            or self.max_clusters < 2
        ):
            raise ValueError("max_clusters must be an integer >= 2")
        object.__setattr__(self, "solver_options", dict(self.solver_options))


@dataclass(eq=False)
class SolverDiagnostics:
    source: Literal["density", "margin"]
    status: str
    P: NDArray[np.float64] | None = None
    mu: float | None = None
    gamma: float | None = None
    max_force_error: float | None = None
    min_restoring: float | None = None
    min_cluster_eigenvalue: float | None = None
    residuals: dict[str, float] = field(default_factory=dict)
    numerically_valid: bool = False
    stable: bool = False
    solver_agrees: bool = False
    solve_time: float | None = None
    elapsed_seconds: float = 0.0
    message: str = ""
    optimized_mu: float | None = None
    objective_gap: float | None = None


@dataclass(eq=False)
class TemplateEvaluation:
    template: PhaseTemplate
    status: str
    stable_mu: float | None = None
    margin_gamma: float | None = None
    density: SolverDiagnostics | None = None
    margin: SolverDiagnostics | None = None
    sdp_calls: int = 0
    elapsed_seconds: float = 0.0
    message: str = ""
    cached: bool = False

    @property
    def P_source(self) -> str | None:
        if self.stable_mu is not None:
            return "density"
        if self.margin is not None and self.margin.P is not None:
            return "margin"
        return (
            "density"
            if self.density is not None and self.density.P is not None
            else None
        )

    @property
    def P(self) -> NDArray[np.float64] | None:
        diagnostic = self.density if self.P_source == "density" else self.margin
        return diagnostic.P if diagnostic is not None else None


class TemplateEvaluator:
    """Evaluate without caching or rounding subtly different phase templates.

    A stable density requires agreement between the solver flag and independent
    checks. Valid margin solutions remain exploration signals, even if positive;
    only the density solve sets stable_mu. max_solver_calls can suppress fallback
    to respect a caller's remaining budget. Calls count solver invocations,
    including exceptions, rather than internal conic iterations.
    """

    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config if config is not None else EvaluationConfig()

    def _diagnose(
        self,
        template: PhaseTemplate,
        result: SDPResult,
        source: Literal["density", "margin"],
        allowed: NDArray[np.bool_] | None,
    ) -> SolverDiagnostics:
        cfg = self.config
        diagnostic = SolverDiagnostics(
            source, result.status, solve_time=result.solve_time, message=result.message
        )
        if result.P is None:
            return diagnostic
        P = np.array(result.P, dtype=np.float64, copy=True)
        diagnostic.P = P
        if P.shape != (template.n_clusters, template.n_clusters) or not np.all(
            np.isfinite(P)
        ):
            diagnostic.message += " Invalid density matrix shape or nonfinite entries."
            return diagnostic
        metrics = template_metrics(template.x, template.phi, P)
        diagnostic.mu = metrics.mu
        diagnostic.optimized_mu = result.optimized_mu
        diagnostic.objective_gap = (abs(result.optimized_mu - metrics.mu)
                                    if result.optimized_mu is not None else None)
        diagnostic.max_force_error = metrics.max_force_error
        diagnostic.min_restoring = metrics.min_restoring
        diagnostic.min_cluster_eigenvalue = metrics.min_cluster_eigenvalue
        restoring_bound, cluster_bound = cfg.internal_margin, cfg.cluster_margin
        target = metrics.mu
        objective_ok = result.mu is not None and np.isfinite(result.mu)
        if source == "margin":
            if (
                not isinstance(result, StabilitySDPResult)
                or result.gamma is None
                or not np.isfinite(result.gamma)
            ):
                diagnostic.message += " Missing or nonfinite margin objective."
                return diagnostic
            diagnostic.gamma = float(result.gamma)
            restoring_bound = cluster_bound = diagnostic.gamma
            target = cfg.mu_target
            actual_target = result.target_mu
            objective_ok = bool(
                objective_ok
                and actual_target is not None
                and np.isfinite(actual_target)
                and abs(actual_target - cfg.mu_target) <= cfg.residual_tolerance
            )
        residuals = {
            "symmetry": float(np.max(np.abs(P - P.T))),
            "lower_bound": float(max(0, -P.min())),
            "upper_bound": float(max(0, P.max() - 1)),
            "forbidden": (
                float(np.max(np.abs(P[~allowed]), initial=0))
                if allowed is not None
                else 0.0
            ),
            "cliques": (
                float(np.max(np.abs(np.diag(P) - 1))) if cfg.force_cliques else 0.0
            ),
            "equilibrium": metrics.max_force_error,
            "degree": float(max(0, target - metrics.mu)),
            "restoring": float(max(0, restoring_bound - metrics.min_restoring)),
            "cluster_eigenvalue": float(
                max(0, cluster_bound - metrics.min_cluster_eigenvalue)
            ),
            "reported_mu": (
                abs(result.mu - metrics.mu)
                if objective_ok and result.mu is not None
                else float("inf")
            ),
        }
        if source == "density" and diagnostic.objective_gap is not None:
            residuals["optimized_objective"] = diagnostic.objective_gap
        diagnostic.residuals = residuals
        diagnostic.numerically_valid = bool(
            result.status in ("optimal", "optimal_inaccurate")
            and objective_ok
            and all(
                np.isfinite(r) and r <= cfg.residual_tolerance
                for r in residuals.values()
            )
            and (source != "density" or (
                residuals["restoring"] <= min(cfg.residual_tolerance, 0.05 * cfg.internal_margin)
                and residuals["cluster_eigenvalue"] <= min(cfg.residual_tolerance, 0.05 * cfg.cluster_margin)
            ))
        )
        positive = (
            metrics.min_restoring > cfg.stability_tolerance
            and metrics.min_cluster_eigenvalue > cfg.stability_tolerance
            and not in_open_semicircle(template.phi)
        )
        if in_open_semicircle(template.phi):
            diagnostic.message += " Open-semicircle phases cannot form an exact strictly stable nonsynchronous equilibrium."
        if source == "density":
            diagnostic.solver_agrees = bool(
                result.numerically_feasible
                == (diagnostic.numerically_valid and positive)
            )
        else:
            assert isinstance(result, StabilitySDPResult)
            margin_stable = (
                positive
                and diagnostic.gamma is not None
                and diagnostic.gamma > cfg.stability_tolerance
            )
            diagnostic.solver_agrees = bool(
                result.numerically_valid == diagnostic.numerically_valid
                and result.numerically_feasible
                == (diagnostic.numerically_valid and margin_stable)
            )
        diagnostic.stable = bool(
            diagnostic.numerically_valid
            and positive
            and diagnostic.solver_agrees
            and (
                source == "density"
                or (
                    diagnostic.gamma is not None
                    and diagnostic.gamma > cfg.stability_tolerance
                )
            )
        )
        if not diagnostic.numerically_valid or not diagnostic.solver_agrees:
            diagnostic.message += " Independent numerical screening failed or disagreed with solver flags."
        return diagnostic

    def evaluate(
        self,
        template: PhaseTemplate,
        *,
        allowed: NDArray[np.bool_] | None = None,
        max_solver_calls: int | None = None,
    ) -> TemplateEvaluation:
        started = perf_counter()
        evaluation = TemplateEvaluation(template, "invalid")
        cfg = self.config
        if max_solver_calls is not None and max_solver_calls < 1:
            raise ValueError("max_solver_calls must be positive")
        try:
            if template.n_clusters > cfg.max_clusters:
                raise ValueError("Template exceeds max_clusters")
            _, _, allowed = validate_input(template.x, template.phi, allowed)
            if (
                allowed is not None
                and cfg.force_cliques
                and not np.all(np.diag(allowed))
            ):
                raise ValueError("force_cliques conflicts with forbidden diagonal")
        except ValueError as exc:
            evaluation.message = str(exc)
            evaluation.elapsed_seconds = perf_counter() - started
            return evaluation

        def call(source: Literal["density", "margin"]) -> SolverDiagnostics:
            call_started = perf_counter()
            evaluation.sdp_calls += 1
            try:
                common: _SolverArguments = {
                    "allowed": allowed,
                    "force_cliques": cfg.force_cliques,
                    "solver": cfg.solver,
                    "solver_options": dict(cfg.solver_options),
                    "residual_tolerance": cfg.residual_tolerance,
                }
                if source == "density":
                    result = optimize_template(
                        template.x,
                        template.phi,
                        internal_margin=cfg.internal_margin,
                        cluster_margin=cfg.cluster_margin,
                        **common,
                    )
                else:
                    result = maximize_stability_margin(
                        template.x,
                        template.phi,
                        cfg.mu_target,
                        stability_tolerance=cfg.stability_tolerance,
                        **common,
                    )
                diagnostic = self._diagnose(template, result, source, allowed)
            except (KeyboardInterrupt, SystemExit, GeneratorExit, MemoryError):
                raise
            except BaseException as exc:  # noqa: BLE001 - conic-solver panics must not abort a run
                # pyo3-backed solvers raise PanicException, which inherits from
                # BaseException rather than Exception. A single failed solve is
                # recorded as a diagnostic; it must not kill a long benchmark.
                diagnostic = SolverDiagnostics(
                    source,
                    "solver_error",
                    message=f"{type(exc).__name__}: {exc}",
                )
            diagnostic.elapsed_seconds = perf_counter() - call_started
            return diagnostic

        evaluation.density = call("density")
        if evaluation.density.stable:
            evaluation.stable_mu = evaluation.density.mu
            evaluation.status = "stable"
        else:
            if cfg.use_margin_fallback and (
                max_solver_calls is None or evaluation.sdp_calls < max_solver_calls
            ):
                evaluation.margin = call("margin")
            diagnostics = [
                d for d in (evaluation.density, evaluation.margin) if d is not None
            ]
            if (
                evaluation.margin is not None
                and evaluation.margin.numerically_valid
                and evaluation.margin.solver_agrees
            ):
                evaluation.margin_gamma = evaluation.margin.gamma
                evaluation.status = "near_feasible"
            elif any(
                d.P is not None or d.status in ("optimal", "optimal_inaccurate")
                for d in diagnostics
            ):
                evaluation.status = "uncertain"
            elif any(d.status == "solver_error" for d in diagnostics):
                evaluation.status = "solver_error"
            elif all(
                d.status in ("infeasible", "infeasible_inaccurate") for d in diagnostics
            ):
                evaluation.status = "infeasible"
            else:
                evaluation.status = "uncertain"
        evaluation.message = " | ".join(
            f"{d.source}: {d.message}"
            for d in (evaluation.density, evaluation.margin)
            if d is not None and d.message
        )
        evaluation.elapsed_seconds = perf_counter() - started
        return evaluation
