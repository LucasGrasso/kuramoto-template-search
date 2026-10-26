"""Three-level numerical refinement for promising template candidates.

Levels
------
``fast``
    The evaluator used during search (float64 Clarabel, standard tolerances),
    reused for screening. It records no audit claim and never makes a second
    solve.
``refined``
    A fresh, uncached evaluator with tighter SDP margins/tolerances plus an
    independent audit of the returned density matrix. No cache is involved, so
    diagnostics cannot belong to a different canonical representative.
``certified``
    The refined evaluation plus a conservative numerical gate: optimal solver
    status, strict stability margins, the target density, independent
    residual/eigenvalue checks, and agreement between the reported and
    independently recomputed objective. The solve itself requests twice the
    certification margin so the result has headroom.

What "certified" means here
---------------------------
It does **not** mean a mathematical proof. The gate only rejects candidates
that fail conservative numerical conditions. ``proof_available`` is always
``False``; interval arithmetic or a rigorous equilibrium/stability proof is
still required before any record claim.

``optimal_inaccurate`` is treated as a refinement candidate, never as a
counterexample. Refinement results are numerical evidence only.
"""

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from kuramoto_template.search.evaluator import (
    EvaluationConfig,
    SolverDiagnostics,
    TemplateEvaluation,
    TemplateEvaluator,
)
from kuramoto_template.template import PhaseTemplate

RefinementLevel = Literal["fast", "refined", "certified"]
INACCURATE_STATUSES = ("optimal_inaccurate", "infeasible_inaccurate")


class RefinementEvaluator(Protocol):
    config: EvaluationConfig

    def evaluate(
        self, template: PhaseTemplate, *, max_solver_calls: int | None = None
    ) -> TemplateEvaluation: ...


@dataclass(frozen=True)
class RefinementConfig:
    """Thresholds and solver settings for candidate refinement."""

    target_mu: float = 0.693
    mu_threshold: float = 0.69
    eigenvalue_threshold: float = 1e-6
    margin_threshold: float = 1e-6
    force_tolerance: float = 1e-8
    certification_margin: float = 1e-6
    max_clusters: int = 12
    solver: str = "CLARABEL"
    solver_options: dict[str, Any] = field(default_factory=dict)
    refined_internal_margin: float = 1e-8
    refined_cluster_margin: float = 1e-9
    refined_residual_tolerance: float = 1e-8
    refined_stability_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not np.isfinite(self.target_mu) or not 0 < self.target_mu < 1:
            raise ValueError("target_mu must be finite and in (0, 1)")
        if not np.isfinite(self.mu_threshold) or not 0 < self.mu_threshold < 1:
            raise ValueError("mu_threshold must be finite and in (0, 1)")
        for name in (
            "eigenvalue_threshold",
            "margin_threshold",
            "force_tolerance",
            "certification_margin",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "refined_internal_margin",
            "refined_cluster_margin",
            "refined_residual_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.refined_stability_tolerance)
            or self.refined_stability_tolerance < 0
        ):
            raise ValueError("refined_stability_tolerance must be nonnegative")
        if (
            isinstance(self.max_clusters, bool)
            or not isinstance(self.max_clusters, int)
            or self.max_clusters < 2
        ):
            raise ValueError("max_clusters must be an integer >= 2")
        if not self.solver:
            raise ValueError("solver must be nonempty")
        object.__setattr__(self, "solver_options", dict(self.solver_options))

    def evaluation_config(self, level: RefinementLevel) -> EvaluationConfig:
        if level == "fast":
            return EvaluationConfig(
                mu_target=self.target_mu,
                max_clusters=self.max_clusters,
                solver=self.solver,
                solver_options=dict(self.solver_options),
            )
        # The certified solve must *ask* for the strict margin, not merely test
        # for it. Requesting twice ``certification_margin`` leaves headroom for
        # solver tolerance, so the gate below can require the real margin
        # without rejecting a solution that sits exactly on its own constraint.
        cluster_margin = self.refined_cluster_margin
        internal_margin = self.refined_internal_margin
        if level == "certified":
            cluster_margin = max(
                cluster_margin, 2.0 * self.certification_margin
            )
            internal_margin = max(internal_margin, 2.0 * self.certification_margin)
        options = {}
        if self.solver.upper() == "CLARABEL":
            options.update(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=300)
        options.update(self.solver_options)
        return EvaluationConfig(
            mu_target=self.target_mu,
            internal_margin=internal_margin,
            cluster_margin=cluster_margin,
            residual_tolerance=self.refined_residual_tolerance,
            stability_tolerance=self.refined_stability_tolerance,
            max_clusters=self.max_clusters,
            solver=self.solver,
            solver_options=options,
        )


def trigger_reasons(
    evaluation: TemplateEvaluation, config: RefinementConfig
) -> list[str]:
    """Return why a candidate deserves refinement, or an empty list."""
    reasons: list[str] = []
    if (
        evaluation.stable_mu is not None
        and np.isfinite(evaluation.stable_mu)
        and evaluation.stable_mu > config.mu_threshold
    ):
        reasons.append(
            f"stable_mu {evaluation.stable_mu:.12g} > threshold {config.mu_threshold:.12g}"
        )
    density = evaluation.density
    if density is not None:
        if density.status in INACCURATE_STATUSES:
            reasons.append(f"density status is {density.status}")
        eigenvalue = density.min_cluster_eigenvalue
        if (
            eigenvalue is not None
            and np.isfinite(eigenvalue)
            and 0 < eigenvalue < config.eigenvalue_threshold
        ):
            reasons.append(
                f"small positive cluster eigenvalue {eigenvalue:.3e}"
            )
        if evaluation.status == "stable" and not density.stable:
            reasons.append("stable label lacks independent stability confirmation")
        if density.status == "optimal" and not density.numerically_valid:
            reasons.append("optimal density solve failed independent residual checks")
        force_error = density.max_force_error
        if (
            force_error is not None
            and np.isfinite(force_error)
            and force_error > config.force_tolerance
        ):
            reasons.append(f"force residual {force_error:.3e} exceeds refinement filter")
    margin = evaluation.margin
    if margin is not None:
        if margin.status in INACCURATE_STATUSES:
            reasons.append(f"margin status is {margin.status}")
        gamma = margin.gamma
        if gamma is not None and np.isfinite(gamma) and abs(gamma) < config.margin_threshold:
            reasons.append(f"near-zero target margin gamma {gamma:.3e}")
        if margin.status == "optimal" and not margin.numerically_valid:
            reasons.append("optimal margin solve failed independent residual checks")
    return reasons


@dataclass(frozen=True)
class RefinementCheck:
    name: str
    passed: bool
    value: float | None
    threshold: float | None
    message: str = ""


@dataclass(eq=False)
class RefinementResult:
    template: PhaseTemplate
    level: RefinementLevel
    evaluation: TemplateEvaluation
    trigger_reasons: list[str]
    checks: list[RefinementCheck]
    gate_passed: bool
    proof_available: bool = False
    message: str = ""

    @property
    def status(self) -> str:
        if self.level == "fast":
            return "screened" if self.gate_passed else "screening_incomplete"
        if self.level == "refined":
            return "refined" if self.gate_passed else "refinement_failed"
        return (
            "certified_numerical_gate_passed"
            if self.gate_passed
            else "certified_numerical_gate_failed"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arithmetic_precision": "float64",
            "rigorous_certificate": False,
            "level": self.level,
            "status": self.status,
            "proof_available": self.proof_available,
            "trigger_reasons": list(self.trigger_reasons),
            "checks": [asdict(check) for check in self.checks],
            "message": self.message,
            "template": {
                "x": self.template.x.tolist(),
                "phi": self.template.phi.tolist(),
                "n_clusters": self.template.n_clusters,
            },
            "evaluation": _evaluation_summary(self.evaluation),
        }


def _evaluation_summary(evaluation: TemplateEvaluation) -> dict[str, Any]:
    return {
        "status": evaluation.status,
        "stable_mu": evaluation.stable_mu,
        "margin_gamma": evaluation.margin_gamma,
        "sdp_calls": evaluation.sdp_calls,
        "cached": evaluation.cached,
        "message": evaluation.message,
        "density": _diagnostic_summary(evaluation.density),
        "margin": _diagnostic_summary(evaluation.margin),
    }


def _diagnostic_summary(diagnostic: SolverDiagnostics | None) -> dict[str, Any] | None:
    if diagnostic is None:
        return None
    return {
        "source": diagnostic.source,
        "status": diagnostic.status,
        "mu": diagnostic.mu,
        "gamma": diagnostic.gamma,
        "max_force_error": diagnostic.max_force_error,
        "min_restoring": diagnostic.min_restoring,
        "min_cluster_eigenvalue": diagnostic.min_cluster_eigenvalue,
        "residuals": dict(diagnostic.residuals),
        "numerically_valid": diagnostic.numerically_valid,
        "stable": diagnostic.stable,
        "solver_agrees": diagnostic.solver_agrees,
        "message": diagnostic.message,
    }


def _check(
    name: str,
    passed: bool,
    value: float | None,
    threshold: float | None,
    message: str = "",
) -> RefinementCheck:
    return RefinementCheck(name, bool(passed), value, threshold, message)


def _independent_metrics(
    template: PhaseTemplate,
    evaluation: TemplateEvaluation,
    tolerance: float,
    *, force_tolerance: float, required_margin: float | None,
) -> tuple[list[RefinementCheck], str]:
    """Recompute density metrics from whichever solve returned a matrix.

    A stable candidate audits its density solve. A near-feasible candidate may
    only have the margin solve's matrix, and that is fine: the margin signal is
    exploratory, so its own matrix is the relevant one. When neither solve
    returned a matrix there is nothing to recompute; the caller decides whether
    that is fatal for the requested level.
    """
    diagnostic = (
        evaluation.margin
        if evaluation.P_source == "margin"
        else evaluation.density
    )
    P = evaluation.P
    if diagnostic is None or P is None:
        return [], "no density matrix"
    P = np.asarray(P, dtype=np.float64)
    if P.shape != (template.n_clusters, template.n_clusters):
        return [
            _check(
                "density_matrix_shape",
                False,
                None,
                None,
                f"expected {(template.n_clusters, template.n_clusters)}, got {P.shape}",
            )
        ], "invalid density matrix shape"
    if not np.all(np.isfinite(P)):
        return [
            _check(
                "density_matrix_finite",
                False,
                None,
                None,
                "density matrix has nonfinite entries",
            )
        ], "nonfinite density matrix"
    source = evaluation.P_source or "unknown"
    # Separate implementation: explicit pair contributions and a normalized
    # Helmert basis rather than reusing the solver/evaluator metric helper.
    # This remains float64 numerical verification, not high precision/proof.
    from types import SimpleNamespace
    m = template.n_clusters
    degrees = np.zeros(m)
    forces = np.zeros(m)
    restoring = np.zeros(m)
    laplacian = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            delta = template.phi[a] - template.phi[b]
            degrees[a] += P[a, b] * template.x[b]
            forces[a] += P[a, b] * template.x[b] * np.sin(delta)
            restoring[a] += P[a, b] * template.x[b] * np.cos(delta)
            if a != b:
                weight = P[a, b] * template.x[a] * template.x[b] * np.cos(delta)
                laplacian[a, a] += weight
                laplacian[a, b] -= weight
    basis = np.zeros((m, m-1))
    for k in range(1, m):
        basis[:k, k-1] = 1 / np.sqrt(k * (k+1))
        basis[k, k-1] = -k / np.sqrt(k * (k+1))
    restricted = basis.T @ laplacian @ basis
    metrics = SimpleNamespace(mu=float(degrees.min()), max_force_error=float(np.abs(forces).max()),
                              min_restoring=float(restoring.min()),
                              min_cluster_eigenvalue=float(np.linalg.eigvalsh((restricted+restricted.T)/2).min()))
    checks = [
        _check("independent_equilibrium_threshold", metrics.max_force_error <= force_tolerance,
               metrics.max_force_error, force_tolerance),
        _check("independent_restoring", diagnostic.min_restoring is not None
               and abs(diagnostic.min_restoring - metrics.min_restoring) <= tolerance,
               metrics.min_restoring, tolerance),
        _check(
            "independent_mu",
            diagnostic.mu is not None
            and np.isfinite(diagnostic.mu)
            and abs(float(diagnostic.mu) - metrics.mu) <= 1e-9,
            metrics.mu,
            None,
            f"recomputed minimum degree vs the reported {source} objective",
        ),
        _check(
            "independent_force_residual",
            abs(float(diagnostic.max_force_error or 0.0) - metrics.max_force_error)
            <= 1e-9,
            metrics.max_force_error,
            None,
        ),
        _check(
            "independent_cluster_eigenvalue",
            diagnostic.min_cluster_eigenvalue is not None
            and abs(
                float(diagnostic.min_cluster_eigenvalue)
                - metrics.min_cluster_eigenvalue
            )
            <= 1e-9,
            metrics.min_cluster_eigenvalue,
            None,
        ),
        _check(
            "density_matrix_symmetric",
            float(np.max(np.abs(P - P.T))) <= tolerance,
            float(np.max(np.abs(P - P.T))),
            tolerance,
        ),
        _check(
            "density_entries_in_unit_interval",
            bool(
                np.all(P >= -tolerance)
                and np.all(P <= 1 + tolerance)
            ),
            None,
            tolerance,
        ),
    ]
    if required_margin is not None:
        checks.extend([
            _check("independent_positive_restoring", metrics.min_restoring > required_margin,
                   metrics.min_restoring, required_margin),
            _check("independent_positive_cluster_eigenvalue", metrics.min_cluster_eigenvalue > required_margin,
                   metrics.min_cluster_eigenvalue, required_margin),
        ])
    return checks, ""


def _stability_checks(
    evaluation: TemplateEvaluation, config: RefinementConfig, level: RefinementLevel
) -> list[RefinementCheck]:
    density = evaluation.density
    if density is None:
        return [
            _check(
                "density_solve_present",
                False,
                None,
                None,
                "no density diagnostics",
            )
        ]
    # ``refined`` asks whether the tighter solve confirms stability at solver
    # tolerance; ``certified`` requires a strict margin and the target density.
    margin = (
        config.certification_margin
        if level == "certified"
        else config.refined_stability_tolerance
    )
    checks = [
        _check(
            "numerically_valid",
            density.numerically_valid,
            None,
            None,
            "independent residual screening",
        ),
        _check(
            "solver_agrees",
            density.solver_agrees,
            None,
            None,
            "solver feasibility flag vs independent screening",
        ),
        _check(
            "strictly_stable",
            evaluation.status == "stable",
            None,
            None,
            f"status is {evaluation.status}",
        ),
        _check(
            "stable_mu_matches_independent_metrics",
            evaluation.stable_mu is not None
            and density.mu is not None
            and abs(float(evaluation.stable_mu) - float(density.mu)) <= 1e-9,
            density.mu,
            None,
            "stable_mu must equal the independently recomputed minimum degree",
        ),
        _check(
            "force_equilibrium",
            density.max_force_error is not None
            and density.max_force_error <= config.force_tolerance,
            density.max_force_error,
            config.force_tolerance,
        ),
        _check(
            "positive_cluster_eigenvalue",
            density.min_cluster_eigenvalue is not None
            and density.min_cluster_eigenvalue > margin,
            density.min_cluster_eigenvalue,
            margin,
        ),
        _check(
            "positive_restoring_force",
            density.min_restoring is not None
            and density.min_restoring > margin,
            density.min_restoring,
            margin,
        ),
    ]
    if level == "certified":
        checks.append(
            _check(
                "target_density",
                evaluation.stable_mu is not None
                and evaluation.stable_mu >= config.target_mu,
                evaluation.stable_mu,
                config.target_mu,
            )
        )
        checks.append(
            _check(
                "optimal_solver_status",
                density.status == "optimal",
                None,
                None,
                f"status is {density.status}",
            )
        )
        checks.append(
            _check(
                "reported_mu_residual",
                density.residuals.get("reported_mu", np.inf)
                <= config.refined_residual_tolerance,
                density.residuals.get("reported_mu"),
                config.refined_residual_tolerance,
            )
        )
    return checks


def audit_candidate(
    template: PhaseTemplate,
    evaluation: TemplateEvaluation,
    config: RefinementConfig,
    level: RefinementLevel,
) -> tuple[list[RefinementCheck], bool, str]:
    """Run independent checks against one evaluation.

    ``certified`` requires every strict stability/target check. ``refined``
    reports the same checks but only requires the independent audit and solver
    agreement, so near-feasible exploratory candidates remain reviewable.
    """
    independent, problem = _independent_metrics(
        template, evaluation, config.refined_residual_tolerance,
        force_tolerance=config.force_tolerance,
        required_margin=(config.certification_margin if level == "certified" else
                         config.refined_stability_tolerance if evaluation.status == "stable" else None),
    )
    checks = [
        _check(
            "mass_sum",
            abs(float(template.x.sum()) - 1.0) <= 1e-10,
            float(template.x.sum()),
            1.0,
        ),
        _check(
            "finite_inputs",
            bool(np.all(np.isfinite(template.x)) and np.all(np.isfinite(template.phi))),
            None,
            None,
        ),
        *independent,
    ]
    if problem and problem != "no density matrix":
        return checks, False, problem

    if level == "fast":
        # Fast level reuses the search evaluation verbatim. It is a screening
        # record, not an audit: no cache bypass and no stability claim.
        checks.append(
            _check(
                "screening_only",
                True,
                None,
                None,
                "fast level reuses the search evaluation and makes no audit claim",
            )
        )
        return checks, all(check.passed for check in checks), ""

    if problem and (
        level == "certified"
        or (level == "refined" and evaluation.status == "stable")
    ):
        # A stability claim needs a density matrix to audit independently.
        checks.append(
            _check(
                "density_matrix_present",
                False,
                None,
                None,
                "no density matrix is available for an independent audit",
            )
        )
        return checks, False, problem

    if level == "certified":
        checks.extend(_stability_checks(evaluation, config, level))
        return checks, all(check.passed for check in checks), ""

    # Refined level: accept stable or near-feasible outcomes, but require an
    # honest signal diagnostic and no inaccurate-status upgrade. A near-feasible
    # candidate whose margin solve returned no matrix is still reviewable; its
    # gate then rests on the solver's own diagnostics, not an independent P.
    density = evaluation.density
    margin = evaluation.margin
    signal = (
        margin
        if evaluation.status == "near_feasible" and margin is not None
        else density
    )
    if signal is None:
        checks.append(
            _check(
                "signal_diagnostic_present",
                False,
                None,
                None,
                "no density or margin diagnostics",
            )
        )
        return checks, False, "missing signal diagnostic"
    checks.extend(
        [
            _check(
                "signal_numerically_valid",
                signal.numerically_valid,
                None,
                None,
                "independent residual screening for the signal solve",
            ),
            _check(
                "signal_status_accurate",
                signal.status not in INACCURATE_STATUSES,
                None,
                None,
                f"status is {signal.status}; inaccurate results require review",
            ),
            _check(
                "evaluated_by_solver",
                evaluation.sdp_calls > 0 and not evaluation.cached,
                float(evaluation.sdp_calls),
                1.0,
                "refinement must bypass the search cache",
            ),
        ]
    )
    if evaluation.status == "stable":
        checks.extend(_stability_checks(evaluation, config, level))
    elif margin is not None:
        checks.append(
            _check(
                "margin_signal_present",
                margin.gamma is not None and np.isfinite(margin.gamma),
                margin.gamma,
                None,
            )
        )
    else:
        checks.append(
            _check(
                "recognized_outcome",
                evaluation.status in ("stable", "near_feasible"),
                None,
                None,
                f"status is {evaluation.status}",
            )
        )
    message = (
        ""
        if not problem
        else "no density matrix; audit limited to solver diagnostics"
    )
    return checks, all(check.passed for check in checks), message


def _check_cluster_limit(
    templates: Sequence[tuple[str, PhaseTemplate]], config: RefinementConfig
) -> None:
    """Fail fast when a candidate exceeds the configured cluster limit.

    The evaluator would otherwise return ``invalid`` for every oversized
    candidate, which looks like a failed audit instead of a configuration
    mistake. This is especially easy to hit when refining runs that searched
    with a raised ``--max-clusters``.
    """
    oversized = [
        (source, template.n_clusters)
        for source, template in templates
        if template.n_clusters > config.max_clusters
    ]
    if not oversized:
        return
    source, largest = max(oversized, key=lambda item: item[1])
    raise ValueError(
        f"{len(oversized)} candidate(s) exceed max_clusters={config.max_clusters} "
        f"(largest has {largest} clusters in {source}); "
        f"pass --max-clusters {largest} or higher"
    )


def refine_candidate(
    template: PhaseTemplate,
    config: RefinementConfig,
    *,
    level: RefinementLevel = "refined",
    evaluator: RefinementEvaluator | None = None,
) -> RefinementResult:
    """Re-evaluate and audit one candidate with a fresh evaluator."""
    _check_cluster_limit([("<candidate>", template)], config)
    target: RefinementEvaluator = (
        evaluator
        if evaluator is not None
        else TemplateEvaluator(config.evaluation_config(level))
    )
    evaluation = target.evaluate(template)
    trigger = trigger_reasons(evaluation, config)
    checks, passed, problem = audit_candidate(template, evaluation, config, level)
    return RefinementResult(
        template=template,
        level=level,
        evaluation=evaluation,
        trigger_reasons=trigger,
        checks=checks,
        gate_passed=passed,
        proof_available=False,
        message=problem,
    )


@dataclass(frozen=True)
class CandidateRefinement:
    """One source candidate and its optional refinement."""

    source: str
    trigger_reasons: list[str]
    fast_status: str
    fast_stable_mu: float | None
    fast_margin_gamma: float | None
    refined: RefinementResult | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "trigger_reasons": list(self.trigger_reasons),
            "fast_status": self.fast_status,
            "fast_stable_mu": self.fast_stable_mu,
            "fast_margin_gamma": self.fast_margin_gamma,
            "refined": None if self.refined is None else self.refined.to_dict(),
        }


def refine_candidates(
    templates: Sequence[tuple[str, PhaseTemplate]],
    config: RefinementConfig,
    *,
    level: RefinementLevel = "refined",
    force: bool = False,
    evaluator_factory: Callable[
        [RefinementConfig, RefinementLevel], RefinementEvaluator
    ] | None = None,
) -> tuple[list[CandidateRefinement], list[str]]:
    """Fast-screen candidates, then refine triggered ones with fresh solves.

    ``evaluator_factory`` exists for tests; the default creates a fresh
    :class:`TemplateEvaluator`, never a cache.
    """
    _check_cluster_limit(templates, config)

    def factory(
        cfg: RefinementConfig, requested: RefinementLevel
    ) -> RefinementEvaluator:
        if evaluator_factory is not None:
            return evaluator_factory(cfg, requested)
        return TemplateEvaluator(cfg.evaluation_config(requested))

    fast = factory(config, "fast")
    refined_evaluator = factory(config, level)
    results: list[CandidateRefinement] = []
    warnings: list[str] = []
    for source, template in templates:
        fast_evaluation = fast.evaluate(template)
        reasons = trigger_reasons(fast_evaluation, config)
        refined: RefinementResult | None = None
        if force or reasons:
            if level == "fast":
                checks, passed, problem = audit_candidate(
                    template, fast_evaluation, config, "fast"
                )
                refined = RefinementResult(
                    template=template,
                    level="fast",
                    evaluation=fast_evaluation,
                    trigger_reasons=reasons,
                    checks=checks,
                    gate_passed=passed,
                    proof_available=False,
                    message=problem,
                )
            else:
                refined = refine_candidate(
                    template, config, level=level, evaluator=refined_evaluator
                )
            if not refined.proof_available and level == "certified":
                warnings.append(
                    f"{source}: numerical gate only; interval/proof certification "
                    "is still required"
                )
        results.append(
            CandidateRefinement(
                source=source,
                trigger_reasons=reasons,
                fast_status=fast_evaluation.status,
                fast_stable_mu=fast_evaluation.stable_mu,
                fast_margin_gamma=fast_evaluation.margin_gamma,
                refined=refined,
            )
        )
    return results, warnings


def load_templates(path: str | Path) -> list[tuple[str, PhaseTemplate]]:
    """Load candidates from a result.json file, a trace.jsonl file, or a run dir."""
    root = Path(path)
    if root.is_dir():
        candidate = root / "result.json"
        if not candidate.exists():
            raise ValueError(
                f"{root} does not contain result.json; pass a run directory or file"
            )
        root = candidate
    if root.name == "result.json":
        data = json.loads(root.read_text(encoding="utf-8"))
        templates: list[tuple[str, PhaseTemplate]] = []
        seen: set[tuple[bytes, bytes]] = set()
        entries = [data.get("best")] if data.get("best") is not None else []
        entries.extend(data.get("archive", []))
        for index, entry in enumerate(entries):
            template_data = entry.get("template") if isinstance(entry, dict) else None
            if not isinstance(template_data, dict):
                continue
            template = PhaseTemplate(
                np.asarray(template_data["x"], dtype=np.float64),
                np.asarray(template_data["phi"], dtype=np.float64),
            )
            signature = (template.x.tobytes(), template.phi.tobytes())
            if signature in seen:
                continue
            seen.add(signature)
            templates.append((f"{root}:{index}", template))
        return templates
    if root.name == "trace.jsonl":
        templates = []
        seen = set()
        for line_number, line in enumerate(
            root.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            template_data = record.get("template")
            if not isinstance(template_data, dict):
                continue
            template = PhaseTemplate(
                np.asarray(template_data["x"], dtype=np.float64),
                np.asarray(template_data["phi"], dtype=np.float64),
            )
            signature = (template.x.tobytes(), template.phi.tobytes())
            if signature in seen:
                continue
            seen.add(signature)
            templates.append((f"{root}:{line_number}", template))
        return templates
    raise ValueError(
        "Expected a run directory, result.json, or trace.jsonl, "
        f"got {root}"
    )


def _write_json(path: Path, data: Any) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return safe(value.tolist())
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(safe(data), handle, indent=2, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        help="Run directory, result.json, or trace.jsonl",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New refinement JSON file"
    )
    parser.add_argument(
        "--level",
        choices=("fast", "refined", "certified"),
        default="refined",
    )
    parser.add_argument("--force", action="store_true", help="Refine every candidate")
    parser.add_argument("--target-mu", type=float, default=0.693)
    parser.add_argument("--mu-threshold", type=float, default=0.69)
    parser.add_argument("--eigenvalue-threshold", type=float, default=1e-6)
    parser.add_argument("--margin-threshold", type=float, default=1e-6)
    parser.add_argument("--force-tolerance", type=float, default=1e-8)
    parser.add_argument("--certification-margin", type=float, default=1e-6)
    parser.add_argument("--max-clusters", type=int, default=12)
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument(
        "--solver-options",
        type=json.loads,
        default={},
        help="JSON object of conic solver settings",
    )
    parser.add_argument("--refined-internal-margin", type=float, default=1e-8)
    parser.add_argument("--refined-cluster-margin", type=float, default=1e-9)
    parser.add_argument("--refined-residual-tolerance", type=float, default=1e-8)
    parser.add_argument("--refined-stability-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    try:
        config = RefinementConfig(
            target_mu=args.target_mu,
            mu_threshold=args.mu_threshold,
            eigenvalue_threshold=args.eigenvalue_threshold,
            margin_threshold=args.margin_threshold,
            force_tolerance=args.force_tolerance,
            certification_margin=args.certification_margin,
            max_clusters=args.max_clusters,
            solver=args.solver,
            solver_options=args.solver_options,
            refined_internal_margin=args.refined_internal_margin,
            refined_cluster_margin=args.refined_cluster_margin,
            refined_residual_tolerance=args.refined_residual_tolerance,
            refined_stability_tolerance=args.refined_stability_tolerance,
        )
        templates = load_templates(args.result)
        results, warnings = refine_candidates(
            templates,
            config,
            level=args.level,
            force=args.force,
        )
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))

    _write_json(
        args.output,
        {
            "schema_version": 1,
            "source": str(args.result),
            "config": {**asdict(config), "level": args.level, "force": args.force},
            "numerical_evidence_only": True,
            "proof_available": False,
            "candidates": [result.to_dict() for result in results],
            "warnings": warnings,
        },
    )
    refined_count = sum(1 for result in results if result.refined is not None)
    print(
        json.dumps(
            {
                "candidates": len(results),
                "refined": refined_count,
                "gate_passed": sum(
                    1
                    for result in results
                    if result.refined is not None and result.refined.gate_passed
                ),
                "certified": 0,
                "warnings": len(warnings),
                "output": str(args.output),
            }
        )
    )
    print("Numerical gate only; no rigorous certificate.")


if __name__ == "__main__":
    main()
