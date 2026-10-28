"""Joint convex limiting density/correction optimization in a fixed gauge.

Only x, s, labels, support, fixed entries and the guard scale are inputs. There
are no conditional/bilinear tangent constraints for the optimized densities.
All strict statuses are numerical, never rigorous certificates.
"""

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

import cvxpy as cp
import numpy as np
from scipy.linalg import eigvalsh

from .expansions import (U4, coefficients, effective_matrix, full_rescaled_matrix,
                         internal_basis, reference_rescaled_matrix)
from .families import winding
from .models import LimitingCandidate, Numerics
from .solvers import SolverLedger, solve
from .physical import mass_quotient_metrics


@dataclass(frozen=True)
class JointLimitConfig:
    mode: str = "density"
    mu_target: float = .6930
    gamma0: float = 1e-6
    hard_floor: float = 1e-6
    restoring_floor: float = 1e-4
    energy_floor: float = 1e-7
    h_guard: float = .003
    r_max: float = 30.
    backbone_density: float = 1e-4
    inward_slack: float = 1e-3
    repair_density: float = 1e-3
    box_buffer: float = 1e-8
    residual_tolerance: float = 1e-9
    strict_tolerance: float = 1e-8
    rank_tolerance: float = 1e-12
    neutral: bool = False
    secondary: bool = True
    density_allowance: float = 1e-7
    margin_metric: str = "euclidean"
    solver: str = "CLARABEL"
    solver_options: dict[str, Any] = field(default_factory=lambda: dict(
        tol_gap_abs=1e-11, tol_gap_rel=1e-11, tol_feas=1e-11, max_iter=500))

    def __post_init__(self):
        if self.mode not in ("density", "margin"):
            raise ValueError("mode must be density or margin")
        for name in self.__dataclass_fields__:
            if name in ("mode", "neutral", "secondary", "solver", "solver_options", "margin_metric"):
                continue
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not self.mu_target <= 1 or self.h_guard >= 1 or self.repair_density >= 1:
            raise ValueError("Invalid target, guard, or repair density")
        if self.margin_metric not in ("euclidean","mass"):
            raise ValueError("margin_metric must be euclidean or mass")
        object.__setattr__(self, "solver_options", dict(self.solver_options))

    def lift_numerics(self):
        return Numerics(residual_tolerance=self.residual_tolerance,
                        rank_tolerance=self.rank_tolerance,
                        strict_gap=self.strict_tolerance,
                        energy_floor=self.energy_floor,
                        connectivity_margin=self.hard_floor,
                        backbone_density=self.backbone_density,
                        r_max=self.r_max, trial_h=self.h_guard,
                        solver=self.solver, solver_options=self.solver_options)


@dataclass
class JointAffineModel:
    geometry: LimitingCandidate
    p_pairs: tuple
    r_pairs: tuple
    p_map: np.ndarray
    r_map: np.ndarray
    f0_map: np.ndarray
    f1_p: np.ndarray
    f1_r: np.ndarray
    cut_map: np.ndarray
    energy_map: np.ndarray
    restoring_map: np.ndarray
    h_p: np.ndarray
    h_r: np.ndarray
    z: np.ndarray
    flow_map: np.ndarray

    def evaluate(self, p, r):
        pv = np.array([p[a, b] for a, b in self.p_pairs])
        rv = np.array([r[a, b] for a, b in self.r_pairs])
        m = self.geometry.m
        return dict(f0=self.f0_map @ pv, f1=self.f1_p @ pv + self.f1_r @ rv,
                    cuts=self.cut_map @ pv, flows=self.flow_map@pv, energy=float(self.energy_map @ pv),
                    restoring=self.geometry.x + self.restoring_map @ pv,
                    hlim=(self.h_p @ pv + self.h_r @ rv).reshape(m-1, m-1))


def build_affine_model(x, s, groups, mask, *, backbone_cycle=None):
    groups = np.asarray(groups)
    if backbone_cycle is None:
        if any(not np.any(groups == j) for j in range(4)):
            raise ValueError("All four groups are required")
        backbone_cycle = tuple(int(np.flatnonzero(groups == j)[0]) for j in range(4))
    c = LimitingCandidate(x, s, groups, np.eye(len(x)), mask, backbone_cycle, "joint-geometry")
    if np.ptp(c.s) <= 1e-12:
        raise ValueError("Nontrivial offsets are required")
    pairs = tuple((a, b) for a in range(c.m) for b in range(a+1, c.m) if c.allowed_mask[a, b])
    neighbors = tuple((a, b) for a, b in pairs if (c.groups[a]-c.groups[b]) % 2 == 1)
    m, n, nr, d = c.m, len(pairs), len(neighbors), c.m-1
    pm, rm = np.zeros((m*m, n)), np.zeros((m*m, nr))
    f0, f1p, f1r = np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, nr))
    K, I, restoring = np.zeros((4, n)), np.zeros(n), np.zeros((m, n))
    flows=np.zeros((4,n))
    hp, hr = np.zeros((d*d, n)), np.zeros((d*d, nr))
    Z, eye, eye4 = internal_basis(c), np.eye(m), np.eye(4)
    r_indices = {pair: i for i, pair in enumerate(neighbors)}
    for i, (a, b) in enumerate(pairs):
        pm[a*m+b, i] = pm[b*m+a, i] = 1
        mass, ds = c.x[a]*c.x[b], c.s[b]-c.s[a]
        H = np.zeros((d, d))
        if c.groups[a] == c.groups[b]:
            vec = Z.T @ (eye[a]-eye[b])
            H[3:, 3:] = mass*np.outer(vec, vec)
            I[i] = mass*ds*ds
            f1p[a, i], f1p[b, i] = c.x[b]*ds, -c.x[a]*ds
            restoring[a, i], restoring[b, i] = c.x[b], c.x[a]
        else:
            ri = r_indices[a, b]
            rm[a*m+b, ri] = rm[b*m+a, ri] = 1
            if (c.groups[b]-c.groups[a]) % 4 != 1:
                a, b = b, a
                ds = -ds
            j = c.groups[a]
            f0[a, i], f0[b, i] = c.x[b], -c.x[a]
            f1r[a, ri], f1r[b, ri] = c.x[b], -c.x[a]
            K[j, i] = mass*ds
            flows[j,i]=mass
            u = U4.T @ (eye4[j]-eye4[(j+1) % 4])
            z = Z.T @ (eye[a]-eye[b])
            H[:3, 3:] = -mass*ds*np.outer(u, z)
            H[3:, :3] = H[:3, 3:].T
            R = np.zeros((d, d))
            R[:3, :3] = -mass*ds*np.outer(u, u)
            hr[:, ri] = R.ravel()
        hp[:, i] = H.ravel()
    return JointAffineModel(c, pairs, neighbors, pm, rm, f0, f1p, f1r, K, I,
                            restoring, hp, hr, Z,flows)


@dataclass(eq=False)
class JointLimitResult:
    candidate: LimitingCandidate | None
    r: np.ndarray | None
    status: str
    solver_status: str
    config: JointLimitConfig
    diagnostics: dict = field(default_factory=dict)
    hlim: np.ndarray | None = None
    spectrum: np.ndarray | None = None
    mu0: float | None = None
    reported_objective: float | None = None
    solve_records: list = field(default_factory=list)
    repair_edges: tuple = ()
    fixed_p0: np.ndarray | None = None
    elapsed_seconds: float = 0.
    proof_available: bool = False
    certification_status: str = "not_attempted"
    message: str = ""
    flow_target: float | None = None

    def data(self):
        return asdict(self)


def result_from_data(data):
    data = dict(data)
    data["config"] = JointLimitConfig(**data["config"])
    if data["candidate"] is not None:
        data["candidate"] = LimitingCandidate(**data["candidate"])
    for name in ("r", "hlim", "spectrum", "fixed_p0"):
        if data.get(name) is not None:
            data[name] = np.asarray(data[name], dtype=float)
    data["repair_edges"] = tuple(tuple(e) for e in data["repair_edges"])
    return JointLimitResult(**data)


def _fixed_entries(c, fixed_p0):
    fixed = np.full((c.m, c.m), np.nan) if fixed_p0 is None else np.array(fixed_p0, dtype=float)
    if fixed.shape != c.p0.shape or not np.array_equal(fixed, fixed.T, equal_nan=True):
        raise ValueError("fixed_p0 must be symmetric, with NaN for free entries")
    given = np.isfinite(fixed)
    if np.any(np.isinf(fixed)) or np.any((fixed[given] < 0) | (fixed[given] > 1)):
        raise ValueError("Fixed densities must lie in [0,1]")
    if np.any(given & ~c.allowed_mask & (fixed != 0)):
        raise ValueError("Fixed entries conflict with support")
    if np.any(np.isfinite(np.diag(fixed)) & (np.diag(fixed) != 1)):
        raise ValueError("The clique diagonal is fixed at one")
    return fixed


def validate_joint(c, r, config=None, *, solver_status="analytic", fixed_p0=None,
                   repair_edges=(), reported_gamma=None,flow_target=None):
    cfg = config or JointLimitConfig()
    r = np.asarray(r, dtype=float)
    result = JointLimitResult(c, np.array(r, copy=True), "failed", solver_status, cfg,
                              repair_edges=tuple(repair_edges), fixed_p0=fixed_p0,flow_target=flow_target)
    if r.shape != c.p0.shape or not np.all(np.isfinite(r)):
        result.message = "Invalid correction matrix"
        return result
    adjacent = (c.groups[:, None]-c.groups[None, :]) % 2 == 1
    if np.any(r[~(adjacent & c.allowed_mask)] != 0) or not np.array_equal(r, r.T):
        result.message = "Correction violates exact structural constants"
        return result
    e = coefficients(c, r)
    H = full_rescaled_matrix(c, r)
    reference = reference_rescaled_matrix(c, r)
    affine = build_affine_model(c.x, c.s, c.groups, c.allowed_mask,
                                backbone_cycle=c.backbone_cycle).evaluate(c.p0, r)
    spectrum = np.linalg.eigvalsh(H)
    hard = H[3:, 3:]
    hard_gap = float(np.linalg.eigvalsh(hard).min()) if hard.size else None
    mass_metric,hard_metric,soft_metric=mass_quotient_metrics(c)
    physical_values=eigvalsh(H,mass_metric)
    hard_physical=float(eigvalsh(hard,hard_metric).min()) if hard.size else None
    selected_gap=float(physical_values.min()) if cfg.margin_metric=="mass" else float(spectrum.min())
    selected_hard=hard_physical if cfg.margin_metric=="mass" else hard_gap
    try:
        effective = U4.T @ effective_matrix(c, r, cfg.rank_tolerance) @ U4
        effective_gap = float(np.linalg.eigvalsh(effective).min())
        schur = H[:3, :3] - H[:3, 3:] @ np.linalg.solve(hard, H[3:, :3])
        schur_error = float(np.max(np.abs(schur-effective)))
    except (ValueError, np.linalg.LinAlgError):
        effective_gap, schur_error = None, float("inf")
    c0 = np.sum(c.p0*c.x[None, :]*(c.groups[:, None] == c.groups[None, :]), axis=1)
    trial = c.p0+cfg.h_guard*r
    fixed = _fixed_entries(c, fixed_p0)
    fixed_error = float(np.max(np.abs(c.p0[np.isfinite(fixed)]-fixed[np.isfinite(fixed)]), initial=0))
    free = c.allowed_mask & ~np.isfinite(fixed) & ~np.eye(c.m,dtype=bool)
    guard_neighbor = adjacent & c.allowed_mask
    buffer_violation = float(max(0., cfg.box_buffer-np.min(c.p0[free],initial=1),
        np.max(c.p0[free],initial=0)-(1-cfg.box_buffer),
        cfg.box_buffer-np.min(trial[guard_neighbor],initial=1),
        np.max(trial[guard_neighbor],initial=0)-(1-cfg.box_buffer)))
    box = float(max(0., -c.p0.min(), c.p0.max()-1, -trial.min(), trial.max()-1))
    permitted = adjacent & c.allowed_mask
    cone = float(max(0., -np.min(r[permitted & (c.p0 == 0)], initial=0),
                     np.max(r[permitted & (c.p0 == 1)], initial=0)))
    backbone_violation = max(0., *(cfg.backbone_density-c.p0[a,b] for a,b in zip(
        c.backbone_cycle,c.backbone_cycle[1:]+c.backbone_cycle[:1])))
    repair_violation = 0.
    for a, b in repair_edges:
        repair_violation = max(repair_violation, r[a, b]+cfg.inward_slack,
                               cfg.repair_density-c.p0[a, b], cfg.repair_density-trial[a, b])
    n_roundoff = float(256*np.finfo(float).eps*(c.m-1)*max(np.linalg.norm(H, 2), 1e-3))
    gamma = 0. if cfg.neutral else cfg.gamma0
    h_gap = float(spectrum.min())
    floor_violation = max(0., cfg.restoring_floor-c0.min(), cfg.energy_floor-e.energy,
                          cfg.hard_floor-selected_hard if selected_hard is not None else 0.,
                          gamma-selected_gap if cfg.mode == "density" else cfg.mu_target-(c.p0@c.x).min())
    flow_error=float(np.max(np.abs(affine["flows"]-flow_target))) if flow_target is not None else 0.
    equation_error = max(float(np.max(np.abs(e.f0))), float(np.max(np.abs(e.f1))),
                         float(np.max(np.abs(e.cuts))), abs(e.k.sum()-e.energy))
    affine_error = max(float(np.max(np.abs(H-reference))), float(np.max(np.abs(H-affine["hlim"]))),
                       float(np.max(np.abs(e.f0-affine["f0"]))),
                       float(np.max(np.abs(e.f1-affine["f1"]))),
                       float(np.max(np.abs(e.cuts-affine["cuts"]))))
    diagnostic = dict(force0_inf=float(np.max(np.abs(e.f0))), force1_inf=float(np.max(np.abs(e.f1))),
                      cut_inf=float(np.max(np.abs(e.cuts))), budget_identity_error=float(abs(e.k.sum()-e.energy)),
                      inner_restoring=float(c0.min()), hard_gap=hard_gap, full_gap=h_gap,
                      effective_gap=effective_gap, schur_error=schur_error, affine_error=affine_error,
                      box_violation=box, cone_violation=cone, backbone_violation=float(backbone_violation),
                      fixed_entry_violation=fixed_error, repair_violation=float(repair_violation),
                      buffer_violation=buffer_violation,
                      floor_violation=float(floor_violation), energy=e.energy, r_norm=float(np.linalg.norm(r)),
                      r_inf=float(np.max(np.abs(r))), roundoff_floor=n_roundoff,
                      h_guard=cfg.h_guard, offset_variance=float(c.x@(c.s-c.x@c.s)**2),
                      gauge="Euclidean U4 on 1-perp; Euclidean within-parent Z",
                      margin_metric=cfg.margin_metric,physical_spectrum=physical_values,
                      physical_gap=float(physical_values.min()),hard_physical_gap=hard_physical,
                      soft_physical_gap=float(eigvalsh(effective,soft_metric).min()) if effective_gap is not None else None,
                      flows=affine["flows"],flow_target_error=flow_error,
                      degrees=c.p0@c.x,restoring0=c0,hard_spectrum=np.linalg.eigvalsh(hard) if hard.size else [],
                      effective_spectrum=np.linalg.eigvalsh(effective) if effective_gap is not None else None,
                      winding_valid=winding(c, trial, cfg.h_guard, min_density=cfg.backbone_density),
                      winding_one=c.backbone_winding==1 and winding(c, trial, cfg.h_guard, min_density=cfg.backbone_density),
                      backbone_winding=c.backbone_winding)
    result.diagnostics, result.hlim, result.spectrum = diagnostic, H, spectrum
    result.mu0 = float((c.p0@c.x).min())
    violations = max(equation_error, affine_error, box, cone, backbone_violation, fixed_error, repair_violation, buffer_violation,flow_error,
                     floor_violation, diagnostic["r_inf"]-cfg.r_max)
    diagnostic["independent_primal_check_inf"]=float(violations)
    positive_floors_resolved = (c0.min() >= .95*cfg.restoring_floor and
                               (selected_hard is None or selected_hard >= .95*cfg.hard_floor))
    if violations > cfg.residual_tolerance or not diagnostic["winding_valid"]:
        result.message = "Independent constraints exceed configured tolerances"
    elif solver_status not in ("analytic", "optimal"):
        result.status, result.message = "uncertain", "Solver status cannot support strict classification"
    elif reported_gamma is not None and (not np.isfinite(reported_gamma) or reported_gamma-selected_gap > cfg.residual_tolerance):
        result.status, result.message = "uncertain", "Reported full margin disagrees with independent spectrum"
    elif not positive_floors_resolved or effective_gap is None or schur_error > cfg.residual_tolerance:
        result.status = "uncertain"
    elif h_gap > max(cfg.strict_tolerance, n_roundoff) and effective_gap > cfg.strict_tolerance:
        if cfg.mode == "density" and not cfg.neutral and selected_gap < .95*cfg.gamma0:
            result.status = "uncertain"
        else:
            result.status = "joint_asymptotic_strict"
    elif h_gap < -max(cfg.strict_tolerance, n_roundoff):
        result.status = "joint_unstable"
    else:
        result.status = "joint_neutral"
    return result


def _solver_diagnostics(problem, outcome, objective):
    data = dict(status=outcome.status, message=outcome.message, objective=objective,
                iterations=getattr(problem.solver_stats, "num_iters", None),
                primal_residual=None, dual_residual=None, gap_abs=None, gap_rel=None,
                backend_residual_source="unavailable")
    # Optional observability only: acceptance never depends on this private cache.
    try:
        backend = problem._solver_cache.get("CLARABEL")
        info = backend.get_info()
        data.update(primal_residual=float(info.res_primal), dual_residual=float(info.res_dual),
                    gap_abs=float(info.gap_abs), gap_rel=float(info.gap_rel),
                    backend_residual_source="Clarabel.get_info")
    except (AttributeError, KeyError, TypeError):
        pass
    return data


def solve_joint_limit(x, s, groups, mask, config=None, *, backbone_cycle=None,
                      repair_edges=(), fixed_p0=None, family_id="joint", ledger=None,flow_target=None):
    cfg = config or JointLimitConfig()
    model = build_affine_model(x, s, groups, mask, backbone_cycle=backbone_cycle)
    c, m = model.geometry, model.geometry.m
    fixed = _fixed_entries(c, fixed_p0)
    if flow_target is not None and (not np.isfinite(flow_target) or flow_target<=0):
        raise ValueError("flow_target must be finite and positive")
    repair_edges = tuple(tuple(sorted(e)) for e in repair_edges)
    for a, b in repair_edges:
        if (a, b) not in model.r_pairs:
            raise ValueError("Repair edges must be allowed adjacent-parent pairs")
    ledger = ledger if ledger is not None else SolverLedger()
    started = perf_counter()
    fixed_values = np.array([fixed[a,b] for a,b in model.p_pairs])
    free_indices = np.flatnonzero(~np.isfinite(fixed_values))
    constants = np.where(np.isfinite(fixed_values),fixed_values,0.)
    free_variable = cp.Variable(len(free_indices)) if len(free_indices) else None
    selection = np.eye(len(model.p_pairs))[:,free_indices]
    pv = cp.Constant(constants) if free_variable is None else constants+selection@free_variable
    rv = cp.Variable(len(model.r_pairs))
    P = cp.reshape(np.eye(m).ravel()+model.p_map@pv, (m, m), order="C")
    R = cp.reshape(model.r_map@rv, (m, m), order="C")
    H = cp.reshape(model.h_p@pv+model.h_r@rv, (m-1, m-1), order="C")
    mu = cp.Variable()
    gamma = cp.Variable() if cfg.mode == "margin" else (0. if cfg.neutral else cfg.gamma0)
    mass_metric,hard_mass,_=mass_quotient_metrics(c)
    metric=mass_metric if cfg.margin_metric=="mass" else np.eye(m-1)
    hard_constraint_metric=hard_mass if cfg.margin_metric=="mass" else np.eye(m-4)
    constraints = [pv >= 0, pv <= 1, rv >= -cfg.r_max, rv <= cfg.r_max,
                   P@c.x >= mu, model.f0_map@pv == 0,
                   model.f1_p@pv+model.f1_r@rv == 0, model.cut_map@pv == 0,
                   H >> gamma*metric, c.x+model.restoring_map@pv >= cfg.restoring_floor,
                   model.energy_map@pv >= cfg.energy_floor,
                   P+cfg.h_guard*R >= 0, P+cfg.h_guard*R <= 1]
    if free_variable is not None:
        constraints += [free_variable >= cfg.box_buffer, free_variable <= 1-cfg.box_buffer]
    p_index = {pair:i for i,pair in enumerate(model.p_pairs)}
    neighbor_p = pv[[p_index[pair] for pair in model.r_pairs]]
    constraints += [neighbor_p+cfg.h_guard*rv >= cfg.box_buffer,
                    neighbor_p+cfg.h_guard*rv <= 1-cfg.box_buffer]
    if m > 4:
        constraints.append(H[3:, 3:] >> cfg.hard_floor*hard_constraint_metric)
    if flow_target is not None:
        constraints.append(model.flow_map@pv==flow_target)
    if cfg.mode == "margin":
        constraints.append(mu >= cfg.mu_target)
    for a, b in zip(c.backbone_cycle, c.backbone_cycle[1:]+c.backbone_cycle[:1]):
        constraints.append(P[a, b] >= cfg.backbone_density)
    for a, b in model.p_pairs:
        if np.isfinite(fixed[a, b]):
            if (a, b) in model.r_pairs and fixed[a, b] in (0, 1):
                constraints.append(R[a, b] >= 0 if fixed[a, b] == 0 else R[a, b] <= 0)
    for a, b in repair_edges:
        constraints.extend([R[a, b] <= -cfg.inward_slack, P[a, b] >= cfg.repair_density,
                            P[a, b]+cfg.h_guard*R[a, b] >= cfg.repair_density])
    objective = mu if cfg.mode == "density" else gamma
    problem = cp.Problem(cp.Maximize(objective), constraints)
    outcome = solve(problem, "joint_sdp", cfg, ledger)
    records = [_solver_diagnostics(problem, outcome, problem.value)]
    def add_duals(record):
        record["dual_degree"]=np.array(constraints[4].dual_value,copy=True) if constraints[4].dual_value is not None else None
        record["dual_full_hessian"]=np.array(constraints[8].dual_value,copy=True) if constraints[8].dual_value is not None else None
    add_duals(records[0])
    if outcome.status not in ("optimal", "optimal_inaccurate") or pv.value is None or rv.value is None:
        return JointLimitResult(None, None,
            "conditional_infeasible" if outcome.status == "infeasible" else "uncertain",
            outcome.status, cfg, solve_records=records, repair_edges=repair_edges,
            fixed_p0=fixed, elapsed_seconds=perf_counter()-started, message=outcome.message,flow_target=flow_target)
    p_values, r_values = pv.value.copy(), rv.value.copy()
    original_objective = float(objective.value)
    reported_gamma = float(gamma.value) if cfg.mode == "margin" else float(gamma)

    def checked(p_data,r_data,margin):
        density,correction = np.eye(m),np.zeros((m,m))
        for (a,b),value in zip(model.p_pairs,p_data):
            density[a,b] = density[b,a] = value
        for (a,b),value in zip(model.r_pairs,r_data):
            correction[a,b] = correction[b,a] = value
        try:
            candidate = LimitingCandidate(c.x,c.s,c.groups,density,c.allowed_mask,c.backbone_cycle,family_id)
            return validate_joint(candidate,correction,cfg,solver_status=outcome.status,
                                  repair_edges=repair_edges,fixed_p0=fixed,reported_gamma=margin,flow_target=flow_target)
        except ValueError as exc:
            return JointLimitResult(None,correction,"failed",outcome.status,cfg,message=str(exc),
                                    fixed_p0=fixed,repair_edges=repair_edges)

    result = checked(p_values,r_values,reported_gamma)
    if cfg.mode=="density" and result.mu0 is not None:
        gap=abs(result.mu0-original_objective)
        result.diagnostics["density_objective_gap"]=gap
        if gap>cfg.residual_tolerance:
            result.status,result.message="uncertain","Primary density objective disagrees with measured degree"
    records[0]["selected"] = True
    if (cfg.secondary and outcome.status == "optimal" and ledger.remaining("joint_sdp") > 0
            and (cfg.mode == "density" or original_objective > 2*cfg.strict_tolerance)):
        hold = (mu >= original_objective-cfg.density_allowance if cfg.mode == "density"
                else gamma >= .95*original_objective)
        secondary = cp.Problem(cp.Minimize(cp.norm1(rv)), constraints+[hold])
        secondary_outcome = solve(secondary, "joint_sdp", cfg, ledger)
        records.append(_solver_diagnostics(secondary, secondary_outcome, secondary.value))
        add_duals(records[-1])
        records[-1]["selected"] = False
        if secondary_outcome.status == "optimal" and pv.value is not None and rv.value is not None:
            p_values, r_values = pv.value.copy(), rv.value.copy()
            reported_gamma = float(gamma.value) if cfg.mode == "margin" else float(gamma)
            trial = checked(p_values,r_values,reported_gamma)
            if cfg.mode=="density" and trial.mu0 is not None:
                hold_violation=max(0.,original_objective-cfg.density_allowance-trial.mu0)
                trial.diagnostics["secondary_density_hold_violation"]=hold_violation
                if hold_violation>cfg.residual_tolerance:
                    trial.status,trial.message="failed","Secondary density hold violated"
            ranking = {"joint_asymptotic_strict":4,"joint_neutral":3,"joint_unstable":2,"uncertain":1,"failed":0}
            if ranking[trial.status] >= ranking[result.status]:
                result = trial
                records[0]["selected"],records[-1]["selected"] = False,True
            else:
                records[-1]["validation_rejection"] = trial.message or trial.status
    result.solve_records = records
    result.reported_objective = original_objective
    result.elapsed_seconds = perf_counter()-started
    return result


def s7_sym_dipoles(q, w):
    """Necessary polynomials ONLY for the explicitly symmetric S7 family.

    Equal halves/anchor masses and the original four equal crossed densities
    are essential. This function is not an outer-search rejection rule.
    """
    if not 0 <= q <= 1 or not np.isfinite(w) or w < 0:
        raise ValueError("Require 0<=q<=1 and finite w>=0")
    return np.array([4+8*q-4*q*q+8*(1-q*q)*w-(2+4*q+6*q*q)*w*w,
                     -8*q*(2+q)+8*(1-q*q)*w+2*(1+2*q-q*q)*w*w])


def main():
    """Solve one saved geometry; full search is the joint_outer module CLI."""
    import argparse
    import json
    from pathlib import Path
    from .persistence import json_safe
    from .persistence import _code_hash,_write

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,help="JSON geometry or saved joint result")
    parser.add_argument("--output",required=True,help="New JSON result file")
    parser.add_argument("--config",help="Optional JointLimitConfig JSON")
    parser.add_argument("--mode",choices=("density","margin"))
    parser.add_argument("--mu-target",type=float)
    parser.add_argument("--h-guard",type=float)
    parser.add_argument("--neutral",action="store_true")
    parser.add_argument("--primary-only",action="store_true")
    parser.add_argument("--max-joint-calls",type=int,default=2)
    args=parser.parse_args()
    destination=Path(args.output)
    if destination.exists():
        parser.error("Output file already exists")
    if args.max_joint_calls<1:
        parser.error("--max-joint-calls must be positive")
    data=json.loads(Path(args.input).read_text(encoding="utf-8"))
    data=data.get("result",data)
    geometry=data.get("candidate",data)
    settings=json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else data.get("config",{})
    settings=settings.get("inner",settings)
    for name in ("mode","mu_target","h_guard"):
        if getattr(args,name) is not None:
            settings[name]=getattr(args,name)
    if args.neutral:
        settings["neutral"]=True
    if args.primary_only:
        settings["secondary"]=False
    cfg=JointLimitConfig(**settings)
    ledger=SolverLedger(dict(joint_sdp=args.max_joint_calls))
    result=solve_joint_limit(geometry["x"],geometry["s"],geometry["groups"],geometry["allowed_mask"],cfg,
        backbone_cycle=geometry.get("backbone_cycle"),repair_edges=data.get("repair_edges",()),
        fixed_p0=data.get("fixed_p0"),family_id=geometry.get("family_id","joint-cli"),ledger=ledger,flow_target=data.get("flow_target"))
    destination.parent.mkdir(parents=True,exist_ok=True)
    _write(destination,dict(schema_version=1,source_hash=_code_hash(),result=result.data(),
                            counts=ledger.counts,solver_events=ledger.events))
    print(json.dumps(json_safe(dict(status=result.status,mu0=result.mu0,
                                    full_gap=result.diagnostics.get("full_gap"),counts=ledger.counts)),allow_nan=False))


if __name__=="__main__":
    main()
