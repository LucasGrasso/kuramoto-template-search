"""Discover limiting templates from C4 splits, without S7 seeds or constants.

This is a separate experiment. The historical benchmark schedule is unchanged.
All inner results, including rejected finite-difference trials, are retained.
"""

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from .families import normalize_offsets, parent_mask
from .joint_limit import JointLimitConfig, solve_joint_limit
from .joint_outer import repair_tree, _write_with_retry as _write
from .models import LimitingCandidate, candidate_data
from .persistence import _code_hash, json_safe
from .solvers import CallLimit, SolverLedger
from .weak_modes import high_precision_limit_audit


def parent_orbits(split):
    """Orbits of C4 automorphisms preserving the chosen split subset."""
    selected = set(split)
    maps = [(sign, turn) for sign in (-1, 1) for turn in range(4)
            if {(sign*j + turn) % 4 for j in selected} == selected]
    remaining, orbits = set(range(4)), []
    while remaining:
        j = min(remaining)
        orbit = tuple(sorted({(sign*j + turn) % 4 for sign, turn in maps}))
        orbits.append(orbit)
        remaining.difference_update(orbit)
    return tuple(orbits)


def reflection_geometry(split, parameters, *, parent_symmetry=False):
    """Split selected C4 parents into equal, opposite-offset pairs.

    Without parent_symmetry, three logits parameterize the parent masses.
    With it, C4 automorphisms preserving the split subset tie parent variables
    within their orbits. No seven-cluster pattern or numerical constant is fixed.
    Remaining parameters
    are log width ratios to the last split parent's width. This fixes the
    redundant offset scale before enforcing unit mass-weighted variance.
    """
    split = tuple(split)
    if not split or len(set(split)) != len(split) or any(j not in range(4) for j in split):
        raise ValueError("Choose a nonempty subset of C4 parents")
    orbits = parent_orbits(split) if parent_symmetry else tuple((j,) for j in range(4))
    split_orbits = tuple(orbit for orbit in orbits if orbit[0] in split)
    mass_dim = len(orbits) - 1
    v = np.asarray(parameters, dtype=float)
    if v.shape != (mass_dim + len(split_orbits) - 1,) or not np.all(np.isfinite(v)):
        raise ValueError("Wrong reflection parameter vector")
    logits = np.r_[v[:mass_dim], 0.]
    masses = np.array([np.exp(logits[next(i for i, orbit in enumerate(orbits) if j in orbit)]
                              - logits.max()) for j in range(4)])
    masses /= masses.sum()  # Explicit outer softmax parameterization.
    widths = np.exp(np.r_[v[mass_dim:], 0.])
    counts = [2 if j in split else 1 for j in range(4)]
    groups = np.repeat(np.arange(4), counts)
    x = np.concatenate([np.repeat(masses[j] / counts[j], counts[j]) for j in range(4)])
    width = {j: widths[i] for i, orbit in enumerate(split_orbits) for j in orbit}
    s = np.concatenate([[-width[j], width[j]]
                        if j in split else [0.] for j in range(4)])
    mask = parent_mask(groups)
    cycle = tuple(int(np.flatnonzero(groups == j)[0]) for j in range(4))
    return LimitingCandidate(x, normalize_offsets(x, s), groups, mask.astype(float),
                             mask, cycle, ("c4/dihedral:" if parent_symmetry else "c4/reflection:")
                             + ",".join(map(str, split)))


def generic_c4_split(size, rng):
    """Independent unequal splits of four equal C4 parent masses.

    Each birth chooses its parent uniformly; within-parent masses are drawn
    from Dirichlet(4), and Gaussian offsets have zero weighted parent mean.
    No saved candidate, prescribed seven-cluster partition or S7 value enters.
    """
    if not isinstance(size, int) or size < 5:
        raise ValueError("A perturbed C4 must have at least five clusters")
    counts = np.ones(4, dtype=int)
    for _ in range(size - 4):
        counts[int(rng.integers(4))] += 1
    groups = np.repeat(np.arange(4), counts)
    x, s = np.empty(size), np.zeros(size)
    for j in range(4):
        idx = np.flatnonzero(groups == j)
        weights = rng.dirichlet(np.full(len(idx), 4.))
        x[idx] = .25 * weights
        if len(idx) > 1:
            s[idx] = rng.normal(size=len(idx))
            s[idx] -= weights @ s[idx]
    mask = parent_mask(groups)
    cycle = tuple(int(np.flatnonzero(groups == j)[0]) for j in range(4))
    return LimitingCandidate(x, normalize_offsets(x, s), groups, mask.astype(float),
                             mask, cycle, "c4/independent-split")


def split_class(split):
    """C4 rotation/reflection orbit, used only to diversify local starts."""
    split = tuple(split)
    return min(tuple(sorted((sign*j + turn) % 4 for j in split))
               for sign in (-1, 1) for turn in range(4))


def generic_parameters(c):
    return np.r_[np.log(c.x[:-1] / c.x[-1]), c.s]


def generic_geometry(c, parameters):
    m = c.m
    logits = np.r_[parameters[:m-1], 0.]
    x = np.exp(logits - logits.max())
    x /= x.sum()
    return replace(c, x=x, s=normalize_offsets(x, parameters[m-1:]))


@dataclass(frozen=True)
class C4DiscoveryConfig:
    seed: int = 20261029
    reflection_repeats: int = 2
    parent_symmetry: bool = False
    generic_starts: int = 40
    generic_sizes: tuple = (6, 7, 8, 9, 10, 12)
    local_calls: int = 650
    generic_local_starts: int = 1
    polish_calls: int = 1000
    max_iterations: int = 80
    derivative_steps: tuple = (1e-4, 1e-6, 1e-7)
    parameter_bound: float = 3.
    mass_floor: float = 1e-5
    inner: JointLimitConfig = field(default_factory=lambda: JointLimitConfig(
        gamma0=1e-7, h_guard=.003, secondary=False))

    def __post_init__(self):
        for name in ("seed", "reflection_repeats", "generic_starts", "local_calls",
                     "generic_local_starts", "polish_calls", "max_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.reflection_repeats or not self.max_iterations:
            raise ValueError("Need reflection starts and positive iteration count")
        if not isinstance(self.parent_symmetry, bool):
            raise ValueError("parent_symmetry must be boolean")
        if not self.generic_sizes or any(isinstance(m, bool) or not isinstance(m, int)
                                         or m < 5 for m in self.generic_sizes):
            raise ValueError("generic_sizes must contain integers >=5")
        if not self.derivative_steps or any(not np.isfinite(v) or v <= 0 for v in self.derivative_steps):
            raise ValueError("derivative_steps must be positive")
        if not np.isfinite(self.parameter_bound) or self.parameter_bound <= 0:
            raise ValueError("parameter_bound must be positive")
        if not 0 < self.mass_floor < .05:
            raise ValueError("Invalid mass floor")
        if self.inner.mode != "density" or self.inner.secondary:
            raise ValueError("Use primary density solves")
        object.__setattr__(self, "generic_sizes", tuple(self.generic_sizes))
        object.__setattr__(self, "derivative_steps", tuple(self.derivative_steps))


def run_c4_discovery(config, *, output):
    """Fresh, bounded multi-start experiment; existing outputs are never reused.

    Every actual conic invocation goes through the common ledger. Exact-byte
    geometry cache hits consume no budget. SLSQP's selected final point is not
    trusted: only independently strict or neutral inner results can seed refinement.
    Boundary density and strict density are stored separately, including when
    the required full-block margin is zero (inner.neutral=True).
    Local stages have separate call budgets so an early family cannot use all
    later families' allocation. There is no claim of global optimality.
    """
    cfg = config
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "joint-results").mkdir()
    (out / "inputs").mkdir()
    rng = np.random.default_rng(cfg.seed)
    ledger = SolverLedger()
    state = dict(schema_version=1, config=asdict(cfg), source_hash=_code_hash(),
                 versions={name: importlib.metadata.version(name)
                           for name in ("numpy", "scipy", "cvxpy", "clarabel")},
                 initialization="C4 equal parent masses; all nonempty binary split subsets plus independent splits",
                 required_gamma=0. if cfg.inner.neutral else cfg.inner.gamma0,
                 s7_seed_used=False, records=[], stages=[], cache_hits=0, counts=ledger.counts,
                 solver_events=ledger.events, best_id=None, best_mu=None,
                 best_search_id=None, best_search_mu=None,
                 stop_reason="active", proof_available=False)
    cache, solutions, initial = {}, {}, []
    started = perf_counter()

    def persist():
        state["elapsed_seconds"] = perf_counter() - started
        receipt = {k: v for k, v in state.items() if k not in ("records", "solver_events")}
        receipt["record_count"] = len(state["records"])
        _write(out / "checkpoint.json", receipt)

    def append_record(data):
        state["records"].append(data)
        with (out / "trace.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(data), allow_nan=False) + "\n")

    def evaluate(c, *, stage, parent, parameters, split=None, limit=None):
        if c.x.min() < cfg.mass_floor:
            append_record(dict(id=len(state["records"]), stage=stage, parent_id=parent,
                status="mass_floor_rejection", parameters=parameters, candidate=candidate_data(c)))
            persist()
            return None
        digest = hashlib.sha256()
        for a in (c.x, c.s, c.groups, c.allowed_mask):
            digest.update(a.tobytes())
        digest.update(str(c.backbone_cycle).encode())
        key = digest.hexdigest()
        if key in cache:
            state["cache_hits"] += 1
            append_record(dict(id=len(state["records"]), stage=stage, parent_id=parent,
                status="exact_cache_hit", source_id=cache[key], parameters=parameters))
            persist()
            return cache[key]
        if limit is not None and ledger.counts.get("joint_sdp", 0) >= limit:
            raise CallLimit("local_joint_sdp")
        index = len(state["records"])
        # Save the proposed geometry before calling the conic solver.
        pending = dict(id=index, stage=stage, parent_id=parent, parameters=parameters,
                       split=split, candidate=candidate_data(c), status="pending")
        tree = repair_tree(c)
        input_file = f"inputs/{index:06d}.json"
        _write(out / input_file, dict(**pending, config=asdict(cfg.inner), repair_edges=tree))
        state["pending"] = pending
        persist()
        before = ledger.counts.get("joint_sdp", 0)
        event_start = len(ledger.events)
        result = solve_joint_limit(c.x, c.s, c.groups, c.allowed_mask, cfg.inner,
            backbone_cycle=c.backbone_cycle, repair_edges=tree,
            family_id=c.family_id, ledger=ledger)
        solutions[index] = result
        cache[key] = index
        file = f"joint-results/{index:06d}.json"
        _write(out / file, result.data())
        append_record(dict(id=index, stage=stage, parent_id=parent,
            parameters=parameters, split=split, status=result.status,
            solver_status=result.solver_status, mu0=result.mu0,
            full_gap=result.diagnostics.get("full_gap"), file=file, input_file=input_file,
            solver_events=ledger.events[event_start:],
            conic_calls=ledger.counts.get("joint_sdp", 0)-before))
        state["pending"] = None
        if result.status in ("joint_asymptotic_strict", "joint_neutral") and (
                state["best_search_mu"] is None or result.mu0 > state["best_search_mu"]):
            state["best_search_id"], state["best_search_mu"] = index, result.mu0
            _write(out / "top-search.json", result.data())
            print(json.dumps(dict(stage=stage, id=index, mu0=result.mu0, status=result.status,
                                  calls=ledger.counts["joint_sdp"])), flush=True)
        if result.status == "joint_asymptotic_strict" and (state["best_mu"] is None or result.mu0 > state["best_mu"]):
            state["best_id"], state["best_mu"] = index, result.mu0
            _write(out / "top-limit.json", result.data())
        persist()
        return index

    def eligible(index):
        return index is not None and solutions[index].status in ("joint_asymptotic_strict", "joint_neutral")

    def local(start, stage, budget, steps):
        seed_record = state["records"][start]
        split = seed_record["split"]
        c = solutions[start].candidate
        parameters = (np.asarray(seed_record["parameters"]) if split is not None
                      else generic_parameters(c))
        incumbent = start
        limit = ledger.counts.get("joint_sdp", 0) + budget
        stage_info = dict(name=stage, parent_id=start, budget=budget, steps=steps,
                          first_call=ledger.counts.get("joint_sdp", 0), optimizers=[])
        state["stages"].append(stage_info)
        if len(parameters) == 0:
            stage_info.update(stop_reason="no_free_geometry_parameters", best_id=start,
                              best_mu=solutions[start].mu0, actual_calls=0)
            persist()
            return start
        for step in steps:
            def objective(v):
                nonlocal incumbent
                geometry = (reflection_geometry(split, v, parent_symmetry=cfg.parent_symmetry)
                            if split is not None else generic_geometry(c, v))
                # Parent denotes the stage's actual source seed. SLSQP proposes
                # absolute coordinates; the current density winner is not its parent.
                index = evaluate(geometry, stage=stage, parent=start, parameters=v,
                                 split=split, limit=limit)
                if not eligible(index):
                    return 1.
                if solutions[index].mu0 > solutions[incumbent].mu0:
                    incumbent = index
                return -solutions[index].mu0
            try:
                optimum = minimize(objective, parameters, method="SLSQP",
                    bounds=[(-cfg.parameter_bound, cfg.parameter_bound)]*len(parameters),
                    options=dict(maxiter=cfg.max_iterations, eps=step, ftol=1e-11))
                stage_info["optimizers"].append(dict(step=step, success=bool(optimum.success),
                    message=str(optimum.message), nfev=int(optimum.nfev)))
            except CallLimit:
                stage_info["stop_reason"] = "local_call_budget"
                break
            parameters = np.asarray(state["records"][incumbent]["parameters"])
        stage_info.update(best_id=incumbent, best_mu=solutions[incumbent].mu0,
                          actual_calls=ledger.counts.get("joint_sdp", 0)-stage_info["first_call"])
        persist()
        return incumbent

    persist()
    try:
        # No preferred parent, number of splits, seven-cluster ordering or width.
        for repeat in range(cfg.reflection_repeats):
            for bits in range(1, 16):
                split = tuple(j for j in range(4) if bits & (1 << j))
                orbits = parent_orbits(split) if cfg.parent_symmetry else tuple((j,) for j in range(4))
                width_count = sum(orbit[0] in split for orbit in orbits)
                widths = rng.uniform(.6, 1.4, width_count)
                v = np.r_[np.zeros(len(orbits)-1), np.log(widths[:-1] / widths[-1])]
                index = evaluate(reflection_geometry(split, v, parent_symmetry=cfg.parent_symmetry), stage="reflection_start",
                                 parent=None, parameters=v, split=split)
                if eligible(index):
                    initial.append(index)
        for i in range(cfg.generic_starts):
            size = cfg.generic_sizes[i % len(cfg.generic_sizes)]
            c = generic_c4_split(size, rng)
            index = evaluate(c, stage="generic_start", parent=None,
                             parameters=generic_parameters(c))
            if eligible(index):
                initial.append(index)
        # One winning start from each C4 symmetry class, regardless of density.
        classes = {}
        for index in initial:
            record = state["records"][index]
            if record["split"] is not None:
                orbit = split_class(record["split"])
                if orbit not in classes or solutions[index].mu0 > solutions[classes[orbit]].mu0:
                    classes[orbit] = index
        for orbit, index in sorted(classes.items()):
            local(index, "reflection_local:" + str(orbit),
                  cfg.local_calls, cfg.derivative_steps[:1])
        generic = sorted((i for i in initial if state["records"][i]["split"] is None),
                         key=lambda i: solutions[i].mu0, reverse=True)
        for index in generic[:cfg.generic_local_starts]:
            local(index, "generic_local", cfg.local_calls, cfg.derivative_steps[:1])
        if state["best_search_id"] is not None and cfg.polish_calls:
            local(state["best_search_id"], "polish", cfg.polish_calls, cfg.derivative_steps[1:] or cfg.derivative_steps)
        state["stop_reason"] = "completed"
        if state["best_search_id"] is not None:
            audit = ledger.invoke("high_precision_audit",
                lambda: high_precision_limit_audit(solutions[state["best_search_id"]]))
            state["best_search_audit"] = audit
    except KeyboardInterrupt:
        state["stop_reason"] = "interrupted"
    except Exception as exc:
        state["stop_reason"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        persist()
        _write(out / "result.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--neutral", action="store_true", help="Require gamma=0; keep boundary results distinct from strict results")
    args = parser.parse_args()
    data = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else dict(seed=args.seed)
    if "inner" in data:
        data["inner"] = JointLimitConfig(**data["inner"])
    if args.neutral:
        data["inner"] = replace(data.get("inner", C4DiscoveryConfig().inner), neutral=True)
    state = run_c4_discovery(C4DiscoveryConfig(**data), output=args.output)
    print(json.dumps(json_safe(dict(best_strict_mu=state["best_mu"], best_search_mu=state["best_search_mu"], calls=state["counts"],
                                    stop_reason=state["stop_reason"])), allow_nan=False))


if __name__ == "__main__":
    main()
