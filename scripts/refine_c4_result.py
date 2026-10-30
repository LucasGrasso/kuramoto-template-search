"""Bounded Nelder-Mead continuation of a C4 discovery winner at a new margin."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from kuramoto_template.asymptotic_search.c4_discovery import (
    C4DiscoveryConfig, generic_geometry, generic_parameters, parent_orbits,
    reflection_geometry,
)
from kuramoto_template.asymptotic_search.joint_limit import (
    JointLimitConfig, result_from_data, solve_joint_limit,
)
from kuramoto_template.asymptotic_search.joint_outer import repair_tree, _write_with_retry as write
from kuramoto_template.asymptotic_search.models import candidate_data
from kuramoto_template.asymptotic_search.persistence import _code_hash, json_safe
from kuramoto_template.asymptotic_search.solvers import CallLimit, SolverLedger
from kuramoto_template.asymptotic_search.weak_modes import high_precision_limit_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed C4 discovery directory")
    parser.add_argument("--output", type=Path, required=True, help="New directory")
    parser.add_argument("--gamma", type=float, default=0.)
    parser.add_argument("--max-calls", type=int, default=1500)
    args = parser.parse_args()
    if not np.isfinite(args.gamma) or args.gamma < 0 or args.max_calls < 1:
        parser.error("Require gamma>=0 and max-calls>=1")
    source = args.run / "result.json"
    previous = json.loads(source.read_text(encoding="utf-8"))
    if previous.get("s7_seed_used") is not False or previous.get("best_search_id") is None:
        parser.error("Need a C4 discovery run with a recorded winner")
    data = dict(previous["config"])
    data["inner"] = JointLimitConfig(**data["inner"])
    cfg = C4DiscoveryConfig(**data)
    record = previous["records"][previous["best_search_id"]]
    saved_path = args.run / record["file"]
    saved = result_from_data(json.loads(saved_path.read_text(encoding="utf-8")))
    split, c = record["split"], saved.candidate
    v0 = np.array(record["parameters"] if split is not None else generic_parameters(c))
    if not len(v0):
        parser.error("This symmetry class has no free geometry parameters")
    inner = replace(cfg.inner, neutral=args.gamma == 0., gamma0=args.gamma or cfg.inner.gamma0)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "joint-results").mkdir()
    (args.output / "inputs").mkdir()
    ledger = SolverLedger(dict(joint_sdp=args.max_calls))
    state = dict(schema_version=1, source_run=str(args.run),
        source_run_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), source_id=record["id"],
        source_witness_sha256=hashlib.sha256(saved_path.read_bytes()).hexdigest(),
        source_hash=_code_hash(), versions=previous["versions"], s7_seed_used=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        required_gamma=args.gamma, max_calls=args.max_calls, inner=inner.__dict__,
        method="Nelder-Mead", xatol=1e-9, fatol=1e-12, adaptive=True,
        records=[], counts=ledger.counts, solver_events=ledger.events,
        best_search_mu=None, best_mu=None, stop_reason="active", proof_available=False)
    cache, best = {}, {}

    def objective(v):
        key = np.asarray(v).tobytes()
        if key in cache:
            with (args.output / "cache-hits.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(json_safe(dict(parameters=v, source_id=cache[key][1]))) + "\n")
            return cache[key][0]
        cnew = (reflection_geometry(split, v, parent_symmetry=cfg.parent_symmetry)
                if split is not None else generic_geometry(c, v))
        if cnew.x.min() < cfg.mass_floor:
            with (args.output / "rejections.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(json_safe(dict(parameters=v, status="mass_floor"))) + "\n")
            return 1.
        if ledger.remaining("joint_sdp") <= 0:
            raise CallLimit("joint_sdp")
        index = len(state["records"])
        input_file = f"inputs/{index:06d}.json"
        write(args.output / input_file, dict(id=index, parameters=v, candidate=candidate_data(cnew)))
        result = solve_joint_limit(cnew.x, cnew.s, cnew.groups, cnew.allowed_mask, inner,
            backbone_cycle=cnew.backbone_cycle, repair_edges=repair_tree(cnew),
            family_id=cnew.family_id, ledger=ledger)
        file = f"joint-results/{index:06d}.json"
        write(args.output / file, result.data())
        state["records"].append(dict(id=index, parameters=v, status=result.status,
            mu0=result.mu0, full_gap=result.diagnostics.get("full_gap"), file=file, input_file=input_file))
        eligible = result.status in ("joint_asymptotic_strict", "joint_neutral")
        for kind in ("search", "strict"):
            if eligible and (kind == "search" or result.status == "joint_asymptotic_strict"):
                if kind not in best or result.mu0 > best[kind].mu0:
                    best[kind] = result
                    write(args.output / ("top-search.json" if kind == "search" else "top-limit.json"), result.data())
                    state["best_search_mu" if kind == "search" else "best_mu"] = result.mu0
                    if kind == "search":
                        print(index, result.mu0, result.status, flush=True)
        write(args.output / "result.json", state)
        value = -result.mu0 if eligible else 1.
        cache[key] = value, index
        return value

    mass_dim = len(parent_orbits(split))-1 if cfg.parent_symmetry and split is not None else (3 if split is not None else c.m-1)
    steps = np.r_[np.full(mass_dim, 1e-5), np.full(len(v0)-mass_dim, 1e-3)]
    simplex = np.tile(v0, (len(v0)+1, 1))
    for i in range(len(v0)):
        simplex[i+1, i] += steps[i] if v0[i]+steps[i] <= cfg.parameter_bound else -steps[i]
    state["initial_simplex"] = simplex
    write(args.output / "result.json", state)
    try:
        optimum = minimize(objective, v0, method="Nelder-Mead",
            bounds=[(-cfg.parameter_bound, cfg.parameter_bound)]*len(v0),
            options=dict(initial_simplex=simplex, maxfev=2*args.max_calls,
                         maxiter=2*args.max_calls, xatol=state["xatol"],
                         fatol=state["fatol"], adaptive=True))
        state.update(stop_reason="completed", optimizer_message=str(optimum.message),
                     optimizer_success=bool(optimum.success))
    except CallLimit:
        state["stop_reason"] = "joint_call_budget"
    except KeyboardInterrupt:
        state["stop_reason"] = "interrupted"
    except Exception as exc:
        state["stop_reason"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if "search" in best:
            state["best_search_audit"] = ledger.invoke("high_precision_audit",
                lambda: high_precision_limit_audit(best["search"]))
        write(args.output / "result.json", state)
    print(json.dumps(json_safe(dict(best_mu=state["best_mu"], best_search_mu=state["best_search_mu"],
                                    counts=state["counts"], stop_reason=state["stop_reason"]))))


if __name__ == "__main__":
    main()
