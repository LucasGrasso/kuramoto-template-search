"""Budgeted two-stage orchestration with exact caches and boundary checkpoints."""

from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from kuramoto_template.experiments.reporting import json_safe
from kuramoto_template.search.evaluator import EvaluationConfig, TemplateEvaluator
from kuramoto_template.search.infrastructure import SearchSession, evaluation_data
from .families import outer_proposals
from .finite_lift import lift_candidate, scaling_slopes, validate_finite
from .models import (Numerics, Stage2Result, candidate_data, candidate_from_data)
from .solvers import SolverLedger
from .stage1 import optimize_limiting, screen_candidate
from .stage2 import solve_correction

KINDS = ("stage1_lp", "stage1_screen_sdp", "stage2_correction_sdp",
         "finite_tree_linear", "full_density_sdp", "full_margin_sdp")


@dataclass(frozen=True)
class TwoStageConfig:
    stage1_proposals: int = 30
    stage1_lp_solves: int = 30
    stage1_screen_sdp_solves: int = 30
    stage2_sdp_solves: int = 30
    finite_h_sdp_solves: int = 0
    shortlist_size: int = 12
    rng_seed: int = 0
    sizes: tuple[int,...] = (7,8,9,10,12)
    h_values: tuple[float,...] = (.01,.003,.001,.0003)
    max_wall_seconds: float | None = None
    audit_finite: bool = False
    numerics: Numerics = field(default_factory=Numerics)

    def __post_init__(self):
        for name in ("stage1_proposals","stage1_lp_solves","stage1_screen_sdp_solves",
                     "stage2_sdp_solves","finite_h_sdp_solves","shortlist_size","rng_seed"):
            v = getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,int) or v < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.sizes or any(isinstance(m,bool) or not isinstance(m,int) or not 7<=m<=12 for m in self.sizes):
            raise ValueError("sizes must contain cluster counts in 7..12")
        if not self.h_values or any(not np.isfinite(h) or h<=0 for h in self.h_values) or len(set(self.h_values)) != len(self.h_values):
            raise ValueError("h_values must contain distinct positive finite scales")
        if self.max_wall_seconds is not None and (not np.isfinite(self.max_wall_seconds) or self.max_wall_seconds<=0):
            raise ValueError("max_wall_seconds must be finite and positive")
        object.__setattr__(self,"sizes",tuple(self.sizes))
        object.__setattr__(self,"h_values",tuple(self.h_values))


def _signature(c):
    digest = hashlib.sha256()
    for name in ("x","s","groups","p0","allowed_mask"):
        digest.update(getattr(c,name).tobytes())
    digest.update(str(c.backbone_cycle).encode())
    return digest.hexdigest()


def _code_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parents[1].rglob("*.py")):
        digest.update(path.relative_to(Path(__file__).resolve().parents[1]).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write(path, data):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(json_safe(data),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def _shortlist(records, cap):
    # Round-robin structural strata, density and dipole headroom within strata.
    strata = {}
    seen = set()
    for index,item in enumerate(records):
        if item["metrics"]["status"] != "stage1_screened":
            continue
        c = candidate_from_data(item["candidate"])
        signature = _signature(c)
        if signature in seen:
            continue
        seen.add(signature)
        strata.setdefault((c.m,c.mask_hash),[]).append(index)
    for group in strata.values():
        group.sort(key=lambda i:(records[i]["metrics"]["mu0"],min(records[i]["metrics"]["dipole_slacks"])),reverse=True)
    # Interleave masks WITHIN each size first, then round-robin sizes. A single
    # size with many masks must not occupy the entire bounded shortlist.
    per_size = {}
    for size in sorted({key[0] for key in strata}):
        groups = [strata[key] for key in sorted(strata) if key[0]==size]
        queue = []
        while any(groups):
            for group in groups:
                if group:
                    queue.append(group.pop(0))
        per_size[size] = queue
    selected = []
    while len(selected)<cap and any(per_size.values()):
        for size in sorted(per_size):
            if per_size[size] and len(selected)<cap:
                selected.append(per_size[size].pop(0))
    return selected


def _full_evaluation(c, h, cap, numerics, seed, ledger):
    """Existing support-constrained SDP through SearchSession, without a tube.

    This can leave the asymptotic family: it is explicitly an independent
    support-only audit/fallback, NEVER a successful local lift.
    """
    evaluator = TemplateEvaluator(EvaluationConfig(
        internal_margin=1e-8,cluster_margin=numerics.finite_sdp_cluster_margin,
        stability_tolerance=1e-13,residual_tolerance=numerics.residual_tolerance,
        solver=numerics.solver,solver_options=numerics.solver_options,
        max_clusters=max(12,c.m),use_margin_fallback=False))

    class MaskedEvaluator:
        config = evaluator.config

        def evaluate(self,template,*,max_solver_calls=None):
            # One full invocation per atomic session, counted BEFORE entry so an
            # interrupted solve remains paid in the next boundary checkpoint.
            return ledger.invoke("full_density_sdp",lambda:evaluator.evaluate(
                template,allowed=c.allowed_mask,max_solver_calls=1))

    with SearchSession([c.template(h)],MaskedEvaluator(),seed=seed,
                       max_sdp_calls=cap,deduplicate=False) as session:
        record = session.evaluate_seeds()[0]
        result = session.finish("audit_completed")
    return record.evaluation, result, evaluator.config


@dataclass
class TwoStageResult:
    state: dict

    @property
    def call_counts(self):
        return self.state["counts"]

    @property
    def best_finite_numeric_mu(self):
        best = self.state["best_finite"]
        return best["mu"] if best else None

    @property
    def best_finite_numeric_template(self):
        best = self.state["best_finite"]
        return candidate_from_data(best["candidate"]).template(best["h"]) if best else None

    @property
    def best_asymptotic_candidate(self):
        best = self.state["best_asymptotic"]
        if best is None:
            return None
        data = {k:v for k,v in best["result"].items() if k != "trace_id"}
        data["candidate"] = candidate_from_data(data["candidate"])
        data["r"] = np.asarray(data["r"])
        return Stage2Result(**data)

    @property
    def stage1_archive(self):
        return self.state["stage1"]

    @property
    def stage2_archive(self):
        return self.state["stage2"]

    @property
    def finite_lift_archive(self):
        return self.state["finite"]

    @property
    def trace(self):
        return self.state["trace"]

    @property
    def stop_reason(self):
        return self.state["stop_reason"]


def run_two_stage(config=None, *, output=None, resume=False, candidates=None, cancelled=None):
    cfg = config or TwoStageConfig()
    destination = Path(output) if output is not None else None
    if resume and destination is None:
        raise ValueError("resume requires an output directory")
    versions = {name:importlib.metadata.version(name) for name in ("numpy","scipy","cvxpy","clarabel")}
    identity = dict(config=json_safe(asdict(cfg)),code_hash=_code_hash(),versions=versions)
    if resume:
        state = json.loads((destination/"checkpoint.json").read_text(encoding="utf-8"))
        if state["identity"] != identity:
            raise ValueError("Checkpoint configuration, code, or solver versions differ")
        if candidates is not None:
            raise ValueError("Resumed candidates come from the checkpoint")
    else:
        if destination is not None:
            destination.mkdir(parents=True,exist_ok=False)
        initial = list(candidates) if candidates is not None else list(outer_proposals(cfg.rng_seed,cfg.stage1_proposals,cfg.sizes))
        state = dict(schema_version=1, identity=identity, phase="stage1",cursor=0,
                     proposals=[candidate_data(c) for c in initial[:cfg.stage1_proposals]],
                     stage1=[],stage2=[],finite=[],trace=[],shortlist=[],cache={},cache_hits=0,
                     counts={kind:0 for kind in KINDS},solver_events=[],best_finite=None,
                     best_asymptotic=None,elapsed_seconds=0.,stop_reason="active",proof_available=False)
    limits = dict(stage1_lp=cfg.stage1_lp_solves,stage1_screen_sdp=cfg.stage1_screen_sdp_solves,
                  stage2_correction_sdp=cfg.stage2_sdp_solves,full_density_sdp=cfg.finite_h_sdp_solves)
    ledger = SolverLedger(limits,state["counts"],state["solver_events"])
    started,previous_time = perf_counter(),state["elapsed_seconds"]

    def persist():
        state["elapsed_seconds"] = previous_time+perf_counter()-started
        state["archives"] = dict(
            stage1_screened=[i for i,v in enumerate(state["stage1"]) if v["metrics"]["status"]=="stage1_screened"],
            asymptotic_full_second_order_strict=[i for i,v in enumerate(state["stage2"]) if v["status"]=="asymptotic_full_second_order_strict"],
            finite_h_numeric_strict=[i for i,v in enumerate(state["finite"]) if v["status"]=="finite_h_numeric_strict"],
            uncertain=[v["id"] for v in state["trace"] if v.get("status") in ("uncertain","asymptotic_degenerate","finite_uncertain")],
            rejected=[v["id"] for v in state["trace"] if v.get("status") in ("screen_rejected","failed","asymptotic_unstable","finite_invalid","finite_unstable")],
            rigorously_certified=[])
        state["scaling"] = [dict(stage2_index=i,slopes=scaling_slopes(
            [v for v in state["finite"] if v["stage2_index"]==i])) for i in range(len(state["stage2"]))]
        state["budget_limits_reached"] = [kind for kind in limits if ledger.remaining(kind)<=0]
        if destination is not None:
            _write(destination/"checkpoint.json",state)

    def event(kind,parent_id,**data):
        record = json_safe(dict(id=len(state["trace"]),parent_id=parent_id,kind=kind,**data))
        state["trace"].append(record)
        return record["id"]

    def stop():
        if cancelled is not None and cancelled():
            state["stop_reason"] = "cancelled"
            return True
        if cfg.max_wall_seconds is not None and previous_time+perf_counter()-started >= cfg.max_wall_seconds:
            state["stop_reason"] = "wall_time"
            return True
        return False

    persist()
    try:
        while state["phase"] == "stage1" and state["cursor"] < len(state["proposals"]):
            if stop():
                break
            i = state["cursor"]
            c = candidate_from_data(state["proposals"][i])
            proposal_id = event("outer_proposal",None,proposal=i,candidate=candidate_data(c))
            key = _signature(c)
            if key in state["cache"]:
                state["cache_hits"] += 1
                event("exact_cache_hit",proposal_id,source=state["cache"][key])
            else:
                metrics = screen_candidate(c,cfg.numerics)
                ident = event("stage1_direct",proposal_id,status=metrics.status)
                state["stage1"].append(dict(candidate=candidate_data(c),metrics=asdict(metrics),trace_id=ident,method="direct"))
                solution = None
                if ledger.remaining("stage1_lp")>0:
                    solution = optimize_limiting(c,method="lp",numerics=cfg.numerics,ledger=ledger)
                    event("stage1_lp",proposal_id,status=solution.status,solver_status=solution.solver_status)
                    if solution.candidate is not None:
                        ident = len(state["trace"])-1
                        state["stage1"].append(dict(candidate=candidate_data(solution.candidate),metrics=asdict(solution.metrics),trace_id=ident,method="lp"))
                if (solution is None or solution.status != "stage1_screened") and ledger.remaining("stage1_screen_sdp")>0:
                    solution = optimize_limiting(c,method="sdp",numerics=cfg.numerics,ledger=ledger)
                    ident = event("stage1_screen_sdp",proposal_id,status=solution.status,solver_status=solution.solver_status)
                    if solution.candidate is not None:
                        state["stage1"].append(dict(candidate=candidate_data(solution.candidate),metrics=asdict(solution.metrics),trace_id=ident,method="sdp"))
                # Never cache a partially completed outer point after an interrupt.
                state["cache"][key] = proposal_id
            state["cursor"] += 1
            persist()
        if state["phase"] == "stage1" and state["cursor"] == len(state["proposals"]):
            state["shortlist"] = _shortlist(state["stage1"],cfg.shortlist_size)
            state["phase"],state["cursor"] = "stage2",0
            persist()
        while state["phase"] == "stage2" and state["cursor"]<len(state["shortlist"]):
            if stop():
                break
            if ledger.remaining("stage2_correction_sdp")<=0:
                state["cursor"] = len(state["shortlist"])
                break
            item = state["stage1"][state["shortlist"][state["cursor"]]]
            c = candidate_from_data(item["candidate"])
            result = solve_correction(c,numerics=cfg.numerics,ledger=ledger)
            ident = event("stage2",item["trace_id"],status=result.status)
            data = asdict(result)
            data["trace_id"] = ident
            state["stage2"].append(data)
            if result.status == "asymptotic_full_second_order_strict":
                if state["best_asymptotic"] is None or item["metrics"]["mu0"]>state["best_asymptotic"]["mu0"]:
                    state["best_asymptotic"] = dict(mu0=item["metrics"]["mu0"],result=data)
            state["cursor"] += 1
            persist()
        if state["phase"] == "stage2" and state["cursor"]==len(state["shortlist"]):
            state["phase"],state["cursor"] = "finite",0
            persist()
        tasks = [(i,h) for i,item in enumerate(state["stage2"])
                 if item["status"]=="asymptotic_full_second_order_strict" for h in cfg.h_values]
        while state["phase"]=="finite" and state["cursor"]<len(tasks):
            if stop():
                break
            i,h = tasks[state["cursor"]]
            item = state["stage2"][i]
            fields = {k:v for k,v in item.items() if k != "trace_id"}
            fields["candidate"] = candidate_from_data(fields["candidate"])
            fields["r"] = np.asarray(fields["r"])
            result = lift_candidate(Stage2Result(**fields),h,numerics=cfg.numerics,ledger=ledger)
            data = asdict(result)
            data["stage2_index"] = i
            # Optional audit/fallback uses the unchanged project SDP. No tube is
            # available in that API, so store it separately from the local lift.
            full_left = cfg.finite_h_sdp_solves-state["counts"]["full_density_sdp"]-state["counts"]["full_margin_sdp"]
            if full_left>0 and (cfg.audit_finite or result.status != "finite_h_numeric_strict"):
                evaluation,session_result,evaluation_config = _full_evaluation(result.candidate,h,full_left,cfg.numerics,cfg.rng_seed,ledger)
                audit = dict(kind="support_only_sdp_audit" if cfg.audit_finite else "support_only_sdp_fallback",
                             family_lift=False,evaluation=evaluation_data(evaluation),
                             session_sdp_calls=session_result.sdp_calls,evaluation_config=asdict(evaluation_config))
                if evaluation.P is not None:
                    audit["validation"] = validate_finite(result.candidate,evaluation.P,h,numerics=cfg.numerics)
                    audit["distance_from_base"] = float(np.max(np.abs(evaluation.P-result.candidate.p0-h*result.r)))
                data["full_sdp"] = audit
            if result.status == "finite_h_numeric_strict":
                if state["best_finite"] is None or result.mu>state["best_finite"]["mu"]:
                    state["best_finite"] = data
            data["trace_id"] = event("finite_lift",item["trace_id"],status=result.status,h=h,
                                     best_mu=state["best_finite"]["mu"] if state["best_finite"] else None)
            state["finite"].append(data)
            state["cursor"] += 1
            persist()
        if state["phase"]=="finite" and state["cursor"]==len(tasks):
            state["phase"],state["stop_reason"] = "done","completed"
    except KeyboardInterrupt:
        state["stop_reason"] = "interrupted"
    finally:
        persist()
    if destination is not None:
        _write(destination/"result.json",state)
        if state["best_finite"] is not None:
            _write(destination/"top-candidate.json",dict(numerical_evidence_only=True,**state["best_finite"]))
        with (destination/"trace.jsonl").open("w",encoding="utf-8") as handle:
            for item in state["trace"]:
                handle.write(json.dumps(json_safe(item),allow_nan=False)+"\n")
    return TwoStageResult(state)
