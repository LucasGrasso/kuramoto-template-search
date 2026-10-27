"""Calibration, fixed-outer paired controls, and a balanced joint research sweep.

uv run python -m kuramoto_template.asymptotic_search.joint_benchmark --output runs/joint-pilot --sweep-proposals 1050
"""

import argparse
from collections import Counter
import csv
from dataclasses import asdict,replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from kuramoto_template.experiments.reporting import json_safe
from .families import appendix_a_correction,s7
from .finite_lift import lift_candidate,scaling_slopes
from .joint_limit import JointLimitConfig,solve_joint_limit,validate_joint
from .joint_outer import FAMILIES,JointOuterConfig,canonical_outer,make_outer_seed,repair_tree,run_joint_outer
from .models import MU_RECORD
from .search import _write
from .solvers import SolverLedger
from .stage1 import optimize_limiting
from .stage2 import solve_correction,validate_correction
from .weak_modes import high_precision_limit_audit


def calibrate_joint(output):
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    ledger=SolverLedger(dict(joint_sdp=40,finite_tree_linear=50,high_precision_audit=3))
    records=[]
    tree=((0,2),(2,4),(4,6),(6,5),(5,3),(3,1))
    for t in (.03,.01,.003,0.,-.01):
        c=s7(t)
        finite_success=False
        for guard in (.01,.003,.001):
            cfg=JointLimitConfig(h_guard=guard,gamma0=1e-5,neutral=t==0,
                                 mode="margin" if t<0 else "density",mu_target=.69,
                                 secondary=t>0)
            result=solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,cfg,fixed_p0=c.p0,
                                     backbone_cycle=c.backbone_cycle,repair_edges=tree,
                                     family_id=c.family_id,ledger=ledger)
            item=dict(t=t,h_guard=guard,joint=result.data(),finite=[])
            if t>0:
                if result.status!="joint_asymptotic_strict":
                    raise RuntimeError(f"S7 calibration unresolved at t={t}, guard={guard}: {result.status}")
                correction=validate_correction(result.candidate,result.r,numerics=cfg.lift_numerics(),solver_status=result.solver_status)
                lifts=[lift_candidate(correction,guard*f,numerics=cfg.lift_numerics(),ledger=ledger) for f in (.3,.1,.03)]
                item["finite"]=[asdict(value) for value in lifts]
                item["scaling_slopes"]=scaling_slopes(lifts)
                finite_success |= any(value.status=="finite_h_numeric_strict" for value in lifts)
            elif t==0:
                analytic=validate_joint(c,appendix_a_correction(c),replace(cfg,mode="density"),fixed_p0=c.p0)
                item["analytic_neutral_status"]=analytic.status
                if result.status=="joint_asymptotic_strict" or analytic.status!="joint_neutral":
                    raise RuntimeError("Neutral S7 was misclassified")
            elif result.status=="joint_asymptotic_strict" or result.diagnostics.get("full_gap",0)>=0:
                raise RuntimeError("Negative S7 control was not falsified")
            records.append(item)
        if t>0 and not finite_success:
            raise RuntimeError(f"No independently strict finite S7 lift for t={t}")
    _write(root/"calibration.json",dict(records=records,counts=ledger.counts,solver_events=ledger.events,
                                         proof_available=False))
    return records,ledger.counts


def _paired_control(c,method,cfg):
    # Both controls have a two-all-conic ceiling: one limiting solve and one R
    # solve. The old secondary correction has no remaining budget and is skipped.
    ledger=SolverLedger(dict(stage1_lp=1,stage1_screen_sdp=1,stage2_correction_sdp=1))
    n=cfg.lift_numerics()
    started=perf_counter()
    screening=optimize_limiting(c,method=method,numerics=n,ledger=ledger)
    correction=None
    if screening.status=="stage1_screened":
        correction=solve_correction(screening.candidate,numerics=n,ledger=ledger)
    strict=correction is not None and correction.status=="asymptotic_full_second_order_strict"
    finite=lift_candidate(correction,cfg.h_guard*.1,numerics=n,ledger=ledger) if strict else None
    return dict(status=correction.status if correction else screening.status,
                mu0=screening.metrics.mu0 if strict else None,
                finite_mu=finite.mu if finite is not None and finite.status=="finite_h_numeric_strict" else None,
                elapsed_seconds=perf_counter()-started,counts=ledger.counts,
                screening=asdict(screening),correction=asdict(correction) if correction else None,
                finite=asdict(finite) if finite else None,settings=asdict(n))


def run_joint_benchmark(output,*,seed=0,sweep_proposals=1050,paired=True):
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    calibration,calibration_counts=calibrate_joint(root/"calibration")
    rng=np.random.default_rng(seed)
    rows=[]
    _write(root/"benchmark-config.json",dict(seed=seed,sweep_proposals=sweep_proposals,paired=paired,
        sizes=[7,8,9,10,12],families=FAMILIES,all_conic_cell_cap=2,
        fixed_full_sdp_calls_per_cell=0,comparison_target=MU_RECORD))
    if paired:
        paired_root=root/"paired"
        paired_root.mkdir()
        for size in (7,8,9,10,12):
            for family in FAMILIES:
                source,fixed=make_outer_seed(size,family,rng)
                c,fixed,order=canonical_outer(source,fixed)
                cfg=JointLimitConfig(h_guard=.003)
                cell=f"m{size}-{family}"
                cell_root=paired_root/cell
                cell_root.mkdir()
                _write(cell_root/"input.json",dict(candidate=asdict(c),fixed_p0=fixed,config=asdict(cfg),input_order=order))
                ledger=SolverLedger(dict(joint_sdp=2))
                started=perf_counter()
                joint=solve_joint_limit(c.x,c.s,c.groups,c.allowed_mask,cfg,backbone_cycle=c.backbone_cycle,
                                         fixed_p0=fixed,repair_edges=repair_tree(c),family_id=family,ledger=ledger)
                finite=None
                if joint.status=="joint_asymptotic_strict":
                    correction=validate_correction(joint.candidate,joint.r,numerics=cfg.lift_numerics(),solver_status=joint.solver_status)
                    finite=lift_candidate(correction,.0003,numerics=cfg.lift_numerics(),ledger=ledger)
                _write(cell_root/"joint.json",joint.data())
                _write(cell_root/"joint-finite.json",asdict(finite) if finite else {})
                rows.append(dict(cell=cell,size=size,family=family,method="joint",status=joint.status,
                    mu0=joint.mu0 if joint.status=="joint_asymptotic_strict" else None,
                    finite_mu=finite.mu if finite is not None and finite.status=="finite_h_numeric_strict" else None,
                    elapsed_seconds=perf_counter()-started,**ledger.counts))
                for method in ("lp","sdp"):
                    control=_paired_control(c,method,cfg)
                    _write(cell_root/f"{method}-then-r.json",control)
                    rows.append(dict(cell=cell,size=size,family=family,method=f"{method}_then_r",
                        status=control["status"],mu0=control["mu0"],finite_mu=control["finite_mu"],
                        elapsed_seconds=control["elapsed_seconds"],**control["counts"]))
    for row in rows:
        row["all_conic_calls"]=sum(row.get(kind,0) for kind in ("joint_sdp","stage1_lp","stage1_screen_sdp","stage2_correction_sdp"))
        row["matched_actual_all_conic"]=row["all_conic_calls"]==2
        row["full_sdp_calls"]=0
    if rows:
        with (root/"paired-summary.jsonl").open("w",encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(json_safe(row),allow_nan=False)+"\n")
        columns=["cell","size","family","method","status","mu0","finite_mu","all_conic_calls",
                 "matched_actual_all_conic","full_sdp_calls","joint_sdp","stage1_lp","stage1_screen_sdp",
                 "stage2_correction_sdp","finite_tree_linear","elapsed_seconds"]
        with (root/"paired-summary.csv").open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=columns,extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    sweep_cfg=JointOuterConfig(proposals=sweep_proposals,seed=seed,joint_sdp_calls=sweep_proposals,
        finite_tree_calls=60,audit_calls=10,shortlist_size=10,inner=JointLimitConfig(secondary=False))
    sweep=run_joint_outer(sweep_cfg,output=root/"sweep")
    _write(root/"sweep-config.json",asdict(sweep_cfg))
    _write(root/"summary.json",dict(calibration_counts=calibration_counts,paired=rows,
        sweep_counts=sweep.call_counts,best_limit_mu=sweep.best_limit_mu,best_finite_mu=sweep.best_finite_mu,
        sweep_outcomes=dict(Counter(r["status"] for r in sweep.state["records"])),proof_available=False))
    lines=["# Joint limiting SDP pilot","",f"Best strict limiting density: {sweep.best_limit_mu}.",
        f"Best independently strict finite lift: {sweep.best_finite_mu}.",
        f"Comparison target: {MU_RECORD}; rigorous certification remains unavailable.","",
        "The calibration passed before the sweep. Original two-stage algorithms are unchanged.",
        "Paired cells share masses, normalized offsets, labels and support, with a two-all-conic ceiling and zero full SDPs.",
        "Early infeasibility/necessary-screen failure spends fewer calls; matched_actual_all_conic identifies spent ceilings.",
        "Joint repair-tree inward conditions, explicit box buffers, and fixed nonuniform-anchor entries are additional inner restrictions.",
        "Old controls optimize their supported unrestricted densities on the same mask; these differences prevent a superiority claim.",
        "Their R solve requests its effective-soft margin; the joint model controls the full fixed-gauge block. All settings are saved.",
        "This is not a matched wall-time experiment. No margin is ranked across cluster sizes.","",
        "| Method | Cells | Strict limits | Best strict mu0 | All conic calls | Wall seconds |",
        "|---|---:|---:|---:|---:|---:|"]
    for name in ("joint","lp_then_r","sdp_then_r"):
        group=[r for r in rows if r["method"]==name]
        successes=[r["mu0"] for r in group if r["mu0"] is not None]
        lines.append(f"| {name} | {len(group)} | {len(successes)} | {max(successes) if successes else 'none'} | {sum(r['all_conic_calls'] for r in group)} | {sum(r['elapsed_seconds'] for r in group):.3f} |")
    lines.extend(["",f"Sweep solver costs: {sweep.call_counts}.",
        f"Sweep outcomes: {dict(Counter(r['status'] for r in sweep.state['records']))}.",
        "Every proposal, selected parent, partial weak-mode prediction and re-solve is recorded.",
        "A limiting improvement is not a finite witness, and a numerical witness is not a proved lower bound."])
    (root/"findings.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return sweep,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--sweep-proposals",type=int,default=1050)
    parser.add_argument("--skip-paired",action="store_true")
    args=parser.parse_args()
    result,rows=run_joint_benchmark(args.output,seed=args.seed,sweep_proposals=args.sweep_proposals,paired=not args.skip_paired)
    print(json.dumps(dict(best_limit_mu=result.best_limit_mu,best_finite_mu=result.best_finite_mu,
                          paired_cells=len(rows),calls=result.call_counts)))


if __name__=="__main__":
    main()
