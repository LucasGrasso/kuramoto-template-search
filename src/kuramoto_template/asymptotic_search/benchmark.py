"""Small reproducible calibration, diverse search, and matched full-call controls.

uv run python -m kuramoto_template.asymptotic_search.benchmark --output runs/asym-pilot
"""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from kuramoto_template.experiments.reporting import json_safe
from kuramoto_template.search.amcs import NestedSearchConfig, run_nested_search
from kuramoto_template.search.evaluator import EvaluationConfig, TemplateEvaluator
from kuramoto_template.search.random_search import RandomSearchConfig, run_random_search
from .audit import high_precision_audit
from .families import appendix_a_correction,s7
from .finite_lift import lift_candidate,scaling_slopes,validate_finite
from .models import MU_RECORD,Numerics
from .search import TwoStageConfig,run_two_stage
from .stage1 import screen_candidate
from .stage2 import validate_correction


def _dump(path,data):
    path.write_text(json.dumps(json_safe(data),indent=2,allow_nan=False)+"\n",encoding="utf-8")


def run_benchmark(output, *, seeds=(0,1,2), proposals=15, full_calls=4):
    root = Path(output)
    root.mkdir(parents=True,exist_ok=False)
    rows,calibrations,top = [],[],None
    numerics = Numerics()
    _dump(root/"benchmark-config.json",dict(seeds=seeds,proposals=proposals,full_calls=full_calls,
                                           numerics=asdict(numerics),comparison_target=MU_RECORD))
    for t in (.03,.01,.003,0.,-.01):
        c = s7(t)
        metrics = screen_candidate(c,numerics)
        data = dict(candidate=asdict(c),stage1=asdict(metrics))
        if t>0:
            result = validate_correction(c,appendix_a_correction(c),numerics=numerics)
            lifts = [lift_candidate(result,h,numerics=numerics) for h in (.01,.003,.001,.0003)]
            data.update(stage2=asdict(result),finite=[asdict(l) for l in lifts],slopes=scaling_slopes(lifts))
        # Raw P0 is a no-refinement negative control, with zero solver cost.
        data["unrepaired_finite"] = validate_finite(c,c.p0,.003,numerics=numerics)
        calibrations.append(data)
    _dump(root/"calibration.json",calibrations)
    for seed in seeds:
        cfg = TwoStageConfig(rng_seed=seed,stage1_proposals=proposals,stage1_lp_solves=proposals,
                             stage1_screen_sdp_solves=proposals,stage2_sdp_solves=12,
                             shortlist_size=6,numerics=numerics)
        started = perf_counter()
        diverse = run_two_stage(cfg,output=root/f"diverse-{seed}")
        rows.append(dict(algorithm="diverse_two_stage",seed=seed,t=None,
                         best_mu=diverse.best_finite_numeric_mu,elapsed_seconds=perf_counter()-started,
                         full_search_calls=0,full_audit_calls=0,**diverse.call_counts,
                         stage1_screened=len(diverse.state["archives"]["stage1_screened"]),
                         asymptotic_strict=len(diverse.state["archives"]["asymptotic_full_second_order_strict"]),
                         finite_strict=len(diverse.state["archives"]["finite_h_numeric_strict"])))
        rows[-1]["asymptotic_per_second"] = rows[-1]["asymptotic_strict"]/rows[-1]["elapsed_seconds"]
        if diverse.state["best_finite"] is not None and (top is None or diverse.best_finite_numeric_mu>top["mu"]):
            top = diverse.state["best_finite"]
        for t in (.03,.01,.003):
            c = s7(t)
            # The same (x,phi) and allowed mask start all methods in this cell.
            # Asymptotic construction does no full SDP search. Its independent
            # support-only audits spend the matched full-call budget separately.
            cell_cfg = TwoStageConfig(rng_seed=seed,stage1_proposals=1,stage1_lp_solves=0,
                                      stage1_screen_sdp_solves=0,stage2_sdp_solves=2,
                                      finite_h_sdp_solves=full_calls,audit_finite=True,
                                      shortlist_size=1,numerics=numerics,h_values=(.003,.001,.0003,.0001))
            started = perf_counter()
            result = run_two_stage(cell_cfg,candidates=[c],output=root/f"two-stage-{seed}-{t}")
            counts = result.call_counts
            total = counts["full_density_sdp"]+counts["full_margin_sdp"]
            rows.append(dict(algorithm="two_stage",seed=seed,t=t,best_mu=result.best_finite_numeric_mu,
                             elapsed_seconds=perf_counter()-started,full_search_calls=0,full_audit_calls=total,
                             **counts,full_budget=full_calls,matched_actual_calls=total==full_calls))
            if result.state["best_finite"] is not None:
                best = result.state["best_finite"]
                audit = high_precision_audit(c.x,best["phi"],best["P"])
                _dump(root/f"decimal-audit-{seed}-{t}.json",audit)
                rows[-1]["high_precision_audit_seconds"] = audit["elapsed_seconds"]
                if top is None or best["mu"]>top["mu"]:
                    top = best

            base_evaluator = TemplateEvaluator(EvaluationConfig(
                cluster_margin=numerics.finite_sdp_cluster_margin,internal_margin=1e-8,stability_tolerance=1e-13,
                residual_tolerance=numerics.residual_tolerance,solver_options=numerics.solver_options,
                use_margin_fallback=False))

            class MaskedEvaluator:
                config = base_evaluator.config

                def evaluate(self,template,*,max_solver_calls=None):
                    return base_evaluator.evaluate(template,allowed=c.allowed_mask,max_solver_calls=max_solver_calls)

            for algorithm in ("no_refinement","random","amcs"):
                started = perf_counter()
                if algorithm in ("no_refinement","random"):
                    settings = RandomSearchConfig(seed=seed,min_clusters=7,max_clusters=7,
                        max_sdp_calls=full_calls,iterations=0 if algorithm=="no_refinement" else 100,
                        restart_probability=0,mutation_weights=(0,0,1,1),phase_scale=.0003,resize_fraction=.001)
                    control = run_random_search([c.template(.003)],MaskedEvaluator(),settings,
                                                output=root/f"{algorithm}-{seed}-{t}")
                else:
                    settings = NestedSearchConfig(seed=seed,min_clusters=7,max_clusters=7,
                        max_sdp_calls=full_calls,iterations=100,level=1,depth=2,branching=2,
                        restart_probability=0,mutation_weights=(0,0,1,1),phase_scale=.0003,resize_fraction=.001)
                    control = run_nested_search([c.template(.003)],MaskedEvaluator(),settings,
                                                output=root/f"{algorithm}-{seed}-{t}")
                statuses = Counter(r.evaluation.status for r in control.trace if r.evaluation is not None)
                rows.append(dict(algorithm=algorithm,seed=seed,t=t,
                    best_mu=control.best.evaluation.stable_mu if control.best else None,
                    elapsed_seconds=perf_counter()-started,full_search_calls=control.sdp_calls,full_audit_calls=0,
                    full_density_sdp=control.sdp_calls,full_margin_sdp=0,full_budget=full_calls,
                    matched_actual_calls=control.sdp_calls==full_calls,statuses=dict(statuses)))
    with (root/"summary.jsonl").open("w",encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row),allow_nan=False)+"\n")
    columns = ["algorithm","seed","t","best_mu","elapsed_seconds","full_search_calls","full_audit_calls",
               "stage1_lp","stage1_screen_sdp","stage2_correction_sdp","finite_tree_linear",
               "full_density_sdp","full_margin_sdp","matched_actual_calls","stage1_screened","asymptotic_strict","finite_strict",
               "asymptotic_per_second","high_precision_audit_seconds"]
    with (root/"summary.csv").open("w",encoding="utf-8",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=columns,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if top is not None:
        _dump(root/"top-candidate.json",top)
        _dump(root/"top-decimal-audit.json",high_precision_audit(top["candidate"]["x"],top["phi"],top["P"]))
    best = max((r["best_mu"] for r in rows if r["best_mu"] is not None),default=None)
    lines = ["# Two-stage pilot", "",f"Numerical best density: {best}. Comparison target: {MU_RECORD}.",
             "No rigorous certificate or new lower-bound claim.","",
             "Full search calls and independent full audit calls are separate columns.",
             "Two-stage calibration cells and random/AMCS controls use the same masses, initial finite phases, support and full-call cap.",
             "Only cells with matched_actual_calls=true spent the full allowance. No-refinement deliberately spends one initialization call.",
             "Two-stage also pays its LP, necessary-screen SDP and correction SDP costs; these are not free.",
             "The diverse outer search is a separate exploration pilot, not a paired benchmark.",
             "Decimal audits recompute stored float inputs at 60 digits; they do not prove equilibrium or stability.",
             "RL is omitted because there is no nominated frozen checkpoint. The repair_sdp branch is not merged for this experiment.","",
             "| Algorithm | Runs | Finite stable runs | Best density | Wall seconds |", "|---|---:|---:|---:|---:|"]
    for algorithm in sorted({r["algorithm"] for r in rows}):
        group = [r for r in rows if r["algorithm"]==algorithm]
        successes = [r["best_mu"] for r in group if r["best_mu"] is not None]
        lines.append(f"| {algorithm} | {len(group)} | {len(successes)} | {max(successes) if successes else 'none'} | {sum(r['elapsed_seconds'] for r in group):.3f} |")
    (root/"findings.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--proposals",type=int,default=15)
    parser.add_argument("--full-calls",type=int,default=4)
    args = parser.parse_args()
    if args.full_calls<1:
        parser.error("--full-calls must be positive")
    rows = run_benchmark(args.output,seeds=tuple(args.seeds),proposals=args.proposals,full_calls=args.full_calls)
    print(json.dumps(dict(runs=len(rows),best_mu=max((r["best_mu"] for r in rows if r["best_mu"] is not None),default=None))))


if __name__=="__main__":
    main()
