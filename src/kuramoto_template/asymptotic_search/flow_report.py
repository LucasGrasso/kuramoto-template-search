"""Evidence tables for the bounded flow pilot; unlike resources are not pooled."""

from collections import Counter,defaultdict
import csv
import json

from kuramoto_template.experiments.reporting import json_safe
from .aggregate_flow import CONCRETE_PAPER_EXAMPLE
from .joint_outer import _write_with_retry
from .models import MU_RECORD


def write_flow_report(result):
    state,root=result.state,result.output
    if root is None:
        return
    rows=[]
    for i,outcome in enumerate(state["outcomes"]):
        meta=outcome["metadata"]; desc=meta["descriptor"]
        comp=meta.get("compensation",{}).get("diagnostics",{})
        rows.append(dict(proposal=i,category=desc["kind"],family=meta.get("family"),policy=meta.get("policy"),
            pair=meta.get("pair"),size=meta.get("size"),q_mass_ratio=meta.get("q_mass_ratio"),q_cross_density=meta.get("q_cross_density"),
            winding=meta.get("backbone_winding"),flow_fraction=meta.get("flow_fraction"),flow_target=meta.get("flow_target"),
            status=outcome["status"],stable_mu0=outcome.get("mu0") if outcome["status"]=="joint_asymptotic_strict" else None,
            primary_calls=outcome["primary_calls"],secondary_conic_calls=meta.get("secondary_conic_calls",0),
            cache_hit=outcome.get("cache_hit",False),aggregate_cap=meta.get("aggregate_screen",outcome.get("screen",{})).get("cap"),
            aggregate_status=meta.get("aggregate_screen",outcome.get("screen",{})).get("status"),
            comparison_only=meta.get("comparison_only",False),parent_reoptimized=meta.get("parent_reoptimized"),
            predicted_degree_gain=comp.get("predicted_degree_gain"),predicted_margin_gain=comp.get("predicted_full_margin_gain"),
            realized_degree_gain=outcome.get("realized_degree_gain"),opened_edges=comp.get("opened_edges"),
            initial_distance_s7=meta.get("initial_distance_s7",{}).get("total"),
            final_distance_s7=(state["records"][outcome["record"]].get("distance_s7") or {}).get("total") if outcome.get("record") is not None else None,
            record=outcome.get("record")))
    if rows:
        with (root/"summary.csv").open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(json_safe(rows))
        with (root/"summary.jsonl").open("w",encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(json_safe(row),allow_nan=False)+"\n")
    groups=defaultdict(list)
    for row in rows:
        if row["pair"]: groups[row["pair"]].append(row)
    paired=[]
    for name,group in groups.items():
        if len(group)!=2: continue
        a,b=group
        matched=(a["primary_calls"]==b["primary_calls"]==1 and a["secondary_conic_calls"]==b["secondary_conic_calls"])
        strict=a["stable_mu0"] is not None and b["stable_mu0"] is not None
        paired.append(dict(pair=name,category=a["category"],matched_actual_conic=matched,both_strict=strict,
            policies=[a["policy"],b["policy"]],mu_difference_a_minus_b=a["stable_mu0"]-b["stable_mu0"] if strict else None,
            primary_calls=[a["primary_calls"],b["primary_calls"]],secondary_calls=[a["secondary_conic_calls"],b["secondary_conic_calls"]]))
    _write_with_retry(root/"paired.json",paired)
    historical=state["filter_comparison"]
    rejected=[x for x in historical if x["screen"]["status"]=="aggregate_rejected"]
    summary=dict(best_limit_mu=state["best_limit_mu"],best_finite_mu=state["best_finite"]["mu"] if state["best_finite"] else None,
        comparison_theorem=MU_RECORD,comparison_concrete_example=CONCRETE_PAPER_EXAMPLE,counts=state["counts"],
        elapsed_seconds=state["elapsed_seconds"],cache_hits=state["cache_hits"],phase=state["phase"],stop_reason=state["stop_reason"],
        main_outcomes=dict(Counter(o["status"] for o in state["outcomes"])),
        saved_filter_audit=dict(inputs=len(historical),rejected=len(rejected),
            actual_unfiltered_primary_calls=len(historical),counterfactual_screened_primary_calls=len(historical)-len(rejected),
            screening_seconds=sum(x["screening_seconds"] for x in historical),
            unfiltered_seconds=sum(x["unfiltered_seconds"] for x in historical)),
        finite_statuses=dict(Counter(v["status"] for v in state["finite"])),
        potential_record_count=sum(r["status"]=="joint_asymptotic_strict" and r["mu0"]>MU_RECORD+1e-8 for r in state["records"]),
        proof_available=False,rigorously_certified_graph_witnesses=0)
    _write_with_retry(root/"summary.json",summary)
    lines=["# Aggregate-flow and higher-winding pilot","",
        f"Best strict numerical limit: {summary['best_limit_mu']}.",f"Best independently strict finite lift: {summary['best_finite_mu']}.",
        f"Theorem comparison: {MU_RECORD}; concrete Proposition-1 example: {CONCRETE_PAPER_EXAMPLE}.",
        "No Decimal recomputation here encloses an equilibrium root or certifies a graph.","",
        f"Actual resource counters: {state['counts']}.",f"Wall seconds including checkpoints: {state['elapsed_seconds']:.3f}.",
        "Primary SDPs, secondary QPs, analytic projections, tree solves and audits are different costs.",
        f"Saved identical-input audit rejected {len(rejected)} of {len(historical)} points; all {len(historical)} unfiltered primary solves were paid to verify it.",
        "Counterfactual screened savings are labeled separately, not subtracted from actual calls.",
        "Split/merge controls are explicitly solved even when the goal bound excludes both, to test fresh physical-margin comparisons.","",
        "| Family | Proposals | Strict | Goal-screened | Primary calls | Best strict mu0 |",
        "|---|---:|---:|---:|---:|---:|"]
    for family in sorted({r["family"] or "unknown" for r in rows}):
        selected=[r for r in rows if (r["family"] or "unknown")==family]
        vals=[r["stable_mu0"] for r in selected if r["stable_mu0"] is not None]
        lines.append(f"| {family} | {len(selected)} | {len(vals)} | {sum(r['status']=='aggregate_rejected' for r in selected)} | {sum(r['primary_calls'] for r in selected)} | {max(vals) if vals else 'none'} |")
    lines+= ["","Paired rows with both actual primary calls and equal secondary calls:"]
    for category in ("balanced","split_merge","compensated"):
        pair=[p for p in paired if p["category"]==category and p["matched_actual_conic"]]
        comparable=[p for p in pair if p["both_strict"]]
        differences=[p["mu_difference_a_minus_b"] for p in comparable]
        lines.append(f"- {category}: {len(pair)} matched, {len(comparable)} both strict; A-minus-B degree differences {min(differences) if differences else None} to {max(differences) if differences else None}.")
    lines += ["","Family labels distinguish legal q_mass_ratio>1 from a crossed-edge density.",
        "Winding-two skeletons have simple eight-edge backbones; their scalar winding does not imply cycle-space independence.",
        "The archive records shortest nonzero cycle lengths and initial/final invariant S7-distance proxies.",
        "Fresh seven-cluster merges retain h/offset units and use the mass quotient margin metric. They are not asserted to be exact twins.",
        "Only independently strict joint outputs update the incumbent; predicted moves never accept a witness."]
    (root/"findings.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
