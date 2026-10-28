"""Validate tracked witnesses and optionally repeat their paid solves and lifts."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from kuramoto_template.asymptotic_search.audit import high_precision_audit
from kuramoto_template.asymptotic_search.finite_lift import lift_candidate, validate_finite
from kuramoto_template.asymptotic_search.joint_limit import (
    result_from_data, solve_joint_limit, validate_joint,
)
from kuramoto_template.asymptotic_search.persistence import _code_hash, _write
from kuramoto_template.asymptotic_search.solvers import SolverLedger
from kuramoto_template.asymptotic_search.stage2 import validate_correction
from kuramoto_template.asymptotic_search.weak_modes import high_precision_limit_audit

ROOT = Path(__file__).resolve().parents[1]
LIMIT = "research/s7-analysis/evidencia/fine-offset-1e-07--0.0001.json"
ORIGINAL = "research/joint-research-final/sweep/top-limit.json"
FINITE = "research/joint-research-final/finalist-refinement.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_manifest(root=ROOT):
    manifest = read(root / "research/manifest.json")
    for relative, expected in manifest["files"].items():
        data = (root / relative).read_bytes()
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError(f"Research artifact changed: {relative}")
    return len(manifest["files"])


def checked_limit(data):
    saved = result_from_data(data.get("result", data))
    result = validate_joint(
        saved.candidate, saved.r, saved.config,
        solver_status=saved.solver_status, repair_edges=saved.repair_edges,
        fixed_p0=saved.fixed_p0,
    )
    if result.status != "joint_asymptotic_strict" or abs(result.mu0 - saved.mu0) > 1e-12:
        raise ValueError(f"Saved limit failed independent validation: {result.status}")
    return saved, result


def reproduce(output, *, verify_only=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    files = verify_manifest()
    best, checked_best = checked_limit(read(ROOT / LIMIT))
    parent, _ = checked_limit(read(ROOT / ORIGINAL))
    refinement_data = read(ROOT / FINITE)
    refined, _ = checked_limit(refinement_data["result"])
    finite = refinement_data["best_finite"]
    status, diagnostics = validate_finite(
        refined.candidate, np.asarray(finite["P"]), finite["h"],
        numerics=refined.config.lift_numerics(),
    )
    if status != "finite_h_numeric_strict" or abs(diagnostics["mu"] - finite["mu"]) > 1e-12:
        raise ValueError(f"Saved finite witness failed independent validation: {status}")

    output.mkdir(parents=True)
    ledger = SolverLedger(dict(joint_sdp=3, finite_tree_linear=3, high_precision_audit=4))
    audits = {
        "saved_limit": ledger.invoke("high_precision_audit", lambda: high_precision_limit_audit(best)),
        "saved_finite": ledger.invoke("high_precision_audit", lambda: high_precision_audit(
            refined.candidate.x, finite["phi"], finite["P"])),
    }
    if not all(audit.get("positive_pivots", audit.get("positive_grounded_pivots", False))
               for audit in audits.values()):
        raise ValueError("A saved witness failed its Decimal pivot audit")
    summary = dict(
        source_hash=_code_hash(), artifact_files_verified=files,
        saved_limit_mu=checked_best.mu0, saved_finite_mu=diagnostics["mu"],
        finite_diagnostics=diagnostics, verify_only=verify_only,
        proof_available=False,
    )
    if not verify_only:
        c = best.candidate
        fresh = solve_joint_limit(
            c.x, c.s, c.groups, c.allowed_mask, best.config,
            backbone_cycle=c.backbone_cycle, repair_edges=best.repair_edges,
            fixed_p0=best.fixed_p0, family_id=c.family_id, ledger=ledger,
        )
        _write(output / "limit.json", fresh.data())
        if fresh.status != "joint_asymptotic_strict" or abs(fresh.mu0 - best.mu0) > 1e-9:
            raise ValueError(f"Fresh limiting solve did not reproduce the record: {fresh.status}")
        c = parent.candidate
        cfg = replace(parent.config, secondary=True, gamma0=.8 * parent.diagnostics["full_gap"])
        fresh_refined = solve_joint_limit(
            c.x, c.s, c.groups, c.allowed_mask, cfg,
            backbone_cycle=c.backbone_cycle, repair_edges=parent.repair_edges,
            fixed_p0=c.p0, family_id=c.family_id, ledger=ledger,
        )
        _write(output / "refined-limit.json", fresh_refined.data())
        if fresh_refined.status != "joint_asymptotic_strict":
            raise ValueError(f"Fresh fixed-P0 refinement failed: {fresh_refined.status}")
        correction = validate_correction(
            fresh_refined.candidate, fresh_refined.r,
            numerics=cfg.lift_numerics(), solver_status=fresh_refined.solver_status,
        )
        lifts = [lift_candidate(correction, h, numerics=cfg.lift_numerics(), ledger=ledger)
                 for h in (.0003, .0001, .00003)]
        _write(output / "finite-lifts.json", [asdict(lift) for lift in lifts])
        strict = [lift for lift in lifts if lift.status == "finite_h_numeric_strict"]
        if not strict:
            raise ValueError("Fresh correction has no independently strict finite lift")
        finite_best = max(strict, key=lambda lift: lift.mu)
        if abs(finite_best.mu - finite["mu"]) > 1e-9:
            raise ValueError("Fresh finite density differs from the saved witness")
        audits["fresh_limit"] = ledger.invoke("high_precision_audit", lambda: high_precision_limit_audit(fresh))
        audits["fresh_finite"] = ledger.invoke("high_precision_audit", lambda: high_precision_audit(
            c.x, finite_best.phi, finite_best.P))
        if not all(audit.get("positive_pivots", audit.get("positive_grounded_pivots", False))
                   for audit in audits.values()):
            raise ValueError("A fresh witness failed its Decimal pivot audit")
        summary.update(fresh_limit_mu=fresh.mu0, fresh_finite_mu=finite_best.mu)
    summary.update(audits=audits, counts=ledger.counts, solver_events=ledger.events)
    _write(output / "verification.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New scratch output directory")
    parser.add_argument("--verify-only", action="store_true", help="Recompute saved diagnostics without conic solves")
    args = parser.parse_args()
    result = reproduce(args.output, verify_only=args.verify_only)
    print(json.dumps({key: result[key] for key in (
        "saved_limit_mu", "saved_finite_mu", "counts", "proof_available",
    )}))


if __name__ == "__main__":
    main()
