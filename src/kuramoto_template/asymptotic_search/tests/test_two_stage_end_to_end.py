from dataclasses import replace
import json

import numpy as np
import pytest

from kuramoto_template.asymptotic_search.families import s7,split_candidate
from kuramoto_template.asymptotic_search.finite_lift import validate_finite
from kuramoto_template.asymptotic_search.models import candidate_data,candidate_from_data
from kuramoto_template.asymptotic_search.search import TwoStageConfig, _shortlist, run_two_stage
from kuramoto_template.asymptotic_search.solvers import SolverLedger


def compact_config(**kwargs):
    return TwoStageConfig(stage1_proposals=4,stage1_lp_solves=1,stage1_screen_sdp_solves=1,
                          stage2_sdp_solves=4,shortlist_size=2,h_values=(.003,.001),**kwargs)


@pytest.mark.integration
def test_budgets_exact_cache_persistence_and_monotone_archive(tmp_path):
    config = compact_config()
    result = run_two_stage(config,output=tmp_path/"run",candidates=[s7(.01),s7(.01),s7(.003),s7(-.01)])
    assert result.stop_reason=="completed"
    assert result.state["cache_hits"]==1
    assert result.call_counts["stage1_lp"]==1
    assert result.call_counts["stage1_screen_sdp"]==1
    assert result.call_counts["stage2_correction_sdp"]==4
    assert len(result.state["shortlist"])<=2
    assert result.best_finite_numeric_mu is not None
    best = [e["best_mu"] for e in result.state["trace"] if e["kind"]=="finite_lift" and e["best_mu"] is not None]
    assert best==sorted(best)
    snapshot = json.loads((tmp_path/"run"/"top-candidate.json").read_text())
    c = candidate_from_data(snapshot["candidate"])
    assert validate_finite(c,np.array(snapshot["P"]),snapshot["h"])[0]=="finite_h_numeric_strict"
    assert snapshot["r"] is not None and snapshot["phi"] is not None
    assert result.call_counts["finite_tree_linear"]>0
    assert all(e["parent_id"] is None or e["parent_id"]<e["id"] for e in result.state["trace"])


@pytest.mark.integration
@pytest.mark.parametrize("cancel_poll",[3,6,8])
def test_cancel_resume_replays_without_repaying_calls(tmp_path,cancel_poll):
    config = compact_config()
    polls = 0

    def cancel():
        nonlocal polls
        polls += 1
        return polls>=cancel_poll

    partial = run_two_stage(config,output=tmp_path/"resume",cancelled=cancel)
    assert partial.stop_reason=="cancelled"
    result = run_two_stage(config,output=tmp_path/"resume",resume=True)
    fresh = run_two_stage(config)
    assert result.stop_reason=="completed"
    assert result.call_counts==fresh.call_counts
    assert result.state["trace"]==fresh.state["trace"]
    assert result.best_finite_numeric_mu==fresh.best_finite_numeric_mu
    again = run_two_stage(config,output=tmp_path/"resume",resume=True)
    assert again.call_counts==result.call_counts
    with pytest.raises(ValueError,match="differ"):
        run_two_stage(replace(config,rng_seed=9),output=tmp_path/"resume",resume=True)


def test_zero_solver_budget_and_failures_counted():
    result = run_two_stage(TwoStageConfig(stage1_proposals=1,stage1_lp_solves=0,
                          stage1_screen_sdp_solves=0,stage2_sdp_solves=0),candidates=[s7(-.01)])
    assert not any(result.call_counts.values())
    assert result.best_finite_numeric_mu is None
    ledger = SolverLedger(dict(example=1))
    with pytest.raises(RuntimeError):
        ledger.invoke("example",lambda: (_ for _ in ()).throw(RuntimeError("solver failed")))
    assert ledger.counts["example"]==1
    assert len(ledger.events)==1


def test_shortlist_covers_sizes_before_extra_masks():
    c = s7()
    mask,p = c.allowed_mask.copy(),c.p0.copy()
    mask[0,3] = mask[3,0] = False
    p[0,3] = p[3,0] = 0
    candidates = [c,replace(c,allowed_mask=mask,p0=p)]
    for size in (8,9,10,12):
        split = c
        while split.m<size:
            split = split_candidate(split,0,.5,.1)
        candidates.append(split)
    # Isolate the archive policy from the mathematical screen, tested separately.
    records = [dict(candidate=candidate_data(candidate),metrics=dict(
        status="stage1_screened",mu0=.69,dipole_slacks=(.01,.01))) for candidate in candidates]
    selected = _shortlist(records,5)
    assert {len(records[i]["candidate"]["x"]) for i in selected}=={7,8,9,10,12}


@pytest.mark.integration
def test_full_sdp_audit_uses_session_and_preserves_separate_accounting():
    cfg = TwoStageConfig(stage1_proposals=1,stage1_lp_solves=0,stage1_screen_sdp_solves=0,
                         stage2_sdp_solves=2,shortlist_size=1,finite_h_sdp_solves=1,
                         audit_finite=True,h_values=(.003,))
    result = run_two_stage(cfg)
    assert result.call_counts["full_density_sdp"]==1
    assert result.call_counts["full_margin_sdp"]==0
    audit = result.state["finite"][0]["full_sdp"]
    assert audit["session_sdp_calls"]==1
    assert not audit["family_lift"]
    assert audit["kind"]=="support_only_sdp_audit"


@pytest.mark.integration
def test_benchmark_controls_match_actual_full_calls_and_export(tmp_path):
    from kuramoto_template.asymptotic_search.benchmark import run_benchmark

    rows = run_benchmark(tmp_path/"pilot",seeds=(0,),proposals=1,full_calls=2)
    paired = [r for r in rows if r["algorithm"] in ("two_stage","random","amcs")]
    assert len(paired)==9
    assert all(r["matched_actual_calls"] for r in paired)
    assert all(r["full_search_calls"]+r["full_audit_calls"]==2 for r in paired)
    assert (tmp_path/"pilot"/"summary.csv").is_file()
    assert (tmp_path/"pilot"/"top-candidate.json").is_file()
    audit = json.loads((tmp_path/"pilot"/"top-decimal-audit.json").read_text())
    assert audit["positive_grounded_pivots"] and not audit["proof_available"]
