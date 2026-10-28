from dataclasses import replace
import json
from pathlib import Path

import pytest

from kuramoto_template.asymptotic_search.families import s7
from kuramoto_template.asymptotic_search.flow_search import FlowSearchConfig,_fallback_parent,run_flow_search


def small_config():
    return FlowSearchConfig(independent=3,balanced_updates=2,split_merge=2,compensated=2,
        saved_audit_count=0,primary_calls=20,secondary_calls=2,tree_calls=4,lift_attempts=2,audit_calls=2,
        shortlist_size=1,h_values=(.0003,))


@pytest.mark.integration
@pytest.mark.parametrize("boundary",["proposal","joint","finite"])
def test_flow_resume_at_paid_work_boundaries(tmp_path,boundary):
    cfg=small_config(); root=tmp_path/"resumed"

    def cancelled():
        checkpoint=root/"checkpoint.json"
        if not checkpoint.exists(): return False
        state=json.loads(checkpoint.read_text())
        if boundary=="proposal":
            return state["phase"]=="main" and state["pending"] is not None
        if boundary=="joint":
            return state["phase"]=="main" and state["cursor"]>=1 and state["pending"] is None
        return len(state["finite"])>=1

    partial=run_flow_search(cfg,output=root,cancelled=cancelled,parents=[("S7",_fallback_parent())])
    assert partial.state["stop_reason"]=="cancelled"
    resumed=run_flow_search(cfg,output=root,resume=True)
    fresh=run_flow_search(cfg,parents=[("S7",_fallback_parent())])
    assert resumed.state["phase"]=="done"
    assert resumed.state["counts"]==fresh.state["counts"]
    assert resumed.state["trace"]==fresh.state["trace"]
    assert resumed.state["best_limit_mu"]==fresh.state["best_limit_mu"]
    again=run_flow_search(cfg,output=root,resume=True)
    assert again.state["counts"]==resumed.state["counts"]
    assert (root/"summary.csv").exists() and (root/"paired.json").exists()
    with pytest.raises(ValueError,match="Incompatible"):
        run_flow_search(replace(cfg,seed=1),output=root,resume=True)


@pytest.mark.integration
def test_exact_cache_and_resource_caps(monkeypatch):
    from kuramoto_template.asymptotic_search import flow_search

    c=s7(0.)
    def repeated(*args,**kwargs):
        return c,dict(family="repeat",q_mass_ratio=None,q_cross_density=None,flow_target=None,
                      backbone_winding=1,operation="restart")
    monkeypatch.setattr(flow_search,"prepare_proposal",repeated)
    cfg=FlowSearchConfig(independent=2,balanced_updates=0,split_merge=0,compensated=0,
        saved_audit_count=0,primary_calls=3,secondary_calls=0,tree_calls=0,audit_calls=0,shortlist_size=0)
    result=run_flow_search(cfg,parents=[("S7",_fallback_parent())])
    assert result.state["cache_hits"]==1
    assert result.state["counts"]["joint_sdp"]==2
    assert not result.state["counts"]["compensation_qp"]
    assert not result.state["counts"]["finite_tree_linear"]
    assert result.state["archives"]["rigorously_certified"]==[]


@pytest.mark.integration
def test_identical_saved_input_filter_comparison_is_paid(tmp_path):
    from kuramoto_template.asymptotic_search.models import candidate_data
    from kuramoto_template.experiments.reporting import json_safe

    source=tmp_path/"saved"/"case"; source.mkdir(parents=True)
    (source/"input.json").write_text(json.dumps(json_safe(dict(candidate=candidate_data(s7(.03))))))
    cfg=FlowSearchConfig(independent=0,balanced_updates=0,split_merge=0,compensated=0,
        saved_inputs=str(tmp_path/"saved"),saved_audit_count=1,primary_calls=2,
        secondary_calls=0,tree_calls=0,audit_calls=0,shortlist_size=0)
    result=run_flow_search(cfg,parents=[("S7",_fallback_parent())])
    comparison=result.state["filter_comparison"][0]
    assert comparison["screen"]["status"]=="aggregate_rejected"
    assert comparison["actual_unfiltered_primary_calls"]==1
    assert comparison["counterfactual_filtered_primary_calls"]==0
    assert result.state["counts"]["joint_sdp"]==2
