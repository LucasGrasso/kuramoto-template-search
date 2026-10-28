"""The committed record must remain independently checkable after cleanup."""

import json
from pathlib import Path

import numpy as np
import pytest

from kuramoto_template.asymptotic_search.finite_lift import validate_finite
from kuramoto_template.asymptotic_search.joint_limit import result_from_data, validate_joint
from kuramoto_template.asymptotic_search.joint_outer import JointOuterConfig

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("relative,mu", [
    ("joint-research-final/sweep/top-limit.json", .6929033189148635),
    ("s7-analysis/evidencia/fine-offset-1e-07--0.0001.json", .6929072637491324),
])
def test_tracked_limits_pass_independent_math_checks(relative, mu):
    data = json.loads((ROOT / "research" / relative).read_text(encoding="utf-8"))
    saved = result_from_data(data.get("result", data))
    checked = validate_joint(saved.candidate, saved.r, saved.config,
                             solver_status=saved.solver_status,
                             fixed_p0=saved.fixed_p0, repair_edges=saved.repair_edges)
    assert checked.status == "joint_asymptotic_strict"
    assert checked.mu0 == pytest.approx(mu, abs=1e-14, rel=0)
    assert not checked.proof_available


def test_tracked_finite_witness_passes_independent_metrics():
    data = json.loads((ROOT / "research/joint-research-final/finalist-refinement.json").read_text())
    refined = result_from_data(data["result"])
    finite = data["best_finite"]
    status, diagnostics = validate_finite(refined.candidate, np.asarray(finite["P"]), finite["h"],
                                          numerics=refined.config.lift_numerics())
    assert status == "finite_h_numeric_strict"
    assert diagnostics["mu"] == pytest.approx(.6926591963503084, abs=1e-14, rel=0)
    assert diagnostics["cluster_gap"] > diagnostics["roundoff_floor"]
    assert not diagnostics["proof_available"]


def test_unused_full_template_fallback_cannot_spend_calls():
    with pytest.raises(ValueError, match="full_sdp_calls must be zero"):
        JointOuterConfig(full_sdp_calls=1)
