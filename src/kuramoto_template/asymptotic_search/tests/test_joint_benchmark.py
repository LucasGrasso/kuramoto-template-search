import json

import pytest

from kuramoto_template.asymptotic_search.joint_benchmark import run_joint_benchmark


@pytest.mark.integration
def test_calibration_precedes_research_and_reports_unmatched_costs(tmp_path):
    result,rows=run_joint_benchmark(tmp_path/"benchmark",sweep_proposals=5,paired=True)
    calibration=json.loads((tmp_path/"benchmark"/"calibration"/"calibration.json").read_text())
    assert len(calibration["records"])==15
    assert len(rows)==90
    assert all(row["all_conic_calls"]<=2 for row in rows)
    assert all(row["full_sdp_calls"]==0 for row in rows)
    assert any(not row["matched_actual_all_conic"] for row in rows)
    assert (tmp_path/"benchmark"/"paired-summary.csv").is_file()
    assert (tmp_path/"benchmark"/"findings.md").is_file()
    assert result.call_counts["joint_sdp"]==5
