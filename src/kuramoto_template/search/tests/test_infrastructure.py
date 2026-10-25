"""Shared run behavior, independent of random-search parent selection."""

import json
from pathlib import Path

import numpy as np
import pytest

from kuramoto_template.mutations import perturb_phase
from kuramoto_template.search import infrastructure
from kuramoto_template.search.evaluator import EvaluationConfig, TemplateEvaluation
from kuramoto_template.search.infrastructure import (
    ArchiveEntry, DiversityArchive, MutationProposal, SearchBudget,
    SearchBudgetExhausted, SearchRecord, SearchResult, SearchSession,
)
from kuramoto_template.search import random_search
from kuramoto_template.template import PhaseTemplate


class CountingEvaluator:
    config = EvaluationConfig()

    def __init__(self) -> None:
        self.calls: list[int | None] = []

    def evaluate(
        self, template: PhaseTemplate, *, max_solver_calls: int | None = None,
    ) -> TemplateEvaluation:
        self.calls.append(max_solver_calls)
        calls = 2 if max_solver_calls is None else min(2, max_solver_calls)
        return TemplateEvaluation(template, "near_feasible", margin_gamma=-0.1, sdp_calls=calls)


def seed_template() -> PhaseTemplate:
    return PhaseTemplate(np.full(5, 0.2), np.arange(5) * 2 * np.pi / 5)


def test_original_import_paths_are_reexported() -> None:
    assert random_search.ArchiveEntry is ArchiveEntry
    assert random_search.DiversityArchive is DiversityArchive
    assert random_search.MutationProposal is MutationProposal
    assert random_search.SearchRecord is SearchRecord
    assert random_search.SearchResult is SearchResult
    assert random_search.evaluation_data is infrastructure.evaluation_data


def test_branches_share_one_global_budget_and_trace() -> None:
    seed = seed_template()
    evaluator = CountingEvaluator()
    with SearchSession([seed], evaluator, seed=42, max_sdp_calls=7) as session:
        root = session.evaluate_seeds()[0]
        first = session.record(MutationProposal("perturb", {"delta": 0.01}, perturb_phase(seed, 0, 0.01)), root.id, "rollout")
        assert first.template is not None
        second = session.record(MutationProposal("perturb", {"delta": 0.02}, perturb_phase(first.template, 1, 0.02)), first.id, "rollout")
        sibling = session.record(MutationProposal("perturb", {"delta": 0.03}, perturb_phase(seed, 0, 0.03)), root.id, "rollout")
        assert evaluator.calls == [7, 5, 3, 1]
        assert session.budget.stop_reason() == "sdp_budget"
        assert [r.parent_id for r in session.trace] == [None, root.id, first.id, root.id]
        with pytest.raises(SearchBudgetExhausted):
            session.record(MutationProposal("perturb", {}, perturb_phase(seed, 2, 0.04)), second.id, "rollout")
        result = session.finish("sdp_budget")
        assert result.sdp_calls == 7 and result.evaluations == 4 and result.proposals == 3
        assert sibling.id == 3
        assert len(result.trace) == 4


def test_seed_reservation_and_duplicates_have_distinct_accounting() -> None:
    seed = seed_template()
    evaluator = CountingEvaluator()
    with SearchSession([seed, seed], evaluator, seed=0, max_sdp_calls=3) as session:
        records = session.evaluate_seeds()
        assert len(records) == 2
        assert evaluator.calls == [2, 1]
        duplicate = session.record(MutationProposal("perturb", {"delta": 0}, seed), records[0].id, "rollout")
        assert duplicate.disposition == "duplicate" and duplicate.evaluation is None
        rejected = session.record(MutationProposal("merge", {}, message="invalid merge"), records[1].id, "rollout")
        assert rejected.disposition == "rejected" and rejected.message == "invalid merge"
        result = session.finish("sdp_budget")
        assert result.evaluations == 2 and result.proposals == 2 and result.sdp_calls == 3
        with pytest.raises(RuntimeError, match="already been evaluated"):
            session.evaluate_seeds()


def test_budget_wall_time_and_sdp_limit_precedence() -> None:
    now = [10.0]
    budget = SearchBudget(2, 5.0, clock=lambda: now[0])
    assert budget.remaining(reserve=1) == 1
    assert budget.stop_reason() is None
    now[0] = 15.0
    assert budget.stop_reason() == "wall_time"
    budget.sdp_calls = 2
    assert budget.stop_reason() == "sdp_budget"


@pytest.mark.parametrize("kwargs", [
    {"max_sdp_calls": 0}, {"max_sdp_calls": 1.5}, {"max_sdp_calls": True},
    {"max_wall_seconds": 0}, {"max_wall_seconds": np.nan},
])
def test_invalid_budget(kwargs) -> None:
    with pytest.raises(ValueError):
        SearchBudget(**kwargs)


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_shared_archive_size(size) -> None:
    with pytest.raises(ValueError):
        DiversityArchive(size)


def test_session_closes_trace_on_exception_without_overwriting(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    session = SearchSession([seed_template()], CountingEvaluator(), seed=4, output=destination)
    with pytest.raises(RuntimeError, match="interrupted"):
        with session:
            session.evaluate_seeds()
            # Each record is already flushed, even before leaving the context.
            rows = (destination / "trace.jsonl").read_text().splitlines()
            assert len(rows) == 1 and json.loads(rows[0])["evaluation"]["margin_gamma"] == -0.1
            raise RuntimeError("interrupted")
    assert not (destination / "result.json").exists()
    with pytest.raises(RuntimeError, match="with block"):
        session.record(MutationProposal("merge", {}), None, "rollout")
    with pytest.raises(RuntimeError, match="cannot be reopened"):
        session.__enter__()
    before = (destination / "trace.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        with SearchSession([seed_template()], CountingEvaluator(), seed=4, output=destination):
            pass
    assert (destination / "trace.jsonl").read_bytes() == before


def test_write_artifact_is_json_safe_exclusive_and_confined(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    with SearchSession(
        [seed_template()], CountingEvaluator(), seed=2, output=destination
    ) as session:
        session.evaluate_seeds()
        path = session.write_artifact(
            "extra.json", {"value": np.float64(1.5), "nonfinite": np.inf}
        )
        assert path == destination / "extra.json"
        assert path is not None
        assert json.loads(path.read_text()) == {"value": 1.5, "nonfinite": None}
        with pytest.raises(FileExistsError):
            session.write_artifact("extra.json", {"value": 2})
        with pytest.raises(ValueError, match="plain filename"):
            session.write_artifact("../escape.json", {"value": 2})


def test_shared_persistence_accepts_algorithm_config(tmp_path: Path) -> None:
    settings = {"algorithm": "future_nested_search", "seed": 17, "level": 2, "depth": 3}
    destination = tmp_path / "nested"
    with SearchSession([seed_template()], CountingEvaluator(), seed=17, output=destination,
                       search_config=settings) as session:
        session.evaluate_seeds()
        result = session.finish("test")
    metadata = json.loads((destination / "metadata.json").read_text())
    saved = json.loads((destination / "result.json").read_text())
    assert metadata["search_config"] == settings
    assert saved["sdp_calls"] == result.sdp_calls == 2
    assert saved["stop_reason"] == "test"
    assert saved["best"] is None
