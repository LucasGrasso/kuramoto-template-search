import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from kuramoto_template.mutations import perturb_phase, split_cluster
from kuramoto_template.search import amcs as module
from kuramoto_template.search.amcs import (
    NestedSearchConfig,
    run_nested_search,
    twisted_seed,
)
from kuramoto_template.search.evaluator import (
    EvaluationConfig,
    SolverDiagnostics,
    TemplateEvaluation,
    TemplateEvaluator,
)
from kuramoto_template.search.infrastructure import MutationProposal
from kuramoto_template.template import PhaseTemplate


class StepLandscapeEvaluator:
    """Deceptive one-coordinate landscape: 0.5 -> 0.4 -> 0.3 -> 0.8."""

    config = EvaluationConfig()

    def __init__(self) -> None:
        self.calls: list[int | None] = []

    def evaluate(
        self, template: PhaseTemplate, *, max_solver_calls: int | None = None,
    ) -> TemplateEvaluation:
        self.calls.append(max_solver_calls)
        calls = 1 if max_solver_calls is None else min(1, max_solver_calls)
        step = round(float(template.phi[0]) / 0.01)
        mu = {0: 0.5, 1: 0.4, 2: 0.3, 3: 0.8}.get(step, 0.1)
        diagnostic = SolverDiagnostics(
            "density", "optimal", P=np.eye(template.n_clusters), mu=mu
        )
        return TemplateEvaluation(
            template, "stable", stable_mu=mu, density=diagnostic, sdp_calls=calls
        )


class InvalidThenStepEvaluator:
    """The first non-seed evaluation is an invalid representation."""

    config = EvaluationConfig()

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(
        self, template: PhaseTemplate, *, max_solver_calls: int | None = None,
    ) -> TemplateEvaluation:
        self.calls += 1
        calls = 1 if max_solver_calls is None else min(1, max_solver_calls)
        if self.calls == 2:
            return TemplateEvaluation(
                template, "invalid", sdp_calls=calls, message="invalid representation"
            )
        step = round(float(template.phi[0]) / 0.01)
        mu = {0: 0.5, 1: 0.4, 2: 0.3, 3: 0.8}.get(step, 0.1)
        diagnostic = SolverDiagnostics(
            "density", "optimal", P=np.eye(template.n_clusters), mu=mu
        )
        return TemplateEvaluation(
            template, "stable", stable_mu=mu, density=diagnostic, sdp_calls=calls
        )


class ConstantNearEvaluator:
    """No stable candidates; useful for graceful-stop and invariant tests."""

    config = EvaluationConfig()

    def evaluate(
        self, template: PhaseTemplate, *, max_solver_calls: int | None = None,
    ) -> TemplateEvaluation:
        calls = 1 if max_solver_calls is None else min(1, max_solver_calls)
        return TemplateEvaluation(
            template,
            "near_feasible",
            margin_gamma=float(-template.n_clusters),
            sdp_calls=calls,
        )


class SolverErrorEvaluator:
    config = EvaluationConfig()

    def evaluate(
        self, template: PhaseTemplate, *, max_solver_calls: int | None = None,
    ) -> TemplateEvaluation:
        calls = 1 if max_solver_calls is None else min(1, max_solver_calls)
        return TemplateEvaluation(
            template, "solver_error", sdp_calls=calls, message="fake failure"
        )


class HugeGammaEvaluator:
    """Even cluster counts look excellent by gamma but are not stable."""

    config = EvaluationConfig()

    def evaluate(
        self, template: PhaseTemplate, *, max_solver_calls: int | None = None,
    ) -> TemplateEvaluation:
        calls = 1 if max_solver_calls is None else min(1, max_solver_calls)
        if template.n_clusters % 2 == 0:
            return TemplateEvaluation(
                template,
                "near_feasible",
                margin_gamma=1e12,
                sdp_calls=calls,
            )
        diagnostic = SolverDiagnostics(
            "density", "optimal", P=np.eye(template.n_clusters), mu=0.6
        )
        return TemplateEvaluation(
            template, "stable", stable_mu=0.6, density=diagnostic, sdp_calls=calls
        )


def step_seed() -> PhaseTemplate:
    return PhaseTemplate(np.full(5, 0.2), np.arange(5) * 2 * np.pi / 5)


def advance_proposal(
    template: PhaseTemplate, rng: np.random.Generator, config: object
) -> MutationProposal:
    return MutationProposal(
        "advance", {"delta": 0.01}, perturb_phase(template, 0, 0.01)
    )


def split_then_perturb(
    template: PhaseTemplate, rng: np.random.Generator, config: object
) -> MutationProposal:
    if template.n_clusters == 5:
        return MutationProposal(
            "split",
            {"index": 0, "ratio": 0.5, "separation": 0.01},
            split_cluster(template, 0, 0.5, 0.01),
        )
    return MutationProposal(
        "perturb", {"index": 0, "delta": 0.01}, perturb_phase(template, 0, 0.01)
    )


def actions(result) -> list[tuple]:
    return [
        (
            record.id,
            record.parent_id,
            record.operation,
            record.parameters,
            record.disposition,
            record.best_mu,
            record.evaluation.status if record.evaluation is not None else None,
        )
        for record in result.trace
    ]


def test_nested_rollout_requires_two_worse_moves(monkeypatch):
    monkeypatch.setattr(module, "sample_mutation", advance_proposal)
    config = NestedSearchConfig(
        iterations=1,
        level=1,
        depth=2,
        branching=3,
        restart_probability=0,
        exploration_probability=0,
        max_sdp_calls=20,
        archive_size=8,
        max_invalid_attempts=2,
    )
    result = run_nested_search([step_seed()], StepLandscapeEvaluator(), config)

    assert result.best is not None
    assert result.best.evaluation.stable_mu == pytest.approx(0.8)
    assert len(result.episodes) == 1
    episode = result.episodes[0]
    path = result.episode_records(episode)
    steps = [
        round(float(record.template.phi[0]) / 0.01)
        for record in path
        if record.template is not None
    ]
    assert steps == [0, 1, 2, 3]
    assert episode.endpoint_id == path[-1].id
    assert episode.best_stable_mu == pytest.approx(0.8)
    assert episode.endpoint_status == "stable"
    assert episode.complete
    best_history = [
        record.best_mu for record in result.trace if record.best_mu is not None
    ]
    assert best_history == sorted(best_history)

    baseline = run_nested_search(
        [step_seed()],
        StepLandscapeEvaluator(),
        NestedSearchConfig(
            iterations=1,
            level=0,
            depth=1,
            branching=1,
            restart_probability=0,
            exploration_probability=0,
            max_sdp_calls=20,
            archive_size=8,
            max_invalid_attempts=2,
        ),
    )
    assert baseline.best is not None
    assert baseline.best.evaluation.stable_mu == pytest.approx(0.5)


def test_same_seed_reproduces_trace_and_episode_paths(monkeypatch):
    monkeypatch.setattr(module, "sample_mutation", advance_proposal)
    config = NestedSearchConfig(
        iterations=3,
        level=1,
        depth=2,
        branching=2,
        restart_probability=0.5,
        exploration_probability=0.3,
        max_sdp_calls=40,
        archive_size=8,
    )
    first = run_nested_search([step_seed()], StepLandscapeEvaluator(), config)
    second = run_nested_search([step_seed()], StepLandscapeEvaluator(), config)

    assert actions(first) == actions(second)
    assert [asdict(episode) for episode in first.episodes] == [
        asdict(episode) for episode in second.episodes
    ]
    assert first.best is not None and second.best is not None
    assert first.best.evaluation.stable_mu == second.best.evaluation.stable_mu


def test_budget_exhaustion_inside_recursion_is_graceful(monkeypatch):
    monkeypatch.setattr(module, "sample_mutation", advance_proposal)
    result = run_nested_search(
        [step_seed()],
        StepLandscapeEvaluator(),
        NestedSearchConfig(
            iterations=1,
            level=1,
            depth=3,
            branching=1,
            restart_probability=0,
            exploration_probability=0,
            max_sdp_calls=3,
            archive_size=4,
            max_invalid_attempts=1,
        ),
    )

    assert result.stop_reason == "sdp_budget"
    assert result.sdp_calls == 3 and result.evaluations == 3
    assert len(result.episodes) == 1 and not result.episodes[0].complete
    assert result.best is not None
    assert result.best.evaluation.stable_mu == pytest.approx(0.5)
    steps = [
        round(float(record.template.phi[0]) / 0.01)
        for record in result.trace
        if record.template is not None
    ]
    assert steps == [0, 1, 2]


def test_huge_gamma_never_displaces_stable_incumbent(monkeypatch):
    monkeypatch.setattr(module, "sample_mutation", split_then_perturb)
    result = run_nested_search(
        [step_seed()],
        HugeGammaEvaluator(),
        NestedSearchConfig(
            iterations=2,
            level=1,
            depth=2,
            branching=2,
            restart_probability=0,
            exploration_probability=0,
            max_sdp_calls=30,
            archive_size=8,
            max_invalid_attempts=2,
        ),
    )

    assert result.best is not None
    assert result.best.evaluation.status == "stable"
    assert result.best.evaluation.stable_mu == pytest.approx(0.6)
    assert any(
        entry.evaluation.status == "near_feasible"
        and entry.evaluation.margin_gamma == pytest.approx(1e12)
        for entry in result.archive
    )


def test_invalid_mutation_is_logged_and_skipped(monkeypatch):
    calls = 0

    def invalid_then_valid(
        template: PhaseTemplate, rng: np.random.Generator, config: object
    ) -> MutationProposal:
        nonlocal calls
        calls += 1
        if calls == 1:
            return MutationProposal(
                "merge", {"i": 0, "j": 2}, None, "antipodal cancellation"
            )
        return advance_proposal(template, rng, config)

    monkeypatch.setattr(module, "sample_mutation", invalid_then_valid)
    result = run_nested_search(
        [step_seed()],
        StepLandscapeEvaluator(),
        NestedSearchConfig(
            iterations=1,
            level=0,
            depth=1,
            branching=1,
            restart_probability=0,
            exploration_probability=0,
            max_sdp_calls=5,
            archive_size=4,
            max_invalid_attempts=2,
        ),
    )

    assert result.trace[0].operation == "seed"
    assert result.trace[1].operation == "merge"
    assert result.trace[1].disposition == "rejected"
    assert result.trace[2].operation == "advance"
    assert result.trace[2].disposition == "evaluated"
    assert result.best is not None
    assert result.best.evaluation.stable_mu == pytest.approx(0.5)


def test_invalid_evaluation_is_logged_but_never_used_as_parent(monkeypatch):
    calls = 0

    def varying_advance(
        template: PhaseTemplate, rng: np.random.Generator, config: object
    ) -> MutationProposal:
        nonlocal calls
        calls += 1
        return MutationProposal(
            "advance",
            {"delta": 0.01 * calls},
            perturb_phase(template, 0, 0.01 * calls),
        )

    monkeypatch.setattr(module, "sample_mutation", varying_advance)
    result = run_nested_search(
        [step_seed()],
        InvalidThenStepEvaluator(),
        NestedSearchConfig(
            iterations=1,
            level=0,
            depth=1,
            branching=1,
            restart_probability=0,
            exploration_probability=0,
            max_sdp_calls=5,
            archive_size=4,
            max_invalid_attempts=2,
        ),
    )

    assert [
        record.evaluation.status
        for record in result.trace
        if record.evaluation is not None
    ] == ["stable", "invalid", "stable"]
    assert result.trace[1].disposition == "evaluated"
    assert result.episodes[0].record_ids == [0, 2]
    assert result.best is not None
    assert result.best.evaluation.stable_mu == pytest.approx(0.5)


class RaisingAfterSeedEvaluator:
    config = EvaluationConfig()

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(
        self, template: PhaseTemplate, *, max_solver_calls: int | None = None,
    ) -> TemplateEvaluation:
        self.calls += 1
        if self.calls == 1:
            diagnostic = SolverDiagnostics(
                "density", "optimal", P=np.eye(template.n_clusters), mu=0.5
            )
            return TemplateEvaluation(
                template, "stable", stable_mu=0.5, density=diagnostic, sdp_calls=1
            )
        raise RuntimeError("unexpected evaluator failure")


def test_evaluator_exception_stops_safely():
    result = run_nested_search(
        [step_seed()],
        RaisingAfterSeedEvaluator(),
        NestedSearchConfig(
            iterations=2,
            level=1,
            depth=1,
            branching=1,
            restart_probability=0,
            exploration_probability=0,
            max_sdp_calls=10,
            archive_size=4,
        ),
    )

    assert result.stop_reason == "evaluator_error"
    assert len(result.trace) == 1
    assert result.best is not None
    assert result.best.evaluation.stable_mu == pytest.approx(0.5)
    assert result.episodes and not result.episodes[0].complete
    assert "RuntimeError: unexpected evaluator failure" in result.episodes[0].message


def test_solver_failures_remain_diagnostics_not_counterexamples():
    result = run_nested_search(
        [twisted_seed()],
        SolverErrorEvaluator(),
        NestedSearchConfig(
            iterations=2,
            level=1,
            depth=1,
            branching=2,
            restart_probability=0,
            exploration_probability=0,
            max_sdp_calls=10,
            archive_size=8,
        ),
    )

    assert result.best is None
    assert result.trace
    assert all(
        record.evaluation is not None
        and record.evaluation.status == "solver_error"
        for record in result.trace
    )
    assert result.episodes
    assert all(episode.endpoint_status == "solver_error" for episode in result.episodes)


def test_real_sampler_respects_cluster_bounds_and_invariants():
    result = run_nested_search(
        [twisted_seed(5)],
        ConstantNearEvaluator(),
        NestedSearchConfig(
            iterations=6,
            level=1,
            depth=2,
            branching=3,
            min_clusters=4,
            max_clusters=5,
            restart_probability=0,
            exploration_probability=0,
            max_sdp_calls=64,
            archive_size=16,
        ),
    )

    assert result.best is None
    assert any(record.template is not None for record in result.trace)
    for record in result.trace:
        if record.template is None:
            continue
        assert 4 <= record.template.n_clusters <= 5
        assert record.template.x.sum() == pytest.approx(1)
        assert np.all(record.template.x > 0)
        assert np.all(np.isfinite(record.template.phi))


def test_no_stable_candidates_is_graceful():
    result = run_nested_search(
        [twisted_seed()],
        ConstantNearEvaluator(),
        NestedSearchConfig(
            iterations=3,
            level=1,
            depth=1,
            branching=2,
            restart_probability=0,
            exploration_probability=0,
            max_sdp_calls=20,
            archive_size=8,
        ),
    )

    assert result.best is None
    assert result.archive
    assert all(record.best_mu is None for record in result.trace)
    assert all(
        record.evaluation is None or record.evaluation.status == "near_feasible"
        for record in result.trace
    )
    assert all(episode.endpoint_stable_mu is None for episode in result.episodes)


def test_wall_limit_stops_after_mandatory_seeds(monkeypatch):
    ticks = itertools.count()
    monkeypatch.setattr(module, "perf_counter", lambda: float(next(ticks)))
    result = run_nested_search(
        [twisted_seed()],
        ConstantNearEvaluator(),
        NestedSearchConfig(
            iterations=5,
            level=1,
            depth=1,
            branching=1,
            restart_probability=0,
            max_sdp_calls=20,
            max_wall_seconds=0.5,
        ),
    )

    assert result.evaluations == 1
    assert result.proposals == 0
    assert result.stop_reason == "wall_time"
    assert result.episodes == []


def test_paths_are_persisted_and_reviewable(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(module, "sample_mutation", advance_proposal)
    destination = tmp_path / "run"
    result = run_nested_search(
        [step_seed()],
        StepLandscapeEvaluator(),
        NestedSearchConfig(
            iterations=1,
            level=1,
            depth=2,
            branching=1,
            restart_probability=0,
            exploration_probability=0,
            max_sdp_calls=10,
            archive_size=8,
            max_invalid_attempts=1,
        ),
        output=destination,
    )

    metadata = json.loads((destination / "metadata.json").read_text())
    paths = json.loads((destination / "paths.json").read_text())
    lines = [
        json.loads(line)
        for line in (destination / "trace.jsonl").read_text().splitlines()
    ]
    saved = json.loads((destination / "result.json").read_text())

    assert metadata["search_config"]["algorithm"] == "nested_rollouts_fixed"
    assert metadata["search_config"]["level"] == 1
    assert paths["algorithm"] == "nested_rollouts_fixed"
    assert paths["level"] == 1 and paths["depth"] == 2
    assert len(paths["episodes"]) == len(result.episodes) == 1
    episode = paths["episodes"][0]
    assert episode["record_ids"] == result.episodes[0].record_ids
    assert all(0 <= record_id < len(lines) for record_id in episode["record_ids"])
    assert saved["best"]["evaluation"]["stable_mu"] == pytest.approx(0.8)
    assert len(lines) == len(result.trace)
    assert "NaN" not in (destination / "paths.json").read_text()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": -1},
        {"seed": -1},
        {"archive_size": 0},
        {"level": -1},
        {"depth": -1},
        {"branching": 0},
        {"max_invalid_attempts": 0},
        {"max_sdp_calls": 0},
        {"max_sdp_calls": True},
        {"max_wall_seconds": np.inf},
        {"restart_probability": 2},
        {"exploration_probability": -1},
        {
            "parent_elite_probability": 0,
            "parent_promising_probability": 0,
            "parent_random_probability": 0,
        },
        {"separation_min": 0},
        {"phase_scale": 0},
        {"resize_fraction": 1},
        {"mutation_weights": (0, 0, 0, 0)},
        {"mutation_weights": (1, -1, 1, 1)},
        {"min_clusters": 1},
        {"min_clusters": 6, "max_clusters": 4},
        {
            "min_clusters": 5,
            "max_clusters": 5,
            "mutation_weights": (1, 1, 0, 0),
            "depth": 1,
        },
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        NestedSearchConfig(**kwargs)


@pytest.mark.parametrize(
    "seeds,config",
    [
        ([], NestedSearchConfig()),
        ([twisted_seed(2)], NestedSearchConfig()),
        ([PhaseTemplate(np.full(5, 0.2), np.zeros(5))], NestedSearchConfig()),
        ([twisted_seed()], NestedSearchConfig(max_clusters=13)),
    ],
)
def test_invalid_run_inputs(seeds, config):
    with pytest.raises(ValueError):
        run_nested_search(seeds, ConstantNearEvaluator(), config)


@pytest.mark.integration
def test_real_nested_smoke():
    result = run_nested_search(
        [twisted_seed()],
        TemplateEvaluator(),
        NestedSearchConfig(
            iterations=1,
            level=1,
            depth=1,
            branching=2,
            restart_probability=0,
            exploration_probability=0,
            max_sdp_calls=12,
        ),
    )

    assert result.best is not None
    assert result.best.evaluation.stable_mu is not None
    assert result.best.evaluation.stable_mu > 0.59
    assert result.sdp_calls <= 12
    assert result.episodes
    assert result.best.evaluation.density is not None
    assert result.best.evaluation.density.min_cluster_eigenvalue is not None
    assert result.best.evaluation.density.min_cluster_eigenvalue > 0
