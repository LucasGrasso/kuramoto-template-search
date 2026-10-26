"""Adaptive nested rollouts: schedule, archives, pruning and accounting."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from kuramoto_template.mutations import (
    dissolve_cluster,
    perturb_phase,
    split_cluster,
)
from kuramoto_template.search import amcs as amcs_module
from kuramoto_template.search.adaptive import (
    AdaptiveArchives,
    AdaptiveController,
    AdaptiveSearchConfig,
    _pruning_proposal,
    run_adaptive_search,
)
from kuramoto_template.search.evaluator import (
    EvaluationConfig,
    SolverDiagnostics,
    TemplateEvaluation,
)
from kuramoto_template.search.infrastructure import MutationProposal, SearchRecord
from kuramoto_template.search.random_search import twisted_seed
from kuramoto_template.template import PhaseTemplate


class StableLandscapeEvaluator:
    """One solver call; stable value depends on cluster count."""

    config = EvaluationConfig()

    def __init__(self, mus: dict[int, float] | None = None) -> None:
        self.mus = mus or {}
        self.calls: list[int | None] = []

    def evaluate(
        self, template: PhaseTemplate, *, max_solver_calls: int | None = None
    ) -> TemplateEvaluation:
        self.calls.append(max_solver_calls)
        calls = 1 if max_solver_calls is None else min(1, max_solver_calls)
        mu = self.mus.get(template.n_clusters, 0.1)
        diagnostic = SolverDiagnostics(
            "density",
            "optimal",
            P=np.eye(template.n_clusters),
            mu=mu,
            numerically_valid=True,
            stable=True,
            solver_agrees=True,
        )
        return TemplateEvaluation(
            template, "stable", stable_mu=mu, density=diagnostic, sdp_calls=calls
        )


class ConstantNearEvaluator:
    config = EvaluationConfig()

    def evaluate(
        self, template: PhaseTemplate, *, max_solver_calls: int | None = None
    ) -> TemplateEvaluation:
        calls = 1 if max_solver_calls is None else min(1, max_solver_calls)
        return TemplateEvaluation(
            template,
            "near_feasible",
            margin_gamma=-float(template.n_clusters),
            sdp_calls=calls,
        )


class HugeGammaEvaluator:
    """Even cluster counts look excellent by gamma but are never stable."""

    config = EvaluationConfig()

    def evaluate(
        self, template: PhaseTemplate, *, max_solver_calls: int | None = None
    ) -> TemplateEvaluation:
        calls = 1 if max_solver_calls is None else min(1, max_solver_calls)
        if template.n_clusters % 2 == 0:
            return TemplateEvaluation(
                template, "near_feasible", margin_gamma=1e12, sdp_calls=calls
            )
        diagnostic = SolverDiagnostics(
            "density",
            "optimal",
            P=np.eye(template.n_clusters),
            mu=0.6,
            numerically_valid=True,
            stable=True,
            solver_agrees=True,
        )
        return TemplateEvaluation(
            template, "stable", stable_mu=0.6, density=diagnostic, sdp_calls=calls
        )


def make_record(
    identifier: int,
    *,
    status: str,
    mu: float | None = None,
    gamma: float | None = None,
    n_clusters: int = 5,
) -> SearchRecord:
    template = twisted_seed(n_clusters)
    evaluation = TemplateEvaluation(
        template,
        status,
        stable_mu=mu,
        margin_gamma=gamma,
    )
    return SearchRecord(
        id=identifier,
        parent_id=None,
        operation="test",
        parameters={},
        selection="test",
        template=template,
        evaluation=evaluation,
        disposition="evaluated",
        best_mu=mu,
    )


def test_controller_escalates_then_resets_on_improvement() -> None:
    config = AdaptiveSearchConfig(
        level=1,
        depth=2,
        max_level=3,
        max_depth=4,
        stall_threshold=2,
        prune_probability=0.1,
        prune_increment=0.2,
        iterations=20,
    )
    controller = AdaptiveController(config, incumbent_mu=0.5)

    assert controller.after_episode(0.5) == "hold"
    assert (controller.level, controller.depth) == (1, 2)
    assert controller.after_episode(0.5) == "deepen"
    assert (controller.level, controller.depth) == (1, 3)
    assert controller.prune_probability == pytest.approx(0.3)

    assert controller.after_episode(0.5) == "hold"
    assert controller.after_episode(0.5) == "nest"
    assert (controller.level, controller.depth) == (2, 3)
    assert controller.prune_probability == pytest.approx(0.5)

    assert controller.after_episode(0.6) == "reset"
    assert (controller.level, controller.depth) == (1, 2)
    assert controller.prune_probability == pytest.approx(0.1)
    assert controller.stall == 0 and controller.escalations == 0
    assert controller.incumbent_mu == pytest.approx(0.6)


def test_controller_cycles_to_baseline_at_caps() -> None:
    config = AdaptiveSearchConfig(
        level=1,
        depth=1,
        max_level=1,
        max_depth=2,
        stall_threshold=1,
        prune_probability=0.0,
        prune_increment=0.5,
        iterations=20,
    )
    controller = AdaptiveController(config, incumbent_mu=0.5)
    decisions = [controller.after_episode(0.5) for _ in range(4)]
    assert decisions == ["deepen", "restart_schedule", "deepen", "restart_schedule"]
    assert (controller.level, controller.depth) == (1, 1)


def test_two_archives_keep_signals_separate() -> None:
    archives = AdaptiveArchives(
        stable_capacity=2,
        promising_capacity=3,
        promising_per_cluster=2,
    )
    archives.observe(
        [
            make_record(0, status="stable", mu=0.60, n_clusters=5),
            make_record(1, status="stable", mu=0.70, n_clusters=5),
            make_record(2, status="stable", mu=0.65, n_clusters=4),
            make_record(3, status="near_feasible", gamma=-0.3, n_clusters=4),
            make_record(4, status="near_feasible", gamma=-0.1, n_clusters=4),
            make_record(5, status="near_feasible", gamma=-0.2, n_clusters=4),
            make_record(6, status="near_feasible", gamma=1e9, n_clusters=6),
        ]
    )

    assert [record.id for record in archives.stable] == [1, 2]
    assert [record.id for record in archives.promising[4]] == [4, 5]
    assert [record.id for record in archives.promising[6]] == [6]
    assert archives.promising_records()[-1].id == 6
    assert all(
        record.evaluation is not None
        and record.evaluation.status == "stable"
        for record in archives.stable
    )


def test_promising_trimming_does_not_rank_across_cluster_buckets() -> None:
    archives = AdaptiveArchives(
        stable_capacity=1,
        promising_capacity=2,
        promising_per_cluster=2,
    )
    archives.observe(
        [
            make_record(0, status="near_feasible", gamma=-100, n_clusters=4),
            make_record(1, status="near_feasible", gamma=-200, n_clusters=4),
            make_record(2, status="near_feasible", gamma=1.0, n_clusters=6),
        ]
    )
    # Capacity pressure removes the weakest entry of the most populated
    # bucket, not the globally weakest gamma.
    assert [record.id for record in archives.promising[4]] == [0]
    assert [record.id for record in archives.promising[6]] == [2]


def test_dissolve_cluster_redistributes_mass() -> None:
    template = PhaseTemplate(
        np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.0, 0.5, 1.0, 1.5])
    )
    dissolved = dissolve_cluster(template, 1)
    assert dissolved.n_clusters == 3
    assert dissolved.x.sum() == pytest.approx(1.0)
    assert np.all(dissolved.x > 0)
    assert np.allclose(dissolved.phi, template.phi[[0, 2, 3]])
    with pytest.raises(ValueError):
        dissolve_cluster(twisted_seed(2), 0)


def test_pruning_proposal_undoes_a_recent_split() -> None:
    split_template = split_cluster(twisted_seed(5), 0, 0.5, 0.02)
    split_record = SearchRecord(
        id=0,
        parent_id=None,
        operation="split",
        parameters={"index": 0, "ratio": 0.5, "separation": 0.02},
        selection="nested_L0",
        template=split_template,
        evaluation=None,
        disposition="evaluated",
        best_mu=None,
    )
    stub = SimpleNamespace(
        rng=np.random.default_rng(7),
        trace=[split_record],
        archive=SimpleNamespace(entries=[]),
    )
    ctx = SimpleNamespace(session=stub)
    config = AdaptiveSearchConfig(
        iterations=1,
        level=0,
        depth=1,
        min_clusters=3,
        prune_probability=1.0,
    )
    archives = AdaptiveArchives()
    proposal = _pruning_proposal(cast(Any, ctx), config, archives)

    assert proposal.operation == "prune_undo_split"
    assert proposal.template is not None
    assert proposal.template.n_clusters == 5
    assert not np.array_equal(proposal.template.phi, twisted_seed(5).phi)


def test_adaptive_search_reevaluates_pruned_candidates(monkeypatch) -> None:
    monkeypatch.setattr(
        amcs_module,
        "sample_mutation",
        lambda template, rng, config: MutationProposal(
            "perturb",
            {"index": 0, "delta": 0.001},
            perturb_phase(template, 0, 0.001),
        ),
    )
    evaluator = StableLandscapeEvaluator({5: 0.60, 4: 0.55, 3: 0.50})
    result = run_adaptive_search(
        [twisted_seed(5)],
        evaluator,
        AdaptiveSearchConfig(
            seed=3,
            iterations=4,
            level=0,
            depth=1,
            branching=1,
            max_level=1,
            max_depth=2,
            stall_threshold=1,
            prune_probability=1.0,
            prune_increment=0.0,
            restart_probability=0,
            min_clusters=2,
            max_clusters=12,
            max_sdp_calls=40,
            archive_size=8,
            max_invalid_attempts=1,
        ),
    )

    pruned = [
        record
        for record in result.trace
        if record.operation in ("prune_merge", "prune_dissolve")
    ]
    assert pruned
    assert all(
        record.evaluation is not None
        for record in pruned
        if record.disposition == "evaluated"
    )
    assert any(episode.prune_operation is not None for episode in result.episodes)
    assert result.sdp_calls == result.evaluations


def test_adaptive_search_is_reproducible_with_same_seed() -> None:
    config = AdaptiveSearchConfig(
        seed=5,
        iterations=4,
        level=0,
        depth=1,
        branching=2,
        max_level=2,
        max_depth=3,
        stall_threshold=1,
        prune_probability=0.5,
        restart_probability=0.2,
        max_sdp_calls=30,
        archive_size=8,
    )
    first = run_adaptive_search(
        [twisted_seed(5)], ConstantNearEvaluator(), config
    )
    second = run_adaptive_search(
        [twisted_seed(5)], ConstantNearEvaluator(), config
    )

    assert [
        (
            record.operation,
            record.disposition,
            record.parent_id,
        )
        for record in first.trace
    ] == [
        (
            record.operation,
            record.disposition,
            record.parent_id,
        )
        for record in second.trace
    ]
    assert [asdict(episode) for episode in first.episodes] == [
        asdict(episode) for episode in second.episodes
    ]


def test_huge_gamma_never_becomes_the_stable_incumbent() -> None:
    result = run_adaptive_search(
        [twisted_seed(5)],
        HugeGammaEvaluator(),
        AdaptiveSearchConfig(
            seed=1,
            iterations=4,
            level=0,
            depth=1,
            branching=1,
            max_level=1,
            max_depth=2,
            stall_threshold=1,
            prune_probability=1.0,
            restart_probability=0,
            min_clusters=4,
            max_clusters=6,
            max_sdp_calls=40,
            archive_size=8,
        ),
    )
    assert result.best is not None
    assert result.best.evaluation.status == "stable"
    assert result.best.evaluation.stable_mu == pytest.approx(0.6)
    assert any(
        record.evaluation is not None
        and record.evaluation.status == "near_feasible"
        and record.evaluation.margin_gamma == pytest.approx(1e12)
        for record in result.trace
    )


def test_budget_is_respected_and_exhaustion_is_graceful() -> None:
    result = run_adaptive_search(
        [twisted_seed(5)],
        ConstantNearEvaluator(),
        AdaptiveSearchConfig(
            seed=2,
            iterations=100,
            level=1,
            depth=2,
            branching=4,
            max_level=2,
            max_depth=3,
            stall_threshold=1,
            prune_probability=1.0,
            restart_probability=0.1,
            max_sdp_calls=7,
            archive_size=8,
        ),
    )
    assert result.sdp_calls == 7
    assert result.stop_reason == "sdp_budget"
    assert result.evaluations <= 7
    assert result.episodes


def test_real_sampler_respects_cluster_bounds_and_mass() -> None:
    result = run_adaptive_search(
        [twisted_seed(5)],
        ConstantNearEvaluator(),
        AdaptiveSearchConfig(
            seed=4,
            iterations=8,
            level=0,
            depth=1,
            branching=2,
            max_level=1,
            max_depth=2,
            stall_threshold=1,
            prune_probability=1.0,
            restart_probability=0,
            min_clusters=4,
            max_clusters=5,
            max_sdp_calls=64,
            archive_size=16,
        ),
    )
    assert result.trace
    for record in result.trace:
        if record.template is None:
            continue
        assert 4 <= record.template.n_clusters <= 5
        assert record.template.x.sum() == pytest.approx(1.0)
        assert np.all(record.template.x > 0)
        assert np.all(np.isfinite(record.template.phi))


def test_adaptive_artifacts_are_persisted(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        amcs_module,
        "sample_mutation",
        lambda template, rng, config: MutationProposal(
            "perturb",
            {"index": 0, "delta": 0.001},
            perturb_phase(template, 0, 0.001),
        ),
    )
    destination = tmp_path / "adaptive"
    result = run_adaptive_search(
        [twisted_seed(5)],
        StableLandscapeEvaluator({5: 0.6, 4: 0.55}),
        AdaptiveSearchConfig(
            seed=9,
            iterations=2,
            level=0,
            depth=1,
            branching=1,
            max_level=1,
            max_depth=2,
            stall_threshold=1,
            prune_probability=1.0,
            restart_probability=0,
            min_clusters=2,
            max_clusters=12,
            max_sdp_calls=20,
            archive_size=8,
        ),
        output=destination,
    )

    metadata = json.loads((destination / "metadata.json").read_text())
    adaptive = json.loads((destination / "adaptive.json").read_text())
    lines = (destination / "trace.jsonl").read_text().splitlines()
    saved = json.loads((destination / "result.json").read_text())

    assert metadata["search_config"]["algorithm"] == "adaptive_amcs"
    assert adaptive["algorithm"] == "adaptive_amcs"
    assert adaptive["baseline_level"] == 0
    assert adaptive["max_level"] == 1
    assert len(adaptive["episodes"]) == len(result.episodes)
    assert adaptive["stable_archive"]
    assert adaptive["numerical_evidence_only"] is True
    assert all(
        0 <= record_id < len(lines)
        for episode in adaptive["episodes"]
        for record_id in episode["record_ids"]
    )
    assert saved["best"]["evaluation"]["stable_mu"] == pytest.approx(0.6)
    assert "NaN" not in (destination / "adaptive.json").read_text()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_level": -1},
        {"max_depth": -1},
        {"stall_threshold": 0},
        {"prune_probability": -0.1},
        {"prune_probability": 1.1},
        {"prune_increment": -1},
        {"prune_phase_scale": 0},
        {"stable_archive_size": 0},
        {"promising_archive_size": 0},
        {"promising_per_cluster": 0},
        {"undo_split_window": 0},
        {"prune_sources": 0},
        {"level": 3, "max_level": 2},
        {"depth": 4, "max_depth": 3},
    ],
)
def test_invalid_adaptive_config(kwargs: dict[str, Any]) -> None:
    with pytest.raises((ValueError, TypeError)):
        AdaptiveSearchConfig(**kwargs)
