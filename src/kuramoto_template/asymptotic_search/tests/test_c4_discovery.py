import json

import numpy as np
import pytest

from kuramoto_template.asymptotic_search.c4_discovery import (
    C4DiscoveryConfig, generic_c4_split, generic_geometry, generic_parameters,
    reflection_geometry, run_c4_discovery, split_class, parent_orbits,
)


@pytest.mark.parametrize("parent_symmetry", [False, True])
def test_all_c4_binary_splits_start_with_equal_parent_masses(parent_symmetry):
    classes = set()
    for bits in range(1, 16):
        split = tuple(j for j in range(4) if bits & (1 << j))
        orbits = parent_orbits(split) if parent_symmetry else tuple((j,) for j in range(4))
        dim = len(orbits)-1 + sum(orbit[0] in split for orbit in orbits)-1
        c = reflection_geometry(split, np.zeros(dim), parent_symmetry=parent_symmetry)
        assert c.m == 4 + len(split)
        assert c.x @ c.s == pytest.approx(0, abs=1e-14)
        assert c.x @ c.s**2 == pytest.approx(1)
        for j in range(4):
            assert c.x[c.groups == j].sum() == pytest.approx(.25)
        assert np.all(c.p0[c.allowed_mask] == 1)
        assert np.all(c.p0[~c.allowed_mask] == 0)
        classes.add(split_class(split))
    assert len(classes) == 5


def test_parent_symmetry_is_derived_from_c4_partition_automorphisms():
    assert parent_orbits((0, 1, 2)) == ((0, 2), (1,), (3,))
    c = reflection_geometry((0, 1, 2), [.2, -.1, .3], parent_symmetry=True)
    assert c.x[0] == c.x[4]
    assert c.s[0] == c.s[4]
    # Splitting every parent has no free variables under all C4 automorphisms.
    assert parent_orbits((0, 1, 2, 3)) == ((0, 1, 2, 3),)
    assert reflection_geometry((0, 1, 2, 3), [], parent_symmetry=True).m == 8


def test_generic_splits_preserve_c4_mass_and_mean_without_s7(monkeypatch):
    from kuramoto_template.asymptotic_search import families, joint_outer

    def forbidden(*args, **kwargs):
        raise AssertionError("S7 seed must not be used")

    monkeypatch.setattr(families, "s7", forbidden)
    monkeypatch.setattr(joint_outer, "s7", forbidden)
    rng = np.random.default_rng(42)
    counts = set()
    for m in (5, 6, 7, 8, 9, 10, 12):
        c = generic_c4_split(m, rng)
        assert c.m == m
        for j in range(4):
            ids = c.groups == j
            assert c.x[ids].sum() == pytest.approx(.25)
            assert c.x[ids] @ c.s[ids] == pytest.approx(0, abs=1e-14)
        restored = generic_geometry(c, generic_parameters(c))
        assert restored.x == pytest.approx(c.x)
        assert restored.s == pytest.approx(c.s)
        counts.add(tuple(np.bincount(c.groups, minlength=4)))
    assert len(counts) == 7


@pytest.mark.integration
@pytest.mark.parametrize("neutral,parent_symmetry", [(False, False), (True, False), (False, True), (True, True)])
def test_discovery_keeps_rejections_and_counts_actual_calls(tmp_path, monkeypatch, neutral, parent_symmetry):
    from kuramoto_template.asymptotic_search.joint_limit import JointLimitConfig
    from kuramoto_template.asymptotic_search import families, joint_outer

    def forbidden(*args, **kwargs):
        raise AssertionError("S7 seed must not be used")

    monkeypatch.setattr(families, "s7", forbidden)
    monkeypatch.setattr(joint_outer, "s7", forbidden)
    cfg = C4DiscoveryConfig(reflection_repeats=1, generic_starts=1,
        parent_symmetry=parent_symmetry,
        local_calls=2, generic_local_starts=0, polish_calls=2, max_iterations=1,
        inner=JointLimitConfig(secondary=False, neutral=neutral))
    out = tmp_path / "discovery"
    state = run_c4_discovery(cfg, output=out)
    records = state["records"]
    solved = [r for r in records if "conic_calls" in r]
    assert state["stop_reason"] == "completed"
    assert state["counts"]["joint_sdp"] == sum(r["conic_calls"] for r in solved)
    assert state["counts"]["joint_sdp"] == sum(e["kind"] == "joint_sdp" for e in state["solver_events"])
    assert any(r["status"] != "joint_asymptotic_strict" for r in solved)
    strict = [r["mu0"] for r in solved if r["status"] == "joint_asymptotic_strict"]
    assert state["best_mu"] == (max(strict) if strict else None)
    assert state["best_search_mu"] == max(r["mu0"] for r in solved
        if r["status"] in ("joint_asymptotic_strict", "joint_neutral"))
    assert state["required_gamma"] == (0 if neutral else cfg.inner.gamma0)
    assert all((out / r["file"]).exists() for r in solved)
    assert all((out / r["input_file"]).exists() for r in solved)
    for r in solved:
        proposed = json.loads((out / r["input_file"]).read_text())["candidate"]
        assert len(proposed["groups"]) == len(proposed["x"])
        assert len(proposed["p0"]) == len(proposed["x"])
    assert len((out / "trace.jsonl").read_text().splitlines()) == len(records)
    assert json.loads((out / "result.json").read_text())["counts"] == state["counts"]
    for r in solved:
        if r["parent_id"] is not None:
            assert records[r["parent_id"]]["status"] in ("joint_asymptotic_strict", "joint_neutral")
    with pytest.raises(FileExistsError):
        run_c4_discovery(cfg, output=out)


@pytest.mark.parametrize("kwargs", [dict(seed=-1), dict(reflection_repeats=0),
    dict(generic_sizes=(4,)), dict(derivative_steps=(0,))])
def test_invalid_discovery_configuration(kwargs):
    with pytest.raises(ValueError):
        C4DiscoveryConfig(**kwargs)
