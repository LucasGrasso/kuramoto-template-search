"""Data and numerical settings for the linear, four-parent asymptotic family.

All results are numerical evidence. No status in this package is a proof.
"""

from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np

from kuramoto_template.template import PhaseTemplate

MU_RECORD = 0.6929077063695864
Q_STAR = 0.6505691669447988
W_STAR = 1.3982781340204612


@dataclass(frozen=True, eq=False)
class LimitingCandidate:
    x: np.ndarray
    s: np.ndarray
    groups: np.ndarray
    p0: np.ndarray
    allowed_mask: np.ndarray
    backbone_cycle: tuple[int, int, int, int]
    family_id: str = "custom"

    def __post_init__(self):
        x, s = np.array(self.x, dtype=np.float64), np.array(self.s, dtype=np.float64)
        PhaseTemplate(x, s)  # Shared mass/finite-array contract; s itself is not wrapped.
        groups = np.array(self.groups)
        m = len(x)
        if groups.shape != (m,) or not np.all(np.isin(groups, [0, 1, 2, 3])):
            raise ValueError("groups must contain integer parent labels 0..3")
        groups = groups.astype(np.int64)
        if len(np.unique(groups)) != 4:
            raise ValueError("All four parent groups must be nonempty")
        p = np.array(self.p0, dtype=np.float64)
        mask = np.array(self.allowed_mask, dtype=bool)
        if p.shape != (m, m) or mask.shape != p.shape or not np.all(np.isfinite(p)):
            raise ValueError("p0 and mask must be finite square matrices matching x")
        if not np.array_equal(mask, mask.T) or not np.all(np.diag(mask)):
            raise ValueError("mask must be symmetric and allow the clique diagonal")
        opposite = (groups[:, None] - groups[None, :]) % 4 == 2
        if np.any(mask & opposite):
            raise ValueError("This expansion prohibits opposite-parent links")
        # Retain raw conic output, including roundoff violations, for independent checks.
        # These tolerances are input sanity checks, NOT acceptance tolerances.
        if not np.array_equal(p,p.T) or np.any(p[~mask] != 0) or np.any(np.diag(p) != 1):
            raise ValueError("Symmetry, forbidden zeros and clique diagonal must hold exactly")
        if max(-p.min(), p.max() - 1) > 1e-7:
            raise ValueError("p0 violates symmetry, box, mask, or clique convention")
        cycle = tuple(self.backbone_cycle)
        if len(cycle) != 4 or any(isinstance(a, bool) or not isinstance(a, (int, np.integer))
                                  or not 0 <= a < m for a in cycle):
            raise ValueError("backbone must contain four cluster indices")
        if tuple(groups[list(cycle)]) != (0, 1, 2, 3):
            raise ValueError("backbone must visit parents 0,1,2,3 in order")
        if any(not mask[a,b] for a,b in zip(cycle,cycle[1:]+cycle[:1])):
            raise ValueError("mask must allow the backbone edges")
        for name, value in (("x", x), ("s", s), ("groups", groups),
                            ("p0", p), ("allowed_mask", mask)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "backbone_cycle", cycle)

    @property
    def m(self):
        return len(self.x)

    @property
    def mask_hash(self):
        return hashlib.sha256(self.allowed_mask.tobytes()).hexdigest()

    def template(self, h: float) -> PhaseTemplate:
        if not np.isfinite(h) or h <= 0:
            raise ValueError("h must be finite and positive")
        return PhaseTemplate(self.x, self.groups * (np.pi / 2) + h * self.s)


@dataclass(frozen=True)
class Numerics:
    residual_tolerance: float = 1e-9
    rank_tolerance: float = 1e-11
    slack_tolerance: float = 1e-8
    strict_gap: float = 1e-7
    energy_floor: float = 1e-8
    min_mass: float = 1e-5
    backbone_density: float = 1e-4
    connectivity_margin: float = 1e-5
    r_max: float = 30.0
    inward_slack: float = 1e-3
    trial_h: float = 0.01
    max_repair_coefficient: float = 100.0
    finite_sdp_cluster_margin: float = 1e-10
    solver: str = "CLARABEL"
    solver_options: dict[str, Any] = field(default_factory=lambda: dict(
        tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=300))

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name in ("solver", "solver_options"):
                continue
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_mass >= 0.25 or self.backbone_density >= 1 or self.trial_h >= 1:
            raise ValueError("Invalid mass, backbone, or h bound")
        object.__setattr__(self, "solver_options", dict(self.solver_options))


@dataclass(frozen=True)
class Stage1Metrics:
    mu0: float
    force0_inf: float
    cut0_inf: float
    hard_gap: float
    internal_energy: float
    dipole_slacks: tuple[float, float]
    solve_residual: float
    range_residual: float
    box_violation: float
    status: str
    message: str = ""


@dataclass(eq=False)
class Stage2Result:
    candidate: LimitingCandidate
    r: np.ndarray | None
    status: str
    solver_status: str = "not_run"
    force1_inf: float | None = None
    budget_identity_error: float | None = None
    effective_gap: float | None = None
    full_rescaled_gap: float | None = None
    hard_gap: float | None = None
    cone_violation: float | None = None
    schur_error: float | None = None
    reported_gamma: float | None = None
    message: str = ""


def candidate_data(c: LimitingCandidate) -> dict:
    return {name: getattr(c, name) for name in c.__dataclass_fields__}


def candidate_from_data(data: dict) -> LimitingCandidate:
    return LimitingCandidate(**data)
