from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, eq=False)
class PhaseTemplate:
    """Candidate masses and phases, independent of any SDP solution.

    Masses must sum to one within absolute tolerance 1e-10 (no
    normalization). Phases are wrapped to [0, 2*pi). Synchronized states
    are allowed here; counterexample evaluators must reject them.
    Arrays are defensively copied and read-only. Equality is by identity.
    """

    x: np.ndarray
    phi: np.ndarray

    def __post_init__(self) -> None:
        x = np.array(self.x, dtype=np.float64, copy=True)
        phi = np.array(self.phi, dtype=np.float64, copy=True)
        if x.ndim != 1 or phi.ndim != 1 or x.shape != phi.shape:
            raise ValueError("x and phi must have the same 1D shape")
        if x.size < 2:
            raise ValueError("At least two clusters are required")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(phi)):
            raise ValueError("Inputs must be finite")
        if np.any(x <= 0):
            raise ValueError("All cluster masses must be positive")
        if not np.isclose(x.sum(), 1.0, atol=1e-10, rtol=0):
            raise ValueError("Cluster masses must sum to one (atol=1e-10)")
        phi = np.remainder(phi, 2 * np.pi)
        # Rounding can map tiny negative angles to exactly 2*pi.
        phi[phi >= 2 * np.pi] = 0.0
        x.setflags(write=False)
        phi.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "phi", phi)

    @property
    def n_clusters(self) -> int:
        return len(self.x)
