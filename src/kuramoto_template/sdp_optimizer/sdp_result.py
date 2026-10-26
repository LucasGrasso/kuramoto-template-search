from dataclasses import dataclass
from typing import Optional

from numpy import float64
from numpy.typing import NDArray


@dataclass
class SDPResult:
    status: str
    mu: Optional[float] = None
    P: Optional[NDArray[float64]] = None
    degrees: Optional[NDArray[float64]] = None
    forces: Optional[NDArray[float64]] = None
    restoring: Optional[NDArray[float64]] = None

    max_force_error: Optional[float] = None
    min_restoring: Optional[float] = None
    min_cluster_eigenvalue: Optional[float] = None

    numerically_feasible: bool = False
    solve_time: Optional[float] = None
    message: str = ""
    optimized_mu: Optional[float] = None
    objective_gap: Optional[float] = None
