"""Results for the fixed-degree stability-margin SDP."""

from dataclasses import dataclass, field

from .sdp_result import SDPResult


@dataclass
class StabilitySDPResult(SDPResult):
    """A valid negative/zero gamma is useful for ranking, not strict stability.

    numerically_valid checks every SDP constraint within residual_tolerance.
    The inherited numerically_feasible additionally requires gamma and both
    independently measured stability margins to exceed stability_tolerance.
    Neither flag is a rigorous mathematical certificate.
    """

    gamma: float | None = None
    target_mu: float | None = None
    numerically_valid: bool = False
    residuals: dict[str, float] = field(default_factory=dict)

