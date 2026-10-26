"""Configuration for the finite SDP-budget mutation task."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class EnvironmentConfig:
    min_clusters: int = 3
    max_clusters: int = 24
    max_sdp_calls: int = 128
    max_steps: int = 1280
    max_zero_cost_steps: int = 64
    max_wall_seconds: float | None = None
    reward_scale: float = 1000.0
    separation_min: float = 1e-4
    separation_max: float = .1
    phase_scale: float = .25
    resize_fraction: float = .1

    def __post_init__(self):
        for name, lower in (("min_clusters", 2), ("max_clusters", 2),
                            ("max_sdp_calls", 3), ("max_steps", 1), ("max_zero_cost_steps", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if self.min_clusters > self.max_clusters:
            raise ValueError("min_clusters must be <= max_clusters")
        for name in ("reward_scale", "separation_min", "separation_max", "phase_scale", "resize_fraction"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.separation_min > self.separation_max:
            raise ValueError("separation_min must be <= separation_max")
        if self.resize_fraction >= 1:
            raise ValueError("resize_fraction must be < 1")
        if self.max_wall_seconds is not None and (
            not np.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0
        ):
            raise ValueError("max_wall_seconds must be finite and positive")
