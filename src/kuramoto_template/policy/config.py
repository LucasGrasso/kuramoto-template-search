from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PolicyConfig:
    hidden: int = 128
    layers: int = 2
    target_mu: float = .6929077
    margin_target: float = .693
    learning_rate: float = 3e-4
    discount: float = 1.0
    tau: float = .01
    temperature: float = 1.0
    max_weight: float = 20.0
    gradient_clip: float = 5.0
    auxiliary_weight: float = 0.0

    def __post_init__(self):
        if self.hidden < 8 or self.layers < 1:
            raise ValueError("hidden >=8 and layers >=1 required")
        if not 0<self.target_mu<1 or not 0<self.margin_target<1:
            raise ValueError("Density targets must be in (0,1)")
        for name in ("learning_rate","temperature","max_weight","gradient_clip"):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 <= self.discount <= 1 or not 0 < self.tau <= 1:
            raise ValueError("Invalid discount or target update rate")
        if self.auxiliary_weight < 0 or not math.isfinite(self.auxiliary_weight):
            raise ValueError("Invalid auxiliary loss weight")
