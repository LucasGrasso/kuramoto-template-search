import numpy as np


def validate_input(x, phi, allowed=None):
    x = np.asarray(x, dtype=float)
    phi = np.asarray(phi, dtype=float)

    if x.ndim != 1 or phi.shape != x.shape:
        raise ValueError("x and phi must have the same 1D shape")

    if len(x) < 2:
        raise ValueError("At least two clusters are required")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(phi)):
        raise ValueError("Inputs must be finite")

    if np.any(x <= 0):
        raise ValueError("All cluster sizes must be positive")

    if not np.isclose(x.sum(), 1.0, atol=1e-10):
        raise ValueError("Cluster sizes must sum to one")

    # Differences modulo 2*pi.
    phase_diff = np.angle(np.exp(1j * (phi - phi[0])))

    if np.max(np.abs(phase_diff)) < 1e-8:
        raise ValueError(
            "All phases are synchronized. " "This cannot be a nonsynchronous witness."
        )

    if allowed is not None:
        allowed = np.asarray(allowed, dtype=bool)

        if allowed.shape != (len(x), len(x)):
            raise ValueError("Invalid allowed matrix shape")

        if not np.array_equal(allowed, allowed.T):
            raise ValueError("allowed must be symmetric")

    return x, phi, allowed
