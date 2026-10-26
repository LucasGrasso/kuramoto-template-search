import numpy as np


def circular_span(phi):
    """Length of the smallest containing arc; independent of cluster order."""
    angles = np.sort(np.remainder(np.asarray(phi, dtype=float), 2 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    return float(max(0.0, 2 * np.pi - gaps.max()))


def in_open_semicircle(phi):
    """An exact connected nonsynchronous equilibrium is impossible here.

    Leave a roundoff guard at the closed-semicircle boundary (e.g. antipodes).
    This is an analytic obstruction to strict stability, not an input error:
    margin optimization can still return a useful nonpositive search signal.
    """
    return circular_span(phi) < np.pi - 64 * np.finfo(float).eps


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

    if not np.isclose(x.sum(), 1.0, atol=1e-10, rtol=0):
        raise ValueError("Cluster sizes must sum to one")

    if circular_span(phi) < 1e-8:
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
