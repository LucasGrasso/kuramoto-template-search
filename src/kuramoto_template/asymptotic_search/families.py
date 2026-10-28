"""S7 calibration, asymmetric splits, and multiple neighbor support masks."""

from dataclasses import replace

import numpy as np

from .models import LimitingCandidate, Q_STAR, W_STAR


def normalize_offsets(x, s):
    s = np.asarray(s, dtype=float) - np.dot(x, s)
    scale = np.sqrt(np.dot(x, s * s))
    if not np.isfinite(scale) or scale <= 1e-14:
        raise ValueError("Offsets have zero or unresolved variance")
    return s / scale


def parent_mask(groups):
    groups = np.asarray(groups)
    return (groups[:, None] - groups[None, :]) % 4 != 2


def s7(t=0.01, *, w=W_STAR):
    q = Q_STAR - t
    if not 0 <= q <= 1 or not np.isfinite(w) or w <= 0:
        raise ValueError("Invalid S7 parameters")
    a = 1 / (7 + q)
    x = np.array([a] * 6 + [1 - 6 * a])
    groups = np.array([0, 0, 1, 1, 2, 2, 3])
    p = parent_mask(groups).astype(float)
    for i, j in ((0, 3), (1, 2), (2, 5), (3, 4)):
        p[i, j] = p[j, i] = q
    return LimitingCandidate(x, [-1, 1, -w, w, -1, 1, 0], groups, p,
                             parent_mask(groups), (0, 2, 4, 6), f"S7:t={t:g}")


def appendix_a_correction(c, kappa=1.9):
    """Optional S7 reference from arXiv:2610.21271v1, Appendix A.1 (12)-(14)."""
    q, w = c.p0[0, 3], c.s[3]
    expected = s7(Q_STAR - q, w=w)
    if c.m != 7 or any(not np.allclose(getattr(c, name), getattr(expected, name),
                                       atol=1e-13, rtol=0)
                      for name in ("x", "s", "groups", "p0")):
        raise ValueError("Appendix correction requires the raw S7 ordering and scale")
    u = .5 * (2*q*q*w*w + 2*q*q*w - q*q - 6*q + 4*w*w - 2*w + 3)
    A1, A2 = w*(w+1)-u/2, u/2-w*(w-1)
    r = np.zeros((7, 7))
    for edges, value in [(((0, 2), (3, 5)), A1-kappa),
                         (((1, 2), (3, 4)), A2-kappa),
                         (((1, 3), (2, 4), (0, 3), (2, 5)), -kappa),
                         (((0, 6), (5, 6)), c.x[0]/c.x[6]*(A1+2-2*kappa)),
                         (((1, 6), (4, 6)), c.x[0]/c.x[6]*(A2-2-2*kappa))]:
        for a, b in edges:
            r[a, b] = r[b, a] = value
    return r


def split_candidate(c, index, ratio, separation, *, min_mass=1e-5):
    if not 0 <= index < c.m or not 0 < ratio < 1 or not np.isfinite(separation):
        raise ValueError("Invalid split")
    x = np.r_[c.x, c.x[index] * (1-ratio)]
    x[index] *= ratio
    if x.min() < min_mass:
        raise ValueError("Split violates mass floor")
    s = np.r_[c.s, c.s[index] + ratio * separation]
    s[index] -= (1-ratio) * separation
    order = np.r_[np.arange(c.m), index]
    return LimitingCandidate(x, normalize_offsets(x, s), c.groups[order],
                             c.p0[np.ix_(order, order)], c.allowed_mask[np.ix_(order, order)],
                             c.backbone_cycle, c.family_id + f"/split:{index}")


def winding(c, P, h, *, min_density=1e-4, tolerance=1e-10):
    """Check this backbone only, on the actual matrix and finite phase path."""
    phi = c.groups * (np.pi/2) + h*c.s
    cycle = c.backbone_cycle
    for a, b in zip(cycle, cycle[1:] + cycle[:1]):
        if not c.allowed_mask[a, b] or P[a, b] < min_density:
            return False
    from .topology import cycle_audit
    audit=cycle_audit(c.groups,cycle,phi=phi,P=P,tolerance=tolerance)
    # Also keep each nonzero parent step on its intended signed branch.
    from .topology import parent_step
    signed=all((parent_step(c.groups[a],c.groups[b])==0 or
                parent_step(c.groups[a],c.groups[b])*v>tolerance)
               for a,b,v in zip(cycle,cycle[1:]+cycle[:1],audit.get("wrapped_steps",[])))
    return bool(audit["valid"] and signed)
