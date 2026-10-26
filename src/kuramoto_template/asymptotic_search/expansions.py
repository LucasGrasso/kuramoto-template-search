"""Analytic coefficients, with clockwise inter-parent orientation.

Here F uses sin(phi_b-phi_a); the project's template_metrics uses its negative.
The equilibrium residual norm and the weighted Hessian normalization coincide.
No finite differences or global pseudoinverses are used to form coefficients.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space

from .models import LimitingCandidate

U4 = null_space(np.ones((1, 4)))
DIPOLES = np.array([[1., 0., -1., 0.], [0., 1., 0., -1.]])


def edges(c):
    within, cuts = [], []
    for a in range(c.m):
        for b in range(a + 1, c.m):
            if c.groups[a] == c.groups[b]:
                within.append((a, b))
            elif (c.groups[b] - c.groups[a]) % 4 == 1:
                cuts.append((a, b, int(c.groups[a])))
            elif (c.groups[a] - c.groups[b]) % 4 == 1:
                cuts.append((b, a, int(c.groups[b])))
    return within, cuts


def internal_basis(c):
    columns = []
    for j in range(4):
        idx = np.flatnonzero(c.groups == j)
        local = null_space(np.ones((1, len(idx))))
        block = np.zeros((c.m, local.shape[1]))
        block[idx] = local
        columns.append(block)
    return np.column_stack(columns)


def group_indicator(c):
    return np.eye(4)[c.groups]


def cut_matrix(k):
    result = np.zeros((4, 4))
    for j in range(4):
        d = np.eye(4)[j] - np.eye(4)[(j + 1) % 4]
        result += k[j] * np.outer(d, d)
    return result


@dataclass
class Expansion:
    f0: np.ndarray
    f1: np.ndarray
    cuts: np.ndarray
    energy: float
    lin: np.ndarray
    g: np.ndarray
    k: np.ndarray


def coefficients(c: LimitingCandidate, r=None) -> Expansion:
    r = np.zeros_like(c.p0) if r is None else np.asarray(r, dtype=float)
    if r.shape != c.p0.shape or not np.all(np.isfinite(r)):
        raise ValueError("r must be finite and match p0")
    f0, f1 = np.zeros(c.m), np.zeros(c.m)
    K, k = np.zeros(4), np.zeros(4)
    L, G, energy = np.zeros_like(c.p0), np.zeros((c.m, 4)), 0.0
    within, cuts = edges(c)
    eye = np.eye(c.m)
    for a, b in within:
        ds, p = c.s[b] - c.s[a], c.p0[a, b]
        v = eye[a] - eye[b]
        w = c.x[a] * c.x[b] * p
        L += w * np.outer(v, v)
        energy += w * ds**2
        f1[a] += c.x[b] * p * ds
        f1[b] -= c.x[a] * p * ds
    for a, b, j in cuts:
        ds = c.s[b] - c.s[a]
        w = c.x[a] * c.x[b]
        p, correction = c.p0[a, b], r[a, b]
        f0[a] += c.x[b] * p
        f0[b] -= c.x[a] * p
        f1[a] += c.x[b] * correction
        f1[b] -= c.x[a] * correction
        K[j] += w * p * ds
        k[j] -= w * correction * ds
        d = np.eye(4)[j] - np.eye(4)[(j + 1) % 4]
        G += w * p * ds * np.outer(eye[a] - eye[b], d)
    return Expansion(f0, f1, K, float(energy), L, G, k)


def internal_solve(c, L, rhs, rank_tolerance=1e-11):
    """Independent zero-mean solves; reject extra internal null modes."""
    rhs = np.asarray(rhs)
    answer = np.zeros_like(rhs)
    gaps = []
    for j in range(4):
        idx = np.flatnonzero(c.groups == j)
        if len(idx) == 1:
            continue
        B = null_space(np.ones((1, len(idx))))
        block = B.T @ L[np.ix_(idx, idx)] @ B
        gap = float(np.linalg.eigvalsh(block).min())
        gaps.append(gap)
        if gap <= rank_tolerance:
            raise ValueError("Extra internal null mode or unresolved hard gap")
        answer[idx] = B @ np.linalg.solve(block, B.T @ rhs[idx])
    gap = min(gaps, default=float("inf"))  # Four singletons: no hard block.
    residual = float(np.max(np.abs(L @ answer - rhs), initial=0))
    return answer, gap, residual


def effective_matrix(c, r, rank_tolerance=1e-11):
    e = coefficients(c, r)
    z, _, _ = internal_solve(c, e.lin, e.g, rank_tolerance)
    Q = cut_matrix(e.k) - e.g.T @ z
    return (Q + Q.T) / 2


def full_rescaled_matrix(c, r):
    """Complete m-1 form in coordinates y=C U4 eta + h H z."""
    e = coefficients(c, r)
    H = internal_basis(c)
    cross = -U4.T @ e.g.T @ H
    return np.block([[U4.T @ cut_matrix(e.k) @ U4, cross],
                     [cross.T, H.T @ e.lin @ H]])


def reference_rescaled_matrix(c, r):
    """Independent dense Taylor route: L0,L1,L2 then coordinate rescaling."""
    same = c.groups[:, None] == c.groups[None, :]
    delta = (c.groups[None, :] - c.groups[:, None]) % 4
    sin0 = np.where(delta == 1, 1., np.where(delta == 3, -1., 0.))
    ds = c.s[None, :] - c.s[:, None]
    mass = np.outer(c.x, c.x)

    def lap(w):
        w = w.copy()
        np.fill_diagonal(w, 0)
        return np.diag(w.sum(axis=1)) - w

    L0 = lap(mass * c.p0 * same)
    L1 = lap(-mass * c.p0 * sin0 * ds)
    L2 = lap(mass * (-r * sin0 * ds - 0.5 * c.p0 * same * ds**2))
    S, H = group_indicator(c) @ U4, internal_basis(c)
    return np.block([[S.T @ L2 @ S, S.T @ L1 @ H],
                     [H.T @ L1 @ S, H.T @ L0 @ H]])


def finite_rescaled_matrix(c, P, h):
    """Exact trig route using parent labels to avoid cos(pi/2) cancellation."""
    ds = h * (c.s[None, :] - c.s[:, None])
    delta = (c.groups[None, :] - c.groups[:, None]) % 4
    cosine = np.choose(delta, [np.cos(ds), -np.sin(ds), -np.cos(ds), np.sin(ds)])
    S, H = group_indicator(c) @ U4, internal_basis(c)
    T = np.column_stack((S, h * H))
    out = np.zeros((c.m - 1, c.m - 1))
    for a in range(c.m):
        for b in range(a + 1, c.m):
            v = (T[a] - T[b]) / h
            out += c.x[a] * c.x[b] * P[a, b] * cosine[a, b] * np.outer(v, v)
    return (out + out.T) / 2
