"""Pure deterministic mutations. Sampling parameters belongs to the caller."""

from operator import index as integer_index

import numpy as np

from .template import PhaseTemplate


def _index(template: PhaseTemplate, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("Cluster index must be an integer")
    value = integer_index(value)
    if not 0 <= value < template.n_clusters:
        raise IndexError("Cluster index out of range")
    return value


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def split_cluster(
    template: PhaseTemplate,
    index: int,
    ratio: float,
    separation: float,
) -> PhaseTemplate:
    """Replace index by two children, preserving their unwrapped mean phase.

    Children occupy index and index+1; other clusters retain their order.
    Separation is signed, in radians, and may be zero.
    """
    index = _index(template, index)
    ratio = _finite(ratio, "ratio")
    separation = _finite(separation, "separation")
    if not 0 < ratio < 1:
        raise ValueError("ratio must be strictly between zero and one")
    x = template.x
    phi = template.phi

    size = x[index]
    phase = phi[index]

    new_x = np.concatenate(
        [
            x[:index],
            [size * ratio, size * (1 - ratio)],
            x[index + 1 :],
        ]
    )

    new_phi = np.concatenate(
        [
            phi[:index],
            [
                phase - (1 - ratio) * separation,
                phase + ratio * separation,
            ],
            phi[index + 1 :],
        ]
    )

    return PhaseTemplate(
        x=new_x,
        phi=np.mod(new_phi, 2 * np.pi),
    )


def merge_clusters(template: PhaseTemplate, i: int, j: int) -> PhaseTemplate:
    """Merge at the smaller index, delete the larger, retain other order.

    Use the mass-weighted circular mean. Reject near cancellation when
    the resultant length / combined mass is <= 1e-12.
    """
    i, j = _index(template, i), _index(template, j)
    if i == j:
        raise ValueError("Merge indices must be distinct")
    if template.n_clusters <= 2:
        raise ValueError("Merge must leave at least two clusters")
    i, j = sorted((i, j))
    mass = template.x[i] + template.x[j]
    resultant = (template.x[i] * np.exp(1j * template.phi[i])
                 + template.x[j] * np.exp(1j * template.phi[j]))
    if abs(resultant) <= 1e-12 * mass:
        raise ValueError("Circular mean is undefined near antipodal cancellation")
    x, phi = template.x.copy(), template.phi.copy()
    x[i], phi[i] = mass, np.angle(resultant)
    return PhaseTemplate(np.delete(x, j), np.delete(phi, j))


def perturb_phase(template: PhaseTemplate, index: int, delta: float) -> PhaseTemplate:
    """Shift one phase by a finite displacement in radians."""
    index = _index(template, index)
    delta = _finite(delta, "delta")
    phi = template.phi.copy()
    phi[index] += delta
    return PhaseTemplate(template.x, phi)


def resize_clusters(
    template: PhaseTemplate, source: int, target: int, amount: float,
    *, min_mass: float = 0.0,
) -> PhaseTemplate:
    """Transfer positive mass, leaving source strictly above min_mass."""
    source, target = _index(template, source), _index(template, target)
    amount = _finite(amount, "amount")
    min_mass = _finite(min_mass, "min_mass")
    if source == target:
        raise ValueError("source and target must be distinct")
    if min_mass < 0:
        raise ValueError("min_mass must be nonnegative")
    if amount <= 0 or template.x[source] - amount <= min_mass:
        raise ValueError("amount must be positive and leave source above min_mass")
    x = template.x.copy()
    x[source] -= amount
    x[target] += amount
    if x[source] == template.x[source] or x[target] == template.x[target]:
        raise ValueError("amount is too small to change both masses in float64")
    return PhaseTemplate(x, template.phi)
