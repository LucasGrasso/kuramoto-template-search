"""Exact rational repairs and Arb interval certificates, separate from search.

The original floating-point snapshots are never relabeled as certified. These
functions construct and verify explicitly recorded nearby rational data.
Requires the optional ``certification`` extra (python-flint / Arb).
"""

import math

from flint import arb, arb_mat, ctx, fmpq, fmpq_mat


class CertificationFailure(ValueError):
    pass


def rational(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CertificationFailure("Nonfinite input")
        return fmpq(*value.as_integer_ratio())
    return fmpq(value)


def qmatrix(rows):
    return [[rational(value) for value in row] for row in rows]


def strings(value):
    if isinstance(value, fmpq):
        return str(value)
    if isinstance(value, list):
        return [strings(item) for item in value]
    return value


def enclosure(value):
    """Exact dyadic endpoints; decimal display is not used for proof decisions."""
    return dict(lower=str(value.lower().fmpq()), upper=str(value.upper().fmpq()),
                display=value.str(25))


def decimal_bounds(value, digits=20):
    # All decisions and this decimal floor/ceiling are exact rational operations.
    from fractions import Fraction
    fraction = Fraction(str(value))
    scale = 10 ** digits
    low = fraction.numerator * scale // fraction.denominator
    high = -((-fraction.numerator * scale) // fraction.denominator)

    def show(integer):
        sign = "-" if integer < 0 else ""
        integer = abs(integer)
        return f"{sign}{integer // scale}.{integer % scale:0{digits}d}"

    return dict(exact=str(value), lower=show(low), upper=show(high))


def interval_cholesky(matrix):
    """Prove positivity for every symmetric matrix in its lower-triangle balls.

    This is the interval Cholesky--Banachiewicz algorithm. Each squared pivot
    must have a strictly positive lower endpoint before taking its square root.
    No midpoint replacement, clipping, tolerance or eigenvalue heuristic is used.
    """
    n = len(matrix)
    if not n or any(len(row) != n for row in matrix):
        raise CertificationFailure("Expected a nonempty square matrix")
    lower = [[arb(0) for _ in range(n)] for _ in range(n)]
    pivots = []
    for i in range(n):
        pivot = arb(matrix[i][i]) - sum((lower[i][k] ** 2 for k in range(i)), arb(0))
        if not pivot.lower() > 0:
            raise CertificationFailure(f"Interval Cholesky pivot {i} is not strictly positive: {pivot}")
        pivots.append(enclosure(pivot))
        lower[i][i] = pivot.sqrt()
        for j in range(i + 1, n):
            lower[j][i] = (arb(matrix[j][i]) - sum(
                (lower[j][k] * lower[i][k] for k in range(i)), arb(0))) / lower[i][i]
    return dict(positive_definite=True, squared_pivots=pivots)


def validate_connections(x, p):
    m = len(x)
    if m < 2 or any(v <= 0 for v in x) or sum(x) != 1:
        raise CertificationFailure("Masses must be exact positive rationals summing to one")
    if len(p) != m or any(len(row) != m for row in p):
        raise CertificationFailure("Invalid density matrix shape")
    if any(p[i][i] != 1 for i in range(m)):
        raise CertificationFailure("Clique diagonal must be exactly one")
    if any(not 0 <= p[i][j] <= 1 or p[i][j] != p[j][i]
           for i in range(m) for j in range(m)):
        raise CertificationFailure("Density box or symmetry failed")


def validate_tree(tree, m):
    if len(tree) != m - 1:
        raise CertificationFailure("Repair tree must have m-1 edges")
    parents = list(range(m))

    def root(a):
        while parents[a] != a:
            a = parents[a]
        return a

    for a, b in tree:
        if not 0 <= a < m or not 0 <= b < m or root(a) == root(b):
            raise CertificationFailure("Repair edges are not a spanning tree")
        parents[root(a)] = root(b)


def density(x, p):
    return min(sum((p[i][j] * x[j] for j in range(len(x))), fmpq(0)) for i in range(len(x)))


def weighted_system(x, p, theta):
    """Ground the last phase. Jacobian of the weighted forces is the Hessian."""
    m = len(x)
    forces = [arb(0) for _ in x]
    hessian = [[arb(0) for _ in x] for _ in x]
    restoring = [arb(x[i] * p[i][i]) for i in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            delta = theta[i] - theta[j]
            cosine = delta.cos()
            w = arb(x[i] * x[j] * p[i][j])
            force, stiffness = w * delta.sin(), w * cosine
            forces[i] += force
            forces[j] -= force
            hessian[i][i] += stiffness
            hessian[j][j] += stiffness
            hessian[i][j] -= stiffness
            hessian[j][i] -= stiffness
            restoring[i] += arb(x[j] * p[i][j]) * cosine
            restoring[j] += arb(x[i] * p[i][j]) * cosine
    return (arb_mat([[v] for v in forces[:-1]]),
            arb_mat([row[:-1] for row in hessian[:-1]]), restoring)


def krawczyk_and_stability(x, p, centers, radius):
    n = len(x) - 1
    if len(centers) != n or radius <= 0:
        raise CertificationFailure("Invalid phase box")
    center = [arb(v) for v in centers] + [arb(0)]
    # Round only a numerical preconditioner to exact dyadics; its quality is
    # subsequently proved by the interval norm of I-C*J(X).
    f, jacobian, _ = weighted_system(x, p, center)
    inverse = jacobian.inv()
    c = arb_mat([[inverse[i, j].mid() for j in range(n)] for i in range(n)])
    box = [arb(v, radius) for v in centers] + [arb(0)]
    _, jbox, restoring = weighted_system(x, p, box)
    identity = arb_mat([[int(i == j) for j in range(n)] for i in range(n)])
    error = identity - c * jbox
    row_bounds = [sum((abs(error[i, j]).upper() for j in range(n)), arb(0)).upper().fmpq()
                  for i in range(n)]
    contraction = max(row_bounds)
    if contraction >= 1:
        raise CertificationFailure("Krawczyk derivative norm is not below one")
    displacement = arb_mat([[arb(0, radius)] for _ in range(n)])
    images = arb_mat([[v] for v in center[:-1]]) - c * f + error * displacement
    # Compare against the exact rational radius, not Arb's possibly enlarged
    # radius when constructing a box or rounding a high-precision center.
    image_bounds = [abs(images[i, 0] - arb(centers[i])).upper().fmpq() for i in range(n)]
    if not all(bound < radius for bound in image_bounds):
        raise CertificationFailure("Krawczyk image is not strictly inside its box")
    cholesky = interval_cholesky([[jbox[i, j] for j in range(n)] for i in range(n)])
    if not all(value.lower() > 0 for value in restoring):
        raise CertificationFailure("Within-cluster restoring strength is not positive on the box")
    separated = next(((i, j) for i in range(n + 1) for j in range(i + 1, n + 1)
                      if (box[i] - box[j]).cos().upper() < 0), None)
    if separated is None:
        raise CertificationFailure("No certified nonsynchronous phase separation")
    return dict(
        unique_equilibrium_modulo_rotation=True, strict_inclusion=True,
        derivative_infinity_norm_upper=str(contraction),
        krawczyk_image=[enclosure(images[i, 0]) for i in range(n)],
        image_distance_from_center_upper=strings(image_bounds),
        box_centers=strings(centers), box_radius=str(radius),
        preconditioner=[[str(c[i, j].fmpq()) for j in range(n)] for i in range(n)],
        interval_cholesky=cholesky, restoring=[enclosure(v) for v in restoring],
        nonsynchronous_pair=list(separated),
    )


def circle_coordinates(t):
    return ([2 * v / (1 + v * v) for v in t],
            [(1 - v * v) / (1 + v * v) for v in t])


def rational_finite_repair(saved, *, coordinate_bits=128, precision_bits=256):
    """Fix rational unit-circle phases, then solve six tree densities exactly."""
    with ctx.workprec(precision_bits):
        raw_x = [rational(v) for v in saved["candidate"]["x"]]
        total = sum(raw_x)
        x = [v / total for v in raw_x]
        p = qmatrix(saved["P"])
        m, n = len(x), len(x) - 1
        anchor = rational(saved["phi"][-1])
        scale = 2 ** coordinate_bits
        t = []
        for phase in saved["phi"]:
            approximate = ((arb(rational(phase) - anchor) / 2).tan() * scale).mid()
            t.append(fmpq(int(approximate.floor().fmpz()), scale))
        sine, cosine = circle_coordinates(t)
        sin_delta = [[sine[i] * cosine[j] - cosine[i] * sine[j] for j in range(m)] for i in range(m)]
        force = [sum((x[i] * x[j] * p[i][j] * sin_delta[i][j] for j in range(m)), fmpq(0))
                 for i in range(m)]
        tree = saved["tree"]
        validate_tree(tree, m)
        matrix = [[fmpq(0) for _ in tree] for _ in range(n)]
        for k, (a, b) in enumerate(tree):
            weight = x[a] * x[b] * sin_delta[a][b]
            if a < n:
                matrix[a][k] += weight
            if b < n:
                matrix[b][k] -= weight
        correction = fmpq_mat(matrix).solve(fmpq_mat([[-v] for v in force[:-1]]))
        for k, (a, b) in enumerate(tree):
            p[a][b] += correction[k, 0]
            p[b][a] = p[a][b]
        data = dict(x=strings(x), P=strings(p), tan_half_phi=strings(t), anchor=m - 1)
        changes = dict(
            exact_original_mass_sum=str(total),
            max_mass_change=str(max(abs(x[i] - raw_x[i]) for i in range(m))),
            max_density_change=str(max(abs(correction[k, 0]) for k in range(n))),
            max_unit_circle_coordinate_change_upper=str(max(
                abs(value - reference).upper().fmpq()
                for i in range(m)
                for value, reference in (
                    (arb(sine[i]), arb(rational(saved["phi"][i]) - anchor).sin()),
                    (arb(cosine[i]), arb(rational(saved["phi"][i]) - anchor).cos()),
                ))),
            repair_tree=tree, coordinate_bits=coordinate_bits,
        )
        return data, changes


def verify_finite(data, *, precision_bits=256, box=None):
    with ctx.workprec(precision_bits):
        x, p = [rational(v) for v in data["x"]], qmatrix(data["P"])
        t = [rational(v) for v in data["tan_half_phi"]]
        validate_connections(x, p)
        if len(t) != len(x) or data["anchor"] != len(x) - 1 or t[-1] != 0:
            raise CertificationFailure("Invalid finite phase gauge")
        sine, cosine = circle_coordinates(t)
        if not all(sine[i] ** 2 + cosine[i] ** 2 == 1 for i in range(len(x))):
            raise CertificationFailure("Unit-circle identities failed")
        forces = [sum((x[i] * x[j] * p[i][j] * (sine[i] * cosine[j] - cosine[i] * sine[j])
                       for j in range(len(x))), fmpq(0)) for i in range(len(x))]
        if any(value != 0 for value in forces):
            raise CertificationFailure("Exact rational equilibrium identities failed")
        if box is None:
            centers = [(2 * arb(value).atan()).mid().fmpq() for value in t[:-1]]
            radius = fmpq(1, 2 ** 140)
        else:
            centers = [rational(value) for value in box["box_centers"]]
            radius = rational(box["box_radius"])
        if len(centers) != len(t) - 1 or radius <= 0 or not all(
                abs(2 * arb(t[i]).atan() - arb(centers[i])).upper().fmpq() < radius
                for i in range(len(centers))):
            raise CertificationFailure("Phase box does not enclose the declared rational unit-circle phases")
        proof = krawczyk_and_stability(x, p, centers, radius)
        return dict(certified=True, scope="nearby exact rational finite template",
                    precision_bits=precision_bits, exact_equilibrium=True,
                    density=decimal_bounds(density(x, p)), proof=proof,
                    original_snapshot_certified=False, graph_realization_certified=False)


def limiting_constraints(x, s, groups, p, r):
    """Independent exact edge sums, not the projection's coefficient matrix."""
    m = len(x)
    f0, f1, cuts, k = [[fmpq(0) for _ in range(size)] for size in (m, m, 4, 4)]
    energy = fmpq(0)
    for a in range(m):
        for b in range(a + 1, m):
            mass = x[a] * x[b]
            if groups[a] == groups[b]:
                delta = s[b] - s[a]
                f1[a] += mass * p[a][b] * delta
                f1[b] -= mass * p[a][b] * delta
                energy += mass * p[a][b] * delta ** 2
            elif (groups[b] - groups[a]) % 2:
                aa, bb = (a, b) if (groups[b] - groups[a]) % 4 == 1 else (b, a)
                delta = s[bb] - s[aa]
                f0[aa] += mass * p[a][b]
                f0[bb] -= mass * p[a][b]
                f1[aa] += mass * r[a][b]
                f1[bb] -= mass * r[a][b]
                cuts[groups[aa]] += mass * p[a][b] * delta
                k[groups[aa]] -= mass * r[a][b] * delta
    return f0 + f1 + cuts + [sum(k) - energy]


def rational_limiting_repair(saved):
    c = saved["candidate"]
    raw_x = [rational(v) for v in c["x"]]
    total = sum(raw_x)
    x = [v / total for v in raw_x]
    s = [rational(v) for v in c["s"]]
    mean = sum(x[i] * s[i] for i in range(len(x)))
    s = [v - mean for v in s]
    g, p, r = c["groups"], qmatrix(c["p0"]), qmatrix(saved["r"])
    m = len(x)
    pp = [(a, b) for a in range(m) for b in range(a + 1, m) if c["allowed_mask"][a][b]]
    rr = [(a, b) for a, b in pp if (g[a] - g[b]) % 2]
    matrix = [[fmpq(0) for _ in pp + rr] for _ in range(2 * m + 4)]
    for k, (a, b) in enumerate(pp):
        mass = x[a] * x[b]
        if g[a] == g[b]:
            delta = s[b] - s[a]
            matrix[m + 4 + a][k] += mass * delta
            matrix[m + 4 + b][k] -= mass * delta
        else:
            aa, bb = (a, b) if (g[b] - g[a]) % 4 == 1 else (b, a)
            matrix[aa][k] += mass
            matrix[bb][k] -= mass
            matrix[m + g[aa]][k] += mass * (s[bb] - s[aa])
    for l, (a, b) in enumerate(rr):
        aa, bb = (a, b) if (g[b] - g[a]) % 4 == 1 else (b, a)
        mass, k = x[a] * x[b], len(pp) + l
        matrix[m + 4 + aa][k] += mass
        matrix[m + 4 + bb][k] -= mass
    a = fmpq_mat(matrix)
    reduced, rank = a.transpose().rref()
    rows = [next(j for j in range(len(matrix)) if reduced[i, j] != 0) for i in range(rank)]
    b = fmpq_mat([matrix[i] for i in rows])
    v = fmpq_mat([[p[a][b]] for a, b in pp] + [[r[a][b]] for a, b in rr])
    adjustment = b.transpose() * (b * b.transpose()).solve(b * v)
    corrected = v - adjustment
    residual = a * corrected
    if any(residual[i, 0] != 0 for i in range(len(matrix))):
        raise CertificationFailure("Exact projection did not cancel its constraints")
    for k, (a, b) in enumerate(pp):
        p[a][b] = p[b][a] = corrected[k, 0]
    for l, (a, b) in enumerate(rr):
        r[a][b] = r[b][a] = corrected[len(pp) + l, 0]
    data = dict(x=strings(x), s=strings(s), groups=g, P0=strings(p), R=strings(r),
                allowed_mask=c["allowed_mask"], repair_tree=saved["repair_edges"])
    changes = dict(exact_original_mass_sum=str(total), projection_rank=rank,
                   max_mass_change=str(max(abs(x[i] - raw_x[i]) for i in range(m))),
                   max_density_change=str(max(abs(adjustment[k, 0]) for k in range(len(pp)))),
                   max_R_change=str(max(abs(adjustment[k, 0]) for k in range(len(pp), len(pp) + len(rr)))))
    return data, changes


def limiting_hessian(x, s, g, p, r):
    """Exact quotient eta_3=0 and zero-sum integer relative bases per parent."""
    m = len(x)
    columns = []
    for j in range(4):
        ids = [i for i in range(m) if g[i] == j]
        for a in ids[:-1]:
            columns.append([fmpq(int(i == a) - int(i == ids[-1])) for i in range(m)])
    h = [[fmpq(0) for _ in range(m - 1)] for _ in range(m - 1)]
    for a in range(m):
        for b in range(a + 1, m):
            mass = x[a] * x[b]
            if g[a] == g[b]:
                z = [v[a] - v[b] for v in columns]
                for i in range(m - 4):
                    for j in range(m - 4):
                        h[i + 3][j + 3] += mass * p[a][b] * z[i] * z[j]
            elif (g[b] - g[a]) % 2:
                aa, bb = (a, b) if (g[b] - g[a]) % 4 == 1 else (b, a)
                delta = s[bb] - s[aa]
                z = [v[aa] - v[bb] for v in columns]
                u = [fmpq(int(g[aa] == j) - int(g[bb] == j)) for j in range(3)]
                for i in range(3):
                    for j in range(3):
                        h[i][j] -= mass * r[a][b] * delta * u[i] * u[j]
                    for j in range(m - 4):
                        value = mass * p[a][b] * delta * u[i] * z[j]
                        h[i][j + 3] -= value
                        h[j + 3][i] -= value
    return h


def verify_limiting(data, *, precision_bits=256):
    x, s = [rational(v) for v in data["x"]], [rational(v) for v in data["s"]]
    g, p, r = data["groups"], qmatrix(data["P0"]), qmatrix(data["R"])
    m = len(x)
    validate_connections(x, p)
    if len(s) != m or len(g) != m or set(g) != {0, 1, 2, 3}:
        raise CertificationFailure("Invalid four-parent geometry")
    if len(r) != m or any(len(row) != m for row in r):
        raise CertificationFailure("Invalid correction matrix")
    mask = data["allowed_mask"]
    if len(mask) != m or any(len(row) != m for row in mask):
        raise CertificationFailure("Invalid support shape")
    for i in range(m):
        for j in range(m):
            if r[i][j] != r[j][i] or mask[i][j] != mask[j][i]:
                raise CertificationFailure("Asymmetric correction or support")
            if (g[i] - g[j]) % 4 == 2 and mask[i][j]:
                raise CertificationFailure("Opposite-parent support is forbidden")
            if not mask[i][j] and (p[i][j] != 0 or r[i][j] != 0):
                raise CertificationFailure("Forbidden connection is nonzero")
            if (g[i] - g[j]) % 2 == 0 and r[i][j] != 0:
                raise CertificationFailure("Correction must be confined to neighbor cuts")
            if mask[i][j] and (g[i] - g[j]) % 2 and not 0 < p[i][j] < 1:
                raise CertificationFailure("Neighbor densities must be strictly inside their box for this lifting proof")
    if any(value != 0 for value in limiting_constraints(x, s, g, p, r)):
        raise CertificationFailure("Exact F0/F1/K/budget identities failed")
    tree = data["repair_tree"]
    validate_tree(tree, m)
    if any(not mask[a][b] or (g[a] - g[b]) % 2 != 1 for a, b in tree):
        raise CertificationFailure("Lifting needs an allowed neighbor-only repair tree")
    restoring = [sum((x[b] * p[a][b] for b in range(m) if g[a] == g[b]), fmpq(0)) for a in range(m)]
    if min(restoring) <= 0:
        raise CertificationFailure("Limiting restoring strength is not positive")
    with ctx.workprec(precision_bits):
        proof = interval_cholesky(limiting_hessian(x, s, g, p, r))
    return dict(certified=True, scope="nearby rational limiting data with a sufficiently-small-h lifting proof",
                precision_bits=precision_bits, exact_F0_F1_K_and_budget=True,
                interval_cholesky=proof, limiting_restoring=strings(restoring),
                density=decimal_bounds(density(x, p)),
                finite_family_for_sufficiently_small_h=True, explicit_finite_h_certified=False,
                original_snapshot_certified=False, graph_realization_certified=False)


def attempt_original_finite(saved, *, precision_bits=256, iterations=30):
    """Try phase-only Newton/Krawczyk at fixed original P; never claim exclusion."""
    with ctx.workprec(precision_bits):
        raw_x = [rational(v) for v in saved["candidate"]["x"]]
        x = [v / sum(raw_x) for v in raw_x]
        p = qmatrix(saved["P"])
        anchor = rational(saved["phi"][-1])
        theta = [arb(rational(v) - anchor) for v in saved["phi"]]
        start = theta[:]
        n = len(x) - 1
        try:
            for iteration in range(iterations):
                f, jacobian, _ = weighted_system(x, p, theta)
                step = jacobian.solve(f)
                theta = [(theta[i] - step[i, 0]).mid() for i in range(n)] + [arb(0)]
                if all(abs(step[i, 0]).upper() < arb((1, -180)) for i in range(n)):
                    break
            proof = krawczyk_and_stability(x, p, [v.fmpq() for v in theta[:-1]], fmpq(1, 2 ** 100))
            return dict(certified=True, scope="original fixed P with exactly normalized mass ratios",
                        iterations=iteration + 1, proof=proof)
        except (CertificationFailure, ZeroDivisionError, ValueError) as exc:
            change = max(abs(theta[i] - start[i]).upper().fmpq() for i in range(n))
            return dict(certified=False, status="unresolved", iterations=iteration + 1,
                        phase_change_upper=str(change), reason=str(exc),
                        interpretation="This failed attempt does not exclude other stable equilibria for the original P.")
