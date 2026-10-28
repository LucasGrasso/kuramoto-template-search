# Exact-data and interval witness certification

The procedure follows the separation in [O'Keeffe, section V.1](https://arxiv.org/html/2610.21271):
first establish an equilibrium, then prove stability over its enclosure.
The optional python-flint backend provides exact rational arithmetic and
[Arb real-ball arithmetic](https://python-flint.readthedocs.io/en/stable/arb.html).
Every interval proof decision uses exact outward endpoint bounds. Floating-point
tolerances, midpoint eigenvalues and solver statuses are not proof criteria.

The [certificate report](../research/certification/README.md) lists results and
reproduction commands. Certificates are for the nearby rational data stored
inside them, not for the unmodified binary64 source snapshots.

## Finite template: exact equilibrium and Krawczyk

Interpret each saved binary64 mass and density as its exact binary rational.
Normalize masses by their exact sum. Fix the last phase to zero. Approximate
each tangent of a half phase by a rational t_i, and set

    sin(theta_i) = 2 t_i / (1+t_i^2),
    cos(theta_i) = (1-t_i^2) / (1+t_i^2).

These are exact unit-circle points, with theta_i = 2 arctan(t_i). Thus all
pairwise sines/cosines are rational. The weighted forces

    f_i = x_i sum_j x_j P_ij sin(theta_i-theta_j)

are rational and sum to zero by symmetry. Adjust P on a spanning tree to set
the first m-1 forces exactly to zero. This is a rational linear system. The
last force is then exactly zero. The verifier separately checks all m force
identities, mass normalization, positivity, symmetry, cliques and box bounds.

After grounding the last phase, Df is exactly the grounded mass-weighted
Laplacian H. Take an exact dyadic center z and rational radius r for a box X.
Let C be an exact dyadic approximate inverse of Df(z). Compute with outward balls

    K(X) = z - C f(z) + (I - C Df(X)) (X-z).

The checker requires K(X) strictly inside X and
||I-C Df(X)||_infinity < 1. The latter also ensures C is nonsingular: for every
represented Jacobian, C Df is invertible by the Neumann-series criterion.
The map T(y)=y-Cf(y) is a contraction mapping of X into itself, so it has a
unique fixed point in X; C's nonsingularity makes this an equilibrium.
The reported box also encloses the declared exact unit-circle phases.

Interval Cholesky--Banachiewicz is then applied to H(X). Each squared pivot
must have strictly positive lower endpoint before its interval square root is
taken. Successful completion proves positive definiteness for every symmetric
H represented in the box. This is the same sufficient condition documented
for [FLINT's interval Cholesky](https://flintlib.org/doc/arb_mat.html).
The checker additionally encloses every restoring coefficient c_i and requires
positive lower endpoints. A pairwise cosine with negative upper endpoint
excludes synchrony. Together these prove a strict nonsynchronous template.

## Limiting data: exact projection and a lifting lemma

For fixed rational x, s, parent labels and support, F0, F1 and K are linear in
the off-diagonal entries of P0 and neighbor correction R. Select independent
rows B using exact rational row reduction. For the saved vector v, use

    v_corrected = v - B^T (B B^T)^(-1) B v.

All arithmetic is rational. Independent edge sums then check F0=F1=K=0 and
sum(k)=I exactly. The source snapshots' small residuals are not accepted as
zeros. Diagonal ones, forbidden zeros and within-parent R=0 are preserved.

For the Hessian, fix eta_3=0 and use columns e_a-e_ref within each parent as
the relative basis Z. These integer bases represent the quotient by global
rotation exactly; no floating orthonormality assumption enters the proof.
Assemble the full limiting quadratic form by exact rational edge sums, then
prove it positive definite using interval Cholesky.

**Lifting lemma.** Suppose these exact identities hold, the limiting quotient
form is positive definite, the limiting restoring coefficients are positive,
all allowed neighbor densities are strictly inside (0,1), and an allowed
neighbor spanning tree exists. Then strict finite templates exist for all
sufficiently small positive h and their minimum degrees tend to min(P0 x).

**Proof.** Set phi_a(h)=j(a) pi/2+h s_a and initially P(h)=P0+hR. The exact
F0=F1=0 identities make its force residual O(h^2). Use the m-1 tree densities
as correction variables. Their weighted-force coefficient matrix is an
incidence matrix whose columns are multiplied by
x_a x_b sin(phi_a(h)-phi_b(h)). At h=0 these factors are nonzero, because
the endpoints are neighboring parents. Ground one row. The resulting square
matrix is invertible at zero and remains analytically invertible nearby.
Consequently there is an exact analytic tree correction Delta P(h)=O(h^2)
that cancels all forces; weighted antisymmetry supplies the last equation.

Tree corrections affect neighbor edges only. On those edges cos(delta phi)
is O(h), so Delta P changes the unscaled Hessian by O(h^3). For displacements
y=C_parent eta+h Z z, K=0 cancels the first-order collective term and the
rescaled quadratic form converges to the verified positive limiting form.
Indeed, in these coordinates T_h^T L(h) T_h / h^2 tends to H_lim; T_h spans
a complement to the global rotation for every h>0. Positivity persists for
all sufficiently small positive h. Restoring positivity persists by continuity.
Interior neighbor densities leave room for hR and Delta P; other entries are
unchanged. Positive masses are fixed and the four separated parents exclude
synchrony for small h. Finally P(h) -> P0, giving the claimed density limit.
This proves the lemma.

The checker verifies every hypothesis of this lemma for both rationalized
limiting witnesses. It does not compute a uniform explicit upper bound on h,
certify an explicit finite member of the improved limiting family, establish
global optimality, or construct a finite simple graph. The certificate scope
records these distinctions.

## Verification and regression coverage

The verifier rereads exact inputs and recomputes the proof rather than trusting
stored success flags. Tests reject indefinite and unresolved interval matrices,
tampered finite equilibrium data, and broken leading identities. A separate
dense Taylor construction checks the rational quotient Hessian assembly.
All certificates pass at 192, 256 and 320 bits. Original numerical witness
flags remain false; the new certificate results are separate typed artifacts.
