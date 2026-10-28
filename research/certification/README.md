# Interval certification of the three best results

These certificates concern **explicitly recorded nearby rational templates**.
The original floating-point snapshots are unchanged and retain their numerical
status. All three companion files record the original file's SHA-256, exact
rational input data, the changes made, and outward interval proof bounds.

| Companion | Certified density or limiting density, rounded down |
| --- | ---: |
| [Original best limit](original-limit.json) | 0.69290331891486345929 |
| [Improved S7 limit](best-limit.json) | 0.69290726374913245157 |
| [Best finite template](best-finite.json) | 0.69265919635030838311 |

## Finite template

The saved phase residual alone was insufficient: at the original fixed P,
phase-only Newton converged to a root that failed the interval Cholesky check.
This attempt is recorded as unresolved; it does not exclude another stable
equilibrium for that original P.

For the certified variant, the unit-circle phase coordinates are exact
rationals and six tree-edge densities are solved for using exact rational
linear algebra. The largest density change is less than 6.024e-16. The sine
and cosine coordinates change by less than 2.703e-39 after fixing the rotation.
The mass ratios are retained and normalized exactly (the saved sum is already
exactly one in this case). Box constraints and symmetry hold exactly.

Exact rational identities establish equilibrium. Independently, at 256 bits,
the Krawczyk operator maps the six-dimensional phase box strictly inside itself.
Its radius is 2^-140 radians, approximately 7.175e-43, and the derivative
infinity-norm bound is less than 7.192e-30. This proves uniqueness within the
box after fixing the last phase. Interval Cholesky proves the grounded weighted
Hessian positive definite throughout the box. Its final squared pivot is
between 1.46629252831307e-13 and 1.46629252831554e-13. Every within-cluster
restoring coefficient is positive, and a negative pairwise cosine proves
nonsynchrony throughout the box.

## Limiting templates

The original leading identities had small nonzero exact residuals. Exact
rational projection corrects P0 and R while preserving the geometry up to
normalization and global translation. For the original best limit, the maximum
P0 change is less than 6.327e-16; for the improved limit it is less than
2.356e-15. F0=F1=K=0 and sum(k)=I then hold **exactly**.

Arb interval Cholesky proves positivity of the full limiting Hessian in an
exact rational quotient basis. Together with positive restoring strengths,
interior neighbor densities and an allowed neighbor spanning tree, these are
the hypotheses of the [small-splitting lifting proof](../../docs/interval-certification.md).
They imply a family of strict finite templates for sufficiently small positive
h, with densities tending to the certified rational limiting value. No explicit
h or finite witness for the improved limit is enclosed by these certificates.

## Recheck from exact data

```powershell
uv sync --locked --extra certification
uv run --locked --extra certification python scripts/certify_witnesses.py --verify research/certification
```

The checker recomputes the exact identities, interval enclosures and positivity;
it does not trust saved success flags. The certificates were also independently
recomputed at 192 and 320 bits. The complete retained suite passes 81 tests with
the certification extra. No SDP is run during certification.

To reconstruct the rational repairs and certificates from the three original
snapshots, use a fresh output directory:

```powershell
uv run --locked --extra certification python scripts/certify_witnesses.py --output runs/new-certificates
```

The arithmetic backend is python-flint 0.9.0 / FLINT-Arb. The trusted components
are exact integer/rational arithmetic, Arb's rigorous outward enclosures, and
the displayed mathematical arguments. No explicit simple graph is constructed
or certified here. See the [method and proof](../../docs/interval-certification.md).
