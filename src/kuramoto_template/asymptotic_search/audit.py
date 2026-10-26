"""Independent Decimal arithmetic audit of a saved finite numerical candidate.

This recomputes forces and a grounded Hessian LDL factorization at increased
precision. It is neither interval arithmetic nor an equilibrium existence proof.
"""

from decimal import Decimal, localcontext
from time import perf_counter

import numpy as np

PI = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"


def high_precision_audit(x, phi, P, *, digits=60):
    if isinstance(digits,bool) or not isinstance(digits,int) or not 30<=digits<=80:
        raise ValueError("digits must be an integer in 30..80")
    x,phi,P = np.asarray(x),np.asarray(phi),np.asarray(P)
    m = len(x)
    if x.shape!=(m,) or phi.shape!=(m,) or P.shape!=(m,m) or not all(np.all(np.isfinite(v)) for v in (x,phi,P)):
        raise ValueError("Invalid audit arrays")
    started = perf_counter()
    with localcontext() as context:
        context.prec = digits+10
        pi,zero,one = Decimal(PI),Decimal(0),Decimal(1)
        threshold = Decimal(10)**(-digits-5)

        def dec(value):
            return Decimal.from_float(float(value))

        def sincos(theta):
            while theta>pi:
                theta -= 2*pi
            while theta < -pi:
                theta += 2*pi
            square = theta*theta
            sine,cosine,sin_term,cos_term = theta,one,theta,one
            for k in range(1,300):
                sin_term *= -square/Decimal((2*k)*(2*k+1))
                cos_term *= -square/Decimal((2*k-1)*(2*k))
                sine += sin_term
                cosine += cos_term
                if max(abs(sin_term),abs(cos_term))<threshold:
                    return sine,cosine
            raise ArithmeticError("Taylor series did not converge")

        masses,phases = list(map(dec,x)),list(map(dec,phi))
        density = [list(map(dec,row)) for row in P]
        force = [zero]*m
        restoring = [masses[a]*density[a][a] for a in range(m)]
        L = [[zero for _ in range(m)] for _ in range(m)]
        degrees = [sum((density[a][b]*masses[b] for b in range(m)),zero) for a in range(m)]
        for a in range(m):
            for b in range(a+1,m):
                sine,cosine = sincos(phases[b]-phases[a])
                force[a] += masses[b]*density[a][b]*sine
                force[b] -= masses[a]*density[a][b]*sine
                restoring[a] += masses[b]*density[a][b]*cosine
                restoring[b] += masses[a]*density[a][b]*cosine
                w = masses[a]*masses[b]*density[a][b]*cosine
                L[a][b] = L[b][a] = -w
                L[a][a] += w
                L[b][b] += w
        # Ground vertex m-1; positive pivots imply numerical positive definiteness
        # off rotations. Perform an independent scalar LDL, not NumPy eigenanalysis.
        lower = [[zero for _ in range(m-1)] for _ in range(m-1)]
        pivots = []
        for i in range(m-1):
            pivot = L[i][i]-sum((lower[i][k]**2*pivots[k] for k in range(i)),zero)
            pivots.append(pivot)
            if pivot == 0:
                break
            lower[i][i] = one
            for j in range(i+1,m-1):
                lower[j][i] = (L[j][i]-sum((lower[j][k]*lower[i][k]*pivots[k] for k in range(i)),zero))/pivot
        stringify = lambda value: format(value,f".{digits-5}g")
        return dict(precision_digits=digits,max_force_error=stringify(max(map(abs,force))),
                    min_restoring=stringify(min(restoring)),mu=stringify(min(degrees)),
                    grounded_ldl_pivots=[stringify(v) for v in pivots],
                    positive_grounded_pivots=len(pivots)==m-1 and min(pivots)>0,
                    box_valid=min(v for row in density for v in row)>=0 and max(v for row in density for v in row)<=1,
                    symmetry_valid=all(density[a][b]==density[b][a] for a in range(m) for b in range(m)),
                    mass_sum_error=stringify(abs(sum(masses)-one)),
                    proof_available=False,elapsed_seconds=perf_counter()-started)
