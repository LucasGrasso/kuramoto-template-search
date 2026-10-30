"""Exact refinement of a two-level pattern learned from a C4-only search.

No S7 constructor, constants, witness or dipole formula is used. The observed
connection pattern defines the ansatz; exact edge sums derive its equations.
The floating critical-point iteration is not itself a proof. Its dyadic output
is moved strictly inward and checked by the independent limiting certifier.
"""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from flint import fmpq,fmpq_mat
import numpy as np

from kuramoto_template.asymptotic_search.interval_certification import (
    decimal_bounds,density,limiting_constraints,limiting_hessian,rational,strings,verify_limiting,
)
from kuramoto_template.asymptotic_search.joint_limit import result_from_data
from kuramoto_template.asymptotic_search.joint_outer import repair_tree,_write_with_retry as write
from kuramoto_template.asymptotic_search.persistence import _code_hash


def dot(a,b):
    return sum((x*y for x,y in zip(a,b)),fmpq(0))


def evaluate(coefficients,q,w):
    return sum((coefficients[3*i+j]*q**i*w**j for i in range(3) for j in range(3)),fmpq(0))


def derivative(coefficients,q,w,variable):
    return sum((coefficients[3*i+j]*(i if variable==0 else j)*
                q**(i-1 if variable==0 else i)*w**(j-1 if variable==1 else j)
                for i in range(3) for j in range(3) if (i if variable==0 else j)>0),fmpq(0))


def dyadic(value,bits):
    v=Fraction(str(value));scale=2**bits
    return fmpq((v.numerator*scale*2+v.denominator)//(2*v.denominator),scale)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--root-bits',type=int,default=512)
    parser.add_argument('--inward-bits',type=int,default=128)
    parser.add_argument('--precision-bits',type=int,default=384)
    args=parser.parse_args()
    if args.root_bits<128 or not 16<=args.inward_bits<args.root_bits//2:
        parser.error('Need high root precision and a resolved inward displacement')
    raw=json.loads(args.input.read_text())
    saved=result_from_data(raw.get('result',raw));c=saved.candidate
    if c is None or not c.family_id.startswith('c4/'):
        parser.error('Only C4-derived source candidates are accepted')
    groups=list(map(int,c.groups));m=c.m
    occupancy=[groups.count(j) for j in range(4)]
    if sorted(occupancy)!=[1,2,2,2]:
        parser.error('The learned active-family fit requires three pairs and one singleton')
    args.output.mkdir(parents=True,exist_ok=False)
    anchor=next(i for i in range(m) if occupancy[groups[i]]==1)
    pairs={j:sorted([i for i,g in enumerate(groups) if g==j],key=lambda i:c.s[i])
           for j in range(4) if occupancy[j]==2}
    widths={j:float((c.s[ids[1]]-c.s[ids[0]])/2) for j,ids in pairs.items()}
    wide=max(widths,key=widths.get)
    narrow=[j for j in pairs if j!=wide]
    if abs(widths[narrow[0]]-widths[narrow[1]])>1e-4:
        parser.error('Source does not have the observed two-width pattern')
    neighbor=[(a,b) for a in range(m) for b in range(a+1,m) if c.allowed_mask[a,b] and (groups[a]-groups[b])%2]
    ordered=sorted(c.p0[a,b] for a,b in neighbor)
    split=int(np.argmax(np.diff(ordered)))
    threshold=(ordered[split]+ordered[split+1])/2
    variable_edges=[(a,b) for a,b in neighbor if c.p0[a,b]<threshold]
    ones=[c.p0[a,b] for a,b in neighbor if (a,b) not in variable_edges]
    interior=[c.p0[a,b] for a,b in variable_edges]
    if not interior or max(ones)-min(ones)>1e-5 or abs(np.mean(ones)-1)>1e-5 or max(interior)-min(interior)>1e-5:
        parser.error('Connection pattern is not resolved into one shared interior level and ones')

    def connections(q,alpha=fmpq(1)):
        p=[[fmpq(0) for _ in range(m)] for _ in range(m)]
        for a in range(m):p[a][a]=fmpq(1)
        for a in range(m):
            for b in range(a+1,m):
                if groups[a]==groups[b]:value=fmpq(1)
                elif (a,b) in neighbor:value=alpha*(q if (a,b) in variable_edges else fmpq(1))
                else:value=fmpq(0)
                p[a][b]=p[b][a]=value
        return p

    def geometry(q,w,normalized=True):
        p=connections(q)
        def force(r):
            x=[fmpq(1) for _ in range(m)];x[anchor]=r
            f=[fmpq(0) for _ in range(m)]
            for a,b in neighbor:
                aa,bb=(a,b) if (groups[b]-groups[a])%4==1 else (b,a)
                f[aa]+=x[bb]*p[a][b];f[bb]-=x[aa]*p[a][b]
            return f
        f0,f1=force(fmpq(0)),force(fmpq(1))
        row=next(i for i in range(m) if f1[i]!=f0[i])
        ratio=-f0[row]/(f1[row]-f0[row])
        if any(v!=0 for v in force(ratio)):
            raise ValueError('The observed density ties do not admit exact leading equilibrium')
        x=[fmpq(1) for _ in range(m)];x[anchor]=ratio
        if normalized:
            total=sum(x);x=[a/total for a in x]
        s=[fmpq(0) for _ in range(m)]
        for j,ids in pairs.items():
            width=w if j==wide else fmpq(1)
            s[ids[0]],s[ids[1]]=-width,width
        return x,s,ratio

    # Fixed parent dipoles and the remaining alternating collective mode.
    modes=[[fmpq(v) for v in row] for row in ((1,0,-1,0),(0,1,0,-1),(1,-1,1,-1))]
    gauge=[[v[j]-v[3] for j in range(3)] for v in modes]
    zero=[[fmpq(0) for _ in range(m)] for _ in range(m)]

    def penalty(x,s,p):
        h=limiting_hessian(x,s,groups,p,zero)
        cross=fmpq_mat([row[3:] for row in h[:3]])
        hard=fmpq_mat([row[3:] for row in h[3:]])
        schur=cross*hard.solve(cross.transpose())
        e=-limiting_constraints(x,s,groups,p,zero)[-1]
        mode_penalty=[[dot(gauge[i],[sum((schur[a,b]*gauge[j][b] for b in range(3)),fmpq(0))
                                      for a in range(3)]) for j in range(3)] for i in range(3)]
        return e,mode_penalty

    def slacks(q,w):
        x,s,_=geometry(q,w,False)
        e,b=penalty(x,s,connections(q))
        return [e-b[0][0],e-b[1][1],4*e-b[2][2]]

    grid=[(fmpq(q),fmpq(w)) for q in range(3) for w in range(3)]
    interpolation=fmpq_mat([[q**i*w**j for i in range(3) for j in range(3)] for q,w in grid])
    values=[slacks(q,w) for q,w in grid]
    polys=[]
    for k in range(3):
        solved=interpolation.solve(fmpq_mat([[v[k]] for v in values]))
        polys.append([solved[i,0] for i in range(9)])
    for q,w in ((fmpq(2,3),fmpq(7,5)),(fmpq(4,5),fmpq(11,7)),(fmpq(1,3),fmpq(8,5))):
        if slacks(q,w)!=[evaluate(poly,q,w) for poly in polys]:
            raise ValueError('Exact interpolation did not reproduce independent edge sums')
    q=rational(float(np.mean(interior)))
    w=rational(widths[wide]/np.mean([widths[j] for j in narrow]))
    trace=[]
    for iteration in range(20):
        value=[evaluate(poly,q,w) for poly in polys[:2]]
        jac=fmpq_mat([[derivative(poly,q,w,k) for k in (0,1)] for poly in polys[:2]])
        correction=jac.solve(fmpq_mat([[v] for v in value]))
        next_q=dyadic(q-correction[0,0],args.root_bits)
        next_w=dyadic(w-correction[1,0],args.root_bits)
        trace.append(dict(iteration=iteration,q=str(next_q),w=str(next_w),residual=str(max(map(abs,value)))))
        if next_q==q and next_w==w:break
        q,w=next_q,next_w
    alpha=1-fmpq(1,2**args.inward_bits)
    x,s,ratio=geometry(q,w)
    p=connections(q,alpha)
    energy,b=penalty(x,s,p)
    actual_slacks=[energy-b[0][0],energy-b[1][1],4*energy-b[2][2]]
    if min(actual_slacks)<=0:
        raise ValueError('The exactly represented inward point does not have positive dipole slacks')
    coefficients=[[fmpq(1)]*4]
    for i,j in ((0,1),(0,2),(1,2)):
        coefficients.append([(modes[i][k]-modes[i][(k+1)%4])*(modes[j][k]-modes[j][(k+1)%4])
                             for k in range(4)])
    k=fmpq_mat(coefficients).solve(fmpq_mat([[energy],[b[0][1]],[b[0][2]],[b[1][2]]]))
    nr=len(neighbor)
    matrix=[[fmpq(0) for _ in range(nr)] for _ in range(m+4)]
    rhs=[-v for v in limiting_constraints(x,s,groups,p,zero)[m:2*m]]+[k[j,0] for j in range(4)]
    for col,(a,bb) in enumerate(neighbor):
        aa,b=(a,bb) if (groups[bb]-groups[a])%4==1 else (bb,a)
        mass=x[aa]*x[b]
        matrix[aa][col]+=mass;matrix[b][col]-=mass
        matrix[m+groups[aa]][col]=-mass*(s[b]-s[aa])
    A=fmpq_mat(matrix)
    reduced,rank=A.transpose().rref()
    rows=[next(j for j in range(m+4) if reduced[i,j]!=0) for i in range(rank)]
    independent=fmpq_mat([matrix[j] for j in rows])
    selected_rhs=fmpq_mat([[rhs[j]] for j in rows])
    rv=independent.transpose()*(independent*independent.transpose()).solve(selected_rhs)
    residual=A*rv-fmpq_mat([[v] for v in rhs])
    if any(residual[j,0]!=0 for j in range(m+4)):
        raise ValueError('Exact correction recovery failed')
    r=[[fmpq(0) for _ in range(m)] for _ in range(m)]
    for j,(a,b) in enumerate(neighbor):r[a][b]=r[b][a]=rv[j,0]
    data=dict(x=strings(x),s=strings(s),groups=groups,P0=strings(p),R=strings(r),
              allowed_mask=c.allowed_mask.tolist(),repair_tree=repair_tree(c))
    construction=dict(kind='C4-derived exact active-family construction',s7_data_used=False,
        input_file=str(args.input),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        source_hash=_code_hash(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        occupancy=occupancy,anchor=anchor,wide_parent=wide,learned_variable_edges=variable_edges,
        learned_threshold=float(threshold),polynomial_coefficients=strings(polys),coefficient_order='q^i*w^j, index 3*i+j',
        interpolation_checks_exact=True,newton_trace=trace,root_bits=args.root_bits,
        q=str(q),w=str(w),singleton_mass_ratio=str(ratio),inward_factor=str(alpha),
        inward_bits=args.inward_bits,positive_slacks=strings(actual_slacks),recovery_rank=rank,
        counters=dict(conic_calls=0,exact_linear_recovery=1,exact_k_solve=1,newton_iterations=len(trace)))
    write(args.output/'construction.json',construction)
    write(args.output/'exact-data.json',data)
    checks={str(bits):verify_limiting(data,precision_bits=bits) for bits in
            sorted(set((192,256,args.precision_bits)))}
    proof=checks[str(args.precision_bits)]
    root=Path(__file__).resolve().parents[1]
    source=args.output/'construction.json'
    certificate=dict(schema_version=1,name='exact C4 active-family refinement',kind='limiting',
        precision_bits=args.precision_bits,arithmetic='exact rational / FLINT-Arb',
        source_file=source.resolve().relative_to(root).as_posix(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_reported_mu=float(density(x,p)),exact_data=data,
        changes=dict(construction='Learned C4 active pattern; exact root approximation and strictly inward neighbor scaling'),
        original_snapshot_attempt=dict(certified=False,reason='New exact construction from a C4-derived pattern'),
        result=proof,graph_realization_certified=False)
    write(args.output/'certificate.json',certificate)
    summary=dict(certified=True,density=decimal_bounds(density(x,p),digits=70),
        checked_bits=list(map(int,checks)),scope=proof['scope'],conic_calls=0,
        source_hash=construction['source_hash'],s7_data_used=False)
    write(args.output/'summary.json',summary)
    print(json.dumps(dict(certified=True,density_lower=summary['density']['lower'],
                         checked_bits=summary['checked_bits'],conic_calls=0)),flush=True)


if __name__=='__main__':main()
