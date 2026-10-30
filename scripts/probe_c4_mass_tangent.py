"""Checked finite probes with first-order mass changes from exact C4 data."""

import argparse,hashlib,json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from flint import fmpq,fmpq_mat,arb,ctx

from kuramoto_template.asymptotic_search.interval_certification import (
    rational,qmatrix,strings,density,decimal_bounds,rational_finite_repair,verify_finite,verify_limiting,
)
from kuramoto_template.asymptotic_search.joint_outer import _write_with_retry as write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--certificate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--h-bits',type=int,nargs='+',default=[132,134,136,137,138,140,144])
    args=parser.parse_args()
    cert=json.loads(args.certificate.read_text());D=cert['exact_data']
    verify_limiting(D,precision_bits=cert['precision_bits'])
    args.output.mkdir(parents=True,exist_ok=False)
    x=list(map(rational,D['x']));s=list(map(rational,D['s']));g=D['groups'];P=qmatrix(D['P0']);R=qmatrix(D['R']);m=len(x)
    pairs=[(a,b) for a in range(m) for b in range(a+1,m) if (g[a]-g[b])%2];n=len(pairs)
    matrix=[[fmpq(0) for _ in range(n)] for _ in range(m+4)]
    for i,(a,b) in enumerate(pairs):
        aa,bb=(a,b) if (g[b]-g[a])%4==1 else (b,a)
        mass=x[aa]*x[bb];matrix[aa][i]=mass;matrix[bb][i]=-mass
        matrix[m+g[aa]][i]=-mass*(s[bb]-s[aa])
    exactA=fmpq_mat(matrix)
    r0=fmpq_mat([[R[a][b]] for a,b in pairs]);target=exactA*r0
    A=np.zeros((m+5,n+m+1));A[:m+4,:n]=np.array([[float(v) for v in row] for row in matrix]);A[-1,n:n+m]=list(map(float,x))
    rhs=np.r_[[float(target[i,0]) for i in range(m+4)],0.]
    ub=[];bub=[]
    for i,(a,b) in enumerate(pairs):
        if P[a][b]>fmpq(9,10):
            row=np.zeros(n+m+1);row[i]=1;row[n+a]=-float(P[a][b]);row[n+b]=-float(P[a][b]);ub.append(row);bub.append(0.)
    degrees=[sum((P[a][b]*x[b] for b in range(m)),fmpq(0)) for a in range(m)]
    for a in range(m):
        if degrees[a]>min(degrees)+fmpq(1,1000000):continue
        row=np.zeros(n+m+1);row[-1]=1
        for i,(aa,b) in enumerate(pairs):
            if a==aa:row[i]-=float(x[b])
            if a==b:row[i]-=float(x[aa])
        row[n+a]+=float(sum((P[a][b]*x[b] for b in range(m) if (g[a]-g[b])%2),fmpq(0)))
        for b in range(m):
            if g[a]==g[b]:row[n+b]-=float(P[a][b]*x[b])
        ub.append(row);bub.append(0.)
    objective=np.zeros(n+m+1);objective[-1]=-1
    lp=linprog(objective,A_ub=ub,b_ub=bub,A_eq=A,b_eq=rhs,bounds=[(-1000,1000)]*(n+m)+[(None,None)],method='highs')
    if not lp.success:raise ValueError(lp.message)
    initial=fmpq_mat([[rational(v)] for v in lp.x[:n]])
    reduced,rank=exactA.transpose().rref()
    rows=[next(j for j in range(m+4) if reduced[i,j]!=0) for i in range(rank)]
    B=fmpq_mat([matrix[j] for j in rows])
    rv=initial-B.transpose()*(B*B.transpose()).solve(B*(initial-r0))
    residual=exactA*(rv-r0)
    if any(residual[i,0]!=0 for i in range(m+4)):raise ValueError('Exact direction projection failed')
    t=[rational(v) for v in lp.x[n:n+m]]
    anchor=next(a for a in range(m) if g.count(g[a])==1)
    t[anchor]-=sum((x[a]*t[a] for a in range(m)),fmpq(0))/x[anchor]
    correction=[[fmpq(0) for _ in range(m)] for _ in range(m)]
    for i,(a,b) in enumerate(pairs):correction[a][b]=correction[b][a]=rv[i,0]-(t[a]+t[b])*P[a][b]
    write(args.output/'direction.json',dict(source=str(args.certificate),source_sha256=hashlib.sha256(args.certificate.read_bytes()).hexdigest(),
        lp_lambda=float(lp.x[-1]),lp_solution=lp.x.tolist(),mass_derivative=strings(t),density_derivative=strings(correction),
        recovered_effective_R=strings([rv[i,0] for i in range(n)]),exact_projection_verified=True,
        counters=dict(linear_programs=1,exact_nullspace_projections=1)))
    rows=[];root=Path(__file__).resolve().parents[1]
    for hb in args.h_bits:
        h=fmpq(1,2**hb);xh=[x[a]*(1+h*t[a]) for a in range(m)]
        ph=[[P[a][b]+h*correction[a][b] for b in range(m)] for a in range(m)]
        with ctx.workprec(4096):phi=[(g[a]*arb.pi()/2+arb(h)*arb(s[a])).mid().fmpq() for a in range(m)]
        order=[a for a in range(m) if a!=anchor]+[anchor]
        inverse={old:new for new,old in enumerate(order)}
        source=args.output/f'path-{hb}.json'
        saved=dict(candidate=dict(x=strings([xh[a] for a in order])),
            P=strings([[ph[a][b] for b in order] for a in order]),phi=strings([phi[a] for a in order]),
            tree=[[inverse[a],inverse[b]] for a,b in D['repair_tree']],h=str(h),input_order=order)
        write(source,saved)
        row=dict(h_bits=hb,density_before_repair=decimal_bounds(density(xh,ph),70))
        exact=None
        try:
            exact,changes=rational_finite_repair(saved,coordinate_bits=1024,precision_bits=4096)
            with ctx.workprec(4096):centers=[(2*arb(rational(v)).atan()).mid().fmpq() for v in exact['tan_half_phi'][:-1]]
            box=dict(box_centers=strings(centers),box_radius=str(fmpq(1,2**1800)))
            proof=verify_finite(exact,precision_bits=4096,box=box)
            certificate=dict(schema_version=1,name='C4 mass-tangent finite probe',kind='finite',precision_bits=4096,
                source_file=source.resolve().relative_to(root).as_posix(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_reported_mu=float(density(xh,ph)),exact_data=exact,changes=changes,result=proof,
                original_snapshot_attempt=dict(certified=False,reason='Explicit rational repair only'),graph_realization_certified=False)
            write(args.output/f'certificate-{hb}.json',certificate)
            row.update(certified=True,density=decimal_bounds(density(list(map(rational,exact['x'])),qmatrix(exact['P'])),70))
        except Exception as exc:
            row.update(certified=False,failure=f'{type(exc).__name__}: {exc}')
            write(args.output/f'failure-{hb}.json',dict(result=row,exact_data=exact))
        rows.append(row);write(args.output/'summary.json',dict(rows=rows,conic_calls=0,exact_tree_solves=len(rows)))
        print(json.dumps(dict(h_bits=hb,certified=row['certified'],lower=row.get('density',{}).get('lower'),failure=row.get('failure'))),flush=True)


if __name__=='__main__':main()
