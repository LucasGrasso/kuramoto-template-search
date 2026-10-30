"""Explicit finite rational lift of independently certified exact C4 data.

Build P0+hR and high-precision C4 phases, round unit-circle coordinates to
rationals, cancel forces on the stored neighbor tree, and independently prove
Krawczyk inclusion plus interval Cholesky. No asymptotic claim is substituted
for a finite certificate.
"""

import argparse,hashlib,json
from pathlib import Path

from flint import arb,ctx,fmpq

from kuramoto_template.asymptotic_search.interval_certification import (
    decimal_bounds,density,qmatrix,rational,rational_finite_repair,strings,verify_finite,verify_limiting,
)
from kuramoto_template.asymptotic_search.joint_outer import _write_with_retry as write
from kuramoto_template.asymptotic_search.persistence import _code_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--certificate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--h-bits',type=int,default=300)
    parser.add_argument('--coordinate-bits',type=int,default=1024)
    parser.add_argument('--phase-box-bits',type=int,default=3500)
    parser.add_argument('--precision-bits',type=int,default=8192)
    args=parser.parse_args()
    certificate=json.loads(args.certificate.read_text())
    if certificate.get('kind')!='limiting':parser.error('Need a limiting certificate')
    data=certificate['exact_data']
    verify_limiting(data,precision_bits=certificate['precision_bits'])
    root=Path(__file__).resolve().parents[1]
    original=root/certificate['source_file']
    if hashlib.sha256(original.read_bytes()).hexdigest()!=certificate['source_sha256']:
        parser.error('Limiting certificate source hash changed')
    construction=json.loads(original.read_text())
    if construction.get('s7_data_used') is not False or 'C4' not in construction.get('kind',''):
        parser.error('Need explicitly C4-derived exact construction provenance')
    args.output.mkdir(parents=True,exist_ok=False)
    x=[rational(v) for v in data['x']]
    s=[rational(v) for v in data['s']]
    p0,r=qmatrix(data['P0']),qmatrix(data['R'])
    h=fmpq(1,2**args.h_bits)
    p=[[p0[i][j]+h*r[i][j] for j in range(len(x))] for i in range(len(x))]
    with ctx.workprec(args.precision_bits):
        phi=[(int(data['groups'][i])*arb.pi()/2+arb(h)*arb(s[i])).mid().fmpq() for i in range(len(x))]
    # A singleton phase avoids an exactly antipodal reference with equal
    # offsets, making the rational half-angle gauge robust near tan's poles.
    anchor=next(i for i in range(len(x)) if data['groups'].count(data['groups'][i])==1)
    order=[i for i in range(len(x)) if i!=anchor]+[anchor]
    inverse={old:new for new,old in enumerate(order)}
    saved=dict(kind='explicit C4 finite path before exact tree repair',candidate=dict(x=strings([x[i] for i in order])),
        P=strings([[p[i][j] for j in order] for i in order]),phi=strings([phi[i] for i in order]),
        tree=[[inverse[a],inverse[b]] for a,b in data['repair_tree']],h=str(h),input_order=order,
        coordinate_bits=args.coordinate_bits,phase_box_bits=args.phase_box_bits,
        limiting_certificate=str(args.certificate),limiting_certificate_sha256=hashlib.sha256(args.certificate.read_bytes()).hexdigest(),
        source_hash=_code_hash(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_snapshot_certified=False,graph_realization_certified=False)
    write(args.output/'finite-path.json',saved)
    exact,changes=rational_finite_repair(saved,coordinate_bits=args.coordinate_bits,precision_bits=args.precision_bits)
    write(args.output/'exact-data.json',exact)
    with ctx.workprec(args.precision_bits):
        t=[rational(v) for v in exact['tan_half_phi']]
        centers=[(2*arb(v).atan()).mid().fmpq() for v in t[:-1]]
    box=dict(box_centers=strings(centers),box_radius=str(fmpq(1,2**args.phase_box_bits)))
    try:
        checks={str(bits):verify_finite(exact,precision_bits=bits,box=box)
                for bits in (args.precision_bits,args.precision_bits+256,args.precision_bits+512)}
    except Exception as exc:
        write(args.output/'verification-failure.json',dict(failure=f'{type(exc).__name__}: {exc}',
            precision_bits=args.precision_bits,phase_box_bits=args.phase_box_bits,counts=dict(exact_tree_solves=1)))
        raise
    proof=checks[str(args.precision_bits)]
    source=args.output/'finite-path.json'
    result=dict(schema_version=1,name='explicit finite C4 rational lift',kind='finite',
        precision_bits=args.precision_bits,arithmetic='exact rational / FLINT-Arb',
        source_file=source.resolve().relative_to(root).as_posix(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_reported_mu=float(density(x,p)),exact_data=exact,changes=changes,result=proof,
        original_snapshot_attempt=dict(certified=False,reason='Independent rational unit-circle and exact tree repair'),
        graph_realization_certified=False)
    write(args.output/'certificate.json',result)
    summary=dict(certified=True,kind='explicit finite template',h=str(h),h_bits=args.h_bits,
        density=decimal_bounds(density([rational(v) for v in exact['x']],qmatrix(exact['P'])),digits=70),
        checked_bits=list(map(int,checks)),krawczyk_strict_inclusion=proof['proof']['strict_inclusion'],
        interval_cholesky=proof['proof']['interval_cholesky']['positive_definite'],
        box_radius=box['box_radius'],counts=dict(conic_calls=0,exact_tree_solves=1,krawczyk_checks=3),
        graph_realization_certified=False)
    write(args.output/'summary.json',summary)
    print(json.dumps(dict(certified=summary['certified'],h_bits=args.h_bits,
        density_lower=summary['density']['lower'],checked_bits=summary['checked_bits'],
        krawczyk=True,interval_cholesky=True)),flush=True)


if __name__=='__main__':main()
